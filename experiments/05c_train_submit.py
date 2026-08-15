#!/usr/bin/env python3
"""Train the second model-specific ensemble and build submit_02.zip."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import platform
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from features import (
    FeatureConfig,
    ID_COL,
    TARGET_COL,
    add_rowwise_features,
    apply_category_maps,
    categorical_columns,
    fit_category_maps,
)


PACKAGE_SCHEMA_VERSION = 2
EXPECTED_TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
REFERENCE_SEASON = 2025
LGBM_WEIGHT = 0.65
CATBOOST_WEIGHT = 0.35
BLEND_METHOD = "logit"
SELECTED_PAIR = "lgb_pitchmix_shape__cat_all_new"
SUBMISSION_FILENAME = "submit_02.zip"
SUBMISSION_REQUIREMENTS = """# Evaluation image supplies numpy, pandas, and joblib.\nlightgbm==4.6.0\ncatboost==1.2.8\n"""


def load_base_submission_module():
    path = PROJECT_ROOT / "experiments/04_train_submit.py"
    specification = importlib.util.spec_from_file_location(
        "aimers_first_submission_builder", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import first submission builder: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


BASE = load_base_submission_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data/train.csv")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/final_submission_v2",
    )
    parser.add_argument(
        "--oof-latest-path",
        type=Path,
        default=PROJECT_ROOT / "outputs/oof_recombination_v1/latest_run.json",
    )
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--lgbm-estimators", type=int, default=500)
    parser.add_argument("--catboost-iterations", type=int, default=85)
    parser.add_argument(
        "--catboost-task-type",
        choices=["CPU", "GPU"],
        default="GPU",
    )
    return parser.parse_args()


def lgbm_feature_config() -> FeatureConfig:
    return FeatureConfig(
        include_season=True,
        include_player_ids=False,
        include_game_type=True,
        include_pitchmix_shape_features=True,
    )


def catboost_feature_config() -> FeatureConfig:
    return FeatureConfig(
        include_season=False,
        include_player_ids=True,
        include_game_type=False,
        include_rate_gap_features=True,
        include_recent_dynamics_features=True,
        include_pitchmix_shape_features=True,
        include_context_interaction_features=True,
        include_team_context_features=True,
    )


def load_runtime_module(path: Path):
    specification = importlib.util.spec_from_file_location(
        "aimers_second_submission_feature_runtime", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import submission runtime: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def verify_training_inference_feature_parity(
    data: pd.DataFrame,
    runtime_path: Path,
) -> dict:
    runtime = load_runtime_module(runtime_path)
    sample = data.sample(n=min(128, len(data)), random_state=2026).copy()
    rows = []
    for model_name, train_config in (
        ("lightgbm", lgbm_feature_config()),
        ("catboost", catboost_feature_config()),
    ):
        train_frame = (
            data[data["game_type"].eq("R")] if model_name == "lightgbm" else data
        )
        prior = float(train_frame[TARGET_COL].mean())
        training_features = add_rowwise_features(
            sample, target_prior=prior, config=train_config
        )
        runtime_config = runtime.FeatureConfig(**asdict(train_config))
        runtime_features = runtime.add_rowwise_features(
            sample, target_prior=prior, config=runtime_config
        )
        pd.testing.assert_frame_equal(
            training_features,
            runtime_features,
            check_exact=True,
        )
        numeric = runtime_features.select_dtypes(include=[np.number]).to_numpy(
            dtype=np.float64
        )
        if np.isinf(numeric).any():
            raise RuntimeError(f"infinite feature found in runtime parity: {model_name}")
        rows.append(
            {
                "model": model_name,
                "rows": int(len(sample)),
                "feature_count": int(training_features.shape[1]),
                "exact_match": True,
                "infinite_numeric_values": 0,
            }
        )
    return {"status": "passed", "models": rows}


def load_and_validate_oof_selection(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"05b latest result is missing: {path}. Run notebook 05b first."
        )
    with path.open(encoding="utf-8") as handle:
        latest = json.load(handle)
    summary = latest.get("summary", {})
    selected = summary.get("selected_safe_fixed_pair")
    if selected != SELECTED_PAIR:
        raise RuntimeError(
            f"unexpected 05b selected pair: expected={SELECTED_PAIR}, got={selected}"
        )
    if float(summary.get("selected_safe_fixed_delta_brier_vs_base", 1.0)) >= 0.0:
        raise RuntimeError("05b selected pair does not improve the base Brier")
    run_dir = Path(latest["run_dir"])
    fixed_summary_path = run_dir / "fixed_pair_summary.csv"
    if not fixed_summary_path.exists():
        raise FileNotFoundError(f"05b fixed pair summary is missing: {fixed_summary_path}")
    fixed_summary = pd.read_csv(fixed_summary_path)
    selected_rows = fixed_summary[fixed_summary["pair"].eq(SELECTED_PAIR)]
    if len(selected_rows) != 1:
        raise RuntimeError(f"expected one 05b row for {SELECTED_PAIR}")
    selected_row = selected_rows.iloc[0]
    if not bool(selected_row["safe_2023_2024"]):
        raise RuntimeError("selected 05b pair is not safe in both 2023 and 2024")
    latest["selected_fixed_metrics"] = {
        "recent_brier": float(selected_row["recent_brier"]),
        "recent_bss_raw": float(selected_row["recent_bss_raw"]),
        "delta_recent_brier_vs_base": float(
            selected_row["delta_recent_brier_vs_base"]
        ),
        "brier_2022": float(selected_row["brier_2022"]),
        "brier_2023": float(selected_row["brier_2023"]),
        "brier_2024": float(selected_row["brier_2024"]),
        "bss_raw_2022": float(selected_row["bss_raw_2022"]),
        "bss_raw_2023": float(selected_row["bss_raw_2023"]),
        "bss_raw_2024": float(selected_row["bss_raw_2024"]),
    }
    return latest


def train_lgbm(
    data: pd.DataFrame,
    *,
    args: argparse.Namespace,
    model_path: Path,
) -> tuple[dict, dict, np.ndarray, pd.DataFrame, pd.DataFrame]:
    configuration = lgbm_feature_config()
    train = data[data["game_type"].eq("R")].copy()
    if train.empty:
        raise ValueError("LightGBM R-only training frame is empty")
    target_prior = float(train[TARGET_COL].mean())
    features = add_rowwise_features(
        train, target_prior=target_prior, config=configuration
    )
    categorical = categorical_columns(features)
    category_maps = fit_category_maps(features, categorical)
    features = apply_category_maps(features, category_maps)
    target = train[TARGET_COL].to_numpy(dtype=np.int8)
    weights = BASE.recency_weights(
        train["season"], reference_season=REFERENCE_SEASON, half_life=2.5
    )
    parameters = BASE.lgbm_parameters(args)
    print(
        f"Training LightGBM pitchmix | R rows={len(train):,} "
        f"features={features.shape[1]} weight=[{weights.min():.6f}, {weights.max():.6f}]"
    )
    started = time.perf_counter()
    model = lgb.LGBMClassifier(**parameters)
    model.fit(
        features,
        target,
        sample_weight=weights,
        categorical_feature=categorical,
    )
    elapsed = time.perf_counter() - started
    model.booster_.save_model(str(model_path))
    audit_indices = np.unique(
        np.linspace(0, len(train) - 1, num=min(256, len(train)), dtype=int)
    )
    audit_raw = train.iloc[audit_indices].copy()
    audit_prediction = np.clip(
        model.predict_proba(features.iloc[audit_indices])[:, 1], 1e-7, 1.0 - 1e-7
    )
    importance = pd.DataFrame(
        {
            "feature": features.columns,
            "importance_gain": model.booster_.feature_importance("gain"),
        }
    ).sort_values("importance_gain", ascending=False)
    metadata = {
        "model_filename": model_path.name,
        "feature_bundle": "pitchmix_shape",
        "feature_config": asdict(configuration),
        "target_prior": target_prior,
        "feature_columns": list(features.columns),
        "categorical_columns": categorical,
        "category_maps": category_maps,
    }
    summary = {
        "train_rows": int(len(train)),
        "train_seasons": sorted(int(value) for value in train["season"].unique()),
        "train_game_type": "R",
        "target_rate": target_prior,
        "half_life": 2.5,
        "reference_season": REFERENCE_SEASON,
        "feature_bundle": "pitchmix_shape",
        "feature_count": int(features.shape[1]),
        "categorical_count": int(len(categorical)),
        "train_seconds": elapsed,
        "parameters": parameters,
    }
    del train, features, target, weights, model
    gc.collect()
    return metadata, summary, audit_prediction, audit_raw, importance


def train_catboost(
    data: pd.DataFrame,
    *,
    args: argparse.Namespace,
    model_path: Path,
) -> tuple[dict, dict, np.ndarray, pd.DataFrame, pd.DataFrame]:
    configuration = catboost_feature_config()
    target_prior = float(data[TARGET_COL].mean())
    features = add_rowwise_features(
        data, target_prior=target_prior, config=configuration
    )
    categorical = categorical_columns(features)
    for column in categorical:
        features[column] = (
            features[column].astype("string").fillna("__MISSING__").astype(str)
        )
    target = data[TARGET_COL].to_numpy(dtype=np.int8)
    weights = BASE.recency_weights(
        data["season"], reference_season=REFERENCE_SEASON, half_life=0.5
    )
    parameters = BASE.catboost_parameters(args)
    print(
        f"Training CatBoost all_new | rows={len(data):,} features={features.shape[1]} "
        f"task={args.catboost_task_type} weight=[{weights.min():.8f}, {weights.max():.6f}]"
    )
    started = time.perf_counter()
    model = CatBoostClassifier(**parameters)
    model.fit(
        features,
        target,
        cat_features=categorical,
        sample_weight=weights,
        verbose=25,
    )
    elapsed = time.perf_counter() - started
    model.save_model(str(model_path))
    audit_indices = np.unique(
        np.linspace(0, len(data) - 1, num=min(256, len(data)), dtype=int)
    )
    audit_raw = data.iloc[audit_indices].copy()
    audit_prediction = np.clip(
        model.predict_proba(features.iloc[audit_indices])[:, 1], 1e-7, 1.0 - 1e-7
    )
    importance = pd.DataFrame(
        {"feature": features.columns, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False)
    metadata = {
        "model_filename": model_path.name,
        "feature_bundle": "all_new",
        "feature_config": asdict(configuration),
        "target_prior": target_prior,
        "feature_columns": list(features.columns),
        "categorical_columns": categorical,
    }
    summary = {
        "train_rows": int(len(data)),
        "train_seasons": sorted(int(value) for value in data["season"].unique()),
        "train_game_type": "all",
        "target_rate": target_prior,
        "half_life": 0.5,
        "reference_season": REFERENCE_SEASON,
        "feature_bundle": "all_new",
        "feature_count": int(features.shape[1]),
        "categorical_count": int(len(categorical)),
        "train_seconds": elapsed,
        "parameters": parameters,
    }
    del features, target, weights, model
    gc.collect()
    return metadata, summary, audit_prediction, audit_raw, importance


def main() -> None:
    args = parse_args()
    args.threads = max(1, min(8, args.threads))
    output_dir = args.output_dir.resolve()
    package_dir = output_dir / "package"
    model_dir = package_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_latest = load_and_validate_oof_selection(args.oof_latest_path)
    runtime_template = SRC_DIR / "submission_entrypoint.py"
    if not runtime_template.exists():
        raise FileNotFoundError(f"submission runtime is missing: {runtime_template}")
    shutil.copy2(runtime_template, package_dir / "script.py")
    (package_dir / "requirements.txt").write_text(
        SUBMISSION_REQUIREMENTS, encoding="utf-8"
    )

    total_started = time.perf_counter()
    print(f"Loading {args.train_path} ...")
    data = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False)
    BASE.validate_training_data(data)
    seasons = sorted(int(value) for value in data["season"].unique())
    if seasons != EXPECTED_TRAIN_SEASONS:
        raise ValueError(
            f"unexpected training seasons: expected={EXPECTED_TRAIN_SEASONS}, got={seasons}"
        )
    raw_input_columns = [column for column in data.columns if column != TARGET_COL]
    print(
        f"rows={len(data):,} seasons={seasons} target_rate={data[TARGET_COL].mean():.6f} "
        f"selected_pair={SELECTED_PAIR}"
    )
    feature_parity = verify_training_inference_feature_parity(
        data, runtime_template
    )
    print(f"training/runtime feature parity: {feature_parity['status']}")

    (
        lgbm_metadata,
        lgbm_summary,
        lgbm_reference,
        lgbm_audit_raw,
        lgbm_importance,
    ) = train_lgbm(data, args=args, model_path=model_dir / "lgbm_model.txt")
    lgbm_importance.to_csv(
        output_dir / "lgbm_feature_importance.csv", index=False
    )
    del lgbm_importance

    (
        catboost_metadata,
        catboost_summary,
        catboost_reference,
        catboost_audit_raw,
        catboost_importance,
    ) = train_catboost(
        data, args=args, model_path=model_dir / "catboost_model.cbm"
    )
    catboost_importance.to_csv(
        output_dir / "catboost_feature_importance.csv", index=False
    )
    del catboost_importance

    artifacts = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "competition": "LG Aimers 9th",
        "submission_version": 2,
        "target_column": TARGET_COL,
        "raw_input_columns": raw_input_columns,
        "selected_pair": SELECTED_PAIR,
        "lgbm": lgbm_metadata,
        "catboost": catboost_metadata,
        "blend": {
            "method": BLEND_METHOD,
            "lgbm_weight": LGBM_WEIGHT,
            "catboost_weight": CATBOOST_WEIGHT,
            "calibration": "none",
        },
    }
    joblib.dump(artifacts, model_dir / "artifacts.joblib", compress=3)

    audit = BASE.verify_serialization_and_row_independence(
        script_path=package_dir / "script.py",
        model_dir=model_dir,
        lgbm_audit_raw=lgbm_audit_raw,
        lgbm_reference=lgbm_reference,
        catboost_audit_raw=catboost_audit_raw,
        catboost_reference=catboost_reference,
    )
    del lgbm_audit_raw, catboost_audit_raw, lgbm_reference, catboost_reference
    gc.collect()

    end_to_end = BASE.verify_end_to_end(package_dir, args.data_dir)
    zip_path = output_dir / SUBMISSION_FILENAME
    zip_summary = BASE.build_and_verify_zip(package_dir, zip_path)
    if len(zip_path.name) > 30:
        raise RuntimeError(f"submission filename exceeds 30 characters: {zip_path.name}")

    oof_summary = oof_latest["summary"]
    selected_oof = oof_latest["selected_fixed_metrics"]
    summary = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "submission_filename": SUBMISSION_FILENAME,
        "selected_pair": SELECTED_PAIR,
        "train_path": str(args.train_path.resolve()),
        "train_file_bytes": int(args.train_path.stat().st_size),
        "train_file_sha256": BASE.sha256_file(args.train_path),
        "all_train_rows": int(len(data)),
        "all_train_seasons": seasons,
        "reference_season": REFERENCE_SEASON,
        "blend": artifacts["blend"],
        "validation_reference": {
            "source_05b_signature": oof_summary["signature"],
            "selection_seasons": [2023, 2024],
            "pair": SELECTED_PAIR,
            "brier": selected_oof["recent_brier"],
            "bss_raw": selected_oof["recent_bss_raw"],
            "delta_brier_vs_base": selected_oof[
                "delta_recent_brier_vs_base"
            ],
            "fold_brier": {
                "2022": selected_oof["brier_2022"],
                "2023": selected_oof["brier_2023"],
                "2024": selected_oof["brier_2024"],
            },
            "fold_bss_raw": {
                "2022": selected_oof["bss_raw_2022"],
                "2023": selected_oof["bss_raw_2023"],
                "2024": selected_oof["bss_raw_2024"],
            },
            "note": "Frozen OOF reference only; no test statistic was used.",
        },
        "lgbm": lgbm_summary,
        "catboost": catboost_summary,
        "training_inference_feature_parity": feature_parity,
        "audit": audit,
        "end_to_end": end_to_end,
        "package": zip_summary,
        "model_files": {
            path.name: {
                "bytes": int(path.stat().st_size),
                "sha256": BASE.sha256_file(path),
            }
            for path in sorted(model_dir.iterdir())
            if path.is_file()
        },
        "source_files": {
            "training_script_sha256": BASE.sha256_file(Path(__file__)),
            "feature_source_sha256": BASE.sha256_file(SRC_DIR / "features.py"),
            "runtime_script_sha256": BASE.sha256_file(runtime_template),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
            "lightgbm": lgb.__version__,
            "catboost": __import__("catboost").__version__,
            "catboost_task_type": args.catboost_task_type,
            "threads": args.threads,
        },
        "total_seconds": time.perf_counter() - total_started,
    }
    with (output_dir / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\nAll checks passed.")
    print(f"Submission ZIP: {zip_path}")
    print(f"SHA-256: {zip_summary['zip_sha256']}")
    print(f"ZIP bytes: {zip_summary['zip_bytes']:,}")
    print(f"Total seconds: {summary['total_seconds']:.1f}")


if __name__ == "__main__":
    main()
