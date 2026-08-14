#!/usr/bin/env python3
"""Train the selected full-data ensemble and build the official submission ZIP."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
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


PACKAGE_SCHEMA_VERSION = 1
EXPECTED_TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
REFERENCE_SEASON = 2025
LGBM_WEIGHT = 0.65
CATBOOST_WEIGHT = 0.35
BLEND_METHOD = "logit"
EXPECTED_ZIP_NAMES = {
    "model/lgbm_model.txt",
    "model/catboost_model.cbm",
    "model/artifacts.joblib",
    "script.py",
    "requirements.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data/train.csv")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/final_submission_v1",
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


def sha256_file(path: Path, block_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def recency_weights(
    seasons: pd.Series,
    *,
    reference_season: int,
    half_life: float,
) -> np.ndarray:
    age = (reference_season - 1) - seasons.to_numpy(dtype=np.float64)
    if (age < 0).any():
        raise ValueError("training data contains a season beyond the reference season")
    weights = np.power(0.5, age / half_life)
    return weights / weights.mean()


def validate_training_data(data: pd.DataFrame) -> None:
    required = {ID_COL, TARGET_COL, "season", "game_type"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"training data is missing columns: {missing}")
    if not set(data[TARGET_COL].dropna().unique()).issubset({0, 1}):
        raise ValueError("control_success must contain only 0 and 1")
    if data[TARGET_COL].isna().any():
        raise ValueError("control_success contains null values")
    if data[ID_COL].isna().any() or data[ID_COL].astype("string").duplicated().any():
        raise ValueError("training row_id must be non-null and unique")


def lgbm_parameters(args: argparse.Namespace) -> dict:
    return {
        "objective": "binary",
        "n_estimators": args.lgbm_estimators,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 1_000,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "max_bin": 127,
        "random_state": 42,
        "n_jobs": args.threads,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


def catboost_parameters(args: argparse.Namespace) -> dict:
    parameters = {
        "loss_function": "Logloss",
        "iterations": args.catboost_iterations,
        "learning_rate": 0.04,
        "depth": 8,
        "l2_leaf_reg": 12.0,
        "random_strength": 0.5,
        "random_seed": 2026,
        "max_ctr_complexity": 1,
        "allow_writing_files": False,
        "task_type": args.catboost_task_type,
        "thread_count": args.threads,
    }
    if args.catboost_task_type == "GPU":
        parameters.update({"devices": "0", "gpu_ram_part": 0.85})
    return parameters


def train_lgbm(
    data: pd.DataFrame,
    *,
    args: argparse.Namespace,
    model_path: Path,
    reference_season: int,
) -> tuple[dict, dict, np.ndarray, pd.DataFrame, pd.DataFrame]:
    configuration = FeatureConfig(
        include_season=True,
        include_player_ids=False,
        include_game_type=True,
    )
    train = data[data["game_type"].eq("R")].copy()
    if train.empty:
        raise ValueError("LightGBM R-only training frame is empty")
    target_prior = float(train[TARGET_COL].mean())
    features = add_rowwise_features(
        train,
        target_prior=target_prior,
        config=configuration,
    )
    categorical = categorical_columns(features)
    category_maps = fit_category_maps(features, categorical)
    features = apply_category_maps(features, category_maps)
    target = train[TARGET_COL].to_numpy(dtype=np.int8)
    weights = recency_weights(
        train["season"], reference_season=reference_season, half_life=2.5
    )

    parameters = lgbm_parameters(args)
    print(
        f"Training LightGBM | R rows={len(train):,} features={features.shape[1]} "
        f"weight=[{weights.min():.6f}, {weights.max():.6f}]"
    )
    started = time.perf_counter()
    model = lgb.LGBMClassifier(**parameters)
    model.fit(
        features,
        target,
        sample_weight=weights,
        categorical_feature=categorical,
    )
    train_seconds = time.perf_counter() - started
    model.booster_.save_model(str(model_path))

    audit_indices = np.unique(
        np.linspace(0, len(train) - 1, num=min(256, len(train)), dtype=int)
    )
    audit_raw = train.iloc[audit_indices].copy()
    audit_prediction = np.clip(
        model.predict_proba(features.iloc[audit_indices])[:, 1], 1e-7, 1 - 1e-7
    )
    importance = pd.DataFrame(
        {
            "feature": features.columns,
            "importance_gain": model.booster_.feature_importance("gain"),
        }
    ).sort_values("importance_gain", ascending=False)
    metadata = {
        "model_filename": model_path.name,
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
        "reference_season": reference_season,
        "feature_count": int(features.shape[1]),
        "categorical_count": int(len(categorical)),
        "train_seconds": train_seconds,
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
    reference_season: int,
) -> tuple[dict, dict, np.ndarray, pd.DataFrame, pd.DataFrame]:
    configuration = FeatureConfig(
        pitcher_prior_strength=200.0,
        batter_prior_strength=300.0,
        include_season=False,
        include_player_ids=True,
        include_game_type=False,
    )
    target_prior = float(data[TARGET_COL].mean())
    features = add_rowwise_features(
        data,
        target_prior=target_prior,
        config=configuration,
    )
    categorical = categorical_columns(features)
    for column in categorical:
        features[column] = (
            features[column].astype("string").fillna("__MISSING__").astype(str)
        )
    target = data[TARGET_COL].to_numpy(dtype=np.int8)
    weights = recency_weights(
        data["season"], reference_season=reference_season, half_life=0.5
    )

    parameters = catboost_parameters(args)
    print(
        f"Training CatBoost | all rows={len(data):,} features={features.shape[1]} "
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
    train_seconds = time.perf_counter() - started
    model.save_model(str(model_path))

    audit_indices = np.unique(
        np.linspace(0, len(data) - 1, num=min(256, len(data)), dtype=int)
    )
    audit_raw = data.iloc[audit_indices].copy()
    audit_prediction = np.clip(
        model.predict_proba(features.iloc[audit_indices])[:, 1], 1e-7, 1 - 1e-7
    )
    importance = pd.DataFrame(
        {
            "feature": features.columns,
            "importance": model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    metadata = {
        "model_filename": model_path.name,
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
        "reference_season": reference_season,
        "feature_count": int(features.shape[1]),
        "categorical_count": int(len(categorical)),
        "train_seconds": train_seconds,
        "parameters": parameters,
    }
    del features, target, weights, model
    gc.collect()
    return metadata, summary, audit_prediction, audit_raw, importance


def load_inference_module(script_path: Path):
    specification = importlib.util.spec_from_file_location(
        "aimers_submission_runtime", script_path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import inference script: {script_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def verify_serialization_and_row_independence(
    *,
    script_path: Path,
    model_dir: Path,
    lgbm_audit_raw: pd.DataFrame,
    lgbm_reference: np.ndarray,
    catboost_audit_raw: pd.DataFrame,
    catboost_reference: np.ndarray,
) -> dict:
    runtime = load_inference_module(script_path)
    predictor = runtime.load_predictor(model_dir)

    lgbm_reloaded, _ = runtime.predict_components(lgbm_audit_raw, predictor)
    _, catboost_reloaded = runtime.predict_components(catboost_audit_raw, predictor)
    lgbm_reload_diff = max_abs_diff(lgbm_reference, lgbm_reloaded)
    catboost_reload_diff = max_abs_diff(catboost_reference, catboost_reloaded)
    if lgbm_reload_diff > 1e-12 or catboost_reload_diff > 1e-12:
        raise RuntimeError(
            "native model reload mismatch: "
            f"lgbm={lgbm_reload_diff}, catboost={catboost_reload_diff}"
        )

    audit = catboost_audit_raw.iloc[: min(64, len(catboost_audit_raw))].copy()
    base_prediction = runtime.predict_probabilities(audit, predictor)

    shuffled = audit.sample(frac=1.0, random_state=2026)
    shuffled_prediction = runtime.predict_probabilities(shuffled, predictor)
    shuffled_by_id = pd.Series(
        shuffled_prediction,
        index=shuffled[ID_COL].astype("string").to_numpy(),
    )
    reordered = shuffled_by_id.reindex(audit[ID_COL].astype("string")).to_numpy()

    chunks = np.array_split(audit, min(7, len(audit)))
    chunked = np.concatenate(
        [runtime.predict_probabilities(chunk, predictor) for chunk in chunks if len(chunk)]
    )
    single = np.array(
        [
            runtime.predict_probabilities(audit.iloc[[index]], predictor)[0]
            for index in range(min(10, len(audit)))
        ]
    )

    shuffle_diff = max_abs_diff(base_prediction, reordered)
    chunk_diff = max_abs_diff(base_prediction, chunked)
    single_diff = max_abs_diff(base_prediction[: len(single)], single)
    if max(shuffle_diff, chunk_diff, single_diff) > 1e-12:
        raise RuntimeError(
            "row-independence test failed: "
            f"shuffle={shuffle_diff}, chunk={chunk_diff}, single={single_diff}"
        )
    return {
        "lgbm_native_reload_max_abs_diff": lgbm_reload_diff,
        "catboost_native_reload_max_abs_diff": catboost_reload_diff,
        "shuffle_max_abs_diff": shuffle_diff,
        "chunk_max_abs_diff": chunk_diff,
        "single_row_max_abs_diff": single_diff,
        "audit_rows": int(len(audit)),
    }


def verify_end_to_end(package_dir: Path, data_dir: Path) -> dict:
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    if not test_path.exists() or not sample_path.exists():
        raise FileNotFoundError("data/test.csv and data/sample_submission.csv are required")
    with tempfile.TemporaryDirectory(prefix="aimers_submit_check_") as temporary:
        runtime_root = Path(temporary)
        shutil.copy2(package_dir / "script.py", runtime_root / "script.py")
        shutil.copy2(package_dir / "requirements.txt", runtime_root / "requirements.txt")
        shutil.copytree(package_dir / "model", runtime_root / "model")
        os.symlink(data_dir.resolve(), runtime_root / "data", target_is_directory=True)
        result = subprocess.run(
            [sys.executable, "script.py"],
            cwd=runtime_root,
            capture_output=True,
            text=True,
            timeout=540,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"packaged script.py failed with code {result.returncode}")
        generated = pd.read_csv(runtime_root / "output/submission.csv")
        sample = pd.read_csv(sample_path)
        if list(generated.columns) != [ID_COL, TARGET_COL]:
            raise RuntimeError("generated submission has invalid columns")
        if not generated[ID_COL].astype("string").equals(sample[ID_COL].astype("string")):
            raise RuntimeError("generated submission row_id order does not match sample")
        prediction = generated[TARGET_COL].to_numpy(dtype=np.float64)
        if not np.isfinite(prediction).all() or ((prediction < 0) | (prediction > 1)).any():
            raise RuntimeError("generated submission probabilities are invalid")
        return {
            "returncode": result.returncode,
            "rows": int(len(generated)),
            "prediction_min": float(prediction.min()),
            "prediction_mean": float(prediction.mean()),
            "prediction_max": float(prediction.max()),
        }


def build_and_verify_zip(package_dir: Path, zip_path: Path) -> dict:
    source_by_name = {
        "model/lgbm_model.txt": package_dir / "model/lgbm_model.txt",
        "model/catboost_model.cbm": package_dir / "model/catboost_model.cbm",
        "model/artifacts.joblib": package_dir / "model/artifacts.joblib",
        "script.py": package_dir / "script.py",
        "requirements.txt": package_dir / "requirements.txt",
    }
    for name, source in source_by_name.items():
        if not source.is_file():
            raise FileNotFoundError(f"missing package member {name}: {source}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source in source_by_name.items():
            archive.write(source, arcname=name)
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        if names != EXPECTED_ZIP_NAMES:
            raise RuntimeError(
                f"ZIP structure mismatch: missing={sorted(EXPECTED_ZIP_NAMES - names)}, "
                f"extra={sorted(names - EXPECTED_ZIP_NAMES)}"
            )
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise RuntimeError(f"corrupt ZIP member: {corrupt_member}")
        members = {
            info.filename: {"bytes": info.file_size, "compressed_bytes": info.compress_size}
            for info in archive.infolist()
        }
    if zip_path.stat().st_size >= 10 * 1024**3:
        raise RuntimeError("submission ZIP exceeds the 10 GiB competition limit")
    return {
        "zip_path": str(zip_path),
        "zip_bytes": int(zip_path.stat().st_size),
        "zip_sha256": sha256_file(zip_path),
        "members": members,
    }


def main() -> None:
    args = parse_args()
    args.threads = max(1, min(8, args.threads))
    output_dir = args.output_dir.resolve()
    package_dir = output_dir / "package"
    model_dir = package_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_template = PROJECT_ROOT / "submission/script.py"
    requirements_template = PROJECT_ROOT / "submission/requirements.txt"
    if not script_template.exists() or not requirements_template.exists():
        raise FileNotFoundError("submission/script.py and submission/requirements.txt are required")
    shutil.copy2(script_template, package_dir / "script.py")
    shutil.copy2(requirements_template, package_dir / "requirements.txt")

    total_started = time.perf_counter()
    print(f"Loading {args.train_path} ...")
    data = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False)
    validate_training_data(data)
    seasons = sorted(int(value) for value in data["season"].unique())
    if seasons != EXPECTED_TRAIN_SEASONS:
        raise ValueError(
            f"unexpected training seasons: expected={EXPECTED_TRAIN_SEASONS}, got={seasons}"
        )
    reference_season = REFERENCE_SEASON
    raw_input_columns = [column for column in data.columns if column != TARGET_COL]
    print(
        f"rows={len(data):,} seasons={seasons} target_rate={data[TARGET_COL].mean():.6f} "
        f"reference_season={reference_season}"
    )

    (
        lgbm_metadata,
        lgbm_summary,
        lgbm_reference,
        lgbm_audit_raw,
        lgbm_importance,
    ) = train_lgbm(
        data,
        args=args,
        model_path=model_dir / "lgbm_model.txt",
        reference_season=reference_season,
    )
    lgbm_importance.to_csv(output_dir / "lgbm_feature_importance.csv", index=False)
    del lgbm_importance

    (
        catboost_metadata,
        catboost_summary,
        catboost_reference,
        catboost_audit_raw,
        catboost_importance,
    ) = train_catboost(
        data,
        args=args,
        model_path=model_dir / "catboost_model.cbm",
        reference_season=reference_season,
    )
    catboost_importance.to_csv(output_dir / "catboost_feature_importance.csv", index=False)
    del catboost_importance

    artifacts = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "competition": "LG Aimers 9th",
        "target_column": TARGET_COL,
        "raw_input_columns": raw_input_columns,
        "lgbm": lgbm_metadata,
        "catboost": catboost_metadata,
        "blend": {
            "method": BLEND_METHOD,
            "lgbm_weight": LGBM_WEIGHT,
            "catboost_weight": CATBOOST_WEIGHT,
        },
    }
    joblib.dump(artifacts, model_dir / "artifacts.joblib", compress=3)

    audit = verify_serialization_and_row_independence(
        script_path=package_dir / "script.py",
        model_dir=model_dir,
        lgbm_audit_raw=lgbm_audit_raw,
        lgbm_reference=lgbm_reference,
        catboost_audit_raw=catboost_audit_raw,
        catboost_reference=catboost_reference,
    )
    del lgbm_audit_raw, catboost_audit_raw, lgbm_reference, catboost_reference
    gc.collect()

    end_to_end = verify_end_to_end(package_dir, args.data_dir)
    zip_path = output_dir / "submit_lgbm_catboost_logit6535.zip"
    zip_summary = build_and_verify_zip(package_dir, zip_path)

    summary = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "train_path": str(args.train_path.resolve()),
        "train_file_bytes": int(args.train_path.stat().st_size),
        "train_file_sha256": sha256_file(args.train_path),
        "all_train_rows": int(len(data)),
        "all_train_seasons": seasons,
        "reference_season": reference_season,
        "blend": artifacts["blend"],
        "validation_reference": {
            "selection_seasons": [2023, 2024],
            "lgbm_brier": 0.2486020739,
            "lgbm_bss_raw": 539.2297,
            "ensemble_brier": 0.2484494257,
            "ensemble_bss_raw": 600.3012,
            "note": "Reference only; copied from frozen forward OOF and not recomputed on test.",
        },
        "lgbm": lgbm_summary,
        "catboost": catboost_summary,
        "audit": audit,
        "end_to_end": end_to_end,
        "package": zip_summary,
        "model_files": {
            path.name: {
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in sorted(model_dir.iterdir())
            if path.is_file()
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
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\nAll checks passed.")
    print(f"Submission ZIP: {zip_path}")
    print(f"SHA-256: {zip_summary['zip_sha256']}")
    print(f"ZIP bytes: {zip_summary['zip_bytes']:,}")
    print(f"Total seconds: {summary['total_seconds']:.1f}")


if __name__ == "__main__":
    main()
