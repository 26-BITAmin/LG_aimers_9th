#!/usr/bin/env python3
"""Train the predeclared 07b R/F route and build submit_03.zip.

The LightGBM component and 65:35 logit blend stay frozen.  The current
all-row CatBoost is used on R rows, while a second CatBoost trained on all R
history plus only 2024 F rows is used on F rows.  Routing uses only the
current test row's game_type.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import platform
import shutil
import sys
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


PACKAGE_SCHEMA_VERSION = 3
EXPECTED_TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
REFERENCE_SEASON = 2025
F_ROUTE_TRAIN_SEASON = 2024
LGBM_WEIGHT = 0.65
CATBOOST_WEIGHT = 0.35
BLEND_METHOD = "logit"
SELECTED_PAIR = "lgb_pitchmix_shape__cat_all_new"
BASE_ROUTE = "current_r__current_f"
SELECTED_ROUTE = "current_r__latest_f1_f"
EXPECTED_ROUTING_SIGNATURE = "46f3d7d0000e"
SUBMISSION_FILENAME = "submit_03.zip"
SUBMISSION_REQUIREMENTS = (
    "# Evaluation image supplies numpy, pandas, and joblib.\n"
    "lightgbm==4.6.0\n"
    "catboost==1.2.8\n"
)
EXPECTED_ZIP_NAMES = {
    "model/lgbm_model.txt",
    "model/catboost_base.cbm",
    "model/catboost_f_route.cbm",
    "model/artifacts.joblib",
    "script.py",
    "requirements.txt",
}


def load_python_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import Python module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


BASE = load_python_module(
    "aimers_first_submission_builder", PROJECT_ROOT / "experiments/04_train_submit.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data/train.csv")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/final_submission_v3",
    )
    parser.add_argument(
        "--routing-latest-path",
        type=Path,
        default=PROJECT_ROOT / "outputs/f_routing_v1/latest_run.json",
    )
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--lgbm-estimators", type=int, default=500)
    parser.add_argument("--catboost-iterations", type=int, default=85)
    parser.add_argument(
        "--catboost-task-type", choices=["CPU", "GPU"], default="GPU"
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


def select_f_route_training_rows(data: pd.DataFrame) -> pd.DataFrame:
    selected = data[
        data["game_type"].eq("R")
        | (data["game_type"].eq("F") & data["season"].eq(F_ROUTE_TRAIN_SEASON))
    ].copy()
    if selected.empty or selected[TARGET_COL].nunique() != 2:
        raise RuntimeError("F-route CatBoost training frame has insufficient labels")
    expected_f_seasons = [F_ROUTE_TRAIN_SEASON]
    found_f_seasons = sorted(
        int(value)
        for value in selected.loc[selected["game_type"].eq("F"), "season"].unique()
    )
    if found_f_seasons != expected_f_seasons:
        raise RuntimeError(
            f"unexpected F-route seasons: expected={expected_f_seasons}, got={found_f_seasons}"
        )
    return selected


def _as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"cannot interpret boolean value: {value!r}")


def load_and_validate_routing_selection(latest_path: Path) -> dict:
    if not latest_path.exists():
        raise FileNotFoundError(
            f"07b latest result is missing: {latest_path}. Run notebook 07b first."
        )
    with latest_path.open(encoding="utf-8") as handle:
        latest = json.load(handle)
    summary = latest.get("summary", {})
    checks = {
        "signature": summary.get("signature") == EXPECTED_ROUTING_SIGNATURE,
        "base_route": summary.get("base_route") == BASE_ROUTE,
        "selected_route": summary.get("selected_route") == SELECTED_ROUTE,
        "selection_status": summary.get("selection_status") == "primary_passed",
        "selected_passes_fixed_gate": summary.get("selected_passes_fixed_gate") is True,
        "fixed_lgbm_weight": abs(
            float(summary.get("fixed_lgbm_weight", -1.0)) - LGBM_WEIGHT
        )
        <= 1e-12,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"07b selection validation failed: {failed}")

    recorded = Path(latest["run_dir"])
    run_candidates = [recorded, latest_path.parent / recorded.name]
    run_dir = next((path for path in run_candidates if path.exists()), None)
    if run_dir is None:
        raise FileNotFoundError(f"07b run directory is missing: {run_candidates}")
    route_summary_path = run_dir / "route_summary.csv"
    integrity_path = run_dir / "integrity_audit.json"
    for path in (route_summary_path, integrity_path):
        if not path.exists():
            raise FileNotFoundError(f"07b audit file is missing: {path}")

    route_summary = pd.read_csv(route_summary_path)
    selected_rows = route_summary[route_summary["route"].eq(SELECTED_ROUTE)]
    if len(selected_rows) != 1:
        raise RuntimeError(f"expected one 07b row for {SELECTED_ROUTE}")
    selected = selected_rows.iloc[0]
    gate_columns = [
        "r_predictions_exact",
        "improves_recent_all",
        "improves_recent_f",
        "safe_all_folds",
        "safe_f_all_folds",
        "passes_fixed_gate",
    ]
    failed_gates = [column for column in gate_columns if not _as_bool(selected[column])]
    if failed_gates:
        raise RuntimeError(f"07b selected route failed gates: {failed_gates}")

    with integrity_path.open(encoding="utf-8") as handle:
        integrity = json.load(handle)
    selected_audit = integrity["route_audits"][SELECTED_ROUTE]
    if int(selected_audit["changed_r_rows"]) != 0:
        raise RuntimeError("07b route changed R-row CatBoost predictions")
    if not bool(integrity.get("routing_uses_only_current_row_game_type")):
        raise RuntimeError("07b route is not row-independent")

    metrics = {
        "recent_brier": float(selected["recent_brier"]),
        "recent_bss_raw": float(selected["recent_bss_raw"]),
        "delta_recent_brier_vs_base": float(selected["delta_recent_brier_vs_base"]),
        "delta_recent_f_brier_vs_base": float(
            selected["delta_recent_f_brier_vs_base"]
        ),
        "fold_brier": {
            str(season): float(selected[f"brier_{season}"])
            for season in (2022, 2023, 2024)
        },
        "fold_delta_brier_vs_base": {
            str(season): float(selected[f"delta_brier_{season}_vs_base"])
            for season in (2022, 2023, 2024)
        },
        "fold_f_delta_brier_vs_base": {
            str(season): float(selected[f"delta_brier_f_{season}_vs_base"])
            for season in (2022, 2023, 2024)
        },
    }
    return {
        "latest": latest,
        "run_dir": run_dir,
        "selection_checks": checks,
        "selected_metrics": metrics,
        "integrity": integrity,
    }


def verify_training_inference_feature_parity(
    data: pd.DataFrame, runtime_path: Path
) -> dict:
    runtime = load_python_module("aimers_v3_feature_runtime", runtime_path)
    sample = data.sample(n=min(128, len(data)), random_state=2026).copy()
    r_mask = data["game_type"].eq("R")
    f_route_mask = r_mask | (
        data["game_type"].eq("F") & data["season"].eq(F_ROUTE_TRAIN_SEASON)
    )
    definitions = (
        ("lightgbm", float(data.loc[r_mask, TARGET_COL].mean()), lgbm_feature_config()),
        ("catboost_base", float(data[TARGET_COL].mean()), catboost_feature_config()),
        (
            "catboost_f_route",
            float(data.loc[f_route_mask, TARGET_COL].mean()),
            catboost_feature_config(),
        ),
    )
    rows = []
    for model_name, prior, train_config in definitions:
        training_features = add_rowwise_features(
            sample, target_prior=prior, config=train_config
        )
        runtime_features = runtime.add_rowwise_features(
            sample,
            target_prior=prior,
            config=runtime.FeatureConfig(**asdict(train_config)),
        )
        pd.testing.assert_frame_equal(training_features, runtime_features, check_exact=True)
        numeric = runtime_features.select_dtypes(include=[np.number]).to_numpy(
            dtype=np.float64
        )
        if np.isinf(numeric).any():
            raise RuntimeError(f"infinite feature found in runtime parity: {model_name}")
        rows.append(
            {
                "model": model_name,
                "rows": int(len(sample)),
                "target_prior": prior,
                "feature_count": int(training_features.shape[1]),
                "exact_match": True,
                "infinite_numeric_values": 0,
            }
        )
    return {"status": "passed", "models": rows}


def train_lgbm(
    data: pd.DataFrame, *, args: argparse.Namespace, model_path: Path
) -> tuple[dict, dict, np.ndarray, pd.DataFrame, pd.DataFrame]:
    configuration = lgbm_feature_config()
    train = data[data["game_type"].eq("R")].copy()
    target_prior = float(train[TARGET_COL].mean())
    features = add_rowwise_features(train, target_prior=target_prior, config=configuration)
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
    train: pd.DataFrame,
    *,
    args: argparse.Namespace,
    model_path: Path,
    model_label: str,
    training_policy: str,
) -> tuple[dict, dict, np.ndarray, pd.DataFrame, pd.DataFrame]:
    configuration = catboost_feature_config()
    target_prior = float(train[TARGET_COL].mean())
    features = add_rowwise_features(train, target_prior=target_prior, config=configuration)
    categorical = categorical_columns(features)
    for column in categorical:
        features[column] = (
            features[column].astype("string").fillna("__MISSING__").astype(str)
        )
    target = train[TARGET_COL].to_numpy(dtype=np.int8)
    weights = BASE.recency_weights(
        train["season"], reference_season=REFERENCE_SEASON, half_life=0.5
    )
    parameters = BASE.catboost_parameters(args)
    print(
        f"Training {model_label} | rows={len(train):,} features={features.shape[1]} "
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
        np.linspace(0, len(train) - 1, num=min(256, len(train)), dtype=int)
    )
    audit_raw = train.iloc[audit_indices].copy()
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
        "training_policy": training_policy,
    }
    f_rows = train["game_type"].eq("F")
    summary = {
        "train_rows": int(len(train)),
        "train_seasons": sorted(int(value) for value in train["season"].unique()),
        "train_game_types": sorted(str(value) for value in train["game_type"].unique()),
        "f_train_rows": int(f_rows.sum()),
        "f_train_seasons": sorted(
            int(value) for value in train.loc[f_rows, "season"].unique()
        ),
        "target_rate": target_prior,
        "half_life": 0.5,
        "reference_season": REFERENCE_SEASON,
        "training_policy": training_policy,
        "feature_bundle": "all_new",
        "feature_count": int(features.shape[1]),
        "categorical_count": int(len(categorical)),
        "train_seconds": elapsed,
        "parameters": parameters,
    }
    del features, target, weights, model
    gc.collect()
    return metadata, summary, audit_prediction, audit_raw, importance


def max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
            )
        )
    )


def verify_serialization_routing_and_row_independence(
    *,
    script_path: Path,
    model_dir: Path,
    full_data: pd.DataFrame,
    lgbm_audit_raw: pd.DataFrame,
    lgbm_reference: np.ndarray,
    base_audit_raw: pd.DataFrame,
    base_reference: np.ndarray,
    f_route_audit_raw: pd.DataFrame,
    f_route_reference: np.ndarray,
) -> dict:
    runtime = load_python_module("aimers_v3_serialized_runtime", script_path)
    predictor = runtime.load_predictor(model_dir)

    lgbm_reloaded, _ = runtime.predict_components(lgbm_audit_raw, predictor)
    base_reloaded = runtime.predict_catboost_model(
        base_audit_raw, predictor, "catboost_base"
    )
    f_route_reloaded = runtime.predict_catboost_model(
        f_route_audit_raw, predictor, "catboost_f_route"
    )
    reload_diffs = {
        "lgbm": max_abs_diff(lgbm_reference, lgbm_reloaded),
        "catboost_base": max_abs_diff(base_reference, base_reloaded),
        "catboost_f_route": max_abs_diff(f_route_reference, f_route_reloaded),
    }
    if max(reload_diffs.values()) > 1e-12:
        raise RuntimeError(f"native model reload mismatch: {reload_diffs}")

    generator = np.random.default_rng(2026)
    r_indices = np.flatnonzero(full_data["game_type"].eq("R").to_numpy())
    f_indices = np.flatnonzero(full_data["game_type"].eq("F").to_numpy())
    r_audit = full_data.iloc[
        generator.choice(r_indices, size=min(32, len(r_indices)), replace=False)
    ]
    f_audit = full_data.iloc[
        generator.choice(f_indices, size=min(32, len(f_indices)), replace=False)
    ]
    audit = pd.concat([r_audit, f_audit], ignore_index=True)
    if not {"R", "F"}.issubset(set(audit["game_type"].astype(str))):
        raise RuntimeError("routing audit requires both R and F rows")

    _, routed_catboost = runtime.predict_components(audit, predictor)
    base_all = runtime.predict_catboost_model(audit, predictor, "catboost_base")
    f_route_all = runtime.predict_catboost_model(
        audit, predictor, "catboost_f_route"
    )
    is_f = audit["game_type"].astype(str).eq("F").to_numpy(dtype=np.bool_)
    is_r = np.logical_not(is_f)
    expected_routed = np.where(is_f, f_route_all, base_all)
    routing_diffs = {
        "all_vs_expected": max_abs_diff(routed_catboost, expected_routed),
        "r_vs_base": max_abs_diff(routed_catboost[is_r], base_all[is_r]),
        "f_vs_f_route": max_abs_diff(routed_catboost[is_f], f_route_all[is_f]),
    }
    if max(routing_diffs.values()) > 1e-12:
        raise RuntimeError(f"serialized R/F routing mismatch: {routing_diffs}")

    base_prediction = runtime.predict_probabilities(audit, predictor)
    shuffled = audit.sample(frac=1.0, random_state=2027)
    shuffled_prediction = runtime.predict_probabilities(shuffled, predictor)
    shuffled_by_id = pd.Series(
        shuffled_prediction, index=shuffled[ID_COL].astype("string").to_numpy()
    )
    reordered = shuffled_by_id.reindex(
        audit[ID_COL].astype("string").to_numpy()
    ).to_numpy()
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
    independence_diffs = {
        "shuffle": max_abs_diff(base_prediction, reordered),
        "chunk": max_abs_diff(base_prediction, chunked),
        "single_row": max_abs_diff(base_prediction[: len(single)], single),
    }
    if max(independence_diffs.values()) > 1e-12:
        raise RuntimeError(f"row-independence test failed: {independence_diffs}")

    metadata = predictor["metadata"]
    base_prior = float(metadata["catboost_base"]["target_prior"])
    f_route_prior = float(metadata["catboost_f_route"]["target_prior"])
    return {
        "status": "passed",
        "native_reload_max_abs_difference": reload_diffs,
        "routing_max_abs_difference": routing_diffs,
        "row_independence_max_abs_difference": independence_diffs,
        "routing_audit_rows": int(len(audit)),
        "routing_audit_r_rows": int(is_r.sum()),
        "routing_audit_f_rows": int(is_f.sum()),
        "catboost_base_target_prior": base_prior,
        "catboost_f_route_target_prior": f_route_prior,
        "catboost_priors_stored_separately": True,
        "routing_uses_only_current_row_game_type": True,
    }


def build_and_verify_zip(package_dir: Path, zip_path: Path) -> dict:
    source_by_name = {
        "model/lgbm_model.txt": package_dir / "model/lgbm_model.txt",
        "model/catboost_base.cbm": package_dir / "model/catboost_base.cbm",
        "model/catboost_f_route.cbm": package_dir / "model/catboost_f_route.cbm",
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
            info.filename: {
                "bytes": info.file_size,
                "compressed_bytes": info.compress_size,
            }
            for info in archive.infolist()
        }
    if zip_path.stat().st_size >= 10 * 1024**3:
        raise RuntimeError("submission ZIP exceeds the 10 GiB competition limit")
    return {
        "zip_path": str(zip_path),
        "zip_bytes": int(zip_path.stat().st_size),
        "zip_sha256": BASE.sha256_file(zip_path),
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

    routing_selection = load_and_validate_routing_selection(args.routing_latest_path)
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
    if not set(data["game_type"].astype(str).unique()).issubset({"R", "F"}):
        raise ValueError("game_type must contain only R and F")
    raw_input_columns = [column for column in data.columns if column != TARGET_COL]
    print(
        f"rows={len(data):,} seasons={seasons} target_rate={data[TARGET_COL].mean():.6f} "
        f"selected_route={SELECTED_ROUTE}"
    )
    feature_parity = verify_training_inference_feature_parity(data, runtime_template)
    print(f"training/runtime feature parity: {feature_parity['status']}")

    (
        lgbm_metadata,
        lgbm_summary,
        lgbm_reference,
        lgbm_audit_raw,
        lgbm_importance,
    ) = train_lgbm(data, args=args, model_path=model_dir / "lgbm_model.txt")
    lgbm_importance.to_csv(output_dir / "lgbm_feature_importance.csv", index=False)
    del lgbm_importance

    (
        base_metadata,
        base_summary,
        base_reference,
        base_audit_raw,
        base_importance,
    ) = train_catboost(
        data,
        args=args,
        model_path=model_dir / "catboost_base.cbm",
        model_label="CatBoost base all_new",
        training_policy="all_2019_2024_R_F",
    )
    base_importance.to_csv(
        output_dir / "catboost_base_feature_importance.csv", index=False
    )
    del base_importance

    f_route_train = select_f_route_training_rows(data)
    (
        f_route_metadata,
        f_route_summary,
        f_route_reference,
        f_route_audit_raw,
        f_route_importance,
    ) = train_catboost(
        f_route_train,
        args=args,
        model_path=model_dir / "catboost_f_route.cbm",
        model_label="CatBoost F route shared_latest_f1",
        training_policy="all_R_2019_2024_plus_F_2024",
    )
    f_route_importance.to_csv(
        output_dir / "catboost_f_route_feature_importance.csv", index=False
    )
    del f_route_importance, f_route_train
    gc.collect()

    artifacts = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "competition": "LG Aimers 9th",
        "submission_version": 3,
        "target_column": TARGET_COL,
        "raw_input_columns": raw_input_columns,
        "selected_pair": SELECTED_PAIR,
        "selected_route": SELECTED_ROUTE,
        "lgbm": lgbm_metadata,
        "catboost_base": base_metadata,
        "catboost_f_route": f_route_metadata,
        "routing": {
            "column": "game_type",
            "base_value": "R",
            "base_model": "catboost_base",
            "f_route_value": "F",
            "f_route_model": "catboost_f_route",
            "row_independent": True,
        },
        "blend": {
            "method": BLEND_METHOD,
            "lgbm_weight": LGBM_WEIGHT,
            "catboost_weight": CATBOOST_WEIGHT,
            "calibration": "none",
        },
    }
    joblib.dump(artifacts, model_dir / "artifacts.joblib", compress=3)

    audit = verify_serialization_routing_and_row_independence(
        script_path=package_dir / "script.py",
        model_dir=model_dir,
        full_data=data,
        lgbm_audit_raw=lgbm_audit_raw,
        lgbm_reference=lgbm_reference,
        base_audit_raw=base_audit_raw,
        base_reference=base_reference,
        f_route_audit_raw=f_route_audit_raw,
        f_route_reference=f_route_reference,
    )
    del (
        lgbm_audit_raw,
        lgbm_reference,
        base_audit_raw,
        base_reference,
        f_route_audit_raw,
        f_route_reference,
    )
    gc.collect()

    end_to_end = BASE.verify_end_to_end(package_dir, args.data_dir)
    zip_path = output_dir / SUBMISSION_FILENAME
    zip_summary = build_and_verify_zip(package_dir, zip_path)
    if len(zip_path.name) > 30:
        raise RuntimeError(f"submission filename exceeds 30 characters: {zip_path.name}")

    summary = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "submission_filename": SUBMISSION_FILENAME,
        "selected_pair": SELECTED_PAIR,
        "selected_route": SELECTED_ROUTE,
        "train_path": str(args.train_path.resolve()),
        "train_file_bytes": int(args.train_path.stat().st_size),
        "train_file_sha256": BASE.sha256_file(args.train_path),
        "all_train_rows": int(len(data)),
        "all_train_seasons": seasons,
        "reference_season": REFERENCE_SEASON,
        "routing": artifacts["routing"],
        "blend": artifacts["blend"],
        "validation_reference": {
            "source_07b_signature": EXPECTED_ROUTING_SIGNATURE,
            "selection_seasons": [2023, 2024],
            "route": SELECTED_ROUTE,
            **routing_selection["selected_metrics"],
            "selection_checks": routing_selection["selection_checks"],
            "note": "Frozen 07b OOF reference only; no test statistic was used.",
        },
        "lgbm": lgbm_summary,
        "catboost_base": base_summary,
        "catboost_f_route": f_route_summary,
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
