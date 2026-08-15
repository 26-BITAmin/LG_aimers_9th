#!/usr/bin/env python3
"""Forward-CV ablation of row-independent feature groups.

This experiment deliberately keeps the model recipes from the first official
submission fixed.  Only the feature bundle changes, so an OOF difference can
be attributed to the feature group rather than simultaneous retuning.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from evaluation import calibration_table, grouped_metrics, probability_metrics
from features import (
    FeatureConfig,
    ID_COL,
    TARGET_COL,
    add_rowwise_features,
    prepare_catboost_fold_features,
    prepare_fold_features,
)


RESULT_SCHEMA_VERSION = 1
SELECTION_MODEL = "fixed_logit_65_35"
SELECTION_SEASONS = (2023, 2024)
LGBM_BLEND_WEIGHT = 0.65

FEATURE_FLAG_NAMES = (
    "include_rate_gap_features",
    "include_recent_dynamics_features",
    "include_pitchmix_shape_features",
    "include_context_interaction_features",
    "include_team_context_features",
)

FEATURE_EXPERIMENTS: dict[str, dict] = {
    "base": {
        "description": "Exact feature setup used by the first submission",
        "feature_flags": {},
        "lgbm_include_game_type": True,
    },
    "base_no_game_type": {
        "description": "Base with the constant R-only LightGBM game_type removed",
        "feature_flags": {},
        "lgbm_include_game_type": False,
    },
    "rate_gaps": {
        "description": "Pitcher/batter rate differences and control balance",
        "feature_flags": {"include_rate_gap_features": True},
        "lgbm_include_game_type": True,
    },
    "recent_dynamics": {
        "description": "Recent form direction, volatility, and weighted form",
        "feature_flags": {"include_recent_dynamics_features": True},
        "lgbm_include_game_type": True,
    },
    "pitchmix_shape": {
        "description": "Pitch-mix entropy, concentration, contrasts, and reliability",
        "feature_flags": {"include_pitchmix_shape_features": True},
        "lgbm_include_game_type": True,
    },
    "context_interactions": {
        "description": "Count/base/inning/score/leverage interactions",
        "feature_flags": {"include_context_interaction_features": True},
        "lgbm_include_game_type": True,
    },
    "team_context": {
        "description": "Pitcher-view win expectancy and compact team interactions",
        "feature_flags": {"include_team_context_features": True},
        "lgbm_include_game_type": True,
    },
    "all_new": {
        "description": "All five new row-independent feature groups",
        "feature_flags": {name: True for name in FEATURE_FLAG_NAMES},
        "lgbm_include_game_type": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data/train.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/feature_ablation_v1",
    )
    parser.add_argument("--valid-seasons", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(FEATURE_EXPERIMENTS),
        default=list(FEATURE_EXPERIMENTS),
    )
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--lgbm-estimators", type=int, default=500)
    parser.add_argument("--catboost-iterations", type=int, default=85)
    parser.add_argument(
        "--catboost-task-type",
        choices=["auto", "CPU", "GPU"],
        default="auto",
    )
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def resolve_catboost_task_type(requested: str) -> str:
    if requested != "auto":
        return requested
    return "GPU" if shutil.which("nvidia-smi") else "CPU"


def experiment_signature(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def validate_training_data(data: pd.DataFrame, valid_seasons: list[int]) -> None:
    required = {ID_COL, TARGET_COL, "season", "game_type"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"training data is missing columns: {missing}")
    if data[ID_COL].isna().any() or data[ID_COL].astype("string").duplicated().any():
        raise ValueError("row_id must be non-null and unique")
    if data[TARGET_COL].isna().any() or not set(data[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError("control_success must be a non-null binary target")
    missing_seasons = sorted(set(valid_seasons) - set(data["season"].unique()))
    if missing_seasons:
        raise ValueError(f"validation seasons are missing: {missing_seasons}")
    required_validation_seasons = {2022, *SELECTION_SEASONS}
    if not required_validation_seasons.issubset(valid_seasons):
        raise ValueError(
            "this ablation requires validation seasons 2022, 2023, and 2024"
        )


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


def feature_config(config_name: str, model_name: str) -> FeatureConfig:
    experiment = FEATURE_EXPERIMENTS[config_name]
    flags = {name: False for name in FEATURE_FLAG_NAMES}
    flags.update(experiment["feature_flags"])
    if model_name == "lightgbm":
        return FeatureConfig(
            include_season=True,
            include_player_ids=False,
            include_game_type=bool(experiment["lgbm_include_game_type"]),
            **flags,
        )
    if model_name == "catboost":
        return FeatureConfig(
            include_season=False,
            include_player_ids=True,
            include_game_type=False,
            **flags,
        )
    raise ValueError(f"unknown model: {model_name}")


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


def catboost_parameters(args: argparse.Namespace, task_type: str) -> dict:
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
        "task_type": task_type,
        "thread_count": args.threads,
    }
    if task_type == "GPU":
        parameters.update({"devices": "0", "gpu_ram_part": 0.85})
    return parameters


def cache_paths(
    run_dir: Path,
    config_name: str,
    valid_season: int,
    model_name: str,
) -> tuple[Path, Path, Path]:
    directory = run_dir / "cache" / config_name
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"fold_{valid_season}_{model_name}"
    return (
        stem.with_suffix(".csv.gz"),
        stem.with_name(stem.name + "_metadata.json"),
        stem.with_name(stem.name + "_importance.csv"),
    )


def save_fold_cache(
    prediction_path: Path,
    metadata_path: Path,
    importance_path: Path,
    valid: pd.DataFrame,
    prediction: np.ndarray,
    metadata: dict,
    importance: pd.DataFrame,
) -> None:
    pd.DataFrame(
        {ID_COL: valid[ID_COL].astype(str), "prediction": prediction}
    ).to_csv(prediction_path, index=False, compression="gzip")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    importance.to_csv(importance_path, index=False)


def load_fold_cache(
    prediction_path: Path,
    metadata_path: Path,
    importance_path: Path,
    valid: pd.DataFrame,
) -> tuple[np.ndarray, dict, pd.DataFrame]:
    cached = pd.read_csv(prediction_path, dtype={ID_COL: "string"})
    expected_ids = valid[ID_COL].astype("string").to_numpy()
    if not np.array_equal(expected_ids, cached[ID_COL].to_numpy()):
        raise RuntimeError(f"cache row order mismatch: {prediction_path}")
    prediction = cached["prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all() or ((prediction <= 0.0) | (prediction >= 1.0)).any():
        raise RuntimeError(f"invalid cached probabilities: {prediction_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    importance = pd.read_csv(importance_path)
    print(f"verified cache: {config_name_from_path(prediction_path)}/{prediction_path.name}")
    return prediction, metadata, importance


def config_name_from_path(path: Path) -> str:
    return path.parent.name


def train_lgbm_fold(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    configuration: FeatureConfig,
    weights: np.ndarray,
    parameters: dict,
) -> tuple[np.ndarray, dict, pd.DataFrame]:
    x_train, x_valid, categorical, _, _ = prepare_fold_features(
        train, valid, config=configuration
    )
    target = train[TARGET_COL].to_numpy(dtype=np.int8)
    model = lgb.LGBMClassifier(**parameters)
    started = time.perf_counter()
    model.fit(
        x_train,
        target,
        sample_weight=weights,
        categorical_feature=categorical,
    )
    elapsed = time.perf_counter() - started
    prediction = np.clip(model.predict_proba(x_valid)[:, 1], 1e-7, 1.0 - 1e-7)
    importance = pd.DataFrame(
        {
            "feature": x_train.columns,
            "importance": model.booster_.feature_importance("gain"),
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    metadata = {
        "model": "lightgbm",
        "feature_config": asdict(configuration),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "feature_count": int(x_train.shape[1]),
        "categorical_count": int(len(categorical)),
        "train_seconds": elapsed,
        "iterations": int(parameters["n_estimators"]),
    }
    del x_train, x_valid, target, model
    gc.collect()
    return prediction, metadata, importance


def train_catboost_fold(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    configuration: FeatureConfig,
    weights: np.ndarray,
    parameters: dict,
) -> tuple[np.ndarray, dict, pd.DataFrame]:
    x_train, x_valid, categorical, _ = prepare_catboost_fold_features(
        train, valid, config=configuration
    )
    target = train[TARGET_COL].to_numpy(dtype=np.int8)
    model = CatBoostClassifier(**parameters)
    started = time.perf_counter()
    model.fit(
        x_train,
        target,
        cat_features=categorical,
        sample_weight=weights,
        verbose=25,
    )
    elapsed = time.perf_counter() - started
    prediction = np.clip(model.predict_proba(x_valid)[:, 1], 1e-7, 1.0 - 1e-7)
    importance = pd.DataFrame(
        {"feature": x_train.columns, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False, ignore_index=True)
    metadata = {
        "model": "catboost",
        "feature_config": asdict(configuration),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "feature_count": int(x_train.shape[1]),
        "categorical_count": int(len(categorical)),
        "train_seconds": elapsed,
        "iterations": int(parameters["iterations"]),
        "task_type": str(parameters["task_type"]),
    }
    del x_train, x_valid, target, model
    gc.collect()
    return prediction, metadata, importance


def logit_blend(
    lgbm_prediction: np.ndarray,
    catboost_prediction: np.ndarray,
    lgbm_weight: float = LGBM_BLEND_WEIGHT,
) -> np.ndarray:
    eps = 1e-7
    lgbm_prediction = np.clip(lgbm_prediction, eps, 1.0 - eps)
    catboost_prediction = np.clip(catboost_prediction, eps, 1.0 - eps)
    lgbm_logit = np.log(lgbm_prediction / (1.0 - lgbm_prediction))
    catboost_logit = np.log(catboost_prediction / (1.0 - catboost_prediction))
    blended_logit = lgbm_weight * lgbm_logit + (1.0 - lgbm_weight) * catboost_logit
    return 1.0 / (1.0 + np.exp(-np.clip(blended_logit, -30.0, 30.0)))


def metric_rows(
    oof: pd.DataFrame,
    *,
    config_name: str,
    valid_seasons: list[int],
) -> pd.DataFrame:
    prediction_columns = {
        "lightgbm": "lgbm_prediction",
        "catboost": "catboost_prediction",
        SELECTION_MODEL: "blend_prediction",
    }
    partitions: list[tuple[str, int | None, pd.DataFrame]] = [
        ("pooled", None, oof),
        (
            "recent_pooled",
            None,
            oof[oof["valid_season"].isin(SELECTION_SEASONS)],
        ),
    ]
    partitions.extend(
        ("fold", season, oof[oof["valid_season"].eq(season)])
        for season in valid_seasons
    )
    rows = []
    for model_name, prediction_column in prediction_columns.items():
        for partition, season, frame in partitions:
            scopes = {
                "all": np.ones(len(frame), dtype=bool),
                "R": frame["game_type"].eq("R").to_numpy(),
                "F": frame["game_type"].eq("F").to_numpy(),
            }
            for scope, mask in scopes.items():
                if not mask.any():
                    continue
                row = probability_metrics(
                    frame.loc[mask, TARGET_COL], frame.loc[mask, prediction_column]
                )
                row.update(
                    {
                        "config": config_name,
                        "model": model_name,
                        "partition": partition,
                        "valid_season": season,
                        "scope": scope,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def group_metric_rows(
    data: pd.DataFrame,
    oof: pd.DataFrame,
    *,
    config_name: str,
    valid_seasons: list[int],
) -> pd.DataFrame:
    prediction_columns = {
        "lightgbm": "lgbm_prediction",
        "catboost": "catboost_prediction",
        SELECTION_MODEL: "blend_prediction",
    }
    parts = []
    for valid_season in valid_seasons:
        valid = data[data["season"].eq(valid_season)]
        fold_oof = oof[oof["valid_season"].eq(valid_season)]
        if not np.array_equal(
            valid[ID_COL].astype("string").to_numpy(),
            fold_oof[ID_COL].astype("string").to_numpy(),
        ):
            raise RuntimeError(f"OOF row order mismatch in fold {valid_season}")
        lgbm_history = data[
            data["season"].lt(valid_season) & data["game_type"].eq("R")
        ]
        frame = grouped_metrics(
            lgbm_history,
            valid,
            fold_oof[prediction_columns["lightgbm"]].to_numpy(dtype=np.float64),
        )
        frame["config"] = config_name
        frame["model"] = "lightgbm"
        frame["valid_season"] = valid_season
        parts.append(frame)
        del lgbm_history, frame

        all_history = data[data["season"].lt(valid_season)]
        for model_name in ("catboost", SELECTION_MODEL):
            prediction_column = prediction_columns[model_name]
            frame = grouped_metrics(
                all_history,
                valid,
                fold_oof[prediction_column].to_numpy(dtype=np.float64),
            )
            frame["config"] = config_name
            frame["model"] = model_name
            frame["valid_season"] = valid_season
            parts.append(frame)
        del all_history, frame, valid, fold_oof
    return pd.concat(parts, ignore_index=True)


def prediction_correlations(
    oof: pd.DataFrame,
    *,
    config_name: str,
    valid_seasons: list[int],
) -> pd.DataFrame:
    partitions: list[tuple[str, int | None, pd.DataFrame]] = [
        ("pooled", None, oof),
        (
            "recent_pooled",
            None,
            oof[oof["valid_season"].isin(SELECTION_SEASONS)],
        ),
    ]
    partitions.extend(
        ("fold", season, oof[oof["valid_season"].eq(season)])
        for season in valid_seasons
    )
    rows = []
    for partition, season, frame in partitions:
        rows.append(
            {
                "config": config_name,
                "partition": partition,
                "valid_season": season,
                "pearson": float(
                    frame[["lgbm_prediction", "catboost_prediction"]]
                    .corr()
                    .iloc[0, 1]
                ),
            }
        )
    return pd.DataFrame(rows)


def calibration_rows(
    oof: pd.DataFrame,
    *,
    config_name: str,
    valid_seasons: list[int],
) -> pd.DataFrame:
    """Build diagnostic calibration bins without fitting any calibrator."""
    prediction_columns = {
        "lightgbm": "lgbm_prediction",
        "catboost": "catboost_prediction",
        SELECTION_MODEL: "blend_prediction",
    }
    partitions: list[tuple[str, int | None, pd.DataFrame]] = [
        (
            "recent_pooled",
            None,
            oof[oof["valid_season"].isin(SELECTION_SEASONS)],
        )
    ]
    partitions.extend(
        ("fold", season, oof[oof["valid_season"].eq(season)])
        for season in valid_seasons
    )
    parts = []
    for model_name, prediction_column in prediction_columns.items():
        for partition, season, frame in partitions:
            scopes = {
                "all": np.ones(len(frame), dtype=bool),
                "R": frame["game_type"].eq("R").to_numpy(),
                "F": frame["game_type"].eq("F").to_numpy(),
            }
            for scope, mask in scopes.items():
                if mask.sum() < 20:
                    continue
                table = calibration_table(
                    frame.loc[mask, TARGET_COL],
                    frame.loc[mask, prediction_column],
                    n_bins=20,
                )
                table.insert(0, "bin_index", np.arange(len(table)))
                table["config"] = config_name
                table["model"] = model_name
                table["partition"] = partition
                table["valid_season"] = season
                table["scope"] = scope
                parts.append(table)
    return pd.concat(parts, ignore_index=True)


def build_feature_manifest(
    data: pd.DataFrame,
    config_names: list[str],
) -> pd.DataFrame:
    sample = data.iloc[: min(64, len(data))]
    rows = []
    base_features: dict[str, set[str]] = {}
    for model_name in ("lightgbm", "catboost"):
        configuration = feature_config("base", model_name)
        frame = add_rowwise_features(
            sample,
            target_prior=float(data[TARGET_COL].mean()),
            config=configuration,
        )
        base_features[model_name] = set(frame.columns)
    for config_name in config_names:
        for model_name in ("lightgbm", "catboost"):
            configuration = feature_config(config_name, model_name)
            frame = add_rowwise_features(
                sample,
                target_prior=float(data[TARGET_COL].mean()),
                config=configuration,
            )
            added = set(frame.columns) - base_features[model_name]
            removed = base_features[model_name] - set(frame.columns)
            for position, column in enumerate(frame.columns):
                rows.append(
                    {
                        "config": config_name,
                        "model": model_name,
                        "position": position,
                        "feature": column,
                        "is_added_vs_base": column in added,
                        "is_removed_vs_base": False,
                    }
                )
            for column in sorted(removed):
                rows.append(
                    {
                        "config": config_name,
                        "model": model_name,
                        "position": -1,
                        "feature": column,
                        "is_added_vs_base": False,
                        "is_removed_vs_base": True,
                    }
                )
    return pd.DataFrame(rows)


def run_row_independence_audit(data: pd.DataFrame) -> dict:
    sample = data.sample(n=min(32, len(data)), random_state=2026).copy()
    prior = float(data[TARGET_COL].mean())
    configuration = feature_config("all_new", "catboost")

    def build(frame: pd.DataFrame) -> pd.DataFrame:
        features = add_rowwise_features(frame, target_prior=prior, config=configuration)
        features.index = frame[ID_COL].astype("string")
        return features.sort_index()

    batched = build(sample)
    shuffled = build(sample.sample(frac=1.0, random_state=42))
    single_rows = pd.concat([build(sample.iloc[[i]]) for i in range(len(sample))]).sort_index()
    pd.testing.assert_frame_equal(batched, shuffled, check_dtype=False, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(batched, single_rows, check_dtype=False, rtol=1e-12, atol=1e-12)
    numeric = batched.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if np.isinf(numeric).any():
        raise RuntimeError("infinite value found in all_new features")
    return {
        "status": "passed",
        "sample_rows": int(len(sample)),
        "feature_count": int(batched.shape[1]),
        "shuffle_invariant": True,
        "single_row_invariant": True,
        "infinite_numeric_values": 0,
    }


def build_config_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    scope = metrics[metrics["scope"].eq("all")]
    rows = []
    for (config_name, model_name), frame in scope.groupby(["config", "model"], sort=False):
        def one(partition: str, season: int | None = None) -> pd.Series:
            mask = frame["partition"].eq(partition)
            if season is not None:
                mask &= frame["valid_season"].eq(season)
            selected = frame.loc[mask]
            if len(selected) != 1:
                raise RuntimeError(
                    f"expected one metric row: config={config_name} model={model_name} "
                    f"partition={partition} season={season}"
                )
            return selected.iloc[0]

        pooled = one("pooled")
        recent = one("recent_pooled")
        fold_2022 = one("fold", 2022)
        fold_2023 = one("fold", 2023)
        fold_2024 = one("fold", 2024)
        rows.append(
            {
                "config": config_name,
                "model": model_name,
                "feature_description": FEATURE_EXPERIMENTS[config_name]["description"],
                "pooled_brier": pooled["brier"],
                "pooled_bss_raw": pooled["bss_raw"],
                "recent_brier": recent["brier"],
                "recent_bss_raw": recent["bss_raw"],
                "recent_mean_gap": recent["mean_gap"],
                "recent_roc_auc": recent["roc_auc"],
                "brier_2022": fold_2022["brier"],
                "brier_2023": fold_2023["brier"],
                "brier_2024": fold_2024["brier"],
                "bss_raw_2022": fold_2022["bss_raw"],
                "bss_raw_2023": fold_2023["bss_raw"],
                "bss_raw_2024": fold_2024["bss_raw"],
            }
        )
    summary = pd.DataFrame(rows)
    reference = summary[summary["config"].eq("base")].set_index("model")
    for index, row in summary.iterrows():
        baseline = reference.loc[row["model"]]
        for column in ("recent_brier", "brier_2022", "brier_2023", "brier_2024"):
            summary.loc[index, f"delta_{column}_vs_base"] = row[column] - baseline[column]
        summary.loc[index, "delta_recent_bss_raw_vs_base"] = (
            row["recent_bss_raw"] - baseline["recent_bss_raw"]
        )
    summary["improves_recent"] = summary["delta_recent_brier_vs_base"].lt(0.0)
    summary["safe_2023_2024"] = (
        summary["delta_brier_2023_vs_base"].le(0.0)
        & summary["delta_brier_2024_vs_base"].le(0.0)
    )
    summary["worst_recent_fold_brier"] = summary[["brier_2023", "brier_2024"]].max(axis=1)
    return summary.sort_values(
        ["model", "recent_brier", "worst_recent_fold_brier"], ignore_index=True
    )


def choose_safe_config(config_summary: pd.DataFrame) -> tuple[str, list[str]]:
    candidates = config_summary[
        config_summary["model"].eq(SELECTION_MODEL)
        & config_summary["improves_recent"]
        & config_summary["safe_2023_2024"]
    ].sort_values(["recent_brier", "worst_recent_fold_brier"])
    safe_configs = candidates["config"].tolist()
    return (safe_configs[0] if safe_configs else "base"), safe_configs


def main() -> None:
    args = parse_args()
    valid_seasons = sorted(set(args.valid_seasons))
    config_names = list(dict.fromkeys(args.configs))
    if "base" not in config_names:
        raise ValueError("the base config is required for delta calculations")
    task_type = resolve_catboost_task_type(args.catboost_task_type)
    lgbm_params = lgbm_parameters(args)
    catboost_params = catboost_parameters(args, task_type)
    source_hashes = {
        "experiment": sha256_file(Path(__file__)),
        "features": sha256_file(SRC_DIR / "features.py"),
        "evaluation": sha256_file(SRC_DIR / "evaluation.py"),
        "train_data": sha256_file(args.train_path),
    }
    signature_payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "valid_seasons": valid_seasons,
        "selection_seasons": list(SELECTION_SEASONS),
        "configs": {name: FEATURE_EXPERIMENTS[name] for name in config_names},
        "lgbm_parameters": lgbm_params,
        "catboost_parameters": catboost_params,
        "source_hashes": source_hashes,
    }
    signature = experiment_signature(signature_payload)
    run_dir = args.output_dir / f"run_{signature}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "configuration.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"signature": signature, **signature_payload},
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Loading {args.train_path} ...")
    data = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False)
    validate_training_data(data, valid_seasons)
    print(
        f"rows={len(data):,} configs={len(config_names)} folds={valid_seasons} "
        f"CatBoost={task_type} run={run_dir.name}"
    )
    audit = run_row_independence_audit(data)
    with (run_dir / "row_independence_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    print(f"row-independence audit: {audit['status']}")

    manifest = build_feature_manifest(data, config_names)
    manifest.to_csv(run_dir / "feature_manifest.csv", index=False)

    all_metrics = []
    all_group_metrics = []
    all_correlations = []
    all_calibration = []
    all_training_metadata = []
    all_importance = []

    for config_index, config_name in enumerate(config_names, start=1):
        print(
            f"\n[{config_index}/{len(config_names)}] {config_name}: "
            f"{FEATURE_EXPERIMENTS[config_name]['description']}"
        )
        oof_parts = []
        for valid_season in valid_seasons:
            valid = data[data["season"].eq(valid_season)].copy()
            print(f"  fold={valid_season} valid={len(valid):,}")
            predictions = {}

            lgbm_train = data[
                data["season"].lt(valid_season) & data["game_type"].eq("R")
            ].copy()
            lgbm_weights = recency_weights(
                lgbm_train["season"], reference_season=valid_season, half_life=2.5
            )
            lgbm_cache = cache_paths(run_dir, config_name, valid_season, "lightgbm")
            if all(path.exists() for path in lgbm_cache) and not args.force_rerun:
                prediction, metadata, importance = load_fold_cache(
                    *lgbm_cache, valid
                )
            else:
                prediction, metadata, importance = train_lgbm_fold(
                    lgbm_train,
                    valid,
                    configuration=feature_config(config_name, "lightgbm"),
                    weights=lgbm_weights,
                    parameters=lgbm_params,
                )
                save_fold_cache(*lgbm_cache, valid, prediction, metadata, importance)
            metadata.update({"config": config_name, "valid_season": valid_season})
            importance = importance.copy()
            importance["config"] = config_name
            importance["valid_season"] = valid_season
            importance["model"] = "lightgbm"
            all_training_metadata.append(metadata)
            all_importance.append(importance)
            predictions["lightgbm"] = prediction
            del lgbm_train, lgbm_weights, prediction, metadata, importance
            gc.collect()

            catboost_train = data[data["season"].lt(valid_season)].copy()
            catboost_weights = recency_weights(
                catboost_train["season"], reference_season=valid_season, half_life=0.5
            )
            # base_no_game_type changes only the R-only LightGBM matrix.  Its
            # CatBoost matrix is exactly the base matrix, so reuse the same
            # cache to avoid a redundant GPU fit and a nondeterminism confound.
            catboost_cache_config = (
                "base" if config_name == "base_no_game_type" else config_name
            )
            catboost_cache = cache_paths(
                run_dir, catboost_cache_config, valid_season, "catboost"
            )
            catboost_cache_ready = all(path.exists() for path in catboost_cache)
            must_reuse_base_catboost = config_name == "base_no_game_type"
            if catboost_cache_ready and (
                not args.force_rerun or must_reuse_base_catboost
            ):
                prediction, metadata, importance = load_fold_cache(
                    *catboost_cache, valid
                )
            else:
                prediction, metadata, importance = train_catboost_fold(
                    catboost_train,
                    valid,
                    configuration=feature_config(config_name, "catboost"),
                    weights=catboost_weights,
                    parameters=catboost_params,
                )
                save_fold_cache(*catboost_cache, valid, prediction, metadata, importance)
            metadata.update({"config": config_name, "valid_season": valid_season})
            importance = importance.copy()
            importance["config"] = config_name
            importance["valid_season"] = valid_season
            importance["model"] = "catboost"
            all_training_metadata.append(metadata)
            all_importance.append(importance)
            predictions["catboost"] = prediction
            del catboost_train, catboost_weights, prediction, metadata, importance
            gc.collect()

            fold_oof = valid[[ID_COL, TARGET_COL, "season", "game_type"]].copy()
            fold_oof["valid_season"] = valid_season
            fold_oof["lgbm_prediction"] = predictions["lightgbm"]
            fold_oof["catboost_prediction"] = predictions["catboost"]
            fold_oof["blend_prediction"] = logit_blend(
                predictions["lightgbm"], predictions["catboost"]
            )
            oof_parts.append(fold_oof)
            del valid, predictions, fold_oof
            gc.collect()

        oof = pd.concat(oof_parts, ignore_index=True)
        oof_path = run_dir / f"oof_{config_name}.csv.gz"
        oof.to_csv(oof_path, index=False, compression="gzip")
        all_metrics.append(
            metric_rows(oof, config_name=config_name, valid_seasons=valid_seasons)
        )
        all_group_metrics.append(
            group_metric_rows(
                data,
                oof,
                config_name=config_name,
                valid_seasons=valid_seasons,
            )
        )
        all_correlations.append(
            prediction_correlations(
                oof,
                config_name=config_name,
                valid_seasons=valid_seasons,
            )
        )
        all_calibration.append(
            calibration_rows(
                oof,
                config_name=config_name,
                valid_seasons=valid_seasons,
            )
        )
        recent = oof[oof["valid_season"].isin(SELECTION_SEASONS)]
        recent_metric = probability_metrics(recent[TARGET_COL], recent["blend_prediction"])
        print(
            f"  fixed blend recent Brier={recent_metric['brier']:.9f} "
            f"BSS_raw={recent_metric['bss_raw']:.3f}"
        )
        del oof, oof_parts, recent, recent_metric
        gc.collect()

    metrics = pd.concat(all_metrics, ignore_index=True)
    group_metrics_frame = pd.concat(all_group_metrics, ignore_index=True)
    correlations = pd.concat(all_correlations, ignore_index=True)
    calibration = pd.concat(all_calibration, ignore_index=True)
    training_metadata = pd.DataFrame(all_training_metadata)
    importance = pd.concat(all_importance, ignore_index=True)
    config_summary = build_config_summary(metrics)
    selected_config, safe_configs = choose_safe_config(config_summary)

    metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    group_metrics_frame.to_csv(run_dir / "group_metrics.csv", index=False)
    correlations.to_csv(run_dir / "prediction_correlations.csv", index=False)
    calibration.to_csv(run_dir / "calibration_tables.csv", index=False)
    training_metadata.to_csv(run_dir / "training_metadata.csv", index=False)
    importance.to_csv(run_dir / "feature_importance.csv", index=False)
    config_summary.to_csv(run_dir / "config_summary.csv", index=False)

    selected_row = config_summary[
        config_summary["config"].eq(selected_config)
        & config_summary["model"].eq(SELECTION_MODEL)
    ].iloc[0]
    summary = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "signature": signature,
        "valid_seasons": valid_seasons,
        "selection_seasons": list(SELECTION_SEASONS),
        "selection_model": SELECTION_MODEL,
        "fixed_lgbm_weight": LGBM_BLEND_WEIGHT,
        "fixed_catboost_weight": 1.0 - LGBM_BLEND_WEIGHT,
        "selected_config": selected_config,
        "safe_improving_configs": safe_configs,
        "selection_rule": (
            "Lowest recent-pooled Brier among configs that improve the base on "
            "recent pooled and do not worsen either 2023 or 2024; otherwise base."
        ),
        "selected_recent_brier": float(selected_row["recent_brier"]),
        "selected_recent_bss_raw": float(selected_row["recent_bss_raw"]),
        "selected_delta_recent_brier_vs_base": float(
            selected_row["delta_recent_brier_vs_base"]
        ),
        "catboost_task_type": task_type,
        "row_independence_audit": audit,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "latest_run.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"signature": signature, "run_dir": str(run_dir), "summary": summary},
            handle,
            ensure_ascii=False,
            indent=2,
        )

    display_columns = [
        "config",
        "recent_brier",
        "recent_bss_raw",
        "delta_recent_brier_vs_base",
        "delta_brier_2023_vs_base",
        "delta_brier_2024_vs_base",
        "safe_2023_2024",
    ]
    print("\nFixed 65:35 logit blend feature comparison")
    print(
        config_summary[config_summary["model"].eq(SELECTION_MODEL)]
        .sort_values("recent_brier")[display_columns]
        .to_string(index=False)
    )
    print(f"\nSelected config: {selected_config}")
    print(f"Safe improving configs: {safe_configs or 'none; base retained'}")
    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
