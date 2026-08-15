#!/usr/bin/env python3
"""Forward-CV audit of CatBoost policies for the shifting F-game distribution.

The current submission-02 LightGBM component and its 65:35 logit blend weight
are kept fixed.  Only the CatBoost training policy is changed.  This separates
the effect of adding ``game_type`` from changes to the F training window,
recency weighting, and a dedicated F expert.  Every holdout prediction is made
using seasons strictly earlier than the holdout season.
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

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from evaluation import probability_metrics
from features import FeatureConfig, ID_COL, TARGET_COL, prepare_catboost_fold_features


RESULT_SCHEMA_VERSION = 1
VALID_SEASONS = (2022, 2023, 2024)
RECENT_SEASONS = (2023, 2024)
LGBM_WEIGHT = 0.65
BASE_POLICY = "current_all_nogt"
KEY_COLUMNS = [ID_COL, TARGET_COL, "season", "game_type", "valid_season"]
PROBABILITY_EPS = 1e-7
COMPARISON_TOLERANCE = 1e-12

FEATURE_FLAGS = {
    "include_rate_gap_features": True,
    "include_recent_dynamics_features": True,
    "include_pitchmix_shape_features": True,
    "include_context_interaction_features": True,
    "include_team_context_features": True,
}

# Policies are fixed before reading their validation scores. The current OOF
# is loaded from experiment 05 as the exact submission-recipe anchor; a
# matched no-game_type rerun separately measures GPU retraining variation.
POLICIES: dict[str, dict] = {
    BASE_POLICY: {
        "description": "Current CatBoost all_new OOF: all prior R/F, half-life 0.5, no game_type",
        "mode": "source",
        "include_game_type": False,
        "prediction_scope": "all",
        "selection_eligible": False,
    },
    "retrain_all_nogt": {
        "description": "Matched rerun control: all prior R/F, half-life 0.5, no game_type",
        "mode": "all",
        "include_game_type": False,
        "half_life": 0.5,
        "prediction_scope": "all",
        "selection_eligible": False,
    },
    "all_with_gt": {
        "description": "Current training rows and weights, with game_type included",
        "mode": "all",
        "include_game_type": True,
        "half_life": 0.5,
        "prediction_scope": "all",
        "selection_eligible": True,
    },
    "r_only": {
        "description": "Train CatBoost on prior R rows only and extrapolate to both R/F",
        "mode": "r_only",
        "include_game_type": False,
        "half_life": 0.5,
        "prediction_scope": "all",
        "selection_eligible": True,
    },
    "shared_f_fast_decay": {
        "description": "Shared model: R half-life 0.5 and F half-life 0.25",
        "mode": "all_group_decay",
        "include_game_type": False,
        "r_half_life": 0.5,
        "f_half_life": 0.25,
        "prediction_scope": "all",
        "selection_eligible": True,
    },
    "shared_latest_f1": {
        "description": "Shared model: all prior R plus only the immediately previous F season",
        "mode": "shared_latest_f",
        "include_game_type": False,
        "f_window": 1,
        "half_life": 0.5,
        "prediction_scope": "all",
        "selection_eligible": True,
    },
    "expert_f_fast_decay": {
        "description": "Keep current CatBoost on R; replace F with F-only half-life 0.25 expert",
        "mode": "f_expert",
        "include_game_type": False,
        "half_life": 0.25,
        "prediction_scope": "F",
        "selection_eligible": True,
    },
    "expert_f_latest1": {
        "description": "Keep current CatBoost on R; F-only expert uses previous one season",
        "mode": "f_expert_latest",
        "include_game_type": False,
        "f_window": 1,
        "half_life": 0.5,
        "prediction_scope": "F",
        "selection_eligible": True,
    },
    "expert_f_latest2": {
        "description": "Keep current CatBoost on R; F-only expert uses previous two seasons",
        "mode": "f_expert_latest",
        "include_game_type": False,
        "f_window": 2,
        "half_life": 0.5,
        "prediction_scope": "F",
        "selection_eligible": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data/train.csv")
    parser.add_argument(
        "--feature-output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/feature_ablation_v1",
    )
    parser.add_argument("--source-run-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/f_shift_cv_v1",
    )
    parser.add_argument("--valid-seasons", type=int, nargs="+", default=list(VALID_SEASONS))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--catboost-iterations", type=int, default=85)
    parser.add_argument(
        "--catboost-task-type", choices=["auto", "CPU", "GPU"], default="auto"
    )
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def experiment_signature(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def resolve_catboost_task_type(requested: str) -> str:
    if requested != "auto":
        return requested
    return "GPU" if shutil.which("nvidia-smi") else "CPU"


def resolve_source_run(args: argparse.Namespace) -> Path:
    if args.source_run_dir is not None:
        candidates = [args.source_run_dir]
    else:
        latest_path = args.feature_output_root / "latest_run.json"
        if not latest_path.exists():
            raise FileNotFoundError(
                f"05 latest_run.json is missing: {latest_path}. Run notebook 05 first."
            )
        with latest_path.open(encoding="utf-8") as handle:
            latest = json.load(handle)
        recorded = Path(latest["run_dir"])
        # The fallback makes a Drive folder portable if its absolute Colab path
        # changed between sessions.
        candidates = [recorded, args.feature_output_root / recorded.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"05 run directory is missing: {candidates}")


def validate_training_data(data: pd.DataFrame, valid_seasons: list[int]) -> None:
    required = {ID_COL, TARGET_COL, "season", "game_type"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"training data is missing columns: {missing}")
    if data[ID_COL].isna().any() or data[ID_COL].duplicated().any():
        raise ValueError("row_id must be non-null and unique")
    if data[TARGET_COL].isna().any() or not set(data[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError("control_success must be a non-null binary target")
    if not set(data["game_type"].astype(str).unique()).issubset({"R", "F"}):
        raise ValueError("game_type must contain only R and F")
    if tuple(valid_seasons) != VALID_SEASONS:
        raise ValueError(
            f"this experiment requires validation seasons {VALID_SEASONS}; got {valid_seasons}"
        )
    missing_seasons = sorted(set(valid_seasons) - set(data["season"].unique()))
    if missing_seasons:
        raise ValueError(f"validation seasons are missing: {missing_seasons}")


def validate_source_oof(frame: pd.DataFrame, path: Path, prediction_column: str) -> None:
    required = set(KEY_COLUMNS + [prediction_column])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"source OOF columns are missing from {path}: {missing}")
    if frame[ID_COL].isna().any() or frame[ID_COL].duplicated().any():
        raise ValueError(f"source OOF row_id must be non-null and unique: {path}")
    if tuple(sorted(int(x) for x in frame["valid_season"].unique())) != VALID_SEASONS:
        raise ValueError(f"unexpected validation seasons in {path}")
    values = frame[prediction_column].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values <= 0.0) | (values >= 1.0)).any():
        raise ValueError(f"invalid probabilities in {path}: {prediction_column}")


def aligned_values(left: pd.Series, right: pd.Series, column: str) -> bool:
    if column in (ID_COL, "game_type"):
        return np.array_equal(
            left.astype("string").to_numpy(), right.astype("string").to_numpy()
        )
    return np.array_equal(left.to_numpy(), right.to_numpy())


def load_fixed_components(source_run_dir: Path) -> tuple[pd.DataFrame, dict]:
    lgbm_path = source_run_dir / "oof_pitchmix_shape.csv.gz"
    catboost_path = source_run_dir / "oof_all_new.csv.gz"
    for path in (lgbm_path, catboost_path):
        if not path.exists():
            raise FileNotFoundError(f"required experiment-05 OOF is missing: {path}")

    print(f"Loading fixed LightGBM component: {lgbm_path.name}")
    lgbm = pd.read_csv(
        lgbm_path,
        usecols=KEY_COLUMNS + ["lgbm_prediction"],
        dtype={ID_COL: "string", "game_type": "string"},
    )
    print(f"Loading current CatBoost component: {catboost_path.name}")
    catboost = pd.read_csv(
        catboost_path,
        usecols=KEY_COLUMNS + ["catboost_prediction"],
        dtype={ID_COL: "string", "game_type": "string"},
    )
    validate_source_oof(lgbm, lgbm_path, "lgbm_prediction")
    validate_source_oof(catboost, catboost_path, "catboost_prediction")
    if len(lgbm) != len(catboost):
        raise RuntimeError("source OOF row counts do not match")
    for column in KEY_COLUMNS:
        if not aligned_values(lgbm[column], catboost[column], column):
            raise RuntimeError(f"source OOF alignment mismatch: {column}")

    components = lgbm.copy()
    components[f"catboost__{BASE_POLICY}"] = catboost["catboost_prediction"].to_numpy(
        dtype=np.float64
    )
    metadata = {
        "source_run_dir": str(source_run_dir),
        "lgbm_source": lgbm_path.name,
        "catboost_source": catboost_path.name,
        "lgbm_oof_sha256": sha256_file(lgbm_path),
        "catboost_oof_sha256": sha256_file(catboost_path),
        "rows": int(len(components)),
    }
    return components, metadata


def feature_config(include_game_type: bool) -> FeatureConfig:
    return FeatureConfig(
        include_season=False,
        include_player_ids=True,
        include_game_type=include_game_type,
        **FEATURE_FLAGS,
    )


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


def recency_weights(
    frame: pd.DataFrame,
    *,
    reference_season: int,
    policy: dict,
) -> np.ndarray:
    age = (reference_season - 1) - frame["season"].to_numpy(dtype=np.float64)
    if (age < 0).any():
        raise ValueError("training data contains a season beyond the reference season")
    if policy["mode"] == "all_group_decay":
        half_life = np.where(
            frame["game_type"].astype(str).eq("F").to_numpy(),
            float(policy["f_half_life"]),
            float(policy["r_half_life"]),
        )
        weights = np.power(0.5, age / half_life)
    else:
        weights = np.power(0.5, age / float(policy["half_life"]))
    if not np.isfinite(weights).all() or weights.sum() <= 0.0:
        raise RuntimeError("invalid sample weights")
    return weights / weights.mean()


def select_training_rows(
    data: pd.DataFrame, *, policy: dict, valid_season: int
) -> pd.DataFrame:
    history = data[data["season"].lt(valid_season)]
    mode = policy["mode"]
    if mode in ("all", "all_group_decay"):
        selected = history
    elif mode == "r_only":
        selected = history[history["game_type"].eq("R")]
    elif mode == "shared_latest_f":
        earliest_f = valid_season - int(policy["f_window"])
        selected = history[
            history["game_type"].eq("R")
            | (
                history["game_type"].eq("F")
                & history["season"].ge(earliest_f)
            )
        ]
    elif mode == "f_expert":
        selected = history[history["game_type"].eq("F")]
    elif mode == "f_expert_latest":
        earliest_f = valid_season - int(policy["f_window"])
        selected = history[
            history["game_type"].eq("F") & history["season"].ge(earliest_f)
        ]
    else:
        raise ValueError(f"unknown policy mode: {mode}")
    selected = selected.copy()
    if selected.empty or selected[TARGET_COL].nunique() != 2:
        raise RuntimeError(
            f"policy has insufficient training labels: mode={mode}, fold={valid_season}"
        )
    return selected


def cache_paths(run_dir: Path, policy_name: str, valid_season: int) -> tuple[Path, Path, Path]:
    directory = run_dir / "cache" / policy_name
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"fold_{valid_season}"
    return (
        stem.with_suffix(".csv.gz"),
        stem.with_name(stem.name + "_metadata.json"),
        stem.with_name(stem.name + "_importance.csv"),
    )


def save_fold_cache(
    paths: tuple[Path, Path, Path],
    valid: pd.DataFrame,
    prediction: np.ndarray,
    metadata: dict,
    importance: pd.DataFrame,
) -> None:
    prediction_path, metadata_path, importance_path = paths
    pd.DataFrame(
        {ID_COL: valid[ID_COL].astype("string"), "prediction": prediction}
    ).to_csv(prediction_path, index=False, compression="gzip")
    write_json(metadata_path, metadata)
    importance.to_csv(importance_path, index=False)


def load_fold_cache(
    paths: tuple[Path, Path, Path], valid: pd.DataFrame
) -> tuple[np.ndarray, dict, pd.DataFrame]:
    prediction_path, metadata_path, importance_path = paths
    cached = pd.read_csv(prediction_path, dtype={ID_COL: "string"})
    if not np.array_equal(
        valid[ID_COL].astype("string").to_numpy(), cached[ID_COL].to_numpy()
    ):
        raise RuntimeError(f"cache row order mismatch: {prediction_path}")
    prediction = cached["prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all() or (
        (prediction <= 0.0) | (prediction >= 1.0)
    ).any():
        raise RuntimeError(f"invalid cached probabilities: {prediction_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    importance = pd.read_csv(importance_path)
    print(f"    verified cache: {prediction_path.parent.name}/{prediction_path.name}")
    return prediction, metadata, importance


def train_catboost_fold(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    configuration: FeatureConfig,
    weights: np.ndarray,
    parameters: dict,
) -> tuple[np.ndarray, dict, pd.DataFrame]:
    x_train, x_valid, categorical, prior = prepare_catboost_fold_features(
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
    prediction = np.clip(
        model.predict_proba(x_valid)[:, 1], PROBABILITY_EPS, 1.0 - PROBABILITY_EPS
    )
    importance = pd.DataFrame(
        {"feature": x_train.columns, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False, ignore_index=True)
    metadata = {
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "train_target_rate": float(target.mean()),
        "valid_target_rate": float(valid[TARGET_COL].mean()),
        "train_f_fraction": float(train["game_type"].eq("F").mean()),
        "feature_count": int(x_train.shape[1]),
        "categorical_count": int(len(categorical)),
        "target_prior": prior,
        "train_seconds": elapsed,
        "iterations": int(parameters["iterations"]),
        "task_type": str(parameters["task_type"]),
        "feature_config": asdict(configuration),
    }
    del x_train, x_valid, target, model
    gc.collect()
    return prediction, metadata, importance


def logit_blend(
    lgbm_prediction: np.ndarray,
    catboost_prediction: np.ndarray,
    lgbm_weight: float = LGBM_WEIGHT,
) -> np.ndarray:
    p_lgbm = np.clip(np.asarray(lgbm_prediction, dtype=np.float64), PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)
    p_cat = np.clip(np.asarray(catboost_prediction, dtype=np.float64), PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)
    lgbm_logit = np.log(p_lgbm / (1.0 - p_lgbm))
    cat_logit = np.log(p_cat / (1.0 - p_cat))
    blended_logit = lgbm_weight * lgbm_logit + (1.0 - lgbm_weight) * cat_logit
    return np.clip(
        1.0 / (1.0 + np.exp(-np.clip(blended_logit, -30.0, 30.0))),
        PROBABILITY_EPS,
        1.0 - PROBABILITY_EPS,
    )


def scope_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(frame), dtype=bool),
        "R": frame["game_type"].astype(str).eq("R").to_numpy(),
        "F": frame["game_type"].astype(str).eq("F").to_numpy(),
    }


def metric_rows(oof: pd.DataFrame, policy_names: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    partitions: list[tuple[str, int | None, np.ndarray]] = [
        ("pooled", None, np.ones(len(oof), dtype=bool)),
        ("recent_pooled", None, oof["valid_season"].isin(RECENT_SEASONS).to_numpy()),
    ]
    partitions.extend(
        ("fold", season, oof["valid_season"].eq(season).to_numpy())
        for season in VALID_SEASONS
    )
    for policy_name in policy_names:
        for model, column in (
            ("catboost", f"catboost__{policy_name}"),
            ("fixed_logit_65_35", f"blend__{policy_name}"),
        ):
            for partition, season, partition_mask in partitions:
                frame = oof.loc[partition_mask]
                prediction = oof.loc[partition_mask, column].to_numpy(dtype=np.float64)
                for scope, scope_mask in scope_masks(frame).items():
                    if not scope_mask.any():
                        continue
                    metrics = probability_metrics(
                        frame.loc[scope_mask, TARGET_COL], prediction[scope_mask]
                    )
                    metrics.update(
                        {
                            "policy": policy_name,
                            "model": model,
                            "partition": partition,
                            "valid_season": season,
                            "scope": scope,
                        }
                    )
                    rows.append(metrics)
    return pd.DataFrame(rows)


def one_metric(
    frame: pd.DataFrame, *, partition: str, scope: str, season: int | None = None
) -> pd.Series:
    mask = frame["partition"].eq(partition) & frame["scope"].eq(scope)
    if season is not None:
        mask &= frame["valid_season"].eq(season)
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise RuntimeError(
            f"expected one metric row: partition={partition}, scope={scope}, season={season}"
        )
    return selected.iloc[0]


def build_policy_summary(metrics: pd.DataFrame, policy_names: list[str]) -> pd.DataFrame:
    blend = metrics[metrics["model"].eq("fixed_logit_65_35")]
    rows = []
    for policy_name in policy_names:
        frame = blend[blend["policy"].eq(policy_name)]
        recent_all = one_metric(frame, partition="recent_pooled", scope="all")
        recent_r = one_metric(frame, partition="recent_pooled", scope="R")
        recent_f = one_metric(frame, partition="recent_pooled", scope="F")
        row = {
            "policy": policy_name,
            "description": POLICIES[policy_name]["description"],
            "recent_brier": recent_all["brier"],
            "recent_bss_raw": recent_all["bss_raw"],
            "recent_mean_gap": recent_all["mean_gap"],
            "recent_r_brier": recent_r["brier"],
            "recent_r_bss_raw": recent_r["bss_raw"],
            "recent_f_brier": recent_f["brier"],
            "recent_f_bss_raw": recent_f["bss_raw"],
        }
        for season in VALID_SEASONS:
            for scope, label in (("all", ""), ("R", "_r"), ("F", "_f")):
                fold = one_metric(frame, partition="fold", scope=scope, season=season)
                row[f"brier{label}_{season}"] = fold["brier"]
                row[f"bss_raw{label}_{season}"] = fold["bss_raw"]
                row[f"mean_gap{label}_{season}"] = fold["mean_gap"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    base = summary[summary["policy"].eq(BASE_POLICY)].iloc[0]
    comparison_columns = [
        "recent_brier",
        "recent_r_brier",
        "recent_f_brier",
        *[f"brier_{season}" for season in VALID_SEASONS],
        *[f"brier_r_{season}" for season in VALID_SEASONS],
        *[f"brier_f_{season}" for season in VALID_SEASONS],
    ]
    for column in comparison_columns:
        summary[f"delta_{column}_vs_base"] = summary[column] - float(base[column])

    summary["improves_recent_all"] = summary["delta_recent_brier_vs_base"].lt(0.0)
    summary["improves_recent_f"] = summary["delta_recent_f_brier_vs_base"].lt(0.0)
    summary["safe_recent_folds_all"] = (
        summary["delta_brier_2023_vs_base"].le(COMPARISON_TOLERANCE)
        & summary["delta_brier_2024_vs_base"].le(COMPARISON_TOLERANCE)
    )
    summary["safe_recent_folds_r"] = (
        summary["delta_brier_r_2023_vs_base"].le(COMPARISON_TOLERANCE)
        & summary["delta_brier_r_2024_vs_base"].le(COMPARISON_TOLERANCE)
    )
    summary["safe_recent_folds_f"] = (
        summary["delta_brier_f_2023_vs_base"].le(COMPARISON_TOLERANCE)
        & summary["delta_brier_f_2024_vs_base"].le(COMPARISON_TOLERANCE)
    )
    summary["selection_eligible"] = summary["policy"].map(
        {name: bool(policy["selection_eligible"]) for name, policy in POLICIES.items()}
    )
    summary["passes_metric_gates"] = (
        summary["improves_recent_all"]
        & summary["improves_recent_f"]
        & summary["safe_recent_folds_all"]
        & summary["safe_recent_folds_r"]
        & summary["safe_recent_folds_f"]
    )
    summary["passes_strict_selection"] = (
        summary["selection_eligible"] & summary["passes_metric_gates"]
    )
    summary["worst_recent_fold_delta"] = summary[
        [
            "delta_brier_2023_vs_base",
            "delta_brier_2024_vs_base",
            "delta_brier_r_2023_vs_base",
            "delta_brier_r_2024_vs_base",
            "delta_brier_f_2023_vs_base",
            "delta_brier_f_2024_vs_base",
        ]
    ].max(axis=1)
    return summary.sort_values(
        ["recent_brier", "worst_recent_fold_delta"], ignore_index=True
    )


def paired_loss_deltas(oof: pd.DataFrame, policy_names: list[str]) -> pd.DataFrame:
    target = oof[TARGET_COL].to_numpy(dtype=np.float64)
    base_prediction = oof[f"blend__{BASE_POLICY}"].to_numpy(dtype=np.float64)
    base_loss = np.square(base_prediction - target)
    partitions: list[tuple[str, int | None, np.ndarray]] = [
        ("recent_pooled", None, oof["valid_season"].isin(RECENT_SEASONS).to_numpy())
    ]
    partitions.extend(
        ("fold", season, oof["valid_season"].eq(season).to_numpy())
        for season in VALID_SEASONS
    )
    rows = []
    for policy_name in policy_names:
        candidate = oof[f"blend__{policy_name}"].to_numpy(dtype=np.float64)
        loss_delta = np.square(candidate - target) - base_loss
        for partition, season, partition_mask in partitions:
            frame = oof.loc[partition_mask]
            values = loss_delta[partition_mask]
            for scope, scope_mask in scope_masks(frame).items():
                if not scope_mask.any():
                    continue
                scoped = values[scope_mask]
                rows.append(
                    {
                        "policy": policy_name,
                        "partition": partition,
                        "valid_season": season,
                        "scope": scope,
                        "n": int(len(scoped)),
                        "mean_loss_delta_vs_base": float(scoped.mean()),
                        "naive_standard_error": float(
                            scoped.std(ddof=1) / np.sqrt(len(scoped))
                        ),
                        "fraction_rows_improved": float((scoped < 0.0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def prediction_correlations(oof: pd.DataFrame, policy_names: list[str]) -> pd.DataFrame:
    rows = []
    partitions: list[tuple[str, int | None, np.ndarray]] = [
        ("recent_pooled", None, oof["valid_season"].isin(RECENT_SEASONS).to_numpy())
    ]
    partitions.extend(
        ("fold", season, oof["valid_season"].eq(season).to_numpy())
        for season in VALID_SEASONS
    )
    for policy_name in policy_names:
        for model_prefix in ("catboost", "blend"):
            base = oof[f"{model_prefix}__{BASE_POLICY}"].to_numpy(dtype=np.float64)
            candidate = oof[f"{model_prefix}__{policy_name}"].to_numpy(dtype=np.float64)
            for partition, season, partition_mask in partitions:
                frame = oof.loc[partition_mask]
                for scope, scope_mask in scope_masks(frame).items():
                    combined = partition_mask.copy()
                    combined[partition_mask] = scope_mask
                    if combined.sum() < 2:
                        continue
                    rows.append(
                        {
                            "policy": policy_name,
                            "model": model_prefix,
                            "partition": partition,
                            "valid_season": season,
                            "scope": scope,
                            "pearson_vs_base": float(
                                np.corrcoef(base[combined], candidate[combined])[0, 1]
                            ),
                            "mean_abs_prediction_delta_vs_base": float(
                                np.mean(np.abs(candidate[combined] - base[combined]))
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def build_f_rate_history(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["season", "game_type"], observed=True)[TARGET_COL]
        .agg(n="size", target_rate="mean")
        .reset_index()
        .sort_values(["game_type", "season"], ignore_index=True)
    )


def main() -> None:
    args = parse_args()
    valid_seasons = sorted(set(args.valid_seasons))
    task_type = resolve_catboost_task_type(args.catboost_task_type)
    source_run_dir = resolve_source_run(args)
    lgbm_source = source_run_dir / "oof_pitchmix_shape.csv.gz"
    catboost_source = source_run_dir / "oof_all_new.csv.gz"
    parameters = catboost_parameters(args, task_type)
    policy_manifest = pd.DataFrame(
        [{"policy": name, **definition} for name, definition in POLICIES.items()]
    )
    signature_payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "valid_seasons": valid_seasons,
        "recent_seasons": list(RECENT_SEASONS),
        "fixed_lgbm_weight": LGBM_WEIGHT,
        "base_policy": BASE_POLICY,
        "policies": POLICIES,
        "feature_flags": FEATURE_FLAGS,
        "catboost_parameters": parameters,
        "script_sha256": sha256_file(Path(__file__)),
        "features_sha256": sha256_file(SRC_DIR / "features.py"),
        "evaluation_sha256": sha256_file(SRC_DIR / "evaluation.py"),
        "train_sha256": sha256_file(args.train_path),
        "lgbm_oof_sha256": sha256_file(lgbm_source),
        "catboost_oof_sha256": sha256_file(catboost_source),
    }
    signature = experiment_signature(signature_payload)
    run_dir = args.output_dir / f"run_{signature}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "configuration.json",
        {
            "signature": signature,
            "source_run_dir": str(source_run_dir),
            **signature_payload,
        },
    )
    policy_manifest.to_csv(run_dir / "policy_manifest.csv", index=False)

    print(f"Loading {args.train_path} ...")
    data = pd.read_csv(
        args.train_path,
        encoding="utf-8-sig",
        low_memory=False,
        dtype={ID_COL: "string", "game_type": "string"},
    )
    validate_training_data(data, valid_seasons)
    components, input_audit = load_fixed_components(source_run_dir)
    for valid_season in valid_seasons:
        expected = data[data["season"].eq(valid_season)]
        actual = components[components["valid_season"].eq(valid_season)]
        for column in (ID_COL, TARGET_COL, "season", "game_type"):
            if not aligned_values(expected[column], actual[column], column):
                raise RuntimeError(
                    f"train/source OOF alignment mismatch: fold={valid_season}, column={column}"
                )
    print(
        f"rows={len(data):,} oof_rows={len(components):,} folds={valid_seasons} "
        f"CatBoost={task_type} policies={len(POLICIES)}"
    )

    all_training_metadata = []
    all_importance = []
    for policy_index, (policy_name, policy) in enumerate(POLICIES.items(), start=1):
        print(f"\n[{policy_index}/{len(POLICIES)}] {policy_name}: {policy['description']}")
        if policy["mode"] == "source":
            print("  using exact experiment-05 OOF; no retraining")
            continue
        fold_predictions = []
        for valid_season in valid_seasons:
            valid_all = data[data["season"].eq(valid_season)].copy()
            prediction_scope = str(policy["prediction_scope"])
            valid_model = (
                valid_all[valid_all["game_type"].eq("F")].copy()
                if prediction_scope == "F"
                else valid_all
            )
            train = select_training_rows(data, policy=policy, valid_season=valid_season)
            weights = recency_weights(
                train, reference_season=valid_season, policy=policy
            )
            paths = cache_paths(run_dir, policy_name, valid_season)
            cache_ready = all(path.exists() for path in paths)
            print(
                f"  fold={valid_season} train={len(train):,} "
                f"valid_{prediction_scope}={len(valid_model):,}"
            )
            if cache_ready and not args.force_rerun:
                prediction, metadata, importance = load_fold_cache(paths, valid_model)
            else:
                prediction, metadata, importance = train_catboost_fold(
                    train,
                    valid_model,
                    configuration=feature_config(bool(policy["include_game_type"])),
                    weights=weights,
                    parameters=parameters,
                )
                save_fold_cache(paths, valid_model, prediction, metadata, importance)
            metadata.update(
                {
                    "policy": policy_name,
                    "valid_season": valid_season,
                    "prediction_scope": prediction_scope,
                    "weight_min": float(weights.min()),
                    "weight_max": float(weights.max()),
                    "effective_sample_size": float(
                        np.square(weights.sum()) / np.square(weights).sum()
                    ),
                }
            )
            importance = importance.copy()
            importance["policy"] = policy_name
            importance["valid_season"] = valid_season
            all_training_metadata.append(metadata)
            all_importance.append(importance)

            if prediction_scope == "F":
                full_prediction = components.loc[
                    components["valid_season"].eq(valid_season),
                    f"catboost__{BASE_POLICY}",
                ].to_numpy(dtype=np.float64).copy()
                valid_mask = valid_all["game_type"].eq("F").to_numpy()
                if valid_mask.sum() != len(prediction):
                    raise RuntimeError(f"F prediction size mismatch: {policy_name}/{valid_season}")
                full_prediction[valid_mask] = prediction
                fold_predictions.append(full_prediction)
            else:
                fold_predictions.append(prediction)
            del valid_all, valid_model, train, weights, prediction, metadata, importance
            gc.collect()
        components[f"catboost__{policy_name}"] = np.concatenate(fold_predictions)
        del fold_predictions
        gc.collect()

    policy_names = list(POLICIES)
    for policy_name in policy_names:
        catboost_prediction = components[f"catboost__{policy_name}"].to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(catboost_prediction).all() or (
            (catboost_prediction <= 0.0) | (catboost_prediction >= 1.0)
        ).any():
            raise RuntimeError(f"invalid combined CatBoost OOF: {policy_name}")
        components[f"blend__{policy_name}"] = logit_blend(
            components["lgbm_prediction"].to_numpy(dtype=np.float64),
            catboost_prediction,
        )

    metrics = metric_rows(components, policy_names)
    policy_summary = build_policy_summary(metrics, policy_names)
    safe = policy_summary[policy_summary["passes_strict_selection"]]
    selected = (
        safe.sort_values(["recent_brier", "worst_recent_fold_delta"]).iloc[0]
        if not safe.empty
        else policy_summary[policy_summary["policy"].eq(BASE_POLICY)].iloc[0]
    )
    selected_policy = str(selected["policy"])

    loss_deltas = paired_loss_deltas(components, policy_names)
    correlations = prediction_correlations(components, policy_names)
    training_metadata = pd.DataFrame(all_training_metadata)
    importance = (
        pd.concat(all_importance, ignore_index=True)
        if all_importance
        else pd.DataFrame(columns=["feature", "importance", "policy", "valid_season"])
    )
    f_rate_history = build_f_rate_history(data)

    components.to_csv(run_dir / "oof_predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(run_dir / "policy_metrics.csv", index=False)
    policy_summary.to_csv(run_dir / "policy_summary.csv", index=False)
    loss_deltas.to_csv(run_dir / "paired_loss_deltas.csv", index=False)
    correlations.to_csv(run_dir / "prediction_correlations.csv", index=False)
    training_metadata.to_csv(run_dir / "training_metadata.csv", index=False)
    importance.to_csv(run_dir / "feature_importance.csv", index=False)
    f_rate_history.to_csv(run_dir / "f_rate_history.csv", index=False)

    base = policy_summary[policy_summary["policy"].eq(BASE_POLICY)].iloc[0]
    summary = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "signature": signature,
        "source_run_dir": str(source_run_dir),
        "input_audit": input_audit,
        "valid_seasons": valid_seasons,
        "selection_seasons": list(RECENT_SEASONS),
        "base_policy": BASE_POLICY,
        "fixed_lgbm_component": "experiment-05 pitchmix_shape LightGBM OOF",
        "fixed_blend": "65% LightGBM + 35% CatBoost in logit space",
        "base_recent_brier": float(base["recent_brier"]),
        "base_recent_bss_raw": float(base["recent_bss_raw"]),
        "selected_policy": selected_policy,
        "selected_recent_brier": float(selected["recent_brier"]),
        "selected_recent_bss_raw": float(selected["recent_bss_raw"]),
        "selected_delta_recent_brier_vs_base": float(
            selected["delta_recent_brier_vs_base"]
        ),
        "strict_safe_improving_policies": safe["policy"].tolist(),
        "selection_rule": (
            "Choose the lowest 2023+2024 pooled Brier only if it improves both "
            "overall and F pooled Brier and does not worsen overall, R, or F "
            "Brier in either 2023 or 2024; otherwise retain the current policy. "
            "The matched no-game_type rerun is a randomness diagnostic and is "
            "not selection-eligible. The 2022 fold is a pre-shift diagnostic "
            "and is not a gate."
        ),
        "catboost_task_type": task_type,
        "next_step": (
            "Review the executed notebook and result ZIP. If a non-base policy is "
            "selected, reproduce it in a final-training notebook; otherwise retain "
            "submit-02 and move to a genuinely different model family."
        ),
    }
    write_json(run_dir / "summary.json", summary)
    write_json(
        args.output_dir / "latest_run.json",
        {"signature": signature, "run_dir": str(run_dir), "summary": summary},
    )

    display_columns = [
        "policy",
        "recent_brier",
        "recent_bss_raw",
        "delta_recent_brier_vs_base",
        "delta_recent_f_brier_vs_base",
        "delta_brier_2023_vs_base",
        "delta_brier_2024_vs_base",
        "passes_strict_selection",
    ]
    print("\nF-shift policy comparison (fixed 65:35 logit blend)")
    print(policy_summary[display_columns].to_string(index=False))
    print(f"\nSelected policy: {selected_policy}")
    print(f"Strict safe improving policies: {safe['policy'].tolist() or 'none; base retained'}")
    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
