#!/usr/bin/env python3
"""Leakage-safe forward CV for an independent TabM probability model.

The experiment trains only on regular-season (R) rows strictly earlier than
each validation season.  ``season`` and ``game_type`` are excluded from the
model input.  Categorical IDs are represented by train-only embeddings and a
small set of target encodings is constructed from strictly earlier seasons.

TabM is evaluated as a standalone model and as a probability blend with the
selected experiment-07b route.  Candidate blends modify R rows only; every F
prediction remains bit-for-bit equal to the frozen 07b anchor.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tabm import EnsembleView, LinearEnsemble, make_tabm_backbone
from torch.utils.data import DataLoader, TensorDataset

from evaluation import probability_metrics
from features import (
    FeatureConfig,
    ID_COL,
    TARGET_COL,
    add_rowwise_features,
    categorical_columns,
)


RESULT_SCHEMA_VERSION = 1
VALID_SEASONS = (2022, 2023, 2024)
RECENT_SEASONS = (2023, 2024)
LOSSES = ("bce", "brier")
BLEND_WEIGHTS = (0.10, 0.20, 0.30, 0.40)
PROBABILITY_EPS = 1e-7
COMPARISON_TOLERANCE = 1e-12
MIN_RECENT_BRIER_IMPROVEMENT = 1e-5
MAX_FOLD_BRIER_REGRESSION = 2e-5
DEFAULT_SEED = 2026

FEATURE_CONFIG = FeatureConfig(
    include_season=False,
    include_player_ids=True,
    include_game_type=False,
    include_rate_gap_features=True,
    include_recent_dynamics_features=True,
    include_pitchmix_shape_features=True,
    include_context_interaction_features=True,
    include_team_context_features=True,
)

# The smoothing strengths are fixed before any TabM OOF result is observed.
TARGET_ENCODING_GROUPS: dict[str, dict] = {
    "pitcher": {"columns": ["pitcher_id"], "strength": 100.0},
    "batter": {"columns": ["batter_id"], "strength": 150.0},
    "pitcher_count": {
        "columns": ["pitcher_id", "count_state"],
        "strength": 75.0,
    },
    "batter_count": {
        "columns": ["batter_id", "count_state"],
        "strength": 100.0,
    },
    "pitcher_vs_batter_hand": {
        "columns": ["pitcher_id", "batter_hand"],
        "strength": 75.0,
    },
    "batter_vs_pitcher_hand": {
        "columns": ["batter_id", "pitcher_hand"],
        "strength": 100.0,
    },
    "team_matchup": {
        "columns": ["pitcher_team_id", "batter_team_id"],
        "strength": 200.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path", type=Path, default=PROJECT_ROOT / "data/train.csv"
    )
    parser.add_argument(
        "--routing-output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/f_routing_v1",
    )
    parser.add_argument("--source-run-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/tabm_cv_v1",
    )
    parser.add_argument(
        "--valid-seasons", type=int, nargs="+", default=list(VALID_SEASONS)
    )
    parser.add_argument("--losses", nargs="+", choices=LOSSES, default=list(LOSSES))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0003)
    parser.add_argument("--half-life", type=float, default=1.5)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--d-block", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device(requested)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def recency_weights(
    seasons: Iterable,
    *,
    reference_season: int,
    half_life: float,
    normalize: bool,
) -> np.ndarray:
    season_values = np.asarray(seasons, dtype=np.float64)
    age = (reference_season - 1) - season_values
    if (age < 0).any():
        raise ValueError("source data contains a season beyond the reference season")
    weights = np.power(0.5, age / half_life)
    if normalize:
        weights = weights / weights.mean()
    return weights.astype(np.float32)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.dot(values, weights) / weights.sum())


def validate_training_data(data: pd.DataFrame, valid_seasons: list[int]) -> None:
    required = {ID_COL, TARGET_COL, "season", "game_type"}
    for definition in TARGET_ENCODING_GROUPS.values():
        required.update(c for c in definition["columns"] if c != "count_state")
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"training data is missing columns: {missing}")
    if data[ID_COL].isna().any() or data[ID_COL].astype("string").duplicated().any():
        raise ValueError("row_id must be non-null and unique")
    if data[TARGET_COL].isna().any():
        raise ValueError("control_success must be non-null")
    if not set(data[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError("control_success must be binary")
    if not set(data["game_type"].astype(str).unique()).issubset({"R", "F"}):
        raise ValueError("game_type must contain only R and F")
    if tuple(valid_seasons) != VALID_SEASONS:
        raise ValueError(
            f"this experiment requires validation seasons {VALID_SEASONS}; "
            f"got {valid_seasons}"
        )
    missing_seasons = sorted(set(valid_seasons) - set(data["season"].unique()))
    if missing_seasons:
        raise ValueError(f"validation seasons are missing: {missing_seasons}")


def resolve_source_run(args: argparse.Namespace) -> Path:
    if args.source_run_dir is not None:
        candidates = [args.source_run_dir]
    else:
        latest_path = args.routing_output_root / "latest_run.json"
        if not latest_path.exists():
            raise FileNotFoundError(
                f"07b latest_run.json is missing: {latest_path}. "
                "Run notebook 07b or restore its outputs first."
            )
        with latest_path.open(encoding="utf-8") as handle:
            latest = json.load(handle)
        recorded = Path(latest["run_dir"])
        candidates = [recorded, args.routing_output_root / recorded.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"07b run directory is missing: {candidates}")


def load_base_oof(source_run_dir: Path) -> tuple[pd.DataFrame, dict]:
    summary_path = source_run_dir / "summary.json"
    oof_path = source_run_dir / "route_oof.csv.gz"
    for path in (summary_path, oof_path):
        if not path.exists():
            raise FileNotFoundError(f"required 07b artifact is missing: {path}")
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    selected_route = summary.get("selected_route")
    if not selected_route:
        raise RuntimeError("07b summary does not contain selected_route")
    prediction_column = f"blend__{selected_route}"
    available = pd.read_csv(oof_path, nrows=0).columns.tolist()
    required = [
        ID_COL,
        TARGET_COL,
        "season",
        "game_type",
        "valid_season",
        prediction_column,
    ]
    missing = sorted(set(required) - set(available))
    if missing:
        raise ValueError(f"07b route OOF columns are missing: {missing}")
    frame = pd.read_csv(
        oof_path,
        usecols=required,
        dtype={ID_COL: "string", "game_type": "string"},
    ).rename(columns={prediction_column: "base_prediction"})
    if frame[ID_COL].isna().any() or frame[ID_COL].duplicated().any():
        raise ValueError("07b OOF row_id must be non-null and unique")
    found_seasons = tuple(sorted(int(x) for x in frame["valid_season"].unique()))
    if found_seasons != VALID_SEASONS:
        raise ValueError(f"unexpected 07b validation seasons: {found_seasons}")
    prediction = frame["base_prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all() or ((prediction <= 0) | (prediction >= 1)).any():
        raise ValueError("07b base probabilities are invalid")
    metadata = {
        "source_run_dir": str(source_run_dir),
        "source_signature": summary.get("signature"),
        "selected_route": selected_route,
        "selection_status": summary.get("selection_status"),
        "route_oof_sha256": sha256_file(oof_path),
        "summary_sha256": sha256_file(summary_path),
        "rows": int(len(frame)),
    }
    return frame, metadata


def align_base_oof(data: pd.DataFrame, base_oof: pd.DataFrame) -> pd.DataFrame:
    expected = data[data["season"].isin(VALID_SEASONS)][
        [ID_COL, TARGET_COL, "season", "game_type"]
    ].copy()
    expected[ID_COL] = expected[ID_COL].astype("string")
    expected["game_type"] = expected["game_type"].astype("string")
    right = base_oof.copy()
    right[ID_COL] = right[ID_COL].astype("string")
    merged = expected.merge(
        right,
        on=ID_COL,
        how="left",
        validate="one_to_one",
        suffixes=("_data", "_oof"),
        sort=False,
    )
    if len(merged) != len(expected) or merged["base_prediction"].isna().any():
        raise RuntimeError("07b OOF rows do not cover all 2022-2024 training rows")
    checks = {
        TARGET_COL: np.array_equal(
            merged[f"{TARGET_COL}_data"].to_numpy(),
            merged[f"{TARGET_COL}_oof"].to_numpy(),
        ),
        "season": np.array_equal(
            merged["season_data"].to_numpy(), merged["season_oof"].to_numpy()
        ),
        "valid_season": np.array_equal(
            merged["season_data"].to_numpy(), merged["valid_season"].to_numpy()
        ),
        "game_type": np.array_equal(
            merged["game_type_data"].astype("string").to_numpy(),
            merged["game_type_oof"].astype("string").to_numpy(),
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"07b OOF key alignment failed: {checks}")
    return pd.DataFrame(
        {
            ID_COL: merged[ID_COL].astype("string"),
            TARGET_COL: merged[f"{TARGET_COL}_data"].astype("int8"),
            "season": merged["season_data"].astype("int16"),
            "game_type": merged["game_type_data"].astype("string"),
            "valid_season": merged["valid_season"].astype("int16"),
            "base_prediction": merged["base_prediction"].astype("float64"),
        }
    )


def target_encoding_keys(frame: pd.DataFrame) -> pd.DataFrame:
    columns = sorted(
        {
            c
            for definition in TARGET_ENCODING_GROUPS.values()
            for c in definition["columns"]
            if c != "count_state"
        }
    )
    keys = frame.loc[:, columns].reset_index(drop=True).copy()
    keys["count_state"] = (
        frame["balls_before"].to_numpy(dtype=np.int16) * 3
        + frame["strikes_before"].to_numpy(dtype=np.int16)
    )
    return keys


def fit_group_statistics(
    source_keys: pd.DataFrame,
    source_target: np.ndarray,
    source_weight: np.ndarray,
    group_columns: list[str],
) -> pd.DataFrame:
    stats = source_keys.loc[:, group_columns].copy()
    stats["__weighted_target"] = (
        np.asarray(source_target, dtype=np.float64)
        * np.asarray(source_weight, dtype=np.float64)
    )
    stats["__effective_count"] = np.asarray(source_weight, dtype=np.float64)
    return (
        stats.groupby(group_columns, dropna=False, observed=True, sort=False)[
            ["__weighted_target", "__effective_count"]
        ]
        .sum()
        .reset_index()
    )


def apply_group_statistics(
    destination_keys: pd.DataFrame,
    stats: pd.DataFrame,
    *,
    group_columns: list[str],
    prior: float,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    destination = destination_keys.loc[:, group_columns].copy()
    destination["__position"] = np.arange(len(destination), dtype=np.int64)
    joined = destination.merge(
        stats,
        on=group_columns,
        how="left",
        sort=False,
        validate="many_to_one",
    ).sort_values("__position", kind="stable")
    if not np.array_equal(
        joined["__position"].to_numpy(), np.arange(len(destination), dtype=np.int64)
    ):
        raise RuntimeError("target-encoding merge changed row order")
    count = joined["__effective_count"].fillna(0.0).to_numpy(dtype=np.float64)
    numerator = joined["__weighted_target"].fillna(0.0).to_numpy(dtype=np.float64)
    encoded = (numerator + strength * prior) / (count + strength)
    return encoded.astype(np.float32), count.astype(np.float32)


def build_temporal_target_encodings(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    valid_season: int,
    half_life: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Encode each row using labels from strictly earlier seasons only."""
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    train_keys = target_encoding_keys(train)
    valid_keys = target_encoding_keys(valid)
    train_target = train[TARGET_COL].to_numpy(dtype=np.float32)
    train_seasons = train["season"].to_numpy(dtype=np.int16)
    valid_weight = recency_weights(
        train_seasons,
        reference_season=valid_season,
        half_life=half_life,
        normalize=False,
    )
    validation_prior = weighted_mean(train_target, valid_weight)

    train_output: dict[str, np.ndarray] = {}
    valid_output: dict[str, np.ndarray] = {}
    coverage: dict[str, dict] = {}
    unique_train_seasons = sorted(int(x) for x in np.unique(train_seasons))
    temporal_train_prior = np.empty(len(train), dtype=np.float32)
    for destination_season in unique_train_seasons:
        destination_mask = train_seasons == destination_season
        source_mask = train_seasons < destination_season
        if not source_mask.any():
            temporal_train_prior[destination_mask] = 0.5
            continue
        source_weight = recency_weights(
            train_seasons[source_mask],
            reference_season=destination_season,
            half_life=half_life,
            normalize=False,
        )
        temporal_train_prior[destination_mask] = weighted_mean(
            train_target[source_mask], source_weight
        )

    for group_name, definition in TARGET_ENCODING_GROUPS.items():
        group_columns = list(definition["columns"])
        strength = float(definition["strength"])
        train_mean = np.empty(len(train), dtype=np.float32)
        train_count = np.empty(len(train), dtype=np.float32)
        source_season_audit = []

        for destination_season in unique_train_seasons:
            destination_mask = train_seasons == destination_season
            source_mask = train_seasons < destination_season
            if not source_mask.any():
                train_mean[destination_mask] = 0.5
                train_count[destination_mask] = 0.0
                source_season_audit.append(
                    {
                        "encoded_season": destination_season,
                        "max_source_season": None,
                        "source_rows": 0,
                        "prior": 0.5,
                    }
                )
                continue
            source_weight = recency_weights(
                train_seasons[source_mask],
                reference_season=destination_season,
                half_life=half_life,
                normalize=False,
            )
            source_prior = weighted_mean(train_target[source_mask], source_weight)
            stats = fit_group_statistics(
                train_keys.loc[source_mask].reset_index(drop=True),
                train_target[source_mask],
                source_weight,
                group_columns,
            )
            encoded, count = apply_group_statistics(
                train_keys.loc[destination_mask].reset_index(drop=True),
                stats,
                group_columns=group_columns,
                prior=source_prior,
                strength=strength,
            )
            train_mean[destination_mask] = encoded
            train_count[destination_mask] = count
            source_season_audit.append(
                {
                    "encoded_season": destination_season,
                    "max_source_season": int(train_seasons[source_mask].max()),
                    "source_rows": int(source_mask.sum()),
                    "prior": source_prior,
                }
            )

        validation_stats = fit_group_statistics(
            train_keys,
            train_target,
            valid_weight,
            group_columns,
        )
        valid_mean, valid_count = apply_group_statistics(
            valid_keys,
            validation_stats,
            group_columns=group_columns,
            prior=validation_prior,
            strength=strength,
        )
        train_output[f"te_{group_name}_mean"] = train_mean
        train_output[f"te_{group_name}_delta"] = train_mean - temporal_train_prior
        train_output[f"te_{group_name}_log_count"] = np.log1p(train_count)
        valid_output[f"te_{group_name}_mean"] = valid_mean
        valid_output[f"te_{group_name}_delta"] = valid_mean - validation_prior
        valid_output[f"te_{group_name}_log_count"] = np.log1p(valid_count)
        coverage[group_name] = {
            "columns": group_columns,
            "strength": strength,
            "validation_known_fraction": float((valid_count > 0).mean()),
            "validation_effective_count_mean": float(valid_count.mean()),
            "validation_effective_count_max": float(valid_count.max()),
            "source_season_audit": source_season_audit,
        }

    leakage_checks = []
    for group in coverage.values():
        for entry in group["source_season_audit"]:
            maximum = entry["max_source_season"]
            leakage_checks.append(
                maximum is None or maximum < entry["encoded_season"]
            )
    audit = {
        "status": "passed" if all(leakage_checks) else "failed",
        "valid_season": valid_season,
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "validation_prior": validation_prior,
        "validation_max_source_season": int(train_seasons.max()),
        "validation_uses_strictly_earlier_seasons": bool(
            train_seasons.max() < valid_season
        ),
        "training_encodings_use_strictly_earlier_seasons": bool(
            all(leakage_checks)
        ),
        "groups": coverage,
    }
    if audit["status"] != "passed":
        raise RuntimeError("temporal target-encoding leakage audit failed")
    return pd.DataFrame(train_output), pd.DataFrame(valid_output), audit


def embedding_dimension(cardinality: int) -> int:
    return int(min(32, max(4, round(4.0 * cardinality**0.25))))


def prepare_neural_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    valid_season: int,
    half_life: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    target_weight_raw = recency_weights(
        train["season"],
        reference_season=valid_season,
        half_life=half_life,
        normalize=False,
    )
    prior = weighted_mean(train[TARGET_COL].to_numpy(), target_weight_raw)
    x_train = add_rowwise_features(train, target_prior=prior, config=FEATURE_CONFIG)
    x_valid = add_rowwise_features(valid, target_prior=prior, config=FEATURE_CONFIG)
    if list(x_train.columns) != list(x_valid.columns):
        raise RuntimeError("train and validation feature columns do not match")

    categorical = categorical_columns(x_train)
    numeric = [column for column in x_train.columns if column not in categorical]
    unexpected_objects = [
        column
        for column in numeric
        if not pd.api.types.is_numeric_dtype(x_train[column])
    ]
    if unexpected_objects:
        raise TypeError(f"unexpected non-numeric columns: {unexpected_objects}")

    train_te, valid_te, te_audit = build_temporal_target_encodings(
        train,
        valid,
        valid_season=valid_season,
        half_life=half_life,
    )
    train_numeric = x_train.loc[:, numeric].replace([np.inf, -np.inf], np.nan)
    valid_numeric = x_valid.loc[:, numeric].replace([np.inf, -np.inf], np.nan)
    train_numeric = pd.concat(
        [train_numeric.reset_index(drop=True), train_te], axis=1
    )
    valid_numeric = pd.concat(
        [valid_numeric.reset_index(drop=True), valid_te], axis=1
    )
    numeric_columns = train_numeric.columns.tolist()
    medians = train_numeric.median(axis=0).fillna(0.0)
    missing_indicator_sources = [
        column for column in numeric_columns if train_numeric[column].isna().any()
    ]
    train_filled = train_numeric.fillna(medians)
    valid_filled = valid_numeric.fillna(medians)
    if missing_indicator_sources:
        indicator_names = {
            column: f"{column}__missing" for column in missing_indicator_sources
        }
        train_indicators = (
            train_numeric.loc[:, missing_indicator_sources]
            .isna()
            .astype("float32")
            .rename(columns=indicator_names)
        )
        valid_indicators = (
            valid_numeric.loc[:, missing_indicator_sources]
            .isna()
            .astype("float32")
            .rename(columns=indicator_names)
        )
        train_filled = pd.concat([train_filled, train_indicators], axis=1)
        valid_filled = pd.concat([valid_filled, valid_indicators], axis=1)

    means = train_filled.mean(axis=0)
    stds = train_filled.std(axis=0, ddof=0).fillna(0.0)
    constant_columns = stds.index[stds < 1e-6].tolist()
    safe_stds = stds.mask(stds < 1e-6, 1.0)
    train_num = ((train_filled - means) / safe_stds).clip(-10.0, 10.0)
    valid_num = ((valid_filled - means) / safe_stds).clip(-10.0, 10.0)
    train_num_array = train_num.to_numpy(dtype=np.float32, copy=True)
    valid_num_array = valid_num.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(train_num_array).all() or not np.isfinite(valid_num_array).all():
        raise RuntimeError("non-finite numerical values remain after preprocessing")

    train_cat = np.empty((len(train), len(categorical)), dtype=np.int32)
    valid_cat = np.empty((len(valid), len(categorical)), dtype=np.int32)
    category_manifest = {}
    cardinalities = []
    embedding_dims = []
    for column_index, column in enumerate(categorical):
        train_values = x_train[column].astype("string").fillna("__MISSING__")
        valid_values = x_valid[column].astype("string").fillna("__MISSING__")
        categories = pd.Index(train_values.unique())
        mapping = pd.Series(
            np.arange(1, len(categories) + 1, dtype=np.int32), index=categories
        )
        train_codes = train_values.map(mapping).fillna(0).to_numpy(dtype=np.int32)
        valid_codes = valid_values.map(mapping).fillna(0).to_numpy(dtype=np.int32)
        cardinality = int(len(categories) + 1)
        dimension = embedding_dimension(cardinality)
        train_cat[:, column_index] = train_codes
        valid_cat[:, column_index] = valid_codes
        cardinalities.append(cardinality)
        embedding_dims.append(dimension)
        category_manifest[column] = {
            "cardinality_with_unknown": cardinality,
            "embedding_dimension": dimension,
            "valid_unknown_fraction": float((valid_codes == 0).mean()),
        }

    preprocessing_manifest = {
        "valid_season": valid_season,
        "feature_config": asdict(FEATURE_CONFIG),
        "target_prior": prior,
        "rowwise_feature_count": int(x_train.shape[1]),
        "numeric_columns": train_filled.columns.tolist(),
        "numeric_feature_count": int(train_filled.shape[1]),
        "categorical_columns": categorical,
        "categorical_feature_count": len(categorical),
        "category_manifest": category_manifest,
        "cardinalities": cardinalities,
        "embedding_dimensions": embedding_dims,
        "missing_indicator_sources": missing_indicator_sources,
        "constant_numeric_columns": constant_columns,
        "numeric_clip": [-10.0, 10.0],
        "category_unknown_code": 0,
        "fit_on_training_rows_only": True,
        "season_feature_excluded": "season" not in x_train.columns,
        "game_type_feature_excluded": "game_type" not in x_train.columns,
    }
    return (
        train_num_array,
        valid_num_array,
        train_cat,
        valid_cat,
        preprocessing_manifest,
        te_audit,
    )


class EmbeddingTabM(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        cardinalities: list[int],
        embedding_dimensions: list[int],
        k: int,
        n_blocks: int,
        d_block: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if len(cardinalities) != len(embedding_dimensions):
            raise ValueError("categorical cardinalities and embedding dimensions differ")
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, dimension)
                for cardinality, dimension in zip(
                    cardinalities, embedding_dimensions, strict=True
                )
            ]
        )
        d_in = n_num_features + sum(embedding_dimensions)
        chunks = [1] * n_num_features + embedding_dimensions
        self.ensemble_view = EnsembleView(k=k)
        self.backbone = make_tabm_backbone(
            d_in=d_in,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
            k=k,
            arch_type="tabm",
            start_scaling_init="normal",
            start_scaling_init_chunks=chunks,
        )
        self.output = LinearEnsemble(d_block, 1, k=k)
        self.k = k
        self.d_in = d_in
        self.start_scaling_chunks = chunks
        self.reset_embedding_parameters()

    def reset_embedding_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [x_num]
        parts.extend(
            embedding(x_cat[:, index])
            for index, embedding in enumerate(self.embeddings)
        )
        x = torch.cat(parts, dim=-1)
        x = self.ensemble_view(x)
        x = self.backbone(x)
        return self.output(x)


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def tabm_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    logits = logits.squeeze(-1)
    expanded_target = target[:, None].expand_as(logits)
    if loss_name == "bce":
        elementwise = nn.functional.binary_cross_entropy_with_logits(
            logits, expanded_target, reduction="none"
        )
    elif loss_name == "brier":
        elementwise = torch.square(torch.sigmoid(logits) - expanded_target)
    else:
        raise ValueError(f"unknown loss: {loss_name}")
    # The loss is computed for each of the k predictions before averaging.
    return (elementwise * weight[:, None]).sum() / (weight.sum() * logits.shape[1])


@torch.inference_mode()
def predict_probabilities(
    model: nn.Module,
    x_num: np.ndarray,
    x_cat: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(x_num), torch.from_numpy(x_cat))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    predictions = []
    for batch_num, batch_cat in loader:
        batch_num = batch_num.to(device, non_blocking=True)
        batch_cat = batch_cat.to(device, dtype=torch.long, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(batch_num, batch_cat).squeeze(-1)
        # Official TabM inference: average probabilities, not logits.
        probability = torch.sigmoid(logits.float()).mean(dim=1)
        predictions.append(probability.cpu().numpy())
    output = np.concatenate(predictions).astype(np.float64)
    return np.clip(output, PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)


def train_one_model(
    train_num: np.ndarray,
    train_cat: np.ndarray,
    train_target: np.ndarray,
    train_weight: np.ndarray,
    valid_num: np.ndarray,
    valid_cat: np.ndarray,
    valid_target: np.ndarray,
    valid_is_r: np.ndarray,
    preprocessing_manifest: dict,
    *,
    loss_name: str,
    valid_season: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, list[dict], dict]:
    seed = args.seed + valid_season
    set_global_seed(seed)
    model = EmbeddingTabM(
        n_num_features=train_num.shape[1],
        cardinalities=preprocessing_manifest["cardinalities"],
        embedding_dimensions=preprocessing_manifest["embedding_dimensions"],
        k=args.k,
        n_blocks=args.n_blocks,
        d_block=args.d_block,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = make_grad_scaler(device.type == "cuda")
    dataset = TensorDataset(
        torch.from_numpy(train_num),
        torch.from_numpy(train_cat),
        torch.from_numpy(train_target.astype(np.float32)),
        torch.from_numpy(train_weight.astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    best_brier = math.inf
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_weight_sum = 0.0
        epoch_started = time.perf_counter()
        for batch_num, batch_cat, batch_target, batch_weight in loader:
            batch_num = batch_num.to(device, non_blocking=True)
            batch_cat = batch_cat.to(device, dtype=torch.long, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            batch_weight = batch_weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(batch_num, batch_cat)
                loss = tabm_loss(logits, batch_target, batch_weight, loss_name)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            weight_sum = float(batch_weight.sum().detach().cpu())
            epoch_loss_sum += float(loss.detach().cpu()) * weight_sum
            epoch_weight_sum += weight_sum

        valid_prediction = predict_probabilities(
            model,
            valid_num,
            valid_cat,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
        )
        r_metrics = probability_metrics(
            valid_target[valid_is_r], valid_prediction[valid_is_r]
        )
        all_metrics = probability_metrics(valid_target, valid_prediction)
        improved = float(r_metrics["brier"]) < best_brier - 1e-7
        if improved:
            best_brier = float(r_metrics["brier"])
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "valid_season": valid_season,
                "loss": loss_name,
                "epoch": epoch,
                "train_loss": epoch_loss_sum / epoch_weight_sum,
                "valid_r_brier": float(r_metrics["brier"]),
                "valid_r_bss_raw": float(r_metrics["bss_raw"]),
                "valid_r_roc_auc": float(r_metrics["roc_auc"]),
                "valid_all_brier": float(all_metrics["brier"]),
                "valid_all_bss_raw": float(all_metrics["bss_raw"]),
                "prediction_mean": float(all_metrics["prediction_mean"]),
                "epoch_seconds": time.perf_counter() - epoch_started,
                "is_best": improved,
            }
        )
        print(
            f"  [{loss_name}] epoch={epoch:02d} "
            f"train_loss={history[-1]['train_loss']:.6f} "
            f"R_brier={r_metrics['brier']:.6f} "
            f"all_brier={all_metrics['brier']:.6f} "
            f"best={best_brier:.6f}"
        )
        if stale_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    best_prediction = predict_probabilities(
        model,
        valid_num,
        valid_cat,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
    )
    metadata = {
        "valid_season": valid_season,
        "loss": loss_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_valid_r_brier": best_brier,
        "parameter_count": int(parameter_count),
        "d_in": int(model.d_in),
        "start_scaling_chunk_count": len(model.start_scaling_chunks),
        "training_seconds": time.perf_counter() - started,
        "tabm_probability_averaging": True,
        "loss_applied_before_k_averaging": True,
    }
    del model, optimizer, scaler, loader, dataset, best_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_prediction, history, metadata


def cache_paths(run_dir: Path, valid_season: int, loss_name: str) -> tuple[Path, Path]:
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = cache_dir / f"fold_{valid_season}_{loss_name}"
    return stem.with_suffix(".csv.gz"), stem.with_name(stem.name + "_metadata.json")


def load_prediction_cache(
    prediction_path: Path,
    metadata_path: Path,
    valid: pd.DataFrame,
    *,
    signature: str,
) -> tuple[np.ndarray, dict] | None:
    if not prediction_path.exists() or not metadata_path.exists():
        return None
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("signature") != signature:
        return None
    cached = pd.read_csv(prediction_path, dtype={ID_COL: "string"})
    expected_ids = valid[ID_COL].astype("string").to_numpy()
    if not np.array_equal(cached[ID_COL].astype("string").to_numpy(), expected_ids):
        return None
    prediction = cached["tabm_prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all():
        return None
    return prediction, metadata


def save_prediction_cache(
    prediction_path: Path,
    metadata_path: Path,
    valid: pd.DataFrame,
    prediction: np.ndarray,
    metadata: dict,
) -> None:
    pd.DataFrame(
        {
            ID_COL: valid[ID_COL].astype("string"),
            "tabm_prediction": prediction,
        }
    ).to_csv(prediction_path, index=False, compression="gzip")
    write_json(metadata_path, metadata)


def scope_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(frame), dtype=bool),
        "R": frame["game_type"].astype(str).eq("R").to_numpy(dtype=np.bool_),
        "F": frame["game_type"].astype(str).eq("F").to_numpy(dtype=np.bool_),
    }


def metric_table(oof: pd.DataFrame, prediction_columns: dict[str, str]) -> pd.DataFrame:
    partitions: list[tuple[str, int | None, np.ndarray]] = [
        ("pooled", None, np.ones(len(oof), dtype=bool)),
        (
            "recent_pooled",
            None,
            oof["valid_season"].isin(RECENT_SEASONS).to_numpy(dtype=np.bool_),
        ),
    ]
    partitions.extend(
        (
            "fold",
            season,
            oof["valid_season"].eq(season).to_numpy(dtype=np.bool_),
        )
        for season in VALID_SEASONS
    )
    rows = []
    for model_name, column in prediction_columns.items():
        for partition, season, partition_mask in partitions:
            frame = oof.loc[partition_mask]
            prediction = oof.loc[partition_mask, column].to_numpy(dtype=np.float64)
            for scope, scope_mask in scope_masks(frame).items():
                if not scope_mask.any():
                    continue
                result = probability_metrics(
                    frame.loc[scope_mask, TARGET_COL], prediction[scope_mask]
                )
                result.update(
                    {
                        "model": model_name,
                        "prediction_column": column,
                        "partition": partition,
                        "valid_season": season,
                        "scope": scope,
                    }
                )
                rows.append(result)
    return pd.DataFrame(rows)


def one_metric(
    metrics: pd.DataFrame,
    *,
    model: str,
    partition: str,
    scope: str,
    season: int | None = None,
) -> pd.Series:
    mask = (
        metrics["model"].eq(model)
        & metrics["partition"].eq(partition)
        & metrics["scope"].eq(scope)
    )
    if season is not None:
        mask &= metrics["valid_season"].eq(season)
    selected = metrics.loc[mask]
    if len(selected) != 1:
        raise RuntimeError(
            f"expected one metric row: {model=}, {partition=}, {scope=}, {season=}"
        )
    return selected.iloc[0]


def build_model_summary(metrics: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    rows = []
    for model in model_names:
        recent_r = one_metric(
            metrics, model=model, partition="recent_pooled", scope="R"
        )
        pooled_r = one_metric(metrics, model=model, partition="pooled", scope="R")
        row = {
            "model": model,
            "pooled_r_brier": pooled_r["brier"],
            "pooled_r_bss_raw": pooled_r["bss_raw"],
            "recent_r_brier": recent_r["brier"],
            "recent_r_bss_raw": recent_r["bss_raw"],
            "recent_r_roc_auc": recent_r["roc_auc"],
            "recent_r_mean_gap": recent_r["mean_gap"],
        }
        for season in VALID_SEASONS:
            fold = one_metric(
                metrics,
                model=model,
                partition="fold",
                scope="R",
                season=season,
            )
            row[f"r_brier_{season}"] = fold["brier"]
            row[f"r_bss_raw_{season}"] = fold["bss_raw"]
            row[f"r_mean_gap_{season}"] = fold["mean_gap"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("recent_r_brier", ignore_index=True)


def build_blend_summary(metrics: pd.DataFrame, candidate_names: list[str]) -> pd.DataFrame:
    base_name = "base_07b"
    base_recent_r = one_metric(
        metrics, model=base_name, partition="recent_pooled", scope="R"
    )
    rows = []
    for model in [base_name, *candidate_names]:
        recent_all = one_metric(
            metrics, model=model, partition="recent_pooled", scope="all"
        )
        recent_r = one_metric(
            metrics, model=model, partition="recent_pooled", scope="R"
        )
        row = {
            "model": model,
            "recent_brier": recent_all["brier"],
            "recent_bss_raw": recent_all["bss_raw"],
            "recent_r_brier": recent_r["brier"],
            "recent_r_bss_raw": recent_r["bss_raw"],
            "delta_recent_r_brier_vs_base": (
                float(recent_r["brier"]) - float(base_recent_r["brier"])
            ),
        }
        for season in VALID_SEASONS:
            candidate_fold = one_metric(
                metrics,
                model=model,
                partition="fold",
                scope="R",
                season=season,
            )
            base_fold = one_metric(
                metrics,
                model=base_name,
                partition="fold",
                scope="R",
                season=season,
            )
            row[f"r_brier_{season}"] = candidate_fold["brier"]
            row[f"delta_r_brier_{season}_vs_base"] = (
                float(candidate_fold["brier"]) - float(base_fold["brier"])
            )
        row["worst_fold_delta"] = max(
            row[f"delta_r_brier_{season}_vs_base"] for season in VALID_SEASONS
        )
        row["improves_recent_enough"] = (
            row["delta_recent_r_brier_vs_base"] <= -MIN_RECENT_BRIER_IMPROVEMENT
        )
        row["stable_across_folds"] = (
            row["worst_fold_delta"] <= MAX_FOLD_BRIER_REGRESSION
        )
        row["passes_fixed_gate"] = bool(
            model != base_name
            and row["improves_recent_enough"]
            and row["stable_across_folds"]
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("recent_r_brier", ignore_index=True)


def prediction_correlations(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = oof["base_prediction"].to_numpy(dtype=np.float64)
    partitions = [
        (
            "recent_pooled",
            None,
            oof["valid_season"].isin(RECENT_SEASONS).to_numpy(dtype=np.bool_),
        )
    ]
    partitions.extend(
        (
            "fold",
            season,
            oof["valid_season"].eq(season).to_numpy(dtype=np.bool_),
        )
        for season in VALID_SEASONS
    )
    for loss_name in LOSSES:
        column = f"tabm_{loss_name}_prediction"
        if column not in oof:
            continue
        candidate = oof[column].to_numpy(dtype=np.float64)
        for partition, season, partition_mask in partitions:
            frame = oof.loc[partition_mask]
            for scope, scope_mask in scope_masks(frame).items():
                combined = np.zeros(len(oof), dtype=np.bool_)
                combined[np.flatnonzero(partition_mask)] = scope_mask
                if combined.sum() < 2:
                    continue
                rows.append(
                    {
                        "model": f"tabm_{loss_name}",
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


def choose_candidate(blend_summary: pd.DataFrame) -> tuple[str, str]:
    passed = blend_summary[blend_summary["passes_fixed_gate"]]
    if passed.empty:
        return "base_07b", "no_tabm_candidate_passed_fixed_gate"
    selected = passed.sort_values(
        ["recent_r_brier", "worst_fold_delta", "model"], ignore_index=True
    ).iloc[0]
    return str(selected["model"]), "best_predeclared_candidate_passed"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.batch_size <= 0 or args.max_epochs <= 0 or args.patience <= 0:
        raise ValueError("batch size, max epochs, and patience must be positive")
    if args.half_life <= 0:
        raise ValueError("half-life must be positive")
    if len(set(args.losses)) != len(args.losses):
        raise ValueError("loss names must be unique")

    source_run_dir = resolve_source_run(args)
    source_oof_path = source_run_dir / "route_oof.csv.gz"
    source_summary_path = source_run_dir / "summary.json"
    signature_payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "valid_seasons": list(args.valid_seasons),
        "recent_seasons": list(RECENT_SEASONS),
        "losses": list(args.losses),
        "blend_weights": list(BLEND_WEIGHTS),
        "feature_config": asdict(FEATURE_CONFIG),
        "target_encoding_groups": TARGET_ENCODING_GROUPS,
        "training_scope": "R_only",
        "validation_scope": "all_rows_early_stop_on_R",
        "half_life": args.half_life,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "k": args.k,
        "n_blocks": args.n_blocks,
        "d_block": args.d_block,
        "dropout": args.dropout,
        "seed": args.seed,
        "blend_gate": {
            "min_recent_brier_improvement": MIN_RECENT_BRIER_IMPROVEMENT,
            "max_fold_brier_regression": MAX_FOLD_BRIER_REGRESSION,
            "F_predictions_exact": True,
        },
        "script_sha256": sha256_file(Path(__file__)),
        "features_sha256": sha256_file(SRC_DIR / "features.py"),
        "evaluation_sha256": sha256_file(SRC_DIR / "evaluation.py"),
        "train_sha256": sha256_file(args.train_path),
        "source_oof_sha256": sha256_file(source_oof_path),
        "source_summary_sha256": sha256_file(source_summary_path),
    }
    signature = experiment_signature(signature_payload)
    run_dir = args.output_dir / f"run_{signature}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "configuration.json",
        {
            "signature": signature,
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_name": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else None
            ),
            "source_run_dir": str(source_run_dir),
            **signature_payload,
        },
    )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading training data: {args.train_path}")
    data = pd.read_csv(args.train_path, dtype={ID_COL: "string"})
    validate_training_data(data, args.valid_seasons)
    base_oof_raw, source_metadata = load_base_oof(source_run_dir)
    oof = align_base_oof(data, base_oof_raw)
    del base_oof_raw

    training_history_rows: list[dict] = []
    fold_training_rows: list[dict] = []
    feature_manifests: dict[str, dict] = {}
    te_audits: dict[str, dict] = {}
    prediction_by_loss: dict[str, dict[str, float]] = {
        loss_name: {} for loss_name in args.losses
    }

    for valid_season in args.valid_seasons:
        train_mask = data["season"].lt(valid_season) & data["game_type"].eq("R")
        valid_mask = data["season"].eq(valid_season)
        train = data.loc[train_mask].reset_index(drop=True)
        valid = data.loc[valid_mask].reset_index(drop=True)
        if train.empty or valid.empty:
            raise RuntimeError(f"empty fold: {valid_season}")
        if int(train["season"].max()) >= valid_season:
            raise RuntimeError("forward-CV season boundary was violated")
        print(
            f"\nFold {valid_season}: train={len(train):,} R-only rows, "
            f"valid={len(valid):,} all rows"
        )

        missing_losses = []
        for loss_name in args.losses:
            prediction_path, metadata_path = cache_paths(
                run_dir, valid_season, loss_name
            )
            cached = None if args.force_rerun else load_prediction_cache(
                prediction_path,
                metadata_path,
                valid,
                signature=signature,
            )
            if cached is None:
                missing_losses.append(loss_name)
            else:
                prediction, metadata = cached
                prediction_by_loss[loss_name].update(
                    zip(valid[ID_COL].astype(str), prediction, strict=True)
                )
                fold_training_rows.append(metadata["training_metadata"])
                training_history_rows.extend(metadata.get("training_history", []))
                feature_manifests[str(valid_season)] = metadata[
                    "preprocessing_manifest"
                ]
                te_audits[str(valid_season)] = metadata["target_encoding_audit"]
                print(f"  [{loss_name}] loaded cache")

        if missing_losses:
            (
                train_num,
                valid_num,
                train_cat,
                valid_cat,
                preprocessing_manifest,
                te_audit,
            ) = prepare_neural_features(
                train,
                valid,
                valid_season=valid_season,
                half_life=args.half_life,
            )
            feature_manifests[str(valid_season)] = preprocessing_manifest
            te_audits[str(valid_season)] = te_audit
            train_target = train[TARGET_COL].to_numpy(dtype=np.float32)
            valid_target = valid[TARGET_COL].to_numpy(dtype=np.float32)
            valid_is_r = valid["game_type"].eq("R").to_numpy(dtype=np.bool_)
            train_weight = recency_weights(
                train["season"],
                reference_season=valid_season,
                half_life=args.half_life,
                normalize=True,
            )
            print(
                f"  inputs: numeric={train_num.shape[1]}, "
                f"categorical={train_cat.shape[1]}, "
                f"target prior={preprocessing_manifest['target_prior']:.6f}"
            )
            for loss_name in missing_losses:
                prediction, history, training_metadata = train_one_model(
                    train_num,
                    train_cat,
                    train_target,
                    train_weight,
                    valid_num,
                    valid_cat,
                    valid_target,
                    valid_is_r,
                    preprocessing_manifest,
                    loss_name=loss_name,
                    valid_season=valid_season,
                    args=args,
                    device=device,
                )
                prediction_by_loss[loss_name].update(
                    zip(valid[ID_COL].astype(str), prediction, strict=True)
                )
                training_history_rows.extend(history)
                fold_training_rows.append(training_metadata)
                prediction_path, metadata_path = cache_paths(
                    run_dir, valid_season, loss_name
                )
                cache_metadata = {
                    "signature": signature,
                    "training_metadata": training_metadata,
                    "training_history": history,
                    "preprocessing_manifest": preprocessing_manifest,
                    "target_encoding_audit": te_audit,
                }
                save_prediction_cache(
                    prediction_path,
                    metadata_path,
                    valid,
                    prediction,
                    cache_metadata,
                )
            del train_num, valid_num, train_cat, valid_cat
        del train, valid
        gc.collect()

    for loss_name in args.losses:
        mapping = prediction_by_loss[loss_name]
        oof[f"tabm_{loss_name}_prediction"] = oof[ID_COL].astype(str).map(mapping)
        if oof[f"tabm_{loss_name}_prediction"].isna().any():
            raise RuntimeError(f"incomplete OOF predictions for loss={loss_name}")

    is_r = oof["game_type"].astype(str).eq("R").to_numpy(dtype=np.bool_)
    base_prediction = oof["base_prediction"].to_numpy(dtype=np.float64)
    prediction_columns = {"base_07b": "base_prediction"}
    standalone_names = []
    candidate_names = []
    for loss_name in args.losses:
        tabm_column = f"tabm_{loss_name}_prediction"
        tabm_prediction = oof[tabm_column].to_numpy(dtype=np.float64)
        standalone_name = f"tabm_{loss_name}"
        standalone_names.append(standalone_name)
        prediction_columns[standalone_name] = tabm_column
        for weight in BLEND_WEIGHTS:
            weight_label = int(round(weight * 100))
            candidate_name = f"base_tabm_{loss_name}_w{weight_label:02d}"
            candidate_column = f"blend_{loss_name}_w{weight_label:02d}"
            candidate = base_prediction.copy()
            candidate[is_r] = (
                (1.0 - weight) * base_prediction[is_r]
                + weight * tabm_prediction[is_r]
            )
            oof[candidate_column] = np.clip(
                candidate, PROBABILITY_EPS, 1.0 - PROBABILITY_EPS
            )
            if not np.array_equal(
                oof.loc[~is_r, candidate_column],
                oof.loc[~is_r, "base_prediction"],
            ):
                raise RuntimeError(f"candidate changed F predictions: {candidate_name}")
            candidate_names.append(candidate_name)
            prediction_columns[candidate_name] = candidate_column

    metrics = metric_table(oof, prediction_columns)
    model_summary = build_model_summary(
        metrics, ["base_07b", *standalone_names]
    )
    blend_summary = build_blend_summary(metrics, candidate_names)
    correlations = prediction_correlations(oof)
    selected_candidate, selection_status = choose_candidate(blend_summary)

    selected_column = prediction_columns[selected_candidate]
    f_max_difference = float(
        np.max(
            np.abs(
                oof.loc[~is_r, selected_column].to_numpy(dtype=np.float64)
                - oof.loc[~is_r, "base_prediction"].to_numpy(dtype=np.float64)
            )
        )
    )
    if f_max_difference != 0.0:
        raise RuntimeError("selected candidate did not preserve F predictions exactly")
    integrity_audit = {
        "status": "passed",
        "rows": int(len(oof)),
        "row_id_unique": bool(oof[ID_COL].is_unique),
        "valid_seasons": sorted(int(x) for x in oof["valid_season"].unique()),
        "training_scope": "R_only",
        "season_feature_excluded_all_folds": all(
            manifest["season_feature_excluded"]
            for manifest in feature_manifests.values()
        ),
        "game_type_feature_excluded_all_folds": all(
            manifest["game_type_feature_excluded"]
            for manifest in feature_manifests.values()
        ),
        "temporal_target_encoding_passed_all_folds": all(
            audit["status"] == "passed" for audit in te_audits.values()
        ),
        "selected_f_prediction_max_abs_difference": f_max_difference,
        "F_predictions_exact": f_max_difference == 0.0,
    }

    pd.DataFrame(training_history_rows).to_csv(
        run_dir / "training_history.csv", index=False
    )
    pd.DataFrame(fold_training_rows).to_csv(
        run_dir / "fold_training.csv", index=False
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    model_summary.to_csv(run_dir / "model_summary.csv", index=False)
    blend_summary.to_csv(run_dir / "blend_summary.csv", index=False)
    correlations.to_csv(run_dir / "prediction_correlations.csv", index=False)
    oof.to_csv(run_dir / "oof_predictions.csv.gz", index=False, compression="gzip")
    write_json(run_dir / "feature_manifest.json", feature_manifests)
    write_json(run_dir / "target_encoding_audit.json", te_audits)
    write_json(run_dir / "integrity_audit.json", integrity_audit)

    selected_row = blend_summary[
        blend_summary["model"].eq(selected_candidate)
    ].iloc[0]
    base_row = blend_summary[blend_summary["model"].eq("base_07b")].iloc[0]
    summary = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "signature": signature,
        "source_metadata": source_metadata,
        "device": str(device),
        "selected_candidate": selected_candidate,
        "selection_status": selection_status,
        "selected_prediction_column": selected_column,
        "selected_recent_r_brier": float(selected_row["recent_r_brier"]),
        "selected_recent_r_bss_raw": float(selected_row["recent_r_bss_raw"]),
        "selected_delta_recent_r_brier_vs_base": float(
            selected_row["delta_recent_r_brier_vs_base"]
        ),
        "selected_worst_fold_delta": float(selected_row["worst_fold_delta"]),
        "base_recent_r_brier": float(base_row["recent_r_brier"]),
        "base_recent_r_bss_raw": float(base_row["recent_r_bss_raw"]),
        "integrity_audit": integrity_audit,
        "selection_gate": {
            "minimum_recent_r_brier_improvement": MIN_RECENT_BRIER_IMPROVEMENT,
            "maximum_fold_r_brier_regression": MAX_FOLD_BRIER_REGRESSION,
            "F_predictions_must_be_exact": True,
        },
        "interpretation": (
            "A selected TabM blend means the independent neural prediction improved "
            "recent R Brier under the fixed stability gate. A base selection means "
            "the tested TabM recipe should not yet be promoted to final training."
        ),
        "next_step": (
            "If a TabM candidate passed, reproduce its loss and blend weight in a "
            "final-training submission notebook. Otherwise inspect standalone metrics, "
            "correlation, and calibration before changing one neural design axis."
        ),
    }
    write_json(run_dir / "summary.json", summary)
    write_json(
        args.output_dir / "latest_run.json",
        {"signature": signature, "run_dir": str(run_dir), "summary": summary},
    )

    print("\nStandalone comparison (R rows)")
    print(model_summary.to_string(index=False))
    print("\nBest blend candidates")
    print(
        blend_summary[
            [
                "model",
                "recent_r_brier",
                "recent_r_bss_raw",
                "delta_recent_r_brier_vs_base",
                "worst_fold_delta",
                "passes_fixed_gate",
            ]
        ].head(10).to_string(index=False)
    )
    print(f"\nSelected: {selected_candidate} ({selection_status})")
    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
