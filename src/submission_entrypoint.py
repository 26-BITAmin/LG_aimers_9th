#!/usr/bin/env python3
"""Self-contained inference entrypoint for DACON submission schemas 2, 3, and 4.

The evaluator places test.csv and sample_submission.csv in ./data and expects
./output/submission.csv. Every engineered feature depends only on the current
row and constants stored in the fitted artifact.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

try:
    import torch
    import torch.nn as nn
    from tabm import EnsembleView, LinearEnsemble, make_tabm_backbone
except ImportError:  # Schemas 2 and 3 do not require the neural runtime.
    torch = None
    nn = None
    EnsembleView = None
    LinearEnsemble = None
    make_tabm_backbone = None


ID_COL = "row_id"
TARGET_COL = "control_success"
EPSILON = 1e-7

RAW_CATEGORICAL_COLUMNS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]
DERIVED_CATEGORICAL_COLUMNS = [
    "count_state",
    "hand_matchup",
    "count_hand_matchup",
    "count_base_state",
    "base_out_state",
    "inning_bucket",
    "score_diff_bucket",
    "inning_score_state",
    "pitcher_batter_team_matchup",
    "pitcher_team_hand",
    "batter_team_hand",
]


if nn is not None:

    class EmbeddingTabM(nn.Module):
        """Submission-time TabM with shared categorical embeddings."""

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
                raise ValueError(
                    "categorical cardinalities and embedding dimensions differ"
                )
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

        def forward(self, x_num, x_cat):
            parts = [x_num]
            parts.extend(
                embedding(x_cat[:, index])
                for index, embedding in enumerate(self.embeddings)
            )
            x = torch.cat(parts, dim=-1)
            x = self.ensemble_view(x)
            x = self.backbone(x)
            return self.output(x)

else:

    class EmbeddingTabM:  # pragma: no cover - used only for a clearer error.
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise ImportError("schema 4 requires torch and tabm")


@dataclass(frozen=True)
class FeatureConfig:
    pitcher_prior_strength: float = 200.0
    batter_prior_strength: float = 300.0
    include_season: bool = True
    include_player_ids: bool = True
    include_game_type: bool = True
    include_rate_gap_features: bool = False
    include_recent_dynamics_features: bool = False
    include_pitchmix_shape_features: bool = False
    include_context_interaction_features: bool = False
    include_team_context_features: bool = False


def _weighted_row_mean(frame: pd.DataFrame, weights: list[float]) -> pd.Series:
    weight = pd.Series(weights, index=frame.columns, dtype="float64")
    numerator = frame.mul(weight, axis=1).sum(axis=1, min_count=1)
    denominator = frame.notna().mul(weight, axis=1).sum(axis=1)
    return numerator / denominator.replace(0.0, np.nan)


def add_rate_gap_features(out: pd.DataFrame) -> None:
    success = out["asof_pitcher_success_rate"]
    strike = out["asof_pitcher_strike_rate"]
    ball = out["asof_pitcher_ball_rate"]
    middle = out["asof_pitcher_middle_rate"]
    reverse = out["asof_pitcher_reverse_rate"]
    out["pitcher_success_minus_ball"] = success - ball
    out["pitcher_strike_minus_ball"] = strike - ball
    out["pitcher_success_minus_middle"] = success - middle
    out["pitcher_success_minus_reverse"] = success - reverse
    out["smoothed_pitcher_batter_success_gap"] = (
        out["smoothed_pitcher_success_rate"]
        - out["smoothed_batter_success_rate"]
    )
    out["pitcher_batter_middle_gap"] = (
        out["asof_pitcher_middle_rate"] - out["asof_batter_middle_rate"]
    )
    out["pitcher_control_balance"] = success + strike - ball - middle - reverse


def add_recent_dynamics_features(out: pd.DataFrame) -> None:
    success_columns = [
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]
    middle_columns = [
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ]
    recent_success = out[success_columns]
    recent_middle = out[middle_columns]
    out["recent_success_1_minus_5"] = recent_success.iloc[:, 0] - recent_success.iloc[:, 2]
    out["recent_success_3_minus_5"] = recent_success.iloc[:, 1] - recent_success.iloc[:, 2]
    out["recent_success_1_minus_career"] = recent_success.iloc[:, 0] - out["asof_pitcher_success_rate"]
    out["recent_success_5_minus_career"] = recent_success.iloc[:, 2] - out["asof_pitcher_success_rate"]
    out["recent_success_std"] = recent_success.std(axis=1, ddof=0)
    out["recent_success_range"] = recent_success.max(axis=1) - recent_success.min(axis=1)
    out["recent_success_weighted"] = _weighted_row_mean(recent_success, [0.50, 0.30, 0.20])
    out["recent_middle_1_minus_5"] = recent_middle.iloc[:, 0] - recent_middle.iloc[:, 2]
    out["recent_middle_3_minus_5"] = recent_middle.iloc[:, 1] - recent_middle.iloc[:, 2]
    out["recent_middle_std"] = recent_middle.std(axis=1, ddof=0)
    out["recent_middle_range"] = recent_middle.max(axis=1) - recent_middle.min(axis=1)
    out["recent_middle_weighted"] = _weighted_row_mean(recent_middle, [0.50, 0.30, 0.20])
    out["reliability_adjusted_form_gap"] = out["pitcher_reliability"] * out["pitcher_form_gap"]


def add_pitchmix_shape_features(out: pd.DataFrame) -> None:
    columns = [
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]
    rates = out[columns].clip(lower=0.0)
    total = rates.sum(axis=1, min_count=1)
    normalized = rates.div(total.replace(0.0, np.nan), axis=0)
    safe = normalized.clip(lower=1e-12)
    out["pitchmix_rate_sum"] = total
    out["pitchmix_max_share"] = normalized.max(axis=1)
    out["pitchmix_min_share"] = normalized.min(axis=1)
    out["pitchmix_entropy"] = -(normalized * np.log(safe)).sum(axis=1, min_count=1)
    out["pitchmix_concentration"] = np.square(normalized).sum(axis=1, min_count=1)
    out["fastball_minus_breaking"] = normalized.iloc[:, 0] - normalized.iloc[:, 1]
    out["fastball_minus_offspeed"] = normalized.iloc[:, 0] - normalized.iloc[:, 2]
    out["breaking_minus_offspeed"] = normalized.iloc[:, 1] - normalized.iloc[:, 2]
    pitchmix_n = out["asof_pitcher_pitchmix_n"].astype("float64")
    out["pitchmix_reliability"] = pitchmix_n / (pitchmix_n + 200.0)
    out["has_pitchmix_history"] = (pitchmix_n > 0).astype("int8")


def add_context_interaction_features(out: pd.DataFrame) -> None:
    out["count_hand_matchup"] = out["count_state"] * 100 + out["hand_matchup"]
    out["count_base_state"] = (
        out["count_state"].astype("string")
        + "_"
        + out["base_state"].astype("string").fillna("__MISSING__")
    )
    out["base_out_state"] = (
        out["base_state"].astype("string").fillna("__MISSING__")
        + "_"
        + out["outs_before"].astype("string")
    )
    out["inning_bucket"] = pd.cut(
        out["inning"], bins=[-np.inf, 3, 6, np.inf], labels=[0, 1, 2]
    ).astype("int8")
    out["score_diff_bucket"] = (out["score_diff_pitcher_team"].clip(-4, 4) + 4).astype("int8")
    out["inning_score_state"] = (
        out["inning_bucket"].astype("int16") * 10
        + out["score_diff_bucket"].astype("int16")
    )
    out["pressure_index"] = out["li"] * (1.0 + out["is_full_count"] + out["is_tied"])
    out["count_ball_propensity"] = out["balls_before"] * out["asof_pitcher_ball_rate"]
    out["count_strike_propensity"] = out["strikes_before"] * out["asof_pitcher_strike_rate"]
    out["leverage_score_closeness"] = out["li"] / (1.0 + out["abs_score_diff"])
    out["runner_pressure"] = out["num_runners_on"] * out["li"]


def add_team_context_features(out: pd.DataFrame) -> None:
    pitcher_is_home = out["top_bottom"].astype("string").eq("T")
    out["pitcher_win_expectancy"] = np.where(
        pitcher_is_home, out["home_win_expectancy"], out["away_win_expectancy"]
    )
    out["batter_win_expectancy"] = np.where(
        pitcher_is_home, out["away_win_expectancy"], out["home_win_expectancy"]
    )
    out["pitcher_batter_expectancy_gap"] = (
        out["pitcher_win_expectancy"] - out["batter_win_expectancy"]
    )
    out["pitcher_batter_team_matchup"] = (
        out["pitcher_team_id"].astype("string")
        + "_"
        + out["batter_team_id"].astype("string")
    )
    out["pitcher_team_hand"] = (
        out["pitcher_team_id"].astype("string")
        + "_"
        + out["pitcher_hand"].astype("string")
    )
    out["batter_team_hand"] = (
        out["batter_team_id"].astype("string")
        + "_"
        + out["batter_hand"].astype("string")
    )


def add_rowwise_features(
    frame: pd.DataFrame,
    *,
    target_prior: float,
    config: FeatureConfig,
) -> pd.DataFrame:
    drop_columns = [column for column in (ID_COL, TARGET_COL) if column in frame]
    out = frame.drop(columns=drop_columns).copy()
    out["count_state"] = out["balls_before"] * 3 + out["strikes_before"]
    out["hand_matchup"] = out["pitcher_hand"] * 10 + out["batter_hand"]
    out["is_two_strike"] = (out["strikes_before"] == 2).astype("int8")
    out["is_three_ball"] = (out["balls_before"] == 3).astype("int8")
    out["is_full_count"] = (
        (out["balls_before"] == 3) & (out["strikes_before"] == 2)
    ).astype("int8")
    out["is_late_inning"] = (out["inning"] >= 7).astype("int8")
    out["is_tied"] = (out["score_diff_pitcher_team"] == 0).astype("int8")
    out["abs_score_diff"] = out["score_diff_pitcher_team"].abs()
    out["is_high_leverage"] = (out["li"] >= 2.0).astype("int8")

    pitcher_n = out["asof_pitcher_n"].astype("float64")
    batter_n = out["asof_batter_n"].astype("float64")
    pitcher_rate = out["asof_pitcher_success_rate"].fillna(target_prior)
    batter_rate = out["asof_batter_success_rate"].fillna(target_prior)
    out["is_pitcher_cold_start"] = (pitcher_n == 0).astype("int8")
    out["is_batter_cold_start"] = (batter_n == 0).astype("int8")
    out["has_pitcher_recent_games"] = out[
        "asof_pitcher_prev1_game_success_rate"
    ].notna().astype("int8")
    out["log1p_pitcher_n"] = np.log1p(pitcher_n)
    out["log1p_batter_n"] = np.log1p(batter_n)
    pitcher_strength = float(config.pitcher_prior_strength)
    batter_strength = float(config.batter_prior_strength)
    out["pitcher_reliability"] = pitcher_n / (pitcher_n + pitcher_strength)
    out["batter_reliability"] = batter_n / (batter_n + batter_strength)
    out["smoothed_pitcher_success_rate"] = (
        pitcher_n * pitcher_rate + pitcher_strength * target_prior
    ) / (pitcher_n + pitcher_strength)
    out["smoothed_batter_success_rate"] = (
        batter_n * batter_rate + batter_strength * target_prior
    ) / (batter_n + batter_strength)

    recent_success = [
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]
    recent_middle = [
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ]
    out["pitcher_recent_success_mean"] = out[recent_success].mean(axis=1)
    out["pitcher_recent_middle_mean"] = out[recent_middle].mean(axis=1)
    out["pitcher_form_gap"] = (
        out["pitcher_recent_success_mean"] - out["asof_pitcher_success_rate"]
    )

    if config.include_rate_gap_features:
        add_rate_gap_features(out)
    if config.include_recent_dynamics_features:
        add_recent_dynamics_features(out)
    if config.include_pitchmix_shape_features:
        add_pitchmix_shape_features(out)
    if config.include_context_interaction_features:
        add_context_interaction_features(out)
    if config.include_team_context_features:
        add_team_context_features(out)
    if not config.include_player_ids:
        out = out.drop(columns=["pitcher_id", "batter_id"])
    if not config.include_season:
        out = out.drop(columns=["season"])
    if not config.include_game_type:
        out = out.drop(columns=["game_type"])
    return out


def select_feature_columns(
    frame: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
) -> pd.DataFrame:
    expected = set(feature_columns)
    actual = set(frame.columns)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{model_name} feature mismatch: missing={missing}, extra={extra}"
        )
    return frame.loc[:, feature_columns]


def prepare_lgbm_features(frame: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    config = FeatureConfig(**metadata["feature_config"])
    out = add_rowwise_features(
        frame, target_prior=float(metadata["target_prior"]), config=config
    )
    category_maps = metadata["category_maps"]
    missing_categorical = sorted(set(category_maps) - set(out.columns))
    if missing_categorical:
        raise ValueError(f"missing LightGBM categorical columns: {missing_categorical}")
    for column, categories in category_maps.items():
        values = out[column].astype("string").fillna("__MISSING__")
        out[column] = pd.Categorical(values, categories=categories).codes.astype("int32")
    for column in out.columns:
        if column in category_maps:
            continue
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].astype("float32")
        elif pd.api.types.is_integer_dtype(out[column]):
            out[column] = pd.to_numeric(out[column], downcast="integer")
    return select_feature_columns(out, metadata["feature_columns"], "LightGBM")


def prepare_catboost_features(frame: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    config = FeatureConfig(**metadata["feature_config"])
    out = add_rowwise_features(
        frame, target_prior=float(metadata["target_prior"]), config=config
    )
    for column in metadata["categorical_columns"]:
        if column not in out:
            raise ValueError(f"missing CatBoost categorical column: {column}")
        out[column] = out[column].astype("string").fillna("__MISSING__").astype(str)
    return select_feature_columns(out, metadata["feature_columns"], "CatBoost")


def _tabm_target_encoding_keys(frame: pd.DataFrame, groups: dict) -> pd.DataFrame:
    columns = sorted(
        {
            column
            for definition in groups.values()
            for column in definition["columns"]
            if column != "count_state"
        }
    )
    keys = frame.loc[:, columns].reset_index(drop=True).copy()
    keys["count_state"] = (
        frame["balls_before"].to_numpy(dtype=np.int16) * 3
        + frame["strikes_before"].to_numpy(dtype=np.int16)
    )
    return keys


def _apply_tabm_group_statistics(
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
        joined["__position"].to_numpy(),
        np.arange(len(destination), dtype=np.int64),
    ):
        raise RuntimeError("TabM target-encoding merge changed row order")
    count = joined["__effective_count"].fillna(0.0).to_numpy(dtype=np.float64)
    numerator = joined["__weighted_target"].fillna(0.0).to_numpy(dtype=np.float64)
    encoded = (numerator + strength * prior) / (count + strength)
    return encoded.astype(np.float32), count.astype(np.float32)


def prepare_tabm_features(
    frame: pd.DataFrame,
    metadata: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply only fitted training constants to the supplied rows."""

    config = FeatureConfig(**metadata["feature_config"])
    rowwise = add_rowwise_features(
        frame,
        target_prior=float(metadata["target_prior"]),
        config=config,
    )
    expected_rowwise = set(metadata["rowwise_feature_columns"])
    if set(rowwise.columns) != expected_rowwise:
        missing = sorted(expected_rowwise - set(rowwise.columns))
        extra = sorted(set(rowwise.columns) - expected_rowwise)
        raise ValueError(
            f"TabM rowwise feature mismatch: missing={missing}, extra={extra}"
        )

    target_encoding = metadata["target_encoding"]
    groups = target_encoding["groups"]
    keys = _tabm_target_encoding_keys(frame, groups)
    encoded_columns: dict[str, np.ndarray] = {}
    prior = float(target_encoding["prior"])
    for group_name, definition in groups.items():
        mean, count = _apply_tabm_group_statistics(
            keys,
            definition["stats"],
            group_columns=list(definition["columns"]),
            prior=prior,
            strength=float(definition["strength"]),
        )
        encoded_columns[f"te_{group_name}_mean"] = mean
        encoded_columns[f"te_{group_name}_delta"] = mean - prior
        encoded_columns[f"te_{group_name}_log_count"] = np.log1p(count)

    numeric_columns = metadata["rowwise_numeric_columns"]
    numeric = rowwise.loc[:, numeric_columns].replace([np.inf, -np.inf], np.nan)
    numeric = pd.concat(
        [numeric.reset_index(drop=True), pd.DataFrame(encoded_columns)], axis=1
    )
    expected_pre_missing = metadata["pre_missing_numeric_columns"]
    if list(numeric.columns) != expected_pre_missing:
        raise RuntimeError("TabM numerical feature order changed before imputation")
    medians = pd.Series(metadata["numeric_medians"], dtype="float64").reindex(
        expected_pre_missing
    )
    if medians.isna().any():
        raise RuntimeError("TabM numerical medians are incomplete")
    filled = numeric.fillna(medians)
    missing_sources = metadata["missing_indicator_sources"]
    if missing_sources:
        indicator_names = {
            column: f"{column}__missing" for column in missing_sources
        }
        indicators = (
            numeric.loc[:, missing_sources]
            .isna()
            .astype("float32")
            .rename(columns=indicator_names)
        )
        filled = pd.concat([filled, indicators], axis=1)
    final_columns = metadata["numeric_feature_columns"]
    if list(filled.columns) != final_columns:
        raise RuntimeError("TabM numerical feature order changed after imputation")
    means = pd.Series(metadata["numeric_means"], dtype="float64").reindex(
        final_columns
    )
    stds = pd.Series(metadata["numeric_stds"], dtype="float64").reindex(
        final_columns
    )
    if means.isna().any() or stds.isna().any() or (stds <= 0.0).any():
        raise RuntimeError("TabM numerical scaling constants are invalid")
    x_num = ((filled - means) / stds).clip(-10.0, 10.0).to_numpy(
        dtype=np.float32, copy=True
    )
    if not np.isfinite(x_num).all():
        raise RuntimeError("TabM numerical features contain non-finite values")

    categorical_columns = metadata["categorical_columns"]
    x_cat = np.empty((len(frame), len(categorical_columns)), dtype=np.int32)
    for index, column in enumerate(categorical_columns):
        values = rowwise[column].astype("string").fillna("__MISSING__")
        categories = metadata["category_maps"][column]
        # pandas code -1 is unknown; shifting by one reserves code 0 for it.
        codes = pd.Categorical(values, categories=categories).codes.astype("int32")
        x_cat[:, index] = codes + 1
    cardinalities = np.asarray(metadata["cardinalities"], dtype=np.int64)
    if len(cardinalities) and (
        (x_cat < 0).any() or (x_cat >= cardinalities[None, :]).any()
    ):
        raise RuntimeError("TabM categorical code is outside its cardinality")
    return x_num, x_cat


def safe_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(clipped) - np.log1p(-clipped)


def blend_logit(
    lgbm_probability: np.ndarray,
    catboost_probability: np.ndarray,
    *,
    lgbm_weight: float,
) -> np.ndarray:
    blended = (
        lgbm_weight * safe_logit(lgbm_probability)
        + (1.0 - lgbm_weight) * safe_logit(catboost_probability)
    )
    return 1.0 / (1.0 + np.exp(-np.clip(blended, -30.0, 30.0)))


def load_predictor(model_dir: Path) -> dict:
    model_dir = Path(model_dir)
    metadata = joblib.load(model_dir / "artifacts.joblib")
    schema_version = metadata.get("schema_version")
    if schema_version not in (2, 3, 4):
        raise RuntimeError(f"unsupported artifact schema: {metadata.get('schema_version')}")
    threads = max(1, min(6, os.cpu_count() or 1))
    lgbm_model = lgb.Booster(
        params={"num_threads": threads},
        model_file=str(model_dir / metadata["lgbm"]["model_filename"]),
    )
    predictor = {
        "metadata": metadata,
        "lgbm_model": lgbm_model,
        "threads": threads,
    }
    if schema_version == 2:
        catboost_model = CatBoostClassifier()
        catboost_model.load_model(
            str(model_dir / metadata["catboost"]["model_filename"])
        )
        predictor["catboost_model"] = catboost_model
    else:
        for model_key in ("catboost_base", "catboost_f_route"):
            model = CatBoostClassifier()
            model.load_model(
                str(model_dir / metadata[model_key]["model_filename"])
            )
            predictor[f"{model_key}_model"] = model
    if schema_version == 4:
        if torch is None or nn is None:
            raise ImportError("schema 4 requires torch and tabm")
        torch.set_num_threads(threads)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tabm_metadata = metadata["tabm"]
        architecture = tabm_metadata["architecture"]
        tabm_model = EmbeddingTabM(
            n_num_features=int(tabm_metadata["n_num_features"]),
            cardinalities=[int(value) for value in tabm_metadata["cardinalities"]],
            embedding_dimensions=[
                int(value) for value in tabm_metadata["embedding_dimensions"]
            ],
            k=int(architecture["k"]),
            n_blocks=int(architecture["n_blocks"]),
            d_block=int(architecture["d_block"]),
            dropout=float(architecture["dropout"]),
        )
        state_path = model_dir / tabm_metadata["model_filename"]
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(state_path, map_location="cpu")
        tabm_model.load_state_dict(state, strict=True)
        tabm_model.to(device)
        tabm_model.eval()
        predictor["tabm_model"] = tabm_model
        predictor["tabm_device"] = device
    return predictor


def validate_raw_input(frame: pd.DataFrame, metadata: dict) -> None:
    required = set(metadata["raw_input_columns"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"test.csv is missing columns: {missing}")
    if ID_COL not in frame:
        raise ValueError(f"test.csv is missing {ID_COL}")
    if frame[ID_COL].isna().any() or frame[ID_COL].astype("string").duplicated().any():
        raise ValueError("test row_id must be non-null and unique")


def predict_catboost_model(
    frame: pd.DataFrame,
    predictor: dict,
    model_key: str,
) -> np.ndarray:
    """Predict every supplied row with one serialized CatBoost model."""

    metadata = predictor["metadata"]
    validate_raw_input(frame, metadata)
    if model_key not in metadata:
        raise ValueError(f"unknown CatBoost metadata key: {model_key}")
    model_lookup = {
        "catboost": "catboost_model",
        "catboost_base": "catboost_base_model",
        "catboost_f_route": "catboost_f_route_model",
    }
    predictor_key = model_lookup.get(model_key)
    if predictor_key is None or predictor_key not in predictor:
        raise ValueError(f"CatBoost model is not loaded: {model_key}")
    features = prepare_catboost_features(frame, metadata[model_key])
    pool = Pool(
        features,
        cat_features=metadata[model_key]["categorical_columns"],
    )
    probability = np.clip(
        predictor[predictor_key].predict_proba(
            pool, thread_count=predictor["threads"]
        )[:, 1],
        EPSILON,
        1.0 - EPSILON,
    )
    return probability


def _predict_routed_catboost(frame: pd.DataFrame, predictor: dict) -> np.ndarray:
    metadata = predictor["metadata"]
    routing = metadata["routing"]
    routing_column = routing["column"]
    if routing_column not in frame:
        raise ValueError(f"test.csv is missing routing column: {routing_column}")
    values = frame[routing_column].astype("string")
    base_value = str(routing["base_value"])
    route_value = str(routing["f_route_value"])
    unexpected = sorted(
        str(value)
        for value in values.dropna().unique()
        if str(value) not in {base_value, route_value}
    )
    if values.isna().any() or unexpected:
        raise ValueError(
            f"unsupported {routing_column} values: {unexpected}; null={values.isna().any()}"
        )

    # Older pandas versions can expose nullable-boolean comparisons as an
    # object ndarray. Applying ``~`` to Python bool objects yields -1/-2,
    # which pandas then interprets as row labels instead of a boolean mask.
    # Force a native bool array and route by integer positions so inference is
    # independent of both the pandas version and the frame index.
    is_f = values.eq(route_value).fillna(False).to_numpy(dtype=np.bool_)
    base_positions = np.flatnonzero(np.logical_not(is_f))
    f_positions = np.flatnonzero(is_f)
    routed = np.empty(len(frame), dtype=np.float64)
    if len(base_positions):
        routed[base_positions] = predict_catboost_model(
            frame.iloc[base_positions], predictor, "catboost_base"
        )
    if len(f_positions):
        routed[f_positions] = predict_catboost_model(
            frame.iloc[f_positions], predictor, "catboost_f_route"
        )
    if not np.isfinite(routed).all():
        raise RuntimeError("routed CatBoost returned non-finite predictions")
    return routed


def predict_components(frame: pd.DataFrame, predictor: dict) -> tuple[np.ndarray, np.ndarray]:
    metadata = predictor["metadata"]
    validate_raw_input(frame, metadata)
    lgbm_features = prepare_lgbm_features(frame, metadata["lgbm"])
    lgbm_probability = np.clip(
        predictor["lgbm_model"].predict(lgbm_features), EPSILON, 1.0 - EPSILON
    )
    del lgbm_features
    if metadata["schema_version"] == 2:
        catboost_probability = predict_catboost_model(
            frame, predictor, "catboost"
        )
    else:
        catboost_probability = _predict_routed_catboost(frame, predictor)
    return lgbm_probability, catboost_probability


def predict_tabm_probabilities(frame: pd.DataFrame, predictor: dict) -> np.ndarray:
    metadata = predictor["metadata"]
    if metadata.get("schema_version") != 4:
        raise RuntimeError("TabM prediction is available only for schema 4")
    validate_raw_input(frame, metadata)
    if len(frame) == 0:
        return np.empty(0, dtype=np.float64)
    x_num, x_cat = prepare_tabm_features(frame, metadata["tabm"])
    model = predictor["tabm_model"]
    device = predictor["tabm_device"]
    batch_size = int(
        metadata["tabm"][
            "inference_batch_size_gpu"
            if device.type == "cuda"
            else "inference_batch_size_cpu"
        ]
    )
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            batch_num = torch.from_numpy(x_num[start:stop]).to(
                device, non_blocking=True
            )
            batch_cat = torch.from_numpy(x_cat[start:stop]).to(
                device, dtype=torch.long, non_blocking=True
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(batch_num, batch_cat).squeeze(-1)
            probability = torch.sigmoid(logits.float()).mean(dim=1)
            predictions.append(probability.cpu().numpy())
    output = np.concatenate(predictions).astype(np.float64)
    if len(output) != len(frame) or not np.isfinite(output).all():
        raise RuntimeError("TabM returned invalid predictions")
    return np.clip(output, EPSILON, 1.0 - EPSILON)


def predict_probabilities(frame: pd.DataFrame, predictor: dict) -> np.ndarray:
    lgbm_probability, catboost_probability = predict_components(frame, predictor)
    blend = predictor["metadata"]["blend"]
    if blend["method"] != "logit":
        raise RuntimeError(f"unsupported blend method: {blend['method']}")
    base_prediction = blend_logit(
        lgbm_probability,
        catboost_probability,
        lgbm_weight=float(blend["lgbm_weight"]),
    )
    metadata = predictor["metadata"]
    if metadata["schema_version"] == 4:
        neural_blend = metadata["neural_blend"]
        if neural_blend["method"] != "probability":
            raise RuntimeError(
                f"unsupported neural blend method: {neural_blend['method']}"
            )
        if neural_blend.get("scope") != "R_only":
            raise RuntimeError(
                f"unsupported neural blend scope: {neural_blend.get('scope')}"
            )
        tree_weight = float(neural_blend["tree_weight"])
        tabm_weight = float(neural_blend["tabm_weight"])
        if (
            tree_weight < 0.0
            or tabm_weight < 0.0
            or not np.isclose(tree_weight + tabm_weight, 1.0, atol=1e-12)
        ):
            raise RuntimeError("invalid neural blend weights")
        routing_column = metadata["routing"]["column"]
        values = frame[routing_column].astype("string")
        is_r = (
            values.eq(str(metadata["routing"]["base_value"]))
            .fillna(False)
            .to_numpy(dtype=np.bool_)
        )
        r_positions = np.flatnonzero(is_r)
        prediction = base_prediction.copy()
        if len(r_positions):
            tabm_probability = predict_tabm_probabilities(
                frame.iloc[r_positions], predictor
            )
            prediction[r_positions] = (
                tree_weight * base_prediction[r_positions]
                + tabm_weight * tabm_probability
            )
    else:
        prediction = base_prediction
    if len(prediction) != len(frame) or not np.isfinite(prediction).all():
        raise RuntimeError("model returned invalid predictions")
    return np.clip(prediction, 0.0, 1.0)


def make_submission(
    sample_submission: pd.DataFrame,
    test: pd.DataFrame,
    prediction: np.ndarray,
) -> pd.DataFrame:
    if list(sample_submission.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission columns must be {[ID_COL, TARGET_COL]}, "
            f"got {list(sample_submission.columns)}"
        )
    test_ids = test[ID_COL].astype("string")
    sample_ids = sample_submission[ID_COL].astype("string")
    if test_ids.duplicated().any() or sample_ids.duplicated().any():
        raise ValueError("row_id must be unique in test and sample_submission")
    if len(test_ids) != len(sample_ids) or set(test_ids) != set(sample_ids):
        raise ValueError("test and sample_submission row_id sets do not match")
    prediction_by_id = pd.Series(prediction, index=test_ids.to_numpy())
    ordered_prediction = prediction_by_id.reindex(sample_ids.to_numpy()).to_numpy()
    if not np.isfinite(ordered_prediction).all():
        raise RuntimeError("submission contains missing or non-finite predictions")
    submission = sample_submission[[ID_COL]].copy()
    submission[TARGET_COL] = ordered_prediction.astype("float64")
    return submission


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    model_dir = root / "model"
    output_dir = root / "output"
    test = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig", low_memory=False)
    sample_submission = pd.read_csv(
        data_dir / "sample_submission.csv", encoding="utf-8-sig", low_memory=False
    )
    print(f"test rows={len(test):,}")
    predictor = load_predictor(model_dir)
    prediction = predict_probabilities(test, predictor)
    submission = make_submission(sample_submission, test, prediction)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "submission.csv"
    submission.to_csv(output_path, index=False, encoding="utf-8")
    print(
        f"Saved {output_path} | rows={len(submission):,} "
        f"min={prediction.min():.6f} mean={prediction.mean():.6f} "
        f"max={prediction.max():.6f}"
    )


if __name__ == "__main__":
    main()
