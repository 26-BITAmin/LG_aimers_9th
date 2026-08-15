"""Row-independent feature engineering for the LG Aimers 9th data.

All features in this module are built from the current row and constants learned
from the training fold. No statistic is calculated from other validation/test
rows, which keeps inference compliant with the competition rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"

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
CategoryMaps = dict[str, list[str]]


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
    """Weighted row mean that renormalizes over the non-missing values."""
    weight = pd.Series(weights, index=frame.columns, dtype="float64")
    numerator = frame.mul(weight, axis=1).sum(axis=1, min_count=1)
    denominator = frame.notna().mul(weight, axis=1).sum(axis=1)
    return numerator / denominator.replace(0.0, np.nan)


def add_rate_gap_features(out: pd.DataFrame) -> None:
    """Add interpretable differences between the official as-of rates."""
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
    """Describe recent direction and volatility using current-row as-of fields."""
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

    out["recent_success_1_minus_5"] = (
        recent_success.iloc[:, 0] - recent_success.iloc[:, 2]
    )
    out["recent_success_3_minus_5"] = (
        recent_success.iloc[:, 1] - recent_success.iloc[:, 2]
    )
    out["recent_success_1_minus_career"] = (
        recent_success.iloc[:, 0] - out["asof_pitcher_success_rate"]
    )
    out["recent_success_5_minus_career"] = (
        recent_success.iloc[:, 2] - out["asof_pitcher_success_rate"]
    )
    out["recent_success_std"] = recent_success.std(axis=1, ddof=0)
    out["recent_success_range"] = (
        recent_success.max(axis=1) - recent_success.min(axis=1)
    )
    out["recent_success_weighted"] = _weighted_row_mean(
        recent_success, [0.50, 0.30, 0.20]
    )

    out["recent_middle_1_minus_5"] = (
        recent_middle.iloc[:, 0] - recent_middle.iloc[:, 2]
    )
    out["recent_middle_3_minus_5"] = (
        recent_middle.iloc[:, 1] - recent_middle.iloc[:, 2]
    )
    out["recent_middle_std"] = recent_middle.std(axis=1, ddof=0)
    out["recent_middle_range"] = (
        recent_middle.max(axis=1) - recent_middle.min(axis=1)
    )
    out["recent_middle_weighted"] = _weighted_row_mean(
        recent_middle, [0.50, 0.30, 0.20]
    )
    out["reliability_adjusted_form_gap"] = (
        out["pitcher_reliability"] * out["pitcher_form_gap"]
    )


def add_pitchmix_shape_features(out: pd.DataFrame) -> None:
    """Summarize the shape and reliability of the official pitch-mix rates."""
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
    out["pitchmix_entropy"] = -(normalized * np.log(safe)).sum(
        axis=1, min_count=1
    )
    out["pitchmix_concentration"] = np.square(normalized).sum(
        axis=1, min_count=1
    )
    out["fastball_minus_breaking"] = normalized.iloc[:, 0] - normalized.iloc[:, 1]
    out["fastball_minus_offspeed"] = normalized.iloc[:, 0] - normalized.iloc[:, 2]
    out["breaking_minus_offspeed"] = normalized.iloc[:, 1] - normalized.iloc[:, 2]
    pitchmix_n = out["asof_pitcher_pitchmix_n"].astype("float64")
    out["pitchmix_reliability"] = pitchmix_n / (pitchmix_n + 200.0)
    out["has_pitchmix_history"] = (pitchmix_n > 0).astype("int8")


def add_context_interaction_features(out: pd.DataFrame) -> None:
    """Add compact interactions for count, base, inning, score, and leverage."""
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
        out["inning"],
        bins=[-np.inf, 3, 6, np.inf],
        labels=[0, 1, 2],
    ).astype("int8")
    out["score_diff_bucket"] = (
        out["score_diff_pitcher_team"].clip(-4, 4) + 4
    ).astype("int8")
    out["inning_score_state"] = (
        out["inning_bucket"].astype("int16") * 10
        + out["score_diff_bucket"].astype("int16")
    )
    out["pressure_index"] = out["li"] * (
        1.0 + out["is_full_count"] + out["is_tied"]
    )
    out["count_ball_propensity"] = (
        out["balls_before"] * out["asof_pitcher_ball_rate"]
    )
    out["count_strike_propensity"] = (
        out["strikes_before"] * out["asof_pitcher_strike_rate"]
    )
    out["leverage_score_closeness"] = out["li"] / (1.0 + out["abs_score_diff"])
    out["runner_pressure"] = out["num_runners_on"] * out["li"]


def add_team_context_features(out: pd.DataFrame) -> None:
    """Express game state from the pitcher's team perspective."""
    pitcher_is_home = out["top_bottom"].astype("string").eq("T")
    out["pitcher_win_expectancy"] = np.where(
        pitcher_is_home,
        out["home_win_expectancy"],
        out["away_win_expectancy"],
    )
    out["batter_win_expectancy"] = np.where(
        pitcher_is_home,
        out["away_win_expectancy"],
        out["home_win_expectancy"],
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
    df: pd.DataFrame,
    *,
    target_prior: float,
    config: FeatureConfig,
) -> pd.DataFrame:
    """Create features using only values from each individual row."""
    drop_cols = [c for c in (ID_COL, TARGET_COL) if c in df.columns]
    out = df.drop(columns=drop_cols).copy()

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

    kp = float(config.pitcher_prior_strength)
    kb = float(config.batter_prior_strength)
    out["pitcher_reliability"] = pitcher_n / (pitcher_n + kp)
    out["batter_reliability"] = batter_n / (batter_n + kb)
    out["smoothed_pitcher_success_rate"] = (
        pitcher_n * pitcher_rate + kp * target_prior
    ) / (pitcher_n + kp)
    out["smoothed_batter_success_rate"] = (
        batter_n * batter_rate + kb * target_prior
    ) / (batter_n + kb)

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


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    """Return categorical columns that are present in a feature frame."""
    return [
        c
        for c in RAW_CATEGORICAL_COLUMNS + DERIVED_CATEGORICAL_COLUMNS
        if c in frame.columns
    ]


def fit_category_maps(train: pd.DataFrame, categorical: list[str]) -> CategoryMaps:
    """Learn stable category vocabularies from the training fold only."""
    return {
        col: train[col].astype("string").fillna("__MISSING__").unique().tolist()
        for col in categorical
    }


def apply_category_maps(frame: pd.DataFrame, category_maps: CategoryMaps) -> pd.DataFrame:
    """Apply stored category vocabularies; unseen values become code -1."""
    out = frame.copy()
    missing_columns = sorted(set(category_maps) - set(out.columns))
    if missing_columns:
        raise ValueError(f"categorical columns are missing: {missing_columns}")

    for col, categories in category_maps.items():
        values = out[col].astype("string").fillna("__MISSING__")
        out[col] = pd.Categorical(values, categories=categories).codes.astype("int32")

    for col in out.columns:
        if col in category_maps:
            continue
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].astype("float32")
        elif pd.api.types.is_integer_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="integer")
    return out


def encode_categories_train_only(
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], CategoryMaps]:
    """Encode categories using vocabularies learned from the training fold."""
    categorical = categorical_columns(train)
    category_maps = fit_category_maps(train, categorical)
    return (
        apply_category_maps(train, category_maps),
        apply_category_maps(valid, category_maps),
        categorical,
        category_maps,
    )


def prepare_fold_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], CategoryMaps, float]:
    """Build a leakage-safe pair of training and validation matrices."""
    prior = float(train_df[TARGET_COL].mean())
    x_train = add_rowwise_features(train_df, target_prior=prior, config=config)
    x_valid = add_rowwise_features(valid_df, target_prior=prior, config=config)
    x_train, x_valid, categorical, category_maps = encode_categories_train_only(
        x_train, x_valid
    )
    if list(x_train.columns) != list(x_valid.columns):
        raise RuntimeError("train and validation feature columns do not match")
    return x_train, x_valid, categorical, category_maps, prior


def prepare_catboost_fold_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], float]:
    """Build CatBoost matrices while preserving native categorical values.

    The target prior is learned from the training fold only. Categorical values
    are converted to strings so CatBoost can build ordered statistics without
    reading or aggregating other validation/test rows.
    """
    prior = float(train_df[TARGET_COL].mean())
    x_train = add_rowwise_features(train_df, target_prior=prior, config=config)
    x_valid = add_rowwise_features(valid_df, target_prior=prior, config=config)
    categorical = categorical_columns(x_train)

    if list(x_train.columns) != list(x_valid.columns):
        raise RuntimeError("train and validation feature columns do not match")
    for col in categorical:
        x_train[col] = (
            x_train[col].astype("string").fillna("__MISSING__").astype(str)
        )
        x_valid[col] = (
            x_valid[col].astype("string").fillna("__MISSING__").astype(str)
        )
    return x_train, x_valid, categorical, prior


def prepare_inference_features(
    df: pd.DataFrame,
    *,
    config: FeatureConfig,
    target_prior: float,
    category_maps: CategoryMaps,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Transform raw validation/test rows with a fitted feature bundle."""
    features = add_rowwise_features(df, target_prior=target_prior, config=config)
    features = apply_category_maps(features, category_maps)
    expected = set(feature_columns)
    actual = set(features.columns)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"inference feature mismatch: missing={missing}, extra={extra}")
    return features.loc[:, feature_columns]
