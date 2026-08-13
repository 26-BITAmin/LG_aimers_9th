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

DERIVED_CATEGORICAL_COLUMNS = ["count_state", "hand_matchup"]


@dataclass(frozen=True)
class FeatureConfig:
    pitcher_prior_strength: float = 200.0
    batter_prior_strength: float = 300.0
    include_season: bool = True
    include_player_ids: bool = True


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

    if not config.include_player_ids:
        out = out.drop(columns=["pitcher_id", "batter_id"])
    if not config.include_season:
        out = out.drop(columns=["season"])
    return out


def encode_categories_train_only(
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Ordinal-encode categories with vocabularies learned from train only.

    Unseen validation/test values become -1, which LightGBM treats as missing
    for categorical features.
    """
    train = train.copy()
    valid = valid.copy()
    categorical = [
        c
        for c in RAW_CATEGORICAL_COLUMNS + DERIVED_CATEGORICAL_COLUMNS
        if c in train.columns
    ]
    for col in categorical:
        tr = train[col].astype("string").fillna("__MISSING__")
        va = valid[col].astype("string").fillna("__MISSING__")
        categories = pd.Index(tr.unique())
        train[col] = pd.Categorical(tr, categories=categories).codes.astype("int32")
        valid[col] = pd.Categorical(va, categories=categories).codes.astype("int32")

    for frame in (train, valid):
        for col in frame.columns:
            if col in categorical:
                continue
            if pd.api.types.is_float_dtype(frame[col]):
                frame[col] = frame[col].astype("float32")
            elif pd.api.types.is_integer_dtype(frame[col]):
                frame[col] = pd.to_numeric(frame[col], downcast="integer")
    return train, valid, categorical


def prepare_fold_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], float]:
    """Build a leakage-safe pair of training and validation matrices."""
    prior = float(train_df[TARGET_COL].mean())
    x_train = add_rowwise_features(train_df, target_prior=prior, config=config)
    x_valid = add_rowwise_features(valid_df, target_prior=prior, config=config)
    x_train, x_valid, categorical = encode_categories_train_only(x_train, x_valid)
    if list(x_train.columns) != list(x_valid.columns):
        raise RuntimeError("train and validation feature columns do not match")
    return x_train, x_valid, categorical, prior
