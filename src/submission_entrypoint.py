#!/usr/bin/env python3
"""Self-contained inference entrypoint for DACON submission schemas 2 and 3.

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
    if schema_version not in (2, 3):
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

    is_f = values.eq(route_value).to_numpy()
    is_base = ~is_f
    routed = np.empty(len(frame), dtype=np.float64)
    if is_base.any():
        routed[is_base] = predict_catboost_model(
            frame.loc[is_base], predictor, "catboost_base"
        )
    if is_f.any():
        routed[is_f] = predict_catboost_model(
            frame.loc[is_f], predictor, "catboost_f_route"
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


def predict_probabilities(frame: pd.DataFrame, predictor: dict) -> np.ndarray:
    lgbm_probability, catboost_probability = predict_components(frame, predictor)
    blend = predictor["metadata"]["blend"]
    if blend["method"] != "logit":
        raise RuntimeError(f"unsupported blend method: {blend['method']}")
    prediction = blend_logit(
        lgbm_probability,
        catboost_probability,
        lgbm_weight=float(blend["lgbm_weight"]),
    )
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
