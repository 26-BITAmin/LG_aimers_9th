#!/usr/bin/env python3
"""Train an honest 2019-2023 -> 2024 LightGBM probability baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = PROJECT_ROOT / ".cache/matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation import calibration_table, grouped_metrics, probability_metrics
from features import FeatureConfig, ID_COL, TARGET_COL, prepare_fold_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data/train.csv")
    parser.add_argument("--valid-season", type=int, default=2024)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/lgbm_holdout_2024")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=1000)
    parser.add_argument("--half-life", type=float, default=2.5)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--no-recency-weight", action="store_true")
    parser.add_argument("--exclude-season", action="store_true")
    parser.add_argument("--exclude-player-ids", action="store_true")
    parser.add_argument(
        "--train-game-type",
        choices=["all", "R", "F"],
        default="all",
        help="Optionally restrict only the training fold; validation remains complete.",
    )
    return parser.parse_args()


def recency_weights(seasons: pd.Series, valid_season: int, half_life: float) -> np.ndarray:
    age = (valid_season - 1) - seasons.to_numpy(dtype=np.float64)
    weights = np.power(0.5, age / half_life)
    return weights / weights.mean()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    print(f"Loading {args.train_path} ...")
    data = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False)
    if not set(data[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError("target must contain only 0 and 1")

    train_df = data[data["season"] < args.valid_season].copy()
    valid_df = data[data["season"] == args.valid_season].copy()
    if args.train_game_type != "all":
        train_df = train_df[train_df["game_type"] == args.train_game_type].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError("the requested temporal split is empty")
    print(f"Train={train_df.shape}, valid={valid_df.shape}")

    feature_config = FeatureConfig(
        include_season=not args.exclude_season,
        include_player_ids=not args.exclude_player_ids,
    )
    x_train, x_valid, categorical, prior = prepare_fold_features(
        train_df, valid_df, config=feature_config
    )
    y_train = train_df[TARGET_COL].to_numpy(dtype=np.int8)
    y_valid = valid_df[TARGET_COL].to_numpy(dtype=np.int8)

    sample_weight = None
    if not args.no_recency_weight:
        sample_weight = recency_weights(train_df["season"], args.valid_season, args.half_life)

    params = {
        "objective": "binary",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
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
    print("Training LightGBM ...")
    model = lgb.LGBMClassifier(**params)
    train_started = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        categorical_feature=categorical,
    )
    train_seconds = time.perf_counter() - train_started
    prediction = np.clip(model.predict_proba(x_valid)[:, 1], 0.0, 1.0)

    overall = probability_metrics(y_valid, prediction)
    overall.update(
        {
            "valid_season": args.valid_season,
            "train_seasons": sorted(int(v) for v in train_df["season"].unique()),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "feature_count": int(x_train.shape[1]),
            "target_prior_from_train": prior,
            "recency_weighted": not args.no_recency_weight,
            "half_life": None if args.no_recency_weight else args.half_life,
            "train_game_type": args.train_game_type,
            "include_season": not args.exclude_season,
            "include_player_ids": not args.exclude_player_ids,
            "train_seconds": train_seconds,
            "total_seconds": time.perf_counter() - started,
            "parameters": params,
        }
    )

    prediction_frame = valid_df[[ID_COL, "season", "game_type", TARGET_COL]].copy()
    prediction_frame["prediction"] = prediction
    prediction_frame.to_csv(args.output_dir / "predictions.csv.gz", index=False, compression="gzip")
    grouped_metrics(train_df, valid_df, prediction).to_csv(
        args.output_dir / "group_metrics.csv", index=False
    )
    calibration_table(y_valid, prediction).to_csv(
        args.output_dir / "calibration.csv", index=False
    )
    pd.DataFrame(
        {"feature": x_train.columns, "importance_gain": model.booster_.feature_importance("gain")}
    ).sort_values("importance_gain", ascending=False).to_csv(
        args.output_dir / "feature_importance.csv", index=False
    )
    joblib.dump(
        {
            "model": model,
            "feature_columns": list(x_train.columns),
            "categorical_columns": categorical,
            "feature_config": asdict(feature_config),
            "target_prior": prior,
        },
        args.output_dir / "model.joblib",
        compress=3,
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
