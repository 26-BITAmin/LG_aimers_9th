#!/usr/bin/env python3
"""Train an honest 2019-2023 -> 2024 LightGBM probability baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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
import sklearn

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation import calibration_table, grouped_metrics, probability_metrics
from features import (
    FeatureConfig,
    ID_COL,
    TARGET_COL,
    prepare_fold_features,
    prepare_inference_features,
)


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
    x_train, x_valid, categorical, category_maps, prior = prepare_fold_features(
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
    configuration = {
        "valid_season": args.valid_season,
        "train_game_type": args.train_game_type,
        "include_season": not args.exclude_season,
        "include_player_ids": not args.exclude_player_ids,
        "recency_weighted": not args.no_recency_weight,
        "half_life": None if args.no_recency_weight else args.half_life,
        "feature_config": asdict(feature_config),
        "parameters": params,
    }
    configuration_id = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:12]
    overall.update(
        {
            "configuration_id": configuration_id,
            "valid_season": args.valid_season,
            "train_seasons": sorted(int(v) for v in train_df["season"].unique()),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "all_data_rows": int(len(data)),
            "all_data_columns": int(data.shape[1]),
            "target_sum": int(data[TARGET_COL].sum()),
            "train_file_size_bytes": int(args.train_path.stat().st_size),
            "feature_count": int(x_train.shape[1]),
            "feature_columns": list(x_train.columns),
            "categorical_columns": categorical,
            "category_map_sizes": {k: len(v) for k, v in category_maps.items()},
            "target_prior_from_train": prior,
            "recency_weighted": not args.no_recency_weight,
            "half_life": None if args.no_recency_weight else args.half_life,
            "train_game_type": args.train_game_type,
            "include_season": not args.exclude_season,
            "include_player_ids": not args.exclude_player_ids,
            "train_seconds": train_seconds,
            "parameters": params,
            "python_version": platform.python_version(),
            "package_versions": {
                "joblib": joblib.__version__,
                "lightgbm": lgb.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
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
    model_path = args.output_dir / "model.joblib"
    feature_columns = list(x_train.columns)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "categorical_columns": categorical,
            "category_maps": category_maps,
            "feature_config": asdict(feature_config),
            "target_prior": prior,
            "configuration_id": configuration_id,
        },
        model_path,
        compress=3,
    )

    # Verify that the serialized bundle can recreate the exact validation input
    # and prediction without fitting or reading any other validation rows.
    reloaded = joblib.load(model_path)
    reload_config = FeatureConfig(**reloaded["feature_config"])
    x_valid_reloaded = prepare_inference_features(
        valid_df,
        config=reload_config,
        target_prior=float(reloaded["target_prior"]),
        category_maps=reloaded["category_maps"],
        feature_columns=reloaded["feature_columns"],
    )
    pd.testing.assert_frame_equal(x_valid, x_valid_reloaded, check_exact=True)
    reloaded_prediction = np.clip(
        reloaded["model"].predict_proba(x_valid_reloaded)[:, 1], 0.0, 1.0
    )
    reload_max_abs_diff = float(np.max(np.abs(prediction - reloaded_prediction)))
    if reload_max_abs_diff > 1e-12:
        raise RuntimeError(
            f"serialized prediction mismatch: max_abs_diff={reload_max_abs_diff}"
        )
    overall.update(
        {
            "reload_feature_frame_equal": True,
            "reload_max_abs_diff": reload_max_abs_diff,
            "model_size_bytes": int(model_path.stat().st_size),
            "total_seconds": time.perf_counter() - started,
        }
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
