#!/usr/bin/env python3
"""Leakage-safe LightGBM + CatBoost forward-CV ensemble experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from evaluation import grouped_metrics, probability_metrics
from features import (
    FeatureConfig,
    ID_COL,
    TARGET_COL,
    prepare_catboost_fold_features,
    prepare_fold_features,
)

RESULT_SCHEMA_VERSION = 1
BLEND_WEIGHTS = np.linspace(0.0, 1.0, 41)
BLEND_METHODS = ("probability", "logit")


@dataclass(frozen=True)
class ExperimentConfiguration:
    name: str = "all_recent_h05_no_game_type"
    half_life: float = 0.5
    include_season: bool = False
    include_player_ids: bool = True
    include_game_type: bool = False
    pitcher_prior_strength: float = 200.0
    batter_prior_strength: float = 300.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data/train.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/lgbm_catboost_ensemble_v1",
    )
    parser.add_argument("--valid-seasons", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--lgbm-estimators", type=int, default=1_500)
    parser.add_argument("--catboost-iterations", type=int, default=1_200)
    parser.add_argument(
        "--catboost-task-type",
        choices=["auto", "CPU", "GPU"],
        default="auto",
    )
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def resolve_catboost_task_type(requested: str) -> str:
    if requested != "auto":
        return requested
    return "GPU" if shutil.which("nvidia-smi") else "CPU"


def recency_weights(
    seasons: pd.Series,
    *,
    valid_season: int,
    half_life: float,
) -> np.ndarray:
    age = (valid_season - 1) - seasons.to_numpy(dtype=np.float64)
    weights = np.power(0.5, age / half_life)
    return weights / weights.mean()


def training_frame(data: pd.DataFrame, valid_season: int) -> pd.DataFrame:
    return data[data["season"] < valid_season].copy()


def lgbm_brier_metric(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
    return "brier", float(np.mean(np.square(y_pred - y_true))), False


def parameter_bundle(args: argparse.Namespace, catboost_task_type: str) -> tuple[dict, dict]:
    lgbm_parameters = {
        "objective": "binary",
        "metric": "None",
        "n_estimators": args.lgbm_estimators,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "min_child_samples": 1_000,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.2,
        "reg_lambda": 10.0,
        "max_bin": 127,
        "random_state": 2026,
        "n_jobs": args.threads,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    catboost_parameters = {
        "loss_function": "Logloss",
        # CatBoost does not support BrierScore as a GPU optimization metric.
        # Final model and blend selection is performed on saved raw OOF Brier.
        "eval_metric": "Logloss",
        "iterations": args.catboost_iterations,
        "learning_rate": 0.04,
        "depth": 8,
        "l2_leaf_reg": 12.0,
        "random_strength": 0.5,
        "random_seed": 2026,
        "max_ctr_complexity": 1,
        "allow_writing_files": False,
        "task_type": catboost_task_type,
        "thread_count": args.threads,
    }
    if catboost_task_type == "GPU":
        catboost_parameters.update({"devices": "0", "gpu_ram_part": 0.85})
    return lgbm_parameters, catboost_parameters


def experiment_signature(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def prediction_cache_paths(run_dir: Path, valid_season: int, model_name: str) -> tuple[Path, Path]:
    stem = run_dir / f"fold_{valid_season}_{model_name}"
    return stem.with_suffix(".csv.gz"), stem.with_name(stem.name + "_metadata.json")


def load_cached_prediction(
    path: Path,
    metadata_path: Path,
    valid: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    cached = pd.read_csv(path, dtype={ID_COL: "string"})
    expected_ids = valid[ID_COL].astype("string").to_numpy()
    cached_ids = cached[ID_COL].to_numpy()
    if not np.array_equal(expected_ids, cached_ids):
        raise RuntimeError(f"cache row order mismatch: {path}")
    prediction = cached["prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all() or ((prediction < 0.0) | (prediction > 1.0)).any():
        raise RuntimeError(f"invalid cached probabilities: {path}")
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)
    print(f"verified cache: {path.name}")
    return prediction, metadata


def save_prediction_cache(
    path: Path,
    metadata_path: Path,
    valid: pd.DataFrame,
    prediction: np.ndarray,
    metadata: dict,
) -> None:
    pd.DataFrame(
        {ID_COL: valid[ID_COL].astype(str), "prediction": prediction}
    ).to_csv(path, index=False, compression="gzip")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def train_lgbm_fold(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    feature_config: FeatureConfig,
    weights: np.ndarray,
    parameters: dict,
) -> tuple[np.ndarray, dict]:
    x_train, x_valid, categorical, _, _ = prepare_fold_features(
        train, valid, config=feature_config
    )
    y_train = train[TARGET_COL].to_numpy(dtype=np.int8)
    y_valid = valid[TARGET_COL].to_numpy(dtype=np.int8)
    model = lgb.LGBMClassifier(**parameters)
    started = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        sample_weight=weights,
        categorical_feature=categorical,
        eval_set=[(x_valid, y_valid)],
        eval_metric=lgbm_brier_metric,
        callbacks=[
            lgb.early_stopping(150, first_metric_only=True, verbose=False),
            lgb.log_evaluation(100),
        ],
    )
    prediction = np.clip(model.predict_proba(x_valid)[:, 1], 1e-7, 1.0 - 1e-7)
    metadata = {
        "model": "lightgbm",
        "best_iteration": int(model.best_iteration_ or parameters["n_estimators"]),
        "feature_count": int(x_train.shape[1]),
        "categorical_count": int(len(categorical)),
        "train_seconds": time.perf_counter() - started,
    }
    del x_train, x_valid, model
    gc.collect()
    return prediction, metadata


def train_catboost_fold(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    feature_config: FeatureConfig,
    weights: np.ndarray,
    parameters: dict,
) -> tuple[np.ndarray, dict]:
    x_train, x_valid, categorical, _ = prepare_catboost_fold_features(
        train, valid, config=feature_config
    )
    y_train = train[TARGET_COL].to_numpy(dtype=np.int8)
    y_valid = valid[TARGET_COL].to_numpy(dtype=np.int8)
    model = CatBoostClassifier(**parameters)
    started = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        cat_features=categorical,
        sample_weight=weights,
        eval_set=(x_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=150,
        verbose=100,
    )
    prediction = np.clip(model.predict_proba(x_valid)[:, 1], 1e-7, 1.0 - 1e-7)
    best_iteration = int(model.get_best_iteration())
    metadata = {
        "model": "catboost",
        "best_iteration": best_iteration + 1 if best_iteration >= 0 else parameters["iterations"],
        "feature_count": int(x_train.shape[1]),
        "categorical_count": int(len(categorical)),
        "train_seconds": time.perf_counter() - started,
        "task_type": parameters["task_type"],
    }
    del x_train, x_valid, model
    gc.collect()
    return prediction, metadata


def blend_predictions(
    lgbm_prediction: np.ndarray,
    catboost_prediction: np.ndarray,
    *,
    lgbm_weight: float,
    method: str,
) -> np.ndarray:
    if method == "probability":
        return lgbm_weight * lgbm_prediction + (1.0 - lgbm_weight) * catboost_prediction
    if method == "logit":
        eps = 1e-7
        lgbm_logit = np.log(np.clip(lgbm_prediction, eps, 1 - eps) / np.clip(1 - lgbm_prediction, eps, 1 - eps))
        catboost_logit = np.log(np.clip(catboost_prediction, eps, 1 - eps) / np.clip(1 - catboost_prediction, eps, 1 - eps))
        blended_logit = lgbm_weight * lgbm_logit + (1.0 - lgbm_weight) * catboost_logit
        return 1.0 / (1.0 + np.exp(-np.clip(blended_logit, -30.0, 30.0)))
    raise ValueError(f"unknown blend method: {method}")


def build_blend_grid(
    oof: pd.DataFrame,
    valid_seasons: list[int],
    selection_seasons: list[int],
) -> pd.DataFrame:
    rows = []
    y = oof[TARGET_COL].to_numpy(dtype=np.int8)
    lgbm_prediction = oof["lgbm_prediction"].to_numpy(dtype=np.float64)
    catboost_prediction = oof["catboost_prediction"].to_numpy(dtype=np.float64)
    latest_season = max(valid_seasons)
    selection_mask = oof["valid_season"].isin(selection_seasons).to_numpy()
    for method in BLEND_METHODS:
        for lgbm_weight in BLEND_WEIGHTS:
            prediction = blend_predictions(
                lgbm_prediction,
                catboost_prediction,
                lgbm_weight=float(lgbm_weight),
                method=method,
            )
            all_folds = probability_metrics(y, prediction)
            selection = probability_metrics(y[selection_mask], prediction[selection_mask])
            row = {
                "method": method,
                "lgbm_weight": float(lgbm_weight),
                "catboost_weight": float(1.0 - lgbm_weight),
                "selection_brier": selection["brier"],
                "selection_bss_raw": selection["bss_raw"],
                "selection_mean_gap": selection["mean_gap"],
                "all_folds_brier": all_folds["brier"],
                "all_folds_bss_raw": all_folds["bss_raw"],
            }
            selection_fold_briers = []
            for season in valid_seasons:
                mask = oof["valid_season"].eq(season).to_numpy()
                fold = probability_metrics(y[mask], prediction[mask])
                row[f"brier_{season}"] = fold["brier"]
                row[f"bss_raw_{season}"] = fold["bss_raw"]
                if season in selection_seasons:
                    selection_fold_briers.append(fold["brier"])
            row["latest_brier"] = row[f"brier_{latest_season}"]
            row["worst_selection_fold_brier"] = max(selection_fold_briers)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["selection_brier", "worst_selection_fold_brier", "latest_brier"],
        ignore_index=True,
    )


def choose_blend(frame: pd.DataFrame) -> dict:
    y = frame[TARGET_COL].to_numpy(dtype=np.int8)
    lgbm_prediction = frame["lgbm_prediction"].to_numpy(dtype=np.float64)
    catboost_prediction = frame["catboost_prediction"].to_numpy(dtype=np.float64)
    candidates = []
    for method in BLEND_METHODS:
        for lgbm_weight in BLEND_WEIGHTS:
            prediction = blend_predictions(
                lgbm_prediction,
                catboost_prediction,
                lgbm_weight=float(lgbm_weight),
                method=method,
            )
            candidates.append(
                {
                    "method": method,
                    "lgbm_weight": float(lgbm_weight),
                    "brier": probability_metrics(y, prediction)["brier"],
                }
            )
    return min(candidates, key=lambda item: (item["brier"], item["lgbm_weight"]))


def make_metric_rows(
    oof: pd.DataFrame,
    prediction_columns: dict[str, str],
    valid_seasons: list[int],
    selection_seasons: list[int],
) -> pd.DataFrame:
    rows = []
    partitions: list[tuple[str, int | None, pd.DataFrame]] = [
        ("pooled", None, oof),
        (
            "recent_pooled",
            None,
            oof[oof["valid_season"].isin(selection_seasons)],
        ),
    ]
    partitions.extend(
        ("fold", season, oof[oof["valid_season"].eq(season)])
        for season in valid_seasons
    )
    for model_name, prediction_column in prediction_columns.items():
        for partition, season, frame in partitions:
            scope_masks = {
                "all": np.ones(len(frame), dtype=bool),
                "R": frame["game_type"].eq("R").to_numpy(),
                "F": frame["game_type"].eq("F").to_numpy(),
            }
            for scope, mask in scope_masks.items():
                if not mask.any():
                    continue
                metric = probability_metrics(
                    frame.loc[mask, TARGET_COL], frame.loc[mask, prediction_column]
                )
                metric.update(
                    {
                        "model": model_name,
                        "partition": partition,
                        "valid_season": season,
                        "scope": scope,
                    }
                )
                rows.append(metric)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    valid_seasons = sorted(set(args.valid_seasons))
    if len(valid_seasons) < 2:
        raise ValueError("at least two validation seasons are required for blend checks")
    selection_seasons = valid_seasons[-2:]
    catboost_task_type = resolve_catboost_task_type(args.catboost_task_type)
    configuration = ExperimentConfiguration()
    feature_config = FeatureConfig(
        pitcher_prior_strength=configuration.pitcher_prior_strength,
        batter_prior_strength=configuration.batter_prior_strength,
        include_season=configuration.include_season,
        include_player_ids=configuration.include_player_ids,
        include_game_type=configuration.include_game_type,
    )
    lgbm_parameters, catboost_parameters = parameter_bundle(args, catboost_task_type)
    signature_payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "configuration": asdict(configuration),
        "valid_seasons": valid_seasons,
        "selection_seasons": selection_seasons,
        "lgbm_parameters": lgbm_parameters,
        "catboost_parameters": catboost_parameters,
    }
    signature = experiment_signature(signature_payload)
    run_dir = args.output_dir / f"run_{signature}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "configuration.json").open("w", encoding="utf-8") as f:
        json.dump({"signature": signature, **signature_payload}, f, ensure_ascii=False, indent=2)

    print(f"Loading {args.train_path} ...")
    data = pd.read_csv(args.train_path, encoding="utf-8-sig", low_memory=False)
    if not set(data[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError("target must contain only 0 and 1")
    missing_seasons = [season for season in valid_seasons if season not in set(data["season"])]
    if missing_seasons:
        raise ValueError(f"validation seasons missing from training data: {missing_seasons}")
    print(
        f"configuration={configuration.name} folds={valid_seasons} "
        f"CatBoost={catboost_task_type} rows={len(data):,}"
    )

    oof_parts: list[pd.DataFrame] = []
    training_metadata: list[dict] = []
    for valid_season in valid_seasons:
        train = training_frame(data, valid_season)
        valid = data[data["season"].eq(valid_season)].copy()
        weights = recency_weights(
            train["season"],
            valid_season=valid_season,
            half_life=configuration.half_life,
        )
        print(
            f"\nfold={valid_season} train={len(train):,} valid={len(valid):,} "
            f"weight_min={weights.min():.6f} weight_max={weights.max():.3f}"
        )

        predictions: dict[str, np.ndarray] = {}
        for model_name, trainer, parameters in [
            ("lgbm", train_lgbm_fold, lgbm_parameters),
            ("catboost", train_catboost_fold, catboost_parameters),
        ]:
            cache_path, metadata_path = prediction_cache_paths(
                run_dir, valid_season, model_name
            )
            if cache_path.exists() and not args.force_rerun:
                prediction, metadata = load_cached_prediction(
                    cache_path, metadata_path, valid
                )
            else:
                prediction, metadata = trainer(
                    train,
                    valid,
                    feature_config=feature_config,
                    weights=weights,
                    parameters=parameters,
                )
                save_prediction_cache(
                    cache_path, metadata_path, valid, prediction, metadata
                )
            metadata.update({"valid_season": valid_season, "model": model_name})
            training_metadata.append(metadata)
            predictions[model_name] = prediction

        fold_oof = valid[[ID_COL, TARGET_COL, "season", "game_type"]].copy()
        fold_oof["valid_season"] = valid_season
        fold_oof["lgbm_prediction"] = predictions["lgbm"]
        fold_oof["catboost_prediction"] = predictions["catboost"]
        oof_parts.append(fold_oof)
        del train, valid, weights, predictions, fold_oof
        gc.collect()

    oof = pd.concat(oof_parts, ignore_index=True)
    blend_grid = build_blend_grid(oof, valid_seasons, selection_seasons)
    selected = blend_grid.iloc[0]
    selected_method = str(selected["method"])
    selected_weight = float(selected["lgbm_weight"])
    oof["blend_selected_recent_oof"] = blend_predictions(
        oof["lgbm_prediction"].to_numpy(),
        oof["catboost_prediction"].to_numpy(),
        lgbm_weight=selected_weight,
        method=selected_method,
    )
    oof["blend_probability_50_50"] = blend_predictions(
        oof["lgbm_prediction"].to_numpy(),
        oof["catboost_prediction"].to_numpy(),
        lgbm_weight=0.5,
        method="probability",
    )

    oof["blend_cross_fitted"] = np.nan
    meta_rows = []
    for holdout_season in valid_seasons:
        meta_train = oof[
            ~oof["valid_season"].eq(holdout_season)
            & oof["valid_season"].isin(selection_seasons)
        ]
        if meta_train.empty:
            meta_train = oof[~oof["valid_season"].eq(holdout_season)]
        holdout_mask = oof["valid_season"].eq(holdout_season).to_numpy()
        choice = choose_blend(meta_train)
        holdout_prediction = blend_predictions(
            oof.loc[holdout_mask, "lgbm_prediction"].to_numpy(),
            oof.loc[holdout_mask, "catboost_prediction"].to_numpy(),
            lgbm_weight=choice["lgbm_weight"],
            method=choice["method"],
        )
        oof.loc[holdout_mask, "blend_cross_fitted"] = holdout_prediction
        holdout_metric = probability_metrics(
            oof.loc[holdout_mask, TARGET_COL], holdout_prediction
        )
        meta_rows.append(
            {
                "holdout_season": holdout_season,
                "selected_method": choice["method"],
                "selected_lgbm_weight": choice["lgbm_weight"],
                "selected_catboost_weight": 1.0 - choice["lgbm_weight"],
                "meta_train_brier": choice["brier"],
                "holdout_brier": holdout_metric["brier"],
                "holdout_bss_raw": holdout_metric["bss_raw"],
            }
        )
    if oof["blend_cross_fitted"].isna().any():
        raise RuntimeError("cross-fitted blend predictions are incomplete")
    meta_weights = pd.DataFrame(meta_rows)

    prediction_columns = {
        "lightgbm": "lgbm_prediction",
        "catboost": "catboost_prediction",
        "probability_50_50": "blend_probability_50_50",
        "selected_recent_oof": "blend_selected_recent_oof",
        "cross_fitted_blend": "blend_cross_fitted",
    }
    fold_metrics = make_metric_rows(
        oof, prediction_columns, valid_seasons, selection_seasons
    )

    correlations = []
    for label, frame in [("pooled", oof)]:
        correlations.append(
            {
                "partition": label,
                "valid_season": None,
                "pearson": float(frame[["lgbm_prediction", "catboost_prediction"]].corr().iloc[0, 1]),
            }
        )
    for season in valid_seasons:
        frame = oof[oof["valid_season"].eq(season)]
        correlations.append(
            {
                "partition": "fold",
                "valid_season": season,
                "pearson": float(frame[["lgbm_prediction", "catboost_prediction"]].corr().iloc[0, 1]),
            }
        )
    correlations = pd.DataFrame(correlations)

    group_parts = []
    for valid_season in valid_seasons:
        train = training_frame(data, valid_season)
        valid = data[data["season"].eq(valid_season)].copy()
        fold_oof = oof[oof["valid_season"].eq(valid_season)]
        if not np.array_equal(
            valid[ID_COL].astype(str).to_numpy(), fold_oof[ID_COL].astype(str).to_numpy()
        ):
            raise RuntimeError(f"OOF row order mismatch for grouped metrics: {valid_season}")
        for model_name, prediction_column in prediction_columns.items():
            group = grouped_metrics(
                train, valid, fold_oof[prediction_column].to_numpy()
            )
            group["valid_season"] = valid_season
            group["model"] = model_name
            group_parts.append(group)
        del train, valid, fold_oof
    group_metrics = pd.concat(group_parts, ignore_index=True)

    model_summary = fold_metrics[
        fold_metrics["partition"].eq("recent_pooled")
        & fold_metrics["scope"].eq("all")
    ].sort_values("brier")
    recent_mask = oof["valid_season"].isin(selection_seasons)
    summary = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "signature": signature,
        "configuration": asdict(configuration),
        "valid_seasons": valid_seasons,
        "selection_seasons": selection_seasons,
        "selected_recent_oof_method": selected_method,
        "selected_recent_oof_lgbm_weight": selected_weight,
        "selected_recent_oof_catboost_weight": 1.0 - selected_weight,
        "selected_recent_oof_brier": float(selected["selection_brier"]),
        "selected_recent_oof_bss_raw": float(selected["selection_bss_raw"]),
        "cross_fitted_blend_brier": float(
            probability_metrics(
                oof.loc[recent_mask, TARGET_COL],
                oof.loc[recent_mask, "blend_cross_fitted"],
            )["brier"]
        ),
        "pooled_prediction_correlation": float(
            correlations.loc[correlations["partition"].eq("pooled"), "pearson"].iloc[0]
        ),
        "catboost_task_type": catboost_task_type,
    }

    pd.DataFrame(training_metadata).to_csv(run_dir / "training_metadata.csv", index=False)
    blend_grid.to_csv(run_dir / "blend_grid.csv", index=False)
    meta_weights.to_csv(run_dir / "cross_fitted_blend_weights.csv", index=False)
    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    model_summary.to_csv(run_dir / "model_summary.csv", index=False)
    correlations.to_csv(run_dir / "prediction_correlations.csv", index=False)
    group_metrics.to_csv(run_dir / "group_metrics.csv", index=False)
    oof.to_csv(run_dir / "oof_predictions.csv.gz", index=False, compression="gzip")
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (args.output_dir / "latest_run.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"signature": signature, "run_dir": str(run_dir), "summary": summary},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nRecent-season comparison: {selection_seasons}")
    print(model_summary[["model", "brier", "bss_raw", "mean_gap", "roc_auc"]].to_string(index=False))
    print("\nCross-fitted blend weights")
    print(meta_weights.to_string(index=False))
    print("\nPrediction correlations")
    print(correlations.to_string(index=False))
    print(f"\nSaved results to {run_dir}")


if __name__ == "__main__":
    main()
