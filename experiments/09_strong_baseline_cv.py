#!/usr/bin/env python3
"""Build a strong, leakage-safe LightGBM baseline with forward CV.

The experiment is intentionally independent of outputs from notebooks 05-08.
It measures cumulative changes in a fixed order:

1. remove ``season`` and ``game_type`` from a single LightGBM;
2. repair pre-2023 F-game contamination in cumulative as-of rates;
3. add recent-season sample weighting;
4. apply a train-only prevalence-trend logit shift;
5. average four LightGBM structures over multiple seeds.

Every fitted artifact uses seasons strictly earlier than the validation season.
Validation rows are never aggregated during feature generation or inference.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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

from evaluation import probability_metrics
from f_contamination import (
    apply_f_history_correction,
    artifact_summary,
    fit_f_contamination,
)
from features import (
    FeatureConfig,
    ID_COL,
    TARGET_COL,
    add_rowwise_features,
    encode_categories_train_only,
)


RESULT_SCHEMA_VERSION = 1
VALID_SEASONS = (2023, 2024)
RECENCY_HALF_LIFE = 0.5
PROBABILITY_EPS = 1e-7
DEFAULT_SEEDS = (42, 7, 2024, 1234, 31337)
DEFAULT_VARIANTS = ("base", "wide", "extra", "lowff")

VARIANT_OVERRIDES: dict[str, dict] = {
    "base": {},
    "wide": {"num_leaves": 127, "min_child_samples": 8_000},
    "extra": {"extra_trees": True, "reg_lambda": 10.0},
    "lowff": {"colsample_bytree": 0.45, "subsample": 0.60},
}

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

STAGE_ORDER = (
    "raw_single",
    "f_corrected_single",
    "f_corrected_recency_single",
    "f_corrected_recency_prior_single",
    "ensemble_probability",
    "ensemble_logit",
    "ensemble_logit_prior",
)

STAGE_DESCRIPTIONS = {
    "raw_single": "No season/game_type; one base LightGBM; uniform rows",
    "f_corrected_single": "Raw single + train-only pre-2023 F history repair",
    "f_corrected_recency_single": "F repair + half-life recency weights; one member",
    "f_corrected_recency_prior_single": "Recency single + train-trend logit prior shift",
    "ensemble_probability": "Four structures x seeds; probability mean",
    "ensemble_logit": "Four structures x seeds; logit mean",
    "ensemble_logit_prior": "Logit ensemble + train-trend logit prior shift",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path", type=Path, default=PROJECT_ROOT / "data/train.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/strong_baseline_cv_v1",
    )
    parser.add_argument(
        "--valid-seasons", type=int, nargs="+", default=list(VALID_SEASONS)
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--n-estimators", type=int, default=320)
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument("--half-life", type=float, default=RECENCY_HALF_LIFE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANT_OVERRIDES),
        default=list(DEFAULT_VARIANTS),
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


def validate_training_data(data: pd.DataFrame, valid_seasons: list[int]) -> None:
    required = {
        ID_COL,
        TARGET_COL,
        "season",
        "game_type",
        "pitcher_id",
        "batter_id",
        "asof_pitcher_n",
        "asof_batter_n",
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_batter_success_rate",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"training columns are missing: {missing}")
    if data[ID_COL].isna().any() or data[ID_COL].astype("string").duplicated().any():
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


def recency_weights(
    seasons: pd.Series,
    *,
    reference_season: int,
    half_life: float,
) -> np.ndarray:
    if half_life <= 0.0:
        raise ValueError("half_life must be positive")
    values = seasons.to_numpy(dtype=np.float64)
    age = (reference_season - 1) - values
    if (age < 0).any():
        raise ValueError("training data contains a future season")
    weights = np.power(0.5, age / half_life)
    return weights / weights.mean()


def fit_linear_target_prior(
    train: pd.DataFrame,
    *,
    forecast_season: int,
) -> tuple[float, dict]:
    """Extrapolate one global target rate using training seasons only."""

    table = (
        train.groupby("season", observed=True)[TARGET_COL]
        .agg(n="size", target_rate="mean")
        .reset_index()
        .sort_values("season")
    )
    if len(table) < 2:
        prediction = float(train[TARGET_COL].mean())
        slope = 0.0
        intercept = prediction
    else:
        x = table["season"].to_numpy(dtype=np.float64)
        y = table["target_rate"].to_numpy(dtype=np.float64)
        weight = np.sqrt(table["n"].to_numpy(dtype=np.float64))
        slope, intercept = np.polyfit(x, y, deg=1, w=weight)
        prediction = float(intercept + slope * forecast_season)
    clipped = float(np.clip(prediction, 0.35, 0.65))
    return clipped, {
        "forecast_season": int(forecast_season),
        "seasons": table["season"].astype(int).tolist(),
        "rows": table["n"].astype(int).tolist(),
        "target_rates": table["target_rate"].astype(float).tolist(),
        "slope": float(slope),
        "intercept": float(intercept),
        "unclipped_forecast": float(prediction),
        "forecast": clipped,
    }


def logit(probability: np.ndarray | float) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)
    return np.log(value / (1.0 - value))


def sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def apply_prior_shift(
    prediction: np.ndarray,
    *,
    source_prior: float,
    target_prior: float,
) -> np.ndarray:
    return np.clip(
        sigmoid(logit(prediction) + logit(target_prior) - logit(source_prior)),
        PROBABILITY_EPS,
        1.0 - PROBABILITY_EPS,
    )


def build_feature_pair(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    target_prior: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    x_train = add_rowwise_features(
        train, target_prior=target_prior, config=FEATURE_CONFIG
    )
    x_valid = add_rowwise_features(
        valid, target_prior=target_prior, config=FEATURE_CONFIG
    )
    x_train, x_valid, categorical, _ = encode_categories_train_only(
        x_train, x_valid
    )
    if list(x_train.columns) != list(x_valid.columns):
        raise RuntimeError("train and validation feature columns do not match")
    forbidden = sorted({"season", "game_type"} & set(x_train.columns))
    if forbidden:
        raise RuntimeError(f"forbidden feature columns remain: {forbidden}")
    for name, frame in (("train", x_train), ("valid", x_valid)):
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        if np.isinf(numeric).any():
            raise RuntimeError(f"infinite values found in {name} features")
    return x_train, x_valid, categorical


def model_parameters(
    args: argparse.Namespace,
    *,
    variant: str,
    seed: int,
) -> dict:
    parameters = {
        "objective": "binary",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": 31,
        "min_child_samples": 2_000,
        "colsample_bytree": 0.80,
        "subsample": 0.80,
        "subsample_freq": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 30.0,
        "max_bin": 127,
        "random_state": int(seed),
        "n_jobs": args.threads,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    parameters.update(VARIANT_OVERRIDES[variant])
    return parameters


def member_cache_paths(
    run_dir: Path,
    *,
    valid_season: int,
    member_name: str,
) -> tuple[Path, Path, Path]:
    directory = run_dir / "cache" / f"fold_{valid_season}"
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / member_name
    return (
        stem.with_suffix(".csv.gz"),
        stem.with_suffix(".json"),
        stem.with_name(stem.name + "_importance.csv"),
    )


def train_or_load_member(
    *,
    run_dir: Path,
    valid_season: int,
    member_name: str,
    valid_ids: pd.Series,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    categorical: list[str],
    weights: np.ndarray,
    parameters: dict,
    force_rerun: bool,
) -> tuple[np.ndarray, dict, pd.DataFrame]:
    prediction_path, metadata_path, importance_path = member_cache_paths(
        run_dir, valid_season=valid_season, member_name=member_name
    )
    if (
        not force_rerun
        and prediction_path.exists()
        and metadata_path.exists()
        and importance_path.exists()
    ):
        cached = pd.read_csv(prediction_path, dtype={ID_COL: "string"})
        expected = valid_ids.astype("string").to_numpy()
        if not np.array_equal(cached[ID_COL].to_numpy(), expected):
            raise RuntimeError(f"cached row order mismatch: {prediction_path}")
        prediction = cached["prediction"].to_numpy(dtype=np.float64)
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        importance = pd.read_csv(importance_path)
        print(f"verified cache: fold={valid_season} member={member_name}")
        return prediction, metadata, importance

    model = lgb.LGBMClassifier(**parameters)
    started = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        sample_weight=weights,
        categorical_feature=categorical,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    elapsed = time.perf_counter() - started
    prediction = np.clip(
        model.predict_proba(x_valid)[:, 1],
        PROBABILITY_EPS,
        1.0 - PROBABILITY_EPS,
    )
    importance = pd.DataFrame(
        {
            "feature": x_train.columns,
            "importance": model.booster_.feature_importance("gain"),
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    metadata = {
        "valid_season": int(valid_season),
        "member": member_name,
        "train_rows": int(len(x_train)),
        "valid_rows": int(len(x_valid)),
        "feature_count": int(x_train.shape[1]),
        "categorical_count": int(len(categorical)),
        "train_seconds": float(elapsed),
        "parameters": parameters,
    }
    pd.DataFrame(
        {ID_COL: valid_ids.astype("string"), "prediction": prediction}
    ).to_csv(prediction_path, index=False, compression="gzip")
    write_json(metadata_path, metadata)
    importance.to_csv(importance_path, index=False)
    del model
    gc.collect()
    return prediction, metadata, importance


def add_metric_identity(row: dict, **identity) -> dict:
    return {**identity, **row}


def stage_metric_rows(
    fold_oof: pd.DataFrame,
    *,
    valid_season: int,
) -> list[dict]:
    rows = []
    target = fold_oof[TARGET_COL].to_numpy(dtype=np.int8)
    for stage in STAGE_ORDER:
        rows.append(
            add_metric_identity(
                probability_metrics(target, fold_oof[stage].to_numpy(dtype=np.float64)),
                stage=stage,
                partition="fold",
                valid_season=int(valid_season),
                scope="all",
            )
        )
        for game_type in ("R", "F"):
            mask = fold_oof["game_type"].eq(game_type).to_numpy()
            if mask.any():
                rows.append(
                    add_metric_identity(
                        probability_metrics(target[mask], fold_oof.loc[mask, stage]),
                        stage=stage,
                        partition="fold",
                        valid_season=int(valid_season),
                        scope=game_type,
                    )
                )
    return rows


def segment_metric_rows(
    fold_oof: pd.DataFrame,
    *,
    valid_season: int,
) -> list[dict]:
    rows = []
    target = fold_oof[TARGET_COL].to_numpy(dtype=np.int8)
    segment_sets = {
        "game_type": fold_oof["game_type"].astype("string"),
        "count_state": (
            fold_oof["balls_before"].astype(str)
            + "-"
            + fold_oof["strikes_before"].astype(str)
        ),
        "hand_matchup": (
            fold_oof["pitcher_hand"].astype(str)
            + "-"
            + fold_oof["batter_hand"].astype(str)
        ),
    }
    for stage in STAGE_ORDER:
        prediction = fold_oof[stage].to_numpy(dtype=np.float64)
        for group, values in segment_sets.items():
            for value in sorted(values.unique()):
                mask = values.eq(value).to_numpy()
                if mask.sum() < 20:
                    continue
                rows.append(
                    add_metric_identity(
                        probability_metrics(target[mask], prediction[mask]),
                        stage=stage,
                        valid_season=int(valid_season),
                        group=group,
                        value=str(value),
                    )
                )
    return rows


def member_metric_rows(
    fold_oof: pd.DataFrame,
    member_columns: list[str],
    *,
    valid_season: int,
) -> list[dict]:
    target = fold_oof[TARGET_COL].to_numpy(dtype=np.int8)
    return [
        add_metric_identity(
            probability_metrics(target, fold_oof[column].to_numpy(dtype=np.float64)),
            member=column.removeprefix("member__"),
            valid_season=int(valid_season),
        )
        for column in member_columns
    ]


def correction_row_independence_audit(
    valid: pd.DataFrame,
    artifact,
) -> dict:
    sample = valid.sample(n=min(24, len(valid)), random_state=2026).copy()
    columns = [
        ID_COL,
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_batter_success_rate",
    ]

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        corrected, _ = apply_f_history_correction(frame, artifact)
        return corrected.loc[:, columns].set_index(ID_COL).sort_index()

    batched = transform(sample)
    shuffled = transform(sample.sample(frac=1.0, random_state=42))
    singles = pd.concat(
        [transform(sample.iloc[[index]]) for index in range(len(sample))]
    ).sort_index()
    pd.testing.assert_frame_equal(
        batched, shuffled, check_dtype=False, rtol=1e-12, atol=1e-12
    )
    pd.testing.assert_frame_equal(
        batched, singles, check_dtype=False, rtol=1e-12, atol=1e-12
    )
    return {
        "status": "passed",
        "sample_rows": int(len(sample)),
        "shuffle_invariant": True,
        "single_row_invariant": True,
    }


def build_stage_summary(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target = oof[TARGET_COL].to_numpy(dtype=np.int8)
    for stage in STAGE_ORDER:
        pooled = probability_metrics(target, oof[stage].to_numpy(dtype=np.float64))
        row = {
            "stage": stage,
            "description": STAGE_DESCRIPTIONS[stage],
            "pooled_brier": pooled["brier"],
            "pooled_bss_raw": pooled["bss_raw"],
            "pooled_roc_auc": pooled["roc_auc"],
            "pooled_mean_gap": pooled["mean_gap"],
        }
        for season in VALID_SEASONS:
            mask = oof["valid_season"].eq(season).to_numpy()
            metric = probability_metrics(target[mask], oof.loc[mask, stage])
            row[f"brier_{season}"] = metric["brier"]
            row[f"bss_raw_{season}"] = metric["bss_raw"]
            row[f"mean_gap_{season}"] = metric["mean_gap"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    anchor = summary[summary["stage"].eq("raw_single")].iloc[0]
    previous = None
    for index, row in summary.iterrows():
        summary.loc[index, "delta_pooled_brier_vs_raw"] = (
            row["pooled_brier"] - anchor["pooled_brier"]
        )
        summary.loc[index, "delta_pooled_bss_vs_raw"] = (
            row["pooled_bss_raw"] - anchor["pooled_bss_raw"]
        )
        for season in VALID_SEASONS:
            summary.loc[index, f"delta_brier_{season}_vs_raw"] = (
                row[f"brier_{season}"] - anchor[f"brier_{season}"]
            )
        if previous is None:
            summary.loc[index, "delta_pooled_brier_vs_previous"] = 0.0
        else:
            summary.loc[index, "delta_pooled_brier_vs_previous"] = (
                row["pooled_brier"] - previous["pooled_brier"]
            )
        previous = row
    summary["improves_raw_pooled"] = summary["delta_pooled_brier_vs_raw"].le(0.0)
    summary["safe_both_folds_vs_raw"] = np.logical_and.reduce(
        [summary[f"delta_brier_{season}_vs_raw"].le(0.0) for season in VALID_SEASONS]
    )
    summary["passes_anchor_gate"] = (
        summary["improves_raw_pooled"] & summary["safe_both_folds_vs_raw"]
    )
    return summary


def main() -> None:
    args = parse_args()
    valid_seasons = sorted(set(args.valid_seasons))
    seeds = list(dict.fromkeys(args.seeds))
    variants = list(dict.fromkeys(args.variants))
    if not seeds or not variants:
        raise ValueError("at least one seed and variant are required")
    validate_variant = sorted(set(variants) - set(VARIANT_OVERRIDES))
    if validate_variant:
        raise ValueError(f"unknown variants: {validate_variant}")

    source_hashes = {
        "experiment": sha256_file(Path(__file__)),
        "features": sha256_file(SRC_DIR / "features.py"),
        "f_contamination": sha256_file(SRC_DIR / "f_contamination.py"),
        "evaluation": sha256_file(SRC_DIR / "evaluation.py"),
        "metrics": sha256_file(SRC_DIR / "metrics.py"),
        "train_data": sha256_file(args.train_path),
    }
    signature_payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "valid_seasons": valid_seasons,
        "half_life": args.half_life,
        "seeds": seeds,
        "variants": variants,
        "variant_overrides": {name: VARIANT_OVERRIDES[name] for name in variants},
        "feature_config": asdict(FEATURE_CONFIG),
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "threads": args.threads,
        "stage_order": list(STAGE_ORDER),
        "source_hashes": source_hashes,
    }
    signature = experiment_signature(signature_payload)
    run_dir = args.output_dir / f"run_{signature}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "configuration.json",
        {"signature": signature, **signature_payload},
    )

    print(f"Loading training data: {args.train_path}")
    data = pd.read_csv(args.train_path)
    data[ID_COL] = data[ID_COL].astype("string")
    validate_training_data(data, valid_seasons)

    oof_parts: list[pd.DataFrame] = []
    stage_metrics: list[dict] = []
    segment_metrics: list[dict] = []
    member_metrics: list[dict] = []
    member_metadata: list[dict] = []
    importance_parts: list[pd.DataFrame] = []
    contamination_audit: dict[str, dict] = {}
    prior_audit: dict[str, dict] = {}
    integrity_folds: dict[str, dict] = {}

    for valid_season in valid_seasons:
        print(f"\n{'=' * 88}\nForward fold {valid_season}\n{'=' * 88}")
        train = data[data["season"].lt(valid_season)].copy()
        valid = data[data["season"].eq(valid_season)].copy()
        if train.empty or valid.empty or int(train["season"].max()) >= valid_season:
            raise RuntimeError(f"invalid forward split for {valid_season}")
        train_prior = float(train[TARGET_COL].mean())
        uniform_weights = np.ones(len(train), dtype=np.float64)
        recent_weights = recency_weights(
            train["season"],
            reference_season=valid_season,
            half_life=args.half_life,
        )
        weighted_prior = float(
            np.average(train[TARGET_COL].to_numpy(dtype=np.float64), weights=recent_weights)
        )
        trend_prior, trend_audit = fit_linear_target_prior(
            train, forecast_season=valid_season
        )
        trend_audit.update(
            {
                "unweighted_train_prior": train_prior,
                "recency_weighted_train_prior": weighted_prior,
                "validation_target_rate_diagnostic_only": float(valid[TARGET_COL].mean()),
            }
        )
        prior_audit[str(valid_season)] = trend_audit

        artifact = fit_f_contamination(train)
        corrected_train, correction_train_audit = apply_f_history_correction(
            train, artifact
        )
        corrected_valid, correction_valid_audit = apply_f_history_correction(
            valid, artifact
        )
        independence = correction_row_independence_audit(valid, artifact)
        contamination_audit[str(valid_season)] = {
            "artifact": artifact_summary(artifact),
            "train_transform": correction_train_audit,
            "valid_transform": correction_valid_audit,
            "row_independence": independence,
        }

        y_train = train[TARGET_COL].to_numpy(dtype=np.int8)
        valid_ids = valid[ID_COL]

        print("Building raw features...")
        x_train_raw, x_valid_raw, categorical_raw = build_feature_pair(
            train, valid, target_prior=train_prior
        )
        raw_params = model_parameters(
            args, variant="base", seed=seeds[0]
        )
        raw_prediction, raw_metadata, raw_importance = train_or_load_member(
            run_dir=run_dir,
            valid_season=valid_season,
            member_name="raw_uniform__base_s" + str(seeds[0]),
            valid_ids=valid_ids,
            x_train=x_train_raw,
            y_train=y_train,
            x_valid=x_valid_raw,
            categorical=categorical_raw,
            weights=uniform_weights,
            parameters=raw_params,
            force_rerun=args.force_rerun,
        )
        raw_importance["member"] = "raw_uniform__base_s" + str(seeds[0])
        raw_importance["valid_season"] = valid_season
        importance_parts.append(raw_importance)
        member_metadata.append(raw_metadata)
        del x_train_raw, x_valid_raw, categorical_raw
        gc.collect()

        print("Building F-corrected features...")
        x_train, x_valid, categorical = build_feature_pair(
            corrected_train, corrected_valid, target_prior=train_prior
        )
        corrected_uniform, corrected_metadata, corrected_importance = train_or_load_member(
            run_dir=run_dir,
            valid_season=valid_season,
            member_name="corrected_uniform__base_s" + str(seeds[0]),
            valid_ids=valid_ids,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            categorical=categorical,
            weights=uniform_weights,
            parameters=raw_params,
            force_rerun=args.force_rerun,
        )
        corrected_importance["member"] = "corrected_uniform__base_s" + str(seeds[0])
        corrected_importance["valid_season"] = valid_season
        importance_parts.append(corrected_importance)
        member_metadata.append(corrected_metadata)

        ensemble_predictions: dict[str, np.ndarray] = {}
        for variant in variants:
            for seed in seeds:
                short_name = f"{variant}_s{seed}"
                cache_name = "corrected_recency__" + short_name
                prediction, metadata, importance = train_or_load_member(
                    run_dir=run_dir,
                    valid_season=valid_season,
                    member_name=cache_name,
                    valid_ids=valid_ids,
                    x_train=x_train,
                    y_train=y_train,
                    x_valid=x_valid,
                    categorical=categorical,
                    weights=recent_weights,
                    parameters=model_parameters(args, variant=variant, seed=seed),
                    force_rerun=args.force_rerun,
                )
                ensemble_predictions[short_name] = prediction
                importance["member"] = cache_name
                importance["valid_season"] = valid_season
                importance_parts.append(importance)
                member_metadata.append(metadata)

        base_member_name = f"base_s{seeds[0]}"
        if base_member_name not in ensemble_predictions:
            raise RuntimeError("the first base member is required for cumulative stages")
        member_matrix = np.column_stack(list(ensemble_predictions.values()))
        probability_mean = np.mean(member_matrix, axis=1)
        logit_mean = sigmoid(np.mean(logit(member_matrix), axis=1))

        fold_oof = valid[
            [
                ID_COL,
                TARGET_COL,
                "season",
                "game_type",
                "balls_before",
                "strikes_before",
                "pitcher_hand",
                "batter_hand",
            ]
        ].copy()
        fold_oof["valid_season"] = valid_season
        fold_oof["raw_single"] = raw_prediction
        fold_oof["f_corrected_single"] = corrected_uniform
        fold_oof["f_corrected_recency_single"] = ensemble_predictions[base_member_name]
        fold_oof["f_corrected_recency_prior_single"] = apply_prior_shift(
            ensemble_predictions[base_member_name],
            source_prior=weighted_prior,
            target_prior=trend_prior,
        )
        fold_oof["ensemble_probability"] = probability_mean
        fold_oof["ensemble_logit"] = logit_mean
        fold_oof["ensemble_logit_prior"] = apply_prior_shift(
            logit_mean,
            source_prior=weighted_prior,
            target_prior=trend_prior,
        )
        member_columns = []
        for member_name, prediction in ensemble_predictions.items():
            column = "member__" + member_name
            fold_oof[column] = prediction
            member_columns.append(column)

        stage_metrics.extend(stage_metric_rows(fold_oof, valid_season=valid_season))
        segment_metrics.extend(segment_metric_rows(fold_oof, valid_season=valid_season))
        member_metrics.extend(
            member_metric_rows(
                fold_oof, member_columns, valid_season=valid_season
            )
        )
        integrity_folds[str(valid_season)] = {
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "max_train_season": int(train["season"].max()),
            "valid_season": int(valid_season),
            "feature_count": int(x_train.shape[1]),
            "categorical_count": int(len(categorical)),
            "season_feature_excluded": "season" not in x_train.columns,
            "game_type_feature_excluded": "game_type" not in x_train.columns,
            "member_count": int(len(member_columns)),
            "all_probabilities_finite": bool(
                np.isfinite(fold_oof[list(STAGE_ORDER) + member_columns].to_numpy()).all()
            ),
        }
        oof_parts.append(fold_oof)
        del (
            train,
            valid,
            corrected_train,
            corrected_valid,
            x_train,
            x_valid,
            categorical,
            y_train,
            member_matrix,
            ensemble_predictions,
            fold_oof,
        )
        gc.collect()

    oof = pd.concat(oof_parts, ignore_index=True)
    if oof[ID_COL].duplicated().any():
        raise RuntimeError("OOF row_id values are not unique")
    expected = data[data["season"].isin(valid_seasons)][ID_COL].astype("string")
    if set(oof[ID_COL].astype("string")) != set(expected):
        raise RuntimeError("OOF does not cover every validation-season row")

    stage_summary = build_stage_summary(oof)
    safe = stage_summary[stage_summary["passes_anchor_gate"]].sort_values(
        ["pooled_brier", "brier_2024", "brier_2023"]
    )
    selected_stage = safe.iloc[0]["stage"] if len(safe) else "raw_single"
    selection_status = "safe_improvement" if selected_stage != "raw_single" else "anchor_retained"

    pooled_stage_rows = []
    for stage in STAGE_ORDER:
        pooled_stage_rows.append(
            add_metric_identity(
                probability_metrics(oof[TARGET_COL], oof[stage]),
                stage=stage,
                partition="pooled",
                valid_season=None,
                scope="all",
            )
        )
    stage_metrics.extend(pooled_stage_rows)

    member_columns = [c for c in oof.columns if c.startswith("member__")]
    correlation = oof[member_columns + ["ensemble_logit"]].corr()
    correlation_rows = []
    for column in member_columns:
        correlation_rows.append(
            {
                "member": column.removeprefix("member__"),
                "pearson_vs_logit_ensemble": float(correlation.loc[column, "ensemble_logit"]),
                "mean_abs_delta_vs_logit_ensemble": float(
                    np.mean(np.abs(oof[column] - oof["ensemble_logit"]))
                ),
            }
        )

    stage_summary.to_csv(run_dir / "stage_summary.csv", index=False)
    pd.DataFrame(stage_metrics).to_csv(run_dir / "stage_metrics.csv", index=False)
    pd.DataFrame(segment_metrics).to_csv(run_dir / "segment_metrics.csv", index=False)
    pd.DataFrame(member_metrics).to_csv(run_dir / "member_metrics.csv", index=False)
    pd.DataFrame(member_metadata).to_json(
        run_dir / "member_training.json", orient="records", indent=2
    )
    pd.concat(importance_parts, ignore_index=True).to_csv(
        run_dir / "feature_importance.csv", index=False
    )
    pd.DataFrame(correlation_rows).to_csv(
        run_dir / "member_correlations.csv", index=False
    )
    oof.to_csv(run_dir / "oof_predictions.csv.gz", index=False, compression="gzip")
    write_json(run_dir / "contamination_audit.json", contamination_audit)
    write_json(run_dir / "prior_audit.json", prior_audit)

    integrity_audit = {
        "status": "passed",
        "rows": int(len(oof)),
        "unique_row_ids": int(oof[ID_COL].nunique()),
        "valid_seasons": valid_seasons,
        "forward_split_only": True,
        "validation_or_test_aggregation": False,
        "feature_config": asdict(FEATURE_CONFIG),
        "folds": integrity_folds,
    }
    write_json(run_dir / "integrity_audit.json", integrity_audit)

    selected = stage_summary[stage_summary["stage"].eq(selected_stage)].iloc[0]
    summary = {
        "signature": signature,
        "run_dir": str(run_dir.resolve()),
        "selected_stage": selected_stage,
        "selection_status": selection_status,
        "selected_pooled_brier": float(selected["pooled_brier"]),
        "selected_pooled_bss_raw": float(selected["pooled_bss_raw"]),
        "selected_brier_2023": float(selected["brier_2023"]),
        "selected_brier_2024": float(selected["brier_2024"]),
        "raw_pooled_brier": float(
            stage_summary.loc[stage_summary["stage"].eq("raw_single"), "pooled_brier"].iloc[0]
        ),
        "member_count": int(len(member_columns)),
        "integrity_status": integrity_audit["status"],
    }
    write_json(run_dir / "summary.json", summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "latest_run.json",
        {"run_dir": str(run_dir.resolve()), "summary": summary},
    )
    print("\nStrong baseline CV complete")
    print(stage_summary.to_string(index=False))
    print(f"Selected stage: {selected_stage} ({selection_status})")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
