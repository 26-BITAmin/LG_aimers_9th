#!/usr/bin/env python3
"""Recombine feature-ablation OOF predictions without retraining models.

The script tests model-specific feature bundles, conservative blend weights,
rolling season selection, and train-OOF-only calibration.  Every prediction
used for a holdout season is produced by a model that was already trained only
on earlier seasons in experiment 05.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from evaluation import probability_metrics
from features import ID_COL, TARGET_COL


RESULT_SCHEMA_VERSION = 1
VALID_SEASONS = (2022, 2023, 2024)
RECENT_SEASONS = (2023, 2024)
LGBM_CONFIGS = ("base", "pitchmix_shape")
CATBOOST_CONFIGS = ("base", "rate_gaps", "all_new")
BLEND_METHODS = ("probability", "logit")
BLEND_WEIGHTS = tuple(float(value) for value in np.round(np.arange(0.20, 0.901, 0.05), 2))
FIXED_LGBM_WEIGHT = 0.65
BASE_PAIR = "lgb_base__cat_base"

KEY_COLUMNS = [ID_COL, TARGET_COL, "season", "game_type", "valid_season"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/feature_ablation_v1",
    )
    parser.add_argument("--source-run-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/oof_recombination_v1",
    )
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def experiment_signature(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def resolve_source_run(args: argparse.Namespace) -> Path:
    if args.source_run_dir is not None:
        run_dir = args.source_run_dir
    else:
        latest_path = args.feature_output_root / "latest_run.json"
        if not latest_path.exists():
            raise FileNotFoundError(
                f"05 latest_run.json is missing: {latest_path}"
            )
        with latest_path.open(encoding="utf-8") as handle:
            latest = json.load(handle)
        run_dir = Path(latest["run_dir"])
    if not run_dir.exists():
        raise FileNotFoundError(f"05 run directory is missing: {run_dir}")
    return run_dir


def validate_oof(frame: pd.DataFrame, path: Path) -> None:
    required = set(KEY_COLUMNS + ["lgbm_prediction", "catboost_prediction"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OOF columns are missing from {path}: {missing}")
    if frame[ID_COL].isna().any() or frame[ID_COL].astype("string").duplicated().any():
        raise ValueError(f"row_id must be non-null and unique: {path}")
    if not set(frame[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError(f"invalid target in {path}")
    if tuple(sorted(frame["valid_season"].unique())) != VALID_SEASONS:
        raise ValueError(f"unexpected validation seasons in {path}")
    for column in ("lgbm_prediction", "catboost_prediction"):
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or ((values <= 0.0) | (values >= 1.0)).any():
            raise ValueError(f"invalid probabilities in {path}: {column}")


def load_components(source_run_dir: Path) -> tuple[pd.DataFrame, dict]:
    required_configs = sorted(set(LGBM_CONFIGS + CATBOOST_CONFIGS))
    frames = {}
    hashes = {}
    for config_name in required_configs:
        path = source_run_dir / f"oof_{config_name}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(
                f"required 05 OOF is missing: {path}. Re-run notebook 05 first."
            )
        print(f"Loading {path.name} ...")
        frame = pd.read_csv(path, dtype={ID_COL: "string"})
        validate_oof(frame, path)
        frames[config_name] = frame
        hashes[config_name] = sha256_file(path)

    reference = frames["base"]
    components = reference[KEY_COLUMNS].copy()
    for config_name, frame in frames.items():
        for column in KEY_COLUMNS:
            left = reference[column].astype("string") if column in (ID_COL, "game_type") else reference[column]
            right = frame[column].astype("string") if column in (ID_COL, "game_type") else frame[column]
            if not np.array_equal(left.to_numpy(), right.to_numpy()):
                raise RuntimeError(
                    f"OOF alignment mismatch: config={config_name} column={column}"
                )
    for config_name in LGBM_CONFIGS:
        column = f"lgbm__{config_name}"
        components[column] = frames[config_name]["lgbm_prediction"].to_numpy(dtype=np.float64)
    for config_name in CATBOOST_CONFIGS:
        column = f"catboost__{config_name}"
        components[column] = frames[config_name]["catboost_prediction"].to_numpy(dtype=np.float64)

    if "blend_prediction" in reference.columns:
        recomputed = blend_predictions(
            components["lgbm__base"].to_numpy(),
            components["catboost__base"].to_numpy(),
            lgbm_weight=FIXED_LGBM_WEIGHT,
            method="logit",
        )
        max_difference = float(
            np.max(np.abs(recomputed - reference["blend_prediction"].to_numpy()))
        )
        if max_difference > 1e-12:
            raise RuntimeError(
                f"base 65:35 reconstruction mismatch: max_abs_diff={max_difference}"
            )
    else:
        max_difference = float("nan")

    metadata = {
        "oof_sha256": hashes,
        "rows": int(len(components)),
        "base_reconstruction_max_abs_difference": max_difference,
    }
    del frames
    return components, metadata


def pair_definitions() -> list[dict]:
    rows = []
    for lgbm_config in LGBM_CONFIGS:
        for catboost_config in CATBOOST_CONFIGS:
            rows.append(
                {
                    "pair": f"lgb_{lgbm_config}__cat_{catboost_config}",
                    "lgbm_config": lgbm_config,
                    "catboost_config": catboost_config,
                    "lgbm_column": f"lgbm__{lgbm_config}",
                    "catboost_column": f"catboost__{catboost_config}",
                }
            )
    return rows


def blend_predictions(
    lgbm_prediction: np.ndarray,
    catboost_prediction: np.ndarray,
    *,
    lgbm_weight: float,
    method: str,
) -> np.ndarray:
    lgbm_prediction = np.asarray(lgbm_prediction, dtype=np.float64)
    catboost_prediction = np.asarray(catboost_prediction, dtype=np.float64)
    if method == "probability":
        prediction = lgbm_weight * lgbm_prediction + (1.0 - lgbm_weight) * catboost_prediction
    elif method == "logit":
        eps = 1e-7
        p_lgbm = np.clip(lgbm_prediction, eps, 1.0 - eps)
        p_catboost = np.clip(catboost_prediction, eps, 1.0 - eps)
        lgbm_logit = np.log(p_lgbm / (1.0 - p_lgbm))
        catboost_logit = np.log(p_catboost / (1.0 - p_catboost))
        blended_logit = lgbm_weight * lgbm_logit + (1.0 - lgbm_weight) * catboost_logit
        prediction = 1.0 / (1.0 + np.exp(-np.clip(blended_logit, -30.0, 30.0)))
    else:
        raise ValueError(f"unknown blend method: {method}")
    return np.clip(prediction, 1e-7, 1.0 - 1e-7)


def prediction_for_definition(
    components: pd.DataFrame,
    definition: dict,
    *,
    method: str,
    lgbm_weight: float,
) -> np.ndarray:
    return blend_predictions(
        components[definition["lgbm_column"]].to_numpy(dtype=np.float64),
        components[definition["catboost_column"]].to_numpy(dtype=np.float64),
        lgbm_weight=lgbm_weight,
        method=method,
    )


def scopes(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(frame), dtype=bool),
        "R": frame["game_type"].eq("R").to_numpy(),
        "F": frame["game_type"].eq("F").to_numpy(),
    }


def fixed_metric_rows(
    components: pd.DataFrame,
    definitions: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    candidate_oof = components[KEY_COLUMNS].copy()
    partitions: list[tuple[str, int | None, np.ndarray]] = [
        ("pooled", None, np.ones(len(components), dtype=bool)),
        (
            "recent_pooled",
            None,
            components["valid_season"].isin(RECENT_SEASONS).to_numpy(),
        ),
    ]
    partitions.extend(
        (
            "fold",
            season,
            components["valid_season"].eq(season).to_numpy(),
        )
        for season in VALID_SEASONS
    )
    for definition in definitions:
        prediction = prediction_for_definition(
            components,
            definition,
            method="logit",
            lgbm_weight=FIXED_LGBM_WEIGHT,
        )
        prediction_column = f"fixed__{definition['pair']}"
        candidate_oof[prediction_column] = prediction
        for partition, season, partition_mask in partitions:
            partition_frame = components.loc[partition_mask]
            partition_prediction = prediction[partition_mask]
            for scope, scope_mask in scopes(partition_frame).items():
                if not scope_mask.any():
                    continue
                metric = probability_metrics(
                    partition_frame.loc[scope_mask, TARGET_COL],
                    partition_prediction[scope_mask],
                )
                metric.update(
                    {
                        "pair": definition["pair"],
                        "lgbm_config": definition["lgbm_config"],
                        "catboost_config": definition["catboost_config"],
                        "method": "logit",
                        "lgbm_weight": FIXED_LGBM_WEIGHT,
                        "partition": partition,
                        "valid_season": season,
                        "scope": scope,
                    }
                )
                rows.append(metric)
    return pd.DataFrame(rows), candidate_oof


def fast_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    brier = float(np.mean(np.square(p - y)))
    rate = float(y.mean())
    reference = rate * (1.0 - rate)
    return {
        "n": int(len(y)),
        "brier": brier,
        "bss_raw": float(100_000.0 * (1.0 - brier / reference)),
        "target_rate": rate,
        "prediction_mean": float(p.mean()),
        "mean_gap": float(p.mean() - rate),
    }


def blend_grid_rows(
    components: pd.DataFrame,
    definitions: list[dict],
) -> pd.DataFrame:
    rows = []
    y = components[TARGET_COL].to_numpy(dtype=np.float64)
    fold_masks = {
        season: components["valid_season"].eq(season).to_numpy()
        for season in VALID_SEASONS
    }
    type_masks = {
        scope: mask for scope, mask in scopes(components).items() if scope != "all"
    }
    recent_mask = components["valid_season"].isin(RECENT_SEASONS).to_numpy()
    for definition in definitions:
        lgbm_prediction = components[definition["lgbm_column"]].to_numpy(dtype=np.float64)
        catboost_prediction = components[definition["catboost_column"]].to_numpy(dtype=np.float64)
        for method in BLEND_METHODS:
            for weight in BLEND_WEIGHTS:
                prediction = blend_predictions(
                    lgbm_prediction,
                    catboost_prediction,
                    lgbm_weight=weight,
                    method=method,
                )
                candidate_id = f"{definition['pair']}__{method}__w{weight:.2f}"
                metric = fast_metrics(y[recent_mask], prediction[recent_mask])
                rows.append(
                    {
                        **definition,
                        "candidate_id": candidate_id,
                        "method": method,
                        "lgbm_weight": weight,
                        "partition": "recent_pooled",
                        "valid_season": None,
                        "scope": "all",
                        **metric,
                    }
                )
                for season, fold_mask in fold_masks.items():
                    for scope in ("all", "R", "F"):
                        mask = fold_mask if scope == "all" else fold_mask & type_masks[scope]
                        metric = fast_metrics(y[mask], prediction[mask])
                        rows.append(
                            {
                                **definition,
                                "candidate_id": candidate_id,
                                "method": method,
                                "lgbm_weight": weight,
                                "partition": "fold",
                                "valid_season": season,
                                "scope": scope,
                                **metric,
                            }
                        )
    return pd.DataFrame(rows)


def summarize_fixed_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    all_scope = metrics[metrics["scope"].eq("all")]
    rows = []
    for pair, frame in all_scope.groupby("pair", sort=False):
        definition = frame.iloc[0]

        def metric(partition: str, season: int | None = None) -> pd.Series:
            mask = frame["partition"].eq(partition)
            if season is not None:
                mask &= frame["valid_season"].eq(season)
            selected = frame.loc[mask]
            if len(selected) != 1:
                raise RuntimeError(
                    f"expected one fixed metric row: pair={pair} partition={partition} season={season}"
                )
            return selected.iloc[0]

        recent = metric("recent_pooled")
        folds = {season: metric("fold", season) for season in VALID_SEASONS}
        rows.append(
            {
                "pair": pair,
                "lgbm_config": definition["lgbm_config"],
                "catboost_config": definition["catboost_config"],
                "recent_brier": recent["brier"],
                "recent_bss_raw": recent["bss_raw"],
                "recent_mean_gap": recent["mean_gap"],
                "recent_roc_auc": recent["roc_auc"],
                **{f"brier_{season}": folds[season]["brier"] for season in VALID_SEASONS},
                **{f"bss_raw_{season}": folds[season]["bss_raw"] for season in VALID_SEASONS},
            }
        )
    summary = pd.DataFrame(rows)
    base = summary[summary["pair"].eq(BASE_PAIR)].iloc[0]
    for column in ("recent_brier", "brier_2022", "brier_2023", "brier_2024"):
        summary[f"delta_{column}_vs_base"] = summary[column] - base[column]
    summary["delta_recent_bss_raw_vs_base"] = summary["recent_bss_raw"] - base["recent_bss_raw"]
    summary["improves_recent"] = summary["delta_recent_brier_vs_base"].lt(0.0)
    summary["safe_2023_2024"] = (
        summary["delta_brier_2023_vs_base"].le(0.0)
        & summary["delta_brier_2024_vs_base"].le(0.0)
    )
    summary["worst_recent_fold_brier"] = summary[["brier_2023", "brier_2024"]].max(axis=1)
    return summary.sort_values(
        ["recent_brier", "worst_recent_fold_brier"], ignore_index=True
    )


def grid_objective(
    grid: pd.DataFrame,
    *,
    selection_seasons: list[int],
    scope: str,
) -> pd.DataFrame:
    selected = grid[
        grid["partition"].eq("fold")
        & grid["valid_season"].isin(selection_seasons)
        & grid["scope"].eq(scope)
    ].copy()
    if selected.empty:
        raise RuntimeError(
            f"grid selection frame is empty: seasons={selection_seasons} scope={scope}"
        )
    selected["weighted_brier"] = selected["brier"] * selected["n"]
    objective = (
        selected.groupby(
            [
                "candidate_id",
                "pair",
                "lgbm_config",
                "catboost_config",
                "lgbm_column",
                "catboost_column",
                "method",
                "lgbm_weight",
            ],
            as_index=False,
        )
        .agg(
            selection_n=("n", "sum"),
            weighted_brier_sum=("weighted_brier", "sum"),
            latest_selection_brier=("brier", "last"),
        )
    )
    objective["selection_brier"] = (
        objective["weighted_brier_sum"] / objective["selection_n"]
    )
    objective["weight_distance_from_065"] = (
        objective["lgbm_weight"] - FIXED_LGBM_WEIGHT
    ).abs()
    objective["method_order"] = objective["method"].map({"logit": 0, "probability": 1})
    return objective.sort_values(
        [
            "selection_brier",
            "latest_selection_brier",
            "weight_distance_from_065",
            "method_order",
        ],
        ignore_index=True,
    )


def definition_lookup(definitions: list[dict]) -> dict[str, dict]:
    return {definition["pair"]: definition for definition in definitions}


def rolling_selection_predictions(
    components: pd.DataFrame,
    definitions: list[dict],
    grid: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    lookup = definition_lookup(definitions)
    predictions = {
        policy: np.full(len(components), np.nan, dtype=np.float64)
        for policy in (
            "global_previous",
            "global_expanding",
            "expert_previous",
            "expert_expanding",
        )
    }
    selection_rows = []
    for holdout_season in RECENT_SEASONS:
        holdout_mask = components["valid_season"].eq(holdout_season).to_numpy()
        for history_policy in ("previous", "expanding"):
            selection_seasons = (
                [holdout_season - 1]
                if history_policy == "previous"
                else [season for season in VALID_SEASONS if season < holdout_season]
            )
            objective = grid_objective(
                grid, selection_seasons=selection_seasons, scope="all"
            )
            choice = objective.iloc[0]
            definition = lookup[str(choice["pair"])]
            full_prediction = prediction_for_definition(
                components,
                definition,
                method=str(choice["method"]),
                lgbm_weight=float(choice["lgbm_weight"]),
            )
            policy_name = f"global_{history_policy}"
            predictions[policy_name][holdout_mask] = full_prediction[holdout_mask]
            selection_rows.append(
                {
                    "policy": policy_name,
                    "holdout_season": holdout_season,
                    "selection_scope": "all",
                    "selection_seasons": ",".join(map(str, selection_seasons)),
                    **choice.to_dict(),
                }
            )

            expert_policy_name = f"expert_{history_policy}"
            for scope in ("R", "F"):
                objective = grid_objective(
                    grid, selection_seasons=selection_seasons, scope=scope
                )
                choice = objective.iloc[0]
                definition = lookup[str(choice["pair"])]
                full_prediction = prediction_for_definition(
                    components,
                    definition,
                    method=str(choice["method"]),
                    lgbm_weight=float(choice["lgbm_weight"]),
                )
                scope_mask = holdout_mask & components["game_type"].eq(scope).to_numpy()
                predictions[expert_policy_name][scope_mask] = full_prediction[scope_mask]
                selection_rows.append(
                    {
                        "policy": expert_policy_name,
                        "holdout_season": holdout_season,
                        "selection_scope": scope,
                        "selection_seasons": ",".join(map(str, selection_seasons)),
                        **choice.to_dict(),
                    }
                )
    for policy, prediction in predictions.items():
        recent_mask = components["valid_season"].isin(RECENT_SEASONS).to_numpy()
        if np.isnan(prediction[recent_mask]).any():
            raise RuntimeError(f"rolling predictions are incomplete: {policy}")
    return predictions, pd.DataFrame(selection_rows)


def evaluate_policy_predictions(
    components: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    family: str,
) -> pd.DataFrame:
    rows = []
    for policy, prediction in predictions.items():
        partitions = [
            (
                "recent_pooled",
                None,
                components["valid_season"].isin(RECENT_SEASONS).to_numpy(),
            )
        ]
        partitions.extend(
            (
                "fold",
                season,
                components["valid_season"].eq(season).to_numpy(),
            )
            for season in RECENT_SEASONS
        )
        for partition, season, partition_mask in partitions:
            partition_frame = components.loc[partition_mask]
            partition_prediction = prediction[partition_mask]
            for scope, scope_mask in scopes(partition_frame).items():
                if not scope_mask.any():
                    continue
                metric = probability_metrics(
                    partition_frame.loc[scope_mask, TARGET_COL],
                    partition_prediction[scope_mask],
                )
                metric.update(
                    {
                        "family": family,
                        "policy": policy,
                        "partition": partition,
                        "valid_season": season,
                        "scope": scope,
                    }
                )
                rows.append(metric)
    return pd.DataFrame(rows)


def logit_intercept(prediction: np.ndarray, target_rate: float) -> float:
    """Find an intercept that matches the calibrated mean to the target rate."""
    eps = 1e-7
    p = np.clip(np.asarray(prediction, dtype=np.float64), eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p))
    low, high = -8.0, 8.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        calibrated_mean = float(
            (1.0 / (1.0 + np.exp(-np.clip(logits + midpoint, -30.0, 30.0)))).mean()
        )
        if calibrated_mean < target_rate:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def fit_calibration_parameters(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    method: str,
    group_mode: str,
) -> list[dict]:
    groups = ["__ALL__"] if group_mode == "global" else ["R", "F"]
    rows = []
    for group in groups:
        mask = (
            np.ones(len(frame), dtype=bool)
            if group == "__ALL__"
            else frame["game_type"].eq(group).to_numpy()
        )
        y = frame.loc[mask, TARGET_COL].to_numpy(dtype=np.float64)
        p = prediction[mask]
        if method == "probability_offset":
            value = float(np.mean(y - p))
        elif method == "logit_intercept":
            value = logit_intercept(p, float(y.mean()))
        else:
            raise ValueError(f"unknown calibration method: {method}")
        rows.append(
            {
                "group": group,
                "n": int(mask.sum()),
                "target_rate": float(y.mean()),
                "prediction_mean": float(p.mean()),
                "parameter": value,
            }
        )
    return rows


def apply_calibration(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    method: str,
    group_mode: str,
    parameters: list[dict],
) -> np.ndarray:
    result = np.asarray(prediction, dtype=np.float64).copy()
    parameter_map = {row["group"]: float(row["parameter"]) for row in parameters}
    groups = ["__ALL__"] if group_mode == "global" else ["R", "F"]
    for group in groups:
        mask = (
            np.ones(len(frame), dtype=bool)
            if group == "__ALL__"
            else frame["game_type"].eq(group).to_numpy()
        )
        value = parameter_map[group]
        if method == "probability_offset":
            result[mask] = np.clip(result[mask] + value, 1e-7, 1.0 - 1e-7)
        elif method == "logit_intercept":
            eps = 1e-7
            p = np.clip(result[mask], eps, 1.0 - eps)
            logits = np.log(p / (1.0 - p))
            result[mask] = 1.0 / (
                1.0 + np.exp(-np.clip(logits + value, -30.0, 30.0))
            )
        else:
            raise ValueError(f"unknown calibration method: {method}")
    return result


def calibrated_predictions_and_metrics(
    components: pd.DataFrame,
    definitions: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    metrics_parts = []
    parameter_rows = []
    prediction_store = {}
    policies = [
        (history, group_mode, method)
        for history in ("previous", "expanding")
        for group_mode in ("global", "game_type")
        for method in ("probability_offset", "logit_intercept")
    ]
    for definition in definitions:
        raw_prediction = prediction_for_definition(
            components,
            definition,
            method="logit",
            lgbm_weight=FIXED_LGBM_WEIGHT,
        )
        recent_mask = components["valid_season"].isin(RECENT_SEASONS).to_numpy()
        raw_recent = np.full(len(components), np.nan, dtype=np.float64)
        raw_recent[recent_mask] = raw_prediction[recent_mask]
        raw_policy_name = f"{definition['pair']}__none"
        prediction_store[raw_policy_name] = raw_recent

        pair_predictions = {"none": raw_recent}
        for history, group_mode, method in policies:
            policy = f"{history}__{group_mode}__{method}"
            calibrated = np.full(len(components), np.nan, dtype=np.float64)
            for holdout_season in RECENT_SEASONS:
                selection_seasons = (
                    [holdout_season - 1]
                    if history == "previous"
                    else [season for season in VALID_SEASONS if season < holdout_season]
                )
                train_mask = components["valid_season"].isin(selection_seasons).to_numpy()
                holdout_mask = components["valid_season"].eq(holdout_season).to_numpy()
                train_frame = components.loc[train_mask]
                holdout_frame = components.loc[holdout_mask]
                fitted = fit_calibration_parameters(
                    train_frame,
                    raw_prediction[train_mask],
                    method=method,
                    group_mode=group_mode,
                )
                calibrated[holdout_mask] = apply_calibration(
                    holdout_frame,
                    raw_prediction[holdout_mask],
                    method=method,
                    group_mode=group_mode,
                    parameters=fitted,
                )
                for row in fitted:
                    parameter_rows.append(
                        {
                            "pair": definition["pair"],
                            "policy": policy,
                            "holdout_season": holdout_season,
                            "selection_seasons": ",".join(map(str, selection_seasons)),
                            "history": history,
                            "group_mode": group_mode,
                            "method": method,
                            **row,
                        }
                    )
            pair_predictions[policy] = calibrated
            prediction_store[f"{definition['pair']}__{policy}"] = calibrated

        metrics = evaluate_policy_predictions(
            components, pair_predictions, family=definition["pair"]
        )
        metrics_parts.append(metrics)
    return (
        pd.concat(metrics_parts, ignore_index=True),
        pd.DataFrame(parameter_rows),
        prediction_store,
    )


def policy_summary(
    metrics: pd.DataFrame,
    *,
    family_column: str,
    base_briers: dict[int, float],
) -> pd.DataFrame:
    all_scope = metrics[metrics["scope"].eq("all")]
    rows = []
    for (family, policy), frame in all_scope.groupby([family_column, "policy"], sort=False):
        recent = frame[frame["partition"].eq("recent_pooled")].iloc[0]
        fold_2023 = frame[
            frame["partition"].eq("fold") & frame["valid_season"].eq(2023)
        ].iloc[0]
        fold_2024 = frame[
            frame["partition"].eq("fold") & frame["valid_season"].eq(2024)
        ].iloc[0]
        rows.append(
            {
                family_column: family,
                "policy": policy,
                "recent_brier": recent["brier"],
                "recent_bss_raw": recent["bss_raw"],
                "recent_mean_gap": recent["mean_gap"],
                "brier_2023": fold_2023["brier"],
                "brier_2024": fold_2024["brier"],
                "bss_raw_2023": fold_2023["bss_raw"],
                "bss_raw_2024": fold_2024["bss_raw"],
                "delta_brier_2023_vs_fixed_base": fold_2023["brier"] - base_briers[2023],
                "delta_brier_2024_vs_fixed_base": fold_2024["brier"] - base_briers[2024],
            }
        )
    summary = pd.DataFrame(rows)
    summary["safe_2023_2024_vs_fixed_base"] = (
        summary["delta_brier_2023_vs_fixed_base"].le(0.0)
        & summary["delta_brier_2024_vs_fixed_base"].le(0.0)
    )
    return summary.sort_values(["recent_brier", "brier_2024"], ignore_index=True)


def component_correlations(components: pd.DataFrame) -> pd.DataFrame:
    component_columns = [
        column
        for column in components.columns
        if column.startswith("lgbm__") or column.startswith("catboost__")
    ]
    partitions = [
        (
            "recent_pooled",
            None,
            components["valid_season"].isin(RECENT_SEASONS).to_numpy(),
        )
    ]
    partitions.extend(
        (
            "fold",
            season,
            components["valid_season"].eq(season).to_numpy(),
        )
        for season in VALID_SEASONS
    )
    rows = []
    for partition, season, mask in partitions:
        matrix = components.loc[mask, component_columns].corr()
        for left_index, left in enumerate(component_columns):
            for right in component_columns[left_index + 1 :]:
                rows.append(
                    {
                        "partition": partition,
                        "valid_season": season,
                        "left": left,
                        "right": right,
                        "pearson": float(matrix.loc[left, right]),
                    }
                )
    return pd.DataFrame(rows)


def paired_loss_deltas(
    components: pd.DataFrame,
    candidate_oof: pd.DataFrame,
    definitions: list[dict],
) -> pd.DataFrame:
    y = components[TARGET_COL].to_numpy(dtype=np.float64)
    base_prediction = candidate_oof[f"fixed__{BASE_PAIR}"].to_numpy(dtype=np.float64)
    base_loss = np.square(base_prediction - y)
    rows = []
    for definition in definitions:
        pair = definition["pair"]
        prediction = candidate_oof[f"fixed__{pair}"].to_numpy(dtype=np.float64)
        delta = np.square(prediction - y) - base_loss
        for season in VALID_SEASONS:
            fold_mask = components["valid_season"].eq(season).to_numpy()
            for scope in ("all", "R", "F"):
                mask = fold_mask if scope == "all" else fold_mask & components["game_type"].eq(scope).to_numpy()
                values = delta[mask]
                rows.append(
                    {
                        "pair": pair,
                        "valid_season": season,
                        "scope": scope,
                        "n": int(mask.sum()),
                        "mean_loss_delta_vs_base": float(values.mean()),
                        "naive_standard_error": float(values.std(ddof=1) / np.sqrt(len(values))),
                        "fraction_rows_improved": float((values < 0.0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    source_run_dir = resolve_source_run(args)
    required_configs = sorted(set(LGBM_CONFIGS + CATBOOST_CONFIGS))
    source_paths = {
        config: source_run_dir / f"oof_{config}.csv.gz"
        for config in required_configs
    }
    signature_payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "valid_seasons": list(VALID_SEASONS),
        "recent_seasons": list(RECENT_SEASONS),
        "lgbm_configs": list(LGBM_CONFIGS),
        "catboost_configs": list(CATBOOST_CONFIGS),
        "blend_methods": list(BLEND_METHODS),
        "blend_weights": list(BLEND_WEIGHTS),
        "fixed_lgbm_weight": FIXED_LGBM_WEIGHT,
        "script_sha256": sha256_file(Path(__file__)),
        "source_oof_sha256": {
            config: sha256_file(path) for config, path in source_paths.items()
        },
    }
    signature = experiment_signature(signature_payload)
    run_dir = args.output_dir / f"run_{signature}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "configuration.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "signature": signature,
                "source_run_dir": str(source_run_dir),
                **signature_payload,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    components, input_metadata = load_components(source_run_dir)
    definitions = pair_definitions()
    pd.DataFrame(definitions).to_csv(run_dir / "candidate_pairs.csv", index=False)
    print(f"Aligned OOF rows={len(components):,} candidate pairs={len(definitions)}")

    fixed_metrics, candidate_oof = fixed_metric_rows(components, definitions)
    fixed_summary = summarize_fixed_metrics(fixed_metrics)
    fixed_metrics.to_csv(run_dir / "fixed_pair_metrics.csv", index=False)
    fixed_summary.to_csv(run_dir / "fixed_pair_summary.csv", index=False)

    print("Building blend grid ...")
    grid = blend_grid_rows(components, definitions)
    grid.to_csv(run_dir / "blend_grid_metrics.csv", index=False)
    optimistic_grid = grid[
        grid["partition"].eq("recent_pooled") & grid["scope"].eq("all")
    ].sort_values("brier", ignore_index=True)
    optimistic_grid.to_csv(run_dir / "optimistic_grid_ranking.csv", index=False)

    rolling_predictions, rolling_selection = rolling_selection_predictions(
        components, definitions, grid
    )
    rolling_metrics = evaluate_policy_predictions(
        components, rolling_predictions, family="rolling_selection"
    )
    base_briers = {
        season: float(
            fixed_summary.loc[
                fixed_summary["pair"].eq(BASE_PAIR), f"brier_{season}"
            ].iloc[0]
        )
        for season in RECENT_SEASONS
    }
    rolling_summary = policy_summary(
        rolling_metrics,
        family_column="family",
        base_briers=base_briers,
    )
    rolling_selection.to_csv(run_dir / "rolling_selection.csv", index=False)
    rolling_metrics.to_csv(run_dir / "rolling_metrics.csv", index=False)
    rolling_summary.to_csv(run_dir / "rolling_summary.csv", index=False)

    print("Evaluating train-OOF-only calibration ...")
    calibration_metrics, calibration_parameters, calibration_predictions = (
        calibrated_predictions_and_metrics(components, definitions)
    )
    calibration_summary = policy_summary(
        calibration_metrics,
        family_column="family",
        base_briers=base_briers,
    )
    calibration_metrics.to_csv(run_dir / "calibration_metrics.csv", index=False)
    calibration_parameters.to_csv(run_dir / "calibration_parameters.csv", index=False)
    calibration_summary.to_csv(run_dir / "calibration_summary.csv", index=False)

    correlations = component_correlations(components)
    correlations.to_csv(run_dir / "component_correlations.csv", index=False)
    loss_deltas = paired_loss_deltas(components, candidate_oof, definitions)
    loss_deltas.to_csv(run_dir / "paired_loss_deltas.csv", index=False)

    fixed_safe = fixed_summary[
        fixed_summary["improves_recent"] & fixed_summary["safe_2023_2024"]
    ]
    selected_fixed = (
        fixed_safe.sort_values(["recent_brier", "brier_2024"]).iloc[0]
        if not fixed_safe.empty
        else fixed_summary[fixed_summary["pair"].eq(BASE_PAIR)].iloc[0]
    )
    safe_calibration = calibration_summary[
        calibration_summary["safe_2023_2024_vs_fixed_base"]
    ]
    selected_calibration = (
        safe_calibration.sort_values(["recent_brier", "brier_2024"]).iloc[0]
        if not safe_calibration.empty
        else calibration_summary[
            calibration_summary["family"].eq(BASE_PAIR)
            & calibration_summary["policy"].eq("none")
        ].iloc[0]
    )
    safe_rolling = rolling_summary[
        rolling_summary["safe_2023_2024_vs_fixed_base"]
    ]
    selected_rolling = (
        safe_rolling.sort_values(["recent_brier", "brier_2024"]).iloc[0]
        if not safe_rolling.empty
        else rolling_summary.sort_values(["recent_brier", "brier_2024"]).iloc[0]
    )

    best_calibration_key = (
        f"{selected_calibration['family']}__{selected_calibration['policy']}"
    )
    candidate_oof["selected_calibration_prediction"] = calibration_predictions[
        best_calibration_key
    ]
    for policy, prediction in rolling_predictions.items():
        candidate_oof[f"rolling__{policy}"] = prediction
    candidate_oof.to_csv(
        run_dir / "candidate_oof.csv.gz", index=False, compression="gzip"
    )

    best_grid = optimistic_grid.iloc[0]
    summary = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "signature": signature,
        "source_run_dir": str(source_run_dir),
        "input_audit": input_metadata,
        "base_pair": BASE_PAIR,
        "base_recent_brier": float(
            fixed_summary.loc[
                fixed_summary["pair"].eq(BASE_PAIR), "recent_brier"
            ].iloc[0]
        ),
        "base_recent_bss_raw": float(
            fixed_summary.loc[
                fixed_summary["pair"].eq(BASE_PAIR), "recent_bss_raw"
            ].iloc[0]
        ),
        "selected_safe_fixed_pair": str(selected_fixed["pair"]),
        "selected_safe_fixed_recent_brier": float(selected_fixed["recent_brier"]),
        "selected_safe_fixed_recent_bss_raw": float(selected_fixed["recent_bss_raw"]),
        "selected_safe_fixed_delta_brier_vs_base": float(
            selected_fixed["delta_recent_brier_vs_base"]
        ),
        "optimistic_grid_candidate": str(best_grid["candidate_id"]),
        "optimistic_grid_recent_brier": float(best_grid["brier"]),
        "selected_safe_calibration_pair": str(selected_calibration["family"]),
        "selected_safe_calibration_policy": str(selected_calibration["policy"]),
        "selected_safe_calibration_recent_brier": float(
            selected_calibration["recent_brier"]
        ),
        "selected_rolling_policy": str(selected_rolling["policy"]),
        "selected_rolling_recent_brier": float(selected_rolling["recent_brier"]),
        "selection_note": (
            "The optimistic grid uses the same recent OOF for ranking and is diagnostic only. "
            "A final training recipe must prioritize fold stability and rolling results."
        ),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "latest_run.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"signature": signature, "run_dir": str(run_dir), "summary": summary},
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("\nFixed 65:35 model-specific pairs")
    print(
        fixed_summary[
            [
                "pair",
                "recent_brier",
                "recent_bss_raw",
                "delta_recent_brier_vs_base",
                "delta_brier_2023_vs_base",
                "delta_brier_2024_vs_base",
                "safe_2023_2024",
            ]
        ].to_string(index=False)
    )
    print("\nRolling selection")
    print(
        rolling_summary[
            [
                "policy",
                "recent_brier",
                "recent_bss_raw",
                "delta_brier_2023_vs_fixed_base",
                "delta_brier_2024_vs_fixed_base",
                "safe_2023_2024_vs_fixed_base",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved results to {run_dir}")


if __name__ == "__main__":
    main()
