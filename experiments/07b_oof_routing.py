#!/usr/bin/env python3
"""Recombine experiment-07 OOF predictions with fixed row-wise R/F routing.

This experiment does not train a model or tune a blend weight.  It keeps the
current CatBoost prediction on R rows and substitutes one predeclared shared
CatBoost policy on F rows, then recomputes the fixed 65:35 logit blend.  The
primary and backup routes were fixed after the experiment-07 report and before
this script evaluates the saved row-level OOF predictions.
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
LGBM_WEIGHT = 0.65
PROBABILITY_EPS = 1e-7
COMPARISON_TOLERANCE = 1e-12
BASE_CATBOOST_POLICY = "current_all_nogt"
BASE_ROUTE = "current_r__current_f"
PRIMARY_ROUTE = "current_r__latest_f1_f"
BACKUP_ROUTE = "current_r__fast_decay_f"

KEY_COLUMNS = [ID_COL, TARGET_COL, "season", "game_type", "valid_season"]

ROUTES: dict[str, dict] = {
    BASE_ROUTE: {
        "description": "Current CatBoost on both R and F; exact submit-02 OOF anchor",
        "r_policy": BASE_CATBOOST_POLICY,
        "f_policy": BASE_CATBOOST_POLICY,
        "selection_role": "base",
        "selection_eligible": False,
    },
    PRIMARY_ROUTE: {
        "description": "Current CatBoost on R; shared_latest_f1 CatBoost on F",
        "r_policy": BASE_CATBOOST_POLICY,
        "f_policy": "shared_latest_f1",
        "selection_role": "primary",
        "selection_eligible": True,
    },
    BACKUP_ROUTE: {
        "description": "Current CatBoost on R; shared_f_fast_decay CatBoost on F",
        "r_policy": BASE_CATBOOST_POLICY,
        "f_policy": "shared_f_fast_decay",
        "selection_role": "backup",
        "selection_eligible": True,
    },
    "current_r__r_only_f": {
        "description": "Diagnostic only: current CatBoost on R; R-only CatBoost on F",
        "r_policy": BASE_CATBOOST_POLICY,
        "f_policy": "r_only",
        "selection_role": "diagnostic_unstable",
        "selection_eligible": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--f-shift-output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/f_shift_cv_v1",
    )
    parser.add_argument("--source-run-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/f_routing_v1",
    )
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


def resolve_source_run(args: argparse.Namespace) -> Path:
    if args.source_run_dir is not None:
        candidates = [args.source_run_dir]
    else:
        latest_path = args.f_shift_output_root / "latest_run.json"
        if not latest_path.exists():
            raise FileNotFoundError(
                f"07 latest_run.json is missing: {latest_path}. Run notebook 07 first."
            )
        with latest_path.open(encoding="utf-8") as handle:
            latest = json.load(handle)
        recorded = Path(latest["run_dir"])
        candidates = [recorded, args.f_shift_output_root / recorded.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"07 run directory is missing: {candidates}")


def required_source_columns() -> list[str]:
    policies = sorted(
        {definition["r_policy"] for definition in ROUTES.values()}
        | {definition["f_policy"] for definition in ROUTES.values()}
    )
    return KEY_COLUMNS + ["lgbm_prediction"] + [
        f"catboost__{policy}" for policy in policies
    ] + [f"blend__{BASE_CATBOOST_POLICY}"]


def load_source_oof(source_run_dir: Path) -> tuple[pd.DataFrame, dict]:
    oof_path = source_run_dir / "oof_predictions.csv.gz"
    summary_path = source_run_dir / "summary.json"
    for path in (oof_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(f"required experiment-07 result is missing: {path}")
    with summary_path.open(encoding="utf-8") as handle:
        source_summary = json.load(handle)
    if source_summary.get("base_policy") != BASE_CATBOOST_POLICY:
        raise RuntimeError(
            f"unexpected experiment-07 base policy: {source_summary.get('base_policy')}"
        )
    if source_summary.get("selected_policy") != BASE_CATBOOST_POLICY:
        raise RuntimeError(
            "experiment-07 did not retain its base policy; review the routing premise"
        )

    print(f"Loading {oof_path} ...")
    oof = pd.read_csv(
        oof_path,
        usecols=required_source_columns(),
        dtype={ID_COL: "string", "game_type": "string"},
    )
    validate_source_oof(oof)
    metadata = {
        "source_run_dir": str(source_run_dir),
        "source_signature": source_summary.get("signature"),
        "source_oof_sha256": sha256_file(oof_path),
        "source_summary_sha256": sha256_file(summary_path),
        "source_selected_policy": source_summary.get("selected_policy"),
        "source_base_recent_brier": float(source_summary["base_recent_brier"]),
        "source_base_recent_bss_raw": float(source_summary["base_recent_bss_raw"]),
        "rows": int(len(oof)),
    }
    return oof, metadata


def validate_source_oof(oof: pd.DataFrame) -> None:
    missing = sorted(set(required_source_columns()) - set(oof.columns))
    if missing:
        raise ValueError(f"source OOF columns are missing: {missing}")
    if oof[ID_COL].isna().any() or oof[ID_COL].duplicated().any():
        raise ValueError("source OOF row_id must be non-null and unique")
    if not set(oof[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError("source OOF target must be binary")
    found_seasons = tuple(
        sorted(int(value) for value in oof["valid_season"].unique())
    )
    if found_seasons != VALID_SEASONS:
        raise ValueError("source OOF must contain validation seasons 2022, 2023, and 2024")
    if not set(oof["game_type"].astype(str).unique()).issubset({"R", "F"}):
        raise ValueError("source OOF game_type must contain only R and F")
    probability_columns = [
        column
        for column in oof.columns
        if column == "lgbm_prediction"
        or column.startswith("catboost__")
        or column.startswith("blend__")
    ]
    for column in probability_columns:
        values = oof[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or ((values <= 0.0) | (values >= 1.0)).any():
            raise ValueError(f"invalid source probabilities: {column}")


def logit_blend(
    lgbm_prediction: np.ndarray,
    catboost_prediction: np.ndarray,
    lgbm_weight: float = LGBM_WEIGHT,
) -> np.ndarray:
    p_lgbm = np.clip(
        np.asarray(lgbm_prediction, dtype=np.float64),
        PROBABILITY_EPS,
        1.0 - PROBABILITY_EPS,
    )
    p_cat = np.clip(
        np.asarray(catboost_prediction, dtype=np.float64),
        PROBABILITY_EPS,
        1.0 - PROBABILITY_EPS,
    )
    lgbm_logit = np.log(p_lgbm / (1.0 - p_lgbm))
    cat_logit = np.log(p_cat / (1.0 - p_cat))
    blended_logit = lgbm_weight * lgbm_logit + (1.0 - lgbm_weight) * cat_logit
    return np.clip(
        1.0 / (1.0 + np.exp(-np.clip(blended_logit, -30.0, 30.0))),
        PROBABILITY_EPS,
        1.0 - PROBABILITY_EPS,
    )


def build_routed_oof(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    routed = oof[KEY_COLUMNS + ["lgbm_prediction"]].copy()
    is_r = oof["game_type"].astype(str).eq("R").to_numpy()
    is_f = ~is_r
    base_cat = oof[f"catboost__{BASE_CATBOOST_POLICY}"].to_numpy(dtype=np.float64)
    base_blend_source = oof[f"blend__{BASE_CATBOOST_POLICY}"].to_numpy(dtype=np.float64)
    base_blend_recomputed = logit_blend(
        oof["lgbm_prediction"].to_numpy(dtype=np.float64), base_cat
    )
    base_reconstruction_max = float(
        np.max(np.abs(base_blend_recomputed - base_blend_source))
    )
    if base_reconstruction_max > COMPARISON_TOLERANCE:
        raise RuntimeError(
            f"source base blend reconstruction mismatch: {base_reconstruction_max}"
        )

    route_audits = {}
    for route_name, definition in ROUTES.items():
        r_source = oof[f"catboost__{definition['r_policy']}"].to_numpy(
            dtype=np.float64
        )
        f_source = oof[f"catboost__{definition['f_policy']}"].to_numpy(
            dtype=np.float64
        )
        routed_cat = np.where(is_f, f_source, r_source)
        routed_blend = logit_blend(
            oof["lgbm_prediction"].to_numpy(dtype=np.float64), routed_cat
        )
        source_f_blend = logit_blend(
            oof["lgbm_prediction"].to_numpy(dtype=np.float64), f_source
        )
        routed[f"catboost__{route_name}"] = routed_cat
        routed[f"blend__{route_name}"] = routed_blend

        r_cat_max = float(np.max(np.abs(routed_cat[is_r] - base_cat[is_r])))
        f_cat_max = float(np.max(np.abs(routed_cat[is_f] - f_source[is_f])))
        r_blend_max = float(
            np.max(np.abs(routed_blend[is_r] - base_blend_source[is_r]))
        )
        f_blend_max = float(
            np.max(np.abs(routed_blend[is_f] - source_f_blend[is_f]))
        )
        changed_r_rows = int(
            np.count_nonzero(np.abs(routed_cat[is_r] - base_cat[is_r]) > 0.0)
        )
        if max(r_cat_max, f_cat_max, r_blend_max, f_blend_max) > COMPARISON_TOLERANCE:
            raise RuntimeError(f"route splice audit failed: {route_name}")
        if changed_r_rows != 0:
            raise RuntimeError(f"route changed R predictions: {route_name}")
        route_audits[route_name] = {
            "r_rows": int(is_r.sum()),
            "f_rows": int(is_f.sum()),
            "r_catboost_vs_current_max_abs_difference": r_cat_max,
            "f_catboost_vs_policy_max_abs_difference": f_cat_max,
            "r_blend_vs_current_max_abs_difference": r_blend_max,
            "f_blend_vs_policy_max_abs_difference": f_blend_max,
            "changed_r_rows": changed_r_rows,
        }
    audit = {
        "status": "passed",
        "rows": int(len(oof)),
        "r_rows": int(is_r.sum()),
        "f_rows": int(is_f.sum()),
        "base_blend_reconstruction_max_abs_difference": base_reconstruction_max,
        "routing_uses_only_current_row_game_type": True,
        "route_audits": route_audits,
    }
    return routed, audit


def scope_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(frame), dtype=bool),
        "R": frame["game_type"].astype(str).eq("R").to_numpy(),
        "F": frame["game_type"].astype(str).eq("F").to_numpy(),
    }


def metric_rows(routed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    partitions: list[tuple[str, int | None, np.ndarray]] = [
        ("pooled", None, np.ones(len(routed), dtype=bool)),
        (
            "recent_pooled",
            None,
            routed["valid_season"].isin(RECENT_SEASONS).to_numpy(),
        ),
    ]
    partitions.extend(
        ("fold", season, routed["valid_season"].eq(season).to_numpy())
        for season in VALID_SEASONS
    )
    for route_name in ROUTES:
        for model, column in (
            ("catboost_routed", f"catboost__{route_name}"),
            ("fixed_logit_65_35", f"blend__{route_name}"),
        ):
            for partition, season, partition_mask in partitions:
                frame = routed.loc[partition_mask]
                prediction = routed.loc[partition_mask, column].to_numpy(
                    dtype=np.float64
                )
                for scope, scope_mask in scope_masks(frame).items():
                    if not scope_mask.any():
                        continue
                    result = probability_metrics(
                        frame.loc[scope_mask, TARGET_COL], prediction[scope_mask]
                    )
                    result.update(
                        {
                            "route": route_name,
                            "model": model,
                            "partition": partition,
                            "valid_season": season,
                            "scope": scope,
                        }
                    )
                    rows.append(result)
    return pd.DataFrame(rows)


def one_metric(
    frame: pd.DataFrame, *, partition: str, scope: str, season: int | None = None
) -> pd.Series:
    mask = frame["partition"].eq(partition) & frame["scope"].eq(scope)
    if season is not None:
        mask &= frame["valid_season"].eq(season)
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise RuntimeError(
            f"expected one metric row: partition={partition}, scope={scope}, season={season}"
        )
    return selected.iloc[0]


def build_route_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    blend = metrics[metrics["model"].eq("fixed_logit_65_35")]
    rows = []
    for route_name, definition in ROUTES.items():
        frame = blend[blend["route"].eq(route_name)]
        recent_all = one_metric(frame, partition="recent_pooled", scope="all")
        recent_r = one_metric(frame, partition="recent_pooled", scope="R")
        recent_f = one_metric(frame, partition="recent_pooled", scope="F")
        row = {
            "route": route_name,
            "description": definition["description"],
            "selection_role": definition["selection_role"],
            "selection_eligible": bool(definition["selection_eligible"]),
            "recent_brier": recent_all["brier"],
            "recent_bss_raw": recent_all["bss_raw"],
            "recent_mean_gap": recent_all["mean_gap"],
            "recent_r_brier": recent_r["brier"],
            "recent_f_brier": recent_f["brier"],
        }
        for season in VALID_SEASONS:
            for scope, label in (("all", ""), ("R", "_r"), ("F", "_f")):
                fold = one_metric(frame, partition="fold", scope=scope, season=season)
                row[f"brier{label}_{season}"] = fold["brier"]
                row[f"bss_raw{label}_{season}"] = fold["bss_raw"]
                row[f"mean_gap{label}_{season}"] = fold["mean_gap"]
        rows.append(row)

    summary = pd.DataFrame(rows)
    base = summary[summary["route"].eq(BASE_ROUTE)].iloc[0]
    comparison_columns = [
        "recent_brier",
        "recent_r_brier",
        "recent_f_brier",
        *[f"brier_{season}" for season in VALID_SEASONS],
        *[f"brier_r_{season}" for season in VALID_SEASONS],
        *[f"brier_f_{season}" for season in VALID_SEASONS],
    ]
    for column in comparison_columns:
        summary[f"delta_{column}_vs_base"] = summary[column] - float(base[column])
    summary["r_predictions_exact"] = summary[
        [f"delta_brier_r_{season}_vs_base" for season in VALID_SEASONS]
    ].abs().max(axis=1).le(COMPARISON_TOLERANCE)
    summary["improves_recent_all"] = summary["delta_recent_brier_vs_base"].lt(0.0)
    summary["improves_recent_f"] = summary["delta_recent_f_brier_vs_base"].lt(0.0)
    summary["safe_all_folds"] = summary[
        [f"delta_brier_{season}_vs_base" for season in VALID_SEASONS]
    ].max(axis=1).le(COMPARISON_TOLERANCE)
    summary["safe_f_all_folds"] = summary[
        [f"delta_brier_f_{season}_vs_base" for season in VALID_SEASONS]
    ].max(axis=1).le(COMPARISON_TOLERANCE)
    summary["passes_fixed_gate"] = (
        summary["selection_eligible"]
        & summary["r_predictions_exact"]
        & summary["improves_recent_all"]
        & summary["improves_recent_f"]
        & summary["safe_all_folds"]
        & summary["safe_f_all_folds"]
    )
    summary["worst_fold_delta"] = summary[
        [f"delta_brier_{season}_vs_base" for season in VALID_SEASONS]
    ].max(axis=1)
    return summary.sort_values(
        ["selection_role", "recent_brier"], ignore_index=True
    )


def paired_loss_deltas(routed: pd.DataFrame) -> pd.DataFrame:
    target = routed[TARGET_COL].to_numpy(dtype=np.float64)
    base_prediction = routed[f"blend__{BASE_ROUTE}"].to_numpy(dtype=np.float64)
    base_loss = np.square(base_prediction - target)
    partitions: list[tuple[str, int | None, np.ndarray]] = [
        (
            "recent_pooled",
            None,
            routed["valid_season"].isin(RECENT_SEASONS).to_numpy(),
        )
    ]
    partitions.extend(
        ("fold", season, routed["valid_season"].eq(season).to_numpy())
        for season in VALID_SEASONS
    )
    rows = []
    for route_name in ROUTES:
        candidate = routed[f"blend__{route_name}"].to_numpy(dtype=np.float64)
        loss_delta = np.square(candidate - target) - base_loss
        for partition, season, partition_mask in partitions:
            frame = routed.loc[partition_mask]
            values = loss_delta[partition_mask]
            for scope, scope_mask in scope_masks(frame).items():
                if not scope_mask.any():
                    continue
                scoped = values[scope_mask]
                rows.append(
                    {
                        "route": route_name,
                        "partition": partition,
                        "valid_season": season,
                        "scope": scope,
                        "n": int(len(scoped)),
                        "mean_loss_delta_vs_base": float(scoped.mean()),
                        "naive_standard_error": float(
                            scoped.std(ddof=1) / np.sqrt(len(scoped))
                        ),
                        "fraction_rows_improved": float((scoped < 0.0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def prediction_correlations(routed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    partitions: list[tuple[str, int | None, np.ndarray]] = [
        (
            "recent_pooled",
            None,
            routed["valid_season"].isin(RECENT_SEASONS).to_numpy(),
        )
    ]
    partitions.extend(
        ("fold", season, routed["valid_season"].eq(season).to_numpy())
        for season in VALID_SEASONS
    )
    base = routed[f"blend__{BASE_ROUTE}"].to_numpy(dtype=np.float64)
    for route_name in ROUTES:
        candidate = routed[f"blend__{route_name}"].to_numpy(dtype=np.float64)
        for partition, season, partition_mask in partitions:
            frame = routed.loc[partition_mask]
            for scope, scope_mask in scope_masks(frame).items():
                combined = partition_mask.copy()
                combined[partition_mask] = scope_mask
                if combined.sum() < 2:
                    continue
                rows.append(
                    {
                        "route": route_name,
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


def choose_predeclared_route(route_summary: pd.DataFrame) -> tuple[str, str]:
    lookup = route_summary.set_index("route")
    if bool(lookup.loc[PRIMARY_ROUTE, "passes_fixed_gate"]):
        return PRIMARY_ROUTE, "primary_passed"
    if bool(lookup.loc[BACKUP_ROUTE, "passes_fixed_gate"]):
        return BACKUP_ROUTE, "primary_failed_backup_passed"
    return BASE_ROUTE, "no_predeclared_route_passed"


def main() -> None:
    args = parse_args()
    source_run_dir = resolve_source_run(args)
    source_oof_path = source_run_dir / "oof_predictions.csv.gz"
    source_summary_path = source_run_dir / "summary.json"
    route_manifest = pd.DataFrame(
        [{"route": name, **definition} for name, definition in ROUTES.items()]
    )
    signature_payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "valid_seasons": list(VALID_SEASONS),
        "recent_seasons": list(RECENT_SEASONS),
        "fixed_lgbm_weight": LGBM_WEIGHT,
        "base_route": BASE_ROUTE,
        "primary_route": PRIMARY_ROUTE,
        "backup_route": BACKUP_ROUTE,
        "routes": ROUTES,
        "hypothesis_origin": (
            "Routes were fixed from the documented segment findings in report 07. "
            "No route or blend-weight grid is searched in this experiment."
        ),
        "selection_priority": [PRIMARY_ROUTE, BACKUP_ROUTE, BASE_ROUTE],
        "selection_gate": (
            "R predictions exact; recent overall and F Brier improve; overall and F "
            "Brier do not worsen in any of 2022, 2023, or 2024."
        ),
        "script_sha256": sha256_file(Path(__file__)),
        "evaluation_sha256": sha256_file(SRC_DIR / "evaluation.py"),
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
            "source_run_dir": str(source_run_dir),
            **signature_payload,
        },
    )
    route_manifest.to_csv(run_dir / "route_manifest.csv", index=False)

    source_oof, source_audit = load_source_oof(source_run_dir)
    routed, integrity_audit = build_routed_oof(source_oof)
    metrics = metric_rows(routed)
    route_summary = build_route_summary(metrics)
    selected_route, selection_status = choose_predeclared_route(route_summary)
    paired = paired_loss_deltas(routed)
    correlations = prediction_correlations(routed)

    base_recent = route_summary[
        route_summary["route"].eq(BASE_ROUTE)
    ].iloc[0]
    if (
        abs(
            float(base_recent["recent_brier"])
            - float(source_audit["source_base_recent_brier"])
        )
        > COMPARISON_TOLERANCE
    ):
        raise RuntimeError("07b base recent Brier does not match experiment 07")

    routed.to_csv(run_dir / "route_oof.csv.gz", index=False, compression="gzip")
    metrics.to_csv(run_dir / "route_metrics.csv", index=False)
    route_summary.to_csv(run_dir / "route_summary.csv", index=False)
    paired.to_csv(run_dir / "paired_loss_deltas.csv", index=False)
    correlations.to_csv(run_dir / "prediction_correlations.csv", index=False)
    write_json(run_dir / "integrity_audit.json", integrity_audit)

    selected = route_summary[route_summary["route"].eq(selected_route)].iloc[0]
    summary = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "signature": signature,
        "source_run_dir": str(source_run_dir),
        "source_audit": source_audit,
        "integrity_audit": integrity_audit,
        "base_route": BASE_ROUTE,
        "primary_route": PRIMARY_ROUTE,
        "backup_route": BACKUP_ROUTE,
        "selection_status": selection_status,
        "selected_route": selected_route,
        "selected_route_role": str(selected["selection_role"]),
        "selected_recent_brier": float(selected["recent_brier"]),
        "selected_recent_bss_raw": float(selected["recent_bss_raw"]),
        "selected_delta_recent_brier_vs_base": float(
            selected["delta_recent_brier_vs_base"]
        ),
        "selected_passes_fixed_gate": bool(selected["passes_fixed_gate"]),
        "fixed_lgbm_weight": LGBM_WEIGHT,
        "selection_note": (
            "The route priority and safety gate were fixed before this 07b run. "
            "The diagnostic R-only route is never selection-eligible."
        ),
        "next_step": (
            "If the primary route passes, reproduce its two-CatBoost routing recipe "
            "in a final-training submission notebook; otherwise follow the fixed "
            "backup/base priority."
        ),
    }
    write_json(run_dir / "summary.json", summary)
    write_json(
        args.output_dir / "latest_run.json",
        {"signature": signature, "run_dir": str(run_dir), "summary": summary},
    )

    display_columns = [
        "route",
        "selection_role",
        "recent_brier",
        "recent_bss_raw",
        "delta_recent_brier_vs_base",
        "delta_recent_f_brier_vs_base",
        "worst_fold_delta",
        "passes_fixed_gate",
    ]
    print("\nFixed R/F routing comparison")
    print(route_summary[display_columns].to_string(index=False))
    print(f"\nSelected route: {selected_route} ({selection_status})")
    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
