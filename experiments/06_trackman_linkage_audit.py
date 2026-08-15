#!/usr/bin/env python3
"""Audit whether anonymous main-data pitchers can be linked to Trackman pitchers.

This experiment deliberately does not read the target.  It reconstructs historical
pitch groups from the official row-level as-of fields, compares season fingerprints
with Trackman history, and accepts only mutually unique mappings that reproduce in
both early and late seasons.  The output is a feasibility audit, not a submission
model and not proof of player identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RESULT_SCHEMA_VERSION = 1
SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)
EARLY_SEASONS = (2019, 2020, 2021)
LATE_SEASONS = (2022, 2023, 2024)
PITCH_GROUPS = ("fastball", "breaking", "offspeed")
RATE_COLUMNS = (
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)
MAIN_ID = "pitcher_id"
TRACK_ID = "pitcher_trackman_id"

MAIN_COLUMNS = [
    "season",
    MAIN_ID,
    "pitcher_hand",
    "pitcher_team_id",
    "asof_pitcher_pitchmix_n",
    *RATE_COLUMNS,
]
TRACK_COLUMNS = [
    "season",
    TRACK_ID,
    "pitcher_hand",
    "pitcher_team",
    "pitch_type_group",
]

MAIN_HAND_MAP = {"1": "Left", "2": "Right"}
TEAM_FAMILY_MAP = {
    "DOO_BEA": "DOO",
    "MIN_DOO": "DOO",
    "NC_DIN": "NC",
    "MIN_NCD": "NC",
    "LG_TWI": "LG",
    "MIN_LGT": "LG",
    "HAN_EAG": "HAN",
    "MIN_HAN": "HAN",
    "KIW_HER": "KIW",
    "MIN_HER": "KIW",
    "KT_WIZ": "KT",
    "MIN_KTW": "KT",
    "LOT_GIA": "LOT",
    "MIN_LOT": "LOT",
    "SAM_LIO": "SAM",
    "MIN_SAM": "SAM",
    "KIA_TIG": "KIA",
    "MIN_KIA": "KIA",
    "SK_WYV": "SK",
    "MIN_SKW": "SK",
    "SSG_LAN": "SSG",
    "MIN_SSG": "SSG",
    "KBO_ARM": "ARM",
    "MIN_HAW": "ARM",
    "KBO_POL": "POL",
    "ACE_MEX": "ACE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/train.csv",
    )
    parser.add_argument(
        "--trackman-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/trackman_history.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "outputs/trackman_linkage_audit_v1",
    )
    parser.add_argument("--min-pitches-per-season", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-overlap-seasons", type=int, default=2)
    parser.add_argument("--max-pitchmix-rmse", type=float, default=0.055)
    parser.add_argument("--min-score-margin", type=float, default=0.015)
    parser.add_argument("--min-activity-jaccard", type=float, default=0.50)
    parser.add_argument("--min-activity-recall", type=float, default=0.75)
    parser.add_argument("--min-season-vote-share", type=float, default=0.60)
    parser.add_argument("--min-team-match-rate", type=float, default=0.80)
    parser.add_argument("--min-team-evidence-seasons", type=int, default=2)
    parser.add_argument("--proceed-min-row-coverage", type=float, default=0.60)
    parser.add_argument("--partial-min-row-coverage", type=float, default=0.30)
    parser.add_argument("--proceed-min-median-vote-share", type=float, default=0.80)
    parser.add_argument("--proceed-min-median-team-match", type=float, default=0.90)
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


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def stable_mode(
    frame: pd.DataFrame,
    *,
    keys: list[str],
    value: str,
    output_value: str,
    count_name: str,
) -> pd.DataFrame:
    counts = (
        frame.groupby(keys + [value], dropna=False, observed=True)
        .size()
        .rename(count_name)
        .reset_index()
        .sort_values(keys + [count_name, value], ascending=[True] * len(keys) + [False, True])
    )
    return counts.groupby(keys, sort=False, as_index=False).head(1).rename(
        columns={value: output_value}
    )


def validate_seasons(frame: pd.DataFrame, label: str) -> None:
    found = tuple(sorted(int(value) for value in frame["season"].unique()))
    if found != SEASONS:
        raise ValueError(f"unexpected {label} seasons: expected={SEASONS}, found={found}")


def build_main_fingerprints(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    print(f"Loading main as-of columns: {path}")
    frame = pd.read_csv(
        path,
        usecols=MAIN_COLUMNS,
        dtype={MAIN_ID: "string", "pitcher_team_id": "string"},
    )
    validate_seasons(frame, "main")
    if frame[MAIN_COLUMNS].isna().all(axis=0).any():
        missing = frame[MAIN_COLUMNS].isna().all(axis=0)
        raise ValueError(f"all-null main columns: {missing[missing].index.tolist()}")

    hand_text = frame["pitcher_hand"].astype("Int64").astype("string")
    frame["hand_normalized"] = hand_text.map(MAIN_HAND_MAP).fillna("Unknown")

    n = frame["asof_pitcher_pitchmix_n"].to_numpy(dtype=np.float64)
    rate_values = frame[list(RATE_COLUMNS)].fillna(0.0).to_numpy(dtype=np.float64)
    cumulative_counts = rate_values * n[:, None]
    cumulative = pd.DataFrame(cumulative_counts, columns=PITCH_GROUPS, index=frame.index)
    grouped = frame.groupby(MAIN_ID, sort=False, observed=True)
    next_n = grouped["asof_pitcher_pitchmix_n"].shift(-1).to_numpy(dtype=np.float64)
    next_cumulative = cumulative.groupby(frame[MAIN_ID], sort=False).shift(-1)
    deltas = next_cumulative.to_numpy(dtype=np.float64) - cumulative_counts

    has_next = np.isfinite(next_n)
    n_increment = next_n - n
    sequential = has_next & np.isclose(n_increment, 1.0, atol=1e-9)
    delta_sum_error = np.full(len(frame), np.nan, dtype=np.float64)
    delta_max = np.full(len(frame), np.nan, dtype=np.float64)
    delta_min = np.full(len(frame), np.nan, dtype=np.float64)
    delta_sum_error[has_next] = np.abs(np.sum(deltas[has_next], axis=1) - 1.0)
    delta_max[has_next] = np.max(deltas[has_next], axis=1)
    delta_min[has_next] = np.min(deltas[has_next], axis=1)
    recoverable = (
        sequential
        & (delta_sum_error <= 0.02)
        & (delta_max >= 0.98)
        & (delta_min >= -0.02)
    )
    recovered_group_index = np.argmax(np.nan_to_num(deltas, nan=-np.inf), axis=1)

    recovered = frame.loc[recoverable, [MAIN_ID, "season"]].copy()
    recovered["pitch_type_group"] = np.asarray(PITCH_GROUPS)[
        recovered_group_index[recoverable]
    ]
    recovered_counts = (
        recovered.groupby([MAIN_ID, "season", "pitch_type_group"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=PITCH_GROUPS, fill_value=0)
        .reset_index()
    )
    recovered_counts.columns.name = None

    season_rows = (
        frame.groupby([MAIN_ID, "season"], observed=True)
        .size()
        .rename("total_count")
        .reset_index()
    )
    season_fingerprint = season_rows.merge(
        recovered_counts,
        on=[MAIN_ID, "season"],
        how="left",
        validate="one_to_one",
    )
    season_fingerprint[list(PITCH_GROUPS)] = season_fingerprint[
        list(PITCH_GROUPS)
    ].fillna(0).astype("int64")
    season_fingerprint["core_count"] = season_fingerprint[
        list(PITCH_GROUPS)
    ].sum(axis=1)
    for group in PITCH_GROUPS:
        season_fingerprint[f"{group}_rate"] = np.divide(
            season_fingerprint[group],
            season_fingerprint["core_count"],
            out=np.full(len(season_fingerprint), np.nan, dtype=np.float64),
            where=season_fingerprint["core_count"].to_numpy() > 0,
        )

    hand_mode = stable_mode(
        frame,
        keys=[MAIN_ID],
        value="hand_normalized",
        output_value="hand_normalized",
        count_name="hand_mode_rows",
    )
    hand_unique = (
        frame.groupby(MAIN_ID, observed=True)["hand_normalized"]
        .nunique(dropna=False)
        .rename("hand_unique_values")
        .reset_index()
    )
    pitcher_rows = (
        frame.groupby(MAIN_ID, observed=True).size().rename("main_rows").reset_index()
    )
    pitcher_meta = (
        pitcher_rows.merge(hand_mode, on=MAIN_ID, validate="one_to_one")
        .merge(hand_unique, on=MAIN_ID, validate="one_to_one")
    )

    team_season = stable_mode(
        frame,
        keys=[MAIN_ID, "season"],
        value="pitcher_team_id",
        output_value="main_team_id",
        count_name="main_team_rows",
    )

    first_n = grouped["asof_pitcher_pitchmix_n"].first().to_numpy(dtype=np.float64)
    last_n = grouped["asof_pitcher_pitchmix_n"].last().to_numpy(dtype=np.float64)
    row_counts = grouped.size().to_numpy(dtype=np.int64)
    rate_sum = frame[list(RATE_COLUMNS)].sum(axis=1, min_count=1).to_numpy()
    audit = {
        "rows": int(len(frame)),
        "pitchers": int(frame[MAIN_ID].nunique()),
        "target_column_read": False,
        "first_asof_n_zero_fraction": float(np.mean(np.isclose(first_n, 0.0))),
        "last_asof_n_equals_rows_minus_one_fraction": float(
            np.mean(np.isclose(last_n, row_counts - 1))
        ),
        "consecutive_rows": int(has_next.sum()),
        "n_increment_one_fraction": float(sequential.sum() / max(has_next.sum(), 1)),
        "recovered_rows": int(recoverable.sum()),
        "recovered_fraction_of_consecutive": float(
            recoverable.sum() / max(has_next.sum(), 1)
        ),
        "delta_sum_error_max": float(np.nanmax(delta_sum_error[has_next])),
        "delta_min_min": float(np.nanmin(delta_min[has_next])),
        "delta_max_min": float(np.nanmin(delta_max[has_next])),
        "rate_sum_mean": float(np.nanmean(rate_sum)),
        "rate_sum_min": float(np.nanmin(rate_sum)),
        "rate_sum_max": float(np.nanmax(rate_sum)),
        "unknown_hand_rows": int(frame["hand_normalized"].eq("Unknown").sum()),
        "multi_hand_pitchers": int(pitcher_meta["hand_unique_values"].gt(1).sum()),
    }
    return season_fingerprint, pitcher_meta, team_season, audit


def normalize_track_team(value: object) -> str:
    if pd.isna(value):
        return "__MISSING__"
    text = str(value)
    return TEAM_FAMILY_MAP.get(text, text)


def build_trackman_fingerprints(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    print(f"Loading Trackman linkage columns: {path}")
    frame = pd.read_csv(path, usecols=TRACK_COLUMNS, dtype={TRACK_ID: "string"})
    validate_seasons(frame, "Trackman")
    frame["hand_normalized"] = (
        frame["pitcher_hand"].astype("string").fillna("Unknown")
    )
    frame["pitch_type_group"] = (
        frame["pitch_type_group"].astype("string").str.lower()
    )
    frame["track_team_family"] = frame["pitcher_team"].map(normalize_track_team)

    season_rows = (
        frame.groupby([TRACK_ID, "season"], observed=True)
        .size()
        .rename("total_count")
        .reset_index()
    )
    core = frame[frame["pitch_type_group"].isin(PITCH_GROUPS)].copy()
    group_counts = (
        core.groupby([TRACK_ID, "season", "pitch_type_group"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=PITCH_GROUPS, fill_value=0)
        .reset_index()
    )
    group_counts.columns.name = None
    season_fingerprint = season_rows.merge(
        group_counts,
        on=[TRACK_ID, "season"],
        how="left",
        validate="one_to_one",
    )
    season_fingerprint[list(PITCH_GROUPS)] = season_fingerprint[
        list(PITCH_GROUPS)
    ].fillna(0).astype("int64")
    season_fingerprint["core_count"] = season_fingerprint[
        list(PITCH_GROUPS)
    ].sum(axis=1)
    for group in PITCH_GROUPS:
        season_fingerprint[f"{group}_rate"] = np.divide(
            season_fingerprint[group],
            season_fingerprint["core_count"],
            out=np.full(len(season_fingerprint), np.nan, dtype=np.float64),
            where=season_fingerprint["core_count"].to_numpy() > 0,
        )

    hand_mode = stable_mode(
        frame,
        keys=[TRACK_ID],
        value="hand_normalized",
        output_value="hand_normalized",
        count_name="hand_mode_rows",
    )
    hand_unique = (
        frame.groupby(TRACK_ID, observed=True)["hand_normalized"]
        .nunique(dropna=False)
        .rename("hand_unique_values")
        .reset_index()
    )
    pitcher_rows = (
        frame.groupby(TRACK_ID, observed=True)
        .size()
        .rename("trackman_rows")
        .reset_index()
    )
    pitcher_meta = (
        pitcher_rows.merge(hand_mode, on=TRACK_ID, validate="one_to_one")
        .merge(hand_unique, on=TRACK_ID, validate="one_to_one")
    )
    team_season = stable_mode(
        frame,
        keys=[TRACK_ID, "season"],
        value="track_team_family",
        output_value="track_team_family",
        count_name="track_team_rows",
    )
    audit = {
        "rows": int(len(frame)),
        "pitchers": int(frame[TRACK_ID].nunique()),
        "core_rows": int(len(core)),
        "other_or_missing_rows": int(len(frame) - len(core)),
        "core_fraction": float(len(core) / len(frame)),
        "unknown_hand_rows": int(frame["hand_normalized"].eq("Unknown").sum()),
        "multi_hand_pitchers": int(pitcher_meta["hand_unique_values"].gt(1).sum()),
        "raw_teams": int(frame["pitcher_team"].nunique(dropna=True)),
        "normalized_team_families": int(frame["track_team_family"].nunique()),
    }
    return season_fingerprint, pitcher_meta, team_season, audit


def fingerprint_arrays(
    season_frame: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    id_column: str,
    row_count_column: str,
) -> dict:
    ids = meta[id_column].astype("string").to_numpy()
    id_to_index = {value: index for index, value in enumerate(ids)}
    rates = np.full((len(ids), len(SEASONS), len(PITCH_GROUPS)), np.nan)
    total_count = np.zeros((len(ids), len(SEASONS)), dtype=np.float64)
    core_count = np.zeros((len(ids), len(SEASONS)), dtype=np.float64)
    season_to_index = {season: index for index, season in enumerate(SEASONS)}
    rate_names = [f"{group}_rate" for group in PITCH_GROUPS]
    for row in season_frame.itertuples(index=False):
        entity = str(getattr(row, id_column))
        entity_index = id_to_index[entity]
        season_index = season_to_index[int(row.season)]
        rates[entity_index, season_index, :] = [
            float(getattr(row, name)) for name in rate_names
        ]
        total_count[entity_index, season_index] = float(row.total_count)
        core_count[entity_index, season_index] = float(row.core_count)
    return {
        "ids": ids,
        "hands": meta["hand_normalized"].astype("string").to_numpy(),
        "rows": meta[row_count_column].to_numpy(dtype=np.int64),
        "rates": rates,
        "total_count": total_count,
        "core_count": core_count,
    }


def compute_metric_matrices(
    main: dict,
    track: dict,
    *,
    season_indices: list[int],
    min_pitches: int,
) -> dict[str, np.ndarray]:
    n_main = len(main["ids"])
    n_track = len(track["ids"])
    score = np.full((n_main, n_track), np.inf, dtype=np.float64)
    rmse_matrix = np.full_like(score, np.nan)
    count_mae_matrix = np.full_like(score, np.nan)
    jaccard_matrix = np.full_like(score, np.nan)
    recall_matrix = np.full_like(score, np.nan)
    overlap_matrix = np.zeros((n_main, n_track), dtype=np.int8)

    track_total = track["total_count"][:, season_indices]
    track_core = track["core_count"][:, season_indices]
    track_rates = track["rates"][:, season_indices, :]
    track_active = (track_total >= min_pitches) & (track_core >= min_pitches)

    for main_index in range(n_main):
        candidate_indices = np.flatnonzero(track["hands"] == main["hands"][main_index])
        if len(candidate_indices) == 0:
            continue
        main_total = main["total_count"][main_index, season_indices]
        main_core = main["core_count"][main_index, season_indices]
        main_rates = main["rates"][main_index, season_indices, :]
        main_active = (main_total >= min_pitches) & (main_core >= min_pitches)
        candidate_active = track_active[candidate_indices]
        overlap = candidate_active & main_active[None, :]
        overlap_count = overlap.sum(axis=1)
        union_count = (candidate_active | main_active[None, :]).sum(axis=1)

        rate_difference = track_rates[candidate_indices] - main_rates[None, :, :]
        season_squared_error = np.mean(
            np.square(np.nan_to_num(rate_difference, nan=0.0)), axis=2
        )
        weights = np.sqrt(
            np.minimum(
                track_core[candidate_indices],
                main_core[None, :],
            )
        ) * overlap
        weight_sum = weights.sum(axis=1)
        weighted_error = np.nansum(season_squared_error * weights, axis=1)
        pitchmix_rmse = np.sqrt(
            np.divide(
                weighted_error,
                weight_sum,
                out=np.full(len(candidate_indices), np.nan),
                where=weight_sum > 0,
            )
        )

        count_difference = np.abs(
            np.log1p(track_total[candidate_indices]) - np.log1p(main_total[None, :])
        )
        count_log_mae = np.divide(
            np.sum(count_difference * overlap, axis=1),
            overlap_count,
            out=np.full(len(candidate_indices), np.nan),
            where=overlap_count > 0,
        )
        activity_jaccard = np.divide(
            overlap_count,
            union_count,
            out=np.zeros(len(candidate_indices), dtype=np.float64),
            where=union_count > 0,
        )
        main_active_count = int(main_active.sum())
        activity_recall = overlap_count / max(main_active_count, 1)
        candidate_score = (
            pitchmix_rmse
            + 0.06 * count_log_mae
            + 0.08 * (1.0 - activity_jaccard)
            + 0.04 * (1.0 - activity_recall)
        )
        candidate_score[overlap_count == 0] = np.inf

        score[main_index, candidate_indices] = candidate_score
        rmse_matrix[main_index, candidate_indices] = pitchmix_rmse
        count_mae_matrix[main_index, candidate_indices] = count_log_mae
        jaccard_matrix[main_index, candidate_indices] = activity_jaccard
        recall_matrix[main_index, candidate_indices] = activity_recall
        overlap_matrix[main_index, candidate_indices] = overlap_count.astype(np.int8)

    return {
        "score": score,
        "pitchmix_rmse": rmse_matrix,
        "count_log_mae": count_mae_matrix,
        "activity_jaccard": jaccard_matrix,
        "activity_recall": recall_matrix,
        "overlap_seasons": overlap_matrix,
    }


def best_indices(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = np.argmin(score, axis=1)
    value = score[np.arange(len(score)), index]
    index = np.where(np.isfinite(value), index, -1)
    return index, value


def candidate_tables(
    main: dict,
    track: dict,
    *,
    full_metrics: dict,
    early_metrics: dict,
    late_metrics: dict,
    single_season_metrics: list[dict],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score = full_metrics["score"]
    top_k = min(args.top_k, score.shape[1])
    top_order = np.argsort(score, axis=1)[:, :top_k]
    track_best_main = np.argmin(score, axis=0)
    full_best, full_best_score = best_indices(score)
    early_best, early_best_score = best_indices(early_metrics["score"])
    late_best, late_best_score = best_indices(late_metrics["score"])

    vote_available = np.zeros(len(main["ids"]), dtype=np.int16)
    vote_agree = np.zeros(len(main["ids"]), dtype=np.int16)
    for metrics in single_season_metrics:
        season_best, season_score = best_indices(metrics["score"])
        available = np.isfinite(season_score)
        vote_available += available
        vote_agree += available & (season_best == full_best)

    top_track_ids = np.where(
        full_best >= 0,
        track["ids"][np.maximum(full_best, 0)],
        "__NO_CANDIDATE__",
    )
    duplicate_count = pd.Series(top_track_ids).map(
        pd.Series(top_track_ids).value_counts()
    ).to_numpy(dtype=np.int64)

    candidate_rows = []
    mapping_rows = []
    for main_index, main_id in enumerate(main["ids"]):
        ordered = top_order[main_index]
        finite_ordered = ordered[np.isfinite(score[main_index, ordered])]
        for rank, track_index in enumerate(finite_ordered, start=1):
            candidate_rows.append(
                {
                    MAIN_ID: main_id,
                    "rank": rank,
                    TRACK_ID: track["ids"][track_index],
                    "score": float(score[main_index, track_index]),
                    "pitchmix_rmse": float(
                        full_metrics["pitchmix_rmse"][main_index, track_index]
                    ),
                    "count_log_mae": float(
                        full_metrics["count_log_mae"][main_index, track_index]
                    ),
                    "activity_jaccard": float(
                        full_metrics["activity_jaccard"][main_index, track_index]
                    ),
                    "activity_recall": float(
                        full_metrics["activity_recall"][main_index, track_index]
                    ),
                    "overlap_seasons": int(
                        full_metrics["overlap_seasons"][main_index, track_index]
                    ),
                }
            )
        if len(finite_ordered) == 0:
            mapping_rows.append(
                {
                    MAIN_ID: main_id,
                    TRACK_ID: pd.NA,
                    "main_rows": int(main["rows"][main_index]),
                    "score": np.nan,
                    "score_margin": np.nan,
                    "mutual_top1": False,
                    "top1_duplicate_count": 0,
                    "early_late_available": False,
                    "early_late_stable": False,
                    "season_vote_available": int(vote_available[main_index]),
                    "season_vote_share": 0.0,
                }
            )
            continue
        top_index = int(finite_ordered[0])
        second_score = (
            float(score[main_index, finite_ordered[1]])
            if len(finite_ordered) > 1
            else np.inf
        )
        early_late_available = bool(
            np.isfinite(early_best_score[main_index])
            and np.isfinite(late_best_score[main_index])
        )
        early_late_stable = bool(
            early_late_available
            and early_best[main_index] == top_index
            and late_best[main_index] == top_index
        )
        mapping_rows.append(
            {
                MAIN_ID: main_id,
                TRACK_ID: track["ids"][top_index],
                "main_rows": int(main["rows"][main_index]),
                "trackman_rows": int(track["rows"][top_index]),
                "main_hand": main["hands"][main_index],
                "trackman_hand": track["hands"][top_index],
                "score": float(score[main_index, top_index]),
                "score_margin": float(second_score - score[main_index, top_index]),
                "pitchmix_rmse": float(
                    full_metrics["pitchmix_rmse"][main_index, top_index]
                ),
                "count_log_mae": float(
                    full_metrics["count_log_mae"][main_index, top_index]
                ),
                "activity_jaccard": float(
                    full_metrics["activity_jaccard"][main_index, top_index]
                ),
                "activity_recall": float(
                    full_metrics["activity_recall"][main_index, top_index]
                ),
                "overlap_seasons": int(
                    full_metrics["overlap_seasons"][main_index, top_index]
                ),
                "mutual_top1": bool(track_best_main[top_index] == main_index),
                "top1_duplicate_count": int(duplicate_count[main_index]),
                "early_top1": (
                    track["ids"][early_best[main_index]]
                    if early_best[main_index] >= 0
                    else pd.NA
                ),
                "late_top1": (
                    track["ids"][late_best[main_index]]
                    if late_best[main_index] >= 0
                    else pd.NA
                ),
                "early_late_available": early_late_available,
                "early_late_stable": early_late_stable,
                "season_vote_available": int(vote_available[main_index]),
                "season_vote_share": float(
                    vote_agree[main_index] / max(vote_available[main_index], 1)
                ),
            }
        )
    return pd.DataFrame(candidate_rows), pd.DataFrame(mapping_rows)


def add_fingerprint_confidence(
    mapping: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    mapping = mapping.copy()
    mapping["fingerprint_high_confidence"] = (
        mapping[TRACK_ID].notna()
        & mapping["mutual_top1"]
        & mapping["top1_duplicate_count"].eq(1)
        & mapping["early_late_stable"]
        & mapping["overlap_seasons"].ge(args.min_overlap_seasons)
        & mapping["pitchmix_rmse"].le(args.max_pitchmix_rmse)
        & mapping["score_margin"].ge(args.min_score_margin)
        & mapping["activity_jaccard"].ge(args.min_activity_jaccard)
        & mapping["activity_recall"].ge(args.min_activity_recall)
        & mapping["season_vote_available"].ge(args.min_overlap_seasons)
        & mapping["season_vote_share"].ge(args.min_season_vote_share)
    )
    return mapping


def infer_team_season_mapping(
    mapping: pd.DataFrame,
    main_team: pd.DataFrame,
    track_team: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trusted = mapping.loc[
        mapping["fingerprint_high_confidence"], [MAIN_ID, TRACK_ID]
    ]
    evidence = (
        trusted.merge(main_team, on=MAIN_ID, validate="one_to_many")
        .merge(
            track_team,
            on=[TRACK_ID, "season"],
            validate="many_to_one",
        )
    )
    if evidence.empty:
        empty_mapping = pd.DataFrame(
            columns=[
                "main_team_id",
                "season",
                "inferred_track_team_family",
                "support_pairs",
                "support_weight",
                "purity",
                "reliable_team_mapping",
            ]
        )
        return empty_mapping, evidence
    evidence["support_weight"] = np.sqrt(
        evidence["main_team_rows"] * evidence["track_team_rows"]
    )
    votes = (
        evidence.groupby(
            ["main_team_id", "season", "track_team_family"], observed=True
        )
        .agg(
            support_pairs=(MAIN_ID, "nunique"),
            support_weight=("support_weight", "sum"),
        )
        .reset_index()
    )
    totals = (
        votes.groupby(["main_team_id", "season"], observed=True)["support_weight"]
        .sum()
        .rename("total_support_weight")
        .reset_index()
    )
    ranked = (
        votes.sort_values(
            ["main_team_id", "season", "support_weight", "track_team_family"],
            ascending=[True, True, False, True],
        )
        .groupby(["main_team_id", "season"], as_index=False, sort=False)
        .head(1)
        .merge(totals, on=["main_team_id", "season"], validate="one_to_one")
        .rename(columns={"track_team_family": "inferred_track_team_family"})
    )
    ranked["purity"] = ranked["support_weight"] / ranked["total_support_weight"]
    ranked["reliable_team_mapping"] = (
        ranked["support_pairs"].ge(2) & ranked["purity"].ge(0.75)
    )
    return ranked, evidence


def team_family_arrays(
    main: dict,
    track: dict,
    main_team: pd.DataFrame,
    track_team: pd.DataFrame,
    team_mapping: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    main_family = np.full((len(main["ids"]), len(SEASONS)), None, dtype=object)
    track_family = np.full((len(track["ids"]), len(SEASONS)), None, dtype=object)
    main_index = {value: index for index, value in enumerate(main["ids"])}
    track_index = {value: index for index, value in enumerate(track["ids"])}
    season_index = {season: index for index, season in enumerate(SEASONS)}
    reliable = team_mapping[team_mapping["reliable_team_mapping"]].copy()
    reliable_lookup = {
        (str(row.main_team_id), int(row.season)): row.inferred_track_team_family
        for row in reliable.itertuples(index=False)
    }
    for row in main_team.itertuples(index=False):
        expected = reliable_lookup.get((str(row.main_team_id), int(row.season)))
        if expected is not None:
            main_family[
                main_index[str(getattr(row, MAIN_ID))],
                season_index[int(row.season)],
            ] = expected
    for row in track_team.itertuples(index=False):
        track_family[
            track_index[str(getattr(row, TRACK_ID))],
            season_index[int(row.season)],
        ] = row.track_team_family
    return main_family, track_family


def add_team_score_penalty(
    metrics: dict[str, np.ndarray],
    *,
    main_family: np.ndarray,
    track_family: np.ndarray,
    season_indices: list[int],
    mismatch_penalty: float = 0.20,
) -> dict[str, np.ndarray]:
    result = dict(metrics)
    adjusted = metrics["score"].copy()
    evidence_matrix = np.zeros_like(adjusted, dtype=np.int8)
    match_rate_matrix = np.full_like(adjusted, np.nan)
    observed = track_family[:, season_indices]
    observed_valid = observed != None  # noqa: E711 - object array sentinel
    for main_index in range(len(main_family)):
        expected = main_family[main_index, season_indices]
        expected_valid = expected != None  # noqa: E711 - object array sentinel
        evidence = observed_valid & expected_valid[None, :]
        evidence_count = evidence.sum(axis=1)
        matches = (observed == expected[None, :]) & evidence
        match_rate = np.divide(
            matches.sum(axis=1),
            evidence_count,
            out=np.full(len(observed), np.nan),
            where=evidence_count > 0,
        )
        has_evidence = evidence_count > 0
        adjusted[main_index, has_evidence] += mismatch_penalty * (
            1.0 - match_rate[has_evidence]
        )
        adjusted[main_index, has_evidence & (match_rate < 0.50)] = np.inf
        evidence_matrix[main_index] = evidence_count.astype(np.int8)
        match_rate_matrix[main_index] = match_rate
    result["score"] = adjusted
    result["team_evidence_seasons"] = evidence_matrix
    result["team_match_rate"] = match_rate_matrix
    return result


def add_team_consistency(
    mapping: pd.DataFrame,
    main_team: pd.DataFrame,
    track_team: pd.DataFrame,
    team_mapping: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = mapping.copy()
    pair_seasons = (
        mapping[[MAIN_ID, TRACK_ID]]
        .dropna()
        .merge(main_team, on=MAIN_ID, validate="one_to_many")
        .merge(track_team, on=[TRACK_ID, "season"], validate="many_to_one")
        .merge(
            team_mapping[
                [
                    "main_team_id",
                    "season",
                    "inferred_track_team_family",
                    "purity",
                    "reliable_team_mapping",
                ]
            ],
            on=["main_team_id", "season"],
            how="left",
            validate="many_to_one",
        )
    )
    if pair_seasons.empty:
        mapping["team_evidence_seasons"] = 0
        mapping["team_match_rate"] = np.nan
    else:
        reliable_values = (
            pair_seasons["reliable_team_mapping"]
            .astype("boolean")
            .fillna(False)
            .to_numpy(dtype=bool)
        )
        pair_seasons["team_match"] = (
            pair_seasons["track_team_family"]
            == pair_seasons["inferred_track_team_family"]
        ) & reliable_values
        pair_seasons["team_evidence"] = reliable_values
        pair_summary = (
            pair_seasons.groupby(MAIN_ID, observed=True)
            .agg(
                team_evidence_seasons=("team_evidence", "sum"),
                team_matches=("team_match", "sum"),
            )
            .reset_index()
        )
        pair_summary["team_match_rate"] = np.divide(
            pair_summary["team_matches"],
            pair_summary["team_evidence_seasons"],
            out=np.full(len(pair_summary), np.nan),
            where=pair_summary["team_evidence_seasons"].to_numpy() > 0,
        )
        mapping = mapping.merge(
            pair_summary[[MAIN_ID, "team_evidence_seasons", "team_match_rate"]],
            on=MAIN_ID,
            how="left",
            validate="one_to_one",
        )
        mapping["team_evidence_seasons"] = mapping[
            "team_evidence_seasons"
        ].fillna(0).astype(int)
    mapping["high_confidence"] = (
        mapping["fingerprint_high_confidence"]
        & mapping["team_evidence_seasons"].ge(args.min_team_evidence_seasons)
        & mapping["team_match_rate"].ge(args.min_team_match_rate)
    )
    return mapping, pair_seasons


def coverage_by_season(
    main_season: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    marked = main_season.merge(
        mapping[[MAIN_ID, "high_confidence"]],
        on=MAIN_ID,
        how="left",
        validate="many_to_one",
    )
    marked["mapped_rows"] = marked["total_count"] * marked[
        "high_confidence"
    ].fillna(False).astype(int)
    result = (
        marked.groupby("season", observed=True)
        .agg(total_rows=("total_count", "sum"), mapped_rows=("mapped_rows", "sum"))
        .reset_index()
    )
    result["row_coverage"] = result["mapped_rows"] / result["total_rows"]
    return result


def threshold_sensitivity(
    mapping: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    common = (
        mapping[TRACK_ID].notna()
        & mapping["mutual_top1"]
        & mapping["top1_duplicate_count"].eq(1)
        & mapping["early_late_stable"]
        & mapping["overlap_seasons"].ge(args.min_overlap_seasons)
        & mapping["activity_jaccard"].ge(args.min_activity_jaccard)
        & mapping["activity_recall"].ge(args.min_activity_recall)
        & mapping["season_vote_available"].ge(args.min_overlap_seasons)
        & mapping["season_vote_share"].ge(args.min_season_vote_share)
        & mapping["team_evidence_seasons"].ge(args.min_team_evidence_seasons)
        & mapping["team_match_rate"].ge(args.min_team_match_rate)
    )
    total_rows = mapping["main_rows"].sum()
    for max_rmse in (0.035, 0.045, 0.055, 0.070):
        for min_margin in (0.005, 0.010, 0.015, 0.025):
            selected = common & mapping["pitchmix_rmse"].le(max_rmse) & mapping[
                "score_margin"
            ].ge(min_margin)
            rows.append(
                {
                    "max_pitchmix_rmse": max_rmse,
                    "min_score_margin": min_margin,
                    "mapped_pitchers": int(selected.sum()),
                    "pitcher_coverage": float(selected.mean()),
                    "row_coverage": float(
                        mapping.loc[selected, "main_rows"].sum() / total_rows
                    ),
                    "median_vote_share": (
                        float(mapping.loc[selected, "season_vote_share"].median())
                        if selected.any()
                        else np.nan
                    ),
                    "median_team_match_rate": (
                        float(mapping.loc[selected, "team_match_rate"].median())
                        if selected.any()
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def decision_summary(
    mapping: pd.DataFrame,
    coverage: pd.DataFrame,
    team_mapping: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    selected = mapping["high_confidence"]
    total_rows = int(mapping["main_rows"].sum())
    mapped_rows = int(mapping.loc[selected, "main_rows"].sum())
    row_coverage = mapped_rows / max(total_rows, 1)
    median_vote = (
        float(mapping.loc[selected, "season_vote_share"].median())
        if selected.any()
        else 0.0
    )
    median_team = (
        float(mapping.loc[selected, "team_match_rate"].median())
        if selected.any()
        else 0.0
    )
    reliable_team_fraction = (
        float(team_mapping["reliable_team_mapping"].mean())
        if len(team_mapping)
        else 0.0
    )
    duplicate_track_ids = int(
        mapping.loc[selected, TRACK_ID].astype("string").duplicated().sum()
    )
    proceed = (
        row_coverage >= args.proceed_min_row_coverage
        and median_vote >= args.proceed_min_median_vote_share
        and median_team >= args.proceed_min_median_team_match
        and duplicate_track_ids == 0
    )
    if proceed:
        decision = "proceed_high_confidence_only"
        next_step = (
            "Run Trackman feature CV using only high-confidence mapped pitchers; "
            "all other rows must use missing indicators and global fallbacks."
        )
    elif row_coverage >= args.partial_min_row_coverage and duplicate_track_ids == 0:
        decision = "partial_mapping_experiment_only"
        next_step = (
            "Do not use mapping in a final submission yet. Run a mapped-only/fallback "
            "OOF ablation and require fold-safe improvement before reconsidering."
        )
    else:
        decision = "reject_player_level_mapping"
        next_step = (
            "Do not attach player-level Trackman features. Use only league/season/hand "
            "aggregates or move to the F-shift experiment."
        )
    return {
        "decision": decision,
        "next_step": next_step,
        "high_confidence_pitchers": int(selected.sum()),
        "total_main_pitchers": int(len(mapping)),
        "pitcher_coverage": float(selected.mean()),
        "mapped_main_rows": mapped_rows,
        "total_main_rows": total_rows,
        "row_coverage": float(row_coverage),
        "minimum_season_row_coverage": (
            float(coverage["row_coverage"].min()) if len(coverage) else 0.0
        ),
        "median_pitchmix_rmse": (
            float(mapping.loc[selected, "pitchmix_rmse"].median())
            if selected.any()
            else None
        ),
        "median_score_margin": (
            float(mapping.loc[selected, "score_margin"].median())
            if selected.any()
            else None
        ),
        "median_season_vote_share": median_vote,
        "median_team_match_rate": median_team,
        "reliable_team_season_mapping_fraction": reliable_team_fraction,
        "duplicate_trackman_ids": duplicate_track_ids,
        "identity_warning": (
            "A high-confidence row is a reproducible fingerprint match, not ground-truth "
            "identity. It may be used only after forward OOF ablation confirms benefit."
        ),
    }


def main() -> None:
    args = parse_args()
    if not args.train_path.exists():
        raise FileNotFoundError(args.train_path)
    if not args.trackman_path.exists():
        raise FileNotFoundError(args.trackman_path)

    configuration = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "seasons": list(SEASONS),
        "early_seasons": list(EARLY_SEASONS),
        "late_seasons": list(LATE_SEASONS),
        "pitch_groups": list(PITCH_GROUPS),
        "main_columns": MAIN_COLUMNS,
        "trackman_columns": TRACK_COLUMNS,
        "target_column_read": False,
        "external_data_used": False,
        "min_pitches_per_season": args.min_pitches_per_season,
        "top_k": args.top_k,
        "confidence_thresholds": {
            "min_overlap_seasons": args.min_overlap_seasons,
            "max_pitchmix_rmse": args.max_pitchmix_rmse,
            "min_score_margin": args.min_score_margin,
            "min_activity_jaccard": args.min_activity_jaccard,
            "min_activity_recall": args.min_activity_recall,
            "min_season_vote_share": args.min_season_vote_share,
            "min_team_match_rate": args.min_team_match_rate,
            "min_team_evidence_seasons": args.min_team_evidence_seasons,
        },
        "decision_thresholds": {
            "proceed_min_row_coverage": args.proceed_min_row_coverage,
            "partial_min_row_coverage": args.partial_min_row_coverage,
            "proceed_min_median_vote_share": args.proceed_min_median_vote_share,
            "proceed_min_median_team_match": args.proceed_min_median_team_match,
        },
        "score_formula": (
            "pitchmix_rmse + 0.06*count_log_mae + "
            "0.08*(1-activity_jaccard) + 0.04*(1-activity_recall)"
        ),
        "two_stage_team_audit": {
            "enabled": True,
            "description": (
                "Infer anonymous team-season mappings only from strict fingerprint "
                "seeds, then add a 0.20 mismatch penalty and reject candidates with "
                "team match rate below 0.50 before repeating temporal stability tests."
            ),
            "mismatch_penalty": 0.20,
            "hard_reject_below_match_rate": 0.50,
        },
        "team_family_map": TEAM_FAMILY_MAP,
        "source_script_sha256": sha256_file(Path(__file__)),
        "input": {
            "train_path": str(args.train_path.resolve()),
            "train_bytes": args.train_path.stat().st_size,
            "trackman_path": str(args.trackman_path.resolve()),
            "trackman_bytes": args.trackman_path.stat().st_size,
        },
    }
    signature = experiment_signature(configuration)
    run_dir = args.output_dir / f"run_{signature}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "configuration.json", configuration)

    main_season, main_meta, main_team, main_audit = build_main_fingerprints(
        args.train_path
    )
    track_season, track_meta, track_team, track_audit = build_trackman_fingerprints(
        args.trackman_path
    )
    main_arrays = fingerprint_arrays(
        main_season,
        main_meta,
        id_column=MAIN_ID,
        row_count_column="main_rows",
    )
    track_arrays = fingerprint_arrays(
        track_season,
        track_meta,
        id_column=TRACK_ID,
        row_count_column="trackman_rows",
    )
    season_index = {season: index for index, season in enumerate(SEASONS)}
    full_metrics = compute_metric_matrices(
        main_arrays,
        track_arrays,
        season_indices=list(range(len(SEASONS))),
        min_pitches=args.min_pitches_per_season,
    )
    early_metrics = compute_metric_matrices(
        main_arrays,
        track_arrays,
        season_indices=[season_index[value] for value in EARLY_SEASONS],
        min_pitches=args.min_pitches_per_season,
    )
    late_metrics = compute_metric_matrices(
        main_arrays,
        track_arrays,
        season_indices=[season_index[value] for value in LATE_SEASONS],
        min_pitches=args.min_pitches_per_season,
    )
    single_season_metrics = [
        compute_metric_matrices(
            main_arrays,
            track_arrays,
            season_indices=[season_index[season]],
            min_pitches=args.min_pitches_per_season,
        )
        for season in SEASONS
    ]
    seed_candidates, seed_mapping = candidate_tables(
        main_arrays,
        track_arrays,
        full_metrics=full_metrics,
        early_metrics=early_metrics,
        late_metrics=late_metrics,
        single_season_metrics=single_season_metrics,
        args=args,
    )
    seed_mapping = add_fingerprint_confidence(seed_mapping, args)
    seed_team_mapping, seed_team_evidence = infer_team_season_mapping(
        seed_mapping, main_team, track_team
    )
    main_family, track_family = team_family_arrays(
        main_arrays,
        track_arrays,
        main_team,
        track_team,
        seed_team_mapping,
    )
    team_full_metrics = add_team_score_penalty(
        full_metrics,
        main_family=main_family,
        track_family=track_family,
        season_indices=list(range(len(SEASONS))),
    )
    team_early_metrics = add_team_score_penalty(
        early_metrics,
        main_family=main_family,
        track_family=track_family,
        season_indices=[season_index[value] for value in EARLY_SEASONS],
    )
    team_late_metrics = add_team_score_penalty(
        late_metrics,
        main_family=main_family,
        track_family=track_family,
        season_indices=[season_index[value] for value in LATE_SEASONS],
    )
    team_single_season_metrics = [
        add_team_score_penalty(
            metrics,
            main_family=main_family,
            track_family=track_family,
            season_indices=[season_index[season]],
        )
        for metrics, season in zip(single_season_metrics, SEASONS)
    ]
    candidates, mapping = candidate_tables(
        main_arrays,
        track_arrays,
        full_metrics=team_full_metrics,
        early_metrics=team_early_metrics,
        late_metrics=team_late_metrics,
        single_season_metrics=team_single_season_metrics,
        args=args,
    )
    mapping = add_fingerprint_confidence(mapping, args)
    team_mapping, team_evidence = infer_team_season_mapping(
        mapping, main_team, track_team
    )
    mapping, pair_team_seasons = add_team_consistency(
        mapping,
        main_team,
        track_team,
        team_mapping,
        args,
    )
    coverage = coverage_by_season(main_season, mapping)
    sensitivity = threshold_sensitivity(mapping, args)
    decision = decision_summary(mapping, coverage, team_mapping, args)

    input_hashes = {
        "train_sha256": sha256_file(args.train_path),
        "trackman_sha256": sha256_file(args.trackman_path),
    }
    reconstruction_audit = {
        "main": main_audit,
        "trackman": track_audit,
        "input_hashes": input_hashes,
        "input_hashes": input_hashes,
    }
    summary = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "signature": signature,
        "decision": decision,
        "main_reconstruction": main_audit,
        "trackman": track_audit,
        "seed_fingerprint_high_confidence_pitchers": int(
            seed_mapping["fingerprint_high_confidence"].sum()
        ),
        "seed_team_evidence_rows": int(len(seed_team_evidence)),
        "fingerprint_high_confidence_pitchers": int(
            mapping["fingerprint_high_confidence"].sum()
        ),
        "final_high_confidence_pitchers": int(mapping["high_confidence"].sum()),
        "candidate_rows": int(len(candidates)),
        "team_evidence_rows": int(len(team_evidence)),
        "pair_team_season_rows": int(len(pair_team_seasons)),
        "selection_note": (
            "No target was read. Thresholds were fixed before model CV. The mapping "
            "must still pass a future forward-OOF feature ablation."
        ),
    }

    main_season.to_csv(run_dir / "main_pitcher_season_fingerprint.csv", index=False)
    track_season.to_csv(
        run_dir / "trackman_pitcher_season_fingerprint.csv", index=False
    )
    seed_candidates.to_csv(run_dir / "seed_candidate_topk.csv", index=False)
    seed_mapping.sort_values(
        ["fingerprint_high_confidence", "main_rows"], ascending=[False, False]
    ).to_csv(run_dir / "seed_pitcher_mapping_audit.csv", index=False)
    seed_team_mapping.to_csv(run_dir / "seed_team_season_mapping.csv", index=False)
    candidates.to_csv(run_dir / "candidate_topk.csv", index=False)
    mapping.sort_values(
        ["high_confidence", "main_rows"], ascending=[False, False]
    ).to_csv(run_dir / "pitcher_mapping_audit.csv", index=False)
    team_mapping.to_csv(run_dir / "team_season_mapping.csv", index=False)
    coverage.to_csv(run_dir / "coverage_by_season.csv", index=False)
    sensitivity.to_csv(run_dir / "threshold_sensitivity.csv", index=False)
    write_json(run_dir / "reconstruction_audit.json", reconstruction_audit)
    write_json(run_dir / "summary.json", summary)

    latest = {
        "run_dir": str(run_dir.resolve()),
        "summary": summary,
        "configuration": configuration,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "latest_run.json", latest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved Trackman linkage audit: {run_dir}")


if __name__ == "__main__":
    main()
