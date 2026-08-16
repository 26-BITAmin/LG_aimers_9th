"""Leakage-safe repair of pre-2023 F-game cumulative history features.

The operational definition of the ``reverse`` failure mode changed sharply in
2023 for F games.  Career ``asof_*`` rates still contain the older convention,
so later rows inherit a measurement error even when ``game_type`` itself is
removed from the model.

This module estimates the missing reverse mass from a training fold only and
stores it as immutable lookup arrays.  Applying an artifact to validation or
test rows performs independent row-wise lookups; it never aggregates those
rows or reads their targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


KEY_MULTIPLIER = 10_000_000
DIRTY_LAST_SEASON = 2022


@dataclass(frozen=True)
class AxisLookup:
    """Sorted contaminated-pitch positions and their prefix missing mass."""

    keys: np.ndarray
    cumulative_mass: np.ndarray


@dataclass(frozen=True)
class FContaminationArtifact:
    """Training-fold artifact used by both training and inference features."""

    q_by_season: dict[int, float]
    pitcher: AxisLookup
    batter: AxisLookup
    recovery_audit: dict[str, float | int]


def _as_int64(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any():
        raise ValueError(f"{values.name} contains missing values")
    return numeric.to_numpy(dtype=np.int64)


def recover_increment_labels(
    frame: pd.DataFrame,
    *,
    id_column: str,
    count_column: str,
    rate_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the current binary label from the next cumulative as-of row.

    If a row has career count ``n`` before the current pitch, the next row for
    the same player has count ``n + 1`` and contains the cumulative rate after
    the current pitch.  The last observed pitch for each player is intentionally
    left unrecovered because no future cumulative row is available.
    """

    required = {id_column, count_column, rate_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"increment recovery columns are missing: {missing}")

    player = _as_int64(frame[id_column])
    count = _as_int64(frame[count_column])
    rate = pd.to_numeric(frame[rate_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    order = np.lexsort((count, player))
    sorted_player = player[order]
    sorted_count = count[order]
    sorted_rate = rate[order]

    valid_pair = (
        (sorted_player[:-1] == sorted_player[1:])
        & (sorted_count[1:] == sorted_count[:-1] + 1)
        & np.isfinite(sorted_rate[1:])
    )
    before_total = np.where(
        sorted_count[:-1] > 0,
        sorted_count[:-1] * np.nan_to_num(sorted_rate[:-1], nan=0.0),
        0.0,
    )
    after_total = sorted_count[1:] * sorted_rate[1:]
    delta = after_total - before_total

    labels = np.full(len(frame), np.nan, dtype=np.float64)
    recovered_positions = order[:-1][valid_pair]
    labels[recovered_positions] = np.rint(delta[valid_pair])
    recovered = np.isfinite(labels)

    if recovered.any():
        raw = delta[valid_pair]
        max_rounding_error = float(np.max(np.abs(raw - np.rint(raw))))
        invalid = (labels[recovered] < 0.0) | (labels[recovered] > 1.0)
        if invalid.any() or max_rounding_error > 0.05:
            raise RuntimeError(
                f"{rate_column} increments are not binary: "
                f"invalid={int(invalid.sum())}, max_rounding_error={max_rounding_error:.6f}"
            )
    labels[recovered] = np.clip(labels[recovered], 0.0, 1.0)
    return labels, recovered


def _build_axis_lookup(
    player: np.ndarray,
    count: np.ndarray,
    per_pitch_mass: np.ndarray,
) -> AxisLookup:
    dirty = per_pitch_mass > 0.0
    if not dirty.any():
        return AxisLookup(
            keys=np.empty(0, dtype=np.int64),
            cumulative_mass=np.zeros(1, dtype=np.float64),
        )
    selected_count = count[dirty].astype(np.int64, copy=False)
    if selected_count.max(initial=0) >= KEY_MULTIPLIER:
        raise ValueError("as-of count exceeds the lookup key multiplier")
    keys = (
        player[dirty].astype(np.int64, copy=False) * KEY_MULTIPLIER
        + selected_count
    )
    if len(np.unique(keys)) != len(keys):
        raise RuntimeError("duplicate player/count positions in contamination lookup")
    order = np.argsort(keys, kind="stable")
    ordered_mass = per_pitch_mass[dirty][order].astype(np.float64, copy=False)
    cumulative = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(ordered_mass, dtype=np.float64)]
    )
    return AxisLookup(keys=keys[order], cumulative_mass=cumulative)


def fit_f_contamination(frame: pd.DataFrame) -> FContaminationArtifact:
    """Fit missing reverse mass using one forward-CV training fold only."""

    required = {
        "season",
        "game_type",
        "pitcher_id",
        "batter_id",
        "asof_pitcher_n",
        "asof_batter_n",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_success_rate",
        "control_success",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"F-contamination fit columns are missing: {missing}")

    reverse_label, reverse_mask = recover_increment_labels(
        frame,
        id_column="pitcher_id",
        count_column="asof_pitcher_n",
        rate_column="asof_pitcher_reverse_rate",
    )
    success_label, success_mask = recover_increment_labels(
        frame,
        id_column="pitcher_id",
        count_column="asof_pitcher_n",
        rate_column="asof_pitcher_success_rate",
    )
    target = frame["control_success"].to_numpy(dtype=np.int8)
    comparable = success_mask & np.isfinite(success_label)
    agreement = float(np.mean(success_label[comparable] == target[comparable]))
    if agreement < 0.999:
        raise RuntimeError(
            f"cumulative success reconstruction disagrees with target: {agreement:.6%}"
        )

    season = frame["season"].to_numpy(dtype=np.int16)
    game_type = frame["game_type"].astype("string").to_numpy()
    q_by_season: dict[int, float] = {}
    for value in sorted(int(x) for x in np.unique(season) if x <= DIRTY_LAST_SEASON):
        regular = (season == value) & (game_type == "R") & reverse_mask
        final = (season == value) & (game_type == "F") & reverse_mask
        if not regular.any() or not final.any():
            continue
        q_by_season[value] = max(
            0.0,
            float(reverse_label[regular].mean() - reverse_label[final].mean()),
        )

    per_pitch_mass = np.zeros(len(frame), dtype=np.float64)
    dirty = (season <= DIRTY_LAST_SEASON) & (game_type == "F")
    for value, q_value in q_by_season.items():
        per_pitch_mass[dirty & (season == value)] = q_value

    artifact = FContaminationArtifact(
        q_by_season=q_by_season,
        pitcher=_build_axis_lookup(
            _as_int64(frame["pitcher_id"]),
            _as_int64(frame["asof_pitcher_n"]),
            per_pitch_mass,
        ),
        batter=_build_axis_lookup(
            _as_int64(frame["batter_id"]),
            _as_int64(frame["asof_batter_n"]),
            per_pitch_mass,
        ),
        recovery_audit={
            "rows": int(len(frame)),
            "reverse_recovered": int(reverse_mask.sum()),
            "reverse_recovery_fraction": float(reverse_mask.mean()),
            "success_recovered": int(success_mask.sum()),
            "success_recovery_fraction": float(success_mask.mean()),
            "success_target_agreement": agreement,
            "dirty_rows": int(dirty.sum()),
            "estimated_missing_reverse_mass": float(per_pitch_mass.sum()),
        },
    )
    return artifact


def _lookup_adjustment(
    lookup: AxisLookup,
    player: np.ndarray,
    count: np.ndarray,
) -> np.ndarray:
    if lookup.keys.size == 0:
        return np.zeros(len(player), dtype=np.float64)
    if count.max(initial=0) >= KEY_MULTIPLIER:
        raise ValueError("as-of count exceeds the lookup key multiplier")
    base = player.astype(np.int64, copy=False) * KEY_MULTIPLIER
    start = np.searchsorted(lookup.keys, base, side="left")
    stop = np.searchsorted(
        lookup.keys,
        base + count.astype(np.int64, copy=False),
        side="left",
    )
    missing_mass = lookup.cumulative_mass[stop] - lookup.cumulative_mass[start]
    return np.divide(
        missing_mass,
        np.maximum(count, 1),
        out=np.zeros_like(missing_mass),
        where=count > 0,
    )


def correction_amount(
    frame: pd.DataFrame,
    artifact: FContaminationArtifact,
    *,
    axis: str,
) -> np.ndarray:
    """Return the row-independent missing-mass fraction for one player axis."""

    if axis == "pitcher":
        id_column = "pitcher_id"
        count_column = "asof_pitcher_n"
        lookup = artifact.pitcher
    elif axis == "batter":
        id_column = "batter_id"
        count_column = "asof_batter_n"
        lookup = artifact.batter
    else:
        raise ValueError(f"unknown axis: {axis}")
    return _lookup_adjustment(
        lookup,
        _as_int64(frame[id_column]),
        _as_int64(frame[count_column]),
    )


def apply_f_history_correction(
    frame: pd.DataFrame,
    artifact: FContaminationArtifact,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Return a corrected copy and diagnostics for the affected career rates."""

    out = frame.copy()
    pitcher_adjustment = correction_amount(out, artifact, axis="pitcher")
    batter_adjustment = correction_amount(out, artifact, axis="batter")

    original_pitcher_success = pd.to_numeric(
        out["asof_pitcher_success_rate"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    original_pitcher_reverse = pd.to_numeric(
        out["asof_pitcher_reverse_rate"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    original_batter_success = pd.to_numeric(
        out["asof_batter_success_rate"], errors="coerce"
    ).to_numpy(dtype=np.float64)

    out["asof_pitcher_success_rate"] = np.where(
        np.isfinite(original_pitcher_success),
        np.clip(original_pitcher_success - pitcher_adjustment, 0.0, 1.0),
        original_pitcher_success,
    )
    out["asof_pitcher_reverse_rate"] = np.where(
        np.isfinite(original_pitcher_reverse),
        np.clip(original_pitcher_reverse + pitcher_adjustment, 0.0, 1.0),
        original_pitcher_reverse,
    )
    out["asof_batter_success_rate"] = np.where(
        np.isfinite(original_batter_success),
        np.clip(original_batter_success - batter_adjustment, 0.0, 1.0),
        original_batter_success,
    )

    audit: dict[str, float | int] = {
        "rows": int(len(out)),
        "pitcher_rows_adjusted": int(np.count_nonzero(pitcher_adjustment)),
        "batter_rows_adjusted": int(np.count_nonzero(batter_adjustment)),
        "pitcher_adjustment_mean": float(pitcher_adjustment.mean()),
        "pitcher_adjustment_std": float(pitcher_adjustment.std()),
        "pitcher_adjustment_max": float(pitcher_adjustment.max(initial=0.0)),
        "batter_adjustment_mean": float(batter_adjustment.mean()),
        "batter_adjustment_std": float(batter_adjustment.std()),
        "batter_adjustment_max": float(batter_adjustment.max(initial=0.0)),
    }
    return out, audit


def artifact_summary(artifact: FContaminationArtifact) -> dict:
    """Convert an artifact's lightweight diagnostics to JSON-safe values."""

    return {
        "q_by_season": {
            str(season): float(value)
            for season, value in artifact.q_by_season.items()
        },
        "pitcher_lookup_rows": int(artifact.pitcher.keys.size),
        "batter_lookup_rows": int(artifact.batter.keys.size),
        "pitcher_missing_mass": float(artifact.pitcher.cumulative_mass[-1]),
        "batter_missing_mass": float(artifact.batter.cumulative_mass[-1]),
        "recovery_audit": artifact.recovery_audit,
    }
