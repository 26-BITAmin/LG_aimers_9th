"""Diagnostics for out-of-time probability predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import TARGET_COL
from metrics import brier_score, brier_skill_score, raw_brier_skill_score


def probability_metrics(y_true, y_prob) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(y_prob, dtype=np.float64)
    auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
    return {
        "n": int(y.size),
        "target_rate": float(y.mean()),
        "prediction_mean": float(p.mean()),
        "prediction_std": float(p.std()),
        "prediction_min": float(p.min()),
        "prediction_max": float(p.max()),
        "mean_gap": float(p.mean() - y.mean()),
        "brier": brier_score(y, p),
        "bss_raw": raw_brier_skill_score(y, p),
        "bss_official": brier_skill_score(y, p),
        "roc_auc": auc,
    }


def grouped_metrics(train_df: pd.DataFrame, valid_df: pd.DataFrame, y_prob) -> pd.DataFrame:
    """Report the segments most likely to fail under 2025 distribution shift."""
    pred = np.asarray(y_prob, dtype=np.float64)
    y = valid_df[TARGET_COL].to_numpy()
    rows: list[dict] = []

    def add(group: str, value: str, mask) -> None:
        m = np.asarray(mask, dtype=bool)
        if m.sum() == 0:
            return
        row = {"group": group, "value": value}
        row.update(probability_metrics(y[m], pred[m]))
        rows.append(row)

    add("overall", "all", np.ones(len(valid_df), dtype=bool))
    for value in sorted(valid_df["game_type"].astype(str).unique()):
        add("game_type", value, valid_df["game_type"].astype(str).eq(value))

    seen_pitchers = set(train_df["pitcher_id"].unique())
    seen_batters = set(train_df["batter_id"].unique())
    pitcher_seen = valid_df["pitcher_id"].isin(seen_pitchers).to_numpy()
    batter_seen = valid_df["batter_id"].isin(seen_batters).to_numpy()
    add("pitcher_history", "seen", pitcher_seen)
    add("pitcher_history", "unseen", ~pitcher_seen)
    add("batter_history", "seen", batter_seen)
    add("batter_history", "unseen", ~batter_seen)
    add("cold_start", "pitcher_n=0", valid_df["asof_pitcher_n"].eq(0))
    add("cold_start", "batter_n=0", valid_df["asof_batter_n"].eq(0))

    bins = [-1, 0, 50, 200, 500, 1_000, 2_500, 5_000, 10_000, np.inf]
    labels = ["0", "1-50", "51-200", "201-500", "501-1k", "1k-2.5k", "2.5k-5k", "5k-10k", "10k+"]
    bucket = pd.cut(valid_df["asof_pitcher_n"], bins=bins, labels=labels)
    for value in labels:
        add("asof_pitcher_n", value, bucket.eq(value))

    for month in sorted(valid_df["game_month"].unique()):
        add("game_month", str(month), valid_df["game_month"].eq(month))
    return pd.DataFrame(rows)


def calibration_table(y_true, y_prob, n_bins: int = 20) -> pd.DataFrame:
    data = pd.DataFrame({"target": y_true, "prediction": y_prob})
    data["bin"] = pd.qcut(data["prediction"], n_bins, duplicates="drop")
    return (
        data.groupby("bin", observed=True)
        .agg(n=("target", "size"), prediction_mean=("prediction", "mean"), target_rate=("target", "mean"))
        .reset_index(drop=True)
    )
