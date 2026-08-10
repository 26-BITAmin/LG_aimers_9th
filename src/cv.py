"""시즌 홀드아웃(forward chaining) 검증 + ablation.

실행:  .venv/Scripts/python.exe src/cv.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (ALL_SEASONS, FOLDS, RECENCY_HALF_LIFE, SEED, TARGET_COL,
                    TRAIN_CSV)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import brier, bss, decompose, report, shift_to_prior

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=1000,      # 신호가 약하다 → 리프를 크게 잡아 노이즈 암기를 막는다
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    max_bin=127,
    num_threads=6,
    seed=SEED,
    verbosity=-1,
)


def brier_eval(preds, dset):
    y = dset.get_label()
    return "brier", float(np.mean((preds - y) ** 2)), False


def run_fold(df, tr_seasons, va_season, *, use_recency, num_boost_round=3000):
    tr = df[df.season.isin(tr_seasons)]
    va = df[df.season == va_season]

    w = (recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=max(tr_seasons))
         if use_recency else np.ones(len(tr)))

    art = fit_artifacts(tr, weights=w)
    Xtr, Xva = build_features(tr, art), build_features(va, art)
    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)

    dtr = lgb.Dataset(Xtr, ytr, weight=w, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=False)

    booster = lgb.train(
        LGB_PARAMS, dtr, num_boost_round=num_boost_round,
        valid_sets=[dva], valid_names=["valid"], feval=brier_eval,
        callbacks=[lgb.early_stopping(150, first_metric_only=False, verbose=False),
                   lgb.log_evaluation(0)],
    )
    p = booster.predict(Xva, num_iteration=booster.best_iteration)
    return booster, art, Xtr.columns.tolist(), p, yva, tr, va


def prior_rules(hist):
    """시즌 <T 의 양성률 시계열(hist: dict season->rate)로 시즌 T를 추정하는 규칙들."""
    ss = sorted(hist)
    r = [hist[s] for s in ss]
    deltas = np.diff(r)
    return {
        "R1_last": r[-1],
        "R2_last+delta": r[-1] + deltas[-1],
        "R3_last+half_delta": r[-1] + 0.5 * deltas[-1],
        "R4_last+mean_delta": r[-1] + deltas.mean(),
        "R5_last+half_mean_delta": r[-1] + 0.5 * deltas.mean(),
    }


def main():
    t0 = time.time()
    print("Loading train.csv ...")
    df = load_csv(TRAIN_CSV)
    print(f"  {df.shape}  ({time.time()-t0:.1f}s)")

    season_rate = df.groupby("season")[TARGET_COL].mean().to_dict()
    print("\n시즌별 양성률:", {k: round(v, 4) for k, v in season_rate.items()})

    for use_recency in (False, True):
        tag = "recency-weighted" if use_recency else "uniform-weight"
        print(f"\n{'='*100}\n### {tag}\n{'='*100}")
        for tr_seasons, va_season in FOLDS:
            t1 = time.time()
            booster, art, cols, p, yva, tr, va = run_fold(
                df, tr_seasons, va_season, use_recency=use_recency)
            print(f"\n[fold] train {tr_seasons[0]}-{tr_seasons[-1]} → valid {va_season}   "
                  f"best_iter={booster.best_iteration}  ({time.time()-t1:.0f}s)")

            pbar = yva.mean()
            report("const = oracle(valid mean)", yva, np.full_like(p, pbar))
            report("const = last train season", yva,
                   np.full_like(p, season_rate[max(tr_seasons)]))
            report("LGBM raw", yva, p)
            report("LGBM + shift→oracle", yva, shift_to_prior(p, pbar))

            hist = {s: season_rate[s] for s in tr_seasons}
            for name, est in prior_rules(hist).items():
                report(f"LGBM + shift→{name}", yva, shift_to_prior(p, est),
                       extra=f"[est={est:.4f}]")

            imp = pd.Series(booster.feature_importance("gain"), index=cols)
            print("  top15 gain:", ", ".join(
                f"{k}={v:.0f}" for k, v in imp.sort_values(ascending=False).head(15).items()))

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
