"""2차 실험 — 설정 ablation.

fold2(train≤2023 → valid 2024)가 실제 과제와 구조가 같으므로 주 판단 근거로 쓰고,
fold1(valid 2023)은 game_type 정의 변경 구간이라 **강건성 경고등**으로만 본다.

실행:  .venv/Scripts/python.exe -u src/ablate.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import SEED, TARGET_COL, TRAIN_CSV
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import brier, bss, shift_to_prior

BASE_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=1000,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
    max_bin=127, num_threads=6, seed=SEED, verbosity=-1,
)


def brier_eval(preds, dset):
    y = dset.get_label()
    return "brier", float(np.mean((preds - y) ** 2)), False


def r4_prior(season_rate, train_seasons):
    """R4: 직전 시즌 + 평균 시즌간 변화량."""
    r = np.array([season_rate[s] for s in sorted(train_seasons)])
    return float(r[-1] + np.diff(r).mean())


def evaluate(df, season_rate, va_season, *, train_seasons, half_life, params,
             drop_cols=(), add_ids=False, k=200.0, rounds=4000):
    tr = df[df.season.isin(train_seasons)]
    va = df[df.season == va_season]

    w = (np.ones(len(tr)) if half_life is None
         else recency_weight(tr.season.to_numpy(), half_life, ref_season=max(train_seasons)))

    art = fit_artifacts(tr, weights=w, k_pitcher=k, k_batter=k)
    Xtr, Xva = build_features(tr, art), build_features(va, art)

    if add_ids:
        for c in ("pitcher_id", "batter_id"):
            Xtr[c] = tr[c].to_numpy(dtype=np.float32)
            Xva[c] = va[c].to_numpy(dtype=np.float32)
    if drop_cols:
        Xtr = Xtr.drop(columns=list(drop_cols))
        Xva = Xva.drop(columns=list(drop_cols))

    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)

    dtr = lgb.Dataset(Xtr, ytr, weight=w, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=False)
    bst = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                    feval=brier_eval,
                    callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    p = bst.predict(Xva, num_iteration=bst.best_iteration)

    est = r4_prior(season_rate, train_seasons)
    return {
        "iter": bst.best_iteration,
        "bss_raw": bss(yva, p),
        "bss_r4": bss(yva, shift_to_prior(p, est)),
        "bss_oracle": bss(yva, shift_to_prior(p, yva.mean())),
        "n_feat": Xtr.shape[1],
    }


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    season_rate = df.groupby("season")[TARGET_COL].mean().to_dict()
    S = [2019, 2020, 2021, 2022, 2023]

    slow = dict(BASE_PARAMS, learning_rate=0.015, num_leaves=31, min_data_in_leaf=3000,
                lambda_l2=50.0)

    variants = {
        "A base (hl=1.5)":            dict(train_seasons=S, half_life=1.5, params=BASE_PARAMS),
        "B uniform":                  dict(train_seasons=S, half_life=None, params=BASE_PARAMS),
        "C hl=1.0":                   dict(train_seasons=S, half_life=1.0, params=BASE_PARAMS),
        "D hl=2.5":                   dict(train_seasons=S, half_life=2.5, params=BASE_PARAMS),
        "E 2022-23 only":             dict(train_seasons=[2022, 2023], half_life=1.5, params=BASE_PARAMS),
        "F 2023 only":                dict(train_seasons=[2023], half_life=None, params=BASE_PARAMS),
        "G no game_type":             dict(train_seasons=S, half_life=1.5, params=BASE_PARAMS,
                                           drop_cols=["game_type"]),
        "H +pitcher/batter_id":       dict(train_seasons=S, half_life=1.5, params=BASE_PARAMS,
                                           add_ids=True),
        "I slow+regularized":         dict(train_seasons=S, half_life=1.5, params=slow),
        "J slow, no game_type":       dict(train_seasons=S, half_life=1.5, params=slow,
                                           drop_cols=["game_type"]),
        "K l2 objective":             dict(train_seasons=S, half_life=1.5,
                                           params=dict(BASE_PARAMS, objective="regression")),
        "L smooth k=50":              dict(train_seasons=S, half_life=1.5, params=BASE_PARAMS, k=50.0),
        "M smooth k=800":             dict(train_seasons=S, half_life=1.5, params=BASE_PARAMS, k=800.0),
    }

    print(f"{'variant':<26s} {'iter':>5s} {'nf':>4s} {'BSS raw':>9s} {'BSS R4':>9s} {'BSS oracle':>11s}")
    print("-" * 72)
    results = {}
    for name, cfg in variants.items():
        r = evaluate(df, season_rate, 2024, **cfg)
        results[name] = r
        print(f"{name:<26s} {r['iter']:>5d} {r['n_feat']:>4d} "
              f"{r['bss_raw']:>9.1f} {r['bss_r4']:>9.1f} {r['bss_oracle']:>11.1f}   "
              f"({time.time()-t0:.0f}s)")

    # 상위 후보를 fold1(2023)에서 강건성 확인
    print("\n--- fold1 강건성 확인 (train≤2022 → valid 2023) ---")
    S1 = [2019, 2020, 2021, 2022]
    for name in ["A base (hl=1.5)", "G no game_type", "J slow, no game_type", "I slow+regularized"]:
        cfg = dict(variants[name])
        cfg["train_seasons"] = [s for s in cfg["train_seasons"] if s in S1] or S1
        r = evaluate(df, season_rate, 2023, **cfg)
        print(f"{name:<26s} {r['iter']:>5d} {r['n_feat']:>4d} "
              f"{r['bss_raw']:>9.1f} {r['bss_r4']:>9.1f} {r['bss_oracle']:>11.1f}")

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
