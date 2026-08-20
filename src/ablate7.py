"""8차 실험 — Trackman 피처가 실제로 점수를 올리는가.

매핑 자체는 검증됐다 (시즌 홀드아웃 100%, 팀 매핑이 SK→SSG 개명까지 재현).
남은 질문은 하나: **그래서 Brier가 좋아지는가.**

시드 노이즈 std가 4~9이므로 3시드 평균으로 비교한다.

실행:  .venv/Scripts/python.exe -u src/ablate7.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (DROP_FEATURES, FINAL_LGB_PARAMS, MODEL_DIR, N_ROUNDS,
                    RECENCY_HALF_LIFE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import bss, shift_to_prior

SEEDS = (42, 7, 2024)


def run(df, tm_table, train_seasons, valid_season, seeds=SEEDS, extra_drop=()):
    tr = df[df.season.isin(train_seasons)]
    va = df[df.season == valid_season]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=max(train_seasons))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, tm_table=tm_table)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        drop = [c for c in list(DROP_FEATURES) + list(extra_drop) if c in x.columns]
        return x.drop(columns=drop)

    Xtr, Xva = X(tr), X(va)
    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)

    vals, imp = [], None
    for s in seeds:
        p = dict(FINAL_LGB_PARAMS, seed=s, bagging_seed=s,
                 feature_fraction_seed=s, data_random_seed=s)
        bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=N_ROUNDS)
        z = bst.predict(Xva, raw_score=True)
        pr = shift_to_prior(1 / (1 + np.exp(-z)), yva.mean())
        vals.append(bss(yva, pr))
        if imp is None:
            imp = pd.Series(bst.feature_importance("gain"), index=Xtr.columns)
    v = np.array(vals)
    return v, Xtr.shape[1], imp


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    tm = pd.read_csv(MODEL_DIR / "trackman_pitcher.csv")
    print(f"trackman 테이블 {tm.shape}")

    S = [2019, 2020, 2021, 2022, 2023]
    RELEASE = [c for c in tm.columns if "rel" in c or "disp" in c]

    cases = {
        "A) Trackman 없음 (현재 제출)": dict(tm_table=None),
        "B) Trackman 전체 28피처":      dict(tm_table=tm),
        "C) 릴리스포인트 계열만":        dict(tm_table=tm[["pitcher_id", "season"] + RELEASE]),
    }
    print(f"\n{'설정':<30s} {'nf':>4s} {'3시드 BSS oracle':>28s} {'평균':>8s} {'std':>6s}")
    print("-" * 84)
    res = {}
    for name, kw in cases.items():
        v, nf, imp = run(df, kw["tm_table"], S, 2024)
        res[name] = (v, imp)
        print(f"{name:<30s} {nf:>4d} {str(np.round(v,1)):>28s} {v.mean():>8.1f} "
              f"{v.std(ddof=1):>6.1f}   ({time.time()-t0:.0f}s)")

    base = res["A) Trackman 없음 (현재 제출)"][0]
    for name, (v, _) in res.items():
        if name.startswith("A"):
            continue
        d = v.mean() - base.mean()
        se = np.sqrt(v.var(ddof=1) / len(v) + base.var(ddof=1) / len(base))
        print(f"\n{name}: {d:+.1f} BSS  (차이의 표준오차 {se:.1f}, "
              f"{abs(d)/max(se,1e-9):.1f}σ)")

    imp = res["B) Trackman 전체 28피처"][1]
    tmimp = imp[[c for c in imp.index if c.startswith("tm")]].sort_values(ascending=False)
    print(f"\nTrackman 피처 gain 상위 12개 (전체 gain 중 비중 "
          f"{tmimp.sum()/imp.sum():.1%}):")
    print(tmimp.head(12).round(0).to_string())

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
