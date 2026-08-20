"""§7.101 10차 — 판별력 천장으로 TabICL을 닫는다 (또는 못 닫는다)

왜 이 측정인가
    TabICL에 남은 손잡이(온도 1.5 초과, n_estimators 8 초과, NaN 유지)는 **전부
    캘리브레이션 축**이다. 온도는 로짓의 단조 변환이라 AUC를 못 바꾸고,
    앙상블 증가는 분산만 줄인다 (est4 0.5275 → est8 0.5281로 이미 평평).

    그러니 손잡이를 하나씩 더듬는 대신 **상한을 직접 잰다**:
    오라클 등장회귀(isotonic)로 그 순위가 달성할 수 있는 **최선의 Brier**.
    어떤 사후 보정도 이 값을 못 넘는다. 이게 LGBM 아래면 TabICL은 판별력에서
    진 것이고 남은 스윕이 전부 무의미해진다.

    ⚠️ 이 값은 평가셋 자신에 적합한 오라클이라 **낙관 상한**이다. 실제로 도달할 수
       없다. 그래서 "이것도 LGBM보다 낮다"는 결론에만 쓴다 (음성 판정 전용).
       양성 방향으로는 아무것도 주장하지 않는다.

같이 확인
    · TabPFN이 무승부인 게 정말 판별력 대등 때문인지 (천장이 LGBM과 붙는지)
    · LGBM 자신의 천장 — 현재 캘리브레이션이 얼마나 손해를 보고 있는지
    · TabICL 온도 스윕이 천장으로 수렴하는지 (수렴하면 온도 축은 소진된 것)

실행:  .venv-gpu/Scripts/python.exe -u src/calceil.py 2>&1 | tee experiments/calceil.log
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
import truststore

truststore.inject_into_ssl()

from config import (CONTAM_AXES, DROP_FEATURES, RECENCY_HALF_LIFE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import auc, bss, brier, shift_to_prior

ACC_N = 8000
SEEDS = (42, 7, 2024)
CTX = 10_000
BIG_COLS = ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n")
ID_COLS = ("pitcher_id", "batter_id")
TEMPS = (0.9, 1.5, 2.5, 4.0)

SMALL_LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15,
                 min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7,
                 bagging_freq=1, lambda_l2=10.0, max_bin=127, num_threads=6,
                 verbosity=-1)


def isotonic(y, p):
    """PAVA. 순위를 보존하는 **최적** 확률 배정 → 그 순위의 Brier 하한."""
    o = np.argsort(p, kind="mergesort")
    v = y[o].astype(np.float64).copy()
    w = np.ones(len(v))
    # pool adjacent violators
    lvl, lw = [], []
    for i in range(len(v)):
        cv, cw = v[i], w[i]
        while lvl and lvl[-1] > cv:
            pv, pw = lvl.pop(), lw.pop()
            cv = (pv * pw + cv * cw) / (pw + cw)
            cw = pw + cw
        lvl.append(cv); lw.append(cw)
    fit = np.repeat(np.array(lvl), np.array(lw).astype(int))
    out = np.empty(len(v))
    out[o] = fit
    return out


def ceiling(y, p):
    """오라클 등장회귀 후 subBSS — 그 순위로 가능한 최선."""
    q = np.clip(isotonic(y, np.asarray(p, dtype=np.float64)), 1e-6, 1 - 1e-6)
    ref = y.mean() * (1 - y.mean())
    return 1e5 * (1 - brier(y, q) / ref)


def main():
    df = load_csv(TRAIN_CSV)
    tr = df[df.season < 2024]
    va = df[df.season == 2024]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=2023)
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ID_COLS:
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    trF = tr[(tr.season == 2023) & (tr.game_type.astype(str) == "F")]
    vaF = va[va.game_type.astype(str) == "F"]
    acc_i = np.random.default_rng(42).permutation(len(vaF))[:ACC_N]

    xtr = X(trF); cols = list(xtr.columns)
    Xtr = xtr.to_numpy(dtype=np.float32)
    Xq = X(vaF).iloc[acc_i].to_numpy(dtype=np.float32)
    ytr = trF[TARGET_COL].to_numpy(dtype=np.int64)
    ya = vaF[TARGET_COL].to_numpy(dtype=np.float64)[acc_i]

    keep = [i for i, c in enumerate(cols) if c not in ID_COLS]
    big = [j for j, i in enumerate(keep) if cols[i] in BIG_COLS]
    med = np.nanmedian(Xtr[:, keep], axis=0)

    def prep(Z, impute=True):
        W = Z[:, keep].copy()
        W[:, big] = np.log1p(np.maximum(np.nan_to_num(W[:, big], nan=0.0), 0.0))
        if impute:
            bad = ~np.isfinite(W)
            if bad.any():
                W[bad] = np.take(med, np.nonzero(bad)[1])
        return W

    month = trF["game_month"].to_numpy(dtype=np.float64)
    cut = np.quantile(month, 0.8); itr, iva = month <= cut, month > cut

    def lgb_run(Ztr, Zq):
        ps = []
        for s in SEEDS:
            prm = dict(SMALL_LGB, seed=s, bagging_seed=s, feature_fraction_seed=s,
                       data_random_seed=s)
            b0 = lgb.train(prm, lgb.Dataset(Ztr[itr], ytr[itr].astype(np.float64)),
                           num_boost_round=3000,
                           valid_sets=[lgb.Dataset(Ztr[iva], ytr[iva].astype(np.float64))],
                           callbacks=[lgb.early_stopping(100, verbose=False)])
            b = lgb.train(prm, lgb.Dataset(Ztr, ytr.astype(np.float64)),
                          num_boost_round=max(50, b0.best_iteration))
            ps.append(1 / (1 + np.exp(-b.predict(Zq, raw_score=True))))
        return np.mean(ps, axis=0)

    preds = {}
    preds["LGBM raw 3시드"] = lgb_run(Xtr, Xq)
    preds["LGBM drop 3시드"] = lgb_run(Xtr[:, keep], Xq[:, keep])

    Ztr, Zq = prep(Xtr), prep(Xq)
    i = np.random.default_rng(42).choice(len(Ztr), size=CTX, replace=False)

    from tabpfn import TabPFNClassifier
    t0 = time.time()
    c = TabPFNClassifier(device="cuda", n_estimators=1, random_state=42,
                         inference_precision="autocast", ignore_pretraining_limits=True)
    c.fit(Ztr[i], ytr[i])
    preds["TabPFN ctx10k est1"] = c.predict_proba(Zq)[:, 1]
    del c
    print(f"  TabPFN {time.time()-t0:.0f}s")

    # TabPFN NaN 유지 — 9차에서 못 잰 ③번 항목 (est1이 TabPFN 최적이라 est1로 잰다)
    Ztr2, Zq2 = prep(Xtr, impute=False), prep(Xq, impute=False)
    c = TabPFNClassifier(device="cuda", n_estimators=1, random_state=42,
                         inference_precision="autocast", ignore_pretraining_limits=True)
    c.fit(Ztr2[i], ytr[i])
    preds["TabPFN est1 NaN유지"] = c.predict_proba(Zq2)[:, 1]
    del c

    from tabicl import TabICLClassifier
    for tp in TEMPS:
        t0 = time.time()
        c = TabICLClassifier(device="cuda", n_estimators=8, random_state=42,
                             softmax_temperature=tp, verbose=False)
        c.fit(Ztr[i], ytr[i])
        preds[f"TabICL est8 temp{tp}"] = c.predict_proba(Zq)[:, 1]
        del c
        print(f"  TabICL temp{tp} {time.time()-t0:.0f}s")

    # TabICL NaN 유지
    try:
        c = TabICLClassifier(device="cuda", n_estimators=8, random_state=42,
                             softmax_temperature=1.5, verbose=False)
        c.fit(Ztr2[i], ytr[i])
        preds["TabICL est8 t1.5 NaN유지"] = c.predict_proba(Zq2)[:, 1]
        del c
    except Exception as e:
        print(f"  NaN유지 FAIL {type(e).__name__}: {str(e)[:50]}")

    print(f"\n  {'모델':<28}{'subBSS':>9}{'AUC':>8}{'천장(오라클)':>13}{'캘리브 손실':>12}")
    for k, p in preds.items():
        ps = shift_to_prior(p, ya.mean())
        s, ceil_ = bss(ya, ps), ceiling(ya, p)
        print(f"  {k:<28}{s:>9.1f}{auc(ya, ps):>8.4f}{ceil_:>13.1f}{ceil_-s:>12.1f}")

    print("\n  판정 — 천장끼리 비교 (사후 보정으로도 못 넘는 선)")
    cl = {k: ceiling(ya, p) for k, p in preds.items()}
    best_lgb = max(cl["LGBM raw 3시드"], cl["LGBM drop 3시드"])
    for k, v in sorted(cl.items(), key=lambda kv: -kv[1]):
        mark = "" if k.startswith("LGBM") else ("  ← LGBM 천장 초과" if v > best_lgb else "")
        print(f"    {k:<28}{v:>9.1f}{mark}")


if __name__ == "__main__":
    main()
