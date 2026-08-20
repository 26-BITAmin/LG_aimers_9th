"""§7.101 5차 (마지막) — 무승부 모델이 **블렌드에서** 값어치가 있는가

여기까지
    TabPFN 최고 설정(ctx10k est1 정제) = LGBM F전문가와 무승부 (Δ −11.3±89.5, t −0.13).
    TabICL은 위생을 고쳐도 기준선 이하 — 닫았다.

무승부 ≠ 무가치
    단독 점수가 같아도 **탈상관**돼 있으면 앙상블에서 값어치가 있다
    (MODEL.md: 분산 축소 수단은 서로 겹친다 — 상관계수를 같이 봐야 한다).
    그러니 닫기 전에 물어야 할 것은 "TabPFN이 LGBM보다 나은가"가 아니라
    **"이미 가진 것(전역 + LGBM전문가) 위에 TabPFN이 더 얹히는가"**다.

세 예측을 같은 8,000행에서 만들고 상관·블렌드를 본다
    G  전역 배포 모델 (전체 + soft)      ← 현행 F행 예측 주체
    S  LGBM F전문가 (2023 F만)          ← 3차에서 확보. G와 블렌드 시 F행 +54.6
    T  TabPFN ctx10k est1 정제           ← 무승부 후보

    2원 스윕 G+S (재확인) · G+T · 3원 G+S+T
    판정: 3원 최고가 2원(G+S) 최고를 **쌍체 SE의 2배 이상** 넘지 못하면 닫는다.

⚠️ 블렌드 가중치는 평가 fold에서 고른다 = 낙관 편향. 여기서 나온 최고값은
   상한이지 기대값이 아니다. 그래도 "얹히지 않는다"는 결론에는 영향이 없다
   (편향이 TabPFN에 유리한 방향인데도 못 이기면 그건 확실한 음성이다).

실행:  .venv-gpu/Scripts/python.exe -u src/fblend.py 2>&1 | tee experiments/fblend.log
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
import torch
import truststore

truststore.inject_into_ssl()

import soften as soften_mod
from config import (CONTAM_AXES, DROP_FEATURES, FINAL_LGB_PARAMS, N_ROUNDS,
                    RECENCY_HALF_LIFE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import bss, shift_to_prior

ACC_N = 8000
SEEDS = (42, 7, 2024)
BIG_COLS = ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n")
ID_COLS = ("pitcher_id", "batter_id")
GRID = np.arange(0.0, 1.001, 0.05)

SMALL_LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15,
                 min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7,
                 bagging_freq=1, lambda_l2=10.0, max_bin=127, num_threads=6,
                 verbosity=-1)


def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    from tabpfn import TabPFNClassifier

    df = load_csv(TRAIN_CSV)
    tr = df[df.season < 2024]
    va = df[df.season == 2024]
    w_all = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=2023)
    art = fit_artifacts(tr, weights=w_all, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ID_COLS:
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    trF = tr[(tr.season == 2023) & (tr.game_type.astype(str) == "F")]
    vaF = va[va.game_type.astype(str) == "F"]
    xtr = X(trF); cols = list(xtr.columns)
    Xtr = xtr.to_numpy(dtype=np.float32)
    ytr = trF[TARGET_COL].to_numpy(dtype=np.int64)
    XvaF = X(vaF).to_numpy(dtype=np.float32)
    yvaF = vaF[TARGET_COL].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(42)
    acc_i = rng.permutation(len(XvaF))[:ACC_N]
    ya = yvaF[acc_i]
    Xq = XvaF[acc_i]
    ref = ya.mean() * (1 - ya.mean())

    keep = [i for i, c in enumerate(cols) if c not in ID_COLS]
    big = [j for j, i in enumerate(keep) if cols[i] in BIG_COLS]
    med = np.nanmedian(Xtr[:, keep], axis=0)

    def clean(Z):
        Z = Z[:, keep].copy()
        Z[:, big] = np.log1p(np.maximum(np.nan_to_num(Z[:, big], nan=0.0), 0.0))
        bad = ~np.isfinite(Z)
        if bad.any():
            Z[bad] = np.take(med, np.nonzero(bad)[1])
        return Z

    # ── G. 전역 배포 모델 ────────────────────────────────────
    t0 = time.time()
    Xa, ya_tr, wa, _, _ = soften_mod.soften(
        tr, X(tr), tr[TARGET_COL].to_numpy(dtype=np.float64), w_all, None)
    pg = []
    for s in SEEDS:
        prm = dict(FINAL_LGB_PARAMS, seed=s, bagging_seed=s,
                   feature_fraction_seed=s, data_random_seed=s)
        b = lgb.train(prm, lgb.Dataset(Xa, ya_tr, weight=wa), num_boost_round=N_ROUNDS)
        pg.append(1 / (1 + np.exp(-b.predict(Xq, raw_score=True))))
    G = np.mean(pg, axis=0); del Xa
    print(f"  G 전역 배포      subBSS {bss(ya, shift_to_prior(G, ya.mean())):>8.1f}  ({time.time()-t0:.0f}s)")

    # ── S. LGBM F전문가 ─────────────────────────────────────
    month = trF["game_month"].to_numpy(dtype=np.float64)
    cut = np.quantile(month, 0.8); itr, iva = month <= cut, month > cut
    ps = []
    for s in SEEDS:
        prm = dict(SMALL_LGB, seed=s, bagging_seed=s, feature_fraction_seed=s,
                   data_random_seed=s)
        b0 = lgb.train(prm, lgb.Dataset(Xtr[itr], ytr[itr].astype(np.float64)),
                       num_boost_round=3000,
                       valid_sets=[lgb.Dataset(Xtr[iva], ytr[iva].astype(np.float64))],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        b = lgb.train(prm, lgb.Dataset(Xtr, ytr.astype(np.float64)),
                      num_boost_round=max(50, b0.best_iteration))
        ps.append(1 / (1 + np.exp(-b.predict(Xq, raw_score=True))))
    S = np.mean(ps, axis=0)
    print(f"  S LGBM F전문가   subBSS {bss(ya, shift_to_prior(S, ya.mean())):>8.1f}")

    # ── T. TabPFN ctx10k est1 정제 ──────────────────────────
    i = np.random.default_rng(42).choice(len(Xtr), size=10_000, replace=False)
    clf = TabPFNClassifier(device="cuda", n_estimators=1, inference_precision="autocast",
                           random_state=42, ignore_pretraining_limits=True)
    clf.fit(clean(Xtr[i]), ytr[i])
    T = clf.predict_proba(clean(Xq))[:, 1]
    print(f"  T TabPFN 정제    subBSS {bss(ya, shift_to_prior(T, ya.mean())):>8.1f}")

    zg, zs, zt = logit(G), logit(S), logit(T)
    print(f"\n  로짓 상관   G·S {np.corrcoef(zg, zs)[0,1]:.4f}   "
          f"G·T {np.corrcoef(zg, zt)[0,1]:.4f}   S·T {np.corrcoef(zs, zt)[0,1]:.4f}")

    def sc(z):
        return bss(ya, shift_to_prior(1 / (1 + np.exp(-z)), ya.mean()))

    best2 = max(((sc((1 - a) * zg + a * zs), a) for a in GRID))
    best2t = max(((sc((1 - a) * zg + a * zt), a) for a in GRID))
    print(f"\n  2원 G+S  최고 {best2[0]:>8.1f}  (a={best2[1]:.2f})")
    print(f"  2원 G+T  최고 {best2t[0]:>8.1f}  (a={best2t[1]:.2f})")

    best3 = (-1e18, 0, 0)
    for a in GRID:
        for b in GRID:
            if a + b > 1.0 + 1e-9:
                continue
            v = sc((1 - a - b) * zg + a * zs + b * zt)
            if v > best3[0]:
                best3 = (v, a, b)
    print(f"  3원 G+S+T 최고 {best3[0]:>8.1f}  (S={best3[1]:.2f}, T={best3[2]:.2f})")

    # 판정 — 3원 최고 vs 2원(G+S) 최고, 같은 행에서 쌍체
    z2 = (1 - best2[1]) * zg + best2[1] * zs
    z3 = (1 - best3[1] - best3[2]) * zg + best3[1] * zs + best3[2] * zt
    p2, p3 = shift_to_prior(1/(1+np.exp(-z2)), ya.mean()), shift_to_prior(1/(1+np.exp(-z3)), ya.mean())
    d = (p2 - ya) ** 2 - (p3 - ya) ** 2
    m, se = 1e5 * d.mean() / ref, 1e5 * d.std(ddof=1) / np.sqrt(len(d)) / ref
    print(f"\n  3원 − 2원(G+S) 쌍체 Δ = {m:+.1f} ± {se:.1f}  (t={m/se:.2f})")
    print(f"  판정: {'TabPFN이 얹힌다' if m > 2*se else 'TabPFN은 더 얹히지 않는다 → 닫는다'}")


if __name__ == "__main__":
    main()
