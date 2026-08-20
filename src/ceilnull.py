"""§7.101 11차 — "도달 가능한 최선"이 얼마나 부풀어 있나

10차에서 나온 것
    TabPFN의 천장 711.1  >  LGBM의 천장 683.6.  순위 정보는 TabPFN이 더 많다는 뜻이다.
    (TabICL은 416으로 LGBM의 **현재 점수**보다도 낮아 이미 닫혔다.)

믿기 전에 확인할 것
    천장은 평가셋의 정답을 보고 등장회귀를 적합해 얻은 값이라 **부풀어 있다.**
    부풀림의 크기를 모르면 711 vs 684를 해석할 수 없다.

    영가설: 예측을 무작위로 섞어 순위 정보를 **전부 파괴**한 뒤 같은 방식으로 천장을
    계산한다. 그 값이 오라클 적합만으로 생기는 순수 부풀림이다.
    (MODEL.md의 원칙 — 셔플은 0을 주지만 실제 죽은 축은 후보와 같은 프로파일을 준다.
     여기서는 셔플이 맞다. 파괴하려는 게 "순위" 하나뿐이기 때문이다.)

    그리고 711 − 684 = 27 에 오차막대를 붙인다. 같은 행을 부트스트랩으로 다시 뽑아
    두 천장의 **차이**를 반복 계산한다 (쌍체 — 같은 행에서 같이 흔들린다).

실행:  .venv-gpu/Scripts/python.exe -u src/ceilnull.py 2>&1 | tee experiments/ceilnull.log
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

from calceil import ceiling
from config import (CONTAM_AXES, DROP_FEATURES, RECENCY_HALF_LIFE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import auc, bss, shift_to_prior

ACC_N = 8000
SEEDS = (42, 7, 2024)
CTX = 10_000
N_NULL = 40
N_BOOT = 300
BIG_COLS = ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n")
ID_COLS = ("pitcher_id", "batter_id")

SMALL_LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15,
                 min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7,
                 bagging_freq=1, lambda_l2=10.0, max_bin=127, num_threads=6,
                 verbosity=-1)


def main():
    from tabpfn import TabPFNClassifier

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

    def prep(Z):
        W = Z[:, keep].copy()
        W[:, big] = np.log1p(np.maximum(np.nan_to_num(W[:, big], nan=0.0), 0.0))
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

    P = {"LGBM drop": lgb_run(Xtr[:, keep], Xq[:, keep]),
         "LGBM raw": lgb_run(Xtr, Xq)}
    Ztr, Zq = prep(Xtr), prep(Xq)
    i = np.random.default_rng(42).choice(len(Ztr), size=CTX, replace=False)
    c = TabPFNClassifier(device="cuda", n_estimators=1, random_state=42,
                         inference_precision="autocast", ignore_pretraining_limits=True)
    c.fit(Ztr[i], ytr[i]); P["TabPFN"] = c.predict_proba(Zq)[:, 1]; del c

    # ── 영가설: 순위를 파괴한 예측의 천장 ────────────────────
    rng = np.random.default_rng(0)
    null = np.array([ceiling(ya, rng.permutation(P["LGBM drop"])) for _ in range(N_NULL)])
    print(f"  순위 파괴(셔플) 천장   {null.mean():.1f} ± {null.std(ddof=1):.1f}"
          f"   [{null.min():.1f}, {null.max():.1f}]   n={N_NULL}")
    print(f"  → 오라클 적합만으로 생기는 부풀림이 이만큼이다.\n")

    print(f"  {'모델':<14}{'실제':>9}{'천장':>9}{'천장−영가설':>13}{'AUC':>9}")
    for k, p in P.items():
        s = bss(ya, shift_to_prior(p, ya.mean()))
        cl = ceiling(ya, p)
        print(f"  {k:<14}{s:>9.1f}{cl:>9.1f}{cl-null.mean():>13.1f}{auc(ya, p):>9.4f}")

    # ── 부트스트랩: 천장 차이의 오차막대 (쌍체) ──────────────
    print(f"\n  부트스트랩 {N_BOOT}회 — 천장 **차이**, 같은 행 재표집")
    rb = np.random.default_rng(1)
    n = len(ya)
    for a, b in (("TabPFN", "LGBM drop"), ("TabPFN", "LGBM raw"), ("LGBM drop", "LGBM raw")):
        d = np.empty(N_BOOT)
        for t in range(N_BOOT):
            j = rb.integers(0, n, n)
            if ya[j].mean() in (0.0, 1.0):
                d[t] = 0.0
                continue
            d[t] = ceiling(ya[j], P[a][j]) - ceiling(ya[j], P[b][j])
        lo, hi = np.percentile(d, [2.5, 97.5])
        star = "유의" if lo > 0 or hi < 0 else "판별 불가"
        print(f"    {a:<11} − {b:<11} {d.mean():>+8.1f}   95% [{lo:>+7.1f}, {hi:>+7.1f}]   {star}")


if __name__ == "__main__":
    main()
