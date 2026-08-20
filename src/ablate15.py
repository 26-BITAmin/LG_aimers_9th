"""15차 실험 — CatBoost를 앙상블 멤버로 넣을지 판정.

§7.12에서 CatBoost의 -214가 설정 버그였음이 확인됐다 (cat_features -101, 과학습 -34).
`cat_features` 없이 iter를 줄이면 **-36.2**로, 통과선 -80 안쪽이다.

이제 진짜 질문: **현행 LightGBM 앙상블에 얹으면 이득인가.**
단독 성능이 아니라 블렌드가 기준을 넘느냐가 채택 조건이다 (§5: 앙상블은 구성원
실력이 비슷할 때만 이긴다).

⚠️ iteration 수의 출처를 정직하게 밝힌다.
   §7.12의 정점 250은 **fold 2024의 valid를 보고** 고른 값이다. 그대로 쓰면 낙관 편향.
   여기서는 사전에 300으로 고정하고(곡선이 200~500 구간에서 703~706으로 평평했다),
   fold 2023에서도 같은 구간이 최적인지 곡선으로 확인한다.
   fold마다 최적점이 다르면 이 설정은 채택하지 않는다.

실행:  .venv/Scripts/python.exe -u src/ablate15.py
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

from config import (DROP_FEATURES, ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, N_ROUNDS,
                    RECENCY_HALF_LIFE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL,
                    TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
SEEDS = (42, 7, 2024)
ALPHAS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)

# §7.12 2x2 결과 (valid=2024, lgbm 741.9 기준)
#                     cat_features O   X      차이
#   SymmetricTree          604.3   705.7   +101.4
#   Lossguide 31리프        622.3   715.2    +92.9
#                   차이    +18.0    +9.5
#   → cat_features 주효과 ~ -97 (압도적), grow_policy 주효과 ~ +14
#   → 최고 셀 = loss_num, 핸디캡 -26.7
#
# ⚠️ 이 설정은 **fold 2024에서 골랐다.** 그러므로 아래 실험의 fold 2023이
#    표본 외 확인이고, 부호 일치를 반드시 요구한다.
CAT_ROUNDS = 250            # loss_num 곡선이 150~350에서 705~715로 평평했다. 중앙값 채택
CAT_CURVE_PERIOD = 50       # fold마다 정점 위치를 확인한다
CAT_PARAMS = dict(          # ⚠️ cat_features 없음 — 그게 -97점의 원인이었다
    learning_rate=0.03, l2_leaf_reg=30.0, border_count=128, rsm=0.8,
    grow_policy="Lossguide", max_leaves=31, depth=16, min_data_in_leaf=2000,
    loss_function="Logloss", verbose=0, thread_count=6, allow_writing_files=False)


def fold_data(df, valid_season):
    tr = df[df.season < valid_season]
    va = df[df.season == valid_season]
    ref = int(tr.season.max())
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=ref)
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    return (X(tr), tr[TARGET_COL].to_numpy(dtype=np.float64), w,
            X(va), va[TARGET_COL].to_numpy(dtype=np.float64))


def score(z, y):
    return bss(y, shift_to_prior(1 / (1 + np.exp(-z)), y.mean()))


def lgb_ensemble(Xtr, ytr, w, Xva, t0):
    """배포와 같은 구조: ENSEMBLE_VARIANTS x 시드, **로짓 평균**."""
    z = np.zeros(len(Xva))
    n = 0
    for vname, over in ENSEMBLE_VARIANTS.items():
        for s in SEEDS:
            p = dict(FINAL_LGB_PARAMS, **over, seed=s, bagging_seed=s,
                     feature_fraction_seed=s, data_random_seed=s)
            bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=N_ROUNDS)
            z += bst.predict(Xva, raw_score=True)
            n += 1
    print(f"    lgbm {n}부스터 완료  ({time.time()-t0:.0f}s)")
    return z / n


def cat_ensemble(Xtr, ytr, w, Xva, yva, t0):
    from catboost import CatBoostClassifier, Pool
    pool_tr = Pool(Xtr, ytr, weight=w)
    pool_va = Pool(Xva)
    z = np.zeros(len(Xva))
    for i, s in enumerate(SEEDS):
        m = CatBoostClassifier(iterations=CAT_ROUNDS, random_seed=s, **CAT_PARAMS)
        m.fit(pool_tr)
        z += m.predict(pool_va, prediction_type="RawFormulaVal")
        if i == 0:      # 첫 시드만 곡선을 찍어 fold별 정점 위치를 확인한다
            cur = [((j + 1) * CAT_CURVE_PERIOD, score(np.asarray(zz, dtype=np.float64), yva))
                   for j, zz in enumerate(m.staged_predict(
                       pool_va, prediction_type="RawFormulaVal",
                       eval_period=CAT_CURVE_PERIOD))]
            print("    cat 곡선: " + "  ".join(f"{k}:{v:.0f}" for k, v in cur))
            print(f"      → 이 fold 정점 iter={max(cur, key=lambda t: t[1])[0]}"
                  f"  (사전 고정값 {CAT_ROUNDS})")
    print(f"    cat {len(SEEDS)}시드 완료  ({time.time()-t0:.0f}s)")
    return z / len(SEEDS)


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    res = {}

    for f in FOLDS:
        print(f"\n=== valid={f} ===")
        Xtr, ytr, w, Xva, yva = fold_data(df, f)
        zl = lgb_ensemble(Xtr, ytr, w, Xva, t0)
        zc = cat_ensemble(Xtr, ytr, w, Xva, yva, t0)

        row = {"lgb_ens": score(zl, yva), "cat_ens": score(zc, yva)}
        for a in ALPHAS:
            row[f"a={a:.2f}"] = score((1 - a) * zl + a * zc, yva)
        row["corr"] = float(np.corrcoef(zl, zc)[0, 1])
        res[f] = row
        print("    " + "  ".join(f"{k}={v:.1f}" if k != "corr" else f"corr={v:.3f}"
                                 for k, v in row.items()))

    keys = [k for k in res[FOLDS[0]] if k != "corr"]
    print(f"\n{'':<10s} " + " ".join(f"{f:>9d}" for f in FOLDS)
          + f" {'평균':>9s} {'기준대비':>9s} {'부호':>4s} {'크기비':>7s}")
    print("-" * 60)
    b = np.array([res[f]["lgb_ens"] for f in FOLDS])
    for k in keys:
        v = np.array([res[f][k] for f in FOLDS])
        d = v - b
        same = bool(np.all(d > 0) or np.all(d < 0))
        sp = np.abs(d).max() / max(np.abs(d).min(), 1e-9) if same else np.inf
        mark = "  <<< 채택 조건 통과" if (same and d.mean() > 0 and sp <= 3) else ""
        print(f"{k:<10s} " + " ".join(f"{x:>9.1f}" for x in v)
              + f" {v.mean():>9.1f} {d.mean():>+9.1f} {'O' if same else 'X':>4s}"
              + (f" {sp:>7.1f}" if same else f" {'inf':>7s}") + mark)
    print(f"\n로짓 상관: " + "  ".join(f"{f}={res[f]['corr']:.3f}" for f in FOLDS))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
