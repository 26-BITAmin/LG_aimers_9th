"""마지막 노선 — **배포 모델 자신**으로 상황을 주변화해 빠른 성분을 행 독립으로 뽑는다.

errkind가 기각한 것: 따로 학습한 약한 모델(M_p)을 기준선으로 쓰는 것.
   (z − M_p)에는 '빠른 성분' 말고 **투수 레벨 정보 격차**가 섞이고, 그건 유정보라
   수축하면 진짜 신호를 버린다.

기각되지 않은 것: 같은 모델 안의 대조.
   B(row) = 그 행의 상황 컬럼만 **동결된 K개 프로파일**로 갈아끼워 예측한 평균.
   (z − B)는 같은 모델 · 같은 투수/타자 피처의 차이 = 순수 상황 효과.
   레벨 정보 격차가 생길 구조가 아니다. K개 프로파일은 train에서 뽑아 얼린다
   → test 행 간 조회 없음, 규정 문제 없음.

   z' = B + s·(z − B) = z − (1−s)·(상황 효과 추정치)

두 분할을 잰다:  A = 상황만 갈아끼움(타자는 유지) / B = 상황+타자

LGBM 단독으로 잰다 (캐시에 z_lgb가 따로 있어 대조가 깨끗하다).

실행:  LGA_FOLD_KEY=5488ef1b18 LGA_MODEL_DIR=submit/model_marg \
       .venv/Scripts/python.exe -u src/marg.py 2>&1 | tee experiments/marg.log
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

import foldcache
import modelabel
import soften
from clockaxis import gmean, qbucket
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, N_ROUNDS, SOFT_LABEL_F,
                    TRAIN_CSV)
from data import load_csv
from pbase import CACHE, PITCHER_ONLY
from rookie import FOLDS, FOLD_KEY, deployed_pred
from validation import _logit, bss, shift_to_prior

K = 8
SEED = 42
SG = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BATTER = ["batter_id", "batter_team_id", "batter_hand", "asof_batter_n",
          "miss_batter_asof", "asof_batter_success_rate", "asof_batter_success_rate_sm",
          "asof_batter_middle_rate", "asof_batter_middle_rate_sm",
          "matchup_success_edge", "matchup_middle_edge", "hand_same", "hand_matchup"]


def boosters(fold, Xtr, ytr, w, cols):
    """foldcache와 **같은 12개**(4변형 × 3시드). 모델은 캐시 안 하고 즉시 예측에 쓴다."""
    out = []
    for over in ENSEMBLE_VARIANTS.values():
        for s in foldcache.LGB_SEEDS:
            prm = dict(FINAL_LGB_PARAMS, **over, seed=s, bagging_seed=s,
                       feature_fraction_seed=s, data_random_seed=s)
            out.append(lgb.train(prm, lgb.Dataset(Xtr[cols], ytr, weight=w),
                                 num_boost_round=N_ROUNDS))
    return out


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(SEED)
    R = {}

    for fold in FOLDS:
        va, y, p_dep = deployed_pred(df, s, fold, cached)
        Xtr_o, ytr0, w0, Xva, yva, art, tr = foldcache.fold_data(df, fold)
        cols = list(Xva.columns)
        pit = set(PITCHER_ONLY) | {c + "_sm" for c in PITCHER_ONLY}
        sit_a = [c for c in cols if c not in pit and c not in BATTER]
        sit_b = [c for c in cols if c not in pit]
        if fold == FOLDS[0]:
            print(f"A(상황만) {len(sit_a)}개  B(상황+타자) {len(sit_b)}개")

        aux, _ = modelabel.aux_labels(tr)
        Xtr, ytr, w = Xtr_o, ytr0, w0
        if SOFT_LABEL_F:
            Xtr, ytr, w, aux, _ = soften.soften(tr, Xtr_o, ytr0, w0, aux)
        bs = boosters(fold, Xtr, ytr, w, cols)
        print(f"  부스터 {len(bs)}개 학습 ({time.time()-t0:.0f}s)")

        def pred(X):
            z = np.zeros(len(X))
            for b in bs:
                z += b.predict(X, raw_score=True)
            return z / len(bs)

        z_lgb = pred(Xva)
        print(f"  z_lgb 재현 {bss(y, shift_to_prior(1/(1+np.exp(-z_lgb)), y.mean())):.1f}"
              f"  (캐시 {bss(y, shift_to_prior(1/(1+np.exp(-cached[fold][0])), y.mean())):.1f})"
              f"  ({time.time()-t0:.0f}s)")

        # 동결 프로파일 — train 행에서 K개 뽑아 얼린다
        pick = rng.choice(len(Xtr_o), K, replace=False)
        base = bss(y, shift_to_prior(1/(1+np.exp(-z_lgb)), y.mean()))
        _, cl = np.unique(va["pitcher_id"].to_numpy(), return_inverse=True)
        g = cl.astype(np.int64) * 1000 + qbucket(cl, va["asof_pitcher_n"].to_numpy(float), 5)
        _, g = np.unique(g, return_inverse=True)
        zb_or = gmean(g, z_lgb)[g]

        for nm, sc in (("A 상황만", sit_a), ("B 상황+타자", sit_b)):
            B = np.zeros(len(Xva))
            for i in pick:
                Xm = Xva.copy()
                for c in sc:
                    Xm[c] = Xtr_o[c].to_numpy()[i]
                B += pred(Xm)
            B /= K
            fast = z_lgb - B
            solo = bss(y, shift_to_prior(1/(1+np.exp(-B)), y.mean()))
            row = [(s_, bss(y, shift_to_prior(1/(1+np.exp(-(z_lgb - (1-s_)*fast))),
                                              y.mean())) - base) for s_ in SG]
            k = max(row, key=lambda t: t[1])
            print(f"  [{nm}] 기준선 단독 {solo:.1f}  fast std {fast.std():.4f}"
                  f"  (오라클 fast std {np.std(z_lgb-zb_or):.4f})"
                  f"  corr(B, z̄_b) {np.corrcoef(B, zb_or)[0,1]:.3f}")
            print("     " + "  ".join(f"s={a}:{b:+.1f}" for a, b in row)
                  + f"   → 최적 s={k[0]} Δ {k[1]:+.1f}   ({time.time()-t0:.0f}s)")
            R[(fold, nm)] = dict(row)
        del bs

    print("\n" + "=" * 60)
    for nm in ("A 상황만", "B 상황+타자"):
        for s_ in SG:
            v = [R[(f, nm)][s_] for f in FOLDS]
            if min(v) > 0:
                print(f"  {nm}  s={s_}  " +
                      "  ".join(f"fold{f} {R[(f,nm)][s_]:+.1f}" for f in FOLDS) +
                      f"   크기비 {max(v)/min(v):.2f}")


if __name__ == "__main__":
    main()
