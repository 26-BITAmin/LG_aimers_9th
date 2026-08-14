"""최적 용량이 시즌의 **신호량**을 따라 움직이는가

동기 — 지금까지 안 본 숫자
    fold2024 홀드아웃 770 vs LB 975.7. 이걸 "홀드아웃이 과소평가한다"로만 다뤘는데
    분해하면 다른 이야기다. LB에는 레벨 항이 들어 있으므로

        975.7 = R - 400,700·δ²      →      2025의 refinement R ≥ 975.7

    **2025는 2024(770)보다 200점 이상 예측하기 쉬운 시즌이다.**
    `foldsurvey`가 시즌별 신호량을 이미 재놨다 (단일 base 모델, BSS oracle):

        2021  973.2      2022 1849.2      2023  -10.2      2024  741.9

    **2025(≈975+)는 2021(973) 대역이고, 우리가 튜닝에 쓴 2023·2024는 최하위 둘이다.**

    → 가설: **신호가 적은 시즌에서 고른 설정은 과도하게 조여져 있다.**
      우리 설정은 전부 조이는 쪽으로 골라졌다 (lr 0.03→0.015 +11, min_data 2000,
      lambda_l2 30, hl 0.5). 저신호 시즌 최적이라면 고신호 시즌에서는 손해다.

    ⚠️ 이건 §9가 허용하는 LB 독법이다 — 점수로 파라미터를 푸는 게 아니라
      **세계에 대한 사실("2025는 고신호 시즌")을 읽는 것**이다. 용량은 이 스크립트가
      train 안에서만 고른다.

설계
    · **트리 단독, 단일 모델**로 싸게 훑는다 (앙상블·MLP 없음). 여기서 추세가 안 보이면
      비싼 실험을 할 이유가 없다. 보이면 그때 배포 구조로 재확인한다.
    · fold 2021·2022는 §2가 "2025와 다른 레짐"이라 판정 근거에서 뺀 것들이다.
      여기서는 **판정이 아니라 관계(용량 대 신호량)를 재는 데만** 쓴다 —
      §7.22와 같은 용법이고, 절대 점수는 해석하지 않는다.
    · 용량 외 전부 고정 (hl 0.5, N_ROUNDS 320, game_type 제거, ids 사용).

사전 예측
    P1  fold별 최적 용량이 그 fold의 신호량을 따라 단조 증가한다.
        내 예측: **아니오** — §7.5의 용량 스윕이 727~744로 평평했고 §7.18의 재튜닝이
        9전 9패였다. 이 프로젝트의 기저율은 "용량 축은 평평하다"이다.
        (내 긍정 예측은 0승 5패라 방향을 걸 때는 기저율 쪽에 건다.)
    P2  그래도 **신호량 자체는 fold마다 크게 다를 것**이다 (foldsurvey 재현).
        이건 예측이 아니라 정합성 확인이다 — 다르지 않으면 내 전제가 무너진다.

실행:  .venv/Scripts/python.exe -u src/ablate30.py 2>&1 | tee experiments/ablate_run30.log
"""
import gc
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

import ablate23 as A23
import mlp
from config import (FINAL_LGB_PARAMS, N_ROUNDS, RECENCY_HALF_LIFE, SEED,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from features import fit_artifacts

THREADS = 5
FOLDS = [2021, 2022, 2023, 2024]

# 용량 축만 움직인다. 조인 쪽 → 푼 쪽. `base`가 배포값이다.
CAPS = {
    "tighter": dict(num_leaves=15, min_data_in_leaf=8000, lambda_l2=100.0),
    "tight":   dict(num_leaves=15, min_data_in_leaf=4000, lambda_l2=50.0),
    "base":    dict(),                                      # 31 / 2000 / l2 30
    "loose":   dict(num_leaves=63, min_data_in_leaf=500, lambda_l2=10.0),
    "looser":  dict(num_leaves=127, min_data_in_leaf=100, lambda_l2=1.0),
}


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    season = df.season.to_numpy()
    R = {}
    for fold in FOLDS:
        tr, va = df[season < fold], df[season == fold]
        w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=fold - 1)
        art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                            k_batter=SMOOTH_K_BATTER)
        Xtr, Xva = A23.featurize(tr, art), A23.featurize(va, art)
        ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
        yva = va[TARGET_COL].to_numpy(dtype=np.float64)
        print(f"\n=== fold {fold}  학습 {len(Xtr):,}  검증 {len(yva):,}"
              f"  ({time.time()-t0:.0f}s) ===")
        del tr, va
        gc.collect()
        for name, over in CAPS.items():
            p = dict(FINAL_LGB_PARAMS, **over, seed=SEED, bagging_seed=SEED,
                     feature_fraction_seed=SEED, data_random_seed=SEED,
                     num_threads=THREADS)
            bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w),
                            num_boost_round=N_ROUNDS)
            z = bst.predict(Xva, raw_score=True)
            R[(fold, name)] = mlp.score(z, yva)
            del bst, z
            print(f"  {name:>8}  leaves {p['num_leaves']:>3}"
                  f"  min_data {p['min_data_in_leaf']:>5}  l2 {p['lambda_l2']:>5.0f}"
                  f"   BSS {R[(fold,name)]:>8.1f}   ({time.time()-t0:.0f}s)")
        del Xtr, Xva
        gc.collect()
    del df
    gc.collect()

    print("\n" + "=" * 88)
    print("용량 × 시즌 신호량  (BSS oracle, 트리 단일 모델)")
    print("=" * 88)
    print(f"{'fold':>6} " + " ".join(f"{n:>9}" for n in CAPS)
          + f" {'최적':>9} {'base대비':>9}")
    for fold in FOLDS:
        row = [R[(fold, n)] for n in CAPS]
        best = max(CAPS, key=lambda n: R[(fold, n)])
        print(f"{fold:>6} " + " ".join(f"{v:>9.1f}" for v in row)
              + f" {best:>9} {R[(fold,best)]-R[(fold,'base')]:>+9.1f}")

    order = list(CAPS)
    print(f"\n{'fold':>6} {'신호량(base)':>12} {'최적 용량 순위':>14}  (0=가장 조임, 4=가장 풂)")
    for fold in sorted(FOLDS, key=lambda f: R[(f, "base")]):
        best = max(CAPS, key=lambda n: R[(fold, n)])
        print(f"{fold:>6} {R[(fold,'base')]:>12.1f} {order.index(best):>14}")

    sig = np.array([R[(f, "base")] for f in FOLDS])
    rank = np.array([list(CAPS).index(max(CAPS, key=lambda n: R[(f, n)])) for f in FOLDS])
    if len(set(rank)) > 1:
        c = np.corrcoef(sig, rank)[0, 1]
        print(f"\n  신호량 vs 최적용량 순위  상관 {c:+.3f}  (n=4)")
    else:
        print(f"\n  네 fold 모두 최적이 '{order[rank[0]]}'로 같다 → 용량 축은 신호량과 무관")
    print("\n  ⚠️ n=4다. 상관계수 하나로 채택하지 않는다. 추세가 보이면 배포 구조(앙상블+"
          "MLP 블렌드)로 fold2023·2024에서 다시 재고, 조건 ①②③으로 판정한다.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
