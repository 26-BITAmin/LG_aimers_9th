"""MLP 정직한 재판정 — 천장 실험(mlp2)에서 valid를 보고 고른 것들을 전부 제거한다.

mlp2 결과 (valid=2024, 낙관적 상한):
    ID임베딩 4차원+wd   ep7  단독 686.9 (-55.0)  블렌드 a=0.4 → 765.2 (+23.3)  corr 0.882
    소형+강한 정규화     ep10 단독 695.4 (-46.5)  블렌드 a=0.3 → 753.9 (+12.0)  corr 0.912

그 실험은 **에폭도 블렌드 가중치도 valid의 argmax**였다. 낙관 편향이 들어 있다.
여기서는 셋 다 사전에 못 박는다:

  1. **argmax 금지.** 마지막 에폭 값을 쓴다. OneCycle 스케줄이 그 지점에서 끝나도록
     epochs를 설정하므로 LR이 0으로 어닐링된 상태의 값이다.
  2. **에폭 수는 mlp2 곡선에서 사전 선택.** = fold 2024에서 골랐다는 뜻이므로
     **fold 2023이 표본 외 확인**이다. 부호 일치를 반드시 요구한다.
  3. **블렌드 가중치 a=0.3 고정.** mlp2에서 관측된 최적값 0.2~0.4의 중앙.

비교 기준은 단일 LightGBM이 아니라 **배포와 같은 구조의 앙상블**이다
(ENSEMBLE_VARIANTS x 시드, 로짓 평균). 이미 분산이 제거된 위에 얹어도 남는지를 본다.

실행:  .venv/Scripts/python.exe -u src/mlp3.py
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

import mlp
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, N_ROUNDS, TRAIN_CSV)
from data import load_csv

FOLDS = [2023, 2024]
SEEDS = (42, 7)
ALPHA = 0.30                 # 사전 고정
ALPHA_GRID = (0.15, 0.20, 0.30, 0.40, 0.50)   # 참고 출력용 (판정은 ALPHA로만)

EMB4 = dict(mlp.EMB_DIM, pitcher_id=4, batter_id=4)

# 에폭 수는 mlp2 곡선의 평탄 구간에서 고른다.
#   emb4_wd : ep3~10이 644~687로 평평 → 6, 9 두 지점을 본다
#   tiny    : ep10에서도 상승 중(695.4) → 더 준다
VARIANTS = {
    "emb4_wd@6": dict(epochs=6, use_id_emb=True, wd=1e-2, dropout=0.3, hid=128,
                      emb_dim=EMB4),
    "emb4_wd@9": dict(epochs=9, use_id_emb=True, wd=1e-2, dropout=0.3, hid=128,
                      emb_dim=EMB4),
    "tiny@16": dict(epochs=16, use_id_emb=False, hid=64, dropout=0.4, wd=3e-2),
    "tiny@24": dict(epochs=24, use_id_emb=False, hid=64, dropout=0.4, wd=3e-2),
}


def lgb_ensemble(Xtr, ytr, w, Xva, t0):
    """배포와 같은 구조: ENSEMBLE_VARIANTS x 시드, 로짓 평균."""
    z, n = np.zeros(len(Xva)), 0
    for over in ENSEMBLE_VARIANTS.values():
        for s in SEEDS:
            p = dict(FINAL_LGB_PARAMS, **over, seed=s, bagging_seed=s,
                     feature_fraction_seed=s, data_random_seed=s)
            bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=N_ROUNDS)
            z += bst.predict(Xva, raw_score=True)
            n += 1
    print(f"  lgbm {n}부스터 완료  ({time.time()-t0:.0f}s)")
    return z / n


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    res = {}

    for f in FOLDS:
        print(f"\n=== valid={f} ===")
        Xtr, ytr, w, Xva, yva, art = mlp.fold_data(df, f)
        A, Ctr, B, Cva, num_cols, sizes = mlp.encode(Xtr, Xva, art)

        zl = lgb_ensemble(Xtr, ytr, w, Xva, t0)
        row = {"lgb_ens": mlp.score(zl, yva)}
        print(f"  lgb_ens BSS = {row['lgb_ens']:.1f}")

        for name, cfg in VARIANTS.items():
            zs = []
            for s in SEEDS:
                _, curve = mlp.train_mlp(A, Ctr, ytr, w, B, Cva, sizes, seed=s,
                                         verbose=False, **cfg)
                zs.append(curve[-1])          # ⚠️ argmax가 아니라 **마지막 에폭**
            zm = np.mean(zs, axis=0)
            solo = mlp.score(zm, yva)
            row[name] = mlp.score((1 - ALPHA) * zl + ALPHA * zm, yva)
            grid = "  ".join(f"a{a:.2f}:{mlp.score((1-a)*zl + a*zm, yva):.0f}"
                             for a in ALPHA_GRID)
            print(f"  {name:<12s} 단독 {solo:>7.1f}  a={ALPHA} → {row[name]:>7.1f}"
                  f"  corr {np.corrcoef(zl, zm)[0,1]:.3f}   [{grid}]"
                  f"   ({time.time()-t0:.0f}s)")
        res[f] = row

    print(f"\n{'설정':<12s} " + " ".join(f"{f:>9d}" for f in FOLDS)
          + f" {'평균':>9s} {'기준대비':>9s} {'부호':>4s} {'크기비':>7s}")
    print("-" * 62)
    b = np.array([res[f]["lgb_ens"] for f in FOLDS])
    print(f"{'lgb_ens':<12s} " + " ".join(f"{x:>9.1f}" for x in b) + f" {b.mean():>9.1f}")
    for name in VARIANTS:
        v = np.array([res[f][name] for f in FOLDS])
        d = v - b
        same = bool(np.all(d > 0) or np.all(d < 0))
        sp = np.abs(d).max() / max(np.abs(d).min(), 1e-9) if same else np.inf
        mark = "  <<< 채택 조건 통과" if (same and d.mean() > 0 and sp <= 3) else ""
        print(f"{name:<12s} " + " ".join(f"{x:>9.1f}" for x in v)
              + f" {v.mean():>9.1f} {d.mean():>+9.1f} {'O' if same else 'X':>4s}"
              + (f" {sp:>7.1f}" if same else f" {'inf':>7s}") + mark)

    print("\n  ※ 에폭 수는 fold 2024 곡선에서 골랐다 → fold 2023이 표본 외 확인이다.")
    print("     2023에서 부호가 뒤집히면 그 설정은 채택하지 않는다.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
