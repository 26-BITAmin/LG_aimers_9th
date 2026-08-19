"""달력 피처 (game_month, game_dayofweek) 제거 A/B — §7.15 재개봉 (현행 s23 구성).

왜 다시 재나
    §7.15 그룹 LOO에서 달력 제거가 +37.1/+8.5로 **제거 후보 1위**였다 (크기비 4.4).
    그러나 LB 검증은 51개 일괄 제거(s06, −2.13)로만 했고 달력 단독은 미검증.
    게다가 그 측정은 soft label·contam·aux 이전의 옛 구성이다.

방법
    ctl  = foldcache 캐시의 z_lgb (4변형 × 3시드, 현행 구성 그대로)
    drop = 같은 데이터·같은 시드에서 두 컬럼만 빼고 12부스터 재학습
    판정 = 쌍체 Δ(BSS) ± SE (ablate20.se_bss 방식 — 같은 행 차분으로 표본변동 소거)

한계
    LGBM 단독이다. MLP도 이 두 컬럼을 수치 채널로 먹으므로, 여기서 신호가 나오면
    블렌드 단계(MLP 재학습 포함)를 따로 재야 한다. 통과선: fold2024 t ≥ 3.

실행:  .venv/Scripts/python.exe -u src/calprobe.py
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
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, N_ROUNDS, SOFT_LABEL_F,
                    TRAIN_CSV)
from data import load_csv
from mlp import score
from validation import _sigmoid, shift_to_prior

FOLDS = [2023, 2024]
DROP = ["game_month", "game_dayofweek"]


def paired(zA, zB, y):
    """Δ(BSS, B−A) ± SE. 각 팔에 자기 oracle shift를 준 뒤 같은 행에서 차분한다."""
    pA = shift_to_prior(_sigmoid(zA), y.mean())
    pB = shift_to_prior(_sigmoid(zB), y.mean())
    d = (pA - y) ** 2 - (pB - y) ** 2          # 양수 = B가 좋다
    ref = y.mean() * (1 - y.mean())
    k = 1e5 / ref
    return k * d.mean(), k * d.std(ddof=1) / np.sqrt(len(d))


def main():
    t0 = time.time()
    cache = foldcache.load(FOLDS, verbose=False)
    df = load_csv(TRAIN_CSV)

    for f in FOLDS:
        z_ctl, _, y, _ = cache[f]
        Xtr, ytr, w, Xva, yva, art, tr = foldcache.fold_data(df, f)
        assert np.array_equal(y, yva)

        # foldcache.compute와 같은 순서: 보조 라벨 복원 → soften이 행을 늘린다.
        # aux 자체는 LGBM이 안 쓰지만 행 확장이 ctl 팔과 같아야 쌍체가 성립한다.
        aux, _ = modelabel.aux_labels(tr)
        if SOFT_LABEL_F:
            Xtr, ytr, w, aux, _ = soften.soften(tr, Xtr, ytr, w, aux)

        Xtr = Xtr.drop(columns=DROP)
        Xva_d = Xva.drop(columns=DROP)

        z_drop, n = np.zeros(len(Xva)), 0
        for over in ENSEMBLE_VARIANTS.values():
            for s in foldcache.LGB_SEEDS:
                p = dict(FINAL_LGB_PARAMS, **over, seed=s, bagging_seed=s,
                         feature_fraction_seed=s, data_random_seed=s)
                bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w),
                                num_boost_round=N_ROUNDS)
                z_drop += bst.predict(Xva_d, raw_score=True)
                n += 1
                del bst
        z_drop /= n

        dm, se = paired(z_ctl, z_drop, y)
        print(f"\nfold {f}  ctl {score(z_ctl, y):>7.1f}   drop {score(z_drop, y):>7.1f}"
              f"   Δ(drop−ctl) {dm:+.2f} ± {se:.2f}  (t {dm/se:+.1f})"
              f"   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
