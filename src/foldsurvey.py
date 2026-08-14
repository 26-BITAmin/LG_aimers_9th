"""후보 fold들의 **신호량**을 싸게 잰다 — 검증 설계를 다시 짜기 위한 진단.

문제. 지금 fold는 2023과 2024 둘뿐인데 기준 BSS가 40.7 대 769.6으로 **19배** 다르다.
신호가 거의 없는 fold에서는 무엇을 섞어도 좋아 보이고(ablate21에서 25행 전부 그랬다),
그래서 조건 ③이 거의 모든 후보를 기각하고 있다. 판단 근거가 사실상 fold 2024 하나다.

가르려는 것:
    (A) 2023 시즌 자체가 예측하기 어렵다        → 시즌 내 분할도 약할 것
    (B) 2019~2022로 2023을 맞히는 게 어렵다     → 학습에 2023 전반기를 넣으면 강해질 것
둘은 처방이 다르다. (A)면 2023을 버려야 하고, (B)면 드리프트 문제라 fold 구성을 바꾸면 된다.

싸게 잰다: fold당 **base 변형 1개 · 시드 1개**. 신호량을 재는 데는 그걸로 충분하다
(앙상블 12개를 돌리면 fold당 6분이고 6개면 36분이다).
여기 숫자는 fold 간 **비교용**이지 성능 판정용이 아니다.

실행:  .venv/Scripts/python.exe -u src/foldsurvey.py
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
from config import (DROP_FEATURES, FINAL_LGB_PARAMS, N_ROUNDS, RECENCY_HALF_LIFE,
                    SEED, SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV,
                    USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts

SPLIT_MONTH = 7          # 7월 이후를 후반기로 본다 (KBO 정규시즌 3~10월)


def folds(df):
    """(이름, 학습 마스크, 검증 마스크). 학습은 항상 검증보다 시간상 앞이다."""
    s, m = df.season.to_numpy(), df.game_month.to_numpy()
    out = []
    for v in (2021, 2022, 2023, 2024):
        out.append((f"{v} 시즌", s < v, s == v))
    for v in (2023, 2024):
        out.append((f"{v} 후반기", (s < v) | ((s == v) & (m < SPLIT_MONTH)),
                    (s == v) & (m >= SPLIT_MONTH)))
    return out


def run(df, name, mtr, mva, t0):
    tr, va = df[mtr], df[mva]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    p = dict(FINAL_LGB_PARAMS, seed=SEED, bagging_seed=SEED,
             feature_fraction_seed=SEED, data_random_seed=SEED)
    bst = lgb.train(p, lgb.Dataset(X(tr), ytr, weight=w), num_boost_round=N_ROUNDS)
    z = bst.predict(X(va), raw_score=True)

    print(f"{name:<12s} 학습 {len(tr):>9,}  검증 {len(va):>8,}"
          f"  양성률 {ytr.mean():.4f}→{yva.mean():.4f}"
          f"  std(z) {z.std():.4f}   **BSS {mlp.score(z, yva):>7.1f}**"
          f"   ({time.time()-t0:.0f}s)")
    return mlp.score(z, yva)


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    res = {}
    for name, mtr, mva in folds(df):
        res[name] = run(df, name, mtr, mva, t0)

    print("\n  판정:")
    a, b = res.get("2023 시즌"), res.get("2023 후반기")
    if a is not None and b is not None:
        print(f"    2023 시즌 {a:.1f}  vs  2023 후반기 {b:.1f}")
        if b > a + 200:
            print("    → 학습에 2023 전반기를 넣으니 살아났다. **드리프트 문제(B)**.")
            print("       2023 시즌 자체가 예측 불가능한 게 아니다. fold 구성을 바꾸면 된다.")
        else:
            print("    → 전반기를 넣어도 약하다. **2023 시즌 자체의 문제(A)**.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
