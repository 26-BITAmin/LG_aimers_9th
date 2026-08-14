"""exp47 — 학습 seed 노이즈 실측 (wonwoo 브랜치 아이디어 도입).

wonwoo `src/decide.py`: fold별로 3~5개 seed로 재학습해 BSS 표준편차를 실측하고,
그 노이즈보다 작은 Δ는 판정 자체를 보류한다("조건①: 크기 게이트"). 우리는 지금까지
seed=0 고정으로 한 번만 학습해왔고, fold 간 격차 중 일부가 "진짜 신호"가 아니라
"그 seed의 운"일 가능성을 한 번도 재본 적이 없다.

exp04 config(단일 LightGBM, custom 피처, fold A/B/C)를 5개 seed로 재학습해
fold별 score(BSS) 표준편차를 잰다. `run_experiment`가 이미 `seed` 인자와
`log_result`(seed 컬럼 포함)를 지원하므로 배관은 안 건드리고 그대로 반복 호출한다.

결과는 results.csv에 label="exp47_seed_noise"로 15행(5 seed x 3 fold) 쌓이고,
이 스크립트 끝에서 fold별 mean/std를 요약 출력한다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src import features as feat
from src.train_utils import run_experiment

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03,
    num_leaves=31,
    min_data_in_leaf=500,
    lambda_l2=5.0,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
)

SEEDS = (0, 1, 2, 3, 4)

if __name__ == "__main__":
    feature_cols = feat.feature_list(use_custom=True)
    scores = {fn: [] for fn in ("A", "B", "C")}

    for seed in SEEDS:
        results = run_experiment(
            label="exp47_seed_noise",
            feature_cols=feature_cols,
            cat_cols=CAT_COLS,
            lgb_params=LGB_PARAMS,
            use_custom=True,
            fold_names=("A", "B", "C"),
            num_boost_round=3000,
            early_stopping_rounds=100,
            seed=seed,
            notes=f"seed noise measurement (exp04 config), seed={seed} of {SEEDS}",
        )
        for fn, res in results.items():
            scores[fn].append(res["score"])

    print("\n=== seed noise summary (exp04 config, 5 seeds) ===")
    for fn in ("A", "B", "C"):
        arr = np.array(scores[fn])
        print(f"  fold {fn}: scores={np.round(arr, 1).tolist()}  "
              f"mean={arr.mean():.1f}  std={arr.std(ddof=1):.2f}  "
              f"range=[{arr.min():.1f}, {arr.max():.1f}]")
