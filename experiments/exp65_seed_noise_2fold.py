"""exp65 -- 2-fold(wonwoo 방식) 프로토콜 전환 후 seed 노이즈 게이트 재측정.

exp47(A/B/C 롤링 3시즌 창, LightGBM 단일)의 노이즈 게이트(A=9.6/B=12.1/C=15.2)는
2026-08-21 사용자 지시로 폐기된 옛 폴드 전용이었다. `src/folds.py`가 wonwoo의
2-fold 확장 윈도우("2023": train 2019-22 valid 2023, "2024": train 2019-23
valid 2024)로 완전히 바뀌었으니, 판정 조건 ①(크기 게이트)을 다시 쓰려면 이
새 폴드에서 노이즈를 다시 재야 한다. 방법은 exp47과 동일 -- exp04 config
(단일 LightGBM, custom 피처) 5개 seed로 재학습해 fold별 score 표준편차를 잰다.
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
    scores = {fn: [] for fn in ("2023", "2024")}

    for seed in SEEDS:
        results = run_experiment(
            label="exp65_seed_noise_2fold",
            feature_cols=feature_cols,
            cat_cols=CAT_COLS,
            lgb_params=LGB_PARAMS,
            use_custom=True,
            fold_names=("2023", "2024"),
            num_boost_round=3000,
            early_stopping_rounds=100,
            seed=seed,
            notes=f"seed noise measurement (exp04 config) on NEW wonwoo-style 2-fold "
                  f"protocol, seed={seed} of {SEEDS} -- replaces exp47's retired A/B/C gate",
        )
        for fn, res in results.items():
            scores[fn].append(res["score"])

    print("\n=== seed noise summary (exp04 config, 5 seeds, 2-fold protocol) ===")
    for fn in ("2023", "2024"):
        arr = np.array(scores[fn])
        print(f"  fold {fn}: scores={np.round(arr, 1).tolist()}  "
              f"mean={arr.mean():.1f}  std={arr.std(ddof=1):.2f}  "
              f"range=[{arr.min():.1f}, {arr.max():.1f}]")
