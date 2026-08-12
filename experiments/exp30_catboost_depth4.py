"""exp30 — CatBoost 하이퍼파라미터 튜닝 1탄: depth 6→4 (용량 축소).

exp20~29까지 CatBoost 파라미터(depth=6, l2_leaf_reg=5, lr=0.03)는 전부 LightGBM
exp04 용량에 그냥 맞춘 값이었지 CatBoost용으로 따로 튜닝한 적이 없다. fold A에서만
CatBoost가 LightGBM에 유의미하게 졌는데(exp21, Δ=−35.85), 원인 후보 중 하나는
CatBoost가 fold A의 오래된 학습 구간(2019-21)에서 과적합했을 가능성이다.

LightGBM에서는 num_leaves 31→63(exp05, 용량 증가)이 A/C 악화·B만 개선으로
나빴었다 — 반대 방향(용량 축소)이 fold A를 구할 수 있는지 확인. `depth`만 6→4로
낮추고 나머지는 exp20과 동일하게 유지 (한 번에 하나만).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import features as feat
from src.train_utils_cb import run_experiment_cb

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

CB_PARAMS = dict(learning_rate=0.03, depth=4, l2_leaf_reg=5.0)

if __name__ == "__main__":
    run_experiment_cb(
        label="exp30_catboost_depth4",
        feature_cols=feat.feature_list(use_custom=True),
        cat_cols=CAT_COLS,
        cb_params=CB_PARAMS,
        use_custom=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="CatBoost depth 6->4 (capacity reduction) vs exp20, isolating whether fold A "
              "loss to LightGBM was overfitting-driven",
    )
