"""exp31 — CatBoost 하이퍼파라미터 튜닝 2탄: depth 6→8 (용량 증가, 반대 방향).

exp30(depth 6→4, 용량 축소)이 세 폴드 전부 악화시켰다(A −18.8, B −27.0, C −25.3) —
fold A의 손실이 과적합 때문이 아니라는 뜻이고, 오히려 CatBoost가 이 문제에서
더 큰 용량을 원할 가능성을 시사한다. 반대 방향으로 depth 6→8을 시도.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import features as feat
from src.train_utils_cb import run_experiment_cb

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

CB_PARAMS = dict(learning_rate=0.03, depth=8, l2_leaf_reg=5.0)

if __name__ == "__main__":
    run_experiment_cb(
        label="exp31_catboost_depth8",
        feature_cols=feat.feature_list(use_custom=True),
        cat_cols=CAT_COLS,
        cb_params=CB_PARAMS,
        use_custom=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="CatBoost depth 6->8 (capacity increase) vs exp20 -- opposite direction from exp30, "
              "since depth=4 hurt all folds",
    )
