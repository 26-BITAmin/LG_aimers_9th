"""exp37 — CatBoost 하이퍼파라미터 튜닝 3탄: depth 8→10 (용량 증가 트렌드 연장).

exp30(depth=4)은 세 폴드 다 악화, exp31(depth=8)은 fold C +25.6(큰 개선), A +1.9(거의
동일), B −20.7(악화하지만 절대점수는 여전히 LightGBM보다 훨씬 위) -- depth를 올릴수록
fold C가 계속 좋아지는 단조 트렌드. 자연스러운 다음 지점인 depth=10을 시도해 이 트렌드가
계속되는지, 아니면 fold C에서도 꺾이는지 확인.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import features as feat
from src.train_utils_cb import run_experiment_cb

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

CB_PARAMS = dict(learning_rate=0.03, depth=10, l2_leaf_reg=5.0)

if __name__ == "__main__":
    run_experiment_cb(
        label="exp37_catboost_depth10",
        feature_cols=feat.feature_list(use_custom=True),
        cat_cols=CAT_COLS,
        cb_params=CB_PARAMS,
        use_custom=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="CatBoost depth 8->10 (capacity increase, extending exp31's monotonic trend) vs exp20/exp31",
    )
