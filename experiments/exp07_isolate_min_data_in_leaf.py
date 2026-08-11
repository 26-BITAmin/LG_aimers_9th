"""exp07 — exp05은 num_leaves와 min_data_in_leaf를 동시에 바꿔서 어느 쪽이 부호를
망가뜨렸는지 알 수 없었다. num_leaves는 exp04(31)로 되돌리고 min_data_in_leaf만
500->300으로 낮춰서 단독 효과를 본다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import features as feat
from src.train_utils import run_experiment

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03,
    num_leaves=31,
    min_data_in_leaf=300,
    lambda_l2=5.0,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
)

if __name__ == "__main__":
    run_experiment(
        label="exp07_isolate_min_data_in_leaf",
        feature_cols=feat.feature_list(use_custom=True),
        cat_cols=CAT_COLS,
        lgb_params=LGB_PARAMS,
        use_custom=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="isolate: num_leaves back to 31 (exp04), only min_data_in_leaf 500->300",
    )
