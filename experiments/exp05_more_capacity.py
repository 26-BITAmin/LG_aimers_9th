"""exp05 — exp04 대비 트리 용량을 늘려본다.

exp04의 best_iter(125/82/152)가 num_boost_round 상한(3000)에 한참 못 미쳐 일찍 멈췄다.
num_leaves를 31→63으로 늘리고 min_data_in_leaf를 500→300으로 살짝 낮춰서 트리가 더
세밀한 분기를 잡을 여지를 준다. 과적합 여부는 fold A/B/C best_iter와 valid score로 바로
드러난다 (early stopping이 감시).
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
    num_leaves=63,
    min_data_in_leaf=300,
    lambda_l2=5.0,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
)

if __name__ == "__main__":
    run_experiment(
        label="exp05_more_capacity",
        feature_cols=feat.feature_list(use_custom=True),
        cat_cols=CAT_COLS,
        lgb_params=LGB_PARAMS,
        use_custom=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="num_leaves 31->63, min_data_in_leaf 500->300 vs exp04",
    )
