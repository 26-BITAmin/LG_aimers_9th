"""exp02 — exp01 진단: pitcher_id/batter_id를 범주형으로 넣으면 792/830개 카테고리에 대해
개별 분기를 만들 수 있어 24~39라운드 만에 과적합(early stop)했다. RF 베이스라인도 이 두
컬럼을 그냥 숫자로 넣는다 (competition.md §6 파이프라인 참고). 여기서는

1. pitcher_id/batter_id를 수치형으로 되돌리고 (팀ID/손잡이/base_state 등 저카디널리티만 범주형 유지)
2. min_data_in_leaf를 키우고 학습률을 낮춰 더 많은 라운드를 버티게 규제

무엇이 저조한 점수의 원인이었는지 분리해서 본다.
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
    min_data_in_leaf=500,
    lambda_l2=5.0,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
)

if __name__ == "__main__":
    run_experiment(
        label="exp02_numeric_ids_regularized",
        feature_cols=feat.feature_list(use_custom=False),
        cat_cols=CAT_COLS,
        lgb_params=LGB_PARAMS,
        use_custom=False,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="pitcher_id/batter_id numeric (not categorical), min_data_in_leaf=500, lr=0.03",
    )
