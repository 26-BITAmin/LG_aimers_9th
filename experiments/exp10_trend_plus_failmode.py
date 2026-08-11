"""exp10 — exp08(trend)과 exp09(failmode)를 동시에 얹어본다.

둘 다 단독으로는 fold 부호가 갈렸지만(A 약화 / B,C 대체로 개선 쪽), 겹치는 정보가
아니라 서로 다른 각도(최근 폼 vs 실패 성분 구성)라 같이 넣으면 누적될 수도 있다.
아니라면 exp04가 여전히 최선이라는 뜻.
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
        label="exp10_trend_plus_failmode",
        feature_cols=feat.feature_list(use_custom=True, use_trend=True, use_failmode=True),
        cat_cols=CAT_COLS,
        lgb_params=LGB_PARAMS,
        use_custom=True,
        use_trend=True,
        use_failmode=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="exp08 trend + exp09 failmode combined, on top of exp04",
    )
