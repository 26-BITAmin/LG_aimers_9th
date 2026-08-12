"""exp12 — 피처 엔지니어링 3탄: ball/strike의 R 전용 재구성.

exp04 모델의 feature importance(gain)를 보면 `asof_pitcher_ball_rate`가 성공률 다음
2위(9193.6, success_rate 34023.7 대비)였다 — reverse_rate(4위)/middle_rate보다도 높다.
success/reverse/middle과 똑같이 ball_rate/strike_rate도 공식 asof_pitcher_*_rate라
2022 이전 F 오염이 섞여 있는데, 여긴 아직 R 전용으로 안 고쳤다. 중요도가 가장 높았던
성분을 아직 안 건드렸다는 뜻 — reverse/middle(exp09, 마진널)보다 이쪽이 더 유망한 배팅.
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
        label="exp12_ballstrike_features",
        feature_cols=feat.feature_list(use_custom=True, use_ballstrike=True),
        cat_cols=CAT_COLS,
        lgb_params=LGB_PARAMS,
        use_custom=True,
        use_ballstrike=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="+ asof_ball_R_static / asof_strike_R_static (season-frozen, R-only), on top of exp04",
    )
