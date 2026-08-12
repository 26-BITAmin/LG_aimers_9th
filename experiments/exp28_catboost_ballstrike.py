"""exp28 — CatBoost 위에 exp12(ball/strike R전용 재구성)을 재시도.

exp12는 LightGBM(exp04)에서 fold B만 개선, A/C는 flat~악화로 exp13 부트스트랩에서
잡음 판정났다. exp20의 CatBoost 설정 위에 asof_ball_R_static/asof_strike_R_static만
추가해 CatBoost에서 재시도.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import features as feat
from src.train_utils_cb import run_experiment_cb

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

CB_PARAMS = dict(learning_rate=0.03, depth=6, l2_leaf_reg=5.0)

if __name__ == "__main__":
    run_experiment_cb(
        label="exp28_catboost_ballstrike",
        feature_cols=feat.feature_list(use_custom=True, use_ballstrike=True),
        cat_cols=CAT_COLS,
        cb_params=CB_PARAMS,
        use_custom=True,
        use_ballstrike=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="CatBoost + asof_ball_R_static / asof_strike_R_static "
              "(was noise on LightGBM/exp12-13), on top of exp20",
    )
