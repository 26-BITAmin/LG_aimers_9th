"""exp27 — CatBoost 위에 exp14(실패 성분 좌우매치업 분할)를 재시도.

exp14는 LightGBM(exp04)에서 fold B만 크게 개선, A/C는 flat~악화(부호 불일치)로
exp15 부트스트랩에서 잡음 판정났다. exp20의 CatBoost 설정(depth=6, l2_leaf_reg=5,
lr=0.03) 위에 asof_reverse_R_platoon_blend_static/asof_middle_R_platoon_blend_static만
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
        label="exp27_catboost_failmode_platoon",
        feature_cols=feat.feature_list(use_custom=True, use_failmode_platoon=True),
        cat_cols=CAT_COLS,
        cb_params=CB_PARAMS,
        use_custom=True,
        use_failmode_platoon=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="CatBoost + asof_reverse_R_platoon_blend_static / asof_middle_R_platoon_blend_static "
              "(was noise on LightGBM/exp14-15), on top of exp20",
    )
