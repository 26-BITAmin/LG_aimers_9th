"""exp29 — CatBoost 위에 exp18(다년도 시즌 궤적)을 재시도.

exp18은 LightGBM(exp04)에서 fold C가 −14.4로 이 세션 최대 폭 하락, exp19 부트스트랩
CI [−31.98, 4.13]도 0에 가장 가깝게 다가간 음수 쪽이었다. exp20의 CatBoost 설정 위에
asof_success_R_last_season_blend_static/asof_success_R_season_slope_static만 추가해
CatBoost에서 재시도 — 네 피처 중 LightGBM에서 가장 부정적이었던 만큼 CatBoost에서도
가장 위험한 후보.
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
        label="exp29_catboost_season_trend",
        feature_cols=feat.feature_list(use_custom=True, use_season_trend=True),
        cat_cols=CAT_COLS,
        cb_params=CB_PARAMS,
        use_custom=True,
        use_season_trend=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="CatBoost + asof_success_R_last_season_blend_static / asof_success_R_season_slope_static "
              "(was worst LightGBM result, exp18-19), on top of exp20",
    )
