"""exp25 — CatBoost 위에 exp16(frozen 표본 크기 n 노출)을 재시도.

exp16은 LightGBM(exp04)에서 이 세션 중 부호 일관성이 가장 좋았지만(A/B 개선, C 거의
그대로) 부트스트랩 CI(exp17)에서 결국 잡음 판정났다. 하지만 CatBoost는 트리 구조
(symmetric/oblivious)와 정규화 방식이 달라서 같은 피처가 다르게 작동할 수 있다 —
exp20이 이미 fold B/C에서 LightGBM과 완전히 다른 스케일의 반응을 보였다(사용자 지시로
CatBoost에서 재시도).

exp20과 동일한 CatBoost 설정(depth=6, l2_leaf_reg=5, lr=0.03) 위에
`asof_success_R2_n_static`만 추가.
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
        label="exp25_catboost_n_exposed",
        feature_cols=feat.feature_list(use_custom=True, use_n_exposed=True),
        cat_cols=CAT_COLS,
        cb_params=CB_PARAMS,
        use_custom=True,
        use_n_exposed=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="CatBoost + asof_success_R2_n_static (was noise on LightGBM/exp16-17), on top of exp20",
    )
