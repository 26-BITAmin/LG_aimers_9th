"""exp20 — 첫 CatBoost 시도. 사용자가 단일모델/노앙상블 제약을 풀어줬다 (exp12~19 네 갈래
피처 실험이 전부 부트스트랩 CI에서 잡음 판정난 뒤 — model.md 참조).

exp04와 완전히 동일한 피처셋(공식 47 + CUSTOM_STATIC_FEATURES)·동일 폴드로 모델만
LightGBM → CatBoost로 바꿔 순수 알고리즘 비교부터 한다 (피처를 동시에 바꾸면
무엇 때문에 달라졌는지 알 수 없음 — CLAUDE.md "한 번에 하나만"). pitcher_id/batter_id는
exp04처럼 수치형으로 유지 — CatBoost의 ordered target statistics가 고카디널리티
범주형을 LightGBM보다 안전하게 다룰 수 있다는 게 흥미롭지만, 그건 다음 실험에서
따로 격리해서 본다.

`num_leaves=31`(LightGBM)에 준하는 트리 용량으로 `depth=6`, 동일한 `lambda_l2=5`,
`learning_rate=0.03`을 그대로 맞췄다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import features as feat
from src.train_utils_cb import run_experiment_cb

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

CB_PARAMS = dict(
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=5.0,
)

if __name__ == "__main__":
    run_experiment_cb(
        label="exp20_catboost_baseline",
        feature_cols=feat.feature_list(use_custom=True),
        cat_cols=CAT_COLS,
        cb_params=CB_PARAMS,
        use_custom=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="CatBoost, exp04-identical feature set (official 47 + CUSTOM_STATIC_FEATURES), "
              "pitcher_id/batter_id numeric, depth=6/l2=5/lr=0.03 (LightGBM-matched capacity)",
    )
