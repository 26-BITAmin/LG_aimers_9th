"""exp03 — exp02 설정에 커스텀 R 전용/좌우 매치업 피처를 추가.

노트북 05는 이 피처를 검증 시즌 안에서 계속 갱신되는 형태로 측정해 +85~+97점을 봤는데,
그건 실제 배포에서는 금지된 방식이다(competition.md §4-4: test 행 순서 기반 갱신 불가).
여기서는 src/features.py의 올바른 버전 — 학습 구간 끝에서 얼어붙은 투수별 정적 lookup —
을 쓴다. 검증 시즌 내내 값이 그대로 고정되므로 노트북 05보다 이득이 작을 것으로 예상된다.
그 차이 자체가 이번 실험의 관전 포인트.
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
        label="exp03_custom_platoon_frozen",
        feature_cols=feat.feature_list(use_custom=True),
        cat_cols=CAT_COLS,
        lgb_params=LGB_PARAMS,
        use_custom=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="+ asof_success_R2_static / R_platoon_static / R_platoon_blend_static (frozen at train cutoff, deployable)",
    )
