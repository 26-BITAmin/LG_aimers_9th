"""exp14 — 피처 엔지니어링 4탄: 실패 성분(reverse/middle)의 좌우 매치업 분할.

exp09에서 reverse/middle을 R전용으로 재구성했지만(마진널, 잡음권) 좌우 분할은
시도하지 않았다. success 피처(exp04)의 same/diff-hand + credibility-blend가
+85~97점을 낸 것과 같은 메커니즘을 그대로 reverse/middle에 재사용한다.
근거: "reverse"는 포수 요구 코스의 반대쪽으로 빠진 투구라는 정의상, 타자
좌우에 따라 갈릴 개연성이 구조적으로 있다 (success보다 오히려 더 그럴듯할 수
있음 — 몸쪽/바깥쪽 개념 자체가 타자 좌우 기준이라서).

asof_reverse_R_platoon_blend_static / asof_middle_R_platoon_blend_static를
exp04 위에 추가. (exp09의 unsplit R전용 버전과는 별개 — 대체가 아니라 신규 시도.)
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
        label="exp14_failmode_platoon",
        feature_cols=feat.feature_list(use_custom=True, use_failmode_platoon=True),
        cat_cols=CAT_COLS,
        lgb_params=LGB_PARAMS,
        use_custom=True,
        use_failmode_platoon=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="+ asof_reverse_R_platoon_blend_static / asof_middle_R_platoon_blend_static "
              "(same/diff-hand + credibility blend, k=30), on top of exp04",
    )
