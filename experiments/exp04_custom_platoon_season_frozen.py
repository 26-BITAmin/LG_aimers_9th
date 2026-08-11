"""exp04 — exp03의 train/serve skew 버그를 고친 버전.

exp03은 학습 중엔 행 단위 strictly-prior 누적(고유값 27만개, std 0.058)을 쓰고
검증 중엔 투수당 상수 1개(고유값 225개, std 0.047)를 썼다 — 분포 질감이 달라
트리 분기가 어긋났고 fold A/B/C 전부 exp02보다 나빠졌다.

src/features.py를 고쳐 학습 쪽도 "시즌 시작 시점에 얼어붙은 이전 시즌 누적"으로
바꿨다 (배포 시 검증/테스트가 보는 것과 같은 질감). 이 실험은 그 수정이 실제로
exp02 대비 개선을 만드는지 확인한다.
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
        label="exp04_custom_platoon_season_frozen",
        feature_cols=feat.feature_list(use_custom=True),
        cat_cols=CAT_COLS,
        lgb_params=LGB_PARAMS,
        use_custom=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="fixed: season-start frozen snapshot for training rows too (matches deploy texture)",
    )
