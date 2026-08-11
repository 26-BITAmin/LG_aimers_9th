"""exp09 — 피처 엔지니어링 2탄: 실패 성분(reverse/middle)의 R 전용 재구성.

exp02/04에서 성공률 하나만 R 전용으로 다시 만들어도 컸다(RF anchor 대비 압도적 이득).
같은 논리가 실패 성분에도 적용된다: 공식 asof_pitcher_reverse_rate/middle_rate도
success_rate와 똑같이 2022 이전 F 오염이 섞여 있다(EDA_정리.md §5). success와 같은
방식(시즌 단위 frozen, R 전용)으로 reverse/middle 누적률을 다시 만들어 추가한다.

reverse/middle의 실현값은 train.csv에서만 파생 가능한 EDA/artifacts/decoded_pitch_outcomes.parquet
(노트북 02 산출물)를 쓴다 — test 행은 전혀 건드리지 않으므로 배포 규칙과 무관하게 항상 안전.
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
        label="exp09_failmode_features",
        feature_cols=feat.feature_list(use_custom=True, use_failmode=True),
        cat_cols=CAT_COLS,
        lgb_params=LGB_PARAMS,
        use_custom=True,
        use_failmode=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="+ asof_reverse_R_static / asof_middle_R_static (season-frozen, R-only), on top of exp04",
    )
