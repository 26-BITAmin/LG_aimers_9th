"""exp08 — 피처 엔지니어링 1탄: "최근 컨디션" 트렌드 피처.

공식 prev1/3/5게임 성공률은 이미 모델에 들어가 있지만, 절대 수준(career rate와 섞여
해석하기 애매함)으로만 존재한다. `prev*_game_success_rate - asof_pitcher_success_rate`
로 "최근이 커리어 평균보다 얼마나 좋았는지/나빴는지" 편차를 명시적으로 뽑아준다.
전부 이미 제공되는 공식 컬럼의 행 단위 뺄셈이라 frozen lookup 같은 특별 취급이
필요 없고 배포 규칙과도 완전히 무관하게 안전하다. exp04(현재 최선) 위에 얹어서 비교.
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
        label="exp08_trend_features",
        feature_cols=feat.feature_list(use_custom=True, use_trend=True),
        cat_cols=CAT_COLS,
        lgb_params=LGB_PARAMS,
        use_custom=True,
        use_trend=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="+ trend_prev1/3/5 = prevN_game_success_rate - career asof_pitcher_success_rate, on top of exp04",
    )
