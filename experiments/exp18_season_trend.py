"""exp18 — 피처 엔지니어링 6탄: 다년도 시즌 단위 궤적 (질적으로 새로운 축).

exp12/14/16은 전부 "성공률 lookup을 변주"하는 계열(성분 재구성, 좌우분할, 표본크기
노출)이었고 셋 다 CI에서 잡음 판정났다. 이번엔 다른 축: 지금 `TREND_FEATURES`는
게임 단위(prevN game) "최근 컨디션"만 잡고(공식 컬럼이라 2022 이전 F 오염도 섞임),
투수별 다년도 장기 추세는 아직 다룬 적이 없다.

`asof_success_R_last_season_blend_static`: 가장 최근 얼어붙은 시즌 하나의 R전용
성공률 (credibility-blend, exp04 좌우매치업과 같은 메커니즘 재사용 — 표본 작으면
그 이전 career 쪽으로 당김).
`asof_success_R_season_slope_static`: 위 값 − 그 이전 전체 career R전용 성공률.
투수별 "최근 시즌이 이전보다 나아졌는지/나빠졌는지".

가설 근거: EDA_정리.md §3이 확인한 리그 전체 base rate 하락(r: .5495→.4897,
2019→2024)이 투수 개개인의 노화/기량 변화로도 갈릴 수 있다 — 실측 sanity check에서
slope 평균이 음수(-0.016~-0.021)로 나와 이 방향과 일치한다.
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
        label="exp18_season_trend",
        feature_cols=feat.feature_list(use_custom=True, use_season_trend=True),
        cat_cols=CAT_COLS,
        lgb_params=LGB_PARAMS,
        use_custom=True,
        use_season_trend=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="+ asof_success_R_last_season_blend_static / asof_success_R_season_slope_static "
              "(multi-season career trajectory, R-only, season-frozen), on top of exp04",
    )
