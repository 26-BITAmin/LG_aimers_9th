"""exp16 — 피처 엔지니어링 5탄: frozen 표본 크기(n)를 그 자체로 노출.

지금까지 `asof_success_R_platoon_blend_static`은 `k_blend=30` 고정값으로
same/diff-hand 갈래의 신뢰도를 셰이딩했다 — 표본이 적으면 무조건 career rate
쪽으로 당기는 규칙을 하드코딩한 것. 하지만 이 고정 k=30이 정말 맞는 강도인지는
검증한 적이 없고, 트리에 표본 크기 자체를 보여주면 각 리프에서 그 크기별로
다른 신뢰도 임계값을 스스로 배울 수 있다 (예: n<10인 콜드스타트 투수와
n>500인 베테랑을 다르게 취급).

`asof_success_R2_n_static` = 투수의 frozen 구간 전체 R전용 투구 수. 새 정보를
만드는 게 아니라 이미 계산되고 있던 값(`n_R2`)을 컬럼으로 노출만 하는 것이라
가장 저비용/저위험 실험. exp04 위에 이것 하나만 추가.
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
        label="exp16_n_exposed",
        feature_cols=feat.feature_list(use_custom=True, use_n_exposed=True),
        cat_cols=CAT_COLS,
        lgb_params=LGB_PARAMS,
        use_custom=True,
        use_n_exposed=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="+ asof_success_R2_n_static (frozen R-only pitch count, lets tree learn its own "
              "credibility instead of fixed k_blend=30), on top of exp04",
    )
