"""exp49 — 투수→팀→리그 3단 계층적 축소 (신규 피처, 사용자 선택 A).

지금 최고 피처 `asof_success_R_platoon_blend_static`은 투수 표본이 적으면 곧장
**리그 전체 평균**으로 당겨진다(k=30). 그 사이에 "팀" 단계가 없다 — 신인이나
표본이 적은 투수는 자기 팀의 코칭스태프·포수진 등 체계적 효과를 전혀 못 받고
리그 평균으로 뭉개진다. `src/features.py`에 `fit_team_lookup` + `_team_hier_blend`를
추가해 투수→팀→리그 3단 credibility-blend를 만들었다 (신규 피처
`asof_success_R_team_blend_static`, `TEAM_HIER_FEATURES`).

R전용, season-frozen(strictly-prior) — 기존 `fit_pitcher_lookup`과 동일한 안전
패턴(train에서만 fit, valid/test는 조회만). 2025 신인 비율이 과거 fold보다 높을
수 있다는 우려(CLAUDE.md/model.md)에 직접 대응하는 피처.

이 스크립트는 (1) 새 lookup/blend 함수가 직접 계산과 일치하는지 단위 검증하고
(2) LightGBM(exp04 config) 기준선 위에 이 피처 하나만 추가했을 때 fold A/B/C에서
어떻게 바뀌는지를 잰다 — 한 번에 하나만 바꾼다 원칙에 따라 기존 4개 커스텀
피처(platoon_blend 포함)는 그대로 두고 이것만 얹는다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src import features as feat
from src import folds as F
from src.train_utils import load_train, run_experiment

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


def sanity_check(df):
    """fit_team_lookup / _team_hier_blend를 fold C에서 직접 계산과 대조."""
    train_mask, valid_mask = F.split(df, "C")
    fit_raw = df.loc[train_mask]
    team_lookup, league_rate = feat.fit_team_lookup(fit_raw)

    R = fit_raw.loc[fit_raw["game_type"] == "R"]
    direct_league_rate = R[feat.TARGET].mean()
    assert abs(league_rate - direct_league_rate) < 1e-9, \
        f"league_rate mismatch: {league_rate} vs {direct_league_rate}"

    # 표본 많은 팀 하나 골라 직접 계산과 대조
    team_id = R["pitcher_team_id"].value_counts().index[0]
    direct_n = int((R["pitcher_team_id"] == team_id).sum())
    direct_rate = R.loc[R["pitcher_team_id"] == team_id, feat.TARGET].mean()
    got_n = team_lookup.loc[team_id, "n_team"]
    got_rate = team_lookup.loc[team_id, "rate_team"]
    assert got_n == direct_n, f"n_team mismatch: {got_n} vs {direct_n}"
    assert abs(got_rate - direct_rate) < 1e-9, f"rate_team mismatch: {got_rate} vs {direct_rate}"

    # 콜드스타트(팀도 없는 team_id) 시뮬레이션 -- league_rate로 완전히 fall back 하는지
    blend = feat._team_hier_blend(
        pitcher_rate=np.array([np.nan]), pitcher_n=np.array([np.nan]),
        team_rate=np.array([np.nan]), team_n=np.array([np.nan]),
        league_rate=league_rate)
    assert abs(blend[0] - league_rate) < 1e-9, f"cold-start fallback broken: {blend[0]} vs {league_rate}"

    # 표본 아주 많은 투수(k=30 대비)는 거의 자기 rate로 수렴하는지
    pid = R["pitcher_id"].value_counts().index[0]
    direct_p_n = int((R["pitcher_id"] == pid).sum())
    direct_p_rate = R.loc[R["pitcher_id"] == pid, feat.TARGET].mean()
    blend2 = feat._team_hier_blend(
        pitcher_rate=np.array([direct_p_rate]), pitcher_n=np.array([direct_p_n]),
        team_rate=np.array([direct_rate]), team_n=np.array([direct_n]),
        league_rate=league_rate)
    assert abs(blend2[0] - direct_p_rate) < 0.01, \
        f"high-n pitcher should converge to own rate: {blend2[0]} vs {direct_p_rate}"

    print(f"  sanity check OK -- league_rate={league_rate:.4f}  "
          f"team{team_id} n={direct_n} rate={direct_rate:.4f}  "
          f"pitcher{pid} n={direct_p_n} rate={direct_p_rate:.4f} blend={blend2[0]:.4f}")


if __name__ == "__main__":
    df = load_train()
    print("=== sanity check ===")
    sanity_check(df)

    feature_cols = feat.feature_list(use_custom=True, use_team_hier=True)
    print(f"\n=== exp49_team_hier_shrinkage ===  features={len(feature_cols)} "
          f"(champion 51 + 1 new: asof_success_R_team_blend_static)")

    run_experiment(
        label="exp49_team_hier_shrinkage",
        feature_cols=feature_cols,
        cat_cols=CAT_COLS,
        lgb_params=LGB_PARAMS,
        use_custom=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        use_team_hier=True,
        notes="new feature: pitcher-team-league 3-tier hierarchical shrinkage "
              "(asof_success_R_team_blend_static, k_team=500) added on top of exp04 "
              "champion custom features (51->52 feature_cols) -- addresses cold-start "
              "for low-n/rookie pitchers who currently fall straight to league mean",
    )
