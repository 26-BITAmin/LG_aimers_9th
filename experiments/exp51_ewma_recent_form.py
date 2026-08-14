"""exp51 — R전용 투구단위 EWMA "최근 폼" 피처 (사용자 선택 B).

지금 갖고 있는 career 신호(`asof_success_R2_static`, `asof_success_R_platoon_blend_static`)는
전부 "얼어붙은 시점까지의 전체 평균"이라 최근성이 없다. 공식 TREND_FEATURES(prevN 게임)는
2022 이전 F 오염이 섞인 컬럼이라 신뢰 안 함. 게임 ID가 데이터에 없어(competition.md) 게임
단위 집계가 불가능하므로, `row_id` 연대순(CLAUDE.md 불변식)을 이용해 **투구 단위** 지수감쇠
평균을 season-frozen lookup으로 만든다 (half_life=150투구, 튜닝 안 함).

한 번에 하나만 바꾼다 원칙에 따라 기존 4개 커스텀 피처는 그대로 두고 이 피처 하나만 얹는다.
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
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
)


def sanity_check(df):
    """fit_pitcher_ewma를 fold C에서 직접 계산과 대조."""
    train_mask, valid_mask = F.split(df, "C")
    fit_raw = df.loc[train_mask]
    ewma_lookup = feat.fit_pitcher_ewma(fit_raw)

    R = fit_raw.loc[fit_raw["game_type"] == "R"].sort_values("row_id")
    pid = R["pitcher_id"].value_counts().index[0]
    sub = R.loc[R["pitcher_id"] == pid]
    n = len(sub)
    ranks = np.arange(n - 1, -1, -1, dtype=np.float64)  # 0 = 가장 최근
    w = 0.5 ** (ranks / feat.K_EWMA_HALFLIFE)
    y = sub[feat.TARGET].to_numpy(dtype=np.float64)
    direct = float((w * y).sum() / w.sum())
    got = ewma_lookup.loc[pid, "rate_ewma"]
    assert abs(got - direct) < 1e-9, f"ewma mismatch: {got} vs {direct}"

    # career 평균과는 달라야 정상 (감쇠가 실제로 작동한다는 증거) -- 우연히 같을 수도 있으니
    # 그냥 출력만
    career = y.mean()
    print(f"  sanity check OK -- pitcher{pid} n={n}  career_rate={career:.4f}  "
          f"ewma_rate(hl={feat.K_EWMA_HALFLIFE:.0f})={direct:.4f}")


if __name__ == "__main__":
    df = load_train()
    print("=== sanity check ===")
    sanity_check(df)

    feature_cols = feat.feature_list(use_custom=True, use_ewma=True)
    print(f"\n=== exp51_ewma_recent_form ===  features={len(feature_cols)} "
          f"(champion 51 + 1 new: asof_success_R_ewma_static)")

    run_experiment(
        label="exp51_ewma_recent_form",
        feature_cols=feature_cols,
        cat_cols=CAT_COLS,
        lgb_params=LGB_PARAMS,
        use_custom=True,
        fold_names=("A", "B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        use_ewma=True,
        notes="new feature: pitch-level EWMA recent-form (asof_success_R_ewma_static, "
              "half_life=150 pitches, R-only, row_id-ordered) added on top of exp04 "
              "champion custom features (51->52 feature_cols)",
    )
