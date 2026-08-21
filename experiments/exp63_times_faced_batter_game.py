"""exp63 -- 신규 원시 피처: 이 경기에서 이 투수가 이 타자를 몇 번째 상대하는가
(times_faced_batter_this_game, TTO/times-through-the-order 유사 축).

exp62와 같은 game_id 경계 복원을 쓴다 (팀 쌍 변화 + 1회초 클린스타트 패턴, 검증
내용은 exp62 docstring 참조 -- 경기 4,872개, 시즌당 797~825, 경계 100% 일치).

세이버메트릭스에서 잘 알려진 효과: 타자가 같은 투수를 한 경기 안에서 2번째,
3번째 상대할수록 유리해진다(투구 패턴 노출 누적). `control_success`가 "제구
성공"이라는 투수 관점 라벨이라 방향은 선험적으로 불확실하지만(타자가 유리해지는
것과 투수 제구가 무너지는 것은 다른 축일 수 있음), 매치업 반복 자체가 이 데이터에
없던 완전히 새로운 정보라 시도해볼 값어치가 있다.

`times_faced_batter_this_game` = 이 투구 이전에 이 경기에서 이 투수-타자 조합이
이미 몇 번 있었는지 (0-indexed, strictly-prior, target 미사용 -- balls_before와
동급의 구조적 카운터라 train-only fit 불필요).

champion 51피처 + LightGBM/CatBoost 50/50 앙상블(exp57/58/61/62와 동일 구성, 단일
seed=0) 위에 이 피처 하나만 추가해서 fold A/B/C에서 잰다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src import features as feat
from src import metrics as M
from src.train_utils import load_train, run_fold
from src.train_utils_cb import run_fold_cb

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
)
CB_PARAMS = dict(learning_rate=0.03, depth=8, l2_leaf_reg=5.0)


def add_game_context(df):
    """경기 경계 복원 + 당경기 투수-타자 매치업 횟수 (exp62.add_game_context와 동일한
    경계 로직, 이 실험 전용 컬럼만 다름). df는 row_id 오름차순이어야 한다."""
    assert (df["row_id"].to_numpy() == np.sort(df["row_id"].to_numpy())).all(), \
        "df must be row_id-sorted before game context reconstruction"
    d = df.copy()

    pair = [frozenset(p) for p in zip(d["pitcher_team_id"].to_numpy(), d["batter_team_id"].to_numpy())]
    prev_pair = [None] + pair[:-1]
    pair_changed = np.array([a != b for a, b in zip(pair, prev_pair)])
    pair_changed[0] = True

    clean_start = ((d["inning"] == 1) & (d["top_bottom"] == "T") & (d["outs_before"] == 0)
                   & (d["balls_before"] == 0) & (d["strikes_before"] == 0)
                   & (d["run_total_before"] == 0)).to_numpy()
    prev_pitcher = np.r_[np.nan, d["pitcher_id"].to_numpy()[:-1]]
    pitcher_changed = d["pitcher_id"].to_numpy() != prev_pitcher
    boundary = pair_changed | (clean_start & pitcher_changed)
    d["game_id"] = pd.Series(boundary).cumsum().to_numpy()

    d["times_faced_batter_this_game"] = d.groupby(
        ["game_id", "pitcher_id", "batter_id"]).cumcount().to_numpy().astype(np.float32)
    return d


def sanity_check(df):
    d = add_game_context(df)
    n_games = int(d["game_id"].max())
    b = d.drop_duplicates("game_id", keep="first")
    clean = ((b["inning"] == 1) & (b["top_bottom"] == "T") & (b["outs_before"] == 0)
             & (b["balls_before"] == 0) & (b["strikes_before"] == 0) & (b["run_total_before"] == 0))
    assert clean.all(), f"{(~clean).sum()} game boundaries don't match clean-start pattern"

    # 직접 계산 대조: 표본 하나 골라 매치업 반복 횟수를 손으로 세서 비교
    gid0 = d["game_id"].iloc[100000]
    sub = d[d["game_id"] == gid0].sort_values("row_id")
    p0 = sub["pitcher_id"].iloc[0]
    psub = sub[sub["pitcher_id"] == p0]
    b0 = psub["batter_id"].iloc[0]
    seg = psub[psub["batter_id"] == b0]
    direct = np.arange(len(seg))
    got = seg["times_faced_batter_this_game"].to_numpy()
    assert (direct == got).all(), f"cumcount mismatch: {direct} vs {got}"

    dist = d["times_faced_batter_this_game"].value_counts().sort_index()
    print(f"  sanity check OK -- games={n_games}  direct-count match on sample matchup "
          f"(pitcher={p0}, batter={b0}, len={len(seg)}) OK")
    print(f"  value distribution (0=1st encounter this game): "
          f"{dict(dist.head(6))}")
    return d


if __name__ == "__main__":
    df = load_train()
    print("=== sanity check ===")
    df = sanity_check(df)

    feature_cols = feat.feature_list(use_custom=True)
    feature_cols_new = feature_cols + ["times_faced_batter_this_game"]

    for fold_name in ("A", "B", "C"):
        print(f"=== fold {fold_name} ===")
        res_lgb, _, p_lgb, y_lgb, valid = run_fold(df, fold_name, feature_cols, CAT_COLS, LGB_PARAMS,
                                                     use_custom=True, num_boost_round=3000, early_stopping_rounds=100)
        res_cb, _, p_cb, y_cb, valid_cb = run_fold_cb(df, fold_name, feature_cols, CAT_COLS, CB_PARAMS,
                                                        use_custom=True, num_boost_round=3000, early_stopping_rounds=100)
        assert (y_lgb == y_cb).all()
        p_ens = 0.5 * p_lgb + 0.5 * p_cb
        sc_baseline = M.score(p_ens, y_lgb)

        res_lgb2, _, p_lgb2, y_lgb2, _ = run_fold(df, fold_name, feature_cols_new, CAT_COLS, LGB_PARAMS,
                                                    use_custom=True, num_boost_round=3000, early_stopping_rounds=100)
        res_cb2, _, p_cb2, y_cb2, _ = run_fold_cb(df, fold_name, feature_cols_new, CAT_COLS, CB_PARAMS,
                                                    use_custom=True, num_boost_round=3000, early_stopping_rounds=100)
        assert (y_lgb2 == y_cb2).all()
        p_ens2 = 0.5 * p_lgb2 + 0.5 * p_cb2
        sc_new = M.score(p_ens2, y_lgb2)

        print(f"  baseline(champion 51 feat)      = {sc_baseline:.1f}")
        print(f"  + times_faced_batter_this_game  = {sc_new:.1f}  (Delta={sc_new - sc_baseline:+.1f})")
