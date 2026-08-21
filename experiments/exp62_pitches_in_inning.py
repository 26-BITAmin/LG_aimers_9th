"""exp62 -- 신규 원시 피처: 이닝당 투구수 (pitches_in_half_inning).

사용자 제안: game_id가 데이터에 없지만, (pitcher_team_id, batter_team_id) 팀 쌍이
바뀌는 지점 + (이닝=1,초,0-0카운트,0아웃,0득점) 패턴에서 선발투수가 바뀌는 지점을
합쳐 경기 경계를 복원할 수 있다. 검증: 전체 train.csv(147만 행)에 적용해
경기 4,872개 검출, 시즌당 797~825개(6시즌 내내 일정), 경기당 투구수 278~325(중앙값
300, 실제 KBO 한 경기 투구수 범위와 일치), 경계 지점 100%가 "1회초 0-0/0아웃/0득점"
패턴과 역방향으로 일치.

이 피처(`pitches_in_half_inning`)는 target을 전혀 안 쓰는 순수 구조적 카운터라서
(cumsum() - 자기값과 같은 strictly-prior 규칙이 필요 없음 -- balls_before/outs_before와
동급의 "이미 주어진 상황 정보") train 구간에서만 fit할 필요가 없다. df 전체에 대해
한 번에 계산해서 원시 컬럼으로 얹는다.

가설: 이닝 안에서 이미 여러 개 던진 투수는 그 이닝 리듬/피로가 누적돼 제구가
달라질 수 있다 (짧은 스팬이라 등판 간격 피로도보다 신호가 살아있을 가능성).

champion 51피처 + LightGBM/CatBoost 50/50 앙상블(exp57/58/61과 동일 구성, 단일
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
    """경기 경계 복원 + 이닝당 투구수. df는 이미 row_id 오름차순(load_train() 그대로)
    이어야 한다 -- groupby.cumcount()가 행 순서를 그대로 쓰기 때문."""
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

    d["pitches_in_half_inning"] = d.groupby(
        ["game_id", "pitcher_id", "inning", "top_bottom"]).cumcount().to_numpy().astype(np.float32)
    return d


def sanity_check(df):
    d = add_game_context(df)
    n_games = int(d["game_id"].max())
    sizes = d.groupby("game_id").size()
    per_season = d.groupby("game_id")["season"].first().value_counts().sort_index()
    b = d.drop_duplicates("game_id", keep="first")
    clean = ((b["inning"] == 1) & (b["top_bottom"] == "T") & (b["outs_before"] == 0)
             & (b["balls_before"] == 0) & (b["strikes_before"] == 0) & (b["run_total_before"] == 0))
    assert clean.all(), f"{(~clean).sum()} game boundaries don't match clean-start pattern"
    assert 700 <= per_season.min() and per_season.max() <= 900, \
        f"per-season game count out of plausible KBO range: {per_season.to_dict()}"
    assert sizes.median() > 250, f"median game size too small: {sizes.median()}"

    # 직접 계산 대조: 표본 하나 골라 cumcount와 손으로 센 값이 일치하는지
    gid0 = d["game_id"].iloc[100000]
    sub = d[d["game_id"] == gid0].sort_values("row_id")
    p0 = sub["pitcher_id"].iloc[0]
    psub = sub[sub["pitcher_id"] == p0]
    inn0, tb0 = psub["inning"].iloc[0], psub["top_bottom"].iloc[0]
    seg = psub[(psub["inning"] == inn0) & (psub["top_bottom"] == tb0)]
    direct = np.arange(len(seg))
    got = seg["pitches_in_half_inning"].to_numpy()
    assert (direct == got).all(), f"cumcount mismatch: {direct} vs {got}"

    print(f"  sanity check OK -- games={n_games}  per_season={dict(per_season)}  "
          f"size(median/min/max)={sizes.median():.0f}/{sizes.min()}/{sizes.max()}  "
          f"direct-count match on sample segment (len={len(seg)}) OK")
    return d


if __name__ == "__main__":
    df = load_train()
    print("=== sanity check ===")
    df = sanity_check(df)

    feature_cols = feat.feature_list(use_custom=True)
    feature_cols_new = feature_cols + ["pitches_in_half_inning"]

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

        print(f"  baseline(champion 51 feat) = {sc_baseline:.1f}")
        print(f"  + pitches_in_half_inning   = {sc_new:.1f}  (Delta={sc_new - sc_baseline:+.1f})")
