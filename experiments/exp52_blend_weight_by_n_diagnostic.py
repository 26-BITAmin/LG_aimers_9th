"""exp52 — 표본크기 조건부 앙상블 결합 (사용자 선택 C), 1단계: 진단만.

exp34~35(고정 스칼라 블렌드 가중치 탐색)는 이미 기각됐다 -- nested holdout으로 고른
가중치가 fold A/B에서 유의미하게 나빠지고 fold C만 좋아짐(부호 불일치). 이건 다른
메커니즘이다: 전역 스칼라가 아니라 `asof_pitcher_n`(표본 크기) 구간별로 LightGBM과
CatBoost 중 어느 쪽이 이기는지 본다 -- 패턴이 있으면 그때 조건부 블렌드 규칙을
설계한다. 없으면(50/50이 전 구간에서 이미 최선이면) 여기서 접는다.

CLAUDE.md 층화 축("asof_pitcher_n 구간")을 그대로 사용. 채택 여부를 미리 정하지
않고 순수 진단만 -- 블렌드를 바꾸지 않는다.
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

N_BINS = 4

if __name__ == "__main__":
    df = load_train()
    feature_cols = feat.feature_list(use_custom=True)

    for fold_name in ("A", "B", "C"):
        print(f"=== fold {fold_name} ===")
        res_lgb, _, p_lgb, y_lgb, valid = run_fold(df, fold_name, feature_cols, CAT_COLS, LGB_PARAMS,
                                                     use_custom=True, num_boost_round=3000, early_stopping_rounds=100)
        res_cb, _, p_cb, y_cb, valid_cb = run_fold_cb(df, fold_name, feature_cols, CAT_COLS, CB_PARAMS,
                                                        use_custom=True, num_boost_round=3000, early_stopping_rounds=100)
        assert (y_lgb == y_cb).all()
        assert (valid["pitcher_id"].to_numpy() == valid_cb["pitcher_id"].to_numpy()).all()
        yv = y_lgb
        p_ens = 0.5 * p_lgb + 0.5 * p_cb
        n_p = valid["asof_pitcher_n"].to_numpy()

        print(f"  overall: LightGBM={M.score(p_lgb, yv):.1f}  CatBoost={M.score(p_cb, yv):.1f}  "
              f"50/50={M.score(p_ens, yv):.1f}")

        bins = pd.qcut(pd.Series(n_p), N_BINS, duplicates="drop")
        bin_df = pd.DataFrame({"bin": bins, "p_lgb": p_lgb, "p_cb": p_cb, "p_ens": p_ens, "y": yv})
        for b, g in bin_df.groupby("bin", observed=True):
            sc_lgb = M.score(g["p_lgb"].to_numpy(), g["y"].to_numpy())
            sc_cb = M.score(g["p_cb"].to_numpy(), g["y"].to_numpy())
            sc_ens = M.score(g["p_ens"].to_numpy(), g["y"].to_numpy())
            print(f"    n_bin={str(b):<20s} n_rows={len(g):>7,}  "
                  f"LightGBM={sc_lgb:7.1f}  CatBoost={sc_cb:7.1f}  50/50={sc_ens:7.1f}  "
                  f"(CB-LGB={sc_cb - sc_lgb:+.1f})")
