"""exp39 — exp38(최근 시즌 지수 가중치, half_life=1.0)이 exp04 대비 진짜 신호인지
잡음인지 투수 단위 클러스터 부트스트랩으로 확인 (CLAUDE.md ③).

exp38 raw 결과: A 543.7 (exp04 559.6, Δ-15.9) / B 469.0 (exp04 461.3, Δ+7.7) /
C 633.6 (exp04 631.5, Δ+2.1) — 부호가 이미 엇갈린다(A 악화, B/C 개선). fold A(가장
크게 악화)와 fold C(주근거 폴드)에서 짝지은 Δ와 95% CI를 계산해 크기가 잡음 폭을
넘는지 확인한다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lightgbm as lgb

from src import features as feat
from src import folds as F
from src import metrics as M
from src.train_utils import load_train, run_fold

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS_BASE = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
    objective="binary", metric="binary_logloss", seed=0, verbosity=-1,
)
HALF_LIFE = 1.0


def get_preds_weighted(feature_cols, fold_name, df):
    train_mask, valid_mask = F.split(df, fold_name)
    fit_df, valid_df = feat.build_fold_frames(df, train_mask, valid_mask, use_custom=True)
    Xt = fit_df[feature_cols].copy()
    Xv = valid_df[feature_cols].copy()
    yt = fit_df[feat.TARGET].to_numpy()
    yv = valid_df[feat.TARGET].to_numpy()
    for c in CAT_COLS:
        Xt[c] = Xt[c].astype("category")
        Xv[c] = Xv[c].astype("category")
    train_max_season = fit_df["season"].max()
    w = 0.5 ** ((train_max_season - fit_df["season"]).to_numpy() / HALF_LIFE)
    dtrain = lgb.Dataset(Xt, label=yt, weight=w, categorical_feature=CAT_COLS, free_raw_data=False)
    dvalid = lgb.Dataset(Xv, label=yv, categorical_feature=CAT_COLS, free_raw_data=False, reference=dtrain)
    booster = lgb.train(
        LGB_PARAMS_BASE, dtrain, num_boost_round=3000,
        valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    p = booster.predict(Xv, num_iteration=booster.best_iteration)
    return p, yv, valid_df["pitcher_id"].to_numpy()


def get_preds_base(feature_cols, fold_name, df):
    res, booster, p, yv, valid_df = run_fold(
        df, fold_name, feature_cols, CAT_COLS, LGB_PARAMS_BASE, use_custom=True,
        num_boost_round=3000, early_stopping_rounds=100)
    return p, yv, valid_df["pitcher_id"].to_numpy()


if __name__ == "__main__":
    df = load_train()
    feature_cols = feat.feature_list(use_custom=True)

    for fold_name in ("A", "C"):
        print(f"=== fold {fold_name} ===")
        p_base, y_base, pid_base = get_preds_base(feature_cols, fold_name, df)
        p_w, y_w, pid_w = get_preds_weighted(feature_cols, fold_name, df)
        assert (y_base == y_w).all() and (pid_base == pid_w).all(), "valid set mismatch"

        boot = M.pitcher_cluster_bootstrap(p_w, p_base, y_base, pid_base, n_boot=500, seed=0)
        print(f"  exp38(weighted) - exp04(base): delta={boot['delta']:.2f}  "
              f"95% CI=[{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]  n_pitchers={boot['n_pitchers']}")
        if boot["ci_lo"] > 0:
            print("  -> 유의미하게 좋아짐")
        elif boot["ci_hi"] < 0:
            print("  -> 유의미하게 나빠짐")
        else:
            print("  -> CI가 0을 포함: 잡음과 구분 안 됨")
