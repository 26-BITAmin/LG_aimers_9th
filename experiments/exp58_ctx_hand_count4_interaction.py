"""exp58 — exp57에서 상황 축(outs/base_state/top_bottom/inning3) 전부 기각된 뒤
ctx보정의 마지막 확장 시도: hand와 count4의 **상호작용**.

지금은 δ_hand + δ_count4로 독립 가산한다(로짓 공간 덧셈, 두 축이 서로 무관하다고
가정). 만약 "왼손 상대로 3볼일 때"처럼 두 축이 결합된 상황에서 투수마다 다른
패턴이 있다면, 독립 가산은 그 상호작용을 못 담는다 -- 단일 결합 축(m=8: hand 2 x
count4 4)으로 다시 fit해서 비교.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src import features as feat
from src import metrics as M
from src.folds import split
from src.train_utils import load_train, run_fold
from src.train_utils_cb import run_fold_cb

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
)
CB_PARAMS = dict(learning_rate=0.03, depth=8, l2_leaf_reg=5.0)

K_SHRINK = 200.0


def code_hand(df):
    return df["batter_hand"].to_numpy(dtype=np.int64) - 1, 2


def code_count4(df):
    b = df["balls_before"].to_numpy(dtype=np.int64)
    s = df["strikes_before"].to_numpy(dtype=np.int64)
    return (b >= 3).astype(np.int64) + 2 * (s >= 2).astype(np.int64), 4


def code_hand_count4_joint(df):
    hcode, _ = code_hand(df)
    ccode, _ = code_count4(df)
    return hcode * 4 + ccode, 8


AXES = {"hand": code_hand, "count4": code_count4, "hand_count4_joint": code_hand_count4_joint}


def fit_ctx(fit_df, axis_names, k=K_SHRINK):
    y = fit_df[feat.TARGET].to_numpy(dtype=np.float64)
    pid = fit_df["pitcher_id"].to_numpy(dtype=np.int64)
    upid, invp = np.unique(pid, return_inverse=True)
    n_p = np.bincount(invp).astype(np.float64)
    rate_p = np.bincount(invp, weights=y) / n_p
    out = {}
    for name in axis_names:
        code, m = AXES[name](fit_df)
        key = pid * m + code
        u, inv = np.unique(key, return_inverse=True)
        n = np.bincount(inv).astype(np.float64)
        t = np.bincount(inv, weights=y)
        rp = rate_p[np.searchsorted(upid, u // m)]
        delta = (n / (n + k)) * (t / n - rp)
        out[name] = {"m": m, "keys": u, "delta": delta}
    return out


def ctx_lookup(df, axis_name, art):
    a = art[axis_name]
    code, m = AXES[axis_name](df)
    kv = df["pitcher_id"].to_numpy(dtype=np.int64) * m + code
    keys = a["keys"]
    j = np.clip(np.searchsorted(keys, kv), 0, len(keys) - 1)
    return np.where(keys[j] == kv, a["delta"][j], 0.0)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


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
        p_ens = 0.5 * p_lgb + 0.5 * p_cb
        yv = y_lgb

        train_mask, valid_mask = split(df, fold_name)
        fit_df, valid_df = feat.build_fold_frames(df, train_mask, valid_mask, use_custom=True)

        art_add = fit_ctx(fit_df, ["hand", "count4"])
        z_add = ctx_lookup(valid_df, "hand", art_add) + ctx_lookup(valid_df, "count4", art_add)
        p_add = sigmoid(logit(p_ens) + z_add)
        sc_add = M.score(p_add, yv)

        art_joint = fit_ctx(fit_df, ["hand_count4_joint"])
        z_joint = ctx_lookup(valid_df, "hand_count4_joint", art_joint)
        p_joint = sigmoid(logit(p_ens) + z_joint)
        sc_joint = M.score(p_joint, yv)
        hit = float(np.mean(z_joint != 0.0))

        print(f"  additive(hand+count4)      = {sc_add:.1f}")
        print(f"  joint(hand x count4, m=8)  = {sc_joint:.1f}  (Delta vs additive={sc_joint - sc_add:+.1f})  hit_rate={hit:.3f}")
