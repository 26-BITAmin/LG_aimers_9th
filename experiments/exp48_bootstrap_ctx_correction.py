"""exp48 — exp46(사후 로짓 보정, hand+count4)이 진짜 신호인지 투수 단위 클러스터
부트스트랩으로 확인 (CLAUDE.md ③, exp46이 이미 통과한 ⓪seed게이트/①부호/②크기 다음 단계).

exp46 raw (exp32 앙상블 대비, beta=1.0 고정):
  A: 561.2 -> 582.4 (+21.2)   B: 549.7 -> 576.2 (+26.4)   C: 705.6 -> 726.9 (+21.3)
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


AXES = {"hand": code_hand, "count4": code_count4}


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
        pid_arr = valid["pitcher_id"].to_numpy()
        assert (valid["pitcher_id"].to_numpy() == valid_cb["pitcher_id"].to_numpy()).all()

        train_mask, valid_mask = split(df, fold_name)
        fit_df, valid_df = feat.build_fold_frames(df, train_mask, valid_mask, use_custom=True)
        assert len(valid_df) == len(yv)

        art = fit_ctx(fit_df, ["hand", "count4"])
        z = ctx_lookup(valid_df, "hand", art) + ctx_lookup(valid_df, "count4", art)
        p_corr = sigmoid(logit(p_ens) + 1.0 * z)

        sc_ens = M.score(p_ens, yv)
        sc_corr = M.score(p_corr, yv)
        print(f"  ensemble={sc_ens:.1f}  +hand+count4={sc_corr:.1f}  (raw Delta={sc_corr - sc_ens:+.1f})")

        boot = M.pitcher_cluster_bootstrap(p_corr, p_ens, yv, pid_arr, n_boot=500, seed=0)
        print(f"  bootstrap: delta={boot['delta']:.2f}  95% CI=[{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]  "
              f"n_pitchers={boot['n_pitchers']}")
        if boot["ci_lo"] > 0:
            print("  -> 유의미하게 개선")
        elif boot["ci_hi"] < 0:
            print("  -> 유의미하게 악화")
        else:
            print("  -> CI가 0을 포함: 잡음과 구분 안 됨")
