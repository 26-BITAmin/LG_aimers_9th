"""exp59 — exp56(시드 앙상블 통합 결과 불확실, 특히 fold C가 -10.2)을 분해.

LightGBM·CatBoost 각각 단독으로는 시드평균이 세 fold 모두 확실히 개선이었다
(exp54, exp55). 그런데 둘을 50/50 블렌드하면 fold C만 악화됐다. 어느 쪽 모델의
시드평균이 블렌드를 망가뜨리는지 확인 -- 4가지 조합(LGB seed0/avg x CB seed0/avg)을
전부 만들어서 seed0+seed0(현재 챔피언) 대비 부트스트랩으로 비교한다.

절반만(예: LGB만 평균, CB는 seed0 유지) 살아남으면 그게 안전하게 채택 가능한
"확실한 개선"이 된다.
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

SEEDS = (0, 1, 2, 3, 4)
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
        p_lgb_all, p_cb_all = [], []
        yv_ref, pid_ref = None, None
        for seed in SEEDS:
            res_l, _, p_l, y_l, valid_l = run_fold(df, fold_name, feature_cols, CAT_COLS, LGB_PARAMS,
                                                     use_custom=True, num_boost_round=3000,
                                                     early_stopping_rounds=100, seed=seed)
            res_c, _, p_c, y_c, valid_c = run_fold_cb(df, fold_name, feature_cols, CAT_COLS, CB_PARAMS,
                                                        use_custom=True, num_boost_round=3000,
                                                        early_stopping_rounds=100, seed=seed)
            assert (y_l == y_c).all()
            if yv_ref is None:
                yv_ref = y_l
                pid_ref = valid_l["pitcher_id"].to_numpy()
            p_lgb_all.append(p_l)
            p_cb_all.append(p_c)
        print(f"  trained {len(SEEDS)} seeds each")

        p_lgb_avg = np.mean(p_lgb_all, axis=0)
        p_cb_avg = np.mean(p_cb_all, axis=0)
        p_lgb_s0, p_cb_s0 = p_lgb_all[0], p_cb_all[0]

        train_mask, valid_mask = split(df, fold_name)
        fit_df, valid_df = feat.build_fold_frames(df, train_mask, valid_mask, use_custom=True)
        art = fit_ctx(fit_df, ["hand", "count4"])
        z = ctx_lookup(valid_df, "hand", art) + ctx_lookup(valid_df, "count4", art)

        combos = {
            "LGBs0+CBs0 (champion)": 0.5 * p_lgb_s0 + 0.5 * p_cb_s0,
            "LGBavg+CBs0": 0.5 * p_lgb_avg + 0.5 * p_cb_s0,
            "LGBs0+CBavg": 0.5 * p_lgb_s0 + 0.5 * p_cb_avg,
            "LGBavg+CBavg": 0.5 * p_lgb_avg + 0.5 * p_cb_avg,
        }
        p_champion = sigmoid(logit(combos["LGBs0+CBs0 (champion)"]) + z)
        sc_champion = M.score(p_champion, yv_ref)
        print(f"  champion score = {sc_champion:.1f}")

        for name, p_ens in combos.items():
            if name == "LGBs0+CBs0 (champion)":
                continue
            p_corr = sigmoid(logit(p_ens) + z)
            sc = M.score(p_corr, yv_ref)
            boot = M.pitcher_cluster_bootstrap(p_corr, p_champion, yv_ref, pid_ref, n_boot=500, seed=0)
            print(f"  {name:<16s} score={sc:.1f}  raw Delta={sc - sc_champion:+.1f}  "
                  f"bootstrap Delta={boot['delta']:.2f}  95% CI=[{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]")
