"""exp56 — exp54(LightGBM 시드 앙상블) + exp55(CatBoost 시드 앙상블)를 합친 전체
구성이 현재 챔피언(단일시드 LGB+CB 50/50 + 사후 ctx보정, LB 834.44)을 실제로
이기는지 투수 단위 클러스터 부트스트랩으로 확정 검증.

새 구성 = (LightGBM 5seed 평균) 50% + (CatBoost 5seed 평균) 50% + 사후 ctx보정(hand+count4)
기존 챔피언 = (LightGBM seed0) 50% + (CatBoost seed0) 50% + 사후 ctx보정(hand+count4)

두 구성 다 같은 ctx 보정(fit_ctx는 fit_df 위에서 한 번만 fit -- 앙상블 구성과
무관하게 동일 아티팩트 재사용, 두 비교 대상이 보정 방식 차이가 아니라 순수하게
"시드 앙상블 유무" 차이만 갖게 하기 위함).
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
            print(f"  seed={seed} trained (LGB={res_l['score']:.1f}  CB={res_c['score']:.1f})")

        p_lgb_avg = np.mean(p_lgb_all, axis=0)
        p_cb_avg = np.mean(p_cb_all, axis=0)
        p_ens_new = 0.5 * p_lgb_avg + 0.5 * p_cb_avg
        p_ens_old = 0.5 * p_lgb_all[0] + 0.5 * p_cb_all[0]  # seed=0 only, matches current champion

        train_mask, valid_mask = split(df, fold_name)
        fit_df, valid_df = feat.build_fold_frames(df, train_mask, valid_mask, use_custom=True)
        art = fit_ctx(fit_df, ["hand", "count4"])
        z = ctx_lookup(valid_df, "hand", art) + ctx_lookup(valid_df, "count4", art)

        p_new = sigmoid(logit(p_ens_new) + z)
        p_old = sigmoid(logit(p_ens_old) + z)

        sc_new = M.score(p_new, yv_ref)
        sc_old = M.score(p_old, yv_ref)
        print(f"  new(5seed LGB+CB avg + ctx) = {sc_new:.1f}   old(seed0 LGB+CB + ctx) = {sc_old:.1f}   "
              f"raw Delta = {sc_new - sc_old:+.1f}")

        boot = M.pitcher_cluster_bootstrap(p_new, p_old, yv_ref, pid_ref, n_boot=500, seed=0)
        print(f"  bootstrap: delta={boot['delta']:.2f}  95% CI=[{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]  "
              f"n_pitchers={boot['n_pitchers']}")
        if boot["ci_lo"] > 0:
            print("  -> 유의미하게 개선")
        elif boot["ci_hi"] < 0:
            print("  -> 유의미하게 악화")
        else:
            print("  -> CI가 0을 포함: 잡음과 구분 안 됨")
