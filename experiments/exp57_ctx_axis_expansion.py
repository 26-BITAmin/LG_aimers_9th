"""exp57 — 사후 ctx 로짓보정 축 확장. exp46/48이 hand+count4로 세 fold 전부 CI
확정(LB 실측 +16.5)된 것과 정확히 같은, 이미 검증된 메커니즘을 그대로 재사용해서
축만 늘린다. 시드 앙상블(exp54~56)처럼 결합 시 불확실해지는 리스크가 없다 —
로짓 덧셈이라 새 축은 기존 축과 독립적으로 작동한다.

후보 축 (전부 test 행 자기 자신의 값으로만 조회 -- 리크 없음):
  outs        0/1/2 아웃 상황 (m=3)
  base_state  주자 상황 8가지, runner_on_1b/2b/3b 조합 (m=8)
  top_bottom  공수 여부 (m=2)
  inning3     이닝을 초/중/후반 3구간으로 (m=3)

각 축을 hand+count4(이미 확정된 기준선) 위에 **하나씩 추가**해서 한계 기여를
잰다 -- 한 번에 하나만 바꾼다 원칙. 챔피언과 동일하게 단일 seed(=0) 앙상블 사용
(exp56에서 시드 앙상블은 결합 시 불확실해진다는 게 확인됐으므로 이 실험은 손대지 않음).
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


def code_outs(df):
    return df["outs_before"].to_numpy(dtype=np.int64), 3


def code_base_state(df):
    r1 = df["runner_on_1b"].to_numpy(dtype=np.int64)
    r2 = df["runner_on_2b"].to_numpy(dtype=np.int64)
    r3 = df["runner_on_3b"].to_numpy(dtype=np.int64)
    return r1 + 2 * r2 + 4 * r3, 8


def code_top_bottom(df):
    return (df["top_bottom"].astype(str).to_numpy() == "T").astype(np.int64), 2


def code_inning3(df):
    inning = df["inning"].to_numpy(dtype=np.int64)
    return np.clip((inning - 1) // 3, 0, 2), 3


AXES = {"hand": code_hand, "count4": code_count4, "outs": code_outs,
        "base_state": code_base_state, "top_bottom": code_top_bottom, "inning3": code_inning3}

CANDIDATE_NEW_AXES = ["outs", "base_state", "top_bottom", "inning3"]


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

        art_base = fit_ctx(fit_df, ["hand", "count4"])
        z_base = ctx_lookup(valid_df, "hand", art_base) + ctx_lookup(valid_df, "count4", art_base)
        p_baseline = sigmoid(logit(p_ens) + z_base)
        sc_baseline = M.score(p_baseline, yv)
        print(f"  baseline(hand+count4) = {sc_baseline:.1f}  (ensemble raw={M.score(p_ens, yv):.1f})")

        for new_axis in CANDIDATE_NEW_AXES:
            axes = ["hand", "count4", new_axis]
            art = fit_ctx(fit_df, axes)
            z = sum(ctx_lookup(valid_df, a, art) for a in axes)
            p_new = sigmoid(logit(p_ens) + z)
            sc_new = M.score(p_new, yv)
            hit = float(np.mean(ctx_lookup(valid_df, new_axis, art) != 0.0))
            print(f"    + {new_axis:<12s} score={sc_new:.1f}  (Delta vs baseline={sc_new - sc_baseline:+.1f})  "
                  f"hit_rate={hit:.3f}")
