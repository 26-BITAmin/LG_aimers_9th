"""exp61 — 사후 ctx 로짓보정 축 확장: 초구(0-0 카운트) 스트라이크 성향.

exp57(outs/base_state/top_bottom/inning3 전부 기각)과 exp58(hand×count4 결합
기각) 이후 마지막으로 시도해보는 카운트 관련 축. count4는 0-0을 "평상시"
버킷 하나에 뭉개고 있어서, "이 투수가 초구를 얼마나 적극적으로 스트라이크
존에 넣는가"라는 별도 성향이 아직 한 번도 독립 축으로 분리된 적이 없다.

wonwoo가 count12(카운트 12칸 전부 분리)를 기각한 이유가 "셀당 표본이 count4의
1/3이라 과전이에 약함"이었다 — 초구(0-0)는 그보다는 표본이 크지만(전체 타석의
상당 비중이 0-0에서 시작), 그래도 count4가 이미 커버하는 정보와 겹칠 위험이
있다("채널 포화", wonwoo §7.43 / 우리 exp58). 기대치는 낮게 잡고 진단만 한다.

hand+count4(이미 LB로 확정된 기준선) 위에 초구 축 하나만 추가해서 한계 기여를
잰다. 챔피언과 동일하게 단일 seed(=0) 앙상블 사용.
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


def code_first_pitch(df):
    b = df["balls_before"].to_numpy(dtype=np.int64)
    s = df["strikes_before"].to_numpy(dtype=np.int64)
    return ((b == 0) & (s == 0)).astype(np.int64), 2


AXES = {"hand": code_hand, "count4": code_count4, "first_pitch": code_first_pitch}


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

        art = fit_ctx(fit_df, ["hand", "count4", "first_pitch"])
        z = (ctx_lookup(valid_df, "hand", art) + ctx_lookup(valid_df, "count4", art)
             + ctx_lookup(valid_df, "first_pitch", art))
        p_new = sigmoid(logit(p_ens) + z)
        sc_new = M.score(p_new, yv)
        hit = float(np.mean(ctx_lookup(valid_df, "first_pitch", art) != 0.0))
        first_pitch_share = float(np.mean(code_first_pitch(valid_df)[0]))
        print(f"    + first_pitch score={sc_new:.1f}  (Delta vs baseline={sc_new - sc_baseline:+.1f})  "
              f"hit_rate={hit:.3f}  0-0_share={first_pitch_share:.3f}")

        # 참고: first_pitch 단독(hand 없이 count4 위에만) 한계 기여도 같이 본다
        art_fp_only = fit_ctx(fit_df, ["count4", "first_pitch"])
        z_base_c4 = ctx_lookup(valid_df, "count4", art_fp_only)
        p_c4 = sigmoid(logit(p_ens) + z_base_c4)
        sc_c4 = M.score(p_c4, yv)
        z_c4_fp = z_base_c4 + ctx_lookup(valid_df, "first_pitch", art_fp_only)
        p_c4_fp = sigmoid(logit(p_ens) + z_c4_fp)
        sc_c4_fp = M.score(p_c4_fp, yv)
        print(f"    (참고) count4만={sc_c4:.1f}  count4+first_pitch={sc_c4_fp:.1f}  "
              f"(Delta={sc_c4_fp - sc_c4:+.1f})")
