"""exp46 — wonwoo 브랜치(95f7ab3, LB 987.41)의 새 아이디어: 사후 로짓 보정 (post-hoc
logit correction). 코드는 이식하지 않고 메커니즘만 재구현해 우리 파이프라인에서 검증.

메커니즘 (wonwoo `src/ctxfit.py`):
    δ(투수, 상황코드) = n/(n+k) · (그 셀 성공률 − 그 투수 전체 성공률)
    z = Σ_axis β · δ
    p_corrected = sigmoid(logit(p_raw) + z)

train에서만 fit(축소 강도 k, 투수별 상황-편차), valid에서는 조회만 — 모델 재학습
없이 앙상블 예측 위에 사후로 더하는 보정항이라 학습셋 타깃 리크가 구조적으로
안 생긴다(CLAUDE.md가 명시적으로 허용하는 "train fit, test transform" 패턴).

이게 CLAUDE.md 폐기 목록의 "카운트 조건부 투수 이력(−30.9)"과 같은 것인지 확인함:
그 항목은 실제로는 exp27 "실패성분 좌우분할"(모델 **입력 피처**, 좌우 실패성분 분할)을
가리키는 것으로 model.md에서 추적됨 — 이 사후 로짓 보정(count4 축)은 완전히 다른
메커니즘이라 우리 프로젝트에서 시도된 적이 없음.

축 두 개:
  hand   — 타자 손잡이(사실상 2값, wonwoo는 키 호환 위해 m=8 사용, 우리는 m=2로 충분)
  count4 — 0=평상시 1=3볼 2=2스트라이크 3=풀카운트급 (balls>=3, strikes>=2 조합)

우리는 이미 `asof_success_R_platoon_blend`(투수×타자손 강도, strictly-prior 연속
누적)를 입력 피처로 갖고 있어 hand 축은 일부 중복 가능성이 있음 — 그래도 사후
보정은 트리 앙상블의 정규화가 지운 잔차를 로짓 공간에서 직접 잡는 구조라 다를 수
있어 diagnostic으로 같이 확인. count4 축은 우리 프로젝트에서 완전히 새로운 시도.

β=1.0 고정으로 hand만 / hand+count4 두 단계로 순차 확인 (wonwoo의 s11→s12 순서와
동일) — exp32 앙상블(LightGBM+CatBoost 50/50) 위에 얹는다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

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

K_SHRINK = 200.0  # wonwoo와 동일 축소 강도(k_pitcher와 같은 자릿수) -- 별도 튜닝 안 함


def code_hand(df):
    # batter_hand 실제 값은 1=좌, 2=우 (CLAUDE.md) -- 0-index로 매핑해야 pid*m+code 키가
    # 인접 pitcher_id와 충돌하지 않는다 (원래 코드가 1/2를 그대로 써서 IndexError 발생).
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
        sc_ens = M.score(p_ens, yv)
        print(f"  ensemble(exp32): score={sc_ens:.1f}")

        from src.folds import split
        train_mask, valid_mask = split(df, fold_name)
        fit_df, valid_df = feat.build_fold_frames(df, train_mask, valid_mask, use_custom=True)
        assert len(valid_df) == len(yv)

        art_hand = fit_ctx(fit_df, ["hand"])
        z_hand = ctx_lookup(valid_df, "hand", art_hand)
        p_hand = sigmoid(logit(p_ens) + 1.0 * z_hand)
        sc_hand = M.score(p_hand, yv)
        print(f"  + hand axis (beta=1.0): score={sc_hand:.1f}  (Delta={sc_hand - sc_ens:+.1f})  "
              f"hit_rate={float(np.mean(z_hand != 0.0)):.3f}")

        art_both = fit_ctx(fit_df, ["hand", "count4"])
        z_both = ctx_lookup(valid_df, "hand", art_both) + ctx_lookup(valid_df, "count4", art_both)
        p_both = sigmoid(logit(p_ens) + 1.0 * z_both)
        sc_both = M.score(p_both, yv)
        print(f"  + hand+count4 (beta=1.0 each): score={sc_both:.1f}  (Delta={sc_both - sc_ens:+.1f})  "
              f"count4_hit_rate={float(np.mean(ctx_lookup(valid_df, 'count4', art_both) != 0.0)):.3f}")

        art_count_only = fit_ctx(fit_df, ["count4"])
        z_count = ctx_lookup(valid_df, "count4", art_count_only)
        p_count = sigmoid(logit(p_ens) + 1.0 * z_count)
        sc_count = M.score(p_count, yv)
        print(f"  + count4 axis only (beta=1.0): score={sc_count:.1f}  (Delta={sc_count - sc_ens:+.1f})")
