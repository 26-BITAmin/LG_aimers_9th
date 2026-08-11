"""exp06 — 절편(intercept)만 사후 보정.

EDA_정리.md §8: 기울기/isotonic 보정은 held-out에서 전부 마이너스였고 절편 보정만
유효했다(+83.2). exp04/05 로그의 bias 컬럼을 보면 fold A=+0.00521, C=+0.01012로
0이 아니다 — 그 정도만큼 예측이 평균적으로 높게 잡혀 있다는 뜻.

★ 검증 시즌의 실제 라벨로 절편을 맞추면 test 참조 상수를 만드는 것과 같아 규칙 위반이자
치팅이다 (CLAUDE.md: "절편 상수는 학습 시점에 확정"). 그래서 학습 구간 자체를 한 시즌 더
쪼갠 "중첩 홀드아웃"으로 편향을 추정한다:

  fold C 예시 (train 2021-2023, valid 2024):
    proxy 모델을 2021-2022로만 학습 → 2023(학습 구간의 마지막 시즌, proxy는 못 봄)에 예측
    → bias_hat = mean(pred_2023) - mean(y_2023)
  이 bias_hat을 "실제" 모델(2021-2023 전체로 학습, 2024 예측)의 예측에서 빼준다.
  2024 라벨은 이 과정에서 한 번도 쓰이지 않는다.
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import folds as F
from src import metrics as M
from src import features as feat
from src.train_utils import load_train, log_result

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03,
    num_leaves=31,
    min_data_in_leaf=500,
    lambda_l2=5.0,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    objective="binary",
    metric="binary_logloss",
    verbosity=-1,
)

FEATURE_COLS = feat.feature_list(use_custom=True)


def _fit_predict(fit_df, apply_df, seed=0, num_boost_round=3000, early_stopping_rounds=100,
                  valid_frac=0.1):
    """fit_df로 학습(내부 뒤쪽 10%를 early-stopping 감시용으로 씀), apply_df에 예측."""
    Xt = fit_df[FEATURE_COLS].copy()
    yt = fit_df[feat.TARGET].to_numpy()
    Xa = apply_df[FEATURE_COLS].copy()
    for c in CAT_COLS:
        Xt[c] = Xt[c].astype("category")
        Xa[c] = Xa[c].astype("category")

    rng = np.random.default_rng(seed)
    n = len(Xt)
    perm = rng.permutation(n)
    n_es = max(1000, int(n * valid_frac))
    es_idx, tr_idx = perm[:n_es], perm[n_es:]

    dtrain = lgb.Dataset(Xt.iloc[tr_idx], label=yt[tr_idx], categorical_feature=CAT_COLS,
                          free_raw_data=False)
    des = lgb.Dataset(Xt.iloc[es_idx], label=yt[es_idx], categorical_feature=CAT_COLS,
                       free_raw_data=False, reference=dtrain)
    booster = lgb.train(LGB_PARAMS, dtrain, num_boost_round=num_boost_round,
                         valid_sets=[des], valid_names=["es"],
                         callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                                    lgb.log_evaluation(0)])
    return booster.predict(Xa, num_iteration=booster.best_iteration)


def estimate_bias_nested(df, fold_name, seed=0):
    """train_mask 안의 마지막 시즌을 홀드아웃으로 써서 절편 편향을 추정 (valid 라벨 미사용)."""
    train_mask, _ = F.split(df, fold_name)
    fit_raw = df.loc[train_mask].copy()
    seasons = sorted(fit_raw["season"].unique())
    last_season = seasons[-1]

    proxy_train_raw = fit_raw[fit_raw["season"] < last_season]
    proxy_holdout_raw = fit_raw[(fit_raw["season"] == last_season) & (fit_raw["game_type"] == "R")]

    proxy_fit = feat.build_training_features(proxy_train_raw)
    lookup = feat.fit_pitcher_lookup(proxy_train_raw)
    static_feats = feat.apply_static_features(proxy_holdout_raw, lookup)
    proxy_holdout = pd.concat([proxy_holdout_raw.reset_index(drop=True),
                                static_feats.reset_index(drop=True)], axis=1)

    p_holdout = _fit_predict(proxy_fit, proxy_holdout, seed=seed)
    y_holdout = proxy_holdout[feat.TARGET].to_numpy()
    bias_hat = float(p_holdout.mean() - y_holdout.mean())
    return bias_hat, last_season, len(proxy_holdout)


def run_fold_calibrated(df, fold_name, seed=0):
    bias_hat, last_season, n_holdout = estimate_bias_nested(df, fold_name, seed=seed)

    train_mask, valid_mask = F.split(df, fold_name)
    fit_df, valid_df = feat.build_fold_frames(df, train_mask, valid_mask, use_custom=True)

    Xt = fit_df[FEATURE_COLS].copy()
    yt = fit_df[feat.TARGET].to_numpy()
    Xv = valid_df[FEATURE_COLS].copy()
    yv = valid_df[feat.TARGET].to_numpy()
    for c in CAT_COLS:
        Xt[c] = Xt[c].astype("category")
        Xv[c] = Xv[c].astype("category")

    dtrain = lgb.Dataset(Xt, label=yt, categorical_feature=CAT_COLS, free_raw_data=False)
    dvalid = lgb.Dataset(Xv, label=yv, categorical_feature=CAT_COLS, free_raw_data=False, reference=dtrain)
    booster = lgb.train(LGB_PARAMS, dtrain, num_boost_round=3000,
                         valid_sets=[dvalid], valid_names=["valid"],
                         callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

    p_raw = booster.predict(Xv, num_iteration=booster.best_iteration)
    p_cal = np.clip(p_raw - bias_hat, 0.0, 1.0)

    res_raw = M.score_pair(p_raw, yv)
    res_cal = M.score_pair(p_cal, yv)
    print(f"  fold {fold_name}: bias_hat(nested, season {last_season}, n={n_holdout:,})={bias_hat:.5f}")
    print(f"    raw:        score={M.score(p_raw, yv):.1f}  bias={res_raw['bias']:.5f}")
    print(f"    calibrated: score={M.score(p_cal, yv):.1f}  bias={res_cal['bias']:.5f}")

    out = dict(fold=fold_name, score=M.score(p_cal, yv), n_va=len(yv),
               n_pitchers=int(valid_df["pitcher_id"].nunique()), **res_cal)
    return out


if __name__ == "__main__":
    df = load_train()
    print("=== exp06_intercept_calibration ===")
    for fn in ("A", "B", "C"):
        res = run_fold_calibrated(df, fn)
        log_result("exp06_intercept_calibration", fn, res,
                    notes="nested-holdout intercept correction on top of exp04 config, no valid-label peeking")
