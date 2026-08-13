"""exp34 — 옵션2: LightGBM/CatBoost 50/50 고정 블렌드(exp32) 대신, 블렌드 가중치 w를
검증 라벨(fold A/B/C의 valid)로 직접 fit하지 않고 **train 내부 nested holdout**으로 고른다.

방식 (fold별로 독립):
  fold의 train 구간(예: fold C면 2021-2023) 중 가장 최근 시즌(2023)을 inner holdout으로
  떼어내고, 그 이전 시즌들(2021-2022)만으로 LightGBM/CatBoost를 학습한다. inner holdout에서
  w in {0.0, 0.05, ..., 1.0} 그리드로 raw Brier를 최소화하는 w*를 고른다 (실제 valid 라벨은
  전혀 안 봄). 그 다음 exp32와 동일하게 전체 train 구간으로 최종 LightGBM/CatBoost를 다시
  학습하고, 실제 fold valid에 w*를 적용해 블렌드 점수를 낸다.

"test 기반 가중치" 폐기 사유(CLAUDE.md)를 피하려는 것이 핵심 — inner holdout은 train
구간에서만 떼어내므로 fold의 실제 valid 라벨은 가중치 선택에 전혀 관여하지 않는다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src import features as feat
from src import folds as F
from src import metrics as M
from src.train_utils import load_train, run_fold, log_result
from src.train_utils_cb import run_fold_cb

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
)
CB_PARAMS = dict(learning_rate=0.03, depth=8, l2_leaf_reg=5.0)  # exp32 best

LABEL = "exp34_blend_weight_nested_holdout"
NOTES = "blend weight w chosen per-fold via nested holdout inside train (last train season held out), never touches actual valid labels; compare vs exp32's fixed 0.5"

W_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def _inner_masks(df, fold_name):
    lo, hi = F.FOLDS[fold_name]["train"]
    train_mask, _ = F.split(df, fold_name)
    inner_train_mask = train_mask & df["season"].between(lo, hi - 1)
    inner_holdout_mask = train_mask & (df["season"] == hi) & (df["game_type"] == "R")
    return inner_train_mask, inner_holdout_mask


def _fit_predict_lgb(df, train_mask, valid_mask, feature_cols):
    fit_df, valid_df = feat.build_fold_frames(df, train_mask, valid_mask, use_custom=True)
    import lightgbm as lgb
    Xt = fit_df[feature_cols].copy()
    Xv = valid_df[feature_cols].copy()
    yt = fit_df[feat.TARGET].to_numpy()
    yv = valid_df[feat.TARGET].to_numpy()
    for c in CAT_COLS:
        Xt[c] = Xt[c].astype("category")
        Xv[c] = Xv[c].astype("category")
    dtrain = lgb.Dataset(Xt, label=yt, categorical_feature=CAT_COLS, free_raw_data=False)
    dvalid = lgb.Dataset(Xv, label=yv, categorical_feature=CAT_COLS, free_raw_data=False, reference=dtrain)
    params = dict(LGB_PARAMS)
    params.setdefault("objective", "binary")
    params.setdefault("metric", "binary_logloss")
    params.setdefault("seed", 0)
    params.setdefault("verbosity", -1)
    booster = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dvalid], valid_names=["valid"],
                         callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    p = booster.predict(Xv, num_iteration=booster.best_iteration)
    return p, yv, valid_df


def _fit_predict_cb(df, train_mask, valid_mask, feature_cols):
    import catboost as cb
    fit_df, valid_df = feat.build_fold_frames(df, train_mask, valid_mask, use_custom=True)
    Xt = fit_df[feature_cols].copy()
    Xv = valid_df[feature_cols].copy()
    yt = fit_df[feat.TARGET].to_numpy()
    yv = valid_df[feat.TARGET].to_numpy()
    for c in CAT_COLS:
        Xt[c] = Xt[c].astype(str)
        Xv[c] = Xv[c].astype(str)
    params = dict(CB_PARAMS)
    params.setdefault("loss_function", "Logloss")
    params.setdefault("eval_metric", "Logloss")
    params.setdefault("random_seed", 0)
    params.setdefault("verbose", False)
    params.setdefault("iterations", 3000)
    params.setdefault("early_stopping_rounds", 100)
    model = cb.CatBoostClassifier(**params)
    model.fit(Xt, yt, cat_features=CAT_COLS, eval_set=(Xv, yv), use_best_model=True)
    p = model.predict_proba(Xv)[:, 1]
    return p, yv, valid_df


def pick_weight(df, fold_name, feature_cols):
    inner_train_mask, inner_holdout_mask = _inner_masks(df, fold_name)
    p_lgb, y_lgb, _ = _fit_predict_lgb(df, inner_train_mask, inner_holdout_mask, feature_cols)
    p_cb, y_cb, _ = _fit_predict_cb(df, inner_train_mask, inner_holdout_mask, feature_cols)
    assert (y_lgb == y_cb).all()
    best_w, best_bs = 0.5, np.inf
    for w in W_GRID:
        p = w * p_lgb + (1 - w) * p_cb
        bs = M.brier(p, y_lgb)
        if bs < best_bs:
            best_bs, best_w = bs, w
    return float(best_w)


if __name__ == "__main__":
    df = load_train()
    feature_cols = feat.feature_list(use_custom=True)

    for fold_name in ("A", "B", "C"):
        print(f"=== fold {fold_name} ===")
        w = pick_weight(df, fold_name, feature_cols)
        print(f"  nested-holdout picked w(LightGBM)={w:.2f} (CatBoost weight={1 - w:.2f})")

        res_lgb, _, p_lgb, y_lgb, valid_lgb = run_fold(
            df, fold_name, feature_cols, CAT_COLS, LGB_PARAMS, use_custom=True,
            num_boost_round=3000, early_stopping_rounds=100)
        res_cb, _, p_cb, y_cb, valid_cb = run_fold_cb(
            df, fold_name, feature_cols, CAT_COLS, CB_PARAMS, use_custom=True,
            num_boost_round=3000, early_stopping_rounds=100)
        assert (y_lgb == y_cb).all()
        assert (valid_lgb["pitcher_id"].to_numpy() == valid_cb["pitcher_id"].to_numpy()).all()

        p_ens = w * p_lgb + (1 - w) * p_cb
        sp = M.score_pair(p_ens, y_lgb)
        sc = M.score(p_ens, y_lgb)
        res = dict(fold=fold_name, score=sc, n_va=len(y_lgb),
                   n_pitchers=int(valid_lgb["pitcher_id"].nunique()), best_iter=-1, **sp)
        log_result(LABEL, fold_name, res, notes=f"{NOTES} | w_lgb={w:.2f}")
        print(f"  LightGBM alone: {res_lgb['score']:.1f}   CatBoost alone: {res_cb['score']:.1f}"
              f"   ensemble(w={w:.2f}): {res['score']:.1f}  pts_res={res['pts_res']:.1f}  bias={res['bias']:.5f}"
              f"   (fixed-0.5 exp32 comparison in results.csv)")
