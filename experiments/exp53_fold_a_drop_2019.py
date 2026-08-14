"""exp53 — fold A에서 CatBoost가 지는 원인 재조사: 2019 시즌 자체가 원흉인가?

지금까지 기각된 가설: team_id CTR(exp43/44, 실제론 team_id가 fold C 핵심 성분),
표본크기(n) 구성(exp52, CB의 열세가 n구간 전부에서 균일 — n 문제 아님), depth
과적합(exp30, depth=4도 전 폴드 악화).

새 가설: fold A의 학습 구간(2019-2021)에만 **2019 시즌**이 들어간다 — A/B/C 중
2019는 fold A에만 있다(B=2020-2022, C=2021-2023). 2019는 트래킹 첫 해라 R base
rate가 가장 높고(.5495, EDA_정리.md §3) 데이터 특성이 가장 다를 수 있다.

검증: fold A와 같은 valid(2022, R전용)로 두되 train을 2019-2021(원래) vs
2020-2021(2019 제외)로 바꿔 LightGBM/CatBoost 둘 다 재학습 -- CatBoost만 크게
좋아지면 "CatBoost가 2019 데이터에 특히 취약"이라는 뜻, LightGBM도 비슷하게
좋아지면 "2019 자체가 노이즈, 모델 무관"이라는 뜻.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lightgbm as lgb
import catboost as cb

from src import features as feat
from src import folds as F
from src import metrics as M
from src.train_utils import load_train

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
    objective="binary", metric="binary_logloss", seed=0, verbosity=-1,
)
CB_PARAMS = dict(
    learning_rate=0.03, depth=8, l2_leaf_reg=5.0,
    loss_function="Logloss", random_seed=0, verbose=False,
)


def train_and_score(df, train_mask, valid_mask, label):
    fit_df, valid_df = feat.build_fold_frames(df, train_mask, valid_mask, use_custom=True)
    feature_cols = feat.feature_list(use_custom=True)
    yt = fit_df[feat.TARGET].to_numpy()
    yv = valid_df[feat.TARGET].to_numpy()
    print(f"  [{label}] train_rows={len(fit_df):,}  valid_rows={len(valid_df):,}")

    Xt_lgb = fit_df[feature_cols].copy()
    Xv_lgb = valid_df[feature_cols].copy()
    for c in CAT_COLS:
        Xt_lgb[c] = Xt_lgb[c].astype("category")
        Xv_lgb[c] = Xv_lgb[c].astype("category")
    dtrain = lgb.Dataset(Xt_lgb, label=yt, categorical_feature=CAT_COLS, free_raw_data=False)
    dvalid = lgb.Dataset(Xv_lgb, label=yv, categorical_feature=CAT_COLS, free_raw_data=False, reference=dtrain)
    booster = lgb.train(LGB_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dvalid], valid_names=["valid"],
                         callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    p_lgb = booster.predict(Xv_lgb, num_iteration=booster.best_iteration)
    sc_lgb = M.score(p_lgb, yv)

    Xt_cb = fit_df[feature_cols].copy()
    Xv_cb = valid_df[feature_cols].copy()
    for c in CAT_COLS:
        Xt_cb[c] = Xt_cb[c].astype(str)
        Xv_cb[c] = Xv_cb[c].astype(str)
    cb_model = cb.CatBoostClassifier(iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    cb_model.fit(Xt_cb, yt, cat_features=CAT_COLS, eval_set=(Xv_cb, yv), use_best_model=True)
    p_cb = cb_model.predict_proba(Xv_cb)[:, 1]
    sc_cb = M.score(p_cb, yv)

    print(f"  [{label}] LightGBM={sc_lgb:.1f}  CatBoost={sc_cb:.1f}  (CB-LGB={sc_cb - sc_lgb:+.1f})  "
          f"best_iter_lgb={booster.best_iteration}  best_iter_cb={cb_model.get_best_iteration()}")
    return sc_lgb, sc_cb


if __name__ == "__main__":
    df = load_train()

    print("=== baseline: fold A (train 2019-2021, valid 2022) ===")
    train_mask_a, valid_mask_a = F.split(df, "A")
    lgb_a, cb_a = train_and_score(df, train_mask_a, valid_mask_a, "2019-2021 (fold A original)")

    print("\n=== variant: train 2020-2021 (drop 2019), valid 2022 ===")
    train_mask_v = df["season"].between(2020, 2021) & F._drop_bad_f(df)
    valid_mask_v = (df["season"] == 2022) & (df["game_type"] == "R")
    lgb_v, cb_v = train_and_score(df, train_mask_v, valid_mask_v, "2020-2021 (2019 dropped)")

    print("\n=== summary ===")
    print(f"  LightGBM: {lgb_a:.1f} -> {lgb_v:.1f}  (Delta={lgb_v - lgb_a:+.1f})")
    print(f"  CatBoost: {cb_a:.1f} -> {cb_v:.1f}  (Delta={cb_v - cb_a:+.1f})")
    print(f"  CB-LGB gap: {cb_a - lgb_a:+.1f} -> {cb_v - lgb_v:+.1f}")
