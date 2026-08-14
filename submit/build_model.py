"""전체 train.csv(2019-2024)로 제출용 최종 모델을 만든다. 실험용이 아니라 배포 아티팩트
빌드 스크립트라 experiments/expNN_*.py와 분리해 여기 둔다.

구성은 exp32(현재 최선, fold D 최종 확인 exp36까지 통과) 그대로:
LightGBM(exp04 파라미터) + CatBoost(depth=8, l2_leaf_reg=5, lr=0.03) 50/50 고정 블렌드.

num_boost_round/iterations은 exp36(fold D: 2019-2023 학습 -> 2024 검증, 학습 구간이
전체 배포 구간 2019-2024와 가장 가까움)의 best_iteration을 그대로 씀 — early stopping을
쓸 수 있는 held-out 검증셋이 실제 배포(2025 test)엔 없으므로 고정 라운드 수로 재학습.

산출물 (model/):
- lgb_model.txt      LightGBM booster (텍스트)
- cb_model.cbm       CatBoost model
- pitcher_lookup.parquet   전체 train(R only)으로 만든 얼어붙은 투수별 lookup
  (asof_success_R2_static 등 CUSTOM_STATIC_FEATURES 적용에 필요, features.py의
  fit_pitcher_lookup 출력 그대로)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lightgbm as lgb
import catboost as cb
import pandas as pd

from src import features as feat
from src import folds as F

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
    objective="binary", metric="binary_logloss", seed=0, verbosity=-1,
)
LGB_NUM_ROUND = 212  # exp36 fold D best_iteration

CB_PARAMS = dict(
    learning_rate=0.03, depth=8, l2_leaf_reg=5.0,
    loss_function="Logloss", random_seed=0, verbose=False,
)
CB_ITERATIONS = 195  # exp36 fold D best_iteration


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading full train.csv...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    keep_mask = F._drop_bad_f(train)
    fit_raw = train.loc[keep_mask].copy()
    print(f"  full train rows={len(train):,}  after dropping bad-F={len(fit_raw):,}")

    print("Building season-frozen training features + final frozen pitcher lookup...")
    fit_df = feat.build_training_features(fit_raw)
    lookup = feat.fit_pitcher_lookup(fit_raw)
    print(f"  fit_df rows={len(fit_df):,}  lookup pitchers={len(lookup):,}")

    feature_cols = feat.feature_list(use_custom=True)
    Xt = fit_df[feature_cols].copy()
    yt = fit_df[feat.TARGET].to_numpy()

    print("Training LightGBM (final, full data)...")
    Xt_lgb = Xt.copy()
    for c in CAT_COLS:
        Xt_lgb[c] = Xt_lgb[c].astype("category")
    dtrain = lgb.Dataset(Xt_lgb, label=yt, categorical_feature=CAT_COLS, free_raw_data=False)
    booster = lgb.train(LGB_PARAMS, dtrain, num_boost_round=LGB_NUM_ROUND)
    booster.save_model(os.path.join(MODEL_DIR, "lgb_model.txt"))
    print(f"  saved lgb_model.txt (num_round={LGB_NUM_ROUND})")

    print("Training CatBoost (final, full data)...")
    Xt_cb = Xt.copy()
    for c in CAT_COLS:
        Xt_cb[c] = Xt_cb[c].astype(str)
    cb_model = cb.CatBoostClassifier(iterations=CB_ITERATIONS, **CB_PARAMS)
    cb_model.fit(Xt_cb, yt, cat_features=CAT_COLS)
    cb_model.save_model(os.path.join(MODEL_DIR, "cb_model.cbm"))
    print(f"  saved cb_model.cbm (iterations={CB_ITERATIONS})")

    # CSV로 저장 (평가 서버 사전 설치 패키지 목록에 pyarrow가 없어 parquet 위험 회피)
    lookup.reset_index().to_csv(os.path.join(MODEL_DIR, "pitcher_lookup.csv"), index=False)
    print("  saved pitcher_lookup.csv")

    print("Fitting post-hoc ctx logit correction (hand+count4, exp46/exp48)...")
    ctx_art = feat.fit_ctx(fit_df, ["hand", "count4"])
    for axis_name, a in ctx_art.items():
        pd.DataFrame({"key": a["keys"], "delta": a["delta"]}).to_csv(
            os.path.join(MODEL_DIR, f"ctx_{axis_name}.csv"), index=False)
        print(f"  saved ctx_{axis_name}.csv ({len(a['keys']):,} keys)")

    print("Done.")


if __name__ == "__main__":
    main()
