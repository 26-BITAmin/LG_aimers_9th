"""submit/build_model.py의 변형 -- 현재 챔피언(LightGBM+CatBoost 50/50 + 사후 ctx보정)
위에 투수→팀→리그 3단 축소 피처(exp49/50, 판정 보류·미확정)를 얹은 후보를 빌드한다.

exp50 로컬 결과는 판정 보류(fold C가 작지만 음수)였지만, 사용자가 실제 LB로
직접 확인해보길 원해서 별도 후보로 빌드한다. **챔피언 submit/를 건드리지 않고**
submit_candidates/team_hier/ 에 완전히 격리해서 만든다.
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
MODEL_DIR = os.path.join(os.path.dirname(__file__), "team_hier", "model")

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
    objective="binary", metric="binary_logloss", seed=0, verbosity=-1,
)
LGB_NUM_ROUND = 212  # exp36 fold D best_iteration (챔피언과 동일 -- 재튜닝 안 함)

CB_PARAMS = dict(
    learning_rate=0.03, depth=8, l2_leaf_reg=5.0,
    loss_function="Logloss", random_seed=0, verbose=False,
)
CB_ITERATIONS = 195


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading full train.csv...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    keep_mask = F._drop_bad_f(train)
    fit_raw = train.loc[keep_mask].copy()
    print(f"  full train rows={len(train):,}  after dropping bad-F={len(fit_raw):,}")

    print("Building season-frozen training features (+team-hier) + final frozen lookups...")
    fit_df = feat.build_training_features(fit_raw, with_team_hier=True)
    lookup = feat.fit_pitcher_lookup(fit_raw)
    team_lookup, league_rate = feat.fit_team_lookup(fit_raw)
    print(f"  fit_df rows={len(fit_df):,}  lookup pitchers={len(lookup):,}  "
          f"teams={len(team_lookup):,}  league_rate={league_rate:.4f}")

    feature_cols = feat.feature_list(use_custom=True, use_team_hier=True)
    Xt = fit_df[feature_cols].copy()
    yt = fit_df[feat.TARGET].to_numpy()
    print(f"  feature_cols={len(feature_cols)}")

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

    lookup.reset_index().to_csv(os.path.join(MODEL_DIR, "pitcher_lookup.csv"), index=False)
    print("  saved pitcher_lookup.csv")

    team_out = team_lookup.reset_index()
    team_out["league_rate"] = league_rate
    team_out.to_csv(os.path.join(MODEL_DIR, "team_lookup.csv"), index=False)
    print(f"  saved team_lookup.csv ({len(team_out)} teams, league_rate={league_rate:.4f})")

    print("Fitting post-hoc ctx logit correction (hand+count4, exp46/exp48)...")
    ctx_art = feat.fit_ctx(fit_df, ["hand", "count4"])
    for axis_name, a in ctx_art.items():
        pd.DataFrame({"key": a["keys"], "delta": a["delta"]}).to_csv(
            os.path.join(MODEL_DIR, f"ctx_{axis_name}.csv"), index=False)
        print(f"  saved ctx_{axis_name}.csv ({len(a['keys']):,} keys)")

    print("Done.")


if __name__ == "__main__":
    main()
