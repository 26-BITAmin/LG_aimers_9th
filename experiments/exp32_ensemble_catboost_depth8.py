"""exp32 — exp23 앙상블의 CatBoost 성분을 depth=8(exp31)로 교체했을 때 순효과.

exp30(depth=4)은 세 폴드 다 악화, exp31(depth=8)은 fold C +25.6(주근거 폴드,
큰 개선), B −20.7(하지만 여전히 LightGBM보다 압도적으로 높음: 597.0 vs 461.3),
A +1.9(거의 그대로)였다. 앙상블은 어차피 fold A/B의 CatBoost 약점을 LightGBM으로
보완하는 구조라, depth=8의 fold C 이득이 앙상블 전체에도 살아남는지, fold B 손실이
앙상블 레벨에서도 문제가 되는지 실측 확인.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import features as feat
from src import metrics as M
from src.train_utils import load_train, run_fold, log_result
from src.train_utils_cb import run_fold_cb

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
)
CB_PARAMS = dict(learning_rate=0.03, depth=8, l2_leaf_reg=5.0)

LABEL = "exp32_ensemble_catboost_depth8"
NOTES = "0.5*LightGBM(exp04) + 0.5*CatBoost(depth=8, exp31), vs exp23's depth=6 CatBoost"

if __name__ == "__main__":
    df = load_train()
    feature_cols = feat.feature_list(use_custom=True)

    for fold_name in ("A", "B", "C"):
        print(f"=== fold {fold_name} ===")
        res_lgb, _, p_lgb, y_lgb, valid_lgb = run_fold(
            df, fold_name, feature_cols, CAT_COLS, LGB_PARAMS, use_custom=True,
            num_boost_round=3000, early_stopping_rounds=100)
        res_cb, _, p_cb, y_cb, valid_cb = run_fold_cb(
            df, fold_name, feature_cols, CAT_COLS, CB_PARAMS, use_custom=True,
            num_boost_round=3000, early_stopping_rounds=100)

        assert (y_lgb == y_cb).all()
        assert (valid_lgb["pitcher_id"].to_numpy() == valid_cb["pitcher_id"].to_numpy()).all()

        p_ens = 0.5 * p_lgb + 0.5 * p_cb
        sp = M.score_pair(p_ens, y_lgb)
        sc = M.score(p_ens, y_lgb)
        res = dict(fold=fold_name, score=sc, n_va=len(y_lgb),
                   n_pitchers=int(valid_lgb["pitcher_id"].nunique()), best_iter=-1, **sp)
        log_result(LABEL, fold_name, res, notes=NOTES)
        print(f"  LightGBM alone: {res_lgb['score']:.1f}   CatBoost(depth=8) alone: {res_cb['score']:.1f}"
              f"   ensemble: {res['score']:.1f}  pts_res={res['pts_res']:.1f}  bias={res['bias']:.5f}")
