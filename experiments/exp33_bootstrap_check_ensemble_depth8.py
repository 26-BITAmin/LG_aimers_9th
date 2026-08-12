"""exp33 — exp32(앙상블, CatBoost depth=8)이 exp23(앙상블, CatBoost depth=6) 대비
진짜 신호인지 부트스트랩으로 확인한다. exp32는 fold A/B/C = 561.2/549.7/705.6로
exp23(559.6/560.5/682.2) 대비 A 소폭 개선, B 소폭 악화, C 큰 폭 개선(+23.4) —
fold C(주근거)에서 확인.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import metrics as M
from src import features as feat
from src.train_utils import load_train, run_fold
from src.train_utils_cb import run_fold_cb

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
)
CB_PARAMS_D6 = dict(learning_rate=0.03, depth=6, l2_leaf_reg=5.0)
CB_PARAMS_D8 = dict(learning_rate=0.03, depth=8, l2_leaf_reg=5.0)


def ensemble_preds(fold_name, cb_params):
    df = load_train()
    feature_cols = feat.feature_list(use_custom=True)
    res_lgb, _, p_lgb, y_lgb, valid_lgb = run_fold(
        df, fold_name, feature_cols, CAT_COLS, LGB_PARAMS, use_custom=True,
        num_boost_round=3000, early_stopping_rounds=100)
    res_cb, _, p_cb, y_cb, valid_cb = run_fold_cb(
        df, fold_name, feature_cols, CAT_COLS, cb_params, use_custom=True,
        num_boost_round=3000, early_stopping_rounds=100)
    assert (y_lgb == y_cb).all()
    pid = valid_lgb["pitcher_id"].to_numpy()
    assert (pid == valid_cb["pitcher_id"].to_numpy()).all()
    p_ens = 0.5 * p_lgb + 0.5 * p_cb
    return p_ens, y_lgb, pid


if __name__ == "__main__":
    fold_name = "C"
    print(f"building exp23 ensemble (CatBoost depth=6, fold {fold_name})...")
    p23, y23, pid23 = ensemble_preds(fold_name, CB_PARAMS_D6)
    print(f"  exp23 ensemble score={M.score(p23, y23):.1f}")

    print(f"building exp32 ensemble (CatBoost depth=8, fold {fold_name})...")
    p32, y32, pid32 = ensemble_preds(fold_name, CB_PARAMS_D8)
    print(f"  exp32 ensemble score={M.score(p32, y32):.1f}")

    assert (y23 == y32).all() and (pid23 == pid32).all(), "valid set mismatch between runs"

    boot = M.pitcher_cluster_bootstrap(p32, p23, y23, pid23, n_boot=500, seed=0)
    print()
    print(f"=== exp32(depth=8 ensemble) - exp23(depth=6 ensemble), fold {fold_name}, "
          f"pitcher-cluster bootstrap (n_boot=500) ===")
    print(f"  delta={boot['delta']:.2f}  95% CI=[{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]  "
          f"n_pitchers={boot['n_pitchers']}")
    if boot["ci_lo"] > 0:
        print("  -> CI 전부 양수: depth=8 앙상블이 유의미하게 낫다")
    elif boot["ci_hi"] < 0:
        print("  -> CI 전부 음수: depth=8 앙상블이 유의미하게 나쁘다")
    else:
        print("  -> CI가 0을 포함: 잡음과 구분 안 됨")
