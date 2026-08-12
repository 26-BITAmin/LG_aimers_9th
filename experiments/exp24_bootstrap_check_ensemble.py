"""exp24 — exp23(LightGBM+CatBoost 50/50 앙상블)이 exp04(LightGBM 단독) 대비
진짜 신호인지 부트스트랩으로 확인한다. exp23은 fold A/B/C = 559.6/560.5/682.2로
exp04(559.6/461.3/631.5) 대비 A는 그대로, B/C는 크게 개선 — 이 세션 처음으로
완전히 부호 일치하는 결과다. fold C(주근거)에서 확인.
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
CB_PARAMS = dict(learning_rate=0.03, depth=6, l2_leaf_reg=5.0)


def check(fold_name):
    df = load_train()
    feature_cols = feat.feature_list(use_custom=True)

    print(f"training LightGBM (fold {fold_name})...")
    res_lgb, _, p_lgb, y_lgb, valid_lgb = run_fold(
        df, fold_name, feature_cols, CAT_COLS, LGB_PARAMS, use_custom=True,
        num_boost_round=3000, early_stopping_rounds=100)
    print(f"  LightGBM score={res_lgb['score']:.1f}")

    print(f"training CatBoost (fold {fold_name})...")
    res_cb, _, p_cb, y_cb, valid_cb = run_fold_cb(
        df, fold_name, feature_cols, CAT_COLS, CB_PARAMS, use_custom=True,
        num_boost_round=3000, early_stopping_rounds=100)
    print(f"  CatBoost score={res_cb['score']:.1f}")

    assert (y_lgb == y_cb).all()
    pid = valid_lgb["pitcher_id"].to_numpy()
    assert (pid == valid_cb["pitcher_id"].to_numpy()).all()

    p_ens = 0.5 * p_lgb + 0.5 * p_cb
    ens_score = M.score(p_ens, y_lgb)
    print(f"  ensemble score={ens_score:.1f}")

    boot = M.pitcher_cluster_bootstrap(p_ens, p_lgb, y_lgb, pid, n_boot=500, seed=0)
    print()
    print(f"=== ensemble - LightGBM(exp04), fold {fold_name}, pitcher-cluster bootstrap (n_boot=500) ===")
    print(f"  delta={boot['delta']:.2f}  95% CI=[{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]  "
          f"n_pitchers={boot['n_pitchers']}")
    if boot["ci_lo"] > 0:
        print("  -> CI 전부 양수: 앙상블이 유의미하게 낫다")
    elif boot["ci_hi"] < 0:
        print("  -> CI 전부 음수: 앙상블이 유의미하게 나쁘다")
    else:
        print("  -> CI가 0을 포함: 잡음과 구분 안 됨")
    print()


if __name__ == "__main__":
    check("C")
    check("B")
