"""exp35 — exp34(nested-holdout 블렌드 가중치)이 exp32(고정 50/50) 대비 진짜 신호인지
투수 클러스터 부트스트랩으로 확인.

exp34 결과: fold A w=0.40(557.1, exp32 대비 −4.1), fold B w=0.55(542.7, exp32 대비 −7.0),
fold C w=0.00(722.5, exp32 대비 +16.9). 부호가 A/B는 손해, C는 큰 이득으로 갈려서
CLAUDE.md 판정 기준(① 부호 일관성)을 충족 못 함 — 특히 fold C가 w=0.00(순수 CatBoost)라는
코너 해로 떨어진 게 inner holdout(시즌 1개, 잡음 큼)의 과적합인지 실제 신호인지 셋 다
부트스트랩으로 확인해야 판단 가능.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

NESTED_W = dict(A=0.40, B=0.55, C=0.00)  # exp34 결과

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
        pid = valid_lgb["pitcher_id"].to_numpy()
        assert (pid == valid_cb["pitcher_id"].to_numpy()).all()

        w = NESTED_W[fold_name]
        p_fixed = 0.5 * p_lgb + 0.5 * p_cb
        p_nested = w * p_lgb + (1 - w) * p_cb

        print(f"  fixed-0.5(exp32): {M.score(p_fixed, y_lgb):.1f}   "
              f"nested-w={w:.2f}(exp34): {M.score(p_nested, y_lgb):.1f}")

        boot = M.pitcher_cluster_bootstrap(p_nested, p_fixed, y_lgb, pid, n_boot=500, seed=0)
        print(f"  exp34(nested w={w:.2f}) - exp32(fixed 0.5), fold {fold_name}, "
              f"pitcher-cluster bootstrap (n_boot=500):")
        print(f"    delta={boot['delta']:.2f}  95% CI=[{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]  "
              f"n_pitchers={boot['n_pitchers']}")
        if boot["ci_lo"] > 0:
            print("    -> CI 전부 양수: nested-w 블렌드가 유의미하게 낫다")
        elif boot["ci_hi"] < 0:
            print("    -> CI 전부 음수: nested-w 블렌드가 유의미하게 나쁘다")
        else:
            print("    -> CI가 0을 포함: 잡음과 구분 안 됨")
