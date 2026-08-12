"""exp22 — exp21의 나머지 폴드(B) 확인. exp20 raw 점수는 fold B에서 617.7 vs exp04
461.3로 +156.4라는 이 세션 최대 격차였다 — 버그성 아티팩트가 아닌지도 같이 확인해야
한다. fold C(+65.39, CI 전부 양수)/fold A(−35.85, CI 전부 음수)에 이어 fold B도
부트스트랩으로 검증한다.
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

if __name__ == "__main__":
    df = load_train()
    fold_name = "B"

    print(f"training exp04 (LightGBM) config (fold {fold_name})...")
    res04, booster, p04, y04, valid_df04 = run_fold(
        df, fold_name, feat.feature_list(use_custom=True), CAT_COLS, LGB_PARAMS, use_custom=True,
        num_boost_round=3000, early_stopping_rounds=100)
    pid04 = valid_df04["pitcher_id"].to_numpy()
    print(f"  exp04 score={res04['score']:.1f}")

    print(f"training exp20 (CatBoost) config (fold {fold_name})...")
    res20, model, p20, y20, valid_df20 = run_fold_cb(
        df, fold_name, feat.feature_list(use_custom=True), CAT_COLS, CB_PARAMS, use_custom=True,
        num_boost_round=3000, early_stopping_rounds=100)
    pid20 = valid_df20["pitcher_id"].to_numpy()
    print(f"  exp20 score={res20['score']:.1f}")

    assert (y04 == y20).all() and (pid04 == pid20).all(), "valid set mismatch between runs"

    boot = M.pitcher_cluster_bootstrap(p20, p04, y04, pid04, n_boot=500, seed=0)
    print()
    print(f"=== exp20(CatBoost) - exp04(LightGBM), fold {fold_name}, "
          f"pitcher-cluster bootstrap (n_boot=500) ===")
    print(f"  delta={boot['delta']:.2f}  95% CI=[{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]  "
          f"n_pitchers={boot['n_pitchers']}")
    if boot["ci_lo"] > 0:
        print("  -> CI 전부 양수: CatBoost가 유의미하게 낫다")
    elif boot["ci_hi"] < 0:
        print("  -> CI 전부 음수: CatBoost가 유의미하게 나쁘다")
    else:
        print("  -> CI가 0을 포함: 잡음과 구분 안 됨")
