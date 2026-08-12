"""exp26 — exp25(CatBoost + n 노출)이 exp20(CatBoost 베이스라인) 대비 진짜 신호인지
부트스트랩으로 확인한다. exp25는 fold A/B/C = 527.9/625.1/701.8로 exp20(523.8/617.7/
696.9) 대비 세 폴드 다 소폭 개선(+4.1/+7.4/+4.9) — 처음으로 완전히 부호 일치하는
"새 피처 추가" 결과지만 크기가 작아서(잡음 바닥 ±20~30점 수준) CI 확인이 특히 중요.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import metrics as M
from src import features as feat
from src.train_utils_cb import load_train, run_fold_cb

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

CB_PARAMS = dict(learning_rate=0.03, depth=6, l2_leaf_reg=5.0)


def get_preds(feature_cols, use_n_exposed, fold_name):
    df = load_train()
    res, model, p, yv, valid_df = run_fold_cb(
        df, fold_name, feature_cols, CAT_COLS, CB_PARAMS, use_custom=True,
        num_boost_round=3000, early_stopping_rounds=100, use_n_exposed=use_n_exposed)
    return p, yv, valid_df["pitcher_id"].to_numpy(), res


if __name__ == "__main__":
    print("training exp20 (CatBoost baseline) config (fold C)...")
    p20, y20, pid20, res20 = get_preds(feat.feature_list(use_custom=True), False, "C")
    print(f"  exp20 score={res20['score']:.1f}")

    print("training exp25 (CatBoost + n_exposed) config (fold C)...")
    p25, y25, pid25, res25 = get_preds(
        feat.feature_list(use_custom=True, use_n_exposed=True), True, "C")
    print(f"  exp25 score={res25['score']:.1f}")

    assert (y20 == y25).all() and (pid20 == pid25).all(), "valid set mismatch between runs"

    boot = M.pitcher_cluster_bootstrap(p25, p20, y20, pid20, n_boot=500, seed=0)
    print()
    print("=== exp25 - exp20, fold C, pitcher-cluster bootstrap (n_boot=500) ===")
    print(f"  delta={boot['delta']:.2f}  95% CI=[{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]  "
          f"n_pitchers={boot['n_pitchers']}")
    if boot["ci_lo"] > 0:
        print("  -> CI 전부 양수: exp25가 유의미하게 낫다")
    elif boot["ci_hi"] < 0:
        print("  -> CI 전부 음수: exp25가 유의미하게 나쁘다")
    else:
        print("  -> CI가 0을 포함: 잡음과 구분 안 됨")
