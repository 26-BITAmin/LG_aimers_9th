"""exp42 — exp41(R만 학습, F 25,686행 제외)이 exp04 대비 진짜 신호인지 잡음인지
투수 단위 클러스터 부트스트랩으로 확인 (CLAUDE.md ③). fold C만 영향받으므로 fold C만.

exp41 raw: 618.8 vs exp04 631.5 (Δ-12.7, F 제외가 더 나쁨).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import metrics as M
from src import features as feat
from src.train_utils import load_train, run_fold

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
)

if __name__ == "__main__":
    df = load_train()
    feature_cols = feat.feature_list(use_custom=True)

    print("training exp04 config (R+F, fold C)...")
    res04, _, p04, y04, valid04 = run_fold(df, "C", feature_cols, CAT_COLS, LGB_PARAMS, use_custom=True,
                                            num_boost_round=3000, early_stopping_rounds=100)
    pid04 = valid04["pitcher_id"].to_numpy()
    print(f"  exp04 score={res04['score']:.1f}")

    print("training exp41 config (R only, fold C)...")
    df_r_only = df[df["game_type"] == "R"].copy()
    res41, _, p41, y41, valid41 = run_fold(df_r_only, "C", feature_cols, CAT_COLS, LGB_PARAMS, use_custom=True,
                                            num_boost_round=3000, early_stopping_rounds=100)
    pid41 = valid41["pitcher_id"].to_numpy()
    print(f"  exp41 score={res41['score']:.1f}")

    assert (y04 == y41).all() and (pid04 == pid41).all(), "valid set mismatch between runs"

    boot = M.pitcher_cluster_bootstrap(p41, p04, y04, pid04, n_boot=500, seed=0)
    print()
    print("=== exp41(R only) - exp04(R+F), fold C, pitcher-cluster bootstrap (n_boot=500) ===")
    print(f"  delta={boot['delta']:.2f}  95% CI=[{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]  "
          f"n_pitchers={boot['n_pitchers']}")
    if boot["ci_lo"] > 0:
        print("  -> R only가 유의미하게 낫다")
    elif boot["ci_hi"] < 0:
        print("  -> R only가 유의미하게 나쁘다 (F 포함이 낫다)")
    else:
        print("  -> CI가 0을 포함: 잡음과 구분 안 됨")
