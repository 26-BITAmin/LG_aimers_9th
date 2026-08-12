"""exp19 — exp18(다년도 시즌 궤적)이 exp04 대비 진짜 신호인지 잡음인지
투수 단위 클러스터 부트스트랩으로 확인한다 (exp11/13/15/17과 동일 절차, CLAUDE.md ③).

exp18은 fold A/B/C = 557.0/470.9/617.1, exp04는 559.6/461.3/631.5 — A 소폭 악화,
B 개선, C는 −14.4로 지금까지의 성분 실험 중 가장 큰 폭으로 악화했다. fold C(주근거)에서
exp04 vs exp18의 짝지은 Δ와 95% CI를 계산한다.
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


def get_preds(feature_cols, use_season_trend, fold_name):
    df = load_train()
    res, booster, p, yv, valid_df = run_fold(
        df, fold_name, feature_cols, CAT_COLS, LGB_PARAMS, use_custom=True,
        num_boost_round=3000, early_stopping_rounds=100,
        use_season_trend=use_season_trend)
    return p, yv, valid_df["pitcher_id"].to_numpy(), res


if __name__ == "__main__":
    print("training exp04 config (fold C)...")
    p04, y04, pid04, res04 = get_preds(feat.feature_list(use_custom=True), False, "C")
    print(f"  exp04 score={res04['score']:.1f}")

    print("training exp18 config (fold C)...")
    p18, y18, pid18, res18 = get_preds(
        feat.feature_list(use_custom=True, use_season_trend=True), True, "C")
    print(f"  exp18 score={res18['score']:.1f}")

    assert (y04 == y18).all() and (pid04 == pid18).all(), "valid set mismatch between runs"

    boot = M.pitcher_cluster_bootstrap(p18, p04, y04, pid04, n_boot=500, seed=0)
    print()
    print("=== exp18 - exp04, fold C, pitcher-cluster bootstrap (n_boot=500) ===")
    print(f"  delta={boot['delta']:.2f}  95% CI=[{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]  "
          f"n_pitchers={boot['n_pitchers']}")
    if boot["ci_lo"] > 0:
        print("  -> CI 전부 양수: exp18이 유의미하게 낫다")
    elif boot["ci_hi"] < 0:
        print("  -> CI 전부 음수: exp18이 유의미하게 나쁘다")
    else:
        print("  -> CI가 0을 포함: 잡음과 구분 안 됨, 둘 다 통계적으로 동급")
