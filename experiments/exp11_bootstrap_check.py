"""exp11 — exp04 대비 지금까지의 피처 실험(exp08/09/10)이 진짜 신호인지 잡음인지
투수 단위 클러스터 부트스트랩으로 확인한다 (CLAUDE.md 판정 기준 ③).

exp05~10 전부 fold별 부호가 갈렸다 — 매번 새 피처/설정을 만들기 전에, 애초에
이 정도 폴드 간 흔들림이 "잡음 정상 범위"인지부터 재봐야 억지 끼워맞추기를 피할 수 있다.
fold C(주근거)에서 exp04 vs exp10(trend+failmode)의 짝지은 Δ와 95% CI를 계산한다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import metrics as M
from src.train_utils import load_train, run_fold

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
)


def get_preds(feature_cols, use_trend, use_failmode, fold_name="C"):
    df = load_train()
    res, booster, p, yv, valid_df = run_fold(
        df, fold_name, feature_cols, CAT_COLS, LGB_PARAMS, use_custom=True,
        num_boost_round=3000, early_stopping_rounds=100,
        use_trend=use_trend, use_failmode=use_failmode)
    return p, yv, valid_df["pitcher_id"].to_numpy(), res


if __name__ == "__main__":
    from src import features as feat

    print("training exp04 config (fold C)...")
    p04, y04, pid04, res04 = get_preds(feat.feature_list(use_custom=True), False, False)
    print(f"  exp04 score={res04['score']:.1f}")

    print("training exp10 config (fold C)...")
    p10, y10, pid10, res10 = get_preds(
        feat.feature_list(use_custom=True, use_trend=True, use_failmode=True), True, True)
    print(f"  exp10 score={res10['score']:.1f}")

    assert (y04 == y10).all() and (pid04 == pid10).all(), "valid set mismatch between runs"

    boot = M.pitcher_cluster_bootstrap(p10, p04, y04, pid04, n_boot=500, seed=0)
    print()
    print("=== exp10 - exp04, fold C, pitcher-cluster bootstrap (n_boot=500) ===")
    print(f"  delta={boot['delta']:.2f}  95% CI=[{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]  "
          f"n_pitchers={boot['n_pitchers']}")
    if boot["ci_lo"] > 0:
        print("  -> CI 전부 양수: exp10이 유의미하게 낫다")
    elif boot["ci_hi"] < 0:
        print("  -> CI 전부 음수: exp10이 유의미하게 나쁘다")
    else:
        print("  -> CI가 0을 포함: 잡음과 구분 안 됨, 둘 다 통계적으로 동급")
