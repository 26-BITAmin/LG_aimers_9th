"""exp17 — exp16(frozen 표본 크기 n 노출)이 exp04 대비 진짜 신호인지 잡음인지
투수 단위 클러스터 부트스트랩으로 확인한다 (exp11/13/15와 동일 절차, CLAUDE.md ③).

exp16은 fold A/B/C = 562.6/472.5/630.2, exp04는 559.6/461.3/631.5 — A/B는 개선,
C는 거의 그대로(-1.3)다. exp05~15에서 계속 나왔던 "부호 불일치"가 처음으로
거의 안 나온 경우라 그나마 가장 유망한 후보지만, CLAUDE.md 판정 기준 ③(CI)을
통과하기 전까지는 채택하지 않는다. fold C(주근거)와 fold A에서 exp04 vs exp16의
짝지은 Δ와 95% CI를 계산한다.
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


def get_preds(feature_cols, use_n_exposed, fold_name):
    df = load_train()
    res, booster, p, yv, valid_df = run_fold(
        df, fold_name, feature_cols, CAT_COLS, LGB_PARAMS, use_custom=True,
        num_boost_round=3000, early_stopping_rounds=100,
        use_n_exposed=use_n_exposed)
    return p, yv, valid_df["pitcher_id"].to_numpy(), res


def check(fold_name):
    print(f"training exp04 config (fold {fold_name})...")
    p04, y04, pid04, res04 = get_preds(feat.feature_list(use_custom=True), False, fold_name)
    print(f"  exp04 score={res04['score']:.1f}")

    print(f"training exp16 config (fold {fold_name})...")
    p16, y16, pid16, res16 = get_preds(
        feat.feature_list(use_custom=True, use_n_exposed=True), True, fold_name)
    print(f"  exp16 score={res16['score']:.1f}")

    assert (y04 == y16).all() and (pid04 == pid16).all(), "valid set mismatch between runs"

    boot = M.pitcher_cluster_bootstrap(p16, p04, y04, pid04, n_boot=500, seed=0)
    print()
    print(f"=== exp16 - exp04, fold {fold_name}, pitcher-cluster bootstrap (n_boot=500) ===")
    print(f"  delta={boot['delta']:.2f}  95% CI=[{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]  "
          f"n_pitchers={boot['n_pitchers']}")
    if boot["ci_lo"] > 0:
        print("  -> CI 전부 양수: exp16이 유의미하게 낫다")
    elif boot["ci_hi"] < 0:
        print("  -> CI 전부 음수: exp16이 유의미하게 나쁘다")
    else:
        print("  -> CI가 0을 포함: 잡음과 구분 안 됨, 둘 다 통계적으로 동급")
    print()


if __name__ == "__main__":
    check("C")
    check("A")
