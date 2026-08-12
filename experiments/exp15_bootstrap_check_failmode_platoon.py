"""exp15 — exp14(실패 성분 좌우 매치업 분할)이 exp04 대비 진짜 신호인지 잡음인지
투수 단위 클러스터 부트스트랩으로 확인한다 (exp11/13과 동일 절차, CLAUDE.md ③).

exp14는 fold A/B/C = 550.9/475.0/630.7, exp04는 559.6/461.3/631.5 — A는 내려가고
B는 크게 오르고 C는 거의 그대로다(부호 불일치, 이미 CLAUDE.md 판정 기준 ①에서 탈락
후보). 그래도 절차상 부트스트랩 없이 기각/채택하지 않는다. fold C와, 이번엔 변화가
가장 컸던 fold B에서도 exp04 vs exp14의 짝지은 Δ와 95% CI를 계산한다.
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


def get_preds(feature_cols, use_failmode_platoon, fold_name):
    df = load_train()
    res, booster, p, yv, valid_df = run_fold(
        df, fold_name, feature_cols, CAT_COLS, LGB_PARAMS, use_custom=True,
        num_boost_round=3000, early_stopping_rounds=100,
        use_failmode_platoon=use_failmode_platoon)
    return p, yv, valid_df["pitcher_id"].to_numpy(), res


def check(fold_name):
    print(f"training exp04 config (fold {fold_name})...")
    p04, y04, pid04, res04 = get_preds(feat.feature_list(use_custom=True), False, fold_name)
    print(f"  exp04 score={res04['score']:.1f}")

    print(f"training exp14 config (fold {fold_name})...")
    p14, y14, pid14, res14 = get_preds(
        feat.feature_list(use_custom=True, use_failmode_platoon=True), True, fold_name)
    print(f"  exp14 score={res14['score']:.1f}")

    assert (y04 == y14).all() and (pid04 == pid14).all(), "valid set mismatch between runs"

    boot = M.pitcher_cluster_bootstrap(p14, p04, y04, pid04, n_boot=500, seed=0)
    print()
    print(f"=== exp14 - exp04, fold {fold_name}, pitcher-cluster bootstrap (n_boot=500) ===")
    print(f"  delta={boot['delta']:.2f}  95% CI=[{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]  "
          f"n_pitchers={boot['n_pitchers']}")
    if boot["ci_lo"] > 0:
        print("  -> CI 전부 양수: exp14가 유의미하게 낫다")
    elif boot["ci_hi"] < 0:
        print("  -> CI 전부 음수: exp14가 유의미하게 나쁘다")
    else:
        print("  -> CI가 0을 포함: 잡음과 구분 안 됨, 둘 다 통계적으로 동급")
    print()


if __name__ == "__main__":
    check("C")
    check("B")
