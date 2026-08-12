"""exp21 — exp20(CatBoost, exp04와 동일 피처셋)이 exp04(LightGBM) 대비 진짜 신호인지
잡음인지 투수 단위 클러스터 부트스트랩으로 확인한다 (exp11/13/15/17/19와 동일 절차).

exp20은 fold A/B/C = 523.8/617.7/696.9, exp04는 559.6/461.3/631.5 — A는 −35.8로
꽤 떨어졌지만 B는 +156.4, C는 +65.4로 지금까지 이 세션의 모든 실험(±10~30점 수준)과는
완전히 다른 스케일이다. 이 정도 크기면 잡음 바닥(투수 클러스터 280~291명 기준
대략 ±20~30점)을 크게 웃돌 가능성이 높지만, 확실히 하기 위해 fold C(주근거)와
fold A(유일하게 나빠진 폴드)에서 부트스트랩으로 확인한다.
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


def get_preds_lgb(fold_name):
    df = load_train()
    res, booster, p, yv, valid_df = run_fold(
        df, fold_name, feat.feature_list(use_custom=True), CAT_COLS, LGB_PARAMS, use_custom=True,
        num_boost_round=3000, early_stopping_rounds=100)
    return p, yv, valid_df["pitcher_id"].to_numpy(), res


def get_preds_cb(fold_name):
    df = load_train()
    res, model, p, yv, valid_df = run_fold_cb(
        df, fold_name, feat.feature_list(use_custom=True), CAT_COLS, CB_PARAMS, use_custom=True,
        num_boost_round=3000, early_stopping_rounds=100)
    return p, yv, valid_df["pitcher_id"].to_numpy(), res


def check(fold_name):
    print(f"training exp04 (LightGBM) config (fold {fold_name})...")
    p04, y04, pid04, res04 = get_preds_lgb(fold_name)
    print(f"  exp04 score={res04['score']:.1f}")

    print(f"training exp20 (CatBoost) config (fold {fold_name})...")
    p20, y20, pid20, res20 = get_preds_cb(fold_name)
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
    print()


if __name__ == "__main__":
    check("C")
    check("A")
