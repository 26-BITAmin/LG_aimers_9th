"""exp23 — LightGBM(exp04) + CatBoost(exp20) 단순 평균 앙상블.

exp20/21/22에서 확인한 것: CatBoost는 fold B(+156.4, CI 전부 양수)와 fold C(+65.4,
CI 전부 양수, 주근거 폴드)에서 LightGBM을 크게, 통계적으로 유의미하게 앞섰지만
fold A(−35.8, CI 전부 음수)에서는 유의미하게 졌다. CLAUDE.md 판정 기준상 fold A/B
단독 판단은 금지되고, C가 "드리프트 대응 주근거"라 최근 시즌(2025 배포 타깃)에
더 가까운 폴드일수록 CatBoost가 이긴다는 점이 특히 중요하다.

두 모델이 fold별로 상반된 강점을 보이는 건 앙상블의 교과서적 사용 사례다 — 검증
라벨을 보고 blend 가중치를 맞추는 게 아니라(그건 test 기반 가중치 폐기 사유와
같은 함정), 가장 단순하고 원칙적인 50/50 단순 평균으로 시작한다. fold A의 손실을
줄이면서 B/C의 이득을 얼마나 보존하는지 확인.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import features as feat
from src import metrics as M
from src.train_utils import load_train, run_fold, log_result
from src.train_utils_cb import run_fold_cb

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
)
CB_PARAMS = dict(learning_rate=0.03, depth=6, l2_leaf_reg=5.0)

LABEL = "exp23_ensemble_lgb_catboost"
NOTES = "0.5*LightGBM(exp04) + 0.5*CatBoost(exp20), identical feature set both models, fixed unweighted blend"

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
        assert (valid_lgb["pitcher_id"].to_numpy() == valid_cb["pitcher_id"].to_numpy()).all()

        p_ens = 0.5 * p_lgb + 0.5 * p_cb
        sp = M.score_pair(p_ens, y_lgb)
        sc = M.score(p_ens, y_lgb)
        res = dict(fold=fold_name, score=sc, n_va=len(y_lgb),
                   n_pitchers=int(valid_lgb["pitcher_id"].nunique()), best_iter=-1, **sp)
        log_result(LABEL, fold_name, res, notes=NOTES)
        print(f"  LightGBM alone: {res_lgb['score']:.1f}   CatBoost alone: {res_cb['score']:.1f}"
              f"   ensemble: {res['score']:.1f}  pts_res={res['pts_res']:.1f}  bias={res['bias']:.5f}")
