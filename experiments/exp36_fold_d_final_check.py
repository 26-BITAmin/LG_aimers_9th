"""exp36 — fold D 최종 확인 (CLAUDE.md: 재튜닝 금지, 한 번만).

exp32(LightGBM exp04 + CatBoost depth=8, 50/50 고정 블렌드)이 A/B/C 튜닝을 거친 현재
최선 설정. exp34(nested-holdout 블렌드 가중치)는 exp35에서 부호 불일치로 기각됐다.
이제 fold D(2019-2023 학습 -> 2024 검증, 최종 확인 전용)에서 딱 한 번 확인한다.
이 결과로 하이퍼파라미터를 다시 바꾸지 않는다 — 실제 배포(2025 test) 앞의 최종 리허설.

best_iteration도 같이 기록해서, 전체 데이터(2019-2024)로 최종 재학습할 때 쓸
num_boost_round 참고치로 삼는다 (fold D의 학습 구간 2019-2023이 전체 학습 구간과
가장 가까움).
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
CB_PARAMS = dict(learning_rate=0.03, depth=8, l2_leaf_reg=5.0)

LABEL = "exp36_fold_d_final_check"
NOTES = "exp32 config (LightGBM exp04 + CatBoost depth=8, 50/50 fixed blend) on fold D -- FINAL CHECK ONLY, never retuned from this"

if __name__ == "__main__":
    df = load_train()
    feature_cols = feat.feature_list(use_custom=True)
    fold_name = "D"

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
    print(f"fold D -- LightGBM alone: {res_lgb['score']:.1f} (best_iter={res_lgb['best_iter']})   "
          f"CatBoost alone: {res_cb['score']:.1f} (best_iter={res_cb['best_iter']})")
    print(f"  ensemble: {res['score']:.1f}  pts_res={res['pts_res']:.1f}  bias={res['bias']:.5f}")
