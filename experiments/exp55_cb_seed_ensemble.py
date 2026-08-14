"""exp55 — exp54(LightGBM 시드 앙상블)와 동일 방법론을 CatBoost에도 적용.

exp54: LightGBM 5시드 평균 예측이 세 fold 모두에서 개별 시드 평균 점수보다
우수(+4.8~+8.9) -- 진짜 분산 축소 효과 확인. CatBoost도 같은 메커니즘이
작동하는지 검증.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src import features as feat
from src import metrics as M
from src.train_utils import load_train
from src.train_utils_cb import run_fold_cb

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

CB_PARAMS = dict(learning_rate=0.03, depth=8, l2_leaf_reg=5.0)

SEEDS = (0, 1, 2, 3, 4)
BASELINE_SEED0 = {"A": 525.7, "B": 597.0, "C": 722.5}  # exp31 CatBoost depth=8, seed=0

if __name__ == "__main__":
    df = load_train()
    feature_cols = feat.feature_list(use_custom=True)

    for fold_name in ("A", "B", "C"):
        print(f"=== fold {fold_name} ===")
        preds = []
        singles = []
        yv_ref = None
        for seed in SEEDS:
            res, model, p, yv, valid_df = run_fold_cb(df, fold_name, feature_cols, CAT_COLS, CB_PARAMS,
                                                        use_custom=True, num_boost_round=3000,
                                                        early_stopping_rounds=100, seed=seed)
            if yv_ref is None:
                yv_ref = yv
            else:
                assert (yv == yv_ref).all()
            preds.append(p)
            singles.append(res["score"])
            print(f"  seed={seed}: score={res['score']:.1f}  best_iter={res['best_iter']}")

        p_avg = np.mean(preds, axis=0)
        sc_avg = M.score(p_avg, yv_ref)
        mean_single = float(np.mean(singles))
        baseline = BASELINE_SEED0[fold_name]

        print(f"  -> 5-seed averaged-prediction score = {sc_avg:.1f}")
        print(f"  -> mean of 5 single-seed scores      = {mean_single:.1f}")
        print(f"  -> baseline (seed=0 only)             = {baseline:.1f}")
        print(f"  -> Delta(avg_pred - baseline)        = {sc_avg - baseline:+.1f}")
        print(f"  -> Delta(avg_pred - mean_single)     = {sc_avg - mean_single:+.1f}")
