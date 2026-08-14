"""exp50 — exp49(팀 계층 축소 피처)를 5seed 평균으로 재확인.

exp49 단일 seed(=0) 결과: A +1.8, B +13.7, C -5.1 (vs exp04 베이스라인).
exp47 seed 게이트(2σ: A=9.6/B=12.1/C=15.2)로 걸러보면 B만 통과 -- A/C는 판정 보류
(seed=0의 운일 수 있음). exp47과 동일하게 5seed 평균을 내서 노이즈를 sqrt(5)만큼
줄이고 다시 비교한다. 베이스라인(exp04, 팀 피처 없음)도 exp47에서 이미 5seed
평균을 냈으므로 그 값(A=557.4, B=466.5, C=623.5)을 그대로 기준선으로 쓴다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src import features as feat
from src.train_utils import run_experiment

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
)

SEEDS = (0, 1, 2, 3, 4)
BASELINE_MEAN = {"A": 557.4, "B": 466.5, "C": 623.5}  # exp47, 5-seed mean, no team-hier feature

if __name__ == "__main__":
    feature_cols = feat.feature_list(use_custom=True, use_team_hier=True)
    scores = {fn: [] for fn in ("A", "B", "C")}

    for seed in SEEDS:
        results = run_experiment(
            label="exp50_team_hier_seed_avg",
            feature_cols=feature_cols,
            cat_cols=CAT_COLS,
            lgb_params=LGB_PARAMS,
            use_custom=True,
            fold_names=("A", "B", "C"),
            num_boost_round=3000,
            early_stopping_rounds=100,
            seed=seed,
            use_team_hier=True,
            notes=f"5-seed reconfirm of exp49 (team-hier feature) vs exp47 baseline mean, seed={seed}",
        )
        for fn, res in results.items():
            scores[fn].append(res["score"])

    print("\n=== exp50 summary: team-hier feature, 5-seed mean vs exp47 baseline mean ===")
    for fn in ("A", "B", "C"):
        arr = np.array(scores[fn])
        delta = arr.mean() - BASELINE_MEAN[fn]
        print(f"  fold {fn}: scores={np.round(arr, 1).tolist()}  mean={arr.mean():.1f}  "
              f"std={arr.std(ddof=1):.2f}  baseline={BASELINE_MEAN[fn]:.1f}  Delta={delta:+.2f}")
