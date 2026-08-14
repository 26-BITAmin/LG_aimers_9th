"""exp54 — LightGBM 시드 앙상블: 확실한 근본 개선 후보.

exp47에서 실측한 사실: 우리 LightGBM(exp04 config) 단일모델의 학습 seed 노이즈가
fold당 std 4.78~7.59점이나 된다. 이건 "판정 임계값"으로만 썼는데, 뒤집어보면
**모델 자체의 분산이 그만큼 크다**는 뜻이고, 여러 seed의 예측을 평균하면 그
분산을 줄여 Brier score 자체를 낮출 수 있다 (LGB+CatBoost 50/50 블렌드가
+99~156점을 낸 것과 같은 메커니즘 — 서로 다른 예측의 평균은 분산을 줄인다).

wonwoo 브랜치가 검증한 법칙("분산 축소는 항상 LB로 전이된다, 정보 손실과 다른
카테고리다")에 정확히 해당하는 저위험 레버. 피처·전처리는 전혀 안 바꾸고 seed만
바꾼 5개 모델의 예측을 평균 — "한 번에 하나만 바꾼다" 원칙 완벽히 준수.

검증: fold A/B/C에서 5개 seed(0~4)로 LightGBM(exp04 config) 각각 학습 후
확률 예측을 산술평균 -- 이 평균 예측의 점수를 (a) 단일 seed=0 베이스라인,
(b) 5개 단일모델 점수의 평균과 비교한다. (a)(b) 둘 다보다 좋아야 진짜 이득.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src import features as feat
from src import metrics as M
from src.train_utils import load_train, run_fold

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
)

SEEDS = (0, 1, 2, 3, 4)
BASELINE_SEED0 = {"A": 559.6, "B": 461.3, "C": 631.5}  # exp04/exp47 seed=0

if __name__ == "__main__":
    df = load_train()
    feature_cols = feat.feature_list(use_custom=True)

    for fold_name in ("A", "B", "C"):
        print(f"=== fold {fold_name} ===")
        preds = []
        singles = []
        yv_ref = None
        for seed in SEEDS:
            res, booster, p, yv, valid_df = run_fold(df, fold_name, feature_cols, CAT_COLS, LGB_PARAMS,
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
        print(f"  -> Delta(avg_pred - mean_single)     = {sc_avg - mean_single:+.1f}  "
              f"(양수면 '평균 예측'이 '평균 점수'보다 우월 -- 진짜 앙상블 효과)")
