"""exp01 — 가장 기본 LightGBM 베이스라인.

주최 측 제공 47피처 그대로, 커스텀 피처 없음. RF 베이스라인(fold C anchor 420.48,
fold D anchor 415.57)과 비교하기 위한 출발점. 하이퍼파라미터는 LightGBM 기본값에
가까운 수수한 값 — 튜닝은 다음 실험부터.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import features as feat
from src.train_utils import run_experiment

LGB_PARAMS = dict(
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=200,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
)

if __name__ == "__main__":
    run_experiment(
        label="exp01_baseline_official",
        feature_cols=feat.feature_list(use_custom=False),
        cat_cols=feat.categorical_list(use_custom=False),
        lgb_params=LGB_PARAMS,
        use_custom=False,
        fold_names=("A", "B", "C"),
        notes="47 official features only, no custom",
    )
