"""exp43 — CatBoost가 fold A에서만 지는 원인 진단: team_id 범주형 인코딩 가설.

model.md "왜 CatBoost가 fold별로 이렇게 갈렸나" (미해결): fold A(2019-21→2022)에서만
CatBoost가 유의미하게 졌다(exp21: Δ=-35.85, CI=[-70.58,-3.83]). 가설: CatBoost의
categorical target statistics(CTR)가 `pitcher_team_id`/`batter_team_id`를
학습 구간의 팀 성적으로 인코딩하는데, fold A는 가장 오래된 학습 구간(2019-2021)이라
그 인코딩이 2022(배포 타깃과 더 가까운 시즌)에 가장 안 맞았을 수 있다.

진단: CatBoost(exp31 depth=8 설정 그대로)에서 team_id 두 컬럼만 빼고 fold A만 재실행.
회복되면 가설 지지, 그대로면 다른 원인.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import features as feat
from src.train_utils_cb import run_experiment_cb

CAT_COLS_NO_TEAM = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
                     "pitcher_hand", "batter_hand"]

CB_PARAMS = dict(learning_rate=0.03, depth=8, l2_leaf_reg=5.0)

if __name__ == "__main__":
    all_feats = feat.feature_list(use_custom=True)
    feature_cols = [c for c in all_feats if c not in ("pitcher_team_id", "batter_team_id")]
    print(f"features: {len(all_feats)} -> {len(feature_cols)} (dropped pitcher_team_id, batter_team_id)")

    run_experiment_cb(
        label="exp43_catboost_drop_team_id",
        feature_cols=feature_cols,
        cat_cols=CAT_COLS_NO_TEAM,
        cb_params=CB_PARAMS,
        use_custom=True,
        fold_names=("A",),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="CatBoost depth=8 (exp31 params) with pitcher_team_id/batter_team_id dropped entirely, "
              "fold A only -- diagnosing whether team_id CTR causes CatBoost's fold-A weakness "
              "(exp21: CatBoost lost fold A by -35.85, CI excludes 0)",
    )
