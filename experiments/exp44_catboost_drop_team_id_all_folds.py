"""exp44 — exp43(fold A만, team_id 제거로 +15.7)을 A/B/C 전체로 확장.

exp43은 fold A만 봤다(541.4 vs exp31 525.7, Δ+15.7) -- team_id CTR이 오래된 학습
구간에서 CatBoost를 해친다는 가설과 일치. 하지만 fold A만으로는 "fold 부호 일관성"을
판단할 수 없다. B/C에서 team_id를 빼는 게 손해인지(원래 팀 정보가 유용했다면) 이득인지
확인해야 전체 그림이 나온다.
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

    run_experiment_cb(
        label="exp44_catboost_drop_team_id_all_folds",
        feature_cols=feature_cols,
        cat_cols=CAT_COLS_NO_TEAM,
        cb_params=CB_PARAMS,
        use_custom=True,
        fold_names=("B", "C"),
        num_boost_round=3000,
        early_stopping_rounds=100,
        notes="CatBoost depth=8 (exp31 params) with pitcher_team_id/batter_team_id dropped, "
              "B/C to check sign consistency after exp43's fold A +15.7 -- vs exp31 (597.0/722.5)",
    )
