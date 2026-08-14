"""exp41 — 학습 데이터에서 F를 완전히 뺀다 (R만으로 학습). 한 번도 안 해본 구조적 변경.

지금까지 `folds._drop_bad_f()`로 2022 이전 F(라벨 생성 방식이 다름)만 제외했고,
2023~2024의 "정상" F는 학습에 그대로 남아있었다. valid는 항상 R 전용인데, 학습에
F를 섞는 게 실제로 이득인지 손해인지 한 번도 측정한 적이 없다.

fold별 영향:
- fold A(train 2019-2021), fold B(train 2020-2022) — 학습 구간 전체가 이미
  `_drop_bad_f` 대상(season<=2022)이라 F가 원래도 0행. **이 실험은 무영향** —
  재실행해도 exp04와 완전히 같은 숫자가 나올 뿐이라 fold C만 돈다.
- fold C(train 2021-2023) — 2023 F 25,686행(학습 데이터의 3.75%)이 이번에 빠진다.
  EDA.md §3: F 2023 rate .4729 vs R 2023 rate .5031 — 실제 격차가 있는 다른
  분포다. 공유 트리 분기에 섞이는 게 노이즈인지 유용한 추가 데이터인지 확인.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import features as feat
from src import folds as F
from src.train_utils import load_train, run_fold, log_result

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
)

LABEL = "exp41_train_r_only"
NOTES = "drop all game_type=='F' rows from training (not just pre-2023 bad-F), fold C only -- A/B train range has no F left after _drop_bad_f anyway"

if __name__ == "__main__":
    df = load_train()
    # R만 남기고, 기존 _drop_bad_f는 이미 R 행에 영향 없으므로 그대로 둬도 무방하지만
    # 명시적으로 game_type=='R'만 남긴다.
    df_r_only = df[df["game_type"] == "R"].copy()
    feature_cols = feat.feature_list(use_custom=True)

    for fold_name in ("C",):
        res, booster, p, yv, valid_df = run_fold(
            df_r_only, fold_name, feature_cols, CAT_COLS, LGB_PARAMS, use_custom=True,
            num_boost_round=3000, early_stopping_rounds=100)
        log_result(LABEL, fold_name, res, notes=NOTES)
        print(f"  fold {fold_name}: score={res['score']:.1f}  pts_res={res['pts_res']:.1f}  "
              f"bias={res['bias']:.5f}  best_iter={res['best_iter']}  n_va={res['n_va']:,}")
