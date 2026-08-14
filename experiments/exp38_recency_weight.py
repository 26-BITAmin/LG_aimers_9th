"""exp38 — 학습 표본에 최근 시즌 지수 가중치를 준다 (wonwoo 브랜치 아이디어 참고, 새로 실측).

wonwoo 브랜치(origin/wonwoo, LB 975.74)의 MODEL.md 7.2절이 이 축에서 큰 효과를
보고했다(동일 가중 대비 BSS oracle +130~+240, half_life=1.0 근방). 코드는 그대로
가져오지 않고, 우리 폴드 프로토콜(A/B/C, R-only valid, exp04 파라미터)에서 우리
자체로 재검증한다.

주의할 차이점: 우리 fold의 train 구간은 3개 시즌뿐이다(wonwoo는 최대 6개 시즌
2019-2024를 씀) — 감쇠 효과가 그쪽만큼 크지 않을 수 있다. 비교 기준은 exp04
(동일 피처·동일 LightGBM 파라미터, 가중치만 없음, results.csv에 이미 기록됨).

가중치: w = 0.5 ** ((train_max_season - season) / half_life), half_life=1.0.
game_type은 손대지 않음(별도 축, 한 번에 하나만 바꾼다).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lightgbm as lgb

from src import features as feat
from src import folds as F
from src import metrics as M
from src.train_utils import load_train, log_result

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
    objective="binary", metric="binary_logloss", seed=0, verbosity=-1,
)
HALF_LIFE = 1.0

LABEL = "exp38_recency_weight_hl1.0"
NOTES = "exp04 + exponential recency sample_weight (half_life=1.0 season), vs exp04 (no weight)"

if __name__ == "__main__":
    df = load_train()
    feature_cols = feat.feature_list(use_custom=True)

    for fold_name in ("A", "B", "C"):
        train_mask, valid_mask = F.split(df, fold_name)
        fit_df, valid_df = feat.build_fold_frames(df, train_mask, valid_mask, use_custom=True)

        Xt = fit_df[feature_cols].copy()
        Xv = valid_df[feature_cols].copy()
        yt = fit_df[feat.TARGET].to_numpy()
        yv = valid_df[feat.TARGET].to_numpy()
        for c in CAT_COLS:
            Xt[c] = Xt[c].astype("category")
            Xv[c] = Xv[c].astype("category")

        train_max_season = fit_df["season"].max()
        w = 0.5 ** ((train_max_season - fit_df["season"]).to_numpy() / HALF_LIFE)

        dtrain = lgb.Dataset(Xt, label=yt, weight=w, categorical_feature=CAT_COLS, free_raw_data=False)
        dvalid = lgb.Dataset(Xv, label=yv, categorical_feature=CAT_COLS, free_raw_data=False, reference=dtrain)

        booster = lgb.train(
            LGB_PARAMS, dtrain, num_boost_round=3000,
            valid_sets=[dvalid], valid_names=["valid"],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        p = booster.predict(Xv, num_iteration=booster.best_iteration)
        sp = M.score_pair(p, yv)
        sc = M.score(p, yv)
        res = dict(fold=fold_name, score=sc, n_va=len(yv),
                   n_pitchers=int(valid_df["pitcher_id"].nunique()),
                   best_iter=booster.best_iteration, **sp)
        log_result(LABEL, fold_name, res, notes=NOTES)
        print(f"  fold {fold_name}: score={res['score']:.1f}  pts_res={res['pts_res']:.1f}  "
              f"bias={res['bias']:.5f}  best_iter={res['best_iter']}  "
              f"season_weight_range=[{w.min():.3f},{w.max():.3f}]")
