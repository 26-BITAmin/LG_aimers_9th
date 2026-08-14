"""exp40 — 절편(intercept) 보정 재시도: 추세선 외삽 방식.

exp06(nested-holdout 프록시 모델로 bias_hat 추정)은 세 폴드 전부 악화시켰다
(A -5.7, B -10.1, C -38.0, results.csv 참조). 그런데 fold C의 pts_res(672.5) -
score(631.5) = 41점, fold A는 10.9점 격차가 있다 — "완벽한 절편 보정"의 이론적
여지는 실재한다(exp04의 bias 컬럼: A=+0.00521, C=+0.01012). exp06이 실패한 건
아이디어가 아니라 추정법일 가능성: 프록시 모델을 학습 구간에서 시즌 하나를 통째로
빼서 만들었는데, 이건 (a) 더 적은 데이터로 학습된 약한 모델의 편향과 (b) 진짜
드리프트 편향을 뒤섞고, (c) 단일 시즌 관측치라 추정 분산이 크다.

EDA.md §3: R 전용 base rate는 6개 시즌에 걸쳐 매끄러운 단조 하락
(.5495/.5269/.5128/.5037/.5031/.4897) — 노이즈가 아니라 추세다. 프록시 모델 대신
**그 추세선을 선형 외삽**해서 valid 시즌의 목표 prior를 추정하고, 실제 모델(약화
안 된 풀파워)을 자기 학습 구간의 마지막 시즌(R only, in-sample)에 적용해 현재
레벨(ref_mean)을 잰다. delta = logit(target) - logit(ref_mean)을 valid 예측에
로짓 이동으로 적용.

리키지 없음: target(추세선)과 ref_mean(자기 학습 구간 마지막 시즌 예측) 둘 다
valid 라벨을 전혀 안 쓴다. valid 시즌 season 값 자체도 안 쓴다(추세선 외삽으로
독립 추정).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src import folds as F
from src import metrics as M
from src import features as feat
from src.train_utils import load_train, run_fold, log_result

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
)

LABEL = "exp40_trend_intercept_calibration"
NOTES = ("trend-extrapolated logit-shift intercept correction on top of exp04, "
         "vs exp06's failed nested-proxy bias estimate")


def trend_target_prior(fit_raw, valid_year):
    """fold train 구간의 시즌별 R base rate에 선형 회귀를 적합해 valid_year로 외삽."""
    r = fit_raw.loc[fit_raw["game_type"] == "R"]
    by_season = r.groupby("season")[feat.TARGET].mean()
    seasons = by_season.index.to_numpy(dtype=float)
    rates = by_season.to_numpy(dtype=float)
    slope, intercept = np.polyfit(seasons, rates, 1)
    target = slope * valid_year + intercept
    return float(np.clip(target, 0.01, 0.99)), dict(zip(by_season.index.tolist(), rates.tolist())), slope


if __name__ == "__main__":
    df = load_train()
    feature_cols = feat.feature_list(use_custom=True)

    for fold_name in ("A", "B", "C"):
        train_mask, valid_mask = F.split(df, fold_name)
        fit_raw = df.loc[train_mask].copy()
        valid_year = int(df.loc[valid_mask, "season"].iloc[0])

        target_prior, by_season, slope = trend_target_prior(fit_raw, valid_year)

        # 실제(풀파워) 모델을 학습하고, 자기 학습구간 마지막 시즌(R only, in-sample)에 적용해
        # 현재 레벨을 잰다. run_fold이 이미 fit_df/booster/valid 예측을 주므로 재사용.
        res_raw, booster, p_valid, y_valid, valid_df = run_fold(
            df, fold_name, feature_cols, CAT_COLS, LGB_PARAMS, use_custom=True,
            num_boost_round=3000, early_stopping_rounds=100)

        last_season = max(by_season.keys())
        ref_raw = fit_raw[(fit_raw["season"] == last_season) & (fit_raw["game_type"] == "R")].copy()
        lookup = feat.fit_pitcher_lookup(fit_raw[fit_raw["season"] < last_season])
        static_feats = feat.apply_static_features(ref_raw, lookup)
        import pandas as pd
        ref_df = pd.concat([ref_raw.reset_index(drop=True), static_feats.reset_index(drop=True)], axis=1)
        Xref = ref_df[feature_cols].copy()
        for c in CAT_COLS:
            Xref[c] = Xref[c].astype("category")
        p_ref = booster.predict(Xref, num_iteration=booster.best_iteration)
        ref_mean = float(np.clip(p_ref.mean(), 1e-6, 1 - 1e-6))

        target_clipped = float(np.clip(target_prior, 1e-6, 1 - 1e-6))
        delta = np.log(target_clipped / (1 - target_clipped)) - np.log(ref_mean / (1 - ref_mean))

        logit_p = np.log(np.clip(p_valid, 1e-6, 1 - 1e-6) / (1 - np.clip(p_valid, 1e-6, 1 - 1e-6)))
        p_cal = 1.0 / (1.0 + np.exp(-(logit_p + delta)))

        sp = M.score_pair(p_cal, y_valid)
        sc = M.score(p_cal, y_valid)
        res = dict(fold=fold_name, score=sc, n_va=len(y_valid),
                   n_pitchers=int(valid_df["pitcher_id"].nunique()), best_iter=res_raw["best_iter"], **sp)
        log_result(LABEL, fold_name, res, notes=NOTES)
        print(f"  fold {fold_name}: by_season={by_season}  slope={slope:.5f}  "
              f"target_prior={target_prior:.4f}  ref_mean={ref_mean:.4f}  delta={delta:.4f}")
        print(f"    raw: score={res_raw['score']:.1f}  calibrated: score={res['score']:.1f}  "
              f"(Δ={res['score']-res_raw['score']:+.1f})  bias_cal={res['bias']:.5f}")
