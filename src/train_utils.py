"""실험 스크립트 공통 러너. 검증 하니스(folds/metrics)는 실험 중 건드리지 않는다는 원칙에 따라
여기 있는 로직도 "고정 배관"으로 취급 — 실험마다 달라지는 건 피처 목록/LightGBM 파라미터뿐이고
그건 각 experiments/expNN_*.py 파일에서 정한다.
"""
import csv
import os
import subprocess
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import folds as F
from . import metrics as M
from . import features as feat

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "..", "results.csv")


def load_train():
    return pd.read_csv(os.path.join(DATA_DIR, "train.csv"))


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        cwd=os.path.dirname(__file__)).decode().strip()
    except Exception:
        return "unknown"


def run_fold(df, fold_name, feature_cols, cat_cols, lgb_params, use_custom,
             num_boost_round=2000, early_stopping_rounds=50, seed=0,
             use_trend=False, use_failmode=False, use_ballstrike=False,
             use_failmode_platoon=False, use_n_exposed=False, use_season_trend=False,
             use_team_hier=False):
    train_mask, valid_mask = F.split(df, fold_name)
    fit_df, valid_df = feat.build_fold_frames(df, train_mask, valid_mask, use_custom=use_custom,
                                               use_trend=use_trend, use_failmode=use_failmode,
                                               use_ballstrike=use_ballstrike,
                                               use_failmode_platoon=use_failmode_platoon,
                                               use_n_exposed=use_n_exposed,
                                               use_season_trend=use_season_trend,
                                               use_team_hier=use_team_hier)

    Xt = fit_df[feature_cols].copy()
    Xv = valid_df[feature_cols].copy()
    yt = fit_df[feat.TARGET].to_numpy()
    yv = valid_df[feat.TARGET].to_numpy()

    for c in cat_cols:
        Xt[c] = Xt[c].astype("category")
        Xv[c] = Xv[c].astype("category")

    dtrain = lgb.Dataset(Xt, label=yt, categorical_feature=cat_cols, free_raw_data=False)
    dvalid = lgb.Dataset(Xv, label=yv, categorical_feature=cat_cols, free_raw_data=False, reference=dtrain)

    params = dict(lgb_params)
    params.setdefault("objective", "binary")
    params.setdefault("metric", "binary_logloss")
    params.setdefault("seed", seed)
    params.setdefault("verbosity", -1)

    booster = lgb.train(
        params, dtrain, num_boost_round=num_boost_round,
        valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
    )

    p = booster.predict(Xv, num_iteration=booster.best_iteration)
    sp = M.score_pair(p, yv)
    sc = M.score(p, yv)
    return dict(fold=fold_name, score=sc, n_va=len(yv),
                n_pitchers=int(valid_df["pitcher_id"].nunique()),
                best_iter=booster.best_iteration, **sp), booster, p, yv, valid_df


def log_result(label, fold, res, seed=0, notes=""):
    row = dict(label=label, fold=fold, score=round(res["score"], 3),
               pts_res=round(res["pts_res"], 3), pts_raw=round(res["pts_raw"], 3),
               bias=round(res["bias"], 5), n_va=res["n_va"], n_pitchers=res["n_pitchers"],
               seed=seed, git_sha=git_sha(), notes=notes)
    write_header = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "fold", "score", "pts_res", "pts_raw",
                                           "bias", "n_va", "n_pitchers", "seed", "git_sha", "notes"])
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_experiment(label, feature_cols, cat_cols, lgb_params, use_custom,
                    fold_names=("A", "B", "C"), num_boost_round=2000,
                    early_stopping_rounds=50, seed=0, notes="",
                    use_trend=False, use_failmode=False, use_ballstrike=False,
                    use_failmode_platoon=False, use_n_exposed=False, use_season_trend=False,
                    use_team_hier=False):
    t0 = time.time()
    df = load_train()
    print(f"=== {label} ===  features={len(feature_cols)}  custom={use_custom}")
    results = {}
    for fn in fold_names:
        res, booster, p, yv, valid_df = run_fold(df, fn, feature_cols, cat_cols, lgb_params,
                                                   use_custom, num_boost_round,
                                                   early_stopping_rounds, seed,
                                                   use_trend=use_trend, use_failmode=use_failmode,
                                                   use_ballstrike=use_ballstrike,
                                                   use_failmode_platoon=use_failmode_platoon,
                                                   use_n_exposed=use_n_exposed,
                                                   use_season_trend=use_season_trend,
                                                   use_team_hier=use_team_hier)
        results[fn] = res
        log_result(label, fn, res, seed=seed, notes=notes)
        print(f"  fold {fn}: score={res['score']:.1f}  pts_res={res['pts_res']:.1f}  "
              f"bias={res['bias']:.5f}  best_iter={res['best_iter']}  n_va={res['n_va']:,}")
    print(f"  elapsed {time.time() - t0:.1f}s")
    return results
