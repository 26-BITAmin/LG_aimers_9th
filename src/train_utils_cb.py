"""CatBoost용 실험 러너. `train_utils.py`(LightGBM 고정 배관)는 19개 실험이 이미 의존하고
있어 손대지 않는다 — 이 파일은 그 옆에 완전히 별개로 추가된 것이다. folds/metrics/features는
동일하게 재사용(검증 하니스는 모델과 무관하게 고정)하고, 모델 학습·저장 로직만 CatBoost로 바꾼다.
`log_result`는 `train_utils.py`의 것을 그대로 재사용해 `results.csv` 스키마를 통일한다.
"""
import time

import catboost as cb
import numpy as np

from . import folds as F
from . import metrics as M
from . import features as feat
from .train_utils import load_train, log_result


def run_fold_cb(df, fold_name, feature_cols, cat_cols, cb_params, use_custom,
                num_boost_round=3000, early_stopping_rounds=100, seed=0,
                use_trend=False, use_failmode=False, use_ballstrike=False,
                use_failmode_platoon=False, use_n_exposed=False, use_season_trend=False):
    train_mask, valid_mask = F.split(df, fold_name)
    fit_df, valid_df = feat.build_fold_frames(df, train_mask, valid_mask, use_custom=use_custom,
                                               use_trend=use_trend, use_failmode=use_failmode,
                                               use_ballstrike=use_ballstrike,
                                               use_failmode_platoon=use_failmode_platoon,
                                               use_n_exposed=use_n_exposed,
                                               use_season_trend=use_season_trend)

    Xt = fit_df[feature_cols].copy()
    Xv = valid_df[feature_cols].copy()
    yt = fit_df[feat.TARGET].to_numpy()
    yv = valid_df[feat.TARGET].to_numpy()

    for c in cat_cols:
        Xt[c] = Xt[c].astype(str)
        Xv[c] = Xv[c].astype(str)

    params = dict(cb_params)
    params.setdefault("loss_function", "Logloss")
    params.setdefault("eval_metric", "Logloss")
    params.setdefault("random_seed", seed)
    params.setdefault("verbose", False)
    params.setdefault("iterations", num_boost_round)
    params.setdefault("early_stopping_rounds", early_stopping_rounds)

    model = cb.CatBoostClassifier(**params)
    model.fit(Xt, yt, cat_features=cat_cols, eval_set=(Xv, yv), use_best_model=True)

    p = model.predict_proba(Xv)[:, 1]
    sp = M.score_pair(p, yv)
    sc = M.score(p, yv)
    return dict(fold=fold_name, score=sc, n_va=len(yv),
                n_pitchers=int(valid_df["pitcher_id"].nunique()),
                best_iter=model.get_best_iteration(), **sp), model, p, yv, valid_df


def run_experiment_cb(label, feature_cols, cat_cols, cb_params, use_custom,
                       fold_names=("A", "B", "C"), num_boost_round=3000,
                       early_stopping_rounds=100, seed=0, notes="",
                       use_trend=False, use_failmode=False, use_ballstrike=False,
                       use_failmode_platoon=False, use_n_exposed=False, use_season_trend=False):
    t0 = time.time()
    df = load_train()
    print(f"=== {label} (CatBoost) ===  features={len(feature_cols)}  custom={use_custom}")
    results = {}
    for fn in fold_names:
        res, model, p, yv, valid_df = run_fold_cb(
            df, fn, feature_cols, cat_cols, cb_params, use_custom, num_boost_round,
            early_stopping_rounds, seed, use_trend=use_trend, use_failmode=use_failmode,
            use_ballstrike=use_ballstrike, use_failmode_platoon=use_failmode_platoon,
            use_n_exposed=use_n_exposed, use_season_trend=use_season_trend)
        results[fn] = res
        log_result(label, fn, res, seed=seed, notes=notes)
        print(f"  fold {fn}: score={res['score']:.1f}  pts_res={res['pts_res']:.1f}  "
              f"bias={res['bias']:.5f}  best_iter={res['best_iter']}  n_va={res['n_va']:,}")
    print(f"  elapsed {time.time() - t0:.1f}s")
    return results
