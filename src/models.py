"""모델 패밀리별 학습·예측 래퍼 — 앙상블 다양성 실험용.

왜 모델 다양성인가
    Trackman 실패의 원인은 **피처와 타깃의 관계가 2025에서 달라진 것**이었다.
    같은 피처 위에서 다른 학습기를 섞는 것은 새로운 분포 의존성을 만들지 않고
    주로 분산을 줄이므로, 그 이득은 전이될 가능성이 높다.

    특히 CatBoost는 단순한 다양성 이상일 수 있다. `pitcher_id`/`batter_id`를
    native categorical로 처리하며 ordered target statistics를 쓴다 —
    정수를 그대로 트리에 넣는 현재 방식과 근본적으로 다른 처리다.

모든 래퍼는 **로짓(raw score)** 을 반환한다. 앙상블은 로짓 공간에서 평균한다
(확률 평균은 예측을 0.5 쪽으로 수축시켜 캘리브레이션을 흐트러뜨린다).
"""
import numpy as np

# 고카디널리티 ID — CatBoost에서 native categorical로 넘긴다
CAT_ID_COLS = ["pitcher_id", "batter_id"]


def fit_lgbm(Xtr, ytr, w, seed, params, rounds):
    import lightgbm as lgb
    p = dict(params, seed=seed, bagging_seed=seed,
             feature_fraction_seed=seed, data_random_seed=seed)
    bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=rounds)
    return lambda X: bst.predict(X, raw_score=True)


def fit_xgb(Xtr, ytr, w, seed, params=None, rounds=600):
    import xgboost as xgb
    p = dict(objective="binary:logistic", eta=0.02, max_depth=6,
             min_child_weight=200, subsample=0.8, colsample_bytree=0.8,
             reg_lambda=30.0, max_bin=128, tree_method="hist",
             nthread=6, seed=seed, **(params or {}))
    d = xgb.DMatrix(Xtr, label=ytr, weight=w)
    bst = xgb.train(p, d, num_boost_round=rounds)
    return lambda X: bst.predict(xgb.DMatrix(X), output_margin=True)


def fit_cat(Xtr, ytr, w, seed, params=None, rounds=800, cat_cols=CAT_ID_COLS):
    """ID를 native categorical로 처리한다 — 이게 LightGBM과의 핵심 차이."""
    from catboost import CatBoostClassifier, Pool
    cats = [c for c in cat_cols if c in Xtr.columns]
    Xtr = Xtr.copy()
    for c in cats:
        Xtr[c] = Xtr[c].astype(np.int64).astype(str)
    idx = [Xtr.columns.get_loc(c) for c in cats]

    m = CatBoostClassifier(
        iterations=rounds, learning_rate=0.03, depth=6, l2_leaf_reg=30.0,
        min_data_in_leaf=2000, border_count=128, random_seed=seed,
        loss_function="Logloss", verbose=0, thread_count=6,
        **(params or {}))
    m.fit(Pool(Xtr, ytr, weight=w, cat_features=idx))

    def predict(X):
        X = X.copy()
        for c in cats:
            X[c] = X[c].astype(np.int64).astype(str)
        return m.predict(Pool(X, cat_features=idx), prediction_type="RawFormulaVal")

    return predict


FAMILIES = {"lgbm": fit_lgbm, "xgb": fit_xgb, "cat": fit_cat}
