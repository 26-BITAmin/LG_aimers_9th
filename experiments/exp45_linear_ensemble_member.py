"""exp45 — 완전히 다른 함수족(선형)을 세 번째 앙상블 멤버 후보로 진단.

wonwoo 브랜치(§7.20, FM 실험)의 핵심 진단: "k=0 선형이 모든 FM보다 낫다" — 쌍
상호작용 항은 기여가 없고, 오히려 고카디널리티 ID 잠재벡터가 과적합 통로가 된다.
MLP도 1차 시도는 참패(-4386)했다가 극단적 정규화 후에야 겨우 살아났다. 두 경우
모두 "암기 용량이 있으면 이 노이즈 지배적 문제에서 죽는다"는 패턴.

로지스틱 회귀는 애초에 암기 용량이 없어 그 실패 모드 자체가 없다. 트리(LightGBM/
CatBoost)와 완전히 다른 귀납 편향(축 정렬 계단함수 vs 선형 결합)이라 로짓 상관이
낮을 가능성 있고, wonwoo의 앙상블 법칙("핸디캡 대비 탈상관 보너스 비율이 중요하지
핸디캡 자체가 작아야 하는 건 아니다")을 따르면 단독 성능이 약해도 블렌드 이득이
있을 수 있다.

이 실험은 블렌드를 바로 적용하지 않고 **진단만** 한다: 단독 스코어 + exp32 앙상블
(LightGBM+CatBoost 50/50)과의 로짓 상관계수. 상관이 매우 높으면(>0.95) 탈상관
보너스가 거의 없어 블렌드 무의미 — wonwoo의 CatBoost 앙상블(corr 0.985, 보너스
겨우 +3.5, 전부 마이너스) 사례와 같은 결론이 나올 수 있다.

인코딩: sklearn은 결측/네이티브 범주형을 못 다루므로 이 실험 내부에서만 로컬
전처리(원-핫 인코딩, 결측 median 대치 + is_missing 플래그, 표준화)를 한다 —
features.py나 검증 하니스는 안 건드림.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import features as feat
from src import folds as F
from src import metrics as M
from src.train_utils import load_train, run_fold
from src.train_utils_cb import run_fold_cb

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

LGB_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
)
CB_PARAMS = dict(learning_rate=0.03, depth=8, l2_leaf_reg=5.0)


def encode_linear(fit_df, valid_df, feature_cols, cat_cols):
    """train에서 fit한 원-핫/median/표준화를 valid에 적용. 결측은 median + is_missing 플래그."""
    num_cols = [c for c in feature_cols if c not in cat_cols]

    Xt_num = fit_df[num_cols].astype(float).copy()
    Xv_num = valid_df[num_cols].astype(float).copy()
    medians = Xt_num.median()
    for c in num_cols:
        na_t = Xt_num[c].isna()
        na_v = Xv_num[c].isna()
        if na_t.any() or na_v.any():
            Xt_num[f"{c}_isna"] = na_t.astype(float)
            Xv_num[f"{c}_isna"] = na_v.astype(float)
        Xt_num[c] = Xt_num[c].fillna(medians[c])
        Xv_num[c] = Xv_num[c].fillna(medians[c])

    scaler = StandardScaler()
    Xt_num_scaled = pd.DataFrame(scaler.fit_transform(Xt_num), columns=Xt_num.columns, index=Xt_num.index)
    Xv_num_scaled = pd.DataFrame(scaler.transform(Xv_num), columns=Xv_num.columns, index=Xv_num.index)

    Xt_cat = pd.get_dummies(fit_df[cat_cols].astype(str), prefix=cat_cols)
    Xv_cat = pd.get_dummies(valid_df[cat_cols].astype(str), prefix=cat_cols)
    Xv_cat = Xv_cat.reindex(columns=Xt_cat.columns, fill_value=0)

    Xt = pd.concat([Xt_num_scaled.reset_index(drop=True), Xt_cat.reset_index(drop=True)], axis=1)
    Xv = pd.concat([Xv_num_scaled.reset_index(drop=True), Xv_cat.reset_index(drop=True)], axis=1)
    return Xt, Xv


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


if __name__ == "__main__":
    df = load_train()
    feature_cols = feat.feature_list(use_custom=True)

    for fold_name in ("A", "B", "C"):
        print(f"=== fold {fold_name} ===")
        train_mask, valid_mask = F.split(df, fold_name)
        fit_df, valid_df = feat.build_fold_frames(df, train_mask, valid_mask, use_custom=True)
        yv = valid_df[feat.TARGET].to_numpy()

        # --- 선형 모델 ---
        Xt_lin, Xv_lin = encode_linear(fit_df, valid_df, feature_cols, CAT_COLS)
        yt = fit_df[feat.TARGET].to_numpy()
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(Xt_lin, yt)
        p_lin = clf.predict_proba(Xv_lin)[:, 1]
        sc_lin = M.score(p_lin, yv)
        sp_lin = M.score_pair(p_lin, yv)
        print(f"  linear: score={sc_lin:.1f}  bias={sp_lin['bias']:.5f}  n_features={Xt_lin.shape[1]}")

        # --- 기존 앙상블(LightGBM+CatBoost 50/50, exp32와 동일 설정) ---
        res_lgb, _, p_lgb, y_lgb, _ = run_fold(df, fold_name, feature_cols, CAT_COLS, LGB_PARAMS,
                                                use_custom=True, num_boost_round=3000, early_stopping_rounds=100)
        res_cb, _, p_cb, y_cb, _ = run_fold_cb(df, fold_name, feature_cols, CAT_COLS, CB_PARAMS,
                                                use_custom=True, num_boost_round=3000, early_stopping_rounds=100)
        assert (y_lgb == yv).all() and (y_cb == yv).all()
        p_ens = 0.5 * p_lgb + 0.5 * p_cb
        sc_ens = M.score(p_ens, yv)
        print(f"  ensemble(exp32): score={sc_ens:.1f}")

        corr = np.corrcoef(logit(p_lin), logit(p_ens))[0, 1]
        print(f"  logit correlation(linear, ensemble) = {corr:.4f}")

        # --- 참고: 몇 가지 고정 블렌드 가중치로 이득/손해 미리보기 (채택은 별도 실험에서) ---
        for a in (0.05, 0.10, 0.20, 0.30):
            p_blend = (1 - a) * p_ens + a * p_lin
            sc_blend = M.score(p_blend, yv)
            print(f"    blend a={a:.2f}: score={sc_blend:.1f}  (Δ vs ensemble={sc_blend - sc_ens:+.1f})")
