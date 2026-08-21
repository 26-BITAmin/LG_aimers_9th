"""exp64 -- 진단: 경험 티어(asof_pitcher_n)별로 챔피언 앙상블의 캘리브레이션
기울기가 정말로 갈라지는가 (wonwoo MODEL.md 7.81의 "이속(二速) 미스캘리브레이션"이
우리 파이프라인에도 전이되는지 확인).

wonwoo 쪽 발견: 신인/베테랑의 예측-실제 관계 기울기가 반대 방향으로 어긋나 있어서
전역 온도(T) 보정 하나로는 거의 안 잡히고(+1.5), 티어를 나눠야 큰 채널(오라클
+47~57)이 보인다. 단, 그 오라클 자체는 검증 fold의 여러 행을 묶어야 계산되는
값이라 행 독립 규정과 정면 충돌 -- 구현 가능한 대체물은 전부 실패해서 채널 자체는
wonwoo 쪽에서 종결됐다(§7.82).

이 스크립트는 그 오라클을 재현하려는 게 아니다. **우리 모델이 애초에 같은 증상을
보이는지**만 확인하는 순수 진단이다 -- "전역으로는 안 보이던 게 티어로 쪼개면
보이는가" 검사. asof_pitcher_n은 이미 행에 있는 strictly-prior 값이라 이 진단
자체는 행 독립을 어기지 않는다 (오라클처럼 fold 내 다른 행을 참조하지 않음).

방법: 챔피언 51피처 + LightGBM/CatBoost 50/50 앙상블(단일 seed=0, exp57~63과 동일
구성)을 fold A/B/C에서 학습 -> 검증 예측 p_ens를 얻는다. asof_pitcher_n으로 3개
티어(신인<200 / 저경력 200-1k / 베테랑 1k+, 지난 대화의 raw-rate 진단과 동일 경계)로
나눠, 각 티어 안에서 logit(p_ens) -> y 1변수 로지스틱 회귀의 기울기(calibration
slope, 1이면 완벽 캘리브레이션)를 잰다. 세 fold 걸쳐 티어 간 기울기 격차가
방향 일관되게 나오는지가 핵심 판정 기준.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.linear_model import LogisticRegression

from src import features as feat
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

TIERS = [("rookie(<200)", 0, 200), ("low(200-1k)", 200, 1000), ("veteran(1k+)", 1000, np.inf)]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calib_slope(p, y):
    """logit(p) -> y 1변수 로지스틱 회귀 기울기. 1=완벽 캘리브레이션,
    <1=모델이 과신(실제보다 더 크게 흔듦), >1=모델이 과소반응."""
    x = logit(p).reshape(-1, 1)
    lr = LogisticRegression(fit_intercept=True, C=1e6, max_iter=1000)
    lr.fit(x, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


if __name__ == "__main__":
    df = load_train()
    feature_cols = feat.feature_list(use_custom=True)

    print(f"{'fold':4s} {'tier':16s} {'n':>7s} {'p_mean':>8s} {'y_mean':>8s} {'slope':>8s} {'intercept':>10s}")
    summary = {}
    for fold_name in ("A", "B", "C"):
        res_lgb, _, p_lgb, y_lgb, valid = run_fold(df, fold_name, feature_cols, CAT_COLS, LGB_PARAMS,
                                                     use_custom=True, num_boost_round=3000, early_stopping_rounds=100)
        res_cb, _, p_cb, y_cb, valid_cb = run_fold_cb(df, fold_name, feature_cols, CAT_COLS, CB_PARAMS,
                                                        use_custom=True, num_boost_round=3000, early_stopping_rounds=100)
        assert (y_lgb == y_cb).all()
        p_ens = 0.5 * p_lgb + 0.5 * p_cb
        yv = y_lgb
        overall_sc = M.score(p_ens, yv)

        asof_n = valid["asof_pitcher_n"].to_numpy()
        valid_asof = ~np.isnan(asof_n)

        for tier_name, lo, hi in TIERS:
            m = valid_asof & (asof_n >= lo) & (asof_n < hi)
            if m.sum() < 200:
                continue
            slope, icpt = calib_slope(p_ens[m], yv[m])
            print(f"{fold_name:4s} {tier_name:16s} {m.sum():7d} {p_ens[m].mean():8.4f} "
                  f"{yv[m].mean():8.4f} {slope:8.3f} {icpt:10.3f}")
            summary.setdefault(tier_name, {})[fold_name] = slope
        print(f"  (overall ensemble score this fold: {overall_sc:.1f})")

    print("\n=== 요약: 티어별 기울기, fold 간 (기울기-1) 부호 일관성 ===")
    for tier_name, _, _ in TIERS:
        vals = summary.get(tier_name, {})
        devs = {k: v - 1.0 for k, v in vals.items()}
        signs = set(np.sign(v) for v in devs.values())
        consistent = len(signs) <= 1
        print(f"  {tier_name:16s} slopes={vals}  (slope-1) signs={devs}  "
              f"cross-fold sign consistent={consistent}")
