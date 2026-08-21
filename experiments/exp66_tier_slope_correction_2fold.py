"""exp66 -- 티어(asof_pitcher_n) 슬로프 사후보정을 우리 챔피언 파이프라인에 시도.

wonwoo_repro에서 발견한 것(§7.81 이속 미스캘리브레이션의 legal 대체판): 절편은
버리고 순수 기울기만 티어별로 lambda 댐핑해서 적용 + 전역 레벨 1회 재정렬하면,
fit fold != eval fold 교차검증(wonwoo §7.83 원칙) 양방향에서 처음으로 부호가
일관되게 양수였다(BSS oracle 스케일 +2~3).

이 실험은 그 메커니즘을 **우리 자체 LightGBM+CatBoost 챔피언 앙상블**(raw,
ctx보정 이전) 위에서, 새 2-fold 프로토콜(src/folds.py "2023"/"2024")로 재현되는지
확인한다. exp65의 seed 노이즈 게이트가 있어야 최종 판정을 내릴 수 있지만, 이
스크립트는 원 수치(raw delta)만 낸다 -- 판정은 exp65 결과와 같이 본다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.linear_model import LogisticRegression

from src import features as feat
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
LAMBDAS = [0.25, 0.5, 0.75, 1.0]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bss_oracle(y, p):
    r = y.mean()
    bs = np.mean((p - y) ** 2)
    return 1e5 * (1 - bs / (r * (1 - r)))


def shift_to_prior(p, target_mean):
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        m = sigmoid(logit(p) + mid).mean()
        if m < target_mean:
            lo = mid
        else:
            hi = mid
    return sigmoid(logit(p) + (lo + hi) / 2)


def tier_mask(asof_n, lo, hi):
    return (~np.isnan(asof_n)) & (asof_n >= lo) & (asof_n < hi)


def fit_tier_slopes(p_oracle, y, asof_n):
    out = {}
    for name, lo, hi in TIERS:
        m = tier_mask(asof_n, lo, hi)
        x = logit(p_oracle[m]).reshape(-1, 1)
        lr = LogisticRegression(fit_intercept=True, C=1e6, max_iter=1000).fit(x, y[m])
        out[(lo, hi)] = float(lr.coef_[0, 0])
    return out


def apply_damped(p_oracle, asof_n, coefs, lam, target_mean):
    z = logit(p_oracle)
    zc = z.copy()
    for (lo, hi), slope in coefs.items():
        m = tier_mask(asof_n, lo, hi)
        s = 1 + lam * (slope - 1)
        zc[m] = s * z[m]
    return shift_to_prior(sigmoid(zc), target_mean)


if __name__ == "__main__":
    df = load_train()
    feature_cols = feat.feature_list(use_custom=True)

    fold_data = {}
    for fold_name in ("2023", "2024"):
        print(f"=== training fold {fold_name} ===")
        res_lgb, _, p_lgb, y_lgb, valid = run_fold(df, fold_name, feature_cols, CAT_COLS, LGB_PARAMS,
                                                     use_custom=True, num_boost_round=3000, early_stopping_rounds=100)
        res_cb, _, p_cb, y_cb, valid_cb = run_fold_cb(df, fold_name, feature_cols, CAT_COLS, CB_PARAMS,
                                                        use_custom=True, num_boost_round=3000, early_stopping_rounds=100)
        assert (y_lgb == y_cb).all()
        p_ens = 0.5 * p_lgb + 0.5 * p_cb
        yv = y_lgb
        p_oracle = shift_to_prior(p_ens, yv.mean())
        asof_n = valid["asof_pitcher_n"].to_numpy(dtype=np.float64)
        fold_data[fold_name] = dict(p_oracle=p_oracle, y=yv, asof_n=asof_n)
        print(f"  fold {fold_name}: n={len(yv)}  BSS(oracle)={bss_oracle(yv, p_oracle):.1f}")

    print("\n=== 경험 티어별 캘리브레이션 기울기 (우리 앙상블, p_oracle 기준) ===")
    tier_slopes = {}
    for fold_name in ("2023", "2024"):
        d = fold_data[fold_name]
        coefs = fit_tier_slopes(d["p_oracle"], d["y"], d["asof_n"])
        tier_slopes[fold_name] = coefs
        for (lo, hi), slope in coefs.items():
            n = int(tier_mask(d["asof_n"], lo, hi).sum())
            print(f"  fold {fold_name}  tier=[{lo},{hi})  n={n}  slope={slope:.3f}")

    print("\n=== lambda 댐핑 교차검증 (fit fold != eval fold, wonwoo 7.83 원칙) ===")
    print(f"{'lambda':>8s} {'fit2023->eval2024':>20s} {'fit2024->eval2023':>20s}")
    for lam in LAMBDAS:
        d24 = fold_data["2024"]
        base24 = bss_oracle(d24["y"], d24["p_oracle"])
        p24c = apply_damped(d24["p_oracle"], d24["asof_n"], tier_slopes["2023"], lam, d24["y"].mean())
        delta1 = bss_oracle(d24["y"], p24c) - base24

        d23 = fold_data["2023"]
        base23 = bss_oracle(d23["y"], d23["p_oracle"])
        p23c = apply_damped(d23["p_oracle"], d23["asof_n"], tier_slopes["2024"], lam, d23["y"].mean())
        delta2 = bss_oracle(d23["y"], p23c) - base23

        print(f"{lam:8.2f} {delta1:+20.1f} {delta2:+20.1f}")
