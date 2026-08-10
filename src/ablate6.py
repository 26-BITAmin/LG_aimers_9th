"""7차 실험 — sharpening (로짓 스케일링).

동기:
    완벽히 캘리브레이션된 모델의 Brier는  BS = θ(1-θ) - Var(p)  이다.
    따라서 달성 가능한 최대 점수는  BSS_max = 100000 * Var(p) / θ(1-θ).
    우리 예측은 std ≈ 0.042 → Var ≈ 0.00176 → 상한 ≈ 706.
    그런데 실측은 2024에서 748, LB(2025)에서 933이 나왔다. **상한을 넘었다.**
    → 모델이 과소확신이다. 실제 조건부 확률이 예측보다 넓게 퍼져 있다.

방법:
    logit(p') = alpha * (logit(p) - z_bar) + z_bar,  이후 prior shift.
    alpha > 1이면 예측을 넓히고(sharpening), < 1이면 좁힌다(shrinkage).
    alpha와 z_bar는 스칼라이므로 학습 시점에 고정할 수 있다 → 행 독립성 유지.

정직한 프로토콜:
    alpha를 fold1(valid 2023)에서 고르고 **fold2(valid 2024)에 적용**해서 전이되는지 본다.
    fold2에서 고른 alpha를 fold2에 쓰면 당연히 좋아 보이므로 그건 상한 참고용으로만 본다.

실행:  .venv/Scripts/python.exe -u src/ablate6.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import lightgbm as lgb
import numpy as np

from config import (DROP_FEATURES, FINAL_LGB_PARAMS, N_ROUNDS, RECENCY_HALF_LIFE,
                    SEED_ENSEMBLE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL,
                    TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import _logit, _sigmoid, bss, shift_to_prior

ALPHAS = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]


def build_X(df, art):
    X = build_features(df, art)
    if USE_PLAYER_IDS:
        for c in ("pitcher_id", "batter_id"):
            X[c] = df[c].to_numpy(dtype=np.float32)
    return X.drop(columns=list(DROP_FEATURES))


def fit_predict(df, train_seasons, valid_season, seeds=SEED_ENSEMBLE):
    """최종 설정 그대로 학습해 valid 시즌의 로짓(raw score 평균)을 반환."""
    tr = df[df.season.isin(train_seasons)]
    va = df[df.season == valid_season]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=max(train_seasons))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER, k_batter=SMOOTH_K_BATTER)
    Xtr, Xva = build_X(tr, art), build_X(va, art)
    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)

    z = np.zeros(len(Xva))
    for s in seeds:
        params = dict(FINAL_LGB_PARAMS, seed=s, bagging_seed=s,
                      feature_fraction_seed=s, data_random_seed=s)
        bst = lgb.train(params, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=N_ROUNDS)
        z += bst.predict(Xva, raw_score=True)
    return z / len(seeds), yva


def sweep(z, y, label):
    """alpha별 BSS(oracle shift 적용). 레벨 성분을 뺐으므로 순수 sharpening 효과다."""
    zbar = z.mean()
    V = y.mean() * (1 - y.mean())
    rows = []
    for a in ALPHAS:
        p = _sigmoid(a * (z - zbar) + zbar)
        p = shift_to_prior(p, y.mean())          # 레벨은 오라클로 고정
        rows.append((a, bss(y, p), float(p.std())))
    best = max(rows, key=lambda r: r[1])
    print(f"\n[{label}]  실제양성률={y.mean():.4f}  raw pred std={_sigmoid(z).std():.4f}")
    print(f"  캘리브레이션 상한(=100000*Var/V) = {100000*_sigmoid(z).var()/V:.1f}"
          f"  ← 이걸 넘으면 과소확신이다")
    print(f"  {'alpha':>6s} {'BSS':>9s} {'pred_std':>9s}")
    for a, s, sd in rows:
        mark = "  <<<" if a == best[0] else ""
        print(f"  {a:>6.2f} {s:>9.1f} {sd:>9.4f}{mark}")
    return best[0], dict((a, s) for a, s, _ in rows)


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)

    print("fold1 학습 (train 2019-2022 → valid 2023) ...")
    z1, y1 = fit_predict(df, [2019, 2020, 2021, 2022], 2023)
    print(f"  ({time.time()-t0:.0f}s)")
    a1, s1 = sweep(z1, y1, "fold1 valid=2023")

    print(f"\nfold2 학습 (train 2019-2023 → valid 2024) ...  ({time.time()-t0:.0f}s)")
    z2, y2 = fit_predict(df, [2019, 2020, 2021, 2022, 2023], 2024)
    print(f"  ({time.time()-t0:.0f}s)")
    a2, s2 = sweep(z2, y2, "fold2 valid=2024")

    print("\n=== 전이 검정 (이게 채택 근거다) ===")
    print(f"  fold1에서 고른 alpha = {a1}")
    print(f"    → fold2 적용:  BSS {s2[1.0]:.1f} (alpha=1) → {s2[a1]:.1f}  "
          f"({s2[a1]-s2[1.0]:+.1f})")
    print(f"  fold2 자체 최적 alpha = {a2} → {s2[a2]:.1f} "
          f"({s2[a2]-s2[1.0]:+.1f})  [상한 참고용, 채택 근거 아님]")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
