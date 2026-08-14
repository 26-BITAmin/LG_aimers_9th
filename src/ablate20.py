"""20차 실험 — **로짓 온도 c.** 앙상블이 만든 수축을 되돌린다.

동기. `verify_pipeline`의 fold2024 캘리브레이션 표에서 예측 십분위 폭이 0.1495인데
실제 폭은 0.1602였다 (비 1.072). 순수한 레벨 이동이면 폭은 그대로여야 하므로
**예측이 실제보다 좁다.** 게다가 예측에 노이즈가 있으면 이 회귀는 기울기를 1 쪽으로
끌어내리므로 진짜 압축은 1.072보다 심하다.

메커니즘이 있다. 로짓 공간에서 상관 0.95인 두 예측을 평균하면 분산이 각 성분보다
줄어든다 (실측: blend 0.1842 < lgb 0.1854, mlp 0.1881). 앙상블은 순위를 지키면서
예측을 중앙으로 수축시키고, Brier는 proper scoring rule이라 그 수축에 벌점을 매긴다.

    z' = c * z + delta        c > 1 이면 수축을 되돌린다

⚠️ 이건 **정보를 더하는 변경이 아니다.** 지금까지 LB에서 뒤집힌 것은 전부
   "2019~2024의 관계가 2025에서도 같다"에 기댄 변경이었다 (Trackman, 피처 삭감).
   c는 앙상블이 만든 왜곡을 푸는 것이라 그 부류가 아니다 — α와 같은 부류다.

⚠️ **레벨과 기울기를 분리해서 잰다.** c를 바꾸면 예측 평균도 바뀌므로 c마다
   레벨을 다시 맞춰야 한다. 안 그러면 기울기 효과와 레벨 오차가 섞인다.
   여기서는 oracle shift(예측 평균 = 실제 평균)로 레벨을 지우고 기울기만 본다.

⚠️ c는 **train fold에서 고른다.** LB를 보고 정하지 않는다.

실행:  .venv/Scripts/python.exe -u src/ablate20.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
from scipy.optimize import brentq

import foldcache
from config import MLP_ALPHA

FOLDS = [2023, 2024]
CS = (0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30)


def oracle_shift(z, y):
    """예측 평균이 실제 평균과 같아지도록 로짓에 상수를 더한다 (레벨 제거)."""
    t = y.mean()
    f = lambda d: (1.0 / (1.0 + np.exp(-(z + d)))).mean() - t
    lo, hi = -5.0, 5.0
    return brentq(f, lo, hi, xtol=1e-12)


def bss(z, y):
    d = oracle_shift(z, y)
    p = 1.0 / (1.0 + np.exp(-(z + d)))
    return (1 - ((p - y) ** 2).mean() / (y.mean() * (1 - y.mean()))) * 1e5


def se_bss(zA, zB, y):
    """두 설정의 쌍체 차이에 대한 표준오차 (BSS 단위). 크기만 보고 판단하지 않기 위해."""
    pA = 1.0 / (1.0 + np.exp(-(zA + oracle_shift(zA, y))))
    pB = 1.0 / (1.0 + np.exp(-(zB + oracle_shift(zB, y))))
    di = (pA - y) ** 2 - (pB - y) ** 2
    return di.std(ddof=1) / np.sqrt(len(di)) / (y.mean() * (1 - y.mean())) * 1e5


def main():
    data = foldcache.load(FOLDS)
    Z, Y = {}, {}
    for f in FOLDS:
        z_lgb, z_mlp, y, _ = data[f]
        Z[f] = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
        Y[f] = y
        print(f"fold {f}: n={len(y):,}  std(z_lgb)={z_lgb.std():.4f} "
              f" std(z_mlp)={z_mlp.std():.4f}  std(blend)={Z[f].std():.4f}"
              f"  corr={np.corrcoef(z_lgb, z_mlp)[0,1]:.4f}")

    base = np.array([bss(Z[f] - Z[f].mean(), Y[f]) for f in FOLDS])
    print(f"\n기준 (c=1.00): " + "  ".join(f"{f}={b:.1f}" for f, b in zip(FOLDS, base)))

    print(f"\n{'c':>6s} " + " ".join(f"{f:>9d}" for f in FOLDS)
          + f" {'평균Δ':>8s} {'크기비':>7s} {'SE':>6s}")
    print("-" * 52)
    rows = []
    for c in CS:
        v, ses = [], []
        for f in FOLDS:
            zc = c * (Z[f] - Z[f].mean())
            v.append(bss(zc, Y[f]))
            ses.append(se_bss(zc, Z[f] - Z[f].mean(), Y[f]))
        v = np.array(v)
        d = v - base
        same = bool(np.all(d > 0) or np.all(d < 0))
        sp = np.abs(d).max() / max(np.abs(d).min(), 1e-9) if same else np.inf
        ok = same and d.mean() > 0 and sp <= 3
        rows.append((c, v, d.mean(), ok))
        print(f"{c:>6.2f} " + " ".join(f"{x:>9.1f}" for x in v)
              + f" {d.mean():>+8.2f} " + (f"{sp:>7.1f}" if same else f"{'inf':>7s}")
              + f" {np.mean(ses):>6.2f}" + ("  <<< 채택 조건 통과" if ok else ""))

    print("\n  fold별 최적 c (표에서):")
    for i, f in enumerate(FOLDS):
        bi = int(np.argmax([r[1][i] for r in rows]))
        print(f"    {f}: c={rows[bi][0]:.2f}  (BSS {rows[bi][1][i]:.1f})")
    print("\n  ※ 두 fold의 최적 c가 크게 다르면 채택하지 않는다 —")
    print("     c가 데이터마다 흔들린다는 뜻이고, 2025에서 어디로 갈지 알 수 없다.")


if __name__ == "__main__":
    main()
