"""jointbeta 후속 — 결합 적합이 **뭉쳐놓은 것**을 축별로 분리한다.

phase 1(`jointbeta.py`)이 남긴 두 가지
    ① 교차 fold가 정면으로 엇갈렸다 (적합2023→2024 −25.4 / 적합2024→2023 +19.8).
       그런데 이건 4축을 **한 덩어리로** 옮긴 결과다. 축마다 사정이 다르면
       하나의 이득과 다른 하나의 손실이 상쇄돼 아무것도 안 보인다.
    ② 계수를 보면 실제로 다르다:
           count4        f2023 1.68 / f2024 1.79   비 1.06   ← 두 fold가 동의. 배포 1.25
           gap_success   f2023 0.065 / f2024 0.040  비 1.66   ← 갈린다.     배포 0.030
       ①의 −25.4는 gap_success를, +19.8은 count4를 반영한 것일 수 있다.

여기서 재는 것
    D   변형 D = β4 + c (절편 **고정**). phase 1의 B/C가 A보다 나빴는데
        B와 C가 거의 같았다 → 손해는 c가 아니라 **절편**에서 왔다는 뜻이다.
        c만 풀면 어떻게 되는지 분리한다. (§7.19는 보정축이 하나도 없던 시절
        측정이고, phase 1에서 c가 0.997/1.039로 **처음 부호가 맞았다**.)

    S   축별 1차원 스윕. 나머지 3축은 배포값에 고정하고 한 축만 움직인다.
        저장소의 채택 규칙을 그대로 적용한다:
          조건 ②  양 fold 이득 양수 구간(=고원)의 폭
          조건 ③  두 fold 이득의 크기비 3배 이내
          밴드 규칙(§7.37)  ±50% 오차 밴드 [β/1.5, β·1.5]가 통째로 고원 안
        → 규칙이 뱉는 β를 배포값과 나란히 찍는다. **점수로 고르지 않는다.**

실행:  .venv/Scripts/python.exe -u src/jointbeta2.py 2>&1 | tee experiments/jointbeta2.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import jointbeta as J
from config import (ASOF_AXES, CTX_AXES, MLP_ALPHA, TARGET_COL, TRAIN_CSV)
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = J.FOLDS
# 축별 스윕 격자. 배포값을 반드시 포함하고 양쪽으로 충분히 넓게 간다.
GRID = {
    "hand":         [0.0, 0.5, 1.0, 1.25, 1.5, 1.875, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0],
    "count4":       [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0],
    "gap_success":  [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.13],
    "gap_bsuccess": [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
}


def band_pick(grid, d23, d24):
    """저장소 채택 규칙을 격자에 적용한다 → (고원, 폭, 밴드허용구간, 추천 β).

    고원 = 양 fold 이득이 **둘 다 양수**인 연속 구간 (0은 정의상 이득 0이라 제외).
    밴드 = ±50% 오차에 대비해 [β/1.5, β·1.5]가 통째로 고원 안에 드는 β.
    추천 = 허용 구간 안에서 f2024 이득이 최대인 값의 **한 칸 아래**(§7.60 관례).
    """
    ok = [g for g, a, b in zip(grid, d23, d24) if g > 0 and a > 0 and b > 0]
    if not ok:
        return None, 0.0, None, None
    lo, hi = min(ok), max(ok)
    allow = [g for g in ok if g / 1.5 >= lo and g * 1.5 <= hi]
    if not allow:
        return (lo, hi), hi / max(lo, 1e-12), None, None
    sc = {g: d24[grid.index(g)] for g in allow}
    best = max(allow, key=lambda g: sc[g])
    i = allow.index(best)
    rec = allow[i - 1] if i > 0 else best
    return (lo, hi), hi / max(lo, 1e-12), (min(allow), max(allow)), rec


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = J.foldcache.load(FOLDS, verbose=False)

    D = {}
    for f in FOLDS:
        X, names, dep, va = J.axis_matrix(df, s, f)
        y = cached[f][2]
        z0 = (1 - MLP_ALPHA) * cached[f][0] + MLP_ALPHA * cached[f][1]
        assert np.array_equal(y, va[TARGET_COL].to_numpy(dtype=np.float64))
        D[f] = dict(X=X, Xs=X / X.std(axis=0), sd=X.std(axis=0), y=y, z0=z0)
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    base = {f: bss(D[f]["y"], shift_to_prior(
        1 / (1 + np.exp(-(D[f]["z0"] + D[f]["X"] @ dep))), D[f]["y"].mean()))
        for f in FOLDS}
    print(f"축 {names}   배포 β {dict(zip(names, dep))}")
    print(f"배포 기준선  f2023 {base[2023]:.1f}   f2024 {base[2024]:.1f}")

    # ── D. c만 푼다 (절편 고정) ─────────────────────────────────────────
    print(f"\n{'='*78}\n■ 변형 D — β4 + c (절편 고정 0)\n{'='*78}")
    print("  phase 1: A(β4만) −25.4/+19.8   B(+절편) −49.6/+12.2   C(+절편+c) −49.7/+9.9")
    print("  B와 C가 거의 같았다 → 손해는 절편에서 왔다는 가설. c만 풀어 확인한다.\n")
    free_d = [True, False, True, True, True, True]
    print(f"  {'λ':>7} │ {'적합2023→평가2024':>18} {'적합2024→평가2023':>18} │"
          f" {'c(f2023)':>9} {'c(f2024)':>9}")
    for lam in J.LAMBDAS:
        th = {f: J.fit(D[f]["Xs"], D[f]["z0"], D[f]["y"], free_d, lam, "brier")
              for f in FOLDS}
        x23 = J.score(th[2023], D[2024]["Xs"], D[2024]["z0"], D[2024]["y"])
        x24 = J.score(th[2024], D[2023]["Xs"], D[2023]["z0"], D[2023]["y"])
        print(f"  {lam:>7.5f} │ {x23:>9.1f} {x23-base[2024]:>+8.1f}"
              f" {x24:>9.1f} {x24-base[2023]:>+8.1f} │"
              f" {th[2023][0]:>9.4f} {th[2024][0]:>9.4f}")

    # ── S. 축별 1차원 스윕 ──────────────────────────────────────────────
    print(f"\n{'='*78}\n■ 축별 1차원 스윕 — 나머지 3축은 배포값 고정\n{'='*78}")
    rec = {}
    for i, nm in enumerate(names):
        grid = GRID[nm]
        d23, d24 = [], []
        print(f"\n── {nm}  (배포 {dep[i]}) " + "─" * 40)
        print(f"  {'β':>8} │ {'f2023':>9} {'Δ':>8} │ {'f2024':>9} {'Δ':>8} │"
              f" {'크기비':>7} {'부호':>5}")
        for g in grid:
            b = dep.copy()
            b[i] = g
            sc = {}
            for f in FOLDS:
                z = D[f]["z0"] + D[f]["X"] @ b
                sc[f] = bss(D[f]["y"], shift_to_prior(
                    1 / (1 + np.exp(-z)), D[f]["y"].mean()))
            a, c = sc[2023] - base[2023], sc[2024] - base[2024]
            d23.append(a)
            d24.append(c)
            r = (max(abs(a), abs(c)) / max(min(abs(a), abs(c)), 1e-9)
                 if a * c > 0 else float("nan"))
            mark = "  ←배포" if abs(g - dep[i]) < 1e-9 else ""
            print(f"  {g:>8.4f} │ {sc[2023]:>9.1f} {a:>+8.1f} │"
                  f" {sc[2024]:>9.1f} {c:>+8.1f} │ {r:>7.2f}"
                  f" {'✓' if a*c > 0 else '✗':>5}{mark}")
        pl, w, allow, r_ = band_pick(grid, d23, d24)
        rec[nm] = r_
        print(f"    고원 {pl}  폭 {w:.1f}배   밴드허용 {allow}   → 규칙 추천 β = {r_}"
              f"   (배포 {dep[i]})")

    # ── 규칙 추천을 전부 적용하면 ───────────────────────────────────────
    print(f"\n{'='*78}\n■ 규칙 추천 β를 동시 적용\n{'='*78}")
    b = np.array([rec[n] if rec[n] is not None else dep[i]
                  for i, n in enumerate(names)], dtype=float)
    print(f"  배포 {dict(zip(names, dep))}\n  추천 {dict(zip(names, b))}")
    for f in FOLDS:
        z = D[f]["z0"] + D[f]["X"] @ b
        sc = bss(D[f]["y"], shift_to_prior(1 / (1 + np.exp(-z)), D[f]["y"].mean()))
        print(f"  fold {f}: {sc:.1f}  (배포 {base[f]:.1f}, Δ {sc-base[f]:+.1f})")

    # 축을 하나씩만 바꿔서 얹었을 때의 증분도 같이 (동시 적용과 갈리는지 확인)
    print("\n  축 하나씩만 추천값으로 (나머지 배포):")
    for i, nm in enumerate(names):
        if rec[nm] is None or abs(rec[nm] - dep[i]) < 1e-9:
            print(f"    {nm:<16} 변화 없음")
            continue
        bb = dep.copy()
        bb[i] = rec[nm]
        out = []
        for f in FOLDS:
            z = D[f]["z0"] + D[f]["X"] @ bb
            out.append(bss(D[f]["y"], shift_to_prior(
                1 / (1 + np.exp(-z)), D[f]["y"].mean())) - base[f])
        print(f"    {nm:<16} {dep[i]:>7.4f} → {rec[nm]:<8.4f}"
              f" f2023 {out[0]:+8.1f}   f2024 {out[1]:+8.1f}")


if __name__ == "__main__":
    main()
