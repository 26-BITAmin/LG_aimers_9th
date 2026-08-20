"""jointbeta 수정 — **전방향 쌍만** 쓴다 (사용자 지적).

무엇이 틀렸나
    phase 1은 (적합2023→평가2024)와 (적합2024→평가2023)을 나란히 놓고
    "부호가 엇갈린다"고 읽었다. **틀린 프레이밍이다.**

    배포는 `≤2024로 얼린 것 → 2025 예측`이다. 시간이 앞으로만 간다.
    적합2024→평가2023은 미래로 과거를 맞히는 것이라 **전이 추정치가 아니다.**
    게다가 fold2023은 2022→2023 구조적 단절이 있는 fold라(블렌드만 641.9 vs
    2024의 837.6) 그 방향의 +19.8은 "정상 시즌 β를 이상한 시즌에 갖다 댔더니
    배포값보다 나았다"는 얘기다. 방향성 결론에 못 쓴다.

    → 유효한 관측은 **적합2023→평가2024 하나뿐**이었다. 그리고 −25.4다.

여기서 고치는 것
    fold를 2021·2022까지 늘려 전방향 쌍을 3개로 만든다.
        2021 → 2022     2022 → 2023     2023 → 2024
    각 쌍에서 "적합 fold의 결합 MLE β"를 다음 시즌에 적용해 배포 β와 비교한다.
    ⚠️ 배포 β는 2023·2024를 **보고** 골랐으므로 뒤 두 쌍에서 배포 쪽이 유리하다.
       첫 쌍(2021→2022)은 배포 β가 보지 않은 fold라 그나마 공정하다.

    같이 찍는 것: 각 fold의 **자기 최적 β**. 이게 시즌을 따라 어떻게 움직이는지가
    "β가 드리프트하는가"에 대한 직접 관측이다 (전이 배율의 근원).

⏱ fold 2021·2022는 캐시가 없어 신규 학습이 필요하다 (fold당 ~25분).

실행:  .venv/Scripts/python.exe -u src/jointbeta3.py 2>&1 | tee experiments/jointbeta3.log
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
from config import ASOF_AXES, CTX_AXES, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

ALL = [2021, 2022, 2023, 2024]
PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024)]     # 전방향만
LAMS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
FREE_A = [False, False, True, True, True, True]        # β4만 (절편은 전이를 망친다)


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = J.foldcache.load(ALL, verbose=False)
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])

    D = {}
    for f in ALL:
        X, names, _, va = J.axis_matrix(df, s, f)
        y = cached[f][2]
        z0 = (1 - MLP_ALPHA) * cached[f][0] + MLP_ALPHA * cached[f][1]
        assert np.array_equal(y, va[TARGET_COL].to_numpy(dtype=np.float64))
        sd = X.std(axis=0)
        raw = bss(y, shift_to_prior(1 / (1 + np.exp(-z0)), y.mean()))
        base = bss(y, shift_to_prior(1 / (1 + np.exp(-(z0 + X @ dep))), y.mean()))
        D[f] = dict(X=X, Xs=X / sd, sd=sd, y=y, z0=z0, base=base)
        print(f"fold {f}: n={len(y):,}  ȳ={y.mean():.4f}  블렌드만 {raw:.1f}"
              f"  +배포4축 {base:.1f}  (기여 {base-raw:+.1f})")
    print(f"\n축 {names}   배포 β {dict(zip(names, dep))}")

    # ── 각 fold의 자기 최적 β — 드리프트 직접 관측 ──────────────────────
    print(f"\n{'='*78}\n■ fold별 결합 MLE β (변형 A, λ=0) — 시즌을 따라 움직이는가\n{'='*78}")
    th0 = {f: J.fit(D[f]["Xs"], D[f]["z0"], D[f]["y"], FREE_A, 0.0, "brier")
           for f in ALL}
    print(f"  {'축':<16}{'배포':>9}" + "".join(f"{f:>10}" for f in ALL) + f"{'최대/최소':>11}")
    for i, nm in enumerate(names):
        v = [th0[f][2 + i] / D[f]["sd"][i] for f in ALL]
        print(f"  {nm:<16}{dep[i]:>9.4f}" + "".join(f"{x:>10.4f}" for x in v)
              + f"{max(v)/max(min(v), 1e-12):>11.2f}")

    # ── 전방향 쌍 ───────────────────────────────────────────────────────
    print(f"\n{'='*78}\n■ 전방향 전이 — 적합 fold의 β를 **다음 시즌**에 적용\n{'='*78}")
    print("  (배포 β는 2023·2024를 보고 골랐다. 2021→2022 쌍만 그 이점이 없다.)\n")
    print(f"  {'λ':>7} │ " + " │ ".join(f"{a}→{b:<6}" for a, b in PAIRS))
    print(f"  {'':>7} │ " + " │ ".join(f"{'BSS   배포Δ':>13}" for _ in PAIRS))
    acc = {lam: [] for lam in LAMS}
    for lam in LAMS:
        cells = []
        for a, b in PAIRS:
            th = J.fit(D[a]["Xs"], D[a]["z0"], D[a]["y"], FREE_A, lam, "brier")
            sc = J.score(th, D[b]["Xs"], D[b]["z0"], D[b]["y"])
            d = sc - D[b]["base"]
            acc[lam].append(d)
            cells.append(f"{sc:>7.1f} {d:>+6.1f}")
        print(f"  {lam:>7.5f} │ " + " │ ".join(cells)
              + f"   평균 {np.mean(acc[lam]):+.1f}  양수 {sum(x > 0 for x in acc[lam])}/3")

    # ── 대조: 배포 β를 그 fold에서 1차원으로 재확인 ─────────────────────
    print(f"\n{'='*78}\n■ 축별 1차원 — 전방향 쌍에서도 배포값이 서는가\n{'='*78}")
    print("  각 축을 격자로 움직이며 **네 fold 전부**의 β=0 대비 이득을 본다.")
    for i, nm in enumerate(names):
        grid = J.__dict__.get("GRID", None)
        from jointbeta2 import GRID
        print(f"\n── {nm}  (배포 {dep[i]}) " + "─" * 40)
        print(f"  {'β':>8} │ " + "".join(f"{f:>10}" for f in ALL)
              + f"{'부호':>6}{'크기비':>9}")
        z0g = {}
        for g in GRID[nm]:
            b = dep.copy()
            b[i] = g
            row = []
            for f in ALL:
                z = D[f]["z0"] + D[f]["X"] @ b
                row.append(bss(D[f]["y"], shift_to_prior(
                    1 / (1 + np.exp(-z)), D[f]["y"].mean())))
            z0g[g] = row
        z00 = z0g[0.0]
        for g in GRID[nm]:
            d = [z0g[g][k] - z00[k] for k in range(len(ALL))]
            pos = all(x > 0 for x in d)
            r = (max(d) / min(d)) if pos else float("nan")
            mark = "  ←배포" if abs(g - dep[i]) < 1e-9 else ""
            print(f"  {g:>8.4f} │ " + "".join(f"{x:>+10.1f}" for x in d)
                  + f"{'✓' if pos else '✗':>6}{r:>9.2f}{mark}")
        ok = [g for g in GRID[nm] if g > 0 and all(x > 0 for x in
              [z0g[g][k] - z00[k] for k in range(len(ALL))])]
        if ok:
            lo, hi = min(ok), max(ok)
            allow = [g for g in ok if g / 1.5 >= lo and g * 1.5 <= hi]
            print(f"    4-fold 고원 [{lo:g}, {hi:g}]  폭 {hi/lo:.1f}배"
                  f"   밴드허용 {f'[{min(allow):g}, {max(allow):g}]' if allow else '없음'}"
                  f"   배포 {dep[i]}")
        else:
            print("    4-fold 고원 없음")


if __name__ == "__main__":
    main()
