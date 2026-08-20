"""jointbeta3 수정 — **pre-2023 F 라벨 오염** 때문에 평가 모집단을 갈아낀다 (사용자 지적).

무엇이 틀렸나
    §7.84에서 전방향 쌍을 3개로 늘리려고 fold 2021·2022를 추가했다.
    **하필 오염이 제일 심한 두 시즌이다.**

        시즌      F비중    q_s     전체행 라벨오염율    ȳ(R)     ȳ(F)
        2019     10.9%   .064          0.70%         .5495    .6892
        2020      9.5%   .065          0.62%         .5269    .5878
        2021     10.5%   .171          1.79%         .5128    .7038
        2022     12.3%   .211          2.60%         .5037    .7087
        2023     10.5%    —            0%            .5031    .4729
        2024     11.8%    —            0%            .4897    .4593

    ȳ(F) 열 하나로 다 보인다 — .70을 유지하다 2023에 .47로 떨어진다.

    피처 쪽은 이미 막혀 있다 (`axis_matrix`가 `contam_axes`를 넘긴다).
    **뚫린 것은 라벨 쪽이다** — `soften`은 `Xtr`만 늘리고 `yva`는 원본이다.
    fold2022의 BSS는 전체 행의 2.6%가 뒤집힌 라벨에 대고 잰 점수다.
    Brier 0.0001 = BSS 40점인 지표에서 이건 해상도의 260배짜리 교란이다.

해결 — R 전용 평가, 단 **게이트를 먼저 통과해야 한다**
    R 라벨은 전 시즌 오염이 없다. 그런데 "R-only β = 전체행 β"인지는 별개 질문이다.
    **2023·2024가 그 답을 준다** — 그 두 fold는 F도 깨끗하므로, 거기서 R-only와
    ALL이 같은 β를 주면 R-only는 유효한 대리 측정이다. 다르면 못 쓴다.

    ⛔ 게이트 실패 시 2021/2022 숫자를 읽지 않는다. 순서를 지킨다.

같이 찍는 것
    F 전용 BSS. 오염 fold에서 얼마나 망가지는지를 눈으로 확인하는 대조군이다.
    2023·2024의 F와 나란히 놓으면 "F가 원래 어려운 것"과 "오염" 이 갈린다.

실행:  .venv/Scripts/python.exe -u src/jointbeta4.py 2>&1 | tee experiments/jointbeta4.log
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
from jointbeta2 import GRID
from config import ASOF_AXES, CTX_AXES, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

ALL_F = [2021, 2022, 2023, 2024]
CLEAN = [2023, 2024]                  # 오염 없음 — 게이트는 여기서만 판정한다
PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024)]
LAMS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
FREE_A = [False, False, True, True, True, True]
POPS = ["ALL", "R", "F"]


def sc(y, z):
    return bss(y, shift_to_prior(1 / (1 + np.exp(-z)), y.mean()))


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = J.foldcache.load(ALL_F, verbose=False)
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])

    D = {}
    for f in ALL_F:
        X, names, _, va = J.axis_matrix(df, s, f)
        y = cached[f][2]
        z0 = (1 - MLP_ALPHA) * cached[f][0] + MLP_ALPHA * cached[f][1]
        assert np.array_equal(y, va[TARGET_COL].to_numpy(dtype=np.float64))
        isF = va["game_type"].to_numpy() == "F"
        sd = X.std(axis=0)
        D[f] = dict(X=X, sd=sd, y=y, z0=z0,
                    m={"ALL": np.ones(len(y), bool), "R": ~isF, "F": isF})
        row = []
        for p in POPS:
            m = D[f]["m"][p]
            row.append(f"{p} n={m.sum():>7,} ȳ={y[m].mean():.4f}"
                       f" 배포 {sc(y[m], (z0 + X @ dep)[m]):>8.1f}")
        print(f"fold {f}  " + "  │ ".join(row))
    # 모집단별 기준선 · 표준화행렬을 미리 만든다 (모집단이 바뀌면 sd도 바뀐다)
    for f in ALL_F:
        D[f]["base"], D[f]["Xs"] = {}, {}
        for p in POPS:
            m = D[f]["m"][p]
            D[f]["base"][p] = sc(D[f]["y"][m], (D[f]["z0"] + D[f]["X"] @ dep)[m])
            D[f]["Xs"][p] = D[f]["X"][m] / D[f]["X"][m].std(axis=0)
    print(f"\n축 {names}   배포 β {dict(zip(names, dep))}")

    # ── 게이트: 깨끗한 fold에서 R-only가 ALL과 같은 β를 주는가 ───────────
    print(f"\n{'='*78}\n■ 게이트 — 오염 없는 fold(2023·2024)에서 R-only ≟ ALL\n{'='*78}")
    print("  통과 조건: R-only β가 ALL β와 같은 부호이고 비가 1.3배 이내.")
    print("  (F는 참고용. 2023·2024의 F는 깨끗하므로 순수 모집단 효과다.)\n")
    gate = True
    for f in CLEAN:
        th = {p: J.fit(D[f]["Xs"][p], D[f]["z0"][D[f]["m"][p]],
                       D[f]["y"][D[f]["m"][p]], FREE_A, 0.0, "brier") for p in POPS}
        print(f"  fold {f}   원단위 β")
        print(f"    {'축':<16}{'배포':>9}{'ALL':>10}{'R':>10}{'F':>10}{'R/ALL':>9}")
        for i, nm in enumerate(names):
            v = {}
            for p in POPS:
                m = D[f]["m"][p]
                v[p] = th[p][2 + i] / D[f]["X"][m].std(axis=0)[i]
            r = v["R"] / v["ALL"] if v["ALL"] else float("nan")
            ok = np.isfinite(r) and 1 / 1.3 <= r <= 1.3
            gate &= bool(ok)
            print(f"    {nm:<16}{dep[i]:>9.4f}{v['ALL']:>10.4f}{v['R']:>10.4f}"
                  f"{v['F']:>10.4f}{r:>8.2f}{'✓' if ok else ' ✗'}")
    print(f"\n  → 게이트 {'통과 ✅' if gate else '실패 ❌'}")
    if not gate:
        print("  ⛔ R-only는 ALL의 대리가 되지 못한다. 2021/2022는 읽지 않는다.")
        print("     (전방향 관측은 2023→2024 하나로 남는다 — §7.83 ②의 −25.4.)")

    # ── 전방향 전이 (모집단별) ──────────────────────────────────────────
    print(f"\n{'='*78}\n■ 전방향 전이 — 적합 fold의 결합 MLE β를 **다음 시즌**에\n{'='*78}")
    for p in POPS:
        tag = {"ALL": "⚠️ 2021·2022는 라벨 오염", "R": "✅ 라벨 깨끗",
               "F": "⚠️ 2021·2022는 오염 그 자체"}[p]
        print(f"\n── 모집단 {p}  ({tag}) " + "─" * 34)
        print(f"  {'λ':>7} │ " + " │ ".join(f"{a}→{b}".ljust(14) for a, b in PAIRS)
              + " │  평균  양수")
        for lam in LAMS:
            cells, ds = [], []
            for a, b in PAIRS:
                ma, mb = D[a]["m"][p], D[b]["m"][p]
                th = J.fit(D[a]["Xs"][p], D[a]["z0"][ma], D[a]["y"][ma],
                           FREE_A, lam, "brier")
                v = J.score(th, D[b]["Xs"][p], D[b]["z0"][mb], D[b]["y"][mb])
                d = v - D[b]["base"][p]
                ds.append(d)
                cells.append(f"{v:>7.1f} {d:>+6.1f}")
            print(f"  {lam:>7.5f} │ " + " │ ".join(cells)
                  + f" │ {np.mean(ds):>+6.1f}  {sum(x > 0 for x in ds)}/3")

    # ── fold별 자기 최적 β — 드리프트하는가 ─────────────────────────────
    print(f"\n{'='*78}\n■ fold별 결합 MLE β (R-only, λ=0) — 시즌을 따라 움직이는가\n{'='*78}")
    th0 = {}
    for f in ALL_F:
        m = D[f]["m"]["R"]
        th0[f] = J.fit(D[f]["Xs"]["R"], D[f]["z0"][m], D[f]["y"][m],
                       FREE_A, 0.0, "brier")
    print(f"  {'축':<16}{'배포':>9}" + "".join(f"{f:>10}" for f in ALL_F)
          + f"{'최대/최소':>11}")
    for i, nm in enumerate(names):
        v = [th0[f][2 + i] / D[f]["X"][D[f]["m"]["R"]].std(axis=0)[i] for f in ALL_F]
        print(f"  {nm:<16}{dep[i]:>9.4f}" + "".join(f"{x:>10.4f}" for x in v)
              + f"{max(v)/max(min(v), 1e-12):>11.2f}")

    # ── 축별 4-fold 고원 (R-only) ───────────────────────────────────────
    print(f"\n{'='*78}\n■ 축별 1차원 — 4 fold 전부 R-only. 배포값이 밴드 안인가\n{'='*78}")
    for i, nm in enumerate(names):
        print(f"\n── {nm}  (배포 {dep[i]}) " + "─" * 40)
        print(f"  {'β':>8} │ " + "".join(f"{f:>10}" for f in ALL_F)
              + f"{'부호':>6}{'크기비':>9}")
        tab = {}
        for g in GRID[nm]:
            b = dep.copy()
            b[i] = g
            tab[g] = [sc(D[f]["y"][D[f]["m"]["R"]],
                         (D[f]["z0"] + D[f]["X"] @ b)[D[f]["m"]["R"]]) for f in ALL_F]
        z00 = tab[0.0]
        for g in GRID[nm]:
            d = [tab[g][k] - z00[k] for k in range(len(ALL_F))]
            pos = all(x > 0 for x in d)
            print(f"  {g:>8.4f} │ " + "".join(f"{x:>+10.1f}" for x in d)
                  + f"{'✓' if pos else '✗':>6}"
                  + (f"{max(d)/min(d):>9.2f}" if pos else f"{'—':>9}")
                  + ("  ←배포" if abs(g - dep[i]) < 1e-9 else ""))
        ok = [g for g in GRID[nm] if g > 0
              and all(tab[g][k] - z00[k] > 0 for k in range(len(ALL_F)))]
        if ok:
            lo, hi = min(ok), max(ok)
            al = [g for g in ok if g / 1.5 >= lo and g * 1.5 <= hi]
            print(f"    4-fold 고원 [{lo:g}, {hi:g}]  폭 {hi/lo:.1f}배   밴드허용 "
                  f"{f'[{min(al):g}, {max(al):g}]' if al else '없음'}"
                  f"   배포 {dep[i]}"
                  f"   {'✅ 안' if al and min(al) <= dep[i] <= max(al) else '❌ 밖'}")
        else:
            print("    4-fold 고원 없음")


if __name__ == "__main__":
    main()
