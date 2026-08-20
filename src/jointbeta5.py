"""jointbeta4 수정 — 고원 경계를 **격자에 스냅하지 않는다**.

무엇이 틀렸나
    §7.84의 4-fold 밴드 판정에서 `hand`가 탈락했다.

        고원 [0.5, 2.75]  →  허용 [고원하한·1.5, 고원상한/1.5] = [0.75, 1.833]
        배포 1.875  →  1.833 를 2.3% 초과  →  ❌

    그런데 고원 상한 2.75는 **격자점**이다. 실제 경계는 2.75(f2022 +4.8)와
    3.0(f2022 −0.7) 사이에 있고, **2.8125 이상이면 배포값이 통과한다.**
    격자 간격이 판정을 뒤집는 자리에서 격자로 판정한 것이다.

    → 경계를 **선형 보간**으로 잡는다. 고원은 연속량이고 판정도 연속이어야 한다.

같이 고치는 것
    `gap_bsuccess`도 ❌였다 (허용 [0.01, 0.02], 배포 0.03). 이쪽은 격자 문제가
    아니라 **진짜 경계 밖**일 가능성이 크다 — f2021·f2022가 0.04에서 +1.1/+4.6로
    꺾이기 때문이다. 같은 방법으로 다시 재서 확인한다.

⛔ 판정이 바뀌어도 **배포값을 고치지 않는다.** 이 스크립트는 §7.84의 판정 오류를
   교정할 뿐이다. β 변경은 별도 근거가 필요하다 (§7.84 ⑤ 참조 — 후보들의
   기대 이득이 전부 LB 쌍체 SE ±1.16 아래다).

실행:  .venv/Scripts/python.exe -u src/jointbeta5.py 2>&1 | tee experiments/jointbeta5.log
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

FOLDS = [2021, 2022, 2023, 2024]
# 조밀 격자. 각 축의 고원 경계를 충분히 넘어가도록 잡는다.
FINE = {
    "hand":         np.round(np.arange(0.0, 4.01, 0.0625), 4),
    "count4":       np.round(np.arange(0.0, 3.51, 0.0625), 4),
    "gap_success":  np.round(np.arange(0.0, 0.1301, 0.0025), 5),
    "gap_bsuccess": np.round(np.arange(0.0, 0.1001, 0.0025), 5),
}


def edge(g, d, i0, step):
    """d[i0] > 0 에서 출발해 부호가 바뀌는 지점을 **선형 보간**으로 찾는다."""
    i = i0
    while 0 <= i + step < len(g) and d[i + step] > 0:
        i += step
    j = i + step
    if not (0 <= j < len(g)):
        return float(g[i])                     # 격자 끝까지 양수 — 경계 미관측
    t = d[i] / (d[i] - d[j])                   # d[j] <= 0
    return float(g[i] + t * (g[j] - g[i]))


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = J.foldcache.load(FOLDS, verbose=False)
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])

    D = {}
    for f in FOLDS:
        X, names, _, va = J.axis_matrix(df, s, f)
        y = cached[f][2]
        z0 = (1 - MLP_ALPHA) * cached[f][0] + MLP_ALPHA * cached[f][1]
        assert np.array_equal(y, va[TARGET_COL].to_numpy(dtype=np.float64))
        m = va["game_type"].to_numpy() != "F"        # R 전용 (§7.84 게이트 통과)
        D[f] = dict(X=X[m], y=y[m], z0=z0[m])
        print(f"fold {f}  R n={m.sum():,}")
    print(f"\n축 {names}   배포 β {dict(zip(names, dep))}\n")

    # ⚠️ 밴드 규칙은 **위쪽만** 구속한다. β→0에서 이득은 0으로 수렴할 뿐
    #    음수가 되지 않으므로 아래쪽 경계는 진짜 경계가 아니다 (손실 곡선 비대칭,
    #    §7.58 ②). 격자판에서 "고원 하한"으로 보였던 것은 **측정한 가장 작은 β**일
    #    뿐이고, 그걸 구속으로 쓰면 밴드가 인위적으로 좁아진다.
    print(f"{'축':<15}{'배포':>8}{'고원 상한(보간)':>18}"
          f"{'허용 상한 =/1.5':>18}{'판정':>7}{'여유':>10}")
    for i, nm in enumerate(names):
        g = FINE[nm]
        cur, base0 = np.zeros(len(g)), None
        for k, b_ in enumerate(g):
            b = dep.copy()
            b[i] = b_
            tot = [bss(D[f]["y"], shift_to_prior(
                1 / (1 + np.exp(-(D[f]["z0"] + D[f]["X"] @ b))), D[f]["y"].mean()))
                for f in FOLDS]
            if base0 is None:          # g[0] == 0 → 이 축을 끈 상태가 기준이다
                base0 = tot
            cur[k] = min(t - b0 for t, b0 in zip(tot, base0))
        # 배포값이 든 양수 구간의 경계를 보간으로 잡는다
        i0 = int(np.argmin(np.abs(g - dep[i])))
        if cur[i0] <= 0:
            print(f"{nm:<15}{dep[i]:>8.4f}{'배포값이 고원 밖':>22}")
            continue
        hi = edge(g, cur, i0, +1)
        ahi = hi / 1.5
        ok = dep[i] <= ahi
        print(f"{nm:<15}{dep[i]:>8.4f}{hi:>18.4g}{ahi:>18.4g}"
              f"{'✅' if ok else '❌':>7}{ahi/dep[i]:>9.2f}배")

    # 축별 조밀 표 — min-over-folds 곡선을 그대로 보여준다 (판정의 근거)
    print(f"\n{'='*78}\n■ min-over-4folds 이득 곡선 (β=0 대비). 판정은 이 곡선의 부호다\n{'='*78}")
    for i, nm in enumerate(names):
        g = FINE[nm]
        vals, base0 = [], None
        for b_ in g:
            b = dep.copy()
            b[i] = b_
            tot = [bss(D[f]["y"], shift_to_prior(
                1 / (1 + np.exp(-(D[f]["z0"] + D[f]["X"] @ b))), D[f]["y"].mean()))
                for f in FOLDS]
            if base0 is None:
                base0 = tot
            vals.append([t - b0 for t, b0 in zip(tot, base0)])
        print(f"\n── {nm} (배포 {dep[i]}) ─ 부호 전환 근방만 ─")
        print(f"  {'β':>9}" + "".join(f"{f:>10}" for f in FOLDS) + f"{'min':>10}")
        mn = [min(v) for v in vals]
        show = [k for k in range(1, len(g))
                if (mn[k] > 0) != (mn[k - 1] > 0) or abs(g[k] - dep[i]) < 1e-9]
        idx = sorted({k + o for k in show for o in (-2, -1, 0, 1, 2)
                      if 0 <= k + o < len(g)})
        for k in idx:
            print(f"  {g[k]:>9.4f}" + "".join(f"{v:>+10.1f}" for v in vals[k])
                  + f"{mn[k]:>+10.1f}"
                  + ("  ←배포" if abs(g[k] - dep[i]) < 1e-9 else ""))


if __name__ == "__main__":
    main()
