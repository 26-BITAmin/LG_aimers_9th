"""§7.60 — 오염 보정판 gap 축의 **β 재선정** (단계별 증분).

왜 다시 잡나
    보정하면 피처의 스케일과 모양이 둘 다 바뀐다. 기존 β(0.02 / 0.015)는
    **원본 판본의 고원**에서 고른 값이라 그대로 쓸 근거가 없다.

설계 — §7.37이 count4를 δ_hand 위의 증분으로 잰 것과 같다
    A  gap_success   기준선 = 블렌드 + δ           → β_A 확정
    B  gap_bsuccess  기준선 = A까지                → β_B 확정
    C  gap_reverse   기준선 = A+B까지              → 채택/기각
    축을 쌓을 때 기준선에 이전 축이 빠져 있으면 같은 신호를 두 번 센다.

β 선정 규칙 (§7.37)
    양 fold 양수 구간(고원)을 찾고, **[0.5β, 1.5β] 전체가 그 안에** 들어가는 β.
    CV argmax를 고르지 않는다. 손실 곡선이 비대칭이면(오른쪽이 절벽) 왼쪽으로 치우친다.

⚠️ fold2023은 기준 130.2의 단절 fold다. 대조군조차 +25.9를 받는다 (§7.51).
   **판정은 fold2024 중심**, fold2023은 부호 확인용.

조건 ④ (이미 측정, `src/asofcontam.py`)
    gap_success  0.80 → 0.93 · gap_bsuccess 0.78 → 0.91 · gap_reverse 0.64 → 0.83

실행:  .venv/Scripts/python.exe -u src/asofcbeta.py 2>&1 | tee experiments/asofcbeta.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import asofdelta
import ctxfit
import features as F
import foldcache
from config import (ASOF_STRENGTH, CONTAM_AXES, CTX_AXES, CTX_K, MLP_ALPHA, SEED,
                    TARGET_COL, TRAIN_CSV)
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
# 위쪽 끝을 넉넉히 — 보정판은 β=0.06에서도 양수였다. 고원 경계를 못 찾으면
# 밴드를 그릴 수 없다 (§7.58에서 겪은 그대로).
GRID = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.14]

# (축, 부호). reverse는 잔차 기울기가 음수라 β도 음수로 얹는다 (§7.57).
STAGES = [("gap_success", +1), ("gap_bsuccess", +1), ("gap_reverse", -1)]


def pick(a23, a24, grid):
    """양 fold 양수 구간 + 밴드 규칙을 통과하는 β 후보들."""
    both = [(a23[i] > 0) and (a24[i] > 0) for i in range(len(grid))]
    pl = [grid[i] for i in range(len(grid)) if both[i]]
    if not pl:
        return None, [], (None, None)
    lo, hi = min(pl), max(pl)
    ok = [b for b in grid if 0.5 * b >= lo and 1.5 * b <= hi]
    return (lo, hi), ok, (lo, hi)


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(SEED)

    # fold별 상태를 한 번만 만든다 (contam fit이 제일 비싸다)
    st = {}
    for fold in FOLDS:
        tr, va = s < fold, s == fold
        d = df[va].copy()
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        ct = F.fit_artifacts(df[tr], contam_axes=CONTAM_AXES)["contam"]
        adj, _ = F.logit_adjust(d, {"ctx": ctx})
        z0 = z + adj
        V = {}
        for name, _ in STAGES:
            a = asofdelta.fit_asof_axes(df[tr], [(name, 1.0)], ASOF_STRENGTH, ct=ct)[0]
            V[name] = F.asof_lookup(d, a, ct=ct)
        # 대조군 — 값 분포는 그대로, 행과의 연결만 끊는다
        V["_shuf"] = V["gap_success"][rng.permutation(len(d))]
        st[fold] = dict(yv=yv, z0=z0, V=V, n=len(d))
        print(f"fold {fold}  행 {len(d):,}  상태 준비 완료")

    chosen = []          # [(name, beta), ...] — 확정된 축
    for stage, (name, sign) in enumerate(STAGES, 1):
        rows = {}
        print("\n" + "=" * 104)
        print(f"단계 {stage} — {name}   (기준선: 블렌드 + δ"
              + (" + " + " + ".join(f"{n}@{b:+.3f}" for n, b in chosen) if chosen else "")
              + ")")
        print("=" * 104)
        for fold in FOLDS:
            x = st[fold]
            fixed = sum(b * x["V"][n] for n, b in chosen) if chosen else 0.0
            zb = x["z0"] + fixed
            base = bss(x["yv"], shift_to_prior(1 / (1 + np.exp(-zb)), x["yv"].mean()))

            def gain(v, b):
                p = shift_to_prior(1 / (1 + np.exp(-(zb + b * v))), x["yv"].mean())
                return bss(x["yv"], p) - base

            print(f"  fold {fold}  기준 {base:7.1f}")
            print(f"    {'':<10}" + "".join(f"{sign*b:>+8.3f}" for b in GRID))
            for tag, v in ((name, x["V"][name]), ("[대조군]", x["V"]["_shuf"])):
                r = [gain(v, sign * b) for b in GRID]
                rows.setdefault(tag, {})[fold] = r
                print(f"    {tag:<10}" + "".join(f"{g:>+8.2f}" for g in r))

        a23 = np.array(rows[name][2023]); a24 = np.array(rows[name][2024])
        _, ok, (lo, hi) = pick(a23, a24, GRID)
        print(f"\n  양 fold 양수 |β| 구간  [{lo}, {hi}]"
              + (f"   폭 {hi/lo:.0f}배" if lo else "   → 없음"))
        if not ok:
            print(f"  ❌ 밴드 규칙을 만족하는 β 없음 → {name} 기각")
            continue
        print(f"  밴드 규칙 허용 |β|  {ok}")
        print(f"    {'|β|':<8}{'f2023':>10}{'f2024':>10}{'크기비':>9}")
        for b in ok:
            i = GRID.index(b)
            ratio = max(abs(a23[i]), abs(a24[i])) / max(min(abs(a23[i]), abs(a24[i])), 1e-9)
            print(f"    {b:<8}{a23[i]:>+10.2f}{a24[i]:>+10.2f}{ratio:>9.2f}")
        # ⚠️ 앵커를 **점수가 아니라 고원 기하**에 둔다.
        #    초안은 "f2024 argmax − 한 칸"이었는데, 그건 결국 검증 fold의 점수를
        #    보고 고르는 것이다 (§7.37이 금지한 것). 밴드 허용 구간은 고원 경계만으로
        #    정해지므로 점수를 안 본다. 거기서 **최대값의 한 칸 아래**를 쓴다 —
        #    위쪽 끝은 절벽이고(β 과대 = 손실), 아래쪽 끝은 신호를 버린다.
        i = max(len(ok) - 2, 0)
        sug = ok[i]
        print(f"  → 밴드 허용 최대 {ok[-1]},  채택 {sug} (한 칸 아래 — 절벽 회피)")
        print(f"     참고: f2024 argmax = {GRID[int(np.argmax(a24))]} "
              f"(고르지 않는다)")
        chosen.append((name, sign * sug))

    print("\n" + "=" * 104)
    print("확정")
    print("=" * 104)
    for n, b in chosen:
        print(f"  {n:<16} β = {b:+.3f}")
    print("\n  ⛔ LB를 보고 다시 고르지 않는다. 위는 전부 홀드아웃에서 나왔다.")


if __name__ == "__main__":
    main()
