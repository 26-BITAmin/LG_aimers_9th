"""§7.64 — `CTX_K`도 한 번도 스윕되지 않았다. §7.63과 **같은 자리**다.

    δ(p,c) = n_eff/(n_eff + K) · (rate_pc − rate_p)

K는 저표본 셀을 0으로 수축하는 강도다. 200으로 박혀 있고 스윕 기록이 없다.
`SMOOTH_K`(=200)에서 가져온 값인데, §7.63이 `ASOF_STRENGTH`에서 확인한 것과
**같은 종류의 오류**다:

    SMOOTH_K는 **레벨**(투수의 통산 성공률)을 리그 평균으로 수축한다.
    δ는 **차분**(셀 − 투수 자기 평균)이다. 차분을 0으로 수축하면 신호가 곧바로 깎인다.
    레벨의 최적 수축량이 차분의 최적 수축량과 같을 이유가 없다.

§7.63에서 같은 예측(더 작아야 한다)이 맞았다 — S 200 → 35, f2024 +9.08.
여기서도 같은 방향이 나오면 **기전이 두 번 확인**되는 것이고, 안 나오면
"차분이니까 K가 작아야 한다"는 일반화가 §7.63에 국한된다는 뜻이다.
(§7.61에서 "차분이니까 편향에 강하다"가 한쪽에서만 성립한 전례가 있다.)

⚠️ K를 바꾸면 δ의 스케일이 바뀐다 → **g(전역 배율) 재선정 필수.**
   K마다 밴드 규칙으로 g를 다시 고른다.

기준선 = 블렌드 + asof(배포 s21: S=35, β 0.03/0.03) — δ의 배포 위치에서의 증분.

실행:  .venv/Scripts/python.exe -u src/ctxk.py 2>&1 | tee experiments/ctxk.log
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
from config import (ASOF_AXES, ASOF_STRENGTH, CONTAM_AXES, CTX_AXES, CTX_HALF_LIFE,
                    CTX_K, MLP_ALPHA, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
KS = [10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0]
GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]


def band_pick(a23, a24, grid):
    both = [(a23[i] > 0) and (a24[i] > 0) for i in range(len(grid))]
    pl = [grid[i] for i in range(len(grid)) if both[i]]
    if not pl:
        return None, None, []
    lo, hi = min(pl), max(pl)
    ok = [g for g in grid if 0.5 * g >= lo and 1.5 * g <= hi]
    if not ok:
        return None, (lo, hi), []
    return ok[max(len(ok) - 2, 0)], (lo, hi), ok


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)

    st = {}
    for fold in FOLDS:
        tr_m, va = s < fold, s == fold
        tr, d = df[tr_m], df[va].copy()
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ref = int(s[tr_m].max())
        art = F.fit_artifacts(tr, contam_axes=CONTAM_AXES)
        art["asof"] = asofdelta.fit_asof_axes(tr, ASOF_AXES, ASOF_STRENGTH,
                                              ct=art.get("contam"))
        a_asof, _ = F.logit_adjust(d, {"asof": art["asof"]})
        z0 = z + a_asof
        st[fold] = dict(tr=tr, d=d, yv=yv, z0=z0,
                        base=bss(yv, shift_to_prior(1 / (1 + np.exp(-z0)), yv.mean())),
                        w=recency_weight(s[tr_m], CTX_HALF_LIFE, ref_season=ref))
        print(f"fold {fold}  기준 {st[fold]['base']:.1f}  (블렌드 + asof 배포판, δ 없음)")

    print("\n" + "=" * 100)
    print("K별 g 재선정 (밴드 규칙) 후 δ 증분")
    print("=" * 100)
    print(f"  {'K':<8}{'g':>7}{'f2023':>10}{'f2024':>10}{'크기비':>8}{'고원':>16}"
          f"{'허용 g':>24}")
    out = {}
    for K in KS:
        rows = {}
        for fold in FOLDS:
            x = st[fold]
            ctx = ctxfit.fit_ctx(x["tr"], TARGET_COL, CTX_AXES, K, weights=x["w"])
            v, _ = F.logit_adjust(x["d"], {"ctx": ctx})
            rows[fold] = [
                bss(x["yv"], shift_to_prior(1 / (1 + np.exp(-(x["z0"] + g * v))),
                                            x["yv"].mean())) - x["base"]
                for g in GRID]
        a23, a24 = np.array(rows[2023]), np.array(rows[2024])
        g, pl, ok = band_pick(a23, a24, GRID)
        if g is None:
            print(f"  {K:<8}  밴드 규칙 만족 g 없음 → 제외")
            continue
        i = GRID.index(g)
        r = max(abs(a23[i]), abs(a24[i])) / max(min(abs(a23[i]), abs(a24[i])), 1e-9)
        out[K] = (g, a23[i], a24[i], r)
        print(f"  {K:<8}{g:>7}{a23[i]:>+10.2f}{a24[i]:>+10.2f}{r:>8.2f}"
              f"{str(pl):>16}{str(ok):>24}")

    if CTX_K in out:
        g0, b23, b24, _ = out[CTX_K]
        print(f"\n  배포값 K={CTX_K} (g={g0}) 대비:")
        for K, (g, a, b, r) in sorted(out.items()):
            print(f"    K={K:<8} g={g:<6} Δf2023 {a-b23:>+8.2f}   Δf2024 {b-b24:>+8.2f}")
    print("\n  §7.63과 같은 방향(K가 작아야 한다)이 나오면 기전이 두 번 확인된다.")
    print("  반대거나 평탄하면 그 일반화는 §7.63에 국한된다 (§7.61의 전례).")


if __name__ == "__main__":
    main()
