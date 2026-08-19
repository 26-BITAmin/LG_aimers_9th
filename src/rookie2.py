"""rookie.py 후속 — 그룹 간 차이에 **투수 클러스터 부트스트랩 SE**를 붙인다.

왜 필요한가
    rookie.py의 로컬 BSS는 fold2023과 fold2024가 반대 방향이다
    (R 투수 신인 404 vs 919). 그룹 비교는 쌍체가 아니고, 한 투수의 수천 행은
    서로 독립이 아니다 → 행 단위 SE는 과소평가다. 클러스터(투수) 단위로 잰다.

내는 값 (fold별, R/F별)
    Δ로컬BSS = BSS(신인) − BSS(2+시즌)          ± 클러스터 부트스트랩 SE
    Δ편향    = (p̄−ȳ)신인 − (p̄−ȳ)2+시즌         ± 같은 방식
                양수 = 신인을 상대적으로 **과대예측**

실행:  .venv/Scripts/python.exe -u src/rookie2.py 2>&1 | tee experiments/rookie2.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache
from config import TRAIN_CSV
from data import load_csv
from rookie import FOLDS, deployed_pred, prior_seasons

B = 1000
RNG = np.random.default_rng(42)


def cell_stats(cl, ncl, cells, sq, y):
    """클러스터 × 셀 집계. 부트스트랩은 이 표를 다시 합치기만 하면 된다."""
    K = len(cells)
    N = np.zeros((ncl, K)); S = np.zeros((ncl, K)); Y = np.zeros((ncl, K))
    for j, m in enumerate(cells):
        N[:, j] = np.bincount(cl[m], minlength=ncl)
        S[:, j] = np.bincount(cl[m], weights=sq[m], minlength=ncl)
        Y[:, j] = np.bincount(cl[m], weights=y[m], minlength=ncl)
    return N, S, Y


def stats_from(N, S, Y, P):
    """셀별 (로컬BSS, 편향). 0으로 나누는 셀은 nan."""
    n = N.sum(0); s = S.sum(0); yb = Y.sum(0); pb = P.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ybar = yb / n
        ref = ybar * (1 - ybar)
        loc = 1e5 * (1 - (s / n) / ref)
        bias = pb / n - ybar
    return loc, bias


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)

    for fold in FOLDS:
        va, y, p = deployed_pred(df, s, fold, cached)
        gt_f = va["game_type"].astype("object").to_numpy() == "F"
        pp = prior_seasons(df, s, fold, "pitcher_id")
        sq = (p - y) ** 2
        _, cl = np.unique(va["pitcher_id"].to_numpy(), return_inverse=True)
        ncl = cl.max() + 1

        cells, names = [], []
        for gname, gm in (("R", ~gt_f), ("F", gt_f)):
            cells += [gm & (pp == 0), gm & (pp >= 2)]
            names += [f"{gname} 신인", f"{gname} 2+시즌"]

        N, S, Y = cell_stats(cl, ncl, cells, sq, y)
        _, P, _ = cell_stats(cl, ncl, cells, p, y)     # P 합만 재사용

        loc0, bias0 = stats_from(N, S, Y, P)
        boot = np.empty((B, len(cells), 2))
        for b in range(B):
            idx = RNG.integers(0, ncl, ncl)
            boot[b, :, 0], boot[b, :, 1] = stats_from(N[idx], S[idx], Y[idx], P[idx])

        print("\n" + "=" * 84)
        print(f"fold {fold}   투수 {ncl:,}명 클러스터 부트스트랩 B={B}")
        print(f"  {'셀':<12}{'n':>10}{'로컬BSS':>12}{'±SE':>9}{'편향':>11}{'±SE':>9}")
        for j, nm in enumerate(names):
            print(f"  {nm:<12}{int(N[:,j].sum()):>10,}{loc0[j]:>12.1f}"
                  f"{np.nanstd(boot[:,j,0],ddof=1):>9.1f}"
                  f"{bias0[j]:>+11.4f}{np.nanstd(boot[:,j,1],ddof=1):>9.4f}")

        for gi, gname in ((0, "R"), (2, "F")):
            a, v = gi, gi + 1                     # 신인 − 2+시즌
            dl = loc0[a] - loc0[v]
            db = bias0[a] - bias0[v]
            sl = np.nanstd(boot[:, a, 0] - boot[:, v, 0], ddof=1)
            sb = np.nanstd(boot[:, a, 1] - boot[:, v, 1], ddof=1)
            print(f"  [{gname}] 신인−2+시즌   Δ로컬BSS {dl:>+8.1f} ± {sl:>6.1f} (t {dl/sl:+.1f})"
                  f"   Δ편향 {db:>+.4f} ± {sb:.4f} (t {db/sb:+.1f})")

        # R vs F 자체도 같은 방식으로 (신인 여부 무관, 전체 행)
        cells2 = [~gt_f, gt_f]
        N2, S2, Y2 = cell_stats(cl, ncl, cells2, sq, y)
        _, P2, _ = cell_stats(cl, ncl, cells2, p, y)
        l2, b2 = stats_from(N2, S2, Y2, P2)
        bt = np.empty((B, 2, 2))
        for b in range(B):
            idx = RNG.integers(0, ncl, ncl)
            bt[b, :, 0], bt[b, :, 1] = stats_from(N2[idx], S2[idx], Y2[idx], P2[idx])
        dl = l2[1] - l2[0]; db = b2[1] - b2[0]
        sl = np.nanstd(bt[:, 1, 0] - bt[:, 0, 0], ddof=1)
        sb = np.nanstd(bt[:, 1, 1] - bt[:, 0, 1], ddof=1)
        print(f"  [전체] F−R        Δ로컬BSS {dl:>+8.1f} ± {sl:>6.1f} (t {dl/sl:+.1f})"
              f"   Δ편향 {db:>+.4f} ± {sb:.4f} (t {db/sb:+.1f})")


if __name__ == "__main__":
    main()
