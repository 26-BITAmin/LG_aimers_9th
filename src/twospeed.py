"""두 속도 재보정 상한 — 느린 성분 증폭(a) × 빠른 성분 수축(s).

clockaxis.py 측정:
    느린 성분(투수 × 누적투구수 5분위) 기울기  신인 .7~.9 / 베테랑 1.4~1.8
    빠른 성분(그 안의 잔차)          기울기  신인 .36~.41 / 베테랑 .55~.74
→ 방향이 반대다. 하나의 s로는 못 고친다.

    z' = z̄_j + a·(z̄_b − z̄_j) + s·(z − z̄_b)
         투수평균   느린 성분        빠른 성분

⚠️ **상한이다.** z̄_j·z̄_b를 검증 fold 행에서 계산한다 (추론에선 금지).
   구현 가능한 형태는 행 독립 대체물이 필요하다. 천장부터 본다.

실행:  LGA_MODEL_DIR=submit/model_twospeed .venv/Scripts/python.exe -u src/twospeed.py
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
from clockaxis import gmean, qbucket
from config import TRAIN_CSV
from data import load_csv
from rookie import FOLDS, deployed_pred, prior_seasons
from validation import _logit, bss, shift_to_prior

B = 1000
RNG = np.random.default_rng(42)
AGRID = [1.0, 1.2, 1.4, 1.6, 1.8]
SGRID = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    R = {}

    for fold in FOLDS:
        va, y, p = deployed_pred(df, s, fold, cached)
        z = _logit(p)
        _, cl = np.unique(va["pitcher_id"].to_numpy(), return_inverse=True)
        an = va["asof_pitcher_n"].to_numpy().astype(float)
        pp = prior_seasons(df, s, fold, "pitcher_id")

        zj = gmean(cl, z)[cl]
        g = cl.astype(np.int64) * 1000 + qbucket(cl, an, 5)
        _, g = np.unique(g, return_inverse=True)
        zb = gmean(g, z)[g]
        slow, fast = zb - zj, z - zb
        base = bss(y, p)

        def ev(a, sc):
            zz = zj + a * slow + sc * fast
            return bss(y, shift_to_prior(1 / (1 + np.exp(-zz)), y.mean())) - base

        print(f"\n=== fold {fold}  기준 {base:.1f} ===  (Δ BSS, 행=a 느린성분, 열=s 빠른성분)")
        print("       " + "".join(f"{x:>9.1f}" for x in SGRID))
        grid = {}
        for a in AGRID:
            row = [ev(a, sc) for sc in SGRID]
            for sc, v in zip(SGRID, row):
                grid[(a, sc)] = v
            print(f"  a={a:.1f}" + "".join(f"{v:>9.1f}" for v in row))
        bestk = max(grid, key=grid.get)
        print(f"  최적 (a={bestk[0]:.1f}, s={bestk[1]:.1f})  Δ {grid[bestk]:+.1f}")
        R[fold] = (grid, ev, cl, y, p, pp, zj, slow, fast)

    print("\n" + "=" * 62)
    print("교차 적용 — 한 fold에서 고른 (a,s)를 다른 fold에")
    for src, dst in ((2023, 2024), (2024, 2023)):
        k = max(R[src][0], key=R[src][0].get)
        print(f"  fold{src} 최적 (a={k[0]:.1f}, s={k[1]:.1f})"
              f"  → fold{dst}에서 Δ {R[dst][1](*k):+.1f}"
              f"   (fold{src} 자체 {R[src][0][k]:+.1f})")

    # 공통 설정을 하나 고르고 두 fold 쌍체 SE
    common = max(SGRID and AGRID and
                 [(a, sc) for a in AGRID for sc in SGRID],
                 key=lambda k: min(R[2023][0][k], R[2024][0][k]))
    print(f"\n두 fold 최소값 최대화 설정: a={common[0]:.1f}, s={common[1]:.1f}")
    for fold in FOLDS:
        grid, ev, cl, y, p, pp, zj, slow, fast = R[fold]
        zz = zj + common[0] * slow + common[1] * fast
        pa = shift_to_prior(1 / (1 + np.exp(-zz)), y.mean())
        d = (p - y) ** 2 - (pa - y) ** 2
        C = cl.max() + 1
        num, cnt = np.bincount(cl, weights=d, minlength=C), np.bincount(cl, minlength=C)
        bt = np.array([num[i].sum() / cnt[i].sum()
                       for i in (RNG.integers(0, C, C) for _ in range(B))])
        k = 1e5 / (y.mean() * (1 - y.mean()))
        print(f"  fold{fold}  Δ {k*num.sum()/cnt.sum():+.1f} ± {k*bt.std(ddof=1):.1f}"
              f"   (신인 {(pp==0).mean()*100:.0f}%)")


if __name__ == "__main__":
    main()
