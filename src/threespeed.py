"""3축 완전판 — 투수간(b) × 투수내 느린성분(a) × 빠른성분(s).

tempnull ②에서 투수간 스케일이 T=1.1까지 단조 증가했다 (스윕이 짧았다).
twospeed는 그 축을 1.0으로 고정한 채였으므로 천장을 과소평가했을 수 있다.

    z' = z̄_all + b·(z̄_j − z̄_all) + a·(z̄_b − z̄_j) + s·(z − z̄_b)

⚠️ 여전히 오라클 (z̄_j·z̄_b가 검증 fold 행에서 나온다).

실행:  LGA_MODEL_DIR=submit/model_threespeed .venv/Scripts/python.exe -u src/threespeed.py
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
from rookie import FOLDS, deployed_pred
from validation import _logit, bss, shift_to_prior

BG = [1.0, 1.1, 1.2, 1.3, 1.4]
AG = [1.0, 1.2, 1.4, 1.6]
SG = [0.5, 0.6, 0.7, 0.8]


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    G = {}
    for fold in FOLDS:
        va, y, p = deployed_pred(df, s, fold, cached)
        z = _logit(p)
        _, cl = np.unique(va["pitcher_id"].to_numpy(), return_inverse=True)
        an = va["asof_pitcher_n"].to_numpy().astype(float)
        zj = gmean(cl, z)[cl]
        g = cl.astype(np.int64) * 1000 + qbucket(cl, an, 5)
        _, g = np.unique(g, return_inverse=True)
        zb = gmean(g, z)[g]
        za = z.mean()
        base = bss(y, p)
        gg = {}
        for b in BG:
            for a in AG:
                for sc in SG:
                    zz = za + b * (zj - za) + a * (zb - zj) + sc * (z - zb)
                    gg[(b, a, sc)] = bss(y, shift_to_prior(
                        1 / (1 + np.exp(-zz)), y.mean())) - base
        G[fold] = gg
        k = max(gg, key=gg.get)
        print(f"fold {fold}  기준 {base:.1f}   최적 b={k[0]:.1f} a={k[1]:.1f} s={k[2]:.1f}"
              f"  Δ {gg[k]:+.1f}")
    common = max(G[2023], key=lambda k: min(G[2023][k], G[2024][k]))
    print(f"\n두 fold 최소값 최대화: b={common[0]:.1f} a={common[1]:.1f} s={common[2]:.1f}"
          f"   fold2023 {G[2023][common]:+.1f}   fold2024 {G[2024][common]:+.1f}"
          f"   (크기비 {max(G[2023][common],G[2024][common])/min(G[2023][common],G[2024][common]):.2f})")
    for src, dst in ((2023, 2024), (2024, 2023)):
        k = max(G[src], key=G[src].get)
        print(f"  교차: fold{src} 최적 → fold{dst} {G[dst][k]:+.1f}  (자체 {G[src][k]:+.1f})")


if __name__ == "__main__":
    main()
