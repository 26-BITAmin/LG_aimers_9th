"""M_p가 왜 실패했나 — 상관은 같은데 랜덤 노이즈와 결과가 다르다.

의심: 오차가 랜덤이 아니라 **척도/레벨이 어긋난** 구조적 오차다.
     z̄_b ≈ α + β·M_p 에서 β≠1이면, (z − M_p)에 β 어긋남이 통째로 섞여
     s로 줄일 때 신호까지 같이 줄인다.

재는 것 (진단용. 검증 fold를 쓰므로 방법이 아니라 부검이다)
    β  = z̄_b를 M_p로 회귀한 기울기, 절편 α
    std 비교, 그리고 **train에서 fit한 α·β로 M_p를 교정**했을 때 이득이 살아나나
    (train fit이면 행 독립이라 그대로 구현 가능하다)

실행:  LGA_FOLD_KEY=5488ef1b18 .venv/Scripts/python.exe -u src/basediag.py
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
from pbase import CACHE
from rookie import FOLDS, FOLD_KEY, deployed_pred
from validation import _logit, bss, shift_to_prior

BG, AG, SG = [1.0, 1.1, 1.2, 1.3], [1.0, 1.2, 1.4, 1.6], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    for fold in FOLDS:
        f = CACHE / f"pbase_f{fold}_{FOLD_KEY}_38.npz"
        if not f.exists():
            print(f"없음: {f.name}"); continue
        mp = np.load(f)["va"]

        va, y, p = deployed_pred(df, s, fold, cached)
        z = _logit(p)
        _, cl = np.unique(va["pitcher_id"].to_numpy(), return_inverse=True)
        g = cl.astype(np.int64) * 1000 + qbucket(cl, va["asof_pitcher_n"].to_numpy(float), 5)
        _, g = np.unique(g, return_inverse=True)
        zb, zj = gmean(g, z)[g], gmean(cl, z)[cl]
        za, base = z.mean(), bss(y, p)

        b1 = np.cov(mp, zb)[0, 1] / np.var(mp)
        a1 = zb.mean() - b1 * mp.mean()
        print(f"\n=== fold {fold} ===")
        print(f"  std: z̄_b {zb.std():.4f}  M_p {mp.std():.4f}   "
              f"z̄_b ~ M_p 회귀 기울기 {b1:.3f}  절편 {a1:+.4f}  corr {np.corrcoef(mp,zb)[0,1]:.3f}")
        # M_p를 z̄_b 척도로 교정한 뒤 잔차의 성질
        mpc = a1 + b1 * mp
        print(f"  교정 후 잔차 std {np.std(zb-mpc):.4f}"
              f"   (z의 투수내 빠른 성분 std {np.std(z-zb):.4f})")

        for name, bl in (("원본 M_p", mp), ("척도 교정 M_p", mpc)):
            blj = gmean(cl, bl)[cl]
            best = ((1., 1., 1.), 0.0)
            for b in BG:
                for a in AG:
                    for sc in SG:
                        zz = za + b*(blj-za) + a*(bl-blj) + sc*(z-bl)
                        d = bss(y, shift_to_prior(1/(1+np.exp(-zz)), y.mean())) - base
                        if d > best[1]:
                            best = ((b, a, sc), d)
            print(f"  {name:<14} 최적 {best[0]}  Δ {best[1]:+.1f}")


if __name__ == "__main__":
    main()
