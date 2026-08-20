"""pbase 실패 부검 — 이득이 기준선 정확도에 얼마나 가파르게 의존하나.

pbase: corr(M_p, z̄_b) = 0.92~0.93 인데 이득 0. 오라클은 +48~83.
질문: 0.93이 모자란 것인가, 아니면 애초에 오라클 이득이 z̄_b가
**모델 자신의 평균**이라는 사실에만 기대는 인공물인가?

방법: 오라클 z̄_b에 노이즈를 섞어 목표 상관 ρ를 만들고 스윕을 다시 돈다.
      ρ→1에서 오라클, ρ=0.93에서 pbase 근처가 나와야 부검이 성립한다.
      ρ 곡선이 가파르면 "더 정확한 기준선"(같은 모델로 상황 주변화)이 답이고,
      완만한데도 0이면 대체물 노선 자체를 접는다.

실행:  LGA_FOLD_KEY=5488ef1b18 .venv/Scripts/python.exe -u src/basenoise.py
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

RHO = [1.0, 0.995, 0.99, 0.98, 0.96, 0.93, 0.90]
BG, AG, SG = [1.0, 1.1, 1.2], [1.0, 1.2, 1.4, 1.6], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
RNG = np.random.default_rng(42)


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    G = {}
    for fold in FOLDS:
        va, y, p = deployed_pred(df, s, fold, cached)
        z = _logit(p)
        _, cl = np.unique(va["pitcher_id"].to_numpy(), return_inverse=True)
        zj = gmean(cl, z)[cl]
        g = cl.astype(np.int64) * 1000 + qbucket(cl, va["asof_pitcher_n"].to_numpy(float), 5)
        _, g = np.unique(g, return_inverse=True)
        zb = gmean(g, z)[g]
        za = z.mean()
        base = bss(y, p)
        print(f"\n=== fold {fold}  기준 {base:.1f} ===")
        print(f"  {'ρ(기준선~z̄_b)':>14}{'최적 (b,a,s)':>18}{'Δ BSS':>10}")
        row = {}
        for r in RHO:
            if r >= 1.0:
                zb_n, zj_n = zb, zj
            else:
                # 목표 상관 r: zb_n = zb + e,  var(e) = var(zb)(1/r²−1)
                e = RNG.normal(0, np.std(zb) * np.sqrt(1 / r**2 - 1), len(zb))
                zb_n = zb + e
                zj_n = gmean(cl, zb_n)[cl]      # 기준선이 흐려지면 그 위 평균도 흐려진다
            best = ((1., 1., 1.), 0.0)
            for b in BG:
                for a in AG:
                    for sc in SG:
                        zz = za + b*(zj_n-za) + a*(zb_n-zj_n) + sc*(z-zb_n)
                        d = bss(y, shift_to_prior(1/(1+np.exp(-zz)), y.mean())) - base
                        if d > best[1]:
                            best = ((b, a, sc), d)
            row[r] = best
            print(f"  {r:>14.3f}{str(best[0]):>18}{best[1]:>+10.1f}")
        G[fold] = row
    print("\n두 fold 최소값 (같은 ρ, 각자 최적):")
    for r in RHO:
        print(f"  ρ={r:.3f}   " + "   ".join(f"fold{f} {G[f][r][1]:+.1f}" for f in FOLDS))


if __name__ == "__main__":
    main()
