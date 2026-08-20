"""basenoise 부검 — 기준선 오차가 **행 단위**냐 **구간 단위**냐가 갈림길.

basenoise는 오라클 z̄_b에 i.i.d. 행 노이즈를 섞었고 ρ=0.93에서도 +47/+31이 남았다.
실제 M_p는 같은 ρ인데 이득이 거의 0이다. 가설:

    z' = B + s(z − B),  B = z̄_b − δ  일 때
    z' = z̄_b + s·fast − (1−s)·δ
    δ가 **행 단위 백색잡음**이면 구간 평균에서 지워져 레벨을 안 건드린다.
    δ가 **구간 단위 오프셋**이면 그대로 레벨 오차로 남는다 — 수축이 그걸 증폭한다.

M_p는 느리게 변하는 피처의 매끄러운 함수라 오차가 구간 단위다.
같은 크기의 두 노이즈를 나란히 넣어 확인한다. 가설이 맞으면 구간 노이즈에서만 죽는다.

실행:  LGA_FOLD_KEY=5488ef1b18 .venv/Scripts/python.exe -u src/noiseshape.py
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

SD = [0.0, 0.01, 0.02, 0.03, 0.045, 0.055, 0.07]     # 0.055 = 실제 M_p 오차
BG, AG, SG = [1.0, 1.1, 1.2, 1.3], [1.0, 1.2, 1.4, 1.6], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
RNG = np.random.default_rng(42)


def best(z, y, base, bl, cl, za):
    blj = gmean(cl, bl)[cl]
    b0 = 0.0
    for b in BG:
        for a in AG:
            for sc in SG:
                zz = za + b*(blj-za) + a*(bl-blj) + sc*(z-bl)
                d = bss(y, shift_to_prior(1/(1+np.exp(-zz)), y.mean())) - base
                b0 = max(b0, d)
    return b0


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    for fold in FOLDS:
        va, y, p = deployed_pred(df, s, fold, cached)
        z = _logit(p)
        _, cl = np.unique(va["pitcher_id"].to_numpy(), return_inverse=True)
        g = cl.astype(np.int64) * 1000 + qbucket(cl, va["asof_pitcher_n"].to_numpy(float), 5)
        _, g = np.unique(g, return_inverse=True)
        zb = gmean(g, z)[g]
        za, base = z.mean(), bss(y, p)
        ng = g.max() + 1
        print(f"\n=== fold {fold}  기준 {base:.1f}   (빠른 성분 std {np.std(z-zb):.4f}) ===")
        print(f"  {'오차 std':>9}{'행 단위 노이즈':>16}{'구간 단위 오프셋':>18}")
        for sd in SD:
            if sd == 0:
                print(f"  {sd:>9.3f}{best(z,y,base,zb,cl,za):>+16.1f}"
                      f"{best(z,y,base,zb,cl,za):>+18.1f}")
                continue
            row = zb + RNG.normal(0, sd, len(zb))
            off = zb + RNG.normal(0, sd, ng)[g]
            print(f"  {sd:>9.3f}{best(z,y,base,row,cl,za):>+16.1f}"
                  f"{best(z,y,base,off,cl,za):>+18.1f}")


if __name__ == "__main__":
    main()
