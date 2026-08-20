"""noiseshape가 가설을 기각했다 — 오차의 '모양'(행/구간)이 아니면 '성질'이다.

    B = z̄_b − δ 로 두면  z' = z̄_b + s·fast − (1−s)·δ
    δ가 **무정보**면 손실은 2차(분산): (1−s)²·Var(δ) — 작다.
    δ가 **유정보**(y를 예측하는 성분)면 손실은 1차: 진짜 신호를 (1−s)만큼 버린다.
    M_p는 약하지만 진짜 예측기다 → z̄_b − M_p는 유정보일 수밖에 없다.

같은 크기(std)의 두 δ를 나란히 놓는다.
    무정보 δ : 난수
    유정보 δ : (z̄_b − M_p)를 같은 std로 스케일
λ 스윕도 같이 — B_λ = λ·z̄_b + (1−λ)·M_p̂ 로 기준선 품질을 연속으로 낮춘다.

실행:  LGA_FOLD_KEY=5488ef1b18 .venv/Scripts/python.exe -u src/errkind.py
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
from noiseshape import best
from pbase import CACHE
from rookie import FOLDS, FOLD_KEY, deployed_pred
from validation import _logit, bss, shift_to_prior

RNG = np.random.default_rng(42)
LAM = [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    for fold in FOLDS:
        f = CACHE / f"pbase_f{fold}_{FOLD_KEY}_38.npz"
        mp = np.load(f)["va"]
        va, y, p = deployed_pred(df, s, fold, cached)
        z = _logit(p)
        _, cl = np.unique(va["pitcher_id"].to_numpy(), return_inverse=True)
        g = cl.astype(np.int64) * 1000 + qbucket(cl, va["asof_pitcher_n"].to_numpy(float), 5)
        _, g = np.unique(g, return_inverse=True)
        zb = gmean(g, z)[g]
        za, base = z.mean(), bss(y, p)

        b1 = np.cov(mp, zb)[0, 1] / np.var(mp)
        mpc = zb.mean() - b1 * mp.mean() + b1 * mp        # 척도 교정
        d_info = zb - mpc
        sd = d_info.std()
        print(f"\n=== fold {fold}  기준 {base:.1f}  (유정보 δ std {sd:.4f}) ===")
        print(f"  무정보 δ (같은 std)   Δ {best(z,y,base,zb+RNG.normal(0,sd,len(zb)),cl,za):+.1f}")
        print(f"  유정보 δ (z̄_b−M_p̂)   Δ {best(z,y,base,zb-d_info,cl,za):+.1f}")
        print(f"  {'λ (1=오라클, 0=M_p̂)':>22}{'Δ BSS':>9}{'그 기준선 단독 BSS':>18}")
        for lam in LAM:
            B = lam * zb + (1 - lam) * mpc
            solo = bss(y, shift_to_prior(1/(1+np.exp(-B)), y.mean()))
            print(f"  {lam:>22.2f}{best(z,y,base,B,cl,za):>+9.1f}{solo:>18.1f}")


if __name__ == "__main__":
    main()
