"""압축 축의 **크기** — 신인 투수간 기울기를 고치면 in-fold 로 몇 점인가.

rookdiag2 ①이 관계를 확립했다. 남은 질문은 값어치다.
    p'_ij = p_ij + (b−1)·(p̄_j − P_cohort)          b = 그 코호트의 투수간 기울기
in-fold **오라클**이다 (b 를 fold 자신의 라벨로 안다). 그래서 세 팔을 같이 낸다:

    신인       후보
    2+시즌     같은 조작을 베테랑에 걸었을 때 (축이 신인 고유인가)
    위약       투수를 무작위로 뽑아 만든 가짜 코호트 (행 수·투수 수 일치)
               memory: "무의미한 버킷이 서브그룹 튜닝의 영가설"

위약이 후보만큼 나오면 이 숫자는 오라클 자유도를 잰 것이지 축을 잰 것이 아니다.

실행:  .venv/Scripts/python.exe -u src/rookdiag4.py 2>&1 | tee experiments/rookdiag4.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

import rookdiag as R
import rookdiag2 as R2
from config import TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

RNG = np.random.default_rng(2024)
N_PLACEBO = 20


def oracle_gain(y, p, g, n_id, m, base):
    """m 코호트 안에서 투수간 기울기를 1로 되돌렸을 때의 in-fold ΔBSS."""
    n = np.bincount(g[m], minlength=n_id)
    sp = np.bincount(g[m], weights=p[m], minlength=n_id)
    pb = np.where(n > 0, sp / np.maximum(n, 1), 0.0)
    nn, yb, pbb = R2.pitcher_agg(g[m], y[m], p[m], n_id)
    b = R2.slope_of(nn, pbb, yb)
    P = np.average(pbb, weights=nn)
    p2 = p.copy()
    p2[m] = p[m] + (b - 1.0) * (pb[g[m]] - P)
    p2 = np.clip(p2, 1e-6, 1 - 1e-6)
    return b, bss(y, shift_to_prior(p2, y.mean())) - base


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    for fold in R.FOLDS:
        va, y, p, _ = R.build(fold, df, s)
        pp = R.prior_seasons(df, s, fold, "pitcher_id")
        g, uniq = pd.factorize(va["pitcher_id"].to_numpy())
        n_id = len(uniq)
        base = bss(y, p)
        rook, vet = pp == 0, pp >= 2
        print("\n" + "=" * 92)
        print(f"fold {fold}   기준 BSS {base:.1f}")
        print("=" * 92)
        print(f"\n  {'팔':<26}{'투수':>7}{'행':>10}{'비중':>7}{'b̂':>8}{'ΔBSS(오라클)':>14}")
        for lab, m in (("신인", rook), ("2+시즌", vet)):
            b, d = oracle_gain(y, p, g, n_id, m, base)
            print(f"  {lab:<26}{len(np.unique(g[m])):>7,}{m.sum():>10,}"
                  f"{m.mean()*100:>6.1f}%{b:>8.3f}{d:>+14.2f}")
        # 위약 — 신인과 같은 투수 수를 무작위로 뽑아 코호트를 만든다
        k = len(np.unique(g[rook]))
        gains = []
        for _ in range(N_PLACEBO):
            pick = RNG.choice(n_id, k, replace=False)
            m = np.isin(g, pick)
            if m.sum() < 1000:
                continue
            _, d = oracle_gain(y, p, g, n_id, m, base)
            gains.append(d)
        gains = np.array(gains)
        print(f"  {'위약 (무작위 투수 코호트)':<24}{k:>7,}{'—':>10}{'—':>7}{'—':>8}"
              f"{gains.mean():>+14.2f}")
        print(f"      위약 {len(gains)}회 — 평균 {gains.mean():+.2f}  최대 {gains.max():+.2f}"
              f"  sd {gains.std():.2f}")


if __name__ == "__main__":
    main()
