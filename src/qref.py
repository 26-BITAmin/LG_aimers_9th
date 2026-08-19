"""h_s의 기준 보정 — F_true = R + g (사용자 제안: R 기울기 + F 자체 레벨) (§7.70 후속)

배포된 soft label의 h_s = (R_s − F_obs,s)/p̄_s 는 F_true,s = R_s 를 가정한다.
그런데 정상 구간의 F reverse는 R보다 높다 (2023 리그 +.049, 짝지은 +.017) —
F 투수의 제구가 실제로 더 나쁘기 때문. 즉 현재는 **과소 보정**일 수 있다.

사용자 제안의 등가 형태
    F_true,s = (정상 F 앵커) − Σ(R의 시즌 기울기) = R_s + g,   g = 정상 구간 F−R 갭
    → q'_s = max(0, R_s + g − F_obs,s),   h'_s = q'_s / p̄_s
    닫힌형이라 self-distillation 문제가 없다.

제약
    fold2023은 train(≤2022)에 정상 F가 없어 g를 fit할 수 없다 → fold2024만 측정.
    (§7.69의 "정상 fold 1개" 상황 — 최종 판정은 LB 프로브뿐이다.)

팔:  g×0 (배포)  /  g×0.5  /  g×1.0  — 용량-반응의 방향과 곡률을 본다.

실행:  .venv/Scripts/python.exe -u src/qref.py 2>&1 | tee experiments/qref.log
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

from config import TARGET_COL, TRAIN_CSV
from data import load_csv
from modelabel import recover
from softlabel import prep_fold, train_eval

FOLD = 2024
SEEDS = (42, 7, 2024)
MULTS = (0.0, 0.5, 1.0)


def fit_h_g(df, train_mask, mult):
    """h'_s = (q_s + m·g)/p̄_s.  g = train의 정상 F 시즌에서 잰 F−R reverse 갭."""
    lab, _, _ = recover(df)
    rev = np.where(np.isnan(lab["reverse"]), 0.0, np.round(lab["reverse"]))
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    s = df["season"].to_numpy()
    g_ = df["game_type"].to_numpy().astype(str)

    gaps = []
    for yr in range(2023, 2025):
        mF = train_mask & (s == yr) & (g_ == "F")
        mR = train_mask & (s == yr) & (g_ == "R")
        if mF.any() and mR.any():
            gaps.append(float(rev[mF].mean() - rev[mR].mean()))
    assert gaps, "train에 정상 F 시즌이 없다 — 이 fold에서는 fit 불가"
    g = float(np.mean(gaps))

    h = {}
    for yr in range(2019, 2023):
        mR = train_mask & (s == yr) & (g_ == "R")
        mF = train_mask & (s == yr) & (g_ == "F")
        if not mF.any() or not mR.any():
            continue
        q = max(0.0, float(rev[mR].mean() - rev[mF].mean()) + mult * g)
        h[yr] = min(q / max(float(y[mF].mean()), 1e-9), 0.95)
    return h, g


def main():
    df = load_csv(TRAIN_CSV)
    t0 = time.time()
    tr, va, w, X = prep_fold(df, FOLD)
    tmask = (df.season < FOLD).to_numpy()
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    va_f = va["game_type"].to_numpy().astype(str) == "F"
    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    Xtr = X(tr)
    s_tr = tr["season"].to_numpy()
    g_tr = tr["game_type"].to_numpy().astype(str)

    res = {}
    for m in MULTS:
        h, g = fit_h_g(df, tmask, m)
        h_row = np.zeros(len(tr))
        for yr, v in h.items():
            h_row[(s_tr == yr) & (g_tr == "F")] = v
        dup = (h_row > 0) & (ytr == 1)
        w_soft = w.copy()
        w_soft[dup] *= (1 - h_row[dup])
        X2 = pd.concat([Xtr, Xtr[dup]], ignore_index=True)
        y2 = np.concatenate([ytr, np.zeros(int(dup.sum()))])
        w2 = np.concatenate([w_soft, np.asarray(w)[dup] * h_row[dup]])
        print(f"\n  g×{m}:  g={g:.4f}  h = "
              + "  ".join(f"{yr}:{v:.3f}" for yr, v in sorted(h.items())))
        r = train_eval(X2, y2, w2, X(va), yva, va_f, SEEDS)
        res[m] = r
        print(f"  BSS {r[:,0].mean():>8.1f} ± {r[:,0].std(ddof=1):.1f}"
              f"   F행 {r[:,1].mean():>7.1f}   R행 {r[:,2].mean():>7.1f}"
              f"   ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 60)
    print("쌍체 Δ vs g×0 (배포)")
    for m in MULTS[1:]:
        d = res[m] - res[0.0]
        print(f"  g×{m}:  전체 {d[:,0].mean():+.1f}±{d[:,0].std(ddof=1)/np.sqrt(3):.1f}"
              f"   F행 {d[:,1].mean():+.1f}   R행 {d[:,2].mean():+.1f}")


if __name__ == "__main__":
    main()
