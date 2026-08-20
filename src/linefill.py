"""성공률 직선 채움 — F 레벨을 R 기울기−간격의 직선에 앉힌다 (§7.70 후속, 사용자 제안 3)

qref(오프셋)·qref2(표적)와 다른 세 번째 형태. 사용자 원안:
    "오염 구간 F 성공률을 정상 구간에서 뒤로 그은 직선으로 다시 채우자."

reverse 질량이 아니라 **성공률 레벨 자체**를 목표로 잡는다:
    F_target,s = R_s + d,   d = 정상 구간 (F 성공률 − R 성공률)   ← train에서만 fit
    h_s = 1 − F_target,s / p̄_obs,s     (y=1 행을 깎아 시즌 평균을 직선에 맞춤)

배포와의 차이: 시즌마다 용량이 다르다 — 특히 2019를 훨씬 깎는다(.626→.52).
R이 거의 선형이라 "R 기울기 차용 직선"과 동치다.

fold2023은 train에 정상 F가 없어 fit 불가 → fold2024만.

실행:  .venv/Scripts/python.exe -u src/linefill.py 2>&1 | tee experiments/linefill.log
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
from softlabel import prep_fold, train_eval, fit_h

FOLD = 2024
SEEDS = (42, 7, 2024)


def main():
    df = load_csv(TRAIN_CSV)
    t0 = time.time()
    tmask = (df.season < FOLD).to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    s = df["season"].to_numpy()
    g_ = df["game_type"].to_numpy().astype(str)

    # d = 정상 구간 (F − R) 성공률 간격, train에서만
    ds = []
    for yr in range(2023, 2025):
        mF = tmask & (s == yr) & (g_ == "F")
        mR = tmask & (s == yr) & (g_ == "R")
        if mF.any() and mR.any():
            ds.append(float(y[mF].mean() - y[mR].mean()))
    assert ds, "train에 정상 F가 없다"
    d = float(np.mean(ds))
    print(f"정상 구간 F−R 성공률 간격 d = {d:+.4f}")

    h_line = {}
    for yr in range(2019, 2023):
        mF = tmask & (s == yr) & (g_ == "F")
        mR = tmask & (s == yr) & (g_ == "R")
        tgt = float(y[mR].mean()) + d
        pbar = float(y[mF].mean())
        h_line[yr] = float(np.clip(1 - tgt / pbar, 0.0, 0.95))
        print(f"  {yr}: 관측 {pbar:.3f} → 목표 {tgt:.3f}   h {h_line[yr]:.3f}")

    h_dep = fit_h(df, tmask)
    print(f"  배포 h: " + "  ".join(f"{k}:{v:.3f}" for k, v in sorted(h_dep.items())))

    tr, va, w, X = prep_fold(df, FOLD)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    va_f = va["game_type"].to_numpy().astype(str) == "F"
    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    Xtr = X(tr)
    s_tr = tr["season"].to_numpy()
    g_tr = tr["game_type"].to_numpy().astype(str)

    res = {}
    for name, h in [("배포(reverse 질량)", h_dep), ("직선 채움(레벨)", h_line)]:
        h_row = np.zeros(len(tr))
        for yr, v in h.items():
            h_row[(s_tr == yr) & (g_tr == "F")] = v
        dup = (h_row > 0) & (ytr == 1)
        w_soft = np.asarray(w, dtype=np.float64).copy()
        w_soft[dup] *= (1 - h_row[dup])
        X2 = pd.concat([Xtr, Xtr[dup]], ignore_index=True)
        y2 = np.concatenate([ytr, np.zeros(int(dup.sum()))])
        w2 = np.concatenate([w_soft, np.asarray(w)[dup] * h_row[dup]])
        r = train_eval(X2, y2, w2, X(va), yva, va_f, SEEDS)
        res[name] = r
        print(f"\n  {name:<14} BSS {r[:,0].mean():>8.1f} ± {r[:,0].std(ddof=1):.1f}"
              f"   F행 {r[:,1].mean():>7.1f}   R행 {r[:,2].mean():>7.1f}"
              f"   ({time.time()-t0:.0f}s)")

    dd = res["직선 채움(레벨)"] - res["배포(reverse 질량)"]
    print(f"\n쌍체 Δ(직선−배포):  전체 {dd[:,0].mean():+.1f}±{dd[:,0].std(ddof=1)/np.sqrt(3):.1f}"
          f"   F행 {dd[:,1].mean():+.1f}   R행 {dd[:,2].mean():+.1f}")


if __name__ == "__main__":
    main()
