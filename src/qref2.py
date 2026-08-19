"""표적 h 보정 — 추가 절삭을 F전용 투수 행에만 (§7.70 후속, 사용자 제안 2)

균일 오프셋(qref.py)은 기각됐다: 정상 구간 갭 +.048의 상당분이 구성 효과
(F전용 주변부 투수)인데, 시즌 상수 h는 그걸 둘다 뛰는 투수 행(83.5%)에도
균일하게 발라 과잉 절삭이 됐다 (m=0.5 전체 −3.4, m=1.0 −11.0).

사용자 제안: **그러면 둘다 그룹은 제외하고, F전용 그룹에만 얹으면 되지 않나.**

설계 (전부 fold train에서만 fit — fold2024만 가능, 2023은 정상 F 앵커 없음)
    그룹  = train에 R행이 있는 투수(dual) vs 없는 투수(fonly)
    dual 행:  h_s = 배포값 (q_s/p̄_s)
    fonly 행: q'_s = max(0, R_s + g_f − Fobs_fonly,s),  h'_s = q'_s/p̄_fonly,s
              g_f = 정상 구간 fonly 행의 reverse율 − 같은 시즌 R
    즉 fonly의 기준을 "R + fonly 자체의 초과분"으로 잡는다.

실행:  .venv/Scripts/python.exe -u src/qref2.py 2>&1 | tee experiments/qref2.log
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
from softlabel import prep_fold, train_eval, fit_h

FOLD = 2024
SEEDS = (42, 7, 2024)


def main():
    df = load_csv(TRAIN_CSV)
    t0 = time.time()
    tmask = (df.season < FOLD).to_numpy()

    lab, _, _ = recover(df)
    rev = np.where(np.isnan(lab["reverse"]), 0.0, np.round(lab["reverse"]))
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    s = df["season"].to_numpy()
    g_ = df["game_type"].to_numpy().astype(str)
    pid = df["pitcher_id"].to_numpy()

    # 그룹 분류 — train에서만
    has_r = set(pid[tmask & (g_ == "R")])
    fonly_row = tmask & (g_ == "F") & ~pd.Series(pid).isin(has_r).to_numpy()
    dual_row = tmask & (g_ == "F") & ~fonly_row

    # ── 진단: 정상 구간(2023, train 내) 그룹별 reverse율 ──
    mR23 = tmask & (s == 2023) & (g_ == "R")
    mD23 = dual_row & (s == 2023)
    mO23 = fonly_row & (s == 2023)
    r_R, r_D, r_O = rev[mR23].mean(), rev[mD23].mean(), rev[mO23].mean()
    print(f"정상 구간(2023) reverse율:  R {r_R:.4f}   둘다-F {r_D:.4f} "
          f"(n={mD23.sum():,})   F전용 {r_O:.4f} (n={mO23.sum():,})")
    print(f"  → 갭 vs R:  둘다 {r_D-r_R:+.4f}   F전용 {r_O-r_R:+.4f}")
    g_f = float(r_O - r_R)

    # ── 오염 구간 그룹별 관측치와 표적 h ──
    h_base = fit_h(df, tmask)                      # 배포값 (풀드)
    h_fonly = {}
    for yr in range(2019, 2023):
        mR = tmask & (s == yr) & (g_ == "R")
        mO = fonly_row & (s == yr)
        if mO.sum() < 500:
            h_fonly[yr] = h_base[yr]
            continue
        q = max(0.0, float(rev[mR].mean()) + g_f - float(rev[mO].mean()))
        h_fonly[yr] = min(q / max(float(y[mO].mean()), 1e-9), 0.95)
        print(f"  {yr}: fonly n={mO.sum():>6,}  관측rev {rev[mO].mean():.3f}  "
              f"관측suc {y[mO].mean():.3f}  h {h_base[yr]:.3f} → {h_fonly[yr]:.3f}")

    tr, va, w, X = prep_fold(df, FOLD)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    va_f = va["game_type"].to_numpy().astype(str) == "F"
    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    Xtr = X(tr)
    s_tr = tr["season"].to_numpy()
    g_tr = tr["game_type"].to_numpy().astype(str)
    fonly_tr = ~pd.Series(tr["pitcher_id"].to_numpy()).isin(has_r).to_numpy()

    res = {}
    for name, use_target in [("배포(균일)", False), ("표적(F전용만)", True)]:
        h_row = np.zeros(len(tr))
        for yr in range(2019, 2023):
            mF = (s_tr == yr) & (g_tr == "F")
            h_row[mF] = h_base[yr]
            if use_target:
                h_row[mF & fonly_tr] = h_fonly[yr]
        dup = (h_row > 0) & (ytr == 1)
        w_soft = np.asarray(w, dtype=np.float64).copy()
        w_soft[dup] *= (1 - h_row[dup])
        X2 = pd.concat([Xtr, Xtr[dup]], ignore_index=True)
        y2 = np.concatenate([ytr, np.zeros(int(dup.sum()))])
        w2 = np.concatenate([w_soft, np.asarray(w)[dup] * h_row[dup]])
        r = train_eval(X2, y2, w2, X(va), yva, va_f, SEEDS)
        res[name] = r
        print(f"\n  {name:<10} BSS {r[:,0].mean():>8.1f} ± {r[:,0].std(ddof=1):.1f}"
              f"   F행 {r[:,1].mean():>7.1f}   R행 {r[:,2].mean():>7.1f}"
              f"   ({time.time()-t0:.0f}s)")

    d = res["표적(F전용만)"] - res["배포(균일)"]
    print(f"\n쌍체 Δ(표적−배포):  전체 {d[:,0].mean():+.1f}±{d[:,0].std(ddof=1)/np.sqrt(3):.1f}"
          f"   F행 {d[:,1].mean():+.1f}   R행 {d[:,2].mean():+.1f}")


if __name__ == "__main__":
    main()
