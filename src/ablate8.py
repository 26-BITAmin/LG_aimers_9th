"""9차 실험 — Trackman 피처 부분집합 확정 (5시드).

7차에서 릴리스포인트 계열만(+14.9)이 전체 28피처(+12.4)보다 나았다. 다만 3시드라
둘의 차이는 노이즈 안이다. 여기서 5시드로 확정한다.

`tm_ext_std`(익스텐션 표준편차)가 전체 설정에서 gain 1위였는데, 익스텐션도
"릴리스 지점이 얼마나 일정한가"의 한 축이므로 릴리스 계열에 넣는 것이 맞다.

실행:  .venv/Scripts/python.exe -u src/ablate8.py
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

from ablate7 import run
from config import MODEL_DIR, TRAIN_CSV
from data import load_csv
from trackman_features import load_table

SEEDS = (42, 7, 2024, 1234, 31337)


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    S = [2019, 2020, 2021, 2022, 2023]
    full = pd.read_csv(MODEL_DIR / "trackman_pitcher.csv")

    narrow = [c for c in full.columns if "rel" in c or "disp" in c]
    cases = {
        "A) Trackman 없음":                None,
        "C) 릴리스포인트만 (7차 최고)":       full[["pitcher_id", "season"] + narrow],
        "D) 릴리스 + 익스텐션 (release)":    load_table("release"),
        "B) 전체 28피처":                  full,
    }

    print(f"{'설정':<28s} {'nf':>4s} {'평균':>8s} {'std':>6s}   5시드")
    print("-" * 88)
    res = {}
    for name, tbl in cases.items():
        v, nf, imp = run(df, tbl, S, 2024, seeds=SEEDS)
        res[name] = v
        print(f"{name:<28s} {nf:>4d} {v.mean():>8.1f} {v.std(ddof=1):>6.1f}   "
              f"{np.round(v,1)}   ({time.time()-t0:.0f}s)")

    base = res["A) Trackman 없음"]
    print()
    for name, v in res.items():
        if name.startswith("A"):
            continue
        d = v.mean() - base.mean()
        se = np.sqrt(v.var(ddof=1) / len(v) + base.var(ddof=1) / len(base))
        print(f"  {name:<28s} {d:+6.1f} BSS   ({abs(d)/max(se,1e-9):.1f}σ)")

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
