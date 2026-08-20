"""10차 실험 — Trackman 피처의 LB 회귀를 오프라인에서 잡을 수 있었는가.

사실:
    fold2(valid 2024) 홀드아웃  Trackman +10.9 BSS (3.2σ)
    실제 LB(2025)              Trackman -13.6

3.2σ짜리 개선이 반대로 뒤집혔다. 검증 설계의 문제다. 두 가지를 확인한다.

  (1) fold1(valid 2023)에서도 Trackman이 이득인가?
      한쪽에서만 이득이면 "2024 특유의 것"이라는 신호였고, 미리 잡을 수 있었다.
  (2) 누적 피처의 노후화 — 2025용 피처는 2019~2024 **누적**이라 오래된 시즌이 섞인다.
      직전 시즌만 쓰는 버전이 더 나은가? (2024 홀드아웃에서 비교)

실행:  .venv/Scripts/python.exe -u src/ablate9.py
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
    tbl = load_table("release")

    folds = [
        ("fold1  valid=2023", [2019, 2020, 2021, 2022], 2023),
        ("fold2  valid=2024", [2019, 2020, 2021, 2022, 2023], 2024),
    ]

    print(f"{'fold':<20s} {'설정':<14s} {'평균':>8s} {'std':>6s}   5시드")
    print("-" * 84)
    out = {}
    for fname, seasons, valid in folds:
        for label, t in (("Trackman 없음", None), ("Trackman release", tbl)):
            v, nf, _ = run(df, t, seasons, valid, seeds=SEEDS)
            out[(fname, label)] = v
            print(f"{fname:<20s} {label:<14s} {v.mean():>8.1f} {v.std(ddof=1):>6.1f}   "
                  f"{np.round(v,1)}   ({time.time()-t0:.0f}s)")

    print("\n=== fold별 Trackman 효과 ===")
    for fname, _, _ in folds:
        a = out[(fname, "Trackman 없음")]
        b = out[(fname, "Trackman release")]
        d = b.mean() - a.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        print(f"  {fname}: {d:+7.1f} BSS  ({abs(d)/max(se,1e-9):.1f}σ)")
    print("\n  → 두 fold의 부호가 다르면, 단일 시즌 홀드아웃으로 채택한 것이 실수였다는 뜻이다.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
