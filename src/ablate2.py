"""3차 실험 — 최근 시즌 편중을 어디까지 밀 것인가 + 시드 노이즈 측정.

1·2차에서 "오래된 데이터를 덜 쓸수록 좋다"가 단조로 나왔다.
여기서는 (a) 그 끝단을 찾고, (b) **시드 노이즈 폭**을 재서 위 차이가 진짜인지 판정한다.
BSS 1점 = Brier 2.5e-6. 시드 간 편차가 ±50점이면 위 결과 절반은 노이즈다.

실행:  .venv/Scripts/python.exe -u src/ablate2.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from ablate import BASE_PARAMS, evaluate
from config import TARGET_COL, TRAIN_CSV
from data import load_csv


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    season_rate = df.groupby("season")[TARGET_COL].mean().to_dict()

    lean = dict(BASE_PARAMS, num_leaves=31, min_data_in_leaf=2000, lambda_l2=30.0)

    ALL = [2019, 2020, 2021, 2022, 2023]
    NG = ["game_type"]           # 2022→2023에 정의가 바뀐 컬럼. 2차에서 제거 시 +171 BSS.

    # 이미 측정 완료 (fold2 BSS oracle):
    #   G hl=1.5 -gt 663.5 / N hl=1.0 -gt 691.1 / O hl=0.7 -gt 702.1
    #   P hl=0.5 -gt 716.4 / Q hl=0.7 2021-23 -gt 682.8
    # → hl은 짧을수록 좋고, 오래된 시즌을 "낮은 가중으로 남기는" 편이 "버리는" 편보다 낫다.
    variants = {
        "P  hl=0.5 -gt (기준)":    dict(train_seasons=ALL, half_life=0.5, drop_cols=NG),
        "P2 hl=0.35 -gt":          dict(train_seasons=ALL, half_life=0.35, drop_cols=NG),
        "R  2023 only -gt":        dict(train_seasons=[2023], half_life=None, drop_cols=NG),
        # ID 추가 (2차에서 game_type 있는 채로 +76이었다)
        "V  hl=0.5 -gt +ids":      dict(train_seasons=ALL, half_life=0.5, drop_cols=NG,
                                        add_ids=True),
        "W  hl=0.5 -gt +ids lean": dict(train_seasons=ALL, half_life=0.5, drop_cols=NG,
                                        add_ids=True, params=lean),
        "X  hl=0.5 -gt +ids k=50": dict(train_seasons=ALL, half_life=0.5, drop_cols=NG,
                                        add_ids=True, k=50.0),
        "X2 hl=0.5 -gt k=50":      dict(train_seasons=ALL, half_life=0.5, drop_cols=NG, k=50.0),
        "L2 hl=0.5 -gt lean":      dict(train_seasons=ALL, half_life=0.5, drop_cols=NG,
                                        params=lean),
    }

    print(f"{'variant':<26s} {'iter':>5s} {'BSS raw':>9s} {'BSS oracle':>11s}")
    print("-" * 56)
    for name, cfg in variants.items():
        cfg.setdefault("params", BASE_PARAMS)
        r = evaluate(df, season_rate, 2024, **cfg)
        print(f"{name:<26s} {r['iter']:>5d} {r['bss_raw']:>9.1f} {r['bss_oracle']:>11.1f}"
              f"   ({time.time()-t0:.0f}s)")

    # ── 시드 노이즈 ──────────────────────────────────────────
    print("\n--- 시드 노이즈 (같은 설정, seed만 변경) ---")
    for label, cfg in [("hl=0.5 -gt", dict(train_seasons=ALL, half_life=0.5, drop_cols=NG)),
                       ("hl=0.5 -gt +ids", dict(train_seasons=ALL, half_life=0.5,
                                                drop_cols=NG, add_ids=True))]:
        vals = []
        for seed in (42, 7, 2024):
            r = evaluate(df, season_rate, 2024,
                         params=dict(BASE_PARAMS, seed=seed, bagging_seed=seed,
                                     feature_fraction_seed=seed),
                         **cfg)
            vals.append(r["bss_oracle"])
        v = np.array(vals)
        print(f"  {label:<20s} oracle BSS = {np.round(v,1)}  "
              f"mean={v.mean():.1f}  std={v.std(ddof=1):.1f}")

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
