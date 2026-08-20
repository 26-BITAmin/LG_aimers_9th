"""6차 실험 — 최종 후보 결정.

시드 노이즈 std ≈ 4~9 BSS로 측정됐다. 그보다 작은 차이는 단일 실행으로 판정할 수 없다.
따라서 여기서는 **최종 후보만 3시드 평균**으로 비교한다.

두 갈래를 합친다:
  (1) 용량 축소 (leaves/min_data/l2)        — 3차에서 lean이 +12
  (2) init_score로 (시즌 x game_type) 레벨 제거 — 4차에서 uniform 314 → 694

실행:  .venv/Scripts/python.exe -u src/ablate5.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from ablate import BASE_PARAMS, evaluate
from ablate3 import evaluate_init
from config import TARGET_COL, TRAIN_CSV
from data import load_csv

ALL = [2019, 2020, 2021, 2022, 2023]
NG = ["game_type"]
SEEDS = (42, 7, 2024)


def P(**kw):
    return dict(BASE_PARAMS, **kw)


LEAN = P(num_leaves=31, min_data_in_leaf=2000, lambda_l2=30.0)


def seeded(fn, df, season_rate, seeds=SEEDS, **cfg):
    """같은 설정을 여러 시드로 돌려 평균/표준편차를 낸다."""
    base_params = cfg.pop("params", BASE_PARAMS)
    vals = []
    for s in seeds:
        r = fn(df, season_rate, 2024,
               params=dict(base_params, seed=s, bagging_seed=s,
                           feature_fraction_seed=s, data_random_seed=s),
               **cfg)
        vals.append(r["bss_oracle"])
    v = np.array(vals)
    return v.mean(), v.std(ddof=1), v


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    season_rate = df.groupby("season")[TARGET_COL].mean().to_dict()

    # ── 1단계: 용량 스윕 (단일 시드, 대략적 방향 탐색) ──────────
    print("=== 1단계: 용량 스윕  (hl=0.5, -game_type, +ids, 단일 시드) ===")
    FIXED = dict(train_seasons=ALL, half_life=0.5, drop_cols=NG, add_ids=True)
    cap = {
        "63/1000/l2=10 (base)": P(),
        "31/2000/l2=30 (lean)": LEAN,
        "15/4000/l2=50":        P(num_leaves=15, min_data_in_leaf=4000, lambda_l2=50.0),
        "15/8000/l2=100":       P(num_leaves=15, min_data_in_leaf=8000, lambda_l2=100.0),
        "31/2000/l2=200":       P(num_leaves=31, min_data_in_leaf=2000, lambda_l2=200.0),
        "31/2000 lr=0.015":     P(num_leaves=31, min_data_in_leaf=2000, lambda_l2=30.0,
                                  learning_rate=0.015),
        "31/2000 ff=0.6":       P(num_leaves=31, min_data_in_leaf=2000, lambda_l2=30.0,
                                  feature_fraction=0.6),
    }
    cap_scores = {}
    for name, params in cap.items():
        r = evaluate(df, season_rate, 2024, params=params, **FIXED)
        cap_scores[name] = r["bss_oracle"]
        print(f"  {name:<24s} iter={r['iter']:>4d}  BSS oracle={r['bss_oracle']:>7.1f}"
              f"   ({time.time()-t0:.0f}s)")
    best_cap = max(cap_scores, key=cap_scores.get)
    print(f"  → 최고: {best_cap}")

    # ── 2단계: init_score 조합 (단일 시드) ────────────────────
    print("\n=== 2단계: init_score(season x game_type) 조합 ===")
    init_cfgs = {
        "init + hl=0.5 +ids":        dict(train_seasons=ALL, half_life=0.5,
                                          base_by="season_gt", add_ids=True),
        "init + hl=0.5 +ids lean":   dict(train_seasons=ALL, half_life=0.5,
                                          base_by="season_gt", add_ids=True, params=LEAN),
        "init + hl=0.7 +ids":        dict(train_seasons=ALL, half_life=0.7,
                                          base_by="season_gt", add_ids=True),
        "init + hl=0.5 +ids -gt":    dict(train_seasons=ALL, half_life=0.5,
                                          base_by="season_gt", add_ids=True, drop_cols=NG),
    }
    init_scores = {}
    for name, cfg in init_cfgs.items():
        r = evaluate_init(df, season_rate, 2024, **cfg)
        init_scores[name] = r["bss_oracle"]
        print(f"  {name:<26s} iter={r['iter']:>4d}  BSS oracle={r['bss_oracle']:>7.1f}"
              f"   ({time.time()-t0:.0f}s)")

    # ── 3단계: 최종 후보 3시드 평균 ───────────────────────────
    print("\n=== 3단계: 최종 후보 3시드 평균 (이 숫자로만 채택 결정) ===")
    finalists = [
        ("A) -gt, hl=0.5, +ids, best_cap", evaluate,
         dict(train_seasons=ALL, half_life=0.5, drop_cols=NG, add_ids=True,
              params=cap[best_cap])),
        ("B) init, hl=0.5, +ids", evaluate_init,
         dict(train_seasons=ALL, half_life=0.5, base_by="season_gt", add_ids=True)),
        ("C) init, hl=0.5, +ids, lean", evaluate_init,
         dict(train_seasons=ALL, half_life=0.5, base_by="season_gt", add_ids=True,
              params=LEAN)),
    ]
    for name, fn, cfg in finalists:
        m, s, v = seeded(fn, df, season_rate, **cfg)
        print(f"  {name:<32s} {np.round(v,1)}  mean={m:7.1f}  std={s:5.1f}"
              f"   ({time.time()-t0:.0f}s)")

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
