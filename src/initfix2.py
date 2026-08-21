"""§7.109 후속 — `init_score` × 소프트 라벨의 **이중교정** 가설 검정.

§7.109 에서 `base_va` 버그를 고쳤는데도 fold2023 은 `init_score` 를 켜면 무너진다
(620.2 → −106.9). fold2024 는 훨씬 덜 다친다 (811.6 → 776.4). 왜 fold 마다 다른가.

가설 H_이중교정
    fold2023 의 학습셋은 2019~2022 = **전부 F 오염 구간**이다. 그런데 이 파이프라인엔
    **소프트 라벨**이 이미 있다 — 오염된 F 행의 양성을 확률적으로 음성으로 되돌려
    **그 행들의 레벨을 낮추는** 장치다. `init_score` 는 같은 행들의 레벨을 **또 한 번**
    뺀다 (그 시즌 F 의 관측 양성률 .709 를 기준으로).  → 같은 것을 두 번 교정한다.
    fold2024 는 학습셋에 깨끗한 2023 F 가 있어 오염 비중이 낮으니 덜 다쳐야 한다.

검정 = **상호작용**이다. 소프트 라벨의 주효과가 크므로(fold2023 은 끄면 620 → 131)
`init` 의 주효과만 봐서는 안 된다. 2×2 를 다 채우고 Δ_init 을 soft 별로 비교한다.

    Δ_init(soft=ON)  ≪  Δ_init(soft=OFF)   →  H_이중교정 지지
    둘이 비슷                              →  기각. 다른 이유다.

⚠️ soft=OFF 는 fold2023 을 그 자체로 무너뜨린다. **절대값이 아니라 Δ_init 만 읽는다.**

실행:  .venv/Scripts/python.exe -u src/initfix2.py 2>&1 | tee experiments/initfix2.log
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import lightgbm as lgb
import numpy as np

import soften
from config import (CONTAM_AXES, DROP_FEATURES, ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS,
                    N_ROUNDS, RECENCY_HALF_LIFE, SEED_ENSEMBLE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from initfix import season_gt_init
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
LGB_SEEDS = SEED_ENSEMBLE[:3]


def run(df, fold, init, soft, t0):
    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    y_va = va[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xtr, Xva = X(tr), X(va)
    init_tr = season_gt_init(tr, y_tr) if init else None

    if soft:
        h = soften.fit_h(tr)
        s_ = tr["season"].to_numpy()
        g_ = tr["game_type"].to_numpy().astype(str)
        h_row = np.zeros(len(tr))
        for yr, v in h.items():
            h_row[(s_ == yr) & (g_ == "F")] = v
        dup = (h_row > 0) & (y_tr == 1)
        Xtr, y_tr, w, _, _ = soften.soften(tr, Xtr, y_tr, w)
        if init_tr is not None:
            init_tr = np.concatenate([init_tr, init_tr[dup]])
        assert init_tr is None or len(init_tr) == len(Xtr), "soften 정렬 깨짐"

    z, n = np.zeros(len(Xva)), 0
    for over in ENSEMBLE_VARIANTS.values():
        for sd in LGB_SEEDS:
            p = dict(FINAL_LGB_PARAMS, **over, seed=sd, bagging_seed=sd,
                     feature_fraction_seed=sd, data_random_seed=sd)
            bst = lgb.train(p, lgb.Dataset(Xtr, y_tr, weight=w, init_score=init_tr),
                            num_boost_round=N_ROUNDS)
            z += bst.predict(Xva, raw_score=True)
            n += 1
            del bst
    z /= n                                  # base_va = 0 (shift_to_prior 가 흡수)

    gt = va["game_type"].to_numpy()
    sv = va["season"].to_numpy()
    keep = ~((gt == "F") & (sv < 2023))
    pv = shift_to_prior(1 / (1 + np.exp(-z[keep])), y_va[keep].mean())
    sc = bss(y_va[keep], pv)
    print(f"    fold {fold}  init={int(init)} soft={int(soft)}  BSS {sc:9.1f}"
          f"   ({time.time()-t0:.0f}s)")
    return sc


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    print(f"train {df.shape}   center_asof=OFF 고정, k={N_ROUNDS} 고정")
    print("소프트 라벨의 **주효과가 크므로** 절대값이 아니라 Δ_init 만 읽는다.\n")
    R = {}
    for fold in FOLDS:
        for soft in (True, False):
            for init in (False, True):
                R[(fold, init, soft)] = run(df, fold, init, soft, t0)

    print(f"\n{'='*78}\n█ 2×2 — init × soft (clean 행 BSS)\n{'='*78}")
    print(f"  {'fold':<8}{'soft':<7}{'init=0':>11}{'init=1':>11}{'Δ_init':>12}")
    D = {}
    for fold in FOLDS:
        for soft in (True, False):
            a, b = R[(fold, False, soft)], R[(fold, True, soft)]
            D[(fold, soft)] = b - a
            print(f"  {fold:<8}{int(soft):<7}{a:>11.1f}{b:>11.1f}{b-a:>+12.1f}")

    print(f"\n{'='*78}\n█ ★ 상호작용 — Δ_init 이 soft 를 끄면 얼마나 회복되나\n{'='*78}")
    print(f"  {'fold':<8}{'Δ_init(soft=ON)':>18}{'Δ_init(soft=OFF)':>19}{'회복분':>12}")
    for fold in FOLDS:
        on, off = D[(fold, True)], D[(fold, False)]
        print(f"  {fold:<8}{on:>+18.1f}{off:>+19.1f}{off-on:>+12.1f}")
    print("\n  회복분이 크게 양수면 H_이중교정 지지 — 소프트 라벨과 init_score 가 같은")
    print("  오염 레벨을 두 번 뺀 것이다. 0 근처면 기각이고 다른 이유를 찾아야 한다.")
    print("  fold2023 > fold2024 여야 한다 (2023 학습셋이 오염 비중이 훨씬 높다).")


if __name__ == "__main__":
    main()
