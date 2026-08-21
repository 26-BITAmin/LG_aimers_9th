"""§7.109 — `init_score` 를 제대로 만든다 (§7.107 의 후속).

§7.107 에서 `init_score` 단독이 fold2023 −1353.4 / fold2024 −273.6 이었다.
[너무 나쁜 결과도 의심하기] — §7.4 는 같은 조작으로 314 → 694 를 봤다. 버그다.

★ 버그 확정: **검증 시즌 기준 로짓을 game_type 별로 외삽했다.**
   F 양성률은 2022→2023 에 .709 → .473 으로 붕괴한다 (§7.21 정의 변경).
   2019~2022 F 에서 2023 F 를 외삽하면 .70 근처가 나오고 진값은 .47 —
   로짓 약 0.97 어긋난다. fold2023 이 fold2024 보다 5배 나쁜 이유가 정확히 이것이다.
   `ablate3.py:67` 은 이미 피해 갔다: 학습은 season×gt 로 레벨을 빼되 **추론 기준
   로짓은 스칼라 하나**. 게다가 마지막에 `shift_to_prior` 가 있으므로
   **스칼라는 어차피 흡수된다** → `base_va = 0` 으로 두면 된다.

두 번째 문제 — `N_ROUNDS`. init_score 가 있으면 부스팅 출발점이 달라져 320 이 맞는
값이라는 보장이 없다. LightGBM 은 `num_iteration` 으로 **한 번 학습해 여러 라운드에서
예측**할 수 있으므로 1.5배까지 학습해 두면 라운드 곡선이 공짜다.

설계  2×2 (init × center) × fold {2023, 2024},  라운드 곡선 {160,240,320,400,480}
  ★ 하네스: (init=0, center=0, k=320) 이 §7.106/§7.107 의 lgb 층
    **620.2 / 811.6** 을 재현해야 한다. 아니면 아래 숫자는 못 읽는다.
  ★ 정직한 선택: (팔, k) 를 **fold2023 에서만** 고르고 fold2024 에서 평가한다.

실행:  .venv/Scripts/python.exe -u src/initfix.py 2>&1 | tee experiments/initfix.log
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
                    SMOOTH_K_PITCHER, SOFT_LABEL_F, TARGET_COL, TRAIN_CSV,
                    USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
LGB_SEEDS = SEED_ENSEMBLE[:3]
KS = [160, 240, 320, 400, 480]
MAXR = max(KS)
EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def season_gt_init(tr, y):
    """학습 행의 시작 로짓 = logit(그 (시즌, game_type) 의 양성률).

    ⚠️ **추론 기준 로짓은 여기서 만들지 않는다.** game_type 별로 외삽하면 F 정의 변경
       구간(2022→2023)에서 0.97 로짓이 어긋난다 — §7.107 이 밟은 지뢰다.
       `shift_to_prior` 가 스칼라를 흡수하므로 base_va = 0 으로 충분하다.
    """
    ss = tr["season"].to_numpy(dtype=np.int64)
    gg = tr["game_type"].astype("object").to_numpy().astype(str)
    ug = np.unique(gg)
    k = ss * len(ug) + np.searchsorted(ug, gg)
    out = np.empty(len(k), dtype=np.float64)
    for kk in np.unique(k):
        m = k == kk
        out[m] = float(y[m].mean()) if m.sum() >= 500 else float(y.mean())
    return _logit(out)


def run(df, fold, center, init, t0):
    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    y_va = va[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                        center_asof=center)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xtr, Xva = X(tr), X(va)
    init_tr = season_gt_init(tr, y_tr) if init else None

    if SOFT_LABEL_F:
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
        assert len(Xtr) == len(y_tr) == len(w)
        assert init_tr is None or len(init_tr) == len(Xtr), "soften 정렬 깨짐"

    Z = {k: np.zeros(len(Xva)) for k in KS}
    n = 0
    for over in ENSEMBLE_VARIANTS.values():
        for sd in LGB_SEEDS:
            p = dict(FINAL_LGB_PARAMS, **over, seed=sd, bagging_seed=sd,
                     feature_fraction_seed=sd, data_random_seed=sd)
            bst = lgb.train(p, lgb.Dataset(Xtr, y_tr, weight=w, init_score=init_tr),
                            num_boost_round=MAXR)
            for k in KS:
                Z[k] += bst.predict(Xva, raw_score=True, num_iteration=k)
            n += 1
            del bst

    gt = va["game_type"].to_numpy()
    sv = va["season"].to_numpy()
    keep = ~((gt == "F") & (sv < 2023))
    out = {}
    for k in KS:
        z = Z[k] / n                      # base_va = 0 — shift_to_prior 가 흡수한다
        pv = shift_to_prior(1 / (1 + np.exp(-z[keep])), y_va[keep].mean())
        out[k] = bss(y_va[keep], pv)
    print(f"    fold {fold}  center={int(center)} init={int(init)}  "
          + "  ".join(f"k{k}:{out[k]:.1f}" for k in KS) + f"   ({time.time()-t0:.0f}s)")
    return out


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    print(f"train {df.shape}   LGBM 단독 {len(ENSEMBLE_VARIANTS)}변형 × {len(LGB_SEEDS)}시드"
          f"   소프트라벨 {'ON' if SOFT_LABEL_F else 'OFF'}   배포 N_ROUNDS={N_ROUNDS}")
    print("base_va = 0 (shift_to_prior 가 스칼라를 흡수). §7.107 의 gt별 외삽이 버그였다.\n")
    R = {}
    for fold in FOLDS:
        for init in (False, True):
            for center in (False, True):
                R[(fold, init, center)] = run(df, fold, center, init, t0)

    print(f"\n{'='*88}\n█ 라운드 곡선 (clean 행 BSS)\n{'='*88}")
    print(f"  {'fold':<7}{'init':<6}{'ctr':<5}" + "".join(f"{f'k={k}':>10}" for k in KS))
    for fold in FOLDS:
        for init in (False, True):
            for center in (False, True):
                r = R[(fold, init, center)]
                print(f"  {fold:<7}{int(init):<6}{int(center):<5}"
                      + "".join(f"{r[k]:>10.1f}" for k in KS))

    h23 = R[(2023, False, False)][N_ROUNDS]
    h24 = R[(2024, False, False)][N_ROUNDS]
    print(f"\n  ★ 하네스 (init=0,ctr=0,k={N_ROUNDS}): {h23:.1f} / {h24:.1f}"
          f"   §7.106·§7.107 lgb 층: 620.2 / 811.6")
    ok = abs(h23 - 620.2) < 3 and abs(h24 - 811.6) < 3
    print(f"     → {'✅ 통과' if ok else '❌ 실패 — 아래 숫자를 읽지 말 것'}")

    print(f"\n{'='*88}\n█ ★ 정직한 선택 — (팔, k) 를 **fold2023 에서만** 고르고 2024 에서 평가\n{'='*88}")
    base24 = R[(2024, False, False)][N_ROUNDS]
    cand = [(init, ctr) for init in (False, True) for ctr in (False, True)]
    best = max(cand, key=lambda a: max(R[(2023, a[0], a[1])].values()))
    kbest = max(KS, key=lambda k: R[(2023, best[0], best[1])][k])
    print(f"  fold2023 최적 팔 = init={int(best[0])}, center={int(best[1])}, k={kbest}"
          f"   (2023 BSS {R[(2023,)+best][kbest]:.1f})")
    print(f"  → fold2024 평가:  {R[(2024,)+best][kbest]:.1f}"
          f"   vs 배포 기준 {base24:.1f}   Δ {R[(2024,)+best][kbest]-base24:+.1f}")

    print(f"\n  참고 — fold별 각 팔의 최적 k 와 그 값 (오라클. 상한)")
    print(f"  {'fold':<7}{'init':<6}{'ctr':<5}{'최적 k':>8}{'BSS':>10}{'배포기준 대비':>14}")
    for fold in FOLDS:
        b = R[(fold, False, False)][N_ROUNDS]
        for init in (False, True):
            for center in (False, True):
                r = R[(fold, init, center)]
                k = max(KS, key=lambda k: r[k])
                print(f"  {fold:<7}{int(init):<6}{int(center):<5}{k:>8}{r[k]:>10.1f}"
                      f"{r[k]-b:>+14.1f}")


if __name__ == "__main__":
    main()
