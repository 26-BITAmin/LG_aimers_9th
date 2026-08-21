"""§7.110 — 이중교정을 피한 `init_score`: 레벨을 **소프트 라벨 적용 후** 데이터에서 낸다.

`initfix2.py` 가 확정한 것: 소프트 라벨과 `init_score` 는 **대체재**다. 둘 다 "오염된 F
행의 레벨을 낮추는" 같은 일을 하고, 같이 쓰면 두 번 뺀다.

    fold      Δ_init(soft=ON)   Δ_init(soft=OFF)   회복분
    2023           −727.1            +479.4       +1206.5
    2024            −35.3             +31.5          +66.8

원칙적 해법: `init_score` 의 레벨을 **원자료가 아니라 soften 적용 후**에서 계산한다.
소프트 라벨은 오염 F 행의 양성을 확률적으로 음성으로 되돌리므로, 그 뒤의
(시즌 × game_type) **가중** 양성률이 곧 **탈오염 레벨**이다. 그걸 빼면 두 번 빼지 않는다.

팔
    A  soft=1, init=0                     ← 배포. 620.2 / 811.6
    B  soft=1, init=1(**softened 레벨**)   ← 이번 후보
    C  soft=1, init=1(원자료 레벨)         ← 이중교정 재현 (대조). −106.9 / 776.4
    D  soft=0, init=1(원자료 레벨)         ← 대체재 팔 (대조). 610.7 / 825.7

★ 하네스: A 가 620.2 / 811.6, C 가 −106.9 / 776.4, D 가 610.7 / 825.7 를 재현해야 한다.
   세 칸이 다 맞아야 B 를 읽는다.
★ 채택 조건 (§2): B − A 가 **두 fold 모두 양수** + 크기비가 상식 범위.

실행:  .venv/Scripts/python.exe -u src/initfix3.py 2>&1 | tee experiments/initfix3.log
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
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
LGB_SEEDS = SEED_ENSEMBLE[:3]
EPS = 1e-6
ARMS = [("A 배포", True, None), ("B softened레벨", True, "post"),
        ("C 원자료레벨", True, "raw"), ("D soft끔", False, "raw")]


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def gt_key(ss, gg):
    ug = np.unique(gg)
    return ss * len(ug) + np.searchsorted(ug, gg), ug


def level_init(ss, gg, y, w):
    """(시즌 × game_type) **가중** 양성률 → 로짓. w 를 넘기면 soften 후 레벨이 된다."""
    k, _ = gt_key(ss, gg)
    out = np.empty(len(k), dtype=np.float64)
    gm = float(np.average(y, weights=w))
    for kk in np.unique(k):
        m = k == kk
        out[m] = float(np.average(y[m], weights=w[m])) if m.sum() >= 500 else gm
    return _logit(out)


def run(df, fold, soft, mode, t0, lab):
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
    ss = tr["season"].to_numpy(dtype=np.int64)
    gg = tr["game_type"].astype("object").to_numpy().astype(str)

    # 원자료 레벨 — soften 전에 계산한다 (C·D 팔)
    init_raw = level_init(ss, gg, y_tr, w) if mode == "raw" else None

    if soft:
        h = soften.fit_h(tr)
        h_row = np.zeros(len(tr))
        for yr, v in h.items():
            h_row[(ss == yr) & (gg == "F")] = v
        dup = (h_row > 0) & (y_tr == 1)
        Xtr, y2, w2, _, _ = soften.soften(tr, Xtr, y_tr, w)
        ss2 = np.concatenate([ss, ss[dup]])
        gg2 = np.concatenate([gg, gg[dup]])
        if mode == "raw":
            init_tr = np.concatenate([init_raw, init_raw[dup]])
        elif mode == "post":
            # ★ 여기가 이번 실험이다 — **soften 후** 가중 양성률이 탈오염 레벨이다
            init_tr = level_init(ss2, gg2, y2, w2)
        else:
            init_tr = None
        y_tr, w = y2, w2
    else:
        init_tr = init_raw

    assert init_tr is None or len(init_tr) == len(Xtr), "정렬 깨짐"

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
    print(f"    fold {fold}  {lab:<16} BSS {sc:9.1f}   ({time.time()-t0:.0f}s)")
    return sc


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    print(f"train {df.shape}   center_asof=OFF, k={N_ROUNDS} 고정\n")
    R = {}
    for fold in FOLDS:
        for lab, soft, mode in ARMS:
            R[(fold, lab)] = run(df, fold, soft, mode, t0, lab)

    print(f"\n{'='*80}\n█ 판정 (clean 행 BSS)\n{'='*80}")
    print(f"  {'팔':<18}" + "".join(f"{f'fold {f}':>12}" for f in FOLDS)
          + "".join(f"{f'Δ vs A ({f})':>15}" for f in FOLDS))
    for lab, _, _ in ARMS:
        print(f"  {lab:<18}" + "".join(f"{R[(f,lab)]:>12.1f}" for f in FOLDS)
              + "".join(f"{R[(f,lab)]-R[(f,'A 배포')]:>+15.1f}" for f in FOLDS))

    exp = {"A 배포": (620.2, 811.6), "C 원자료레벨": (-106.9, 776.4), "D soft끔": (610.7, 825.7)}
    print(f"\n  ★ 하네스 — 알려진 세 팔 재현")
    ok = True
    for lab, (e23, e24) in exp.items():
        d23, d24 = R[(2023, lab)] - e23, R[(2024, lab)] - e24
        good = abs(d23) < 3 and abs(d24) < 3
        ok &= good
        print(f"     {lab:<16} {R[(2023,lab)]:>8.1f}/{R[(2024,lab)]:>7.1f}"
              f"   기대 {e23:>7.1f}/{e24:>7.1f}   {'✅' if good else '❌'}")
    print(f"     → {'통과. B 를 읽는다.' if ok else '실패 — B 를 읽지 말 것.'}")

    b = [R[(f, "B softened레벨")] - R[(f, "A 배포")] for f in FOLDS]
    print(f"\n  ★ B − A: " + "  ".join(f"{f}:{v:+.1f}" for f, v in zip(FOLDS, b)))
    if all(v > 0 for v in b):
        r = max(abs(x) for x in b) / max(min(abs(x) for x in b), 1e-9)
        print(f"     두 fold 모두 양수 ✅  크기비 {r:.2f}  → §2 조건 ①② 확인 후 후보")
    else:
        print(f"     부호 엇갈림 또는 음수 ❌  → 기각")


if __name__ == "__main__":
    main()
