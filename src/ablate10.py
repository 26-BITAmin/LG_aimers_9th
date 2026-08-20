"""11차 실험 — 모델 다양성 앙상블.

같은 피처 위에서 LightGBM / XGBoost / CatBoost를 돌리고 **로짓 평균**한다.
피처를 늘리는 게 아니므로 새 분포 의존성이 생기지 않는다 — Trackman의 실패 모드를 피한다.

레짐이 같은 fold(2023, 2024)에서만 판정한다 (MODEL.md §2).
채택 조건: 부호 일치 + fold 간 크기 3배 이내 + 노이즈 초과.

실행:  .venv/Scripts/python.exe -u src/ablate10.py
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

from config import (DROP_FEATURES, FINAL_LGB_PARAMS, N_ROUNDS, RECENCY_HALF_LIFE,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV,
                    USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from models import fit_cat, fit_lgbm, fit_xgb
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
SEEDS = (42, 7, 2024)


def fold_logits(df, valid_season, seeds=SEEDS):
    """fold별로 세 패밀리의 로짓을 각각 구해 반환. {family: (n_seeds, n_rows)}"""
    tr = df[df.season < valid_season]
    va = df[df.season == valid_season]
    ref = int(tr.season.max())
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=ref)
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xtr, Xva = X(tr), X(va)
    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)

    out = {}
    t0 = time.time()
    for name, fn, kw in (("lgbm", fit_lgbm, dict(params=FINAL_LGB_PARAMS, rounds=N_ROUNDS)),
                         ("xgb", fit_xgb, {}),
                         ("cat", fit_cat, {})):
        Z = []
        for s in seeds:
            Z.append(fn(Xtr, ytr, w, s, **kw)(Xva))
        out[name] = np.array(Z)
        z = out[name].mean(axis=0)
        p = shift_to_prior(1 / (1 + np.exp(-z)), yva.mean())
        print(f"    {name:<5s} 단독 BSS={bss(yva, p):>8.1f}   ({time.time()-t0:.0f}s)")
    return out, yva


def score(logit, y):
    return bss(y, shift_to_prior(1 / (1 + np.exp(-logit)), y.mean()))


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)

    combos = {
        "lgbm (기준)": ["lgbm"],
        "xgb": ["xgb"],
        "cat": ["cat"],
        "lgbm+cat": ["lgbm", "cat"],
        "lgbm+xgb": ["lgbm", "xgb"],
        "lgbm+xgb+cat": ["lgbm", "xgb", "cat"],
    }
    res = {k: {} for k in combos}

    for f in FOLDS:
        print(f"\n=== valid={f} ===")
        Z, y = fold_logits(df, f)
        for name, fams in combos.items():
            z = np.mean([Z[m].mean(axis=0) for m in fams], axis=0)
            res[name][f] = score(z, y)
        print(f"  조합 점수: " + "  ".join(f"{k}={res[k][f]:.1f}" for k in combos))

    print(f"\n{'조합':<16s} " + " ".join(f"{f:>9d}" for f in FOLDS) + f" {'평균':>9s}")
    print("-" * 50)
    for k in combos:
        v = np.array([res[k][f] for f in FOLDS])
        print(f"{k:<16s} " + " ".join(f"{res[k][f]:>9.1f}" for f in FOLDS)
              + f" {v.mean():>9.1f}")

    base = np.array([res["lgbm (기준)"][f] for f in FOLDS])
    print("\n=== 기준(lgbm) 대비 ===")
    for k in combos:
        if k.startswith("lgbm ("):
            continue
        d = np.array([res[k][f] for f in FOLDS]) - base
        same = bool(np.all(d > 0) or np.all(d < 0))
        spread = np.abs(d).max() / max(np.abs(d).min(), 1e-9) if same else np.inf
        print(f"  {k:<16s} fold별 {np.round(d,1)}  평균 {d.mean():+7.1f}"
              f"   부호일치 {'O' if same else 'X'}  크기비 {spread:.1f}배")

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
