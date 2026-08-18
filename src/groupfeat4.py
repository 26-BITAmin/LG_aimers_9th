"""§7.69 후속 3 — 행 단위 game_type δ (δ_F)의 크기 측정.

groupfeat3의 진단: fold2024 잔여 편향은 오염 노출(D/m)과 무관한
**F 행 레벨 ~−0.018**이다. game_type이 DROP_FEATURES라 모델이 표현 불가.

여기서 재는 것
    δ_F = logit(ȳ) − logit(p̄0)  (F 행에서, fold별)
    · fold2024 δ로 fold2024 평가 = in-fold oracle (상한)
    · fold2023 δ로 fold2024 평가 = 오염 fold에서 fit하면 왜 안 되는지 실증
    β 스윕 포함. row-level 조회라 제출 규정 문제 없음 (test 행 독립).

실행:  .venv/Scripts/python.exe -u src/groupfeat4.py 2>&1 | tee experiments/groupfeat4.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import asofdelta
import ctxfit
import features as F
import foldcache
from config import (ASOF_AXES, ASOF_STRENGTH, CONTAM_AXES, CTX_AXES, CTX_HALF_LIFE,
                    CTX_K, MLP_ALPHA, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
BETAS = [0.25, 0.5, 0.75, 1.0]
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)

    P, Y, GTF = {}, {}, {}
    for fold in FOLDS:
        tr_m, va = s < fold, s == fold
        tr, d = df[tr_m], df[va].copy()
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ref = int(s[tr_m].max())
        ctx = ctxfit.fit_ctx(tr, TARGET_COL, CTX_AXES, CTX_K,
                             weights=recency_weight(s[tr_m], CTX_HALF_LIFE, ref_season=ref))
        art = F.fit_artifacts(tr, ctx=ctx, contam_axes=CONTAM_AXES)
        art["asof"] = asofdelta.fit_asof_axes(tr, ASOF_AXES, ASOF_STRENGTH,
                                              ct=art.get("contam"))
        adj, _ = F.logit_adjust(d, art)
        P[fold] = shift_to_prior(1.0 / (1.0 + np.exp(-(z + adj))), yv.mean())
        Y[fold] = yv
        GTF[fold] = d["game_type"].astype("object").to_numpy() == "F"

    dF = {}
    for fold in FOLDS:
        m = GTF[fold]
        dF[fold] = float(logit(Y[fold][m].mean()) - logit(P[fold][m].mean()))
        print(f"fold {fold}  δ_F = {dF[fold]:+.4f}  (F행 {m.sum():,} = "
              f"{m.mean()*100:.1f}%,  잔차 {Y[fold][m].mean() - P[fold][m].mean():+.5f})")

    base = bss(Y[2024], P[2024])
    z24 = logit(P[2024])
    print(f"\nfold2024 기준 BSS {base:.1f}")
    print(f"  {'β':>6}{'ΔBSS (δ_F 2024 oracle)':>24}{'ΔBSS (δ_F 2023 전이)':>24}")
    for b in BETAS:
        po = 1.0 / (1.0 + np.exp(-(z24 + b * dF[2024] * GTF[2024])))
        pt = 1.0 / (1.0 + np.exp(-(z24 + b * dF[2023] * GTF[2024])))
        print(f"  {b:>6.2f}{bss(Y[2024], po) - base:>+24.2f}{bss(Y[2024], pt) - base:>+24.2f}")


if __name__ == "__main__":
    main()
