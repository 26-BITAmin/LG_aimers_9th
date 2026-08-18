"""§7.69 후속 — 3분할 축의 (1) game_type 교차 분해, (2) 전이 검정.

groupfeat.py가 남긴 두 질문:
    ① F전용 잔차 −0.090/−0.022가 「F 행 일반」의 편향인가, 「F전용 투수」의
      편향인가. 전자면 이 축이 아니라 오염 보정/레벨의 문제다.
      → 현재 행 game_type × 그룹 셀로 잔차를 쪼갠다.
    ② fold2023 잔차로 fit한 그룹 δ가 fold2024에서 BSS를 올리는가.
      효과가 1년에 4배 감쇠했으므로 (−.090 → −.022) 과보정 위험이 있다.
      → β 스윕으로 전이 곡선을 그린다. β 선택은 하지 않는다(보고만).

실행:  .venv/Scripts/python.exe -u src/groupfeat2.py 2>&1 | tee experiments/groupfeat2.log
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
from groupfeat import apply_group, fit_group
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
K_SHRINK = 2000               # 그룹 δ 수축 (행 단위 n이라 큰 값)
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
EPS = 1e-6


def group_codes(feat):
    return np.where(feat["unknown"] == 1, 3,
           np.where(feat["is_F_only"] == 1, 2,
           np.where(feat["is_R_only"] == 1, 0, 1)))


GROUP_NAMES = ["R전용", "둘다", "F전용", "신인"]


def fit_delta(y, p0, code, ncls):
    """셀별 로짓 오프셋 δ_g = logit(ȳ_g) − logit(p̄0_g), n/(n+K) 수축."""
    d = np.zeros(ncls)
    for gidx in range(ncls):
        m = code == gidx
        n = m.sum()
        if n == 0:
            continue
        ybar = np.clip(y[m].mean(), EPS, 1 - EPS)
        pbar = np.clip(p0[m].mean(), EPS, 1 - EPS)
        raw = np.log(ybar / (1 - ybar)) - np.log(pbar / (1 - pbar))
        d[gidx] = raw * n / (n + K_SHRINK)
    return d


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)

    P, Y, CODE, GT = {}, {}, {}, {}
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
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-(z + adj))), yv.mean())

        g = fit_group(tr)
        feat, _ = apply_group(d, g)
        P[fold], Y[fold] = p0, yv
        CODE[fold] = group_codes(feat)
        GT[fold] = (d["game_type"].astype("object").to_numpy() == "F")

        print(f"\nfold {fold}  기준 BSS {bss(yv, p0):.1f}")
        print(f"  {'셀':<12}{'행':>10}{'잔차 평균':>12}")
        r = yv - p0
        for gidx, name in enumerate(GROUP_NAMES):
            for is_f, gt_name in [(False, "R행"), (True, "F행")]:
                m = (CODE[fold] == gidx) & (GT[fold] == is_f)
                if m.sum() < 50:
                    continue
                print(f"  {name}·{gt_name:<6}{m.sum():>10,}{r[m].mean():>+12.5f}")

    # ── 전이 검정: fit 2023 → eval 2024 ─────────────────────────────
    print("\n=== 전이 검정 — 그룹 δ (4셀), fit f2023 → eval f2024 ===")
    d23 = fit_delta(Y[2023], P[2023], CODE[2023], 4)
    d24 = fit_delta(Y[2024], P[2024], CODE[2024], 4)  # in-fold oracle (참고용)
    print("  δ(2023 fit):", np.round(d23, 4), " δ(2024 oracle):", np.round(d24, 4))

    z24 = np.log(np.clip(P[2024], EPS, 1 - EPS) / np.clip(1 - P[2024], EPS, 1 - EPS))
    base = bss(Y[2024], P[2024])
    print(f"  {'β':>6}{'ΔBSS(2023δ 전이)':>20}{'ΔBSS(2024δ oracle)':>20}")
    for b in BETAS:
        pt = 1.0 / (1.0 + np.exp(-(z24 + b * d23[CODE[2024]])))
        po = 1.0 / (1.0 + np.exp(-(z24 + b * d24[CODE[2024]])))
        print(f"  {b:>6.2f}{bss(Y[2024], pt) - base:>+20.2f}{bss(Y[2024], po) - base:>+20.2f}")


if __name__ == "__main__":
    main()
