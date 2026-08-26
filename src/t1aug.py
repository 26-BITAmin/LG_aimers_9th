"""§7.172 — 증강을 **T1 블렌드 구조** 위에서 다시 잰다 (§7.171 기준선 교체 반영)

§7.162 계열은 전부 **s37/s40 스택** 위에서 쟀다. §7.171 이 기준선을 T1(1116.788)
으로 바꿨고 "이후 모든 Δ는 T1 위에서 잰다" 고 못 박았다. s40 기준으로 재면
−5.66 짜리 착시가 낀다.

⚠️ **T1 자체로는 홀드아웃이 불가능하다** — 혜진 스택은 2019~2024 를 전부 학습해서
   fold2024 가 in-sample 이다. 게다가 혜진 LGBM/MLP 학습 코드는 공유본에 없다
   (있는 것: compact-context MLP · 채연 F전문가 · 검증 스크립트).

그래서 여기서 재는 것은 **"T1 의 가중 구조에서 증강의 레버리지가 어떻게 변하는가"**
이지 "T1 + 증강의 절대 점수" 가 아니다. 그 차이를 이름에 박아둔다 (T1LIKE).

무엇이 달라지나 — LGBM 이 최종 로짓에서 차지하는 몫:

    우리(s40)   0.25 × 0.90                     = 22.5%
    T1 · R행    0.90×0.25 + 0.10 (일반 LGBM 12) = 32.5%   ← 44% 더 크다
    T1 · F행    0.60×0.25                       = 15.0%

증강은 **LGBM 에만** 걸리므로 R행(88%)에서 레버리지가 커지고 F행에서 작아진다.
부호가 갈릴 수 있다.

T1 상수 (§7.171 감사에서 artifacts.joblib 직접 확인분)
    base blend      0.25 LGBM / 0.75 MLP
    R 라우팅        0.90 core + 0.10 일반 LGBM
    F 라우팅        0.60 core + 0.40 F전문가
    확률공간 아핀    center .4801 · scale 1.05 · shift +.006839

⚠️ 재현하지 못한 것 — compact-context MLP delta(0.75×0.05), 채연 F전문가(우리
   fspec 으로 대체), 혜진 δ β(hand 1.054 vs 우리 1.875, gap_ball −0.0038 없음).
   전부 **두 팔에 공통**으로 걸리는 항이라 Δ 비교에는 1차적으로 상쇄되지만,
   절대 점수는 T1 이 아니다.

실행:  .venv/Scripts/python.exe -u src/t1aug.py 2>&1 | tee experiments/t1aug.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

import jointbeta as J
from config import (ASOF_AXES, CAT_ALPHA, CTX_AXES, FSPEC_ALPHA, MLP_ALPHA,
                    MLP_SEEDS, MLP_VARIANTS, TARGET_COL, TRAIN_CSV)
from data import load_csv
from hbeta2 import train_fspec
from predict import PROB_CENTER, PROB_SCALE
from smotenc import prep
from validation import bss, shift_to_prior

FOLD, B = 2024, 400
C = Path(__file__).parent.parent / "experiments" / "cache"

# ── T1 상수 (§7.171) ──────────────────────────────────────────────────
T1_MLP_ALPHA = 0.75
T1_R_CORE, T1_R_LGB = 0.90, 0.10
T1_F_CORE, T1_F_SPEC = 0.60, 0.40
T1_CENTER, T1_SCALE, T1_SHIFT = 0.4801, 1.05, 0.006839

ARMS = [("SMOTE", "sfull", "낡음·실패 F 만  (= s40 배포본)"),
        ("UNION", "smrand", "낡음·실패 F+R"),
        ("ALL_NEG", "smrand", "전 시즌 실패 행"),
        ("RANDOM", "smrand", "무표적 (영가설)")]


def main():
    df = load_csv(TRAIN_CSV)
    tr, va, w, X, art = prep(df, FOLD)
    Xva = X(va)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    isF = va["game_type"].to_numpy().astype(str) == "F"
    z_fs, _ = train_fspec(tr, X, Xva)
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    Xax, _, _, _ = J.axis_matrix(df, df["season"].to_numpy(), FOLD)
    ax = Xax @ dep
    mb = np.mean([np.load(C / f"sfull_f{FOLD}_base_mlp_{v}_{s}.npz")["z"]
                  for v, (c, n) in MLP_VARIANTS.items() for s in MLP_SEEDS[:n]], axis=0)
    cb = np.load(C / f"sfull_f{FOLD}_base_cat.npz")["z"]

    def ours(zl):
        """우리 s40 배포 순서 — LGBM 몫 22.5%."""
        z = (1 - CAT_ALPHA) * ((1 - MLP_ALPHA) * zl + MLP_ALPHA * mb) + CAT_ALPHA * cb
        z = z.copy()
        z[isF] = (1 - FSPEC_ALPHA) * z[isF] + FSPEC_ALPHA * z_fs[isF]
        return z + ax

    def t1like(zl):
        """T1 가중 구조 — R행 LGBM 32.5%, F행 15%. CatBoost 는 T1 에 없다."""
        core = (1 - T1_MLP_ALPHA) * zl + T1_MLP_ALPHA * mb
        z = core.copy()
        z[~isF] = T1_R_CORE * core[~isF] + T1_R_LGB * zl[~isF]     # R 라우팅
        z[isF] = T1_F_CORE * core[isF] + T1_F_SPEC * z_fs[isF]     # F 라우팅
        return z + ax

    def sc_ours(z, yy, idx=None):
        if idx is not None:
            z, yy = z[idx], yy[idx]
        m = yy.mean()
        p = shift_to_prior(1 / (1 + np.exp(-z)), m)
        return bss(yy, shift_to_prior(PROB_CENTER + PROB_SCALE * (p - PROB_CENTER), m))

    def sc_t1(z, yy, idx=None):
        """T1 의 확률공간 아핀. shift 가 있어 prior 가 .475 → .4816 으로 밀린다."""
        if idx is not None:
            z, yy = z[idx], yy[idx]
        m = yy.mean()
        p = shift_to_prior(1 / (1 + np.exp(-z)), m)
        p = T1_CENTER + T1_SCALE * (p - T1_CENTER) + T1_SHIFT
        return bss(yy, shift_to_prior(np.clip(p, 1e-6, 1 - 1e-6), m))

    Z = {}
    for k, pre, _ in ARMS + [("base", "sfull", "")]:
        f = C / f"{pre}_f{FOLD}_{k}_lgb.npz"
        if f.exists():
            Z[k] = np.load(f)["z"]
    print("팔:", ", ".join(Z))

    codes, uniq = pd.factorize(va["pitcher_id"].to_numpy())
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    seen = np.where(np.diff(bnd) > 0)[0]
    rng = np.random.default_rng(42)
    boots = [np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                             rng.choice(seen, size=len(seen), replace=True)])
             for _ in range(B)]

    for label, blend, scorer, lev in (
            ("우리 s40 스택 (LGBM 22.5%)", ours, sc_ours, "22.5%"),
            ("T1LIKE 구조 (R행 32.5% / F행 15%)", t1like, sc_t1, "32.5/15%")):
        ZZ = {k: blend(z) for k, z in Z.items()}
        S = {k: np.array([scorer(z, yva, ix) for ix in boots]) for k, z in ZZ.items()}
        P = {k: scorer(z, yva) for k, z in ZZ.items()}
        print("\n" + "=" * 90)
        print(f"{label} — 증강 없음 대비 Δ   부트스트랩 {B}회 · LGBM 몫 {lev}")
        print("=" * 90)
        print("  %-30s %9s %8s %22s %8s" % ("팔", "BSS", "Δ", "95% 구간", "P(Δ>0)"))
        print("  %-30s %9.1f %8s" % ("base (증강 없음)", P["base"], "—"))
        for k, _, desc in ARMS:
            if k not in ZZ:
                continue
            d = S[k] - S["base"]
            lo, hi = np.percentile(d, [2.5, 97.5])
            print("  %-30s %9.1f %+8.2f   [%+7.2f, %+7.2f] %8.3f"
                  % (desc, P[k], P[k] - P["base"], lo, hi, (d > 0).mean()))


if __name__ == "__main__":
    main()
