"""§7.162e — 증강 대상 전수 비교. **s40(배포본) 대비 쌍체**가 s41 판정식이다.

지금까지 Δ 는 전부 `base` 대비였다. s41 을 낼지는 **배포본 s40 = SMOTE 팔** 대비로
물어야 하고, 그 구간이 0 을 배제해야 바꿀 이유가 된다.

2x2 가 이미 갈랐다 — **조건은 실패 클래스(y=0)** 이고 recency 는 주효과가 없다:
    실패 제한:  낡음 +1.36/+1.33/+1.73  ·  최근 +1.85
    라벨 무제한: 낡음 +0.43  ·  최근 −0.56  ·  무표적 −0.51

⚠️ 팔이 9개다. 최댓값 고르기 위험이 있으므로 **기전이 지목하는 팔**(ALL_NEG,
   UNION)을 우선하고, 구간 하한을 폭 대비로 같이 읽는다.

실행:  .venv/Scripts/python.exe -u src/smfinal.py 2>&1 | tee experiments/smfinal.log
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

FOLD, B = 2024, 400   # shift_to_prior 가 호출당 ~0.3s 다. B=400 은 600/2000 과 일치했다
C = Path(__file__).parent.parent / "experiments" / "cache"
#  표시명 → 캐시 파일
ARMS = [
    ("ALL_NEG",    "smrand", "전 시즌 실패 행 (시점 무제한)  ← 기전 지목"),
    ("UNION",      "smrand", "낡음·실패 F+R (같은 용량)      ← 기전 지목"),
    ("BOTH2X",     "smrand", "낡음·실패 F+R (2배 용량)"),
    ("RECENT_NEG", "smrand", "최근(2023+)·실패만"),
    ("SMOTE",      "sfull",  "낡음·실패 F 만  ★ s40 배포본"),
    ("PLACEBO",    "sfull",  "낡음·실패 R 만"),
]


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

    def zf(zl):
        z = (1 - CAT_ALPHA) * ((1 - MLP_ALPHA) * zl + MLP_ALPHA * mb) + CAT_ALPHA * cb
        z = z.copy()
        z[isF] = (1 - FSPEC_ALPHA) * z[isF] + FSPEC_ALPHA * z_fs[isF]
        return z + ax

    def sc(z, yy, idx=None):
        if idx is not None:
            z, yy = z[idx], yy[idx]
        m = yy.mean()
        p = shift_to_prior(1 / (1 + np.exp(-z)), m)
        return bss(yy, shift_to_prior(PROB_CENTER + PROB_SCALE * (p - PROB_CENTER), m))

    Z, miss = {}, []
    for k, pre, _ in ARMS + [("base", "sfull", "")]:
        f = C / f"{pre}_f{FOLD}_{k}_lgb.npz"
        if f.exists():
            Z[k] = zf(np.load(f)["z"])
        else:
            miss.append(k)
    if miss:
        print("⚠️ 캐시 없음:", miss)

    codes, uniq = pd.factorize(va["pitcher_id"].to_numpy())
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    seen = np.where(np.diff(bnd) > 0)[0]
    rng = np.random.default_rng(42)
    boots = [np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                             rng.choice(seen, size=len(seen), replace=True)])
             for _ in range(B)]
    S = {k: np.array([sc(z, yva, ix) for ix in boots]) for k, z in Z.items()}
    P = {k: sc(z, yva) for k, z in Z.items()}

    for ref, title in (("base", "① base(935.8) 대비 — 효과가 실재하는가"),
                       ("SMOTE", "② ★ s40 배포본 대비 — s41 로 바꿀 이유가 있는가")):
        print("\n" + "=" * 92)
        print(f"{title}   부트스트랩 {B}회 · 투수 {len(seen)}명")
        print("=" * 92)
        print("  %-34s %8s %8s %22s %8s %8s"
              % ("팔", "BSS", "Δ", "95% 구간", "P(Δ>0)", "하한/폭"))
        for k, _, desc in ARMS:
            if k not in Z or k == ref:
                continue
            d = S[k] - S[ref]
            lo, hi = np.percentile(d, [2.5, 97.5])
            # 하한이 폭의 10% 안이면 B 에 민감한 경계선이다 (부트스트랩 B 메모)
            frac = lo / (hi - lo) if hi > lo else 0.0
            flag = " ⚠" if 0 < frac < 0.10 else ""
            print("  %-34s %8.1f %+8.2f   [%+7.2f, %+7.2f] %8.3f %7.1f%%%s"
                  % (desc, P[k], P[k] - P[ref], lo, hi, (d > 0).mean(),
                     100 * frac, flag))


if __name__ == "__main__":
    main()
