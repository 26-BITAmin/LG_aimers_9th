"""§7.51 ③단계 — 부호가 일관된 타자 컬럼을 **사후 로짓 가산**으로 잰다.

`tmbresid.py`가 26개 중 7개를 남겼다 (양 fold |t|>2 · 부호 일치).
`ablate42.py`가 본 것은 "컬럼으로 넣으면 fold2024에서 손해"였는데, 가산은
컬럼을 늘리지 않으므로 그 비용을 안 낸다. δ(hand)·δ(count4)와 같은 기계다.

⚠️ δ와 결정적으로 다른 점 — **이건 타자 수준 레벨이지 자기 평균 대비 편차가 아니다.**
   §7.34 G가 `batter_id` 축 천장을 **−25.7**로 쟀다 (in-season 오라클로도 음수).
   그 함정에 정면으로 걸리는 형태다. 통과하더라도 근거를 따로 세워야 한다.

⚠️ 그리고 β 고원을 본다. §7.39에서 `tm_ext_std`는 두 fold 다 부호가 양수였지만
   유효 β가 0.02~0.04뿐이라 기각됐다 — 2025에서 전이가 2배만 돼도 음수인 폭이다.
   δ(count4)가 β 0.25~2.0(8배) 전 구간 양수였던 것과 대비된다.

실행:  .venv/Scripts/python.exe -u src/tmbfit.py 2>&1 | tee experiments/tmbfit.log
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

import ctxfit
import foldcache
from config import CTX_AXES, CTX_K, MLP_ALPHA, MODEL_DIR, TARGET_COL, TRAIN_CSV
from data import load_csv
from features import logit_adjust
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
BETAS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

# tmbresid.py가 남긴 7개. `*_dev`는 fold 안에서 raw와 상수 차이라 t가 동일하므로
# 대표 하나만 남긴다 (velo/spin의 dev는 raw와 같은 컬럼을 다시 재는 것).
CAND = ["tmb_n", "tmb_velo_mean", "tmb_spin_mean", "tmb_hb_std", "tmb_zvelo_mean"]


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    tmb = pd.read_csv(MODEL_DIR / "trackman_batter.csv")
    cached = foldcache.load(FOLDS, verbose=False)
    out = {}

    for fold in FOLDS:
        tr, va = s < fold, s == fold
        d = df[va]
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        adj, _ = logit_adjust(d, {"ctx": ctx})
        z0 = z + adj
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean())
        base = bss(yv, p0)

        sub = tmb[tmb.season == fold].set_index("batter_id")
        bid = d["batter_id"].to_numpy(dtype=np.int64)

        print(f"\nfold {fold}  기준 {base:.1f}   " + "".join(f"{'β'+str(b):>9}" for b in BETAS))
        for c in CAND:
            v = sub[c].reindex(bid).to_numpy(dtype=np.float64)
            m = np.isnan(v)
            # **학습 시즌 통계로만 표준화한다.** 검증 행의 평균/표준편차를 쓰면 리크다.
            trv = tmb[tmb.season <= fold][c].to_numpy(dtype=np.float64)
            mu, sd = np.nanmean(trv), np.nanstd(trv)
            v = np.where(m, 0.0, (v - mu) / (sd if sd > 0 else 1.0))    # 결측 → 0(=평균)
            row = []
            for b in BETAS:
                g = bss(yv, shift_to_prior(1.0 / (1.0 + np.exp(-(z0 + b * v))),
                                           yv.mean())) - base
                row.append(g)
            out.setdefault(c, {})[fold] = row
            print(f"  {c:<18}" + "".join(f"{g:>+9.2f}" for g in row))

    print("\n" + "=" * 92)
    print("판정 — 조건 ②(부호) · ③(크기비) · β 고원")
    print("=" * 92)
    for c, byf in out.items():
        a, b = np.array(byf[2023]), np.array(byf[2024])
        both_pos = (a > 0) & (b > 0)
        i = int(np.argmax(np.minimum(a, b)))
        ratio = (max(abs(a[i]), abs(b[i])) / max(min(abs(a[i]), abs(b[i])), 1e-9))
        plateau = [BETAS[j] for j in range(len(BETAS)) if both_pos[j]]
        span = (max(plateau) / min(plateau)) if plateau else 0.0
        print(f"  {c:<18} 최적 β={BETAS[i]:<5} f2023 {a[i]:+7.2f}  f2024 {b[i]:+7.2f}"
              f"  크기비 {ratio:>5.1f}")
        print(f"  {'':18} 양 fold 양수 구간: {plateau if plateau else '없음'}"
              f"   (폭 {span:.0f}배)")
    print("\n  기준: δ(count4)는 β 0.25~2.0(8배) 전 구간 양수여서 채택됐고,")
    print("        투수 tm_ext_std는 유효 β 폭이 2배뿐이라 기각됐다 (§7.39 ③).")


if __name__ == "__main__":
    main()
