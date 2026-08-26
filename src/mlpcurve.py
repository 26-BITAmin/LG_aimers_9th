"""§7.161 — emb4 부분집합 곡선으로 **장부의 `b` 를 학습 0회로 검증** (`seedvar.py` 후속).

두 추정량이 갈린다:

    이 세션 장부 (`seedvar.py`)   emb4 손실 = **+6.23 BSS @ n=5**  →  b = 6.23×5 = **31.16**
    §7.128 의 `v` (split-half)    MLP 전체 손실 = +3.16~3.47 @ n=8  →  b ≈ **25~28**

1.8배까지는 아니어도 갈리고, 이 `b` 가 확장 결정의 전부다 (`5→N` 이득 = `b(1/5 − 1/N)`).
**학습 없이 판정된다** — 기존 emb4 **5멤버의 부분집합**으로 `BSS(k)` 를 그려
`BSS(k) = A − b/k` 를 적합하면 된다.

### 배포층을 그대로 재구성한다

    z_mlp(k) = (0.875/k)·Σ_{i∈S} z_emb4_i + (0.125/3)·Σ z_noid_j
    z        = (1−α_mlp)·z_lgb + α_mlp·z_mlp
    z        = (1−α_cat)·z + α_cat·z_cat
    p        = shift_to_prior(σ(z + Xax·β), ȳ)
    p        = shift_to_prior(P0 + 1.05(p − P0), ȳ)

`k` 마다 **모든** `C(5,k)` 부분집합을 돌려 평균한다 (최대 10개) — draw 노이즈를
평균해 없앤다. 저장소 교훈: "확률적 후보는 draw 분포로 판정하라."

⚠️ fold2023 만 가능하다 (fold2024 는 MLP 멤버 캐시가 없다).

실행:  .venv/Scripts/python.exe -u src/mlpcurve.py 2>&1 | tee experiments/mlpcurve.log
"""
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np
from scipy.optimize import curve_fit

import deploy29 as D
from catens import SEEDS, VARIANTS
from config import CAT_ALPHA, MLP_ALPHA
from predict import PROB_CENTER, PROB_SCALE
from validation import bss, shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
FOLD = 2023
W_E, W_N = 0.875, 0.125          # §7.120 `_vw` — emb4 / noid 몫


def main():
    df, s, y = D.load_df()
    st = D.state(df, s, y, FOLD)
    yv, mm = st["yv"], st["mm"]
    ax = st["Xax"] @ st["dep"]
    ybar = yv.mean()

    emb = [np.load(f)["z"] for f in sorted(CACHE.glob(f"mem_emb4_f{FOLD}_*.npz"))]
    noi = [np.load(f)["z"] for f in sorted(CACHE.glob(f"mem_noid_f{FOLD}_*.npz"))]
    z_lgb = np.load(CACHE / f"initdep_fold{FOLD}_ctr0.npz")["z"]
    z_cat = np.mean([np.load(CACHE / f"catcpu_fold{FOLD}_{v}_{sd}.npz")["z"]
                     for v in VARIANTS for sd in SEEDS], axis=0)
    print(f"fold{FOLD}  emb4 {len(emb)}멤버 · noid {len(noi)}멤버 · "
          f"clean {len(yv):,}행")
    zn = np.mean(noi, axis=0)

    def score(sub):
        zm = W_E * np.mean([emb[i] for i in sub], axis=0) + W_N * zn
        z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * zm
        z = (1 - CAT_ALPHA) * z + CAT_ALPHA * z_cat
        p = shift_to_prior(1 / (1 + np.exp(-(z[mm] + ax))), ybar)
        p = shift_to_prior(PROB_CENTER + PROB_SCALE * (p - PROB_CENTER), ybar)
        return bss(yv, p)

    n = len(emb)
    print(f"\n  {'k':>3s}{'부분집합':>9s}{'BSS 평균':>11s}{'sd':>8s}"
          f"{'최소':>10s}{'최대':>10s}")
    ks, means = [], []
    for k in range(1, n + 1):
        subs = list(combinations(range(n), k))
        v = np.array([score(su) for su in subs])
        ks.append(k)
        means.append(v.mean())
        print(f"  {k:>3d}{len(subs):>9d}{v.mean():>11.2f}"
              f"{v.std(ddof=1) if len(v) > 1 else 0:>8.2f}{v.min():>10.2f}{v.max():>10.2f}")

    ks_a, m_a = np.array(ks, float), np.array(means)
    (A, b), _ = curve_fit(lambda k, A, b: A - b / k, ks_a, m_a, p0=[m_a[-1] + 30, 30])
    pred = A - b / ks_a
    print(f"\n  적합  BSS(k) = {A:.2f} − {b:.2f}/k     잔차 최대 "
          f"{np.abs(pred - m_a).max():.2f} BSS")
    print(f"  ★ 실측 b = **{b:.2f}**")
    print(f"    장부(seedvar) 예측 b = 6.231 × 5 = **31.16**   비 {b/31.16:.2f}")
    print(f"    §7.128 v (split-half)      ≈ **25~28**        비 {b/26.5:.2f}")
    print(f"\n  확장 곡선 (실측 b 기준):")
    for n2 in (6, 8, 10, 16, 20, 40, 10**9):
        g = b * (1 / n - 1 / n2)
        lab = "∞" if n2 > 10**8 else str(n2)
        print(f"    emb4 5 → {lab:>3s}   {g:+6.2f} BSS")


if __name__ == "__main__":
    main()
