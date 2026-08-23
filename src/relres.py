"""남은 REL 이 정말 0인가 — **아핀 너머의 단조 재보정** (`MODEL.md §7.126`).

§7.120~122 가 건드린 것은 `p' = P0 + c(p − P0) + Δ` 즉 **아핀**이다.
Murphy 분해에서 REL 을 0으로 만드는 것은 **임의의 단조 재보정**이고 아핀은 그
부분집합이다. 혜진 §B 는 "REL 은 다 짜냈다"고 결론했지만 그 근거는 아핀 축
하나(스케일 c)의 소진이었다. 아핀이 못 잡는 곡률이 남아 있을 수 있다.

    BS = REL − RES + UNC
    REL = Σ (n_k/n)(p̄_k − ȳ_k)²      완전 재보정으로 회수 가능한 전액
    RES = Σ (n_k/n)(ȳ_k − ȳ)²        순서 정보. 단조 변환으로는 못 건드린다

⚠️ **REL 은 빈 수만으로 커진다** — 빈 안 표본오차가 그대로 들어간다.
   그래서 세 팔을 나란히 낸다.
     oracle    그 fold 라벨로 잰 REL (부풀어 있다)
     transfer  **직전 시즌**에서 적합한 등장회귀를 적용한 실제 이득
     placebo   라벨을 섞은 직전 시즌에서 적합 — 같은 자유도, 신호 0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import deploy29 as D                                            # noqa: E402
from config import TARGET_COL                                   # noqa: E402
from validation import bss, shift_to_prior                      # noqa: E402


def murphy(p, y, nbin=50):
    """등빈도 빈으로 REL / RES / UNC 를 낸다."""
    q = np.quantile(p, np.linspace(0, 1, nbin + 1)[1:-1])
    k = np.searchsorted(q, p)
    n = len(y)
    cnt = np.bincount(k, minlength=nbin).astype(float)
    ok = cnt > 0
    pk = np.bincount(k, weights=p, minlength=nbin)[ok] / cnt[ok]
    yk = np.bincount(k, weights=y, minlength=nbin)[ok] / cnt[ok]
    w = cnt[ok] / n
    ybar = y.mean()
    return (float(np.sum(w * (pk - yk) ** 2)),
            float(np.sum(w * (yk - ybar) ** 2)),
            float(ybar * (1 - ybar)))


def pav(x, y, w):
    """Pool-Adjacent-Violators — 등장회귀 (x 는 정렬돼 있다고 가정)."""
    yy, ww = y.astype(float).copy(), w.astype(float).copy()
    lvl = list(range(len(yy)))
    i = 0
    while i < len(yy) - 1:
        if yy[i] > yy[i + 1] + 1e-15:
            tw = ww[i] + ww[i + 1]
            yy[i] = (yy[i] * ww[i] + yy[i + 1] * ww[i + 1]) / tw
            ww[i] = tw
            yy = np.delete(yy, i + 1); ww = np.delete(ww, i + 1)
            lvl[i + 1:i + 2] = []
            x = np.delete(x, i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1
    return x, yy


def fit_iso(p, y, nbin=200):
    """등빈도 빈 평균에 PAV → (경계, 값). **적합은 참조 시즌에서만 한다.**"""
    q = np.quantile(p, np.linspace(0, 1, nbin + 1)[1:-1])
    k = np.searchsorted(q, p)
    cnt = np.bincount(k, minlength=nbin).astype(float)
    ok = cnt > 0
    xk = np.bincount(k, weights=p, minlength=nbin)[ok] / cnt[ok]
    yk = np.bincount(k, weights=y, minlength=nbin)[ok] / cnt[ok]
    return pav(xk, yk, cnt[ok])


def apply_iso(p, xk, yk):
    return np.interp(p, xk, yk)


def main(fold=2024, seed=42):
    df, s, y = D.load_df()
    st = D.state(df, s, y, fold)
    p0, yv = st["p0"], st["yv"]
    r = float(yv.mean()); K = 1e5 / (r * (1 - r))
    rel, res, unc = murphy(p0, yv)
    print(f"fold {fold}  n={len(yv):,}  기준 BSS {st['base']:.2f}  K={K:,.0f}")
    print(f"  Murphy:  REL {rel:.3e}   RES {res:.3e}   UNC {unc:.5f}")
    print(f"    완전 재보정의 이론 상한  K·REL = {K*rel:+.2f} BSS  (빈 노이즈 포함)")
    print(f"    RES 가 주는 현재 점수    K·RES = {K*res:+.2f} BSS")

    # 참조 시즌(직전)에서 적합 → fold 에 적용
    prev_i = np.where(s == fold - 1)[0]
    gt = df["game_type"].to_numpy()
    keep = ~((gt[prev_i] == "F") & (fold - 1 < 2023))
    pv = D.state(df, s, y, fold - 1)
    pp, yp = pv["p0"], pv["yv"]

    rng = np.random.default_rng(seed)
    print(f"\n  참조 시즌 {fold-1}  n={len(yp):,}")
    print(f"  {'팔':<26s}{'ΔBSS':>10s}   설명")
    for tag, yfit in (("transfer (등장회귀)", yp),
                      ("placebo (라벨 셔플)", rng.permutation(yp))):
        xk, yk = fit_iso(pp, yfit)
        pc = apply_iso(p0, xk, yk)
        pc = shift_to_prior(np.clip(pc, 1e-6, 1 - 1e-6), r)   # 레벨은 이미 최적
        print(f"  {tag:<26s}{bss(yv, pc) - st['base']:>+10.2f}   "
              f"{'직전 시즌 신뢰도곡선을 그대로 적용' if yfit is yp else '같은 자유도, 신호 0'}")

    # 아핀만으로 얻을 수 있는 최대 (같은 fold, oracle) — 비교용
    best, bc = 0.0, 1.0
    for c in np.linspace(0.90, 1.20, 61):
        pc = shift_to_prior(np.clip(r + c * (p0 - p0.mean()), 1e-6, 1 - 1e-6), r)
        g = bss(yv, pc) - st["base"]
        if g > best:
            best, bc = g, c
    print(f"\n  [대조] 아핀 스케일 oracle 최적  c*={bc:.3f}  {best:+.2f} BSS")
    print(f"  → 단조 재보정 전체의 상한이 아핀 최적보다 크게 높지 않으면 REL 은 닫힌 것이다")


if __name__ == "__main__":
    main()
