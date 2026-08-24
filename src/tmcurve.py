"""Lever 3 · 2.5단계 — **학습곡선 눈금** (§7.158).

§7.157 의 teacher gap `+414 / +464` 는 **student 기대값이 아니다.** student 는 `x*`
를 못 본다. student 가 받는 것은 정보가 아니라 **표본 효율** — 타깃 노이즈가
`Var(y|x)/Var(s|x) = 242배` 줄어드는 것이다. 그것이 BSS 로 얼마인지는
**학습곡선의 기울기**가 정한다.

    이득 ≈ (유효 N 옥타브 수) × dBSS/d log₂N

### 유효 N 배수 (사전 고정, Fisher 정보 합산)

    무가중 전 행   m = 1/(0.86/242 + 0.14) = 6.97   → 2.80 옥타브
    매칭 행 전용   m = 0.86 × 242 = 208             → **7.70 옥타브**  ← 게이트는 이쪽

두 극단이 2.8배 갈리는 것은 **비매칭 14% 의 hard 노이즈가 무가중 평균을 지배**하기
때문이다. 3단계의 행 가중이 자유 노브이므로 낙관 쪽 극단으로 계산한다 — 단측 논리와 정합.

### ⚠️ 편향 장부 — 네 겹 전부 같은 방향(낙관)이다

  (a) 행 제거는 라벨 수와 **x-다양성을 동시에** 줄인다. distillation 은 라벨 노이즈만
      사주므로 서브샘플 기울기는 노이즈-등가 이득을 **과대**평가한다.
  (b) 고정 320 라운드는 소표본에서 과적합을 키워 기울기를 **또** 과대평가한다.
  (c) 국소 기울기 × 여러 옥타브 외삽은 곡선 평탄화를 무시한다.
  (d) Fisher 배수 208 은 추정량이 **라벨-노이즈 제한**일 때의 값이다. 31리프 GBDT 에
      76~97만 행이면 이미 용량·정규화 제한 쪽이라 실제 배수는 더 작다.

→ **이 눈금은 상한이고, 게이트는 "미달 시 폐쇄" 단측으로만 쓴다.**
   통과는 필요조건이지 충분조건이 아니다.

### 사전 등록 게이트

    B_opt = 기울기(마지막 옥타브 50→100%) × 7.7      <  +30 BSS  →  Lever 3 폐쇄

`BSS(N) = A − C·N^(−α)` 멱법칙 외삽은 **2차 출력**으로만 낸다 — 곡률을 반영하므로
`B_opt` 보다 정직하지만, **판정 기준은 사전 등록한 것에서 바꾸지 않는다.**

실행:  .venv/Scripts/python.exe -u src/tmcurve.py 2>&1 | tee experiments/tmcurve.log
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import deploy29 as D
from config import (CENTER_ASOF, CONTAM_AXES, DROP_FEATURES, FINAL_LGB_PARAMS,
                    N_ROUNDS, RECENCY_HALF_LIFE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, USE_PLAYER_IDS)
from data import recency_weight
from features import build_features, fit_artifacts
from validation import bss, shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED = 42
FRACS = [0.25, 0.35, 0.50, 0.70, 1.00]
NDRAW = 2
M_OCT = 7.70                  # log2(208) — 사전 고정
GATE = 30.0                   # 사전 고정
B_BOOT = 4000


def fit_predict(Xtr, ytr, w, Xva, rows):
    p = dict(FINAL_LGB_PARAMS, seed=SEED, bagging_seed=SEED,
             feature_fraction_seed=SEED, data_random_seed=SEED)
    bst = lgb.train(p, lgb.Dataset(Xtr.iloc[rows], ytr[rows], weight=w[rows]),
                    num_boost_round=N_ROUNDS)
    z = bst.predict(Xva, raw_score=True)
    del bst
    return z


def paired_boot(y, p0, p1, cl, n_cl, rng, B=B_BOOT):
    ybar = y.mean()
    ref = ybar * (1 - ybar)
    n = np.bincount(cl, minlength=n_cl)
    s0 = np.bincount(cl, weights=(p0 - y) ** 2, minlength=n_cl)
    s1 = np.bincount(cl, weights=(p1 - y) ** 2, minlength=n_cl)
    seen = np.flatnonzero(n > 0)
    d = 1e5 * (s0.sum() - s1.sum()) / (n.sum() * ref)
    dr = np.empty(B)
    for i in range(B):
        k = rng.choice(seen, size=len(seen), replace=True)
        dr[i] = 1e5 * (s0[k].sum() - s1[k].sum()) / (n[k].sum() * ref)
    return d, *np.percentile(dr, [2.5, 97.5])


def main():
    t0 = time.time()
    df, s, y = D.load_df()
    jn = np.load(CACHE / "tmrow_join.npz")
    trust = jn["trust"]
    gt = df["game_type"].to_numpy()
    clean = ~((gt == "F") & (s < 2023))
    pair = (np.minimum(df.pitcher_team_id, df.batter_team_id).astype(np.int64) * 100
            + np.maximum(df.pitcher_team_id, df.batter_team_id))
    key = df.season.astype(np.int64) * 10000 + df.game_month * 100 + df.game_dayofweek
    half = df.inning.astype(np.int64) * 2 + (df.top_bottom == "B")
    game = (((pair != pair.shift()) | (key != key.shift())
             | (half < half.shift())).cumsum()).to_numpy()

    print(f"유효 N 배수 208 = {M_OCT:.2f} 옥타브 (사전 고정)  ·  게이트 B_opt < {GATE:.0f}")
    rng = np.random.default_rng(20260824)
    for fold in D.FOLDS:
        mtr = (s < fold) & clean & trust        # §7.157 팔 A 와 **완전히 같은 행**
        mva = (s == fold) & clean & trust
        tr, va = df[mtr], df[mva]
        w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                           ref_season=int(tr.season.max()))
        art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                            k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                            center_asof=CENTER_ASOF)

        def X(d):
            x = build_features(d, art)
            if USE_PLAYER_IDS:
                for c in ("pitcher_id", "batter_id"):
                    x[c] = d[c].to_numpy(dtype=np.float32)
            return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

        Xtr, Xva = X(tr), X(va)
        ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
        yva = va[TARGET_COL].to_numpy(dtype=np.float64)
        ybar = yva.mean()
        gtr = game[mtr]
        ug = np.unique(gtr)
        cl, uq = pd.factorize(game[mva])
        n_cl = len(uq)

        print(f"\n{'='*96}\nfold{fold} 라인   학습 {len(tr):,}행 / 경기 {len(ug):,}"
              f"   평가 {len(va):,}행\n{'='*96}")
        print(f"  {'비율':>6s}{'행':>10s}{'draw':>6s}{'BSS':>10s}")
        pts, preds = [], {}
        for f_ in FRACS:
            nd = 1 if f_ >= 1.0 else NDRAW
            for dr_ in range(nd):
                if f_ >= 1.0:
                    rows = np.arange(len(tr))
                else:
                    pick = rng.choice(ug, size=int(round(len(ug) * f_)), replace=False)
                    rows = np.flatnonzero(np.isin(gtr, pick))
                z = fit_predict(Xtr, ytr, w, Xva, rows)
                p_ = shift_to_prior(1 / (1 + np.exp(-z)), ybar)
                sc = bss(yva, p_)
                pts.append((len(rows), sc))
                preds.setdefault(f_, []).append(p_)
                print(f"  {f_:>6.0%}{len(rows):>10,d}{dr_:>6d}{sc:>10.1f}"
                      f"   ({time.time()-t0:.0f}s)")

        # ── 마지막 옥타브 기울기 (사전 등록: 50 → 100%) ──────────────
        b50 = np.mean([bss(yva, p_) for p_ in preds[0.50]])
        b100 = bss(yva, preds[1.00][0])
        slope = b100 - b50
        d_, lo, hi = paired_boot(yva, preds[0.50][0], preds[1.00][0], cl, n_cl, rng)
        B_opt = slope * M_OCT
        print(f"\n  마지막 옥타브 (50%→100%)  기울기 {slope:+.1f} BSS/옥타브"
              f"   쌍체 CI [{lo:+.1f}, {hi:+.1f}]")
        print(f"  ★ B_opt = {slope:+.1f} × {M_OCT:.2f} = **{B_opt:+.1f} BSS**"
              f"   → 게이트 {'통과 ✅' if B_opt >= GATE else '미달 ❌ 폐쇄'}")
        print(f"    (CI 로 환산하면 [{lo*M_OCT:+.1f}, {hi*M_OCT:+.1f}])")

        # ── 2차 출력: 멱법칙 외삽 (판정에 안 쓴다) ───────────────────
        a = np.array(pts, dtype=float)
        N, S = a[:, 0], a[:, 1]
        try:
            def f_pow(n, A, C, al):
                return A - C * n ** (-al)
            popt, _ = curve_fit(f_pow, N, S, p0=[S.max() + 200, 1e4, 0.3],
                                maxfev=40000)
            A_, C_, al_ = popt
            n0 = N.max()
            gain = f_pow(n0 * 208, *popt) - f_pow(n0, *popt)
            print(f"  [2차·판정 미사용] 멱법칙 BSS = {A_:.0f} − {C_:.3g}·N^(−{al_:.3f})"
                  f"   → N×208 외삽 이득 **{gain:+.1f} BSS**")
            print(f"    점근 상한 A = {A_:.0f} (현재 {S.max():.0f} 대비 "
                  f"+{A_-S.max():.0f} 남음)")
        except Exception as e:
            print(f"  [2차] 멱법칙 적합 실패: {e}")


if __name__ == "__main__":
    main()
