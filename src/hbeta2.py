"""§7.161 2단계 — β 의 최적점이 `init_score`·`fspec` 앞에서도 버티나

1단계(hbeta.py)는 §7.70 과 같은 **맨 LGBM** 이었다. 배포 스택에는 β 와 같은
일을 하는 것이 둘 더 있다:

  ① init_score  (시즌 × game_type) 레벨을 타깃에서 뺀다. §7.109 는 소프트
     라벨과 **대체재**라고 판정했다 (같이 켜서 −727 을 맞았다). 지금은
     train.py 가 "soften **후**" 레벨을 만들어 이중 차감을 피한다.
  ② fspec       F 행 로짓의 45.8% 를 2023+ F 전용 전문가가 가져간다.
     → 소프트 라벨의 **F 쪽 이득은 중복**이 되고 **R 쪽 손해만 남는다.**
     둘 다 최적 h 를 아래로 밀 방향이다. 크기를 재야 한다.

주효과가 아니라 **상호작용**을 본다 (§7.109 의 교훈 — 주효과만 보면 갈리지 않는다):

    Δ(β=0.65 − β=1.00)  를  init OFF / ON  에서 각각,  fspec OFF / ON 에서 각각.

fspec 은 β 와 **독립**이다 — 학습셋이 2023+ F 뿐이고 soften 은 ≤2022 만 건드린다.
그래서 fold 당 한 번만 학습해 모든 β 팔에 사후 블렌드한다 (거의 공짜).
fold2023 은 train 이 ≤2022 라 2023+ F 앵커가 없다 → fspec 은 **fold2024 전용**.

⚠️ 남는 한계: MLP(α=0.75)·CatBoost 는 재현하지 않는다. 둘은 앙상블 멤버라
   이득과 손해를 함께 희석할 뿐 부호를 바꿀 기전이 없다. 판정은 LB 프로브다.

실행:  .venv/Scripts/python.exe -u src/hbeta2.py 2>&1 | tee experiments/hbeta2.log
"""
import hashlib
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
from config import (CONTAM_AXES, DROP_FEATURES, FINAL_LGB_PARAMS, FSPEC_ALPHA,
                    FSPEC_MAX_ROUNDS, FSPEC_MIN_SEASON, FSPEC_PARAMS, FSPEC_SEEDS,
                    N_ROUNDS, RECENCY_HALF_LIFE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from hbeta import hbar, make_arm
from validation import bss, brier, shift_to_prior

FOLDS = [2024, 2023]
SEEDS = (42, 7, 2024)
#  0.873 ≈ logit β=1.2 (h̄ .2445) — 1단계에서 **두 fold 다 양수**였던 유일한 지점.
#  0.65 는 fold2024 최적 구간, 1.00 은 현 배포.
BETAS = (0.65, 0.873, 1.00)
CACHE = Path(__file__).parent.parent / "experiments" / "cache" / "hbeta2.npz"


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def season_gt_level(ss, gg, y, w, min_n=500):
    """train.py:_season_gt_level 과 같은 함수. **soften 후** y·w 를 받는 것이 요점."""
    ug = np.unique(gg)
    k = ss * len(ug) + np.searchsorted(ug, gg)
    out = np.empty(len(k), dtype=np.float64)
    gm = float(np.average(y, weights=w))
    for kk in np.unique(k):
        m = k == kk
        out[m] = float(np.average(y[m], weights=w[m])) if m.sum() >= min_n else gm
    return _logit(out)


def prep_fold(df, valid_season):
    tr = df[df.season < valid_season].reset_index(drop=True)
    va = df[df.season == valid_season]
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

    return tr, va, w, X


def train_fspec(tr, X, Xva):
    """F 전용 전문가 — train.py:train_fspec 과 같은 조립. β 와 무관하다."""
    m = ((tr["game_type"].to_numpy().astype(str) == "F")
         & (tr["season"].to_numpy() >= FSPEC_MIN_SEASON))
    if not m.any():
        return None, 0
    sub = tr[m]
    Xf = X(sub)
    yf = sub[TARGET_COL].to_numpy(dtype=np.float64)
    tt = (sub["season"].to_numpy(dtype=np.int64) * 12
          + sub["game_month"].to_numpy(dtype=np.int64))
    cut = np.quantile(tt, 0.75)
    itr, iva = tt <= cut, tt > cut
    z = np.zeros(len(Xva))
    for seed in FSPEC_SEEDS:
        prm = dict(FSPEC_PARAMS, seed=seed, bagging_seed=seed,
                   feature_fraction_seed=seed, data_random_seed=seed)
        b0 = lgb.train(prm, lgb.Dataset(Xf[itr], yf[itr]),
                       num_boost_round=FSPEC_MAX_ROUNDS,
                       valid_sets=[lgb.Dataset(Xf[iva], yf[iva])],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        bst = lgb.train(prm, lgb.Dataset(Xf, yf),
                        num_boost_round=max(50, b0.best_iteration))
        z += bst.predict(Xva, raw_score=True)
        del bst, b0
    return z / len(FSPEC_SEEDS), int(m.sum())


def train_z(X2, y2, w2, init, Xva):
    """시드별 valid 로짓. init_score 는 **soften 후** y·w 로 만들어 넘긴다."""
    out = []
    for sd in SEEDS:
        p = dict(FINAL_LGB_PARAMS, seed=sd, bagging_seed=sd,
                 feature_fraction_seed=sd, data_random_seed=sd)
        bst = lgb.train(p, lgb.Dataset(X2, y2, weight=w2, init_score=init),
                        num_boost_round=N_ROUNDS)
        out.append(bst.predict(Xva, raw_score=True))
        del bst
    return np.stack(out)


def score(z, yva, va_f):
    """오라클 시프트 후 (전체, F행, R행) BSS — 1단계와 같은 눈금."""
    refF = yva[va_f].mean() * (1 - yva[va_f].mean())
    refR = yva[~va_f].mean() * (1 - yva[~va_f].mean())
    o = []
    for zi in z:
        pr = shift_to_prior(1 / (1 + np.exp(-zi)), yva.mean())
        o.append((bss(yva, pr),
                  1e5 * (1 - brier(yva[va_f], pr[va_f]) / refF),
                  1e5 * (1 - brier(yva[~va_f], pr[~va_f]) / refR)))
    return np.array(o)


def main():
    df = load_csv(TRAIN_CSV)
    t0 = time.time()
    cache = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        cache = {tuple(k.split("|")): z[k] for k in z.files}
        print("재개 캐시 %d칸" % len(cache))
    res = {}

    for fold in FOLDS:
        print("\n" + "=" * 86)
        print("=== fold %d ===" % fold)
        print("=" * 86)
        tr, va, w, X = prep_fold(df, fold)
        Xtr, Xva = X(tr), X(va)
        yva = va[TARGET_COL].to_numpy(dtype=np.float64)
        va_f = va["game_type"].to_numpy().astype(str) == "F"
        ss0 = tr["season"].to_numpy(dtype=np.int64)
        gt0 = tr["game_type"].astype("object").to_numpy().astype(str)

        z_fs, n_fs = train_fspec(tr, X, Xva)
        print("  fspec: %s (%.0fs)"
              % ("학습 %s행" % "{:,}".format(n_fs) if z_fs is not None
                 else "SKIP — 2023+ F 없음", time.time() - t0))

        for beta in BETAS:
            h = soften.fit_h(tr, beta=beta, mode="lin")
            X2, y2, w2, nd = make_arm(tr, Xtr, w, h, "lin", None)
            # soften 이 복제한 행만큼 레벨용 (season, game_type) 도 늘린다
            h_row = np.zeros(len(tr))
            for yr, v in h.items():
                h_row[(ss0 == yr) & (gt0 == "F")] = v
            dup = (h_row > 0) & (tr[TARGET_COL].to_numpy() == 1)
            ss2 = np.concatenate([ss0, ss0[dup]]) if nd else ss0
            gt2 = np.concatenate([gt0, gt0[dup]]) if nd else gt0

            for use_init in (False, True):
                init = season_gt_level(ss2, gt2, y2, w2) if use_init else None
                name = "b=%.3f init=%s" % (beta, "ON " if use_init else "OFF")
                fp = hashlib.sha1(np.ascontiguousarray(w2).tobytes()
                                  + np.ascontiguousarray(y2).tobytes()
                                  + (np.ascontiguousarray(init).tobytes()
                                     if init is not None else b"none")).hexdigest()[:8]
                ck = (str(fold), fp)
                if ck in cache:
                    z, tag = cache[ck], "cache"
                else:
                    z = train_z(X2, y2, w2, init, Xva)
                    cache[ck] = z
                    CACHE.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(CACHE, **{"|".join(k): v for k, v in cache.items()})
                    tag = "%.0fs" % (time.time() - t0)

                r = score(z, yva, va_f)
                res[(fold, beta, use_init, False)] = r
                line = ("  %-20s h=%.4f  BSS %8.1f +-%4.1f  F행 %8.1f  R행 %7.1f"
                        % (name, hbar(tr, h), r[:, 0].mean(), r[:, 0].std(ddof=1),
                           r[:, 1].mean(), r[:, 2].mean()))
                if z_fs is not None:
                    zb = z.copy()
                    zb[:, va_f] = ((1 - FSPEC_ALPHA) * zb[:, va_f]
                                   + FSPEC_ALPHA * z_fs[va_f])
                    rb = score(zb, yva, va_f)
                    res[(fold, beta, use_init, True)] = rb
                    line += ("   |+fspec %8.1f  F행 %8.1f"
                             % (rb[:, 0].mean(), rb[:, 1].mean()))
                print(line + "  [%s] (%s)" % (fp, tag))
        del tr, va, w, Xtr, Xva

    # ── 상호작용 표 ───────────────────────────────────────────────────
    print("\n" + "=" * 86)
    print("상호작용 — Δ = (β 대비 β=1.00), 쌍체 3시드.  부호가 갈리면 대체재다.")
    print("=" * 86)
    for use_fs in (False, True):
        rows = [(f, b, i) for (f, b, i, u) in res if u == use_fs and b != 1.00]
        if not rows:
            continue
        print("\n  [fspec %s]" % ("ON" if use_fs else "OFF"))
        print("    %-8s %-9s %10s %10s %8s %10s %10s"
              % ("fold", "init", "BSS(β)", "BSS(1.0)", "Δ", "±SE", "ΔF행"))
        for fold in FOLDS:
            for beta in BETAS:
                if beta == 1.00 or (fold, beta, False, use_fs) not in res:
                    continue
                for ui in (False, True):
                    a = res[(fold, beta, ui, use_fs)]
                    b = res[(fold, 1.00, ui, use_fs)]
                    d = a[:, 0] - b[:, 0]
                    print("    %-8s %-9s %10.1f %10.1f %+8.1f %10.1f %+10.1f"
                          % ("%d b=%.2f" % (fold, beta), "ON" if ui else "OFF",
                             a[:, 0].mean(), b[:, 0].mean(), d.mean(),
                             d.std(ddof=1) / np.sqrt(len(d)),
                             a[:, 1].mean() - b[:, 1].mean()))
    print("\n  총 %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
