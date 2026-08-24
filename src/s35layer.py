"""§7.142 — CatBoost Δ 를 **s35 등가 층**에서 다시 잰다 (사용자 요청 A).

§7.141 의 +2.81 은 `deploy29` = **s29 층**(LB 1084.14) 위의 값이다. 챔피언은
**s35 = 1109.2607832584** 이고 그 사이에 +25.1 이 들어가 있다. 남은 잔차가 그만큼
줄었으므로 희석을 예상해야 한다.

#### 무엇을 더해야 s35 인가 — 하나뿐이다

s29 -> s35 의 구성요소를 나누면:
    `_vw` MLP 변형가중 · `_c` · Δ/PROB_SHIFT · 세그먼트   -> **전부 스칼라/추론측**.
        모델 분기를 늘리지 않으므로 CatBoost 와 탈상관을 다투지 않는다.
    **fspec (F 전용 전문가, α=0.4580, F 행에만)**          -> **유일하게 모델 분기를 늘린다.**
즉 희석 위험은 fspec 하나에 집중돼 있다. 그것만 넣으면 질문에 답한다.

#### ⚠️ 구조적 제약 — s35 층은 **fold2024 에서만 정의된다**

fspec 학습셋은 `game_type=="F" & season>=2023` 이다 (2019~22 F 는 라벨 오염, §7.48).
    fold2024  train 2019-2023 -> 깨끗한 F = 2023 F 있음        ✅
    fold2023  train 2019-2022 -> 깨끗한 F = **없음**            ❌ 정의 불가
fspec_verify 가 이 문제를 "2023H2"(시즌 내부 분할)로 우회했지만, 그건 deploy29 가
캐시를 갖고 있지 않은 **다른 fold** 다.
-> 따라서 **fold2024 = s35 층**, **fold2023 = s29 층(§7.141 의 +1.48)** 로 읽는다.
   배포 규약(LB Δ ≈ fold2024 Δ, fold2023 은 강건성 경고등)과 정합이다.

#### 하네스 검증 (앵커)

fspec 를 얹으면 fold2024 기준이 924.5 에서 올라야 한다. §7.101 실측은 α=0.15 에서
**+2.53 [+1.04, +4.08]** (F행 +19.2), fold 오라클 최적 a=0.45 에서 **+4.4** 였다.
α=0.4580 이므로 **+4 안팎**이 나와야 정상이다. 크게 벗어나면 하네스가 틀린 것이다.

실행:  .venv/Scripts/python.exe -u src/s35layer.py 2>&1 | tee experiments/s35layer.log
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
import lightgbm as lgb

import deploy29 as D
import soften
from catens import SEEDS, VARIANTS, WEIGHTS
from config import (CONTAM_AXES, DROP_FEATURES, FSPEC_ALPHA, FSPEC_MAX_ROUNDS,
                    FSPEC_MIN_SEASON, FSPEC_PARAMS, FSPEC_SEEDS, MLP_ALPHA,
                    RECENCY_HALF_LIFE, SMOOTH_K_PITCHER, SMOOTH_K_BATTER,
                    TARGET_COL, USE_PLAYER_IDS)
from data import recency_weight
from features import build_features, fit_artifacts
from initfix3 import level_init
from validation import bss, shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
FOLD = 2024
B = 400


def prep_full(df, fold):
    """catgpu.prep 과 같되 art·tr·va 도 돌려준다 (fspec 학습에 필요)."""
    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                        center_asof=False)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    return X, tr, va, X(va)


def train_fspec(X, tr, X_ref, t0):
    """train.train_fspec 과 같은 규칙. 2023+ F 만, 라운드는 학습셋 내부 시간분할."""
    pth = CACHE / f"s35_fspec_fold{FOLD}.npz"
    if pth.exists():
        d = np.load(pth)
        print(f"    캐시 적중 (학습 {int(d['n']):,}행)")
        return d["z"], int(d["n"])
    m = ((tr["game_type"].to_numpy().astype(str) == "F")
         & (tr["season"].to_numpy() >= FSPEC_MIN_SEASON))
    sub = tr[m]
    Xf = X(sub)
    yf = sub[TARGET_COL].to_numpy(dtype=np.float64)
    tt = (sub["season"].to_numpy(dtype=np.int64) * 12
          + sub["game_month"].to_numpy(dtype=np.int64))
    cut = np.quantile(tt, 0.75)
    itr, iva = tt <= cut, tt > cut
    z = np.zeros(len(X_ref))
    for seed in FSPEC_SEEDS:
        prm = dict(FSPEC_PARAMS, seed=seed, bagging_seed=seed,
                   feature_fraction_seed=seed, data_random_seed=seed)
        b0 = lgb.train(prm, lgb.Dataset(Xf[itr], yf[itr]),
                       num_boost_round=FSPEC_MAX_ROUNDS,
                       valid_sets=[lgb.Dataset(Xf[iva], yf[iva])],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        nr = max(50, b0.best_iteration)
        bst = lgb.train(prm, lgb.Dataset(Xf, yf), num_boost_round=nr)
        z += bst.predict(X_ref, raw_score=True)
        print(f"    F전문가 seed={seed} rounds={nr} ({time.time()-t0:.0f}s)")
        del bst
    z /= len(FSPEC_SEEDS)
    np.savez_compressed(pth, z=z, n=int(m.sum()))
    return z, int(m.sum())


def main():
    t0 = time.time()
    df, s, y = D.load_df()
    st = D.state(df, s, y, FOLD)
    with np.load(CACHE / f"fold{FOLD}_{D.KEY}.npz") as d0:
        z_mlp = d0["z_mlp"]
    z_lgb = np.load(CACHE / f"initdep_fold{FOLD}_ctr0.npz")["z"]
    mm, yv, dep, cl, n_id = st["mm"], st["yv"], st["dep"], st["cl"], st["n_id"]
    Xa = np.zeros((len(z_lgb), st["Xax"].shape[1]))
    Xa[mm] = st["Xax"]
    ax = Xa[mm] @ dep
    ybar = yv.mean()
    ref = ybar * (1 - ybar)

    def sc(zb):
        return bss(yv, shift_to_prior(1 / (1 + np.exp(-(zb[mm] + ax))), ybar))

    z_s29 = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
    base29 = sc(z_s29)
    assert abs(base29 - D.EXPECT_BASE[FOLD]) < 0.3, f"s29 앵커 실패 {base29:.1f}"
    print(f"fold {FOLD}  s29 앵커 OK {base29:.1f}")

    X, tr, va, X_ref = prep_full(df, FOLD)
    print(f"  피처 준비 ({time.time()-t0:.0f}s)")
    z_fs, n_fs = train_fspec(X, tr, X_ref, t0)

    is_f = va["game_type"].to_numpy().astype(str) == "F"
    z_s35 = z_s29.copy()
    z_s35[is_f] = (1 - FSPEC_ALPHA) * z_s29[is_f] + FSPEC_ALPHA * z_fs[is_f]
    base35 = sc(z_s35)
    print(f"\n  fspec α={FSPEC_ALPHA} · 학습 {n_fs:,}행 · 검증 F {is_f.sum():,}"
          f"/{len(is_f):,} ({is_f.mean():.1%})")
    print(f"  s29 층 {base29:.1f}  ->  s35 등가 층 {base35:.1f}  ({base35-base29:+.2f})")
    print(f"  앵커 기대: §7.101 실측 α=.15 에서 +2.53, fold 오라클 a=.45 에서 +4.4")
    print(f"            α={FSPEC_ALPHA} 이므로 +4 안팎이면 정상\n")

    # CatBoost CPU 12멤버 (§7.141 캐시)
    Z = {(v, sd): np.load(CACHE / f"catcpu_fold{FOLD}_{v}_{sd}.npz")["z"]
         for v in VARIANTS for sd in SEEDS}
    ens = np.mean(list(Z.values()), axis=0)
    print(f"  corr(z_cat, s29블렌드) {np.corrcoef(ens[mm], z_s29[mm])[0,1]:.4f}"
          f"   corr(z_cat, s35블렌드) {np.corrcoef(ens[mm], z_s35[mm])[0,1]:.4f}"
          f"   <- 높아지면 그만큼 희석된다")

    rng = np.random.default_rng(42)
    n = np.bincount(cl, minlength=n_id)
    seen = np.where(n > 0)[0]
    for tag, zb, bse in (("s29 층 (§7.141 재현)", z_s29, base29),
                         ("s35 등가 층", z_s35, base35)):
        p0 = shift_to_prior(1 / (1 + np.exp(-(zb[mm] + ax))), ybar)
        s0 = np.bincount(cl, weights=(p0 - yv) ** 2, minlength=n_id)
        print(f"\n  ── {tag}  기준 {bse:.1f} ──")
        print(f"    {'w':>6s}{'Δ':>9s}{'2.5%':>9s}{'97.5%':>9s}{'P(Δ>0)':>9s}")
        for w in WEIGHTS:
            zn = zb.copy()
            zn = (1 - w) * zb + w * ens
            p1 = shift_to_prior(1 / (1 + np.exp(-(zn[mm] + ax))), ybar)
            d = bss(yv, p1) - bse
            s1 = np.bincount(cl, weights=(p1 - yv) ** 2, minlength=n_id)
            draws = np.empty(B)
            for b in range(B):
                pick = rng.choice(seen, size=len(seen), replace=True)
                N = n[pick].sum()
                draws[b] = 1e5 * (s0[pick].sum() - s1[pick].sum()) / (N * ref)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            mark = "  <<< 사전 고정 w" if abs(w - 0.10) < 1e-9 else ""
            print(f"    {w:>6.2f}{d:>+9.2f}{lo:>+9.2f}{hi:>+9.2f}"
                  f"{np.mean(draws>0):>9.2f}{mark}")

    print("\n  판정: s35 층 Δ 가 s29 층 대비 얼마나 줄었는가가 희석량이다.")
    print("        fold2023 은 fspec 이 정의되지 않아 s29 층 +1.48 [-1.81,+5.00] 을 유지한다.")


if __name__ == "__main__":
    main()
