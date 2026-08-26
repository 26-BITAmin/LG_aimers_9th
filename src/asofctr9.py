"""§7.151 — **중심화 없이** 원시 rate 5개를 p 제곱 (`pow5_2`, `pow5_3`). 용량-반응.

사용자 제안: *"걍 이번에는 센터링 없이 저 5 feature들 제곱해볼까"* → *"아 잠깐
그럼 세제곱 하자"*.

지수 p 는 **하나의 노브**다. p=1 이 현 배포 A 이므로 p∈{1,2,3} 이 용량-반응 곡선이 된다.
세제곱만 돌리면 §7.150 처럼 결과가 노이즈 대역(|Δ|≲10)에 떨어졌을 때 해석이 불가능한데,
두 점이 있으면 **추세**로 읽을 수 있다. 그래서 제곱 팔을 대조로 같이 돈다.

성질:
  · rate 는 [0,1] 이라 p 제곱이 **강단조** — 부호 접힘도 정보 손실도 없다
    (x³ 은 실수 전체에서 강단조라 §7.150 의 부호 문제 자체가 없는 형태다).
  · **그래도 무연산이 아니다.** §7.150 에서 확인했듯 변환이 `_smooth` 앞에 걸려
    `_sm`·`form_*`·`matchup_*` 로 번지고 거기서 단조성이 깨진다.
    (§7.150 에서 "강단조라 트리에 무연산" 이라고 사전 등록했다가 틀렸다.
     그 교훈을 여기서는 처음부터 반영한다 — **무연산 명제를 세우지 않는다.**)
  · 실측 corr(v, v²) = success .985 / reverse .919 / prev1 .975 / prev3 .990 / prev5 .992.
    참고로 §7.150 의 sign(c)·c² 은 .754/.666 이었고 결과가 −7.7 / −4.0 이었다.
  · priors 도 **같은 지수로** 올린다. 안 하면 스무딩이 v^p 를 .526 쪽으로 당기는데
    v² 중심은 .28, v³ 은 .15 다 — §7.115 결함① 과 같은 함정.

팔 (LGBM init_score ON = s29 배포 상태)
    A 현 배포   캐시     p=1
    pow5_2     신규     v²
    pow5_3     신규     v³   ← 사용자가 최종적으로 원한 것

채택 기준(§7.111 개정): 어느 fold 도 0 을 유의하게 밑돌지 않을 것 + fold2024 양수.
검정력: 시드 노이즈 std 4~9 → |Δ| ≲ 10 인 단일 대비는 "미판정". 단 **두 팔의 추세**는
       그보다 약한 기준으로도 방향을 말할 수 있다.

실행:  .venv/Scripts/python.exe -u src/asofctr9.py 2>&1 | tee experiments/asofctr9.log
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

import foldcache as _fc

_fc.CENTER_ASOF = False
KEY = _fc._key()

import lightgbm as lgb                                              # noqa: E402
import pandas as pd                                                 # noqa: E402
import features as F                                                # noqa: E402
import mlp                                                          # noqa: E402
import modelabel                                                    # noqa: E402
import soften                                                       # noqa: E402
import jointbeta as J                                               # noqa: E402
from config import (ASOF_AXES, CONTAM_AXES, CTX_AXES, DROP_FEATURES,            # noqa: E402
                    ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA,
                    MLP_AUX_LAMBDA, MLP_SEEDS, MLP_VARIANTS, N_ROUNDS,
                    RECENCY_HALF_LIFE, SEED_ENSEMBLE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight                           # noqa: E402
from features import build_features, fit_artifacts                  # noqa: E402
from initfix3 import level_init                                     # noqa: E402
from validation import bss, shift_to_prior                          # noqa: E402

FOLDS = [2023, 2024]
LGB_SEEDS = SEED_ENSEMBLE[:3]
MODES = ["pow5_2", "pow5_3"]
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED = 42
VCOL = "asof_pitcher_success_rate"


def verify(art, Xtr, mode):
    """배선 확인. **무연산 명제는 세우지 않는다** (§7.150 교훈)."""
    ac = art["asof_center"]
    cols = list(ac["table"])
    p = F._POW5[mode]
    assert set(cols) == set(F.CENTER_SUBSET5), f"대상 목록 불일치: {cols}"
    for c in cols:
        assert art["priors"][c] > 0.0, f"priors[{c}]=0 — 중심화 팔로 잘못 들어왔다"
    pr0 = art["priors"][VCOL]
    v = Xtr[VCOL].to_numpy(dtype=np.float64)
    fin = np.isfinite(v)
    assert v[fin].min() >= 0.0, "거듭제곱 결과에 음수가 있다"
    assert abs(pr0 - 0.52686 ** p) < 0.03, f"priors 가 p 제곱이 아니다 ({pr0:.5f})"
    print(f"    · p={p}  priors[success]={pr0:.5f}  "
          f"범위 [{v[fin].min():.4f},{v[fin].max():.4f}]  평균 {v[fin].mean():.4f}")


def compute(df, fold, mode, t0):
    pth = CACHE / f"ctr9_fold{fold}_{mode}.npz"
    if pth.exists():
        d = np.load(pth)
        print(f"    캐시 적중 {pth.name}")
        return d["z_lgb"], d["z_mlp"]

    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                        center_asof=mode)
    print(f"    fold {fold} [{mode}] 아티팩트 완료 ({time.time()-t0:.0f}s)")

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xtr, Xva = X(tr), X(va)
    verify(art, Xtr, mode)
    ss = tr["season"].to_numpy(dtype=np.int64)
    gg = tr["game_type"].astype("object").to_numpy().astype(str)
    aux, _ = modelabel.aux_labels(tr)
    h = soften.fit_h(tr)
    h_row = np.zeros(len(tr))
    for yr, v in h.items():
        h_row[(ss == yr) & (gg == "F")] = v
    dup = (h_row > 0) & (y_tr == 1)
    Xtr, y2, w2, aux2, _ = soften.soften(tr, Xtr, y_tr, w, aux)
    init_tr = level_init(np.concatenate([ss, ss[dup]]),
                         np.concatenate([gg, gg[dup]]), y2, w2)
    assert len(init_tr) == len(Xtr), "정렬 깨짐"

    z_lgb, n = np.zeros(len(Xva)), 0
    for over in ENSEMBLE_VARIANTS.values():
        for sd in LGB_SEEDS:
            p = dict(FINAL_LGB_PARAMS, **over, seed=sd, bagging_seed=sd,
                     feature_fraction_seed=sd, data_random_seed=sd)
            bst = lgb.train(p, lgb.Dataset(Xtr, y2, weight=w2, init_score=init_tr),
                            num_boost_round=N_ROUNDS)
            z_lgb += bst.predict(Xva, raw_score=True)
            n += 1
            del bst
    z_lgb /= n
    print(f"    fold {fold} [{mode}] lgbm {n}개 ({time.time()-t0:.0f}s)")

    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    zs = []
    for vname, (cfg, nseed) in MLP_VARIANTS.items():
        for sd in MLP_SEEDS[:nseed]:
            _, curve = mlp.train_mlp(A, C, y2, w2, B, Cv, sizes, seed=sd,
                                     verbose=False, y_aux=aux2,
                                     **dict(cfg, aux=MLP_AUX_LAMBDA))
            zs.append(curve[-1])
    z_mlp = np.mean(zs, axis=0)
    print(f"    fold {fold} [{mode}] mlp {len(zs)}개 ({time.time()-t0:.0f}s)")
    np.savez_compressed(pth, z_lgb=z_lgb, z_mlp=z_mlp)
    return z_lgb, z_mlp


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    codes, uniq = pd.factorize(df["pitcher_id"].to_numpy())
    n_id = len(uniq)
    mclean = ~((gt == "F") & (s < 2023))
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    print(f"train {df.shape}   alpha={MLP_ALPHA}  init_score=ON   팔 {MODES}")
    print(f"대상 {F.CENTER_SUBSET5}\n")

    labs = ["A 현 배포"] + MODES
    R, KEEP = {}, {}
    for fold in FOLDS:
        Xax, _, _, _ = J.axis_matrix(df, s, fold)
        vi = np.where(s == fold)[0]
        mm = mclean[vi]
        yv = y[vi]
        with np.load(CACHE / f"rounds_fold{fold}_r640.npz") as d:
            zl_a = d["k320"]
        with np.load(CACHE / f"fold{fold}_{KEY}.npz") as d:
            zm_a = d["z_mlp"]
            assert np.allclose(d["y"], yv)
        arms = {"A 현 배포": (zl_a, zm_a)}
        for m in MODES:
            arms[m] = compute(df, fold, m, t0)
        for lab, (zl, zm) in arms.items():
            z = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm + Xax @ dep
            pp = shift_to_prior(1 / (1 + np.exp(-z[mm])), yv[mm].mean())
            R[(fold, lab)] = bss(yv[mm], pp)
            KEEP[(fold, lab)] = (yv[mm], z[mm], codes[vi][mm])
        print(f"  fold {fold}: " + "  ".join(f"{k} {R[(fold,k)]:.1f}" for k in labs))

    bar = "=" * 88
    print(f"\n{bar}\n█ 판정 — 배포 층 (alpha={MLP_ALPHA} + delta축 + prior shift, clean 행)\n{bar}")
    print(f"  {'팔':<12}" + "".join(f"{f'fold {f}':>11}" for f in FOLDS)
          + "".join(f"{f'D vs A ({f})':>15}" for f in FOLDS))
    for lab in labs:
        print(f"  {lab:<12}" + "".join(f"{R[(f,lab)]:>11.1f}" for f in FOLDS)
              + "".join(f"{R[(f,lab)]-R[(f,'A 현 배포')]:>+15.1f}" for f in FOLDS))

    def boot(f, lab1, lab0):
        yv, z1, cf = KEEP[(f, lab1)]
        z0 = KEEP[(f, lab0)][1]
        order = np.argsort(cf, kind="stable")
        bnd = np.searchsorted(cf[order], np.arange(n_id + 1))
        seen = np.where(np.diff(bnd) > 0)[0]
        bs = []
        for _ in range(200):
            idx = np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                                  rng.choice(seen, size=len(seen), replace=True)])
            yy = yv[idx]
            bs.append(bss(yy, shift_to_prior(1 / (1 + np.exp(-z1[idx])), yy.mean()))
                      - bss(yy, shift_to_prior(1 / (1 + np.exp(-z0[idx])), yy.mean())))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return R[(f, lab1)] - R[(f, lab0)], lo, hi

    print(f"\n  █ 판정 대비 — 팔 쌍 직접 부트스트랩 (투수 복원추출 200회)")
    for name, aa, bb in [("v^2 (pow5_2 - A)", "pow5_2", "A 현 배포"),
                         ("**v^3 (pow5_3 - A)**", "pow5_3", "A 현 배포"),
                         ("지수 증가분 (v^3 - v^2)", "pow5_3", "pow5_2")]:
        cells = [f"{d:+6.1f} [{lo:+6.1f},{hi:+6.1f}]"
                 for d, lo, hi in (boot(f, aa, bb) for f in FOLDS)]
        print(f"     {name:<26}" + "".join(f"{c:>24}" for c in cells))

    print(f"\n  █ 용량-반응 (p=1 은 정의상 0)")
    for f in FOLDS:
        d1 = R[(f, "pow5_2")] - R[(f, "A 현 배포")]
        d2 = R[(f, "pow5_3")] - R[(f, "A 현 배포")]
        trend = ("단조 악화" if d2 < d1 < 0 else "단조 개선" if d2 > d1 > 0
                 else "비단조/혼합")
        print(f"     fold {f}:  p=1  +0.0   p=2 {d1:+6.1f}   p=3 {d2:+6.1f}   → {trend}")

    print(f"\n  ★ 대조 (같은 배포 층)")
    print(f"     §7.150 sign(c)·c^2 (중심화 위)   -7.7 / -4.0   [corr .754/.666]")
    print(f"     §7.119 lag5 중심화만             -11.3 / -3.1")
    print(f"\n  기준: 단일 대비는 |D|<=10 이면 미판정. 두 팔이 같은 방향이면 추세로 읽는다.")


if __name__ == "__main__":
    main()
