"""§7.116 — **사용자가 원래 제안한 형태** 그대로: 평균 0 + scale 재설정.

사용자 원안: *"시즌·game_type 별 asof 평균을 구하고 **평균을 0으로 설정해서 scale을
다시 설정**하면 모델이 방향성을 더 잘 학습할 것 같다."*

내가 돌린 두 버전은 **둘 다 그게 아니었다:**

| 버전 | 더한 상수 | 스무딩 prior | 스케일 |
| --- | --- | --- | --- |
| §7.106 1차 | μ_global (비가중) | 원본 prior | 안 함 |
| §7.115 2차 | 원본 prior | 원본 prior | 안 함 |
| **원안 = 이번** | **없음 (0)** | **0** | **재설정** |

세 번째가 유일하게 **일관된** 설계다. 평균을 0 으로 두면 스무딩도 `n·v/(n+k)` —
**편차를 0 으로 수축**하는 형태가 되어야 논리가 맞는다 (§7.63 이 `gap` 축에서 쓰는 형태).
앞의 두 버전은 **레벨용 prior 에 편차를 밀어넣은 혼종**이었고, 내 결함 ①②가 전부
"상수를 어디에 맞출까"에서 나왔다. **평균 0 + prior 0 이면 그 질문 자체가 사라진다.**

★ 함정이 한 곳 더 있었다 — `mlp.fit_encoder` 의 `nan_plan` 이 결측을 `art["priors"]` 로
  채운다. 평균 0 으로 중심화해놓고 결측을 .52 로 채우면 **그 행만 10σ 밖으로 날아간다.**
  → `art["priors"][col] = 0.0` 한 줄로 스무딩·결측대체·`form_*` 셋을 동시에 맞췄다.

팔 (LGBM 은 전부 init_score ON = s29 배포 상태)
    A   현 배포        중심화 없음
    Z   zero           v − μ,        priors → 0
    ZS  zero_scale     (v − μ)/σ,    priors → 0

검증 완료 (fold2024 학습셋): zero 는 학습 시즌 μ 가 전부 ±0.0000, priors 0.00000.
zero_scale 은 거기에 sd = 1.0000. 검증 시즌은 외삽이라 zero +0.0026 / zs +0.0538 남는다.

실행:  .venv/Scripts/python.exe -u src/asofctr5.py 2>&1 | tee experiments/asofctr5.log
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
MODES = ["zero", "zero_scale", "zero_off"]   # zero_off = §7.116 기전 진단 (시즌 축 되돌림)
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED = 42


def compute(df, fold, mode, t0):
    pth = CACHE / f"ctr5_fold{fold}_{mode}.npz"
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
    pc = art["priors"]["asof_pitcher_success_rate"]
    print(f"    fold {fold} [{mode}] 아티팩트 완료  priors={pc:.5f} ({time.time()-t0:.0f}s)")

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xtr, Xva = X(tr), X(va)
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
    print(f"train {df.shape}   α={MLP_ALPHA}  init_score=ON  모드 {MODES}\n")

    labs = ["A 현 배포"] + [f"{m}" for m in MODES]
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
            p = shift_to_prior(1 / (1 + np.exp(-z[mm])), yv[mm].mean())
            R[(fold, lab)] = bss(yv[mm], p)
            KEEP[(fold, lab)] = (yv[mm], z[mm], codes[vi][mm])
        print(f"  fold {fold}: " + "  ".join(f"{k} {R[(fold,k)]:.1f}" for k in labs))

    print(f"\n{'='*84}\n█ 판정 — 배포 층 (α={MLP_ALPHA} + δ축 + prior shift, clean 행)\n{'='*84}")
    print(f"  {'팔':<14}" + "".join(f"{f'fold {f}':>11}" for f in FOLDS)
          + "".join(f"{f'Δ ({f})':>12}" for f in FOLDS))
    for lab in labs:
        print(f"  {lab:<14}" + "".join(f"{R[(f,lab)]:>11.1f}" for f in FOLDS)
              + "".join(f"{R[(f,lab)]-R[(f,'A 현 배포')]:>+12.1f}" for f in FOLDS))

    print(f"\n  클러스터 부트스트랩 (투수 복원추출 200회)")
    for lab in labs[1:]:
        cells = []
        for f in FOLDS:
            yv, z1, cf = KEEP[(f, lab)]
            z0 = KEEP[(f, "A 현 배포")][1]
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
            cells.append(f"{R[(f,lab)]-R[(f,'A 현 배포')]:+6.1f} [{lo:+6.1f},{hi:+6.1f}]")
        print(f"     {lab:<12}" + "".join(f"{c:>24}" for c in cells))

    print(f"\n  ★ 앞선 두 버전 (같은 조건, init_score 위에서)")
    print(f"     §7.111  μ_global 혼종   −4.1 / −12.3")
    print(f"     §7.115  prior 혼종      −30.7 / −53.9")
    print(f"\n  판정 기준(§7.111 개정): 어느 fold 도 0 을 유의하게 밑돌지 않을 것 + fold2024 양수.")


if __name__ == "__main__":
    main()
