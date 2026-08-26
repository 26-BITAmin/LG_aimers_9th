"""§7.117 — 지연(누적) 기준 중심화: v ← v − μ_cum(< s).  (§7.116 하네스 재사용)

사용자 제안: *"시즌으로 따지지 말고, 누적으로 해야 하나"*

기각된 네 구현(§7.111/§7.115/§7.116 zero/zero_scale)의 공통 사인은 하나로 좁혀져
있다 — **학습 쪽 정렬이 정확할수록 나빠진다** (학습 시즌은 현재 시즌 평균을 정확히
알고, 검증/2025 는 외삽뿐이라 잔차 생성 규칙이 비대칭). 이 팔은 그 비대칭을
설계로 제거한다:

    기준(시즌 s) = 시즌 s−1 까지의 (행수 가중) 누적 평균     ← train 에서만 계산
    2019       = 자기 시즌 평균 (과거가 없다)
    미지 시즌   = 전체 누적 상수 (fold 검증·2025)             ← **외삽이 없다**

학습 행도 검증 행도 기준이 한 발 늦는다 — 잔차(현 시즌 드리프트)가 같은 규칙으로
생긴다. 채택 기대보다 **기전 판별** 목적이 크다: 비대칭이 손실의 전부였다면 Δ≈0,
여전히 −20~−30 이면 미상의 손실원이 지배적이다. (zero_off 는 시즌 축을 돌려줘도
두 fold 다 음수였다 → 회의적 사전.)

팔 (LGBM 은 전부 init_score ON = s29 배포 상태)
    A    현 배포        중심화 없음 (기존 캐시)
    lag  v − μ_cum(<s), priors → 0

실행:  .venv/Scripts/python.exe -u src/asofctr6.py 2>&1 | tee experiments/asofctr6.log
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
MODES = ["lag"]
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED = 42
VCOL = "asof_pitcher_success_rate"


def verify(art, Xtr, Xva, tr, va, fold):
    """설계 서명 검사 — 잔차가 학습·검증에서 **같은 연속선** 위에 있어야 한다.

    zero 모드의 서명은 '학습 전 시즌 0.0000 / 검증만 잔차' (비대칭)였다.
    lag 모드의 서명은 '2019 ≈ 0, 이후 드리프트 누적만큼 점점 음수, 검증도 그
    연장선' (대칭)이어야 한다. 그리고 기준 표가 시즌 μ 표와 실제로 다른지 본다
    (MEMORY: 캐시 키가 진짜 갈렸는지 검정하라 — 같으면 zero 를 두 번 돌린 것).
    """
    ac = art["asof_center"]
    dif = [abs(ac["lag"][VCOL][k] - ac["table"][VCOL][k])
           for k in ac["lag"][VCOL] if k in ac["table"][VCOL]]
    assert max(dif) > 1e-4, "lag 표 == 시즌 μ 표 — 팔이 갈리지 않았다"
    assert all(art["priors"][c] == 0.0 for c in ac["lag"]), "priors 가 0 이 아니다"
    s2, g2 = tr["season"].to_numpy(), tr["game_type"].astype(str).to_numpy()
    v = Xtr[VCOL].to_numpy(dtype=np.float64)
    line = []
    for yy in sorted(set(s2.tolist())):
        m = (s2 == yy) & (g2 == "R") & np.isfinite(v)
        line.append(f"{yy} {v[m].mean():+.4f}")
    vv = Xva[VCOL].to_numpy(dtype=np.float64)
    mv = (va["game_type"].astype(str).to_numpy() == "R") & np.isfinite(vv)
    print(f"    검증: 잔차(R) {'  '.join(line)}  | va {fold} {vv[mv].mean():+.4f}")


def compute(df, fold, mode, t0):
    pth = CACHE / f"ctr6_fold{fold}_{mode}.npz"
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
    pc = art["priors"][VCOL]
    print(f"    fold {fold} [{mode}] 아티팩트 완료  priors={pc:.5f} ({time.time()-t0:.0f}s)")

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xtr, Xva = X(tr), X(va)
    verify(art, Xtr, Xva, tr, va, fold)
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

    print(f"\n  ★ 대조 (같은 조건, init_score 위에서 — 전부 현재-시즌 정렬 기준)")
    print(f"     §7.111  μ_global 혼종 (73%)    −4.1 / −12.3")
    print(f"     §7.115  prior 혼종 (100%)      −30.7 / −53.9")
    print(f"     §7.116  zero 원안 (100%)       −30.8 / −56.5")
    print(f"\n  판정: Δ≈0 이면 '비대칭이 손실의 전부' 확인. 여전히 큰 음수면 미상 손실원 지배.")
    print(f"  채택 기준(§7.111 개정): 어느 fold 도 0 을 유의하게 밑돌지 않을 것 + fold2024 양수.")


if __name__ == "__main__":
    main()
