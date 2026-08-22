"""§7.113 — `init_score` 위에서 `RECENCY_HALF_LIFE` 재측정 (배포 층, LGBM+MLP 전부).

배포값 0.5 는 `init_score` **없이** 튜닝됐다. §7.4 표가 기전을 말한다:

    init_score 없음 → 있음 :  동일가중 314 → 694,  hl=1.5 492 → 703,  hl=1.0 558 → 714
    → **init_score 를 켜면 recency 의 이득이 줄어든다.** 레벨을 이미 뺐으니
      오래된 데이터의 주된 해악이 사라진 것이다.
    → 예측: **최적 half_life 가 커진다** (오래된 데이터를 더 써도 된다).

기존 스윕(§7.4, game_type 제거): hl 1.5 664 / 1.0 691 / 0.7 702 / **0.5 716**.
단조 개선이라 0.5 가 끝점이었고 **그 아래는 재본 적이 없다.** 양방향을 다 본다.

⚠️ 이 축은 `N_ROUNDS` 와 성질이 다르다 — 가중치가 **모든 것**을 바꾼다:
   LGBM · MLP · `fit_artifacts` 의 prior · soften 가중 · 그리고 `init_score` 의 레벨까지.
   잘라 재는 쌍체 비교가 불가능하고 값마다 전부 재학습해야 한다.
⚠️ **LGBM 만 재면 안 된다.** 배포 블렌드가 α=0.75 라 MLP 가 75% 를 차지한다.
   LGBM 층 스크리닝은 이 축에서 오판의 소지가 크다. → LGBM+MLP 를 매번 다 학습한다.

베이스라인 0.5 는 공짜다 — LGBM z 는 `roundsfix` 캐시(init=1, k=320), MLP z 는 배포
캐시에 있다 (**MLP 는 애초에 init_score 를 안 받으므로** 배포 MLP 가 그대로 유효하다).

실행:  .venv/Scripts/python.exe -u src/recencyfix.py 2>&1 | tee experiments/recencyfix.log
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
HLS = [0.35, 0.5, 0.75, 1.0]          # 0.5 = 배포값(캐시 재사용)
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED = 42


def compute(df, fold, hl, t0):
    """(z_lgb, z_mlp) — half_life=hl 로 전부 재학습. init_score 는 soften 후 레벨."""
    pth = CACHE / f"rec_fold{fold}_hl{hl:.2f}.npz"
    if pth.exists():
        d = np.load(pth)
        print(f"    캐시 적중 {pth.name}")
        return d["z_lgb"], d["z_mlp"]

    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), hl, ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

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
    print(f"    fold {fold} hl={hl} lgbm {n}개 완료 ({time.time()-t0:.0f}s)")

    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    zs = []
    for vname, (cfg, nseed) in MLP_VARIANTS.items():
        for sd in MLP_SEEDS[:nseed]:
            _, curve = mlp.train_mlp(A, C, y2, w2, B, Cv, sizes, seed=sd,
                                     verbose=False, y_aux=aux2,
                                     **dict(cfg, aux=MLP_AUX_LAMBDA))
            zs.append(curve[-1])
    z_mlp = np.mean(zs, axis=0)
    print(f"    fold {fold} hl={hl} mlp {len(zs)}개 완료 ({time.time()-t0:.0f}s)")
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
    print(f"train {df.shape}   배포 hl={RECENCY_HALF_LIFE}  α={MLP_ALPHA}  "
          f"init_score=ON   그리드 {HLS}\n")

    R, KEEP = {}, {}
    for fold in FOLDS:
        Xax, _, _, _ = J.axis_matrix(df, s, fold)
        vi = np.where(s == fold)[0]
        mm = mclean[vi]
        yv = y[vi]
        for hl in HLS:
            if abs(hl - RECENCY_HALF_LIFE) < 1e-9:
                # 배포값 — LGBM 은 roundsfix 캐시(k=320), MLP 는 배포 캐시
                with np.load(CACHE / f"rounds_fold{fold}_r640.npz") as d:
                    zl = d["k320"]
                with np.load(CACHE / f"fold{fold}_{KEY}.npz") as d:
                    zm = d["z_mlp"]
                    assert np.allclose(d["y"], yv)
                print(f"    fold {fold} hl={hl} 캐시 재사용 (배포값)")
            else:
                zl, zm = compute(df, fold, hl, t0)
            z = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm + Xax @ dep
            p = shift_to_prior(1 / (1 + np.exp(-z[mm])), yv[mm].mean())
            R[(fold, hl)] = bss(yv[mm], p)
            KEEP[(fold, hl)] = (yv[mm], z[mm], codes[vi][mm])
        print(f"  fold {fold}: " + "  ".join(f"hl{hl}:{R[(fold,hl)]:.1f}" for hl in HLS))

    print(f"\n{'='*90}\n█ 배포 층 half_life 곡선 (init_score ON, α={MLP_ALPHA} + δ축 + prior shift)\n{'='*90}")
    print(f"  {'fold':<8}" + "".join(f"{f'hl={h}':>12}" for h in HLS))
    for fold in FOLDS:
        best = max(HLS, key=lambda h: R[(fold, h)])
        print(f"  {fold:<8}" + "".join(
            (f"{R[(fold,h)]:>11.1f}" + ("*" if h == best else " ")) for h in HLS))
    print(f"  (* = 그 fold 정점.  배포값은 hl={RECENCY_HALF_LIFE})")

    print(f"\n  배포값 대비 Δ + 클러스터 부트스트랩 (투수 복원추출 200회)")
    print(f"  {'fold':<8}" + "".join(f"{f'hl={h}':>24}" for h in HLS if h != RECENCY_HALF_LIFE))
    for fold in FOLDS:
        yv, z0, cf = KEEP[(fold, RECENCY_HALF_LIFE)]
        order = np.argsort(cf, kind="stable")
        bnd = np.searchsorted(cf[order], np.arange(n_id + 1))
        seen = np.where(np.diff(bnd) > 0)[0]
        idxs = [np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                                rng.choice(seen, size=len(seen), replace=True)])
                for _ in range(200)]
        cells = []
        for h in HLS:
            if abs(h - RECENCY_HALF_LIFE) < 1e-9:
                continue
            z1 = KEEP[(fold, h)][1]
            bs = []
            for idx in idxs:
                yy = yv[idx]
                bs.append(bss(yy, shift_to_prior(1 / (1 + np.exp(-z1[idx])), yy.mean()))
                          - bss(yy, shift_to_prior(1 / (1 + np.exp(-z0[idx])), yy.mean())))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            cells.append(f"{R[(fold,h)]-R[(fold,RECENCY_HALF_LIFE)]:+6.1f} [{lo:+6.1f},{hi:+6.1f}]")
        print(f"  {fold:<8}" + "".join(f"{c:>24}" for c in cells))

    print(f"\n{'='*90}\n■ 판정 (§7.111 개정 기준)\n{'='*90}")
    print("  어느 fold 도 0 을 **유의하게** 밑돌지 않을 것 + fold2024 가 양수.")
    for h in HLS:
        if abs(h - RECENCY_HALF_LIFE) < 1e-9:
            continue
        d = [R[(f, h)] - R[(f, RECENCY_HALF_LIFE)] for f in FOLDS]
        print(f"    hl={h}: " + "  ".join(f"{f}:{v:+.1f}" for f, v in zip(FOLDS, d)))
    print("\n  ⚠️ 이 축은 재학습 축이라 **시드 노이즈(std 4~9)가 낀다.** N_ROUNDS 처럼")
    print("     쌍체가 아니다 — 부트스트랩 구간은 표집 오차만 담고 시드 오차는 별도다.")


if __name__ == "__main__":
    main()
