"""§7.112 — `init_score` 위에서 `N_ROUNDS` 재측정 (배포 층).

`initfix.py` 가 잰 라운드 곡선 중 유효한 것은 `init=0` 것뿐이었다 —
`init=1` 곡선은 **원자료 레벨(이중교정)** 위에서 잰 것이라 못 쓴다.
s29 로 `init_score`(soften 후 레벨)가 배포됐으니 그 위에서 다시 잰다.

기전: `init_score` 는 부스팅 **출발점**을 바꾼다. 시즌×gt 레벨을 미리 빼주므로
초반 라운드가 레벨 맞추는 데 쓰이지 않는다 → 최적 라운드가 **내려갈** 수 있다.
반대로 `init=0` 에서 fold2024 는 이미 400 을 원했다(+2.3) → 양방향을 다 본다.

    init=0 곡선 (LGBM 단독, initfix.py)
      2023   602.2  619.3  **620.2**  611.3  602.7      ← k=320 정점
      2024   756.8  800.1    811.6  **813.9**  812.0    ← k=400 정점

★ 라운드 스윕의 장점: **한 번 학습해 `num_iteration` 으로 잘라 재는 쌍체 비교**다.
  k 끼리는 같은 모델이라 시드 노이즈가 상쇄된다 (이 저장소 시드 노이즈 std 4~9 BSS 가
  k 비교에는 안 낀다). 다른 축보다 훨씬 좋은 조건이다.

★ 정직성: k 선택은 여전히 평가 fold 밖에서. **fold2023 에서 고르고 fold2024 에서 평가.**
  배포 층(blend α + δ축 + prior shift)에서 재는 것도 §7.111 이 세운 규약이다.

⚠️ `jointbeta` 는 `foldcache._key` 를 못 박는다 → 지문은 import 전에 잡는다.

실행:  .venv/Scripts/python.exe -u src/roundsfix.py 2>&1 | tee experiments/roundsfix.log
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
import soften                                                       # noqa: E402
import jointbeta as J                                               # noqa: E402
from config import (ASOF_AXES, CONTAM_AXES, CTX_AXES, DROP_FEATURES,            # noqa: E402
                    ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA, N_ROUNDS,
                    RECENCY_HALF_LIFE, SEED_ENSEMBLE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight                           # noqa: E402
from features import build_features, fit_artifacts                  # noqa: E402
from initfix3 import level_init                                     # noqa: E402
from validation import bss, shift_to_prior                          # noqa: E402

FOLDS = [2023, 2024]
LGB_SEEDS = SEED_ENSEMBLE[:3]
KS = [160, 240, 320, 400, 480, 560, 640]
MAXR = max(KS)
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED = 42


def train_curve(df, fold, t0):
    """init_score(soften 후 레벨) LGBM 을 MAXR 까지 학습하고 k 별 z 를 돌려준다."""
    pth = CACHE / f"rounds_fold{fold}_r{MAXR}.npz"
    if pth.exists():
        d = np.load(pth)
        print(f"    캐시 적중 {pth.name}")
        return {k: d[f"k{k}"] for k in KS}

    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
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

    Xtr, Xva = X(tr), X(va)
    ss = tr["season"].to_numpy(dtype=np.int64)
    gg = tr["game_type"].astype("object").to_numpy().astype(str)
    h = soften.fit_h(tr)
    h_row = np.zeros(len(tr))
    for yr, v in h.items():
        h_row[(ss == yr) & (gg == "F")] = v
    dup = (h_row > 0) & (y_tr == 1)
    Xtr, y2, w2, _, _ = soften.soften(tr, Xtr, y_tr, w)
    init_tr = level_init(np.concatenate([ss, ss[dup]]),
                         np.concatenate([gg, gg[dup]]), y2, w2)
    assert len(init_tr) == len(Xtr), "정렬 깨짐"

    Z = {k: np.zeros(len(Xva)) for k in KS}
    n = 0
    for over in ENSEMBLE_VARIANTS.values():
        for sd in LGB_SEEDS:
            p = dict(FINAL_LGB_PARAMS, **over, seed=sd, bagging_seed=sd,
                     feature_fraction_seed=sd, data_random_seed=sd)
            bst = lgb.train(p, lgb.Dataset(Xtr, y2, weight=w2, init_score=init_tr),
                            num_boost_round=MAXR)
            for k in KS:
                Z[k] += bst.predict(Xva, raw_score=True, num_iteration=k)
            n += 1
            del bst
    Z = {k: v / n for k, v in Z.items()}
    np.savez_compressed(pth, **{f"k{k}": v for k, v in Z.items()})
    print(f"    fold {fold} 학습 완료 ({time.time()-t0:.0f}s)")
    return Z


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
    print(f"train {df.shape}   α={MLP_ALPHA}   배포 N_ROUNDS={N_ROUNDS}   "
          f"init_score=ON(soften 후 레벨)\n")

    R, KEEP = {}, {}
    for fold in FOLDS:
        Xax, _, _, _ = J.axis_matrix(df, s, fold)
        vi = np.where(s == fold)[0]
        mm = mclean[vi]
        yv = y[vi]
        with np.load(CACHE / f"fold{fold}_{KEY}.npz") as d:
            zm = d["z_mlp"]
            assert np.allclose(d["y"], yv)
        Z = train_curve(df, fold, t0)
        for k in KS:
            z = (1 - MLP_ALPHA) * Z[k] + MLP_ALPHA * zm + Xax @ dep
            p = shift_to_prior(1 / (1 + np.exp(-z[mm])), yv[mm].mean())
            R[(fold, k)] = bss(yv[mm], p)
            KEEP[(fold, k)] = (yv[mm], z[mm], codes[vi][mm])
        print(f"  fold {fold}: " + "  ".join(f"k{k}:{R[(fold,k)]:.1f}" for k in KS))

    print(f"\n{'='*94}\n█ 배포 층 라운드 곡선 (init_score ON, blend α={MLP_ALPHA} + δ축 + prior shift)\n{'='*94}")
    print(f"  {'fold':<8}" + "".join(f"{f'k={k}':>10}" for k in KS))
    for fold in FOLDS:
        best = max(KS, key=lambda k: R[(fold, k)])
        print(f"  {fold:<8}" + "".join(
            (f"{R[(fold,k)]:>9.1f}" + ("*" if k == best else " ")) for k in KS))
    print("  (* = 그 fold 의 정점.  배포값은 k=320)")

    print(f"\n{'='*94}\n█ 배포값(320) 대비 Δ\n{'='*94}")
    print(f"  {'fold':<8}" + "".join(f"{f'k={k}':>10}" for k in KS))
    for fold in FOLDS:
        print(f"  {fold:<8}" + "".join(f"{R[(fold,k)]-R[(fold,320)]:>+10.1f}" for k in KS))

    # ── 정직한 선택: fold2023 에서 고르고 fold2024 에서 평가 ────────────
    ksel = max(KS, key=lambda k: R[(2023, k)])
    print(f"\n{'='*94}\n█ ★ 정직한 선택 — k 를 **fold2023 에서만** 고르고 fold2024 에서 평가\n{'='*94}")
    print(f"  fold2023 정점 k={ksel}  (BSS {R[(2023,ksel)]:.1f}, 320 대비 {R[(2023,ksel)]-R[(2023,320)]:+.1f})")
    d24 = R[(2024, ksel)] - R[(2024, 320)]
    print(f"  → fold2024 평가: {R[(2024,ksel)]:.1f}  vs k=320 {R[(2024,320)]:.1f}   Δ {d24:+.1f}")

    # 클러스터 부트스트랩 — k=320 대비 (쌍체: 같은 모델을 자른 것이라 시드 노이즈 없음)
    print(f"\n  클러스터 부트스트랩 (투수 복원추출 200회), k=320 대비:")
    print(f"  {'fold':<8}" + "".join(f"{f'k={k}':>22}" for k in KS if k != 320))
    for fold in FOLDS:
        yv, _, cf = KEEP[(fold, 320)]
        order = np.argsort(cf, kind="stable")
        bnd = np.searchsorted(cf[order], np.arange(n_id + 1))
        seen = np.where(np.diff(bnd) > 0)[0]
        picks = [rng.choice(seen, size=len(seen), replace=True) for _ in range(200)]
        idxs = [np.concatenate([order[bnd[j]:bnd[j + 1]] for j in pk]) for pk in picks]
        cells = []
        for k in KS:
            if k == 320:
                continue
            z1, z0 = KEEP[(fold, k)][1], KEEP[(fold, 320)][1]
            bs = []
            for idx in idxs:
                yy = yv[idx]
                bs.append(bss(yy, shift_to_prior(1 / (1 + np.exp(-z1[idx])), yy.mean()))
                          - bss(yy, shift_to_prior(1 / (1 + np.exp(-z0[idx])), yy.mean())))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            cells.append(f"{R[(fold,k)]-R[(fold,320)]:+6.1f} [{lo:+5.1f},{hi:+5.1f}]")
        print(f"  {fold:<8}" + "".join(f"{c:>22}" for c in cells))

    print(f"\n{'='*94}\n■ 읽는 법\n{'='*94}")
    print("  k 끼리는 **같은 모델을 자른 쌍체 비교**라 시드 노이즈가 상쇄된다.")
    print("  채택 기준(§7.111 개정): 두 fold 중 어느 쪽도 0 을 **유의하게** 밑돌지 않을 것")
    print("  + fold2024 가 양수. 부호만으로 자르지 않는다 — 구간이 0 을 포함하면 증거가 아니다.")


if __name__ == "__main__":
    main()
