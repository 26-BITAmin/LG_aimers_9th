"""§7.111 — `init_score`(softened 레벨) × 중심화를 **배포 층**에서 재고, LB 예측을 미리 적는다.

§7.109 의 +15.2 는 **LGBM 단독 층**이다. 배포는 LGBM + MLP(8개) + δ축 4개 + prior shift 다.
`init_score` 는 LGBM 에만 걸리므로 블렌드에서 희석된다. 전례가 정확히 이 크기다 —
§7.101 F전문가는 F행 +19.2 인데 전체 환산 +2.53 (8배 희석)이었다.

2×2 (init × center) 를 배포 층에서 채운다. §7.109 가 빠뜨린 (init=1, center=1) 칸 포함.

    팔  A  배포            z_lgb(5488ef1b18)          z_mlp(5488ef1b18)
    팔  C  중심화만        z_lgb(26bcbefa56)          z_mlp(26bcbefa56)     ← §7.106 캐시
    팔  B  init만          z_lgb 신규(init, ctr=0)    z_mlp(5488ef1b18)
    팔  E  init+중심화     z_lgb 신규(init, ctr=1)    z_mlp(26bcbefa56)     ← 빠졌던 칸

⚠️ 정직성 장치
  · α는 **배포값 고정**. z_lgb 가 바뀌면 최적 α 도 바뀌지만 평가 fold 를 보고 다시
    고르면 §7.101 실측대로 **+2.0 이 부풀려진다.**
  · F전문가(a=0.15)는 뺀다 — 네 팔에 똑같이 빠지므로 Δ 는 보존되고 절대값만 ~2.5 낮다.
  · 오차막대는 **투수 클러스터 부트스트랩** (효과가 투수 단위면 유효 표본이 400명이다).
  · **LB 예측을 미리 적는다.** §7.101 규약: LB Δ ≈ fold2024 배포 층 Δ,
    fold2023 은 강건성 경고등, 부트스트랩 구간이 오차막대.

⚠️ `jointbeta` 는 `foldcache._key` 를 "5488ef1b18" 로 못 박는다 → 지문은 **import 전에** 잡는다.

실행:  .venv/Scripts/python.exe -u src/initdeploy.py 2>&1 | tee experiments/initdeploy.log
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

# ── 지문 확보 (jointbeta import 전) ─────────────────────────────────────
import foldcache as _fc

KEY = {}
for _f in (False, True):
    _fc.CENTER_ASOF = _f
    KEY[_f] = _fc._key()
_fc.CENTER_ASOF = False
assert KEY[False] != KEY[True], "지문이 안 갈렸다"

import lightgbm as lgb                                             # noqa: E402
import soften                                                      # noqa: E402
import jointbeta as J                                              # noqa: E402
from config import (ASOF_AXES, CONTAM_AXES, CTX_AXES, DROP_FEATURES,             # noqa: E402
                    ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA, N_ROUNDS,
                    RECENCY_HALF_LIFE, SEED_ENSEMBLE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight                          # noqa: E402
from features import build_features, fit_artifacts                 # noqa: E402
from initfix3 import level_init                                    # noqa: E402
from validation import bss, shift_to_prior                         # noqa: E402

FOLDS = [2023, 2024]
LGB_SEEDS = SEED_ENSEMBLE[:3]
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED = 42


def train_lgb(df, fold, center, t0):
    """init_score(softened 레벨) LGBM. 결과 z 를 디스크에 캐시한다."""
    pth = CACHE / f"initdep_fold{fold}_ctr{int(center)}.npz"
    if pth.exists():
        print(f"    캐시 적중 {pth.name}")
        return np.load(pth)["z"]

    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                        center_asof=center)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xtr, Xva = X(tr), X(va)
    ss = tr["season"].to_numpy(dtype=np.int64)
    gg = tr["game_type"].astype("object").to_numpy().astype(str)
    Xtr, y2, w2, _, _ = soften.soften(tr, Xtr, y_tr, w)
    h = soften.fit_h(tr)
    h_row = np.zeros(len(tr))
    for yr, v in h.items():
        h_row[(ss == yr) & (gg == "F")] = v
    dup = (h_row > 0) & (y_tr == 1)
    ss2, gg2 = np.concatenate([ss, ss[dup]]), np.concatenate([gg, gg[dup]])
    init_tr = level_init(ss2, gg2, y2, w2)      # ★ soften **후** 레벨 = 탈오염 레벨
    assert len(init_tr) == len(Xtr)

    z, n = np.zeros(len(Xva)), 0
    for over in ENSEMBLE_VARIANTS.values():
        for sd in LGB_SEEDS:
            p = dict(FINAL_LGB_PARAMS, **over, seed=sd, bagging_seed=sd,
                     feature_fraction_seed=sd, data_random_seed=sd)
            bst = lgb.train(p, lgb.Dataset(Xtr, y2, weight=w2, init_score=init_tr),
                            num_boost_round=N_ROUNDS)
            z += bst.predict(Xva, raw_score=True)
            n += 1
            del bst
    z /= n
    np.savez_compressed(pth, z=z)
    print(f"    fold {fold} ctr={int(center)} 학습 완료 ({time.time()-t0:.0f}s)")
    return z


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    import pandas as pd
    codes, uniq = pd.factorize(df["pitcher_id"].to_numpy())
    n_id = len(uniq)
    mclean = ~((gt == "F") & (s < 2023))
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    print(f"train {df.shape}   α={MLP_ALPHA} (배포값 고정)   지문 "
          f"raw={KEY[False]} ctr={KEY[True]}\n")

    R, DL = {}, {}
    for fold in FOLDS:
        Xax, _, _, _ = J.axis_matrix(df, s, fold)
        vi = np.where(s == fold)[0]
        mm = mclean[vi]
        yv = y[vi]

        def cache(k):
            with np.load(CACHE / f"fold{fold}_{k}.npz") as d:
                return d["z_lgb"], d["z_mlp"], d["y"]

        zl_a, zm_a, yc = cache(KEY[False])
        assert np.allclose(yc, yv)
        zl_c, zm_c, _ = cache(KEY[True])
        zl_b = train_lgb(df, fold, False, t0)
        zl_e = train_lgb(df, fold, True, t0)

        arms = {"A 배포": (zl_a, zm_a), "C 중심화만": (zl_c, zm_c),
                "B init만": (zl_b, zm_a), "E init+중심화": (zl_e, zm_c)}
        for lab, (zl, zm) in arms.items():
            z = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm + Xax @ dep
            p = shift_to_prior(1 / (1 + np.exp(-z[mm])), yv[mm].mean())
            R[(fold, lab)] = bss(yv[mm], p)
            DL[(fold, lab)] = (yv[mm], z[mm], codes[vi][mm])
        print(f"  fold {fold}: " + "  ".join(f"{k} {R[(fold,k)]:.1f}" for k in arms))

    print(f"\n{'='*84}\n█ 배포 층 판정 (blend α={MLP_ALPHA} + δ축 + prior shift, clean 행)\n{'='*84}")
    print(f"  {'팔':<16}" + "".join(f"{f'fold {f}':>11}" for f in FOLDS)
          + "".join(f"{f'Δ vs A ({f})':>15}" for f in FOLDS))
    labs = ["A 배포", "C 중심화만", "B init만", "E init+중심화"]
    for lab in labs:
        print(f"  {lab:<16}" + "".join(f"{R[(f,lab)]:>11.1f}" for f in FOLDS)
              + "".join(f"{R[(f,lab)]-R[(f,'A 배포')]:>+15.1f}" for f in FOLDS))

    print(f"\n{'='*84}\n█ 클러스터 부트스트랩 (투수 복원추출, 200회) — Δ vs A\n{'='*84}")
    print(f"  {'팔':<16}" + "".join(f"{f'fold {f}  Δ [95%]':>30}" for f in FOLDS))
    for lab in labs[1:]:
        cells = []
        for f in FOLDS:
            yv, z1, cf = DL[(f, lab)]
            _, z0, _ = DL[(f, "A 배포")]
            order = np.argsort(cf, kind="stable")
            bnd = np.searchsorted(cf[order], np.arange(n_id + 1))
            seen = np.where(np.diff(bnd) > 0)[0]
            bs = []
            for _ in range(200):
                pick = rng.choice(seen, size=len(seen), replace=True)
                idx = np.concatenate([order[bnd[j]:bnd[j + 1]] for j in pick])
                yy = yv[idx]
                a = bss(yy, shift_to_prior(1 / (1 + np.exp(-z1[idx])), yy.mean()))
                b = bss(yy, shift_to_prior(1 / (1 + np.exp(-z0[idx])), yy.mean()))
                bs.append(a - b)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            cells.append(f"{R[(f,lab)]-R[(f,'A 배포')]:+7.1f}  [{lo:+6.1f}, {hi:+6.1f}]")
        print(f"  {lab:<16}" + "".join(f"{c:>30}" for c in cells))

    print(f"\n{'='*84}\n█ ★ LB 예측 — **미리 적는다** (§7.101 규약)\n{'='*84}")
    print("  LB Δ ≈ fold2024 배포 층 Δ. fold2023 은 강건성 경고등. 부트스트랩이 오차막대.")
    print("  §7.101 실적: 예측 +2.53 [+1.04, +4.08] → 실측 +2.2751 (구간 안).\n")
    print(f"  {'팔':<16}{'LB 예측':>10}{'§2 조건 ② 부호일치':>20}{'③ 크기비':>12}{'판정':>10}")
    for lab in labs[1:]:
        d23, d24 = R[(2023, lab)] - R[(2023, "A 배포")], R[(2024, lab)] - R[(2024, "A 배포")]
        same = (d23 > 0) == (d24 > 0)
        ratio = max(abs(d23), abs(d24)) / max(min(abs(d23), abs(d24)), 1e-9)
        ok = same and ratio <= 3.0 and d24 > 0
        print(f"  {lab:<16}{d24:>+10.1f}{'O' if same else 'X':>20}{ratio:>12.2f}"
              f"{'후보' if ok else '기각':>10}")


if __name__ == "__main__":
    main()
