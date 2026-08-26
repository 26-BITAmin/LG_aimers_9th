"""§7.120 — `reverse_rate x success_rate` 곱 상호작용을 **배포 층에서** 측정한다.

사용자 제안:
    df["reverse_x_success"] = df["asof_pitcher_reverse_rate"] * df["asof_pitcher_success_rate"]

    (저장소를 전수 검색했으나 이 피처가 채택된 기록은 없다 — 코드에도 git 전
     히스토리에도 문자열이 0건. 그래서 기억이 아니라 **측정**으로 답한다.)

사전 정보 (학습 없이 산출)
    corr(reverse, success)  = -0.810      두 주효과는 이미 배포 피처다
    corr(prod, reverse)     = +0.890      <- §7.58 companion check(>0.5) 에 걸린다
    corr(prod, success)     = -0.575
    -> 사전은 낮다. "새 정보"가 아니라 **기존 축의 재가중**일 가능성이 높다.
      곱이 실재하면 그래도 이긴다. 그걸 재는 것이 이 스크립트다.

팔 (전부 같은 코드 경로. 바뀌는 것은 `features.EXTRA_PROD` 하나뿐)
    A         현 배포 캐시 (하네스 앵커. base 가 여기 붙는지가 검증이다)
    base      EXTRA_PROD=None      — 짝지은 대조군. **두 주효과는 이미 들어 있다**
    prod      EXTRA_PROD="prod"    — 후보. reverse x success
    placebo   EXTRA_PROD="placebo" — 영가설. success x **죽은 축(요일)**
                                     §7.78 에서 죽은 것이 확인된 실축이고 공유항
                                     (success_rate)을 남긴다. 컬럼이 하나 더 붙는
                                     것 자체의 효과(feature_fraction 재추첨, MLP
                                     입력차원 +1)를 여기서 뺀다.

두 학습기 모두에 들어간다 — LGBM 12개(4변형x3시드, init_score ON)와
MLP 8개(emb4x5 + noidx3)가 **같은 X** 를 본다. 블렌드(a=0.75) + delta 4축 +
prior shift 까지 얹은 배포 층에서 BSS 를 잰다.

채택 기준 (실행 전에 박아둔다)
    (1) prod - base 가 **양 fold 양수**
    (2) 양 fold 에서 prod - placebo > 0 (위약을 이겨야 한다)
    (3) 두 fold 의 이득 크기비 <= 3배
    (4) 클러스터 부트스트랩 95% 구간이 0 을 유의하게 밑돌지 않을 것
    내 예측: **(1)에서 죽는다** (corr 0.89 = 재가중). 시드 노이즈 std 4~9 을 감안하면
    |Δ| < 10 의 무승부가 가장 그럴듯하다.

실행:  .venv/Scripts/python.exe -u src/revxsuc.py 2>&1 | tee experiments/revxsuc.log
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
from initfix3 import level_init                                     # noqa: E402
from validation import bss, shift_to_prior                          # noqa: E402

FOLDS = [2023, 2024]
LGB_SEEDS = SEED_ENSEMBLE[:3]
MODES = ["base", "prod", "placebo"]
FLAG = {"base": None, "prod": "prod", "placebo": "placebo"}
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED = 42
# 하네스 앵커 — §7.115/§7.117 이 같은 캐시로 찍은 A 팔 값 (memory: 간이 재구현은
# 알려진 수치 재현부터). base 가 여기서 ±15 이상 벗어나면 팔이 아니라 코드가 문제다.
ANCHOR = {2023: 807.2, 2024: 924.5}

# 팔이 진짜 갈렸는지 증명하는 지문 (memory: 캐시 키가 진짜 갈렸는지 검정하라)
FP = {}


def compute(df, fold, mode, t0):
    pth = CACHE / f"revxsuc_fold{fold}_{mode}.npz"
    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = F.fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                          k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

    F.EXTRA_PROD = FLAG[mode]                    # <- 팔을 가르는 유일한 스위치

    def X(d):
        x = F.build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xva = X(va)
    # ── 팔 검증 ──────────────────────────────────────────────
    if mode == "base":
        assert "revxsuc" not in Xva.columns, "base 팔에 후보 컬럼이 들어갔다"
        FP[(fold, mode)] = (len(Xva.columns), None)
    else:
        assert "revxsuc" in Xva.columns, f"{mode} 팔에 후보 컬럼이 없다"
        v = Xva["revxsuc"].to_numpy(dtype=np.float64)
        FP[(fold, mode)] = (len(Xva.columns),
                            (round(float(np.nanmean(v)), 8),
                             round(float(np.nanstd(v)), 8)))
        print(f"    [{mode}] 피처 {len(Xva.columns)}개  revxsuc "
              f"mean {np.nanmean(v):.5f} std {np.nanstd(v):.5f} "
              f"NaN {np.isnan(v).mean():.4%}")

    if pth.exists():
        print(f"    캐시 적중 {pth.name}")
        with np.load(pth) as d:
            out = (d["z_lgb"], d["z_mlp"])
        F.EXTRA_PROD = None
        return out

    Xtr = X(tr)
    print(f"    fold {fold} [{mode}] 피처 완료 ({time.time()-t0:.0f}s)")
    ss = tr["season"].to_numpy(dtype=np.int64)
    gg = tr["game_type"].astype("object").to_numpy().astype(str)
    aux, _ = modelabel.aux_labels(tr)
    h = soften.fit_h(tr)
    h_row = np.zeros(len(tr))
    for yr, v2 in h.items():
        h_row[(ss == yr) & (gg == "F")] = v2
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
    F.EXTRA_PROD = None
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
    print(f"train {df.shape}   a={MLP_ALPHA}  init_score=ON  팔 {MODES}\n")

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
        # 팔이 갈렸는가 — prod 와 placebo 의 컬럼이 실제로 다른 값이어야 한다
        assert FP[(fold, "prod")][1] != FP[(fold, "placebo")][1], "위약이 후보와 같다"
        assert (FP[(fold, "base")][0] + 1 == FP[(fold, "prod")][0]
                == FP[(fold, "placebo")][0]), "컬럼 수가 안 맞는다"
        for lab, (zl, zm) in arms.items():
            z = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm + Xax @ dep
            p = shift_to_prior(1 / (1 + np.exp(-z[mm])), yv[mm].mean())
            R[(fold, lab)] = bss(yv[mm], p)
            KEEP[(fold, lab)] = (yv[mm], z[mm], codes[vi][mm])
        print(f"  fold {fold}: " + "  ".join(f"{k} {R[(fold,k)]:.1f}" for k in labs))
        d0 = R[(fold, "base")] - ANCHOR[fold]
        flag = "OK" if abs(d0) < 15 else "!! 앵커 이탈 — 코드 문제 의심"
        print(f"    앵커 검증: base - A(기록 {ANCHOR[fold]}) = {d0:+.1f}  {flag}")

    print(f"\n{'='*92}\n█ 판정 — 배포 층 (a={MLP_ALPHA} + delta축 + prior shift, clean 행)\n{'='*92}")
    print(f"  {'팔':<12}" + "".join(f"{f'fold {f}':>11}" for f in FOLDS)
          + "".join(f"{f'D vs base ({f})':>17}" for f in FOLDS))
    for lab in labs:
        print(f"  {lab:<12}" + "".join(f"{R[(f,lab)]:>11.1f}" for f in FOLDS)
              + "".join(f"{R[(f,lab)]-R[(f,'base')]:>+17.1f}" for f in FOLDS))

    print(f"\n  클러스터 부트스트랩 (투수 복원추출 200회, 기준 = base)")
    for lab in ("prod", "placebo"):
        cells = []
        for f in FOLDS:
            yv, z1, cf = KEEP[(f, lab)]
            z0 = KEEP[(f, "base")][1]
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
            cells.append(f"{R[(f,lab)]-R[(f,'base')]:+6.1f} [{lo:+6.1f},{hi:+6.1f}]")
        print(f"     {lab:<10}" + "".join(f"{c:>26}" for c in cells))

    print(f"\n  후보 - 위약 (컬럼 하나 더 붙는 효과를 뺀 순수 내용 성분)")
    for f in FOLDS:
        print(f"     fold {f}: {R[(f,'prod')]-R[(f,'placebo')]:+6.1f}")

    print(f"\n  채택 기준: (1) prod-base 양 fold 양수  (2) prod-placebo 양 fold 양수"
          f"  (3) 크기비 <=3  (4) 부트스트랩 구간이 0 을 유의하게 밑돌지 않을 것")
    print(f"  !! 재학습 축이라 시드 노이즈(std 4~9)가 낀다. |D|<10 은 무승부로 읽는다.")


if __name__ == "__main__":
    main()
