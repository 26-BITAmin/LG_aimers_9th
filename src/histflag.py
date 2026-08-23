"""§7.130 후속 — `miss_prev_game` 을 **컬럼 안 늘리고** 서수로 승격한다.

왜 이 팔인가 (§7.130 판정에서 바로 따라 나온다)
    `depth` 팔은 내용이 있었다:  depth - dplacebo = **+3.6 / +0.6** (관문 2 통과)
    그런데 위약 자체가 -5.1 / -2.6 이었다 — **컬럼 하나 추가하는 비용**이 내용보다 컸다.
    -> 컬럼을 안 늘리면 그 비용을 안 낸다. 마침 §7.130 2x2 가 같은 자리에서
       `miss_prev_game` 단독 가치를 **+4.7 / +7.9** 로 확정해줬다.

    miss_prev_game = 1 if prev 결측 else 0                (현행, 1.98% 만 1)
    ->  (3 - hist_depth) / 3                              (승격판)
        depth 0(이력없음) 1.0 / 1(1경기) 0.667 / 2(2~3경기) 0.333 / 3(4+) 0.0
    방향을 원래와 맞췄다(1 = 이력 없음). 트리는 0.5 에서 쪼개면 원래 플래그를 정확히
    복원하므로 **엄밀한 상위집합**이고, MLP 도 기존에 학습된 관계의 부호가 유지된다.

팔
    base       §7.130 의 base 캐시를 **재사용**한다 — 같은 코드 경로·같은 스위치 off.
               **이게 앵커다**: 채점 경로가 그때와 같다면 807.2 / 924.5 가 정확히
               재현되어야 한다. 안 나오면 사후 보정 축이 그 사이 바뀐 것이고 비교 무효.
               (A 팔은 뺐다 — §7.128 로 config 지문이 갈려 옛 foldcache 가 없다.
                아래 MLP 멤버 고정 주석 참조.)
    depthflag  후보. 컬럼 수·순서 불변, `miss_prev_game` 의 값만 바뀐다.
    dfplacebo  영가설. depth 0 행은 **1.0 그대로 두고**(이진 결측 정보 보존),
               중간 등급을 받을 행만 (시즌 x gt) 안에서 재배정한다.
               -> 주변분포도 이진 결측정보도 같고 **'어느 행이 얕은가'만 끊긴다.**
               §7.130 에서 위약 없이 봤으면 depth 를 "내용이 해롭다"로 오독했을 것이다
               ([[noise-floor-is-learner-specific]]).

채택 기준 (실행 전에 박아둔다)
    (1) depthflag - base 가 **양 fold 양수**
    (2) 양 fold 에서 depthflag - dfplacebo > 0
    (3) 두 fold 이득 크기비 <= 3
    (4) 클러스터 부트스트랩 구간이 0 을 크게 밑돌지 않을 것
    !! |Δ| < 10 은 무승부 (재학습 시드 노이즈 std 4~9)
    !! 부호가 엇갈려도 각 fold 구간을 보고 판단한다
       ([[sign-isnt-evidence-when-the-interval-spans-zero]])

내 예측 (§7.130 에서 셋 다 틀렸으므로 근거를 좁혀 적는다):
    컬럼 비용이 없어졌으니 base 대비 부호는 depth(-1.4/-1.9)보다 **위로 간다**.
    내용 성분 +3.6/+0.6 이 그대로 남는다면 +0~+4 대. 다만 fold2024 내용이 +0.6 뿐이라
    **fold2023 만 양수인 갈림**이 가장 그럴듯하다. 그러면 (1) 에서 걸린다.

실행:  .venv/Scripts/python.exe -u src/histflag.py 2>&1 | tee experiments/histflag.log
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
                    MLP_AUX_LAMBDA, N_ROUNDS,
                    RECENCY_HALF_LIFE, SEED_ENSEMBLE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from config import MLP_SEEDS as _MLP_SEEDS_CUR                      # noqa: E402
from config import MLP_VARIANTS as _MLP_VARIANTS_CUR                # noqa: E402

# ⚠️ **MLP 멤버 수를 8 로 고정한다.** §7.128(커밋 09efe0f)이 k=8->16 을 채택하면서
#    config 의 시드 배정이 바뀌었다. 이 실험의 `base` 는 §7.130 이 **8 멤버**로 학습한
#    캐시를 재사용하므로, 여기서 config 를 그대로 읽으면 base(8) vs 후보(16) 를
#    비교하게 된다 — 팔의 차이가 아니라 **앙상블 크기의 차이**를 재는 꼴이다.
#    (그래서 foldcache 지문도 5488ef1b18 -> d477b355b6 으로 갈렸고, 옛 A 팔 캐시가
#     새 키로는 존재하지 않는다. 아래에서 A 팔을 빼고 base 앵커로 대신한다.)
#    ⇒ 이 라운드의 수치는 **pre-s32 스택** 위의 값이다. 채택하려면 현행 스택에서
#      다시 재야 한다. 여기서 묻는 것은 "컬럼 비용을 없애면 내용이 살아나나" 뿐이다.
MLP_SEEDS = _MLP_SEEDS_CUR[:5]
MLP_VARIANTS = {"emb4": (_MLP_VARIANTS_CUR["emb4"][0], 5),
                "noid": (_MLP_VARIANTS_CUR["noid"][0], 3)}
assert sum(n for _, n in MLP_VARIANTS.values()) == 8, "멤버 수가 8 이 아니다"
from data import load_csv, recency_weight                           # noqa: E402
from initfix3 import level_init                                     # noqa: E402
from validation import bss, shift_to_prior                          # noqa: E402

FOLDS = [2023, 2024]
LGB_SEEDS = SEED_ENSEMBLE[:3]
MODES = ["base", "depthflag", "dfplacebo"]
HIST = {"base": None, "depthflag": "depthflag", "dfplacebo": "depthflag"}
PERM = {"dfplacebo"}
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
# base 는 §7.130 이 이미 학습해뒀다. 같은 코드 경로·같은 스위치 off 라 재사용한다.
REUSE = {"base": "histdepth"}
SEED = 42
ANCHOR = {2023: 807.2, 2024: 924.5}
FP = {}


def _sig(v):
    return hashlib.sha1(np.ascontiguousarray(v, dtype=np.float64).tobytes()).hexdigest()[:10]


def _placebo_flag(x, d, seed):
    """depth 0(진짜 결측)은 1.0 그대로 두고, **중간 등급만** (시즌 x gt) 안에서 재배정.

    이진 결측 정보(=현행 플래그가 주는 것)는 손대지 않는다. 그래야 위약이 재는 것이
    '승격의 내용'이 되고, '플래그를 망가뜨린 손해'가 안 섞인다 — 후자는 §7.130 의
    basenf(-5.0/-8.3)가 이미 재놨다.
    """
    rng = np.random.default_rng(seed)
    v = x["miss_prev_game"].to_numpy().copy()
    key = (d["season"].to_numpy(dtype=np.int64) * 2
           + (d["game_type"].astype("object").to_numpy().astype(str) == "F"))
    for k in np.unique(key):
        idx = np.where((key == k) & (v < 1.0))[0]      # depth >= 1 인 행만
        v[idx] = v[idx][rng.permutation(len(idx))]
    x = x.copy()
    x["miss_prev_game"] = v
    return x


def compute(df, fold, mode, t0):
    pth = CACHE / (f"{REUSE[mode]}_fold{fold}_{mode}.npz" if mode in REUSE
                   else f"histflag_fold{fold}_{mode}.npz")
    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = F.fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                          k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

    F.HIST_MODE = HIST[mode]

    def X(d, seed):
        x = F.build_features(d, art)
        if mode in PERM:
            x = _placebo_flag(x, d, seed)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xva = X(va, SEED)
    v = Xva["miss_prev_game"].to_numpy(dtype=np.float64)
    assert "hist_depth" not in Xva.columns, "이 실험은 컬럼을 늘리면 안 된다"
    FP[(fold, mode)] = {"ncol": len(Xva.columns), "cols": tuple(Xva.columns),
                        "flag": _sig(v),
                        "hist": tuple(np.round(np.bincount(
                            np.round(v * 3).astype(int), minlength=4) / len(v), 6).tolist())}
    print(f"    [{mode:9s}] 피처 {len(Xva.columns)}개  flag {FP[(fold,mode)]['flag']}"
          f"  분포 {FP[(fold,mode)]['hist']}")

    if pth.exists():
        print(f"    캐시 적중 {pth.name}")
        with np.load(pth) as d:
            out = (d["z_lgb"], d["z_mlp"])
        F.HIST_MODE = None
        return out

    Xtr = X(tr, SEED + 1)
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
    F.HIST_MODE = None
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

    labs = MODES
    R, KEEP = {}, {}
    for fold in FOLDS:
        Xax, _, _, _ = J.axis_matrix(df, s, fold)
        vi = np.where(s == fold)[0]
        mm = mclean[vi]
        yv = y[vi]
        # A 팔은 뺐다 — 새 config 지문(d477b355b6)으로는 foldcache 가 없다.
        # 대신 base 자체가 앵커다: §7.130 캐시를 그대로 읽으므로 채점 경로(delta 축 +
        # prior shift)가 그때와 같다면 **807.2 / 924.5 가 정확히 재현되어야 한다.**
        # 안 나오면 그 사이에 사후 보정 축이 바뀐 것이고, 그때는 비교 자체가 무효다.
        arms = {}
        for m in MODES:
            arms[m] = compute(df, fold, m, t0)

        # ── 팔이 진짜 갈렸는가 ([[assert-the-cache-key-actually-split]]) ──
        f = FP
        assert f[(fold, "base")]["cols"] == f[(fold, "depthflag")]["cols"] \
            == f[(fold, "dfplacebo")]["cols"], "컬럼 집합·순서가 팔마다 다르다"
        assert f[(fold, "base")]["hist"][3] == f[(fold, "depthflag")]["hist"][3] \
            == f[(fold, "dfplacebo")]["hist"][3], "이진 결측 비율이 팔마다 다르다"
        assert f[(fold, "base")]["flag"] != f[(fold, "depthflag")]["flag"], \
            "승격이 안 걸렸다 (base 와 같은 플래그)"
        assert f[(fold, "depthflag")]["hist"] == f[(fold, "dfplacebo")]["hist"], \
            "위약의 주변분포가 후보와 다르다"
        assert f[(fold, "depthflag")]["flag"] != f[(fold, "dfplacebo")]["flag"], \
            "위약이 후보와 같은 배열이다 (재배정이 안 걸렸다)"

        for lab, (zl, zm) in arms.items():
            z = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm + Xax @ dep
            p = shift_to_prior(1 / (1 + np.exp(-z[mm])), yv[mm].mean())
            R[(fold, lab)] = bss(yv[mm], p)
            KEEP[(fold, lab)] = (yv[mm], z[mm], codes[vi][mm])
        print(f"  fold {fold}: " + "  ".join(f"{k} {R[(fold,k)]:.1f}" for k in labs))
        d0 = R[(fold, "base")] - ANCHOR[fold]
        print(f"    앵커 검증: base - §7.130 기록({ANCHOR[fold]}) = {d0:+.2f}  "
              f"{'OK' if abs(d0) < 0.1 else '!! 채점 경로가 그 사이 바뀌었다 — 비교 무효'}")

    print(f"\n{'='*92}\n█ 판정 — 배포 층 (a={MLP_ALPHA} + delta축 + prior shift, clean 행)\n{'='*92}")
    print(f"  {'팔':<12}" + "".join(f"{f'fold {f}':>11}" for f in FOLDS)
          + "".join(f"{f'Δ vs base ({f})':>17}" for f in FOLDS))
    for lab in labs:
        print(f"  {lab:<12}" + "".join(f"{R[(f,lab)]:>11.1f}" for f in FOLDS)
              + "".join(f"{R[(f,lab)]-R[(f,'base')]:>+17.1f}" for f in FOLDS))

    print(f"\n  클러스터 부트스트랩 (투수 복원추출 200회, 기준 = base)")
    for lab in MODES[1:]:
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
        print(f"     {lab:<12}" + "".join(f"{c:>26}" for c in cells))

    print(f"\n  후보 - 위약 (승격의 순수 내용 성분)")
    for f in FOLDS:
        print(f"     fold {f}: {R[(f,'depthflag')]-R[(f,'dfplacebo')]:+6.1f}"
              f"     (참고 §7.130 depth-dplacebo: "
              f"{'+3.6' if f == 2023 else '+0.6'})")

    print(f"\n  채택 기준: (1) depthflag-base 양 fold 양수 (2) depthflag-dfplacebo 양 fold 양수"
          f" (3) 크기비 <=3 (4) 부트스트랩 구간이 0 을 크게 밑돌지 않을 것")
    print(f"  !! |Δ| < 10 은 무승부. 부호가 엇갈려도 각 fold 구간을 보고 판단한다.")
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
