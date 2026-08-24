"""§7.133 후속 3 — §7.94 가 남긴 열린 질문 + §7.92 묶음의 마지막 빈 칸.

이 스크립트는 **원래 계획(drop3)이 prior work 에 반증돼서** 다시 짠 것이다.
경위를 남긴다 — 같은 실수를 또 하지 않으려면 왜 폐기됐는지가 결과보다 중요하다.

폐기된 계획과 그 이유
    계획: miss_batter_asof + miss_pitcher_asof + top_bottom 을 묶어 드랍.
    근거로 "top_bottom 은 is_pitcher_home 과 완전 중복이라 제거가 해로울 이유가
    없다"고 썼다. **그건 §7.93 이 세웠다가 틀린 바로 그 사전 예측이다**
    (Δ ∈ [-2,+4] 예상 -> 배포층 실측 **-5.21 / -3.18**). 반증된 예측을 다시 썼다.
    게다가 알려진 음수(top_bottom)를 미측정 둘과 묶는 것은 §7.92 를 비판한
    바로 그 묶음 오류다 ([[knobs-vs-channels]]).

    그리고 내가 계산한 "나머지 3개 = +3.3/+5.2" 도 철회한다. 배포층 수치로
    다시 하면 +8.1/+8.4 가 되는데, 이는 **세 실험(§7.92·§7.93·§7.133)을 가로지르는
    삼중 소거**이고 하네스가 셋 다 다르며 가법성을 가정한다 — §7.133 이 방금
    그 가법성이 깨지는 사례를 보여줬다 ([[telescoping-chains-are-not-evidence]]).

팔 (base 는 §7.133 캐시 재사용)

    base       현행 78 컬럼

    dupsig     `asof_pitcher_success_rate` 를 그대로 복사해 79 컬럼 (§7.94 의 열린 질문)
               > "신호 있는 축을 의도적으로 1->2 복제하면 §7.93 방향의 이득이 나오는가.
               >  죽은 축 복제가 0 인 것과 모순 없이 양립 가능한 가설이고, 아직 안 재봤다."
               고른 축 근거: §C 신호 지도에서 단독 BSS 198.4 / 315.4 로 **최상위**이고
               §7.94 가 예시로 지목한 바로 그 컬럼이다.
               예측 근거 두 개가 서로 맞물린다 —
                 §7.93  신호 축 중복 **제거** = -5.21 / -3.18 (아프다)
                 §7.94  죽은 축 복제 **추가** = LGBM 비트 동일, MLP t -0.6/-0.2 (0)
               -> "신호 축 복제 추가"는 양수를 기대할 근거가 있다. 그게 이 팔이다.
               ⚠️ 알려진 비대칭 하나: 원본은 `NAN_COLS` 라 MLP `nan_plan` 이 `_sm`->prior
                  로 채우지만 복사본은 거기 없어 0 으로 떨어진다. 해당 행이 **792 개
                  (0.054%)** 라 ±7~9 노이즈 대비 무시할 수준이지만 기록해 둔다.
                  LGBM 쪽은 NaN 네이티브라 두 복사본이 완전히 동일하다.

    dropmiss2  miss_batter_asof + miss_pitcher_asof 드랍 (76 컬럼)
               §7.92 묶음 4개 중 **개별로 재본 적 없는 유일한 칸**이다
               (miss_prev_game -> §7.133, top_bottom/is_pitcher_home -> §7.93).
               ⚠️ 기대값은 낮다. §7.94 가 `miss_batter_asof` 의 total_gain 이 **정확히 0**
                  이고, 복제 추가 시 **LGBM 12 부스터 예측이 비트 동일**임을 보였다 —
                  gain 0 컬럼은 feature_fraction 샘플링 경로에조차 개입하지 않는다.
               -> **falsifiable 예측: 이 팔의 z_lgb 는 base 와 비트 동일해야 한다.**
                  아래에서 직접 검사한다. 어긋나면 §7.94 의 기전 설명이 틀린 것이고,
                  그건 이 팔의 점수보다 중요한 발견이다.

검정력 사전 고지 (§7.133 에서 측정)
    컬럼 수를 **바꾸는** 조작의 부트스트랩 폭은 ±7~9 다 (값만 바꾸면 ±1.4~3.3).
    두 팔 다 컬럼 수를 바꾸므로 이 라운드가 확실히 잡는 것은 |효과| >= 8 뿐이다.
    무승부는 "효과 없음"이 아니라 **"이 하네스로는 못 잰다"** 로 읽는다.

양성 대조군 (재실행 없음, 하네스가 진짜 손실을 잡는다는 증거)
    §7.133 basenf (miss_prev_game 단독 제거) = -5.0 / -8.3

채택 기준 (실행 전에 박아둔다)
    (1) 양 fold 양수  (2) 크기비 <= 3  (3) 부트스트랩 구간이 0 을 크게 밑돌지 않을 것
    !! |Δ| < 8 은 분해능 밖. 부호가 엇갈려도 각 fold 구간을 보고 판단한다
       ([[sign-isnt-evidence-when-the-interval-spans-zero]] — 이번 라운드에서 이미
        한 번, **채택 쪽으로** 이 함정에 빠졌다).

내 예측
    dupsig     가장 볼 만한 팔. 그래도 무승부(|Δ| < 8)가 최빈값이라고 본다.
               §7.93 의 -5.2/-3.2 는 **제거** 방향이고, 추가 방향이 대칭이라는 보장이 없다.
    dropmiss2  LGBM 비트 동일 + MLP 무효 -> 블렌드 |Δ| < 3. +8.1/+8.4 는 소거 산술의
               인공물일 것이다.

실행:  .venv/Scripts/python.exe -u src/dupdrop.py 2>&1 | tee experiments/dupdrop.log
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

# MLP 멤버 8 고정 — base 가 §7.133 의 8멤버 캐시다. 지금 config 도 m8 이지만
# (§7.130 이 s32 를 -2.30 으로 재고 롤백했다) **다른 세션이 실행 중에 또 바꿀 수
# 있으므로** 읽지 말고 고정한다. §7.133 후속의 첫 실행이 정확히 그렇게 죽었다.
MLP_SEEDS = _MLP_SEEDS_CUR[:5]
MLP_VARIANTS = {"emb4": (_MLP_VARIANTS_CUR["emb4"][0], 5),
                "noid": (_MLP_VARIANTS_CUR["noid"][0], 3)}
assert sum(n for _, n in MLP_VARIANTS.values()) == 8, "멤버 수가 8 이 아니다"

from data import load_csv, recency_weight                           # noqa: E402
from initfix3 import level_init                                     # noqa: E402
from validation import bss, shift_to_prior                          # noqa: E402

FOLDS = [2023, 2024]
LGB_SEEDS = SEED_ENSEMBLE[:3]
MODES = ["base", "dupsig", "dropmiss2"]
DUP_SRC = "asof_pitcher_success_rate"
DUP_COL = DUP_SRC + "_dup"
DROPCOL = {"dropmiss2": ["miss_batter_asof", "miss_pitcher_asof"]}
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
REUSE = {"base": "histdepth"}
SEED = 42
ANCHOR = {2023: 807.2, 2024: 924.5}
POSCTL = {2023: -5.0, 2024: -8.3}       # §7.133 basenf
FP = {}


def _sig(v):
    return hashlib.sha1(np.ascontiguousarray(np.nan_to_num(
        np.asarray(v, dtype=np.float64), nan=-9.0)).tobytes()).hexdigest()[:10]


def compute(df, fold, mode, t0):
    pth = CACHE / (f"{REUSE[mode]}_fold{fold}_{mode}.npz" if mode in REUSE
                   else f"dupdrop_fold{fold}_{mode}.npz")
    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = F.fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                          k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

    assert F.HIST_MODE is None and F.EXTRA_PROD is None, "다른 스위치가 켜져 있다"

    def X(d):
        x = F.build_features(d, art)
        if mode == "dupsig":
            # 순수 복제 — NaN 패턴까지 그대로. 파생도 변환도 없다.
            x[DUP_COL] = x[DUP_SRC].to_numpy(dtype=np.float32)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        drop = [c for c in DROP_FEATURES if c in x.columns] + DROPCOL.get(mode, [])
        return x.drop(columns=drop)

    Xva = X(va)
    # ── 팔 검증 ──────────────────────────────────────────────
    if mode == "dupsig":
        a = Xva[DUP_SRC].to_numpy(dtype=np.float32)
        b = Xva[DUP_COL].to_numpy(dtype=np.float32)
        assert np.array_equal(a, b, equal_nan=True), "복제본이 원본과 다르다"
        assert a.dtype == b.dtype, "dtype 이 다르다"
    else:
        assert DUP_COL not in Xva.columns
    for c in DROPCOL.get(mode, []):
        assert c not in Xva.columns, f"{mode}: {c} 가 안 빠졌다"
    assert "miss_prev_game" in Xva.columns, "miss_prev_game 은 전 팔에서 유지"
    assert "top_bottom" in Xva.columns and "is_pitcher_home" in Xva.columns, \
        "§7.93 이 닫은 중복쌍은 이번에 안 건드린다"
    FP[(fold, mode)] = {"ncol": len(Xva.columns), "mp": _sig(Xva["miss_prev_game"])}
    print(f"    [{mode:9s}] 피처 {len(Xva.columns)}개  "
          f"{'복제 ' + DUP_COL if mode == 'dupsig' else ('드랍 ' + str(DROPCOL[mode]) if mode in DROPCOL else '변경 없음')}")

    if pth.exists():
        print(f"    캐시 적중 {pth.name}")
        with np.load(pth) as d:
            return d["z_lgb"], d["z_mlp"]

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
    print(f"train {df.shape}   a={MLP_ALPHA}  init_score=ON  팔 {MODES}")
    print(f"양성 대조군: §7.133 basenf = {POSCTL[2023]:+.1f} / {POSCTL[2024]:+.1f}")
    print(f"분해능 고지: 컬럼 수 변경 조작의 부트스트랩 폭 ±7~9 -> |Δ|<8 은 못 잰다\n")

    R, KEEP, ZL = {}, {}, {}
    for fold in FOLDS:
        Xax, _, _, _ = J.axis_matrix(df, s, fold)
        vi = np.where(s == fold)[0]
        mm = mclean[vi]
        yv = y[vi]
        arms = {m: compute(df, fold, m, t0) for m in MODES}
        for m, (zl, _) in arms.items():
            ZL[(fold, m)] = zl

        assert FP[(fold, "base")]["ncol"] + 1 == FP[(fold, "dupsig")]["ncol"] \
            == FP[(fold, "dropmiss2")]["ncol"] + 3, "컬럼 수가 설계와 다르다"
        assert FP[(fold, "base")]["mp"] == FP[(fold, "dupsig")]["mp"] \
            == FP[(fold, "dropmiss2")]["mp"], "miss_prev_game 이 팔마다 다르다"

        for lab, (zl, zm) in arms.items():
            z = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm + Xax @ dep
            p = shift_to_prior(1 / (1 + np.exp(-z[mm])), yv[mm].mean())
            R[(fold, lab)] = bss(yv[mm], p)
            KEEP[(fold, lab)] = (yv[mm], z[mm], codes[vi][mm])
        print(f"  fold {fold}: " + "  ".join(f"{k} {R[(fold,k)]:.1f}" for k in MODES))
        d0 = R[(fold, "base")] - ANCHOR[fold]
        print(f"    앵커 검증: base - §7.133 기록({ANCHOR[fold]}) = {d0:+.2f}  "
              f"{'OK' if abs(d0) < 0.1 else '!! 채점 경로가 바뀌었다 — 비교 무효'}")

    print(f"\n{'='*92}\n█ 판정 — 배포 층 (a={MLP_ALPHA} + delta축 + prior shift, clean 행)\n{'='*92}")
    print(f"  {'팔':<11}{'컬럼':>6}" + "".join(f"{f'fold {f}':>11}" for f in FOLDS)
          + "".join(f"{f'Δ vs base ({f})':>17}" for f in FOLDS))
    for lab in MODES:
        print(f"  {lab:<11}{FP[(FOLDS[0],lab)]['ncol']:>6}"
              + "".join(f"{R[(f,lab)]:>11.1f}" for f in FOLDS)
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
        print(f"     {lab:<11}" + "".join(f"{c:>26}" for c in cells))

    # ── §7.94 기전의 falsifiable 예측 ────────────────────────
    print(f"\n  ★ §7.94 기전 검정 — gain 0 컬럼 제거는 LGBM 에 무개입이어야 한다")
    for f in FOLDS:
        zb, zd = ZL[(f, "base")], ZL[(f, "dropmiss2")]
        same = np.array_equal(zb, zd)
        mad = float(np.abs(zb - zd).max())
        print(f"     fold {f}: z_lgb 비트 동일 = {same}   max|Δz| = {mad:.3e}"
              f"   {'예측 적중' if same else '!! §7.94 의 기전 설명이 틀렸다 — 점수보다 중요한 발견'}")

    print(f"\n  소거 산술 대조 (하네스가 달라 직접 비교 불가, 방향만)")
    print(f"     삼중 소거 예측: dropmiss2 = +8.1 / +8.4")
    print(f"     기전  예측: dropmiss2 ≈ 0 (LGBM 무개입 + MLP null)")
    print(f"     실측       : {R[(2023,'dropmiss2')]-R[(2023,'base')]:+.1f} / "
          f"{R[(2024,'dropmiss2')]-R[(2024,'base')]:+.1f}")

    print(f"\n  채택 기준: (1) 양 fold 양수 (2) 크기비 <=3 (3) 부트스트랩 구간이 0 을 크게 밑돌지 않을 것")
    print(f"  !! |Δ| < 8 은 분해능 밖 — '효과 없음'이 아니라 '못 잰다'로 읽는다.")
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
