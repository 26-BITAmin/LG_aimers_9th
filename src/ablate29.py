"""LightGBM 단조 제약 — 정보를 그대로 두고 가설 공간만 좁힌다

동기
    이 프로젝트에서 이긴 수는 일관되게 **"더 조이는 것"**이었다:
      lr 0.03→0.015 (+11) · min_data_in_leaf 2000 · lambda_l2 30 ·
      임베딩 16→4차원 + wd 1e-2 (−4670 → −55) · CatBoost iter 800→250 ·
      최대 이득이 피처 **제거**(+172).
    그리고 진 수는 전부 반대였다: 폭 확장 −75(§7.25), 입력차원 확장 −44~−285(§7.26).

    §7.13이 추정한 신호 총량 ≈ **0.004 nat**. 맞출 게 거의 없으면 가설 공간을 좁히는
    것이 곧 성능이다. **단조 제약은 정보를 하나도 안 버리면서 가설 공간만 좁힌다.**

    지금까지 죽은 사유 넷 중 어디에도 안 걸린다:
      정보 변경 ✗ (컬럼 그대로) · 용량 변경 ✗ (트리 수·리프 수 동일) ·
      학습분포 변경 ✗ (가중 동일) · 새 함수족 ✗ (같은 LightGBM)

    부수 기대: **단조 관계는 국소 평균보다 드리프트에 안정적이다.** 트리는 잎마다 학습
    분포의 지역 평균을 박고, 분포가 밀리면 그 값이 낡는다(§7.22의 가설). 단조 제약은
    그 잎 값들이 최소한 **순서**는 지키게 강제한다. 검증은 조건 ①②③이 한다.

    배포 영향 0 — 학습 파라미터일 뿐이고 Booster 포맷·추론 코드가 그대로다.

제약 방향 (도메인 지식. 다른 조건이 같을 때 제구 성공 확률에 대한 부호)
    +  통산/최근 제구 성공률, 스트라이크 비율
    -  볼 비율, 한가운데 비율, 포수 반대 방향 비율, (볼−스트라이크)
    타자 쪽 지표는 부호가 모호해서 **넣지 않는다** (asof_batter_success_rate가
    "타자 상대 성공률"인지 "타자의 성공률"인지 data_description으로 단정하기 어렵다).
    확신 없는 부호를 넣으면 제약이 아니라 오정보다.

후보
    mono_core    투수 제구율 계열 6개만            (가장 보수적)
    mono_wide    + 최근 3경기·구성비까지 12개       (용량 대조가 아니라 **범위 대조**)

사전 예측 (실행 전에 박아둔다)
    P1  mono_core가 트리 단독에서 양 fold + .      내 예측: **판단 보류**
        (§7.26에서 내 긍정 예측이 0승 4패였다. 이번엔 방향을 걸지 않는다 —
         근거 없는 예측을 적는 것은 채점을 오염시킬 뿐이다)
    P2  mono_wide < mono_core (부호가 덜 확실한 것을 넣을수록 나빠진다)
        내 예측: **그렇다** — 이건 도메인 확신도에 대한 예측이라 방향이 있다
    P3  트리 단독에서의 효과가 블렌드에서 **줄어든다** (MLP가 일부 흡수)
        내 예측: **그렇다** (§7.14의 "단일 측정은 앙상블 예측이 아니다")

실행:  .venv/Scripts/python.exe -u src/ablate29.py 2>&1 | tee experiments/ablate_run29.log
"""
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import lightgbm as lgb
import numpy as np

import ablate23 as A23
import foldcache
import mlp
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA, N_ROUNDS,
                    RECENCY_HALF_LIFE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from features import fit_artifacts
from foldcache import CACHE, LGB_SEEDS, _key

THREADS = 4
FOLDS = [2023, 2024]

# 부호가 도메인적으로 확실한 것만. 확신 없는 부호는 제약이 아니라 오정보다.
CORE = {
    "asof_pitcher_success_rate": +1,
    "asof_pitcher_success_rate_sm": +1,
    "asof_pitcher_strike_rate": +1,
    "asof_pitcher_strike_rate_sm": +1,
    "asof_pitcher_ball_rate": -1,
    "asof_pitcher_ball_rate_sm": -1,
}
WIDE = dict(CORE, **{
    "asof_pitcher_middle_rate": -1,
    "asof_pitcher_middle_rate_sm": -1,
    "asof_pitcher_reverse_rate": -1,
    "asof_pitcher_reverse_rate_sm": -1,
    "asof_pitcher_prev1_game_success_rate": +1,
    "asof_pitcher_prev3_game_success_rate": +1,
    "pitcher_ball_minus_strike": -1,
})
SETS = {"mono_core": CORE, "mono_wide": WIDE}


def ck(fold, name, vname, seed):
    return CACHE / f"mono_{name}_f{fold}_{_key()}_{vname}_{seed}.npz"


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = foldcache.load(FOLDS)

    need = [(f, n) for f in FOLDS for n in SETS
            if not all(ck(f, n, v, s).exists()
                       for v in ENSEMBLE_VARIANTS for s in LGB_SEEDS)]
    df = load_csv(TRAIN_CSV) if need else None

    Zn = {}
    for fold in FOLDS:
        D = None
        for name, spec in SETS.items():
            if all(ck(fold, name, v, s).exists()
                   for v in ENSEMBLE_VARIANTS for s in LGB_SEEDS):
                Zn[(fold, name)] = np.mean(
                    [np.load(ck(fold, name, v, s))["z"]
                     for v in ENSEMBLE_VARIANTS for s in LGB_SEEDS], axis=0)
                continue
            if D is None:
                season = df.season.to_numpy()
                tr, va = df[season < fold], df[season == fold]
                w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                                   ref_season=fold - 1)
                art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                                    k_batter=SMOOTH_K_BATTER)
                D = (A23.featurize(tr, art), tr[TARGET_COL].to_numpy(np.float64), w,
                     A23.featurize(va, art))
                del tr, va
                gc.collect()
                print(f"\n=== fold {fold}  피처 {D[0].shape[1]}개"
                      f"  ({time.time()-t0:.0f}s) ===")
            Xtr, ytr, w, Xva = D
            miss = [c for c in spec if c not in Xtr.columns]
            assert not miss, f"제약 컬럼이 피처에 없다: {miss}"   # 조용히 무시되면 안 된다
            mc = [spec.get(c, 0) for c in Xtr.columns]
            print(f"  [{name}] 제약 {sum(1 for v in mc if v)}개"
                  f"  (+{sum(1 for v in mc if v > 0)} / -{sum(1 for v in mc if v < 0)})")
            zs = []
            for vname, over in ENSEMBLE_VARIANTS.items():
                for s in LGB_SEEDS:
                    p = ck(fold, name, vname, s)
                    if p.exists():
                        zs.append(np.load(p)["z"])
                        continue
                    par = dict(FINAL_LGB_PARAMS, **over, seed=s, bagging_seed=s,
                               feature_fraction_seed=s, data_random_seed=s,
                               num_threads=THREADS, monotone_constraints=mc)
                    bst = lgb.train(par, lgb.Dataset(Xtr, ytr, weight=w),
                                    num_boost_round=N_ROUNDS)
                    z = bst.predict(Xva, raw_score=True)
                    np.savez_compressed(p, z=z)
                    zs.append(z)
                    del bst
            Zn[(fold, name)] = np.mean(zs, axis=0)
            print(f"  [{name}] 트리 단독 {mlp.score(Zn[(fold,name)], cached[fold][2]):>8.1f}"
                  f"  ({time.time()-t0:.0f}s)")
        del D
        gc.collect()
    del df
    gc.collect()

    print("\n" + "=" * 92)
    print(f"판정 — 트리 단독과 블렌드(α={MLP_ALPHA}) 양쪽에서 본다")
    print("=" * 92)
    print(f"{'설정':>12} " + " ".join(f"{'단독'+str(f)[-2:]:>10}" for f in FOLDS)
          + " ".join(f"{'블렌드'+str(f)[-2:]:>10}" for f in FOLDS)
          + f" {'Δ단독':>16} {'Δ블렌드':>16}")

    def bl(f, zl):
        zm, y = cached[f][1], cached[f][2]
        return mlp.score((1 - MLP_ALPHA) * zl + MLP_ALPHA * zm, y)

    b_solo = {f: mlp.score(cached[f][0], cached[f][2]) for f in FOLDS}
    b_bl = {f: bl(f, cached[f][0]) for f in FOLDS}
    print(f"{'기준(제약없음)':>12} " + " ".join(f"{b_solo[f]:>10.1f}" for f in FOLDS)
          + " ".join(f"{b_bl[f]:>10.1f}" for f in FOLDS))

    for name in SETS:
        so = {f: mlp.score(Zn[(f, name)], cached[f][2]) for f in FOLDS}
        bb = {f: bl(f, Zn[(f, name)]) for f in FOLDS}
        ds = {f: so[f] - b_solo[f] for f in FOLDS}
        db = {f: bb[f] - b_bl[f] for f in FOLDS}
        sign = len({np.sign(v) for v in db.values()}) == 1
        mags = sorted(abs(v) for v in db.values())
        ratio = mags[-1] / mags[0] if mags[0] > 1e-9 else float("inf")
        print(f"{name:>12} " + " ".join(f"{so[f]:>10.1f}" for f in FOLDS)
              + " ".join(f"{bb[f]:>10.1f}" for f in FOLDS)
              + f" {ds[FOLDS[0]]:>+7.1f}/{ds[FOLDS[1]]:>+7.1f}"
              + f" {db[FOLDS[0]]:>+7.1f}/{db[FOLDS[1]]:>+7.1f}"
              + f"   부호 {'O' if sign else 'X'}  크기비 {ratio:>5.1f}"
              + f"  → {'채택 후보' if sign and np.mean(list(db.values())) > 0 and ratio <= 3 else '기각'}")

    print("\n사전 예측 채점")
    dw = {f: bl(f, Zn[(f, 'mono_wide')]) - b_bl[f] for f in FOLDS}
    dc = {f: bl(f, Zn[(f, 'mono_core')]) - b_bl[f] for f in FOLDS}
    print(f"  P2 mono_wide < mono_core   "
          + "  ".join(f"fold{f} {dw[f]:+.2f} vs {dc[f]:+.2f}" for f in FOLDS)
          + f"   → {'적중' if all(dw[f] < dc[f] for f in FOLDS) else '빗나감'}")
    for name in SETS:
        ds = {f: mlp.score(Zn[(f, name)], cached[f][2]) - b_solo[f] for f in FOLDS}
        db = {f: bl(f, Zn[(f, name)]) - b_bl[f] for f in FOLDS}
        print(f"  P3 {name}: 단독 {np.mean(list(ds.values())):+.1f}"
              f" → 블렌드 {np.mean(list(db.values())):+.1f}"
              f"   {'줄어듦(적중)' if abs(np.mean(list(db.values()))) < abs(np.mean(list(ds.values()))) else '안 줄어듦'}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
