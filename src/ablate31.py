"""`N_ROUNDS`를 변형별로 — 한 번 학습해서 곡선 전체를 읽는다

동기
    `N_ROUNDS = 320`이 **4개 변형 전부에** 똑같이 적용된다. 그런데 용량이 크게 다르다:
        base  31리프/2000     wide 127리프/8000
        extra 분할점 무작위화   lowff 피처 45%
    같은 지점에서 수렴할 이유가 없다.

    그리고 320의 출처가 `hl=0.5`와 **같은 지위**다 — "fold2에서 best_iteration=292였고
    최종은 소폭만 늘린다"(§5)로 정해진 뒤, **변형 4개 체제와 MLP 블렌드가 생기고 나서
    한 번도 재검토되지 않았다.** hl도 같은 상태였고 재보니 fold2024에서 −67.9였다.

    §7.18이 `rounds 480`을 한 번 봤지만 그건 **전 변형 일괄**이었고 fold2023이 −26.3이라
    fold2024를 재지도 않고 버렸다. 변형별로는 본 적이 없다.

설계 — 한 번 학습으로 곡선 전체
    600라운드까지 학습해두고 `predict(num_iteration=k)`로 읽으면 라운드별 z가 공짜다.
    변형 × 라운드 조합을 전부 훑어도 학습은 4회(변형 수)뿐이다.

    · 블렌드까지 본다. 캐시된 `z_mlp`를 그대로 쓰므로 추가 비용이 없다.
      §7.14: 트리 단독 측정은 앙상블 시스템에 대한 예측이 아니다.
    · **쌍체 SE를 같이 낸다** (§7.28의 규약). Δ만 보고 정점을 고르면
      SE가 함께 커지는 것을 놓친다 — PLE에서 실제로 밟은 함정이다.
    · 1시드다. 시드 노이즈 std가 4~9(§7.3)이므로 **여기서 나온 것은 스크리닝**이고,
      통과선을 넘으면 배포 구조(4변형 × 3시드)로 다시 잰다.

사전 예측
    P1  변형별 최적 라운드가 서로 다르다 (`wide`가 `base`보다 늦게 정점).
        내 예측: **그렇다** — 용량이 4배 차이 나는데 같을 리 없다. 이건 방향에
        확신이 있는 예측이다 (구조적 근거).
    P2  그럼에도 **블렌드 기준으로 fold2024에서 t ≥ 3을 넘는 조합은 없다.**
        내 예측: **그렇다** — §7.18이 라운드 축에서 이미 평평했고, 이 프로젝트의
        기저율이 그렇다. (내 긍정 예측 0승 5패를 감안해 기저율에 건다.)

    P1이 맞고 P2가 틀리면 그게 이번 세션 첫 후보다.
    P1이 맞고 P2도 맞으면 "변형별로 다르긴 한데 그 차이가 점수로는 안 나온다"가 되고,
    라운드 축은 닫힌다.

실행:  .venv/Scripts/python.exe -u src/ablate31.py 2>&1 | tee experiments/ablate_run31.log
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
                    RECENCY_HALF_LIFE, SEED, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from features import fit_artifacts
from foldcache import CACHE, _key
from validation import shift_to_prior

THREADS = 5
FOLDS = [2023, 2024]
MAX_ROUNDS = 600
GRID = [120, 200, 280, 320, 400, 480, 600]      # 320 = 배포값


def ck(fold, vname):
    return CACHE / f"rounds_{vname}_f{fold}_{_key()}_{SEED}.npz"


def paired(pA, pB, y):
    """BSS(A) − BSS(B)와 쌍체 SE. 같은 행이므로 차분에서 표본변동이 소거된다."""
    ref = y.mean() * (1 - y.mean())
    k = 100000.0 / ref
    d = (pB - y) ** 2 - (pA - y) ** 2
    return k * d.mean(), k * d.std(ddof=1) / np.sqrt(len(d))


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = foldcache.load(FOLDS)

    need = [(f, v) for f in FOLDS for v in ENSEMBLE_VARIANTS if not ck(f, v).exists()]
    df = load_csv(TRAIN_CSV) if need else None

    Z = {}          # (fold, variant) -> (라운드, n) 행렬
    for fold in FOLDS:
        D = None
        for vname, over in ENSEMBLE_VARIANTS.items():
            p = ck(fold, vname)
            if p.exists():
                Z[(fold, vname)] = np.load(p)["z"]
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
                print(f"\n=== fold {fold}  ({time.time()-t0:.0f}s) ===")
            Xtr, ytr, w, Xva = D
            par = dict(FINAL_LGB_PARAMS, **over, seed=SEED, bagging_seed=SEED,
                       feature_fraction_seed=SEED, data_random_seed=SEED,
                       num_threads=THREADS)
            bst = lgb.train(par, lgb.Dataset(Xtr, ytr, weight=w),
                            num_boost_round=MAX_ROUNDS)
            # 한 번 학습해두고 라운드별로 읽는다 — 여기가 이 실험이 싼 이유다
            zz = np.stack([bst.predict(Xva, raw_score=True, num_iteration=k)
                           for k in GRID])
            np.savez_compressed(p, z=zz)
            Z[(fold, vname)] = zz
            del bst
            y = cached[fold][2]
            best = GRID[int(np.argmax([mlp.score(zz[i], y) for i in range(len(GRID))]))]
            print(f"  {vname:>6}  정점 {best:>3}라운드"
                  + "  " + " ".join(f"{k}:{mlp.score(zz[i], y):.0f}"
                                    for i, k in enumerate(GRID))
                  + f"   ({time.time()-t0:.0f}s)")
        del D
        gc.collect()
    del df
    gc.collect()

    print("\n" + "=" * 96)
    print("1. 변형별 라운드 곡선 (트리 단독, 1시드, BSS oracle)")
    print("=" * 96)
    print(f"{'fold':>6} {'변형':>7} " + " ".join(f"{k:>7}" for k in GRID) + f" {'정점':>7}")
    best_r = {}
    for fold in FOLDS:
        for vname in ENSEMBLE_VARIANTS:
            y = cached[fold][2]
            s = [mlp.score(Z[(fold, vname)][i], y) for i in range(len(GRID))]
            best_r[(fold, vname)] = GRID[int(np.argmax(s))]
            print(f"{fold:>6} {vname:>7} " + " ".join(f"{v:>7.1f}" for v in s)
                  + f" {best_r[(fold,vname)]:>7}")

    print("\n" + "=" * 96)
    print(f"2. 블렌드 판정 (α={MLP_ALPHA}, 캐시 z_mlp). 기준 = 전 변형 {N_ROUNDS}라운드")
    print("=" * 96)
    i320 = GRID.index(N_ROUNDS)

    def blend_p(fold, pick):
        """pick: 변형 -> 라운드 인덱스"""
        zl = np.mean([Z[(fold, v)][pick[v]] for v in ENSEMBLE_VARIANTS], axis=0)
        zm, y = cached[fold][1], cached[fold][2]
        z = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm
        return shift_to_prior(1 / (1 + np.exp(-z)), y.mean())

    base_pick = {v: i320 for v in ENSEMBLE_VARIANTS}
    cands = {"일괄 320 (기준)": base_pick}
    for k in GRID:
        if k != N_ROUNDS:
            cands[f"일괄 {k}"] = {v: GRID.index(k) for v in ENSEMBLE_VARIANTS}
    # 변형별 최적을 fold마다 따로 고르면 valid argmax다. 두 fold **공통**으로만 쓴다.
    for f in FOLDS:
        cands[f"변형별 최적(fold{f}에서 선택)"] = {v: GRID.index(best_r[(f, v)])
                                             for v in ENSEMBLE_VARIANTS}

    print(f"{'후보':>26} " + " ".join(f"{'Δfold'+str(f):>18}" for f in FOLDS))
    for name, pick in cands.items():
        if name.startswith("일괄 320"):
            continue
        out = []
        for f in FOLDS:
            m, se = paired(blend_p(f, pick), blend_p(f, base_pick), cached[f][2])
            out.append(f"{m:+7.2f} ± {se:4.2f} t{m/se:+5.1f}")
        print(f"{name:>26} " + " ".join(f"{o:>18}" for o in out))

    print("\n사전 예측 채점")
    diff = {f: len({best_r[(f, v)] for v in ENSEMBLE_VARIANTS}) > 1 for f in FOLDS}
    print(f"  P1 변형별 최적 라운드가 다르다   " +
          "  ".join(f"fold{f}: " + "/".join(f"{v}={best_r[(f,v)]}"
                                            for v in ENSEMBLE_VARIANTS) for f in FOLDS))
    print(f"     → {'적중' if any(diff.values()) else '빗나감'}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
