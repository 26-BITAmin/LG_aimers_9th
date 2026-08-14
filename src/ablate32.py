"""마지막 "초기에 정하고 방치된" 파라미터 둘 — `SMOOTH_K`, `max_bin`

동기
    `hl=0.5`와 `N_ROUNDS=320`이 같은 프로필이었다 — 초기에 한 fold에서 정하고 방치.
    재보니 **결론이 반대**였다 (hl은 바꾸면 −67.9, 라운드는 바꿔도 0).
    프로필이 같다고 위험이 같은 게 아니므로 남은 것도 재봐야 안다.

    `SMOOTH_K = 200`  §7.2에서 k=50/200/800을 **단일 fold·단일 모델·MLP 도입 전**에
        보고 "k는 지배적이지 않다"로 끝냈다. hl과 정확히 같은 지위다.
    `max_bin = 127`   LightGBM 기본값은 255다. 왜 127인지 기록이 없다.
        **한 번도 스윕된 적이 없다.** 분할 해상도를 낮추는 정규화 장치다.

기전이 갈리는 지점 — 그래서 MLP까지 다시 학습한다
    `SMOOTH_K`가 트리에 미치는 영향은 작을 것이다: §7.11에서 `_sm` 그룹 **전체 제거**가
    +11.0이었다(거의 중복). 트리는 원본 rate와 `asof_*_n`을 갖고 있어 스무딩을 스스로 한다.

    **그런데 MLP는 다르다.** `mlp.fit_encoder`의 `nan_plan`이 결측을 **`_sm` 값으로 채운다.**
    즉 k는 MLP에게 "이력 없는 선수를 어디로 보낼 것인가"를 정하는 값이다.
    §7.24에서 원본 asof를 빼니 MLP가 −63~−113이었다 — MLP는 이 계열에 민감하다.
    그리고 **2025는 fold들보다 미지 선수 비율이 높다**(CLAUDE.md가 처음부터 짚은 지점).
    → 캐시된 `z_mlp`(k=200 고정)를 그대로 쓰면 이 경로를 못 본다. MLP를 다시 학습한다.

    `max_bin`은 트리 전용이므로 캐시 `z_mlp`를 그대로 쓴다.

설계
    · 스크리닝이다 — `base` 변형 1개 × 1시드. 통과선을 넘으면 배포 구조로 다시 잰다.
    · 기준도 같은 조건(base 1시드)으로 다시 계산한다. 캐시 z_lgb(12부스터)와 섞지 않는다.
    · **Δ와 쌍체 SE를 같이 낸다** (§7.28 규약).

사전 예측
    P1  `max_bin` 255가 fold2024에서 |t| < 2.   내 예측: **그렇다** (기저율)
    P2  `SMOOTH_K` 변경이 **트리보다 MLP에서 크게** 움직인다.
        내 예측: **그렇다** — 위 기전에 근거가 있다. 방향은 걸지 않는다.
    P3  블렌드 기준 fold2024에서 t ≥ 3인 조합이 하나도 없다.  내 예측: **그렇다**

실행:  .venv/Scripts/python.exe -u src/ablate32.py 2>&1 | tee experiments/ablate_run32.log
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
from config import (FINAL_LGB_PARAMS, MLP_ALPHA, MLP_SEEDS, MLP_VARIANTS,
                    N_ROUNDS, RECENCY_HALF_LIFE, SEED, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from features import fit_artifacts
from foldcache import CACHE, _key
from mlp_infer import encode_apply
from validation import shift_to_prior

THREADS = 5
mlp.THREADS = THREADS
FOLDS = [2023, 2024]
SEEDS = MLP_SEEDS[:3]

KS = [50, 200, 800]              # 200 = 배포값
BINS = [63, 127, 255]            # 127 = 배포값
EMB4 = MLP_VARIANTS["emb4"][0]


def ck(fold, kind, tag):
    return CACHE / f"sk_{kind}_{tag}_f{fold}_{_key()}.npz"


def paired(pA, pB, y):
    ref = y.mean() * (1 - y.mean())
    c = 100000.0 / ref
    d = (pB - y) ** 2 - (pA - y) ** 2
    return c * d.mean(), c * d.std(ddof=1) / np.sqrt(len(d))


def tree(Xtr, ytr, w, Xva, mb):
    p = dict(FINAL_LGB_PARAMS, seed=SEED, bagging_seed=SEED,
             feature_fraction_seed=SEED, data_random_seed=SEED,
             num_threads=THREADS, max_bin=mb)
    bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=N_ROUNDS)
    z = bst.predict(Xva, raw_score=True)
    del bst
    return z


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = foldcache.load(FOLDS)
    df = load_csv(TRAIN_CSV)
    season = df.season.to_numpy()
    Z = {}

    for fold in FOLDS:
        tr, va = df[season < fold], df[season == fold]
        w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=fold - 1)
        ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
        yva = cached[fold][2]
        print(f"\n=== fold {fold}  ({time.time()-t0:.0f}s) ===")

        for k in KS:
            art = fit_artifacts(tr, weights=w, k_pitcher=float(k), k_batter=float(k))
            Xtr, Xva = A23.featurize(tr, art), A23.featurize(va, art)

            # 트리 (max_bin은 배포값 고정, k만 움직인다)
            p = ck(fold, "tree", f"k{k}_mb127")
            z = np.load(p)["z"] if p.exists() else tree(Xtr, ytr, w, Xva, 127)
            if not p.exists():
                np.savez_compressed(p, z=z)
            Z[(fold, "tree", k, 127)] = z
            print(f"  k={k:<4} 트리          단독 {mlp.score(z, yva):>8.1f}"
                  f"  ({time.time()-t0:.0f}s)")

            # max_bin은 k=배포값에서만 훑는다 (두 축을 같이 흔들지 않는다)
            if k == SMOOTH_K_PITCHER:
                for mb in BINS:
                    if mb == 127:
                        continue
                    p = ck(fold, "tree", f"k{k}_mb{mb}")
                    z = np.load(p)["z"] if p.exists() else tree(Xtr, ytr, w, Xva, mb)
                    if not p.exists():
                        np.savez_compressed(p, z=z)
                    Z[(fold, "tree", k, mb)] = z
                    print(f"  max_bin={mb:<4} 트리      단독 {mlp.score(z, yva):>8.1f}"
                          f"  ({time.time()-t0:.0f}s)")

            # MLP — k가 결측 대체값을 바꾸므로 여기가 진짜 관심사다
            enc = mlp.fit_encoder(Xtr, art)
            A_, C_ = encode_apply(Xtr, enc)
            B_, Cv_ = encode_apply(Xva, enc)
            zs = []
            for s in SEEDS:
                p = ck(fold, "mlp", f"k{k}_{s}")
                if p.exists():
                    zs.append(np.load(p)["z"])
                    continue
                m, _ = mlp.train_mlp(A_, C_, ytr, w, A_[:512], C_[:512], enc["sizes"],
                                     seed=s, verbose=False, **EMB4)
                zs.append(mlp.infer(m, B_, Cv_))
                np.savez_compressed(p, z=zs[-1])
                del m
            Z[(fold, "mlp", k, 0)] = np.mean(zs, axis=0)
            print(f"  k={k:<4} MLP(3시드)    단독 "
                  f"{mlp.score(Z[(fold,'mlp',k,0)], yva):>8.1f}  ({time.time()-t0:.0f}s)")
            del Xtr, Xva, A_, C_, B_, Cv_, enc, art
            gc.collect()
        del tr, va
        gc.collect()
    del df
    gc.collect()

    def bl(fold, zl, zm):
        y = cached[fold][2]
        z = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm
        return shift_to_prior(1 / (1 + np.exp(-z)), y.mean())

    K0, MB0 = int(SMOOTH_K_PITCHER), 127
    print("\n" + "=" * 96)
    print(f"판정 — 기준 = k{K0} / max_bin {MB0}, base 1시드 + MLP 3시드, α={MLP_ALPHA}")
    print("=" * 96)
    print(f"{'변경':>22} " + " ".join(f"{'Δfold'+str(f):>20}" for f in FOLDS))
    base = {f: bl(f, Z[(f, "tree", K0, MB0)], Z[(f, "mlp", K0, 0)]) for f in FOLDS}

    rows = [(f"SMOOTH_K {K0}→{k}", lambda f, k=k: (Z[(f, "tree", k, MB0)],
                                                   Z[(f, "mlp", k, 0)]))
            for k in KS if k != K0]
    rows += [(f"max_bin {MB0}→{mb}", lambda f, mb=mb: (Z[(f, "tree", K0, mb)],
                                                       Z[(f, "mlp", K0, 0)]))
             for mb in BINS if mb != MB0]
    for name, get in rows:
        out = []
        for f in FOLDS:
            zl, zm = get(f)
            m, se = paired(bl(f, zl, zm), base[f], cached[f][2])
            out.append(f"{m:+7.2f} ± {se:4.2f} t{m/se:+5.1f}")
        print(f"{name:>22} " + " ".join(f"{o:>20}" for o in out))

    print("\n  성분별 (단독, 기준 대비) — SMOOTH_K가 트리보다 MLP에서 크게 움직이는가")
    for f in FOLDS:
        y = cached[f][2]
        bt = mlp.score(Z[(f, "tree", K0, MB0)], y)
        bm = mlp.score(Z[(f, "mlp", K0, 0)], y)
        for k in KS:
            if k == K0:
                continue
            print(f"    fold{f}  k={k:<4}  트리 {mlp.score(Z[(f,'tree',k,MB0)], y)-bt:>+7.1f}"
                  f"   MLP {mlp.score(Z[(f,'mlp',k,0)], y)-bm:>+7.1f}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
