"""MLP 깊이·잔차 연결, 그리고 배깅 트리(LightGBM RF 모드)

동기
    §7.24가 판정 기준을 바꿨다. 좋은 앙상블 멤버인지는 "다른가"가 아니라
    **"단독 점수가 오르는가"**로 갈린다 (`noid`가 채택된 진짜 이유). 그리고 현행 MLP는
    fold2024 단독 653인데 LGBM은 752.5다 — **100점 격차**가 있다. 아키텍처로 그걸
    줄일 수 있으면 그 자체가 이득이고, §7.24의 새 기준도 만족한다.

    깊이는 한 번도 스윕된 적이 없다. `ablate17` ②는 임베딩 차원(4/8/없음)과
    폭(128/64)만 봤고 **층 수는 계속 2였다.**

    ⚠️ 반대 방향 증거가 있다. §7.13: 이 문제에서 얻어낼 수 있는 신호 총량은 약
    **0.004 nat**이고 초기 MLP는 **0.048 nat**을 학습했다 (없는 신호의 12배).
    "이긴 수는 항상 더 조이는 것"이었다. 깊이·잔차는 그 반대다.
    → 그래서 **`wide256`을 대조군으로 넣는다.** "깊이가 이득"과 "용량이 이득"을 가른다.
      깊이만 이기고 폭은 지면 구조 이득이고, 둘 다 이기면 그냥 용량 부족이었던 것이다.

후보
    deep3    depth 2 → 3            (폭 고정 128)
    res4     depth 4 + 잔차 연결     (항등 경로로 깊이의 최적화 부담을 덜어준다)
    wide256  hid 128 → 256          ← 용량 대조군
    rf       LightGBM `boosting=rf`  부스팅이 아니라 **배깅**. 학습 동역학이 다르고
                                     배포 코드가 한 줄도 안 는다 (여전히 Booster).

통제
    · MLP 셋은 `emb4`의 cfg를 그대로 물려받고 해당 축 하나만 바꾼다.
    · 에폭은 9 고정, **마지막 에폭 값**을 읽는다 (§7.13 규약: valid argmax 금지).
      OneCycleLR이 그 지점에서 LR을 0으로 어닐링하므로 자연스러운 종점이다.
    · `THREADS=4` — 캐시를 만든 값. 스레드 수는 BLAS 리덕션 순서를 바꿔 결과를
      1e-6 수준에서 흔든다 (실측). 변수를 하나 줄인다.
    · **fold2023용 `emb4` 대조군을 새로 학습한다.** 캐시에는 fold2024만 멤버별 z가
      있어서 fold2023에는 짝지을 대조가 없다. 대조 없는 비교는 하지 않는다.

사전 예측 (실행 전에 박아둔다)
    P1  deep3·res4 단독이 emb4 대조군 대비 양 fold에서 |Δ| < 20 (사실상 중립)
        내 예측: **그렇다** (신호가 얇아 용량이 이득이 아니다)
    P2  wide256 단독 < emb4 (용량 늘리면 진다)          내 예측: **그렇다**
    P3  rf 단독 핸디캡 < -150, corr(z_lgb) > 0.95        내 예측: **그렇다**
    P4  조건 ①②③ 통과 후보 수 = 0                       내 예측: **0개**

    P1이 빗나가면(깊이가 실제로 +) 그건 이 프로젝트에서 처음으로 **용량을 늘려서 이기는**
    사례이고, §7.13의 "항상 조이는 게 이긴다"를 수정해야 한다. 그때는 폭 대조군(P2)이
    같이 이겼는지를 봐서 구조 이득인지 용량 이득인지 가른다.

실행:  .venv/Scripts/python.exe -u src/ablate26.py 2>&1 | tee experiments/ablate_run26.log
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

import ablate23 as A
import foldcache
import mlp
from config import (MLP_ALPHA, MLP_SEEDS, MLP_VARIANTS, RECENCY_HALF_LIFE,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from features import fit_artifacts
from foldcache import CACHE, _key
from mlp_infer import encode_apply

THREADS = 4                      # 캐시를 만든 값. 바꾸면 1e-6 수준에서 결과가 달라진다
mlp.THREADS = THREADS

FOLDS = [2023, 2024]
SEEDS = MLP_SEEDS[:3]
N_OLD = sum(k for _, k in MLP_VARIANTS.values())        # 현행 풀 크기 (8)
EMB4 = MLP_VARIANTS["emb4"][0]

ARCH = {
    "emb4ctl": dict(EMB4),                              # 대조군 (fold2024는 캐시로 검증됨)
    "deep3":   dict(EMB4, depth=3),
    "res4":    dict(EMB4, depth=4, resid=True),
    "wide256": dict(EMB4, hid=256),                     # 용량 대조군
}

RF_PARAMS = dict(objective="binary", boosting="rf", num_leaves=127,
                 min_data_in_leaf=200, bagging_fraction=0.7, bagging_freq=1,
                 feature_fraction=0.7, lambda_l2=0.0, max_bin=127,
                 num_threads=THREADS, verbosity=-1)
RF_ROUNDS = 300

BETAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.273, 0.35, 0.50]


def ck(fold, kind, name, seed):
    return CACHE / f"arch_{kind}_{name}_f{fold}_{_key()}_{seed}.npz"


def fold_inputs(df, fold):
    season = df.season.to_numpy()
    tr, va = df[season < fold], df[season == fold]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=fold - 1)
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER)
    Xtr, Xva = A.featurize(tr, art), A.featurize(va, art)
    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    enc = mlp.fit_encoder(Xtr, art)
    An, Ctr = encode_apply(Xtr, enc)
    B, Cva = encode_apply(Xva, enc)
    del tr, va
    gc.collect()
    return dict(Xtr=Xtr, Xva=Xva, ytr=ytr, w=w, An=An, Ctr=Ctr, B=B, Cva=Cva, enc=enc)


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = foldcache.load(FOLDS)
    Z = {f: (cached[f][0], cached[f][1], cached[f][2]) for f in FOLDS}
    mem24 = cached[2024][3]                     # (8, n) — [0:3]이 emb4_42/7/2024

    todo = [(f, "mlp", n, s) for f in FOLDS for n in ARCH for s in SEEDS]
    todo += [(f, "lgb", "rf", s) for f in FOLDS for s in SEEDS]
    df = load_csv(TRAIN_CSV) if any(not ck(*t).exists() for t in todo) else None

    got = {}
    for fold in FOLDS:
        D = None
        for kind, name, s in [(k, n, s) for (f, k, n, s) in todo if f == fold]:
            p = ck(fold, kind, name, s)
            if p.exists():
                got[(fold, name, s)] = np.load(p)["z"]
                continue
            if D is None:
                D = fold_inputs(df, fold)
                print(f"\n=== fold {fold} 준비 완료 ({time.time()-t0:.0f}s) ===")
            if kind == "lgb":
                par = dict(RF_PARAMS, seed=s, bagging_seed=s,
                           feature_fraction_seed=s, data_random_seed=s)
                bst = lgb.train(par, lgb.Dataset(D["Xtr"], D["ytr"], weight=D["w"]),
                                num_boost_round=RF_ROUNDS)
                z = bst.predict(D["Xva"], raw_score=True)
                del bst
            else:
                m, _ = mlp.train_mlp(D["An"], D["Ctr"], D["ytr"], D["w"],
                                     D["An"][:512], D["Ctr"][:512], D["enc"]["sizes"],
                                     seed=s, verbose=False, **ARCH[name])
                z = mlp.infer(m, D["B"], D["Cva"])
                del m
            np.savez_compressed(p, z=z)
            got[(fold, name, s)] = z
            print(f"  {name}_{s}  단독 {mlp.score(z, Z[fold][2]):>8.1f}"
                  f"  ({time.time()-t0:.0f}s)")
        del D
        gc.collect()
    del df
    gc.collect()

    def avg(fold, name):
        return np.mean([got[(fold, name, s)] for s in SEEDS], axis=0)

    def bl(zl, zm, y, a=MLP_ALPHA):
        return mlp.score((1 - a) * zl + a * zm, y)

    # ── 0. 대조군 정합성 — fold2024의 emb4ctl은 캐시 멤버와 같아야 한다 ──
    d24 = np.abs(avg(2024, "emb4ctl") - mem24[:3].mean(0)).max()
    print(f"\n[정합성] fold2024 emb4ctl(3시드) vs 캐시 멤버[0:3]  max|diff| = {d24:.3e}"
          f"  {'일치' if d24 == 0 else '⚠️ 불일치'}")

    base = {f: bl(*Z[f]) for f in FOLDS}
    print("\n" + "=" * 92)
    print(f"1. 단독 성능 — §7.24의 판정 기준은 '단독 점수가 오르는가'다 (3시드 평균)")
    print("=" * 92)
    print(f"{'변형':>10} " + " ".join(f"{'fold'+str(f):>10}" for f in FOLDS)
          + " ".join(f"{'Δ대조'+str(f)[-2:]:>10}" for f in FOLDS)
          + f" {'corr(z_lgb)':>12} {'corr(블렌드)':>13}")
    ctl = {f: mlp.score(avg(f, "emb4ctl"), Z[f][2]) for f in FOLDS}
    for name in list(ARCH) + ["rf"]:
        row, dl = [], []
        for f in FOLDS:
            zl, zm, y = Z[f]
            sc = mlp.score(avg(f, name), y)
            row.append(sc)
            dl.append(sc - ctl[f])
        f = 2024
        zl, zm, y = Z[f]
        cur = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm
        print(f"{name:>10} " + " ".join(f"{v:>10.1f}" for v in row)
              + " ".join(f"{v:>+10.1f}" for v in dl)
              + f" {np.corrcoef(avg(f, name), zl)[0,1]:>12.3f}"
              + f" {np.corrcoef(avg(f, name), cur)[0,1]:>13.3f}")
    print(f"  (Δ대조 = emb4ctl 대비. rf는 트리라 참고용 — corr(z_lgb)를 봐야 한다)")

    # ── 2. 풀에 넣었을 때 ────────────────────────────────────────────
    print("\n" + "=" * 92)
    print(f"2. 채택 판정 — MLP는 풀에 β로 섞고, rf는 트리 쪽에 γ로 섞는다 (α={MLP_ALPHA} 고정)")
    print("=" * 92)
    print(f"  기준  " + "  ".join(f"fold{f} {base[f]:.1f}" for f in FOLDS))
    print(f"\n{'후보':>10} {'fold':>6} " + " ".join(f"{b:>8.3f}" for b in BETAS))
    for name in list(ARCH)[1:] + ["rf"]:
        for f in FOLDS:
            zl, zm, y = Z[f]
            zn = avg(f, name)
            out = []
            for b in BETAS:
                if name == "rf":
                    out.append(bl((1 - b) * zl + b * zn, zm, y) - base[f])
                else:
                    out.append(bl(zl, (1 - b) * zm + b * zn, y) - base[f])
            out[0] = 0.0
            print(f"{name:>10} {f:>6} " + " ".join(f"{v:>+8.2f}" for v in out))

    print("\n  β(γ)* > 0 이 **양 fold에서** 나와야 후보다. 조건 ③(크기비≤3)은 4전 4중이다.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
