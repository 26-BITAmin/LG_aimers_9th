"""수치 피처 임베딩 (PLE / RFF) — 문헌에서 가져온 첫 후보

출처
    Gorishniy et al., "On Embeddings for Numerical Features in Tabular Deep Learning"
    (NeurIPS 2022, arXiv:2203.05556). 공식 구현 `rtdl_num_embeddings`.
    후속 벤치마크(RealMLP / TabM 계열 평가)에서도 "vanilla MLP는 약하지만 MLP-PLR은
    GBDT와의 격차를 상당히 메운다"가 반복 확인됐다.

왜 이게 우리 문제에 맞는가
    · MLP는 표준화된 스칼라를 그대로 먹고, 트리는 같은 값을 **구간으로 쪼갠다.**
    · fold2024에서 MLP 단독 700.6 vs LGBM 752.5 — **100점 격차.**
      이 문제는 극단적으로 GBDT 친화적이고, 논문의 주장이 정확히 그 레짐을 겨냥한다.
    · §7.24가 정한 후보 조건("단독 점수가 오르는가")을 정면으로 겨냥한다.
    · PLE는 **고정 변환**이라 채택 시 배포가 numpy 몇 줄이다 (`numemb.apply_np`).

왜 §7.25(폭 확장 -75)의 재탕이 아닌가 — 그리고 그 위험을 어떻게 통제했나
    PLE는 첫 층 입력을 늘리므로 용량 증가이기도 하다. `wide256`이 -75였으니 위험하다.
    다만 성격이 다르다 — 자유로운 비선형 용량이 아니라 **각 피처에 대해 단조·구간선형인
    고정 재매개화**다 (트리가 하는 일과 같은 형태).
    → **`ple4`를 `ple8`의 대조군으로 넣는다.** 구간이 많을수록 좋으면 해상도가 병목이고,
      적을수록 좋으면 여전히 용량이 병목이다. §7.25에서 `wide256`이 한 역할과 같다.

    ⚠️ 이 기계는 RAM 15.3GB(여유 2.4GB)다. PLE를 미리 펼치면 fold2024 학습셋이 2.3GB다.
      그래서 **배치 안에서 만든다** (`mlp.Net.numemb_apply`). 공식 구현도 같은 형태다.

구현 검증 (이미 통과)
    학습은 torch, 배포는 numpy를 쓰므로 두 구현이 어긋나면 조용히 성능만 깎인다.
    `numemb.apply_np` vs `Net.numemb_apply`  →  PLE **0.000e+00**, RFF 5.96e-08(1 ULP).
    이 스크립트가 실제 fold 데이터로 다시 대조한다.

사전 예측 (실행 전에 박아둔다. **이번엔 처음으로 긍정 쪽에 건다**)
    P1  ple8 단독이 emb4ctl 대비 **양 fold에서 +**        내 예측: **그렇다**
        (논문의 핵심 주장이고 우리 문제가 정확히 그 레짐이다)
    P2  ple4 ≥ ple8 (구간을 늘려도 이득이 없다)            내 예측: **그렇다**
        (§7.13의 "신호 총량 0.004 nat" — 해상도가 아니라 용량이 병목일 것)
    P3  rff8은 중립~악화                                  내 예측: **그렇다**
        (주기 기저는 asof 비율의 단조 관계와 맞지 않는다)
    P4  조건 ①②③ 통과 후보 ≥ 1개                          내 예측: **1개 이상**

    P1과 P4가 빗나가면 "MLP 함수족은 이 문제에서 포화"(§7.25)가 한 번 더 확인되고,
    문헌 쪽 첫 시도도 닫힌다.

실행:  .venv/Scripts/python.exe -u src/ablate27.py 2>&1 | tee experiments/ablate_run27.log
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

import numpy as np
import torch

import ablate23 as A23
import ablate26 as A26
import foldcache
import mlp
import numemb
from config import (MLP_ALPHA, MLP_SEEDS, MLP_VARIANTS, RECENCY_HALF_LIFE,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from features import fit_artifacts
from foldcache import CACHE, _key
from mlp_infer import encode_apply

THREADS = 4
mlp.THREADS = THREADS

FOLDS = [2023, 2024]
SEEDS = MLP_SEEDS[:3]
N_OLD = sum(k for _, k in MLP_VARIANTS.values())
EMB4 = MLP_VARIANTS["emb4"][0]

# 이름: (임베딩 만드는 함수, 설명)
NEW = {
    "ple8": (lambda An: numemb.fit_ple(An, 8), "구간선형 8bin"),
    "ple4": (lambda An: numemb.fit_ple(An, 4), "구간선형 4bin (용량 대조군)"),
    "rff8": (lambda An: numemb.fit_rff(An, d=8, sigma=0.2), "고정주파수 푸리에 d=8 σ=.2"),
}
BETAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.273, 0.35, 0.50]


def ck(fold, name, seed):
    return CACHE / f"ne_{name}_f{fold}_{_key()}_{seed}.npz"


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = foldcache.load(FOLDS)
    Z = {f: (cached[f][0], cached[f][1], cached[f][2]) for f in FOLDS}

    need = [(f, n, s) for f in FOLDS for n in NEW for s in SEEDS
            if not ck(f, n, s).exists()]
    df = load_csv(TRAIN_CSV) if need else None

    got, dims = {}, {}
    for fold in FOLDS:
        if not any(f == fold for f, _, _ in need):
            for n in NEW:
                for s in SEEDS:
                    got[(fold, n, s)] = np.load(ck(fold, n, s))["z"]
            continue

        season = df.season.to_numpy()
        tr, va = df[season < fold], df[season == fold]
        w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=fold - 1)
        art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                            k_batter=SMOOTH_K_BATTER)
        Xtr, Xva = A23.featurize(tr, art), A23.featurize(va, art)
        ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
        enc = mlp.fit_encoder(Xtr, art)
        An, Ctr = encode_apply(Xtr, enc)
        B, Cva = encode_apply(Xva, enc)
        del tr, va, Xtr, Xva
        gc.collect()
        print(f"\n=== fold {fold}  학습 {An.shape}  검증 {B.shape}"
              f"  ({time.time()-t0:.0f}s) ===")

        for name, (make, desc) in NEW.items():
            spec = make(An)
            dims[(fold, name)] = (len(spec["idx"]), numemb.out_dim(spec, An.shape[1]))
            for s in SEEDS:
                p = ck(fold, name, s)
                if p.exists():
                    got[(fold, name, s)] = np.load(p)["z"]
                    continue
                m, _ = mlp.train_mlp(An, Ctr, ytr, w, An[:512], Ctr[:512],
                                     enc["sizes"], seed=s, verbose=False,
                                     numemb=spec, **EMB4)
                if s == SEEDS[0]:      # 두 구현 대조는 실제 fold 데이터로 한 번씩
                    d = np.abs(m.numemb_apply(torch.from_numpy(B[:20000])).detach()
                               .numpy() - numemb.apply_np(spec, B[:20000])).max()
                    print(f"  [구현대조] {name}  torch vs numpy max|diff| = {d:.3e}")
                z = mlp.infer(m, B, Cva)
                np.savez_compressed(p, z=z)
                got[(fold, name, s)] = z
                del m
                print(f"  {name}_{s}  ({desc}, {dims[(fold,name)][1]}차원)"
                      f"  단독 {mlp.score(z, Z[fold][2]):>8.1f}  ({time.time()-t0:.0f}s)")
        del An, Ctr, B, Cva, enc
        gc.collect()
    del df
    gc.collect()

    def avg(fold, name):
        if name == "emb4ctl":       # ablate26이 이미 학습해뒀다
            return np.mean([np.load(A26.ck(fold, "mlp", "emb4ctl", s))["z"]
                            for s in SEEDS], axis=0)
        return np.mean([got[(fold, name, s)] for s in SEEDS], axis=0)

    def bl(zl, zm, y, a=MLP_ALPHA):
        return mlp.score((1 - a) * zl + a * zm, y)

    base = {f: bl(*Z[f]) for f in FOLDS}
    ctl = {f: mlp.score(avg(f, "emb4ctl"), Z[f][2]) for f in FOLDS}

    print("\n" + "=" * 100)
    print("1. 단독 성능 — §7.24 기준 (3시드 평균, 대조군 emb4ctl = 평범한 표준화 입력)")
    print("=" * 100)
    print(f"{'변형':>10} {'입력차원':>9} " + " ".join(f"{'fold'+str(f):>10}" for f in FOLDS)
          + " ".join(f"{'Δ대조'+str(f)[-2:]:>10}" for f in FOLDS)
          + f" {'corr(z_lgb)':>12} {'corr(블렌드)':>13}")
    print(f"{'emb4ctl':>10} {'73':>9} " + " ".join(f"{ctl[f]:>10.1f}" for f in FOLDS)
          + f"{'+0.0':>10}{'+0.0':>10}")
    for name in NEW:
        sc = [mlp.score(avg(f, name), Z[f][2]) for f in FOLDS]
        zl, zm, y = Z[2024]
        cur = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm
        print(f"{name:>10} {dims.get((2024,name),('','?'))[1]:>9} "
              + " ".join(f"{v:>10.1f}" for v in sc)
              + " ".join(f"{sc[i]-ctl[f]:>+10.1f}" for i, f in enumerate(FOLDS))
              + f" {np.corrcoef(avg(2024, name), zl)[0,1]:>12.3f}"
              + f" {np.corrcoef(avg(2024, name), cur)[0,1]:>13.3f}")

    print("\n" + "=" * 100)
    print(f"2. 채택 판정 — 풀에 β로 섞는다 (기준: 현행 {N_OLD}멤버, α={MLP_ALPHA} 고정)")
    print("=" * 100)
    print("  기준  " + "  ".join(f"fold{f} {base[f]:.1f}" for f in FOLDS))
    print(f"\n{'후보':>10} {'fold':>6} " + " ".join(f"{b:>8.3f}" for b in BETAS))
    verdict = {}
    for name in NEW:
        cur_d = {}
        for f in FOLDS:
            zl, zm, y = Z[f]
            zn = avg(f, name)
            row = [bl(zl, (1 - b) * zm + b * zn, y) - base[f] for b in BETAS]
            row[0] = 0.0
            cur_d[f] = row
            print(f"{name:>10} {f:>6} " + " ".join(f"{v:>+8.2f}" for v in row))
        # 각 fold의 최적 β와, 두 fold 공통으로 이득인 β가 있는가
        both = [i for i in range(1, len(BETAS))
                if all(cur_d[f][i] > 0 for f in FOLDS)]
        verdict[name] = both
        if both:
            i = max(both, key=lambda i: np.mean([cur_d[f][i] for f in FOLDS]))
            d = {f: cur_d[f][i] for f in FOLDS}
            mags = sorted(abs(v) for v in d.values())
            ratio = mags[-1] / mags[0] if mags[0] > 1e-9 else float("inf")
            print(f"{'':>10} {'→':>6} 양 fold 공통 이득 β={BETAS[i]}  "
                  + "  ".join(f"fold{f} {d[f]:+.2f}" for f in FOLDS)
                  + f"   크기비 {ratio:.1f}   {'채택 후보' if ratio <= 3 else '조건③ 걸림'}")
        else:
            print(f"{'':>10} {'→':>6} 양 fold 공통으로 이득인 β 없음 → 기각")

    print("\n" + "=" * 100)
    print("사전 예측 채점")
    print("=" * 100)
    p8 = [mlp.score(avg(f, "ple8"), Z[f][2]) - ctl[f] for f in FOLDS]
    p4 = [mlp.score(avg(f, "ple4"), Z[f][2]) - ctl[f] for f in FOLDS]
    rf = [mlp.score(avg(f, "rff8"), Z[f][2]) - ctl[f] for f in FOLDS]
    print(f"  P1 ple8 단독 양 fold +      {p8[0]:+.1f} / {p8[1]:+.1f}"
          f"   → {'적중' if all(v > 0 for v in p8) else '빗나감'}")
    print(f"  P2 ple4 ≥ ple8              {p4[0]:+.1f} / {p4[1]:+.1f} vs"
          f" {p8[0]:+.1f} / {p8[1]:+.1f}"
          f"   → {'적중' if all(a >= b for a, b in zip(p4, p8)) else '빗나감'}")
    print(f"  P3 rff8 중립~악화           {rf[0]:+.1f} / {rf[1]:+.1f}"
          f"   → {'적중' if all(v <= 5 for v in rf) else '빗나감'}")
    n_ok = sum(1 for n in NEW if verdict[n])
    print(f"  P4 통과 후보 ≥ 1            {n_ok}개"
          f"   → {'적중' if n_ok >= 1 else '빗나감'}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
