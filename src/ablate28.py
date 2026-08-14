"""MLP 풀 안의 `emb4` : `noid` 비율 — 한 번도 스윕된 적 없는 노브

동기
    현행 풀은 `emb4` 5시드 + `noid` 3시드를 **평평하게 평균**한다 (noid 비중 3/8 = .375).
    그런데 **`noid`가 더 세다.**

        fold2024 단독    emb4 653.5    noid 691.3    (+37.8)
        fold2023H2 단독  emb4 537.6    noid 603.1    (+65.5)

    **약한 쪽에 가중이 더 실려 있다.** `ablate17` ②는 "emb4냐 noid냐 섞느냐"만 봤고
    (emb4 +21.2 / noid +27.0(크기비 3.1 ✗) / emb4+noid +24.2 ✓), **비율 자체는
    스윕한 적이 없다.** 5:3은 최적화의 결과가 아니라 시드 수 배정의 부산물이다.

왜 이게 지금까지 기각된 것들과 다른가
    · 정보를 더하지도 빼지도 않는다 (§7.15·§7.24가 죽은 이유가 그것이었다).
    · 용량을 바꾸지 않는다 (§7.25·§7.26이 죽은 이유가 그것이었다).
    · 학습 분포도 안 바꾼다 (§7.23이 죽은 이유).
    → **이미 있는 멤버의 재가중뿐이다.** 배포는 `predict.py`의 MLP 평균에 가중치를
      넣는 것으로 끝나고, 새 학습도 새 아티팩트도 없다.

    ⚠️ §7.23의 교훈을 여기에 적용해 본다: "편향이 다른 것들의 평균은 분산 축소가 아니다."
      emb4/noid는 hl처럼 한 축 위의 두 지점인가? 아니다 — 둘은 **같은 학습 분포·같은
      피처**를 보고 임베딩 구성만 다르다. 그래도 편향이 다를 수 있으므로, 두 fold에서
      w*가 같은 쪽으로 움직이는지를 조건 ②로 확인한다.

사전 예측 (실행 전에 박아둔다)
    P1  noid 단독 > emb4 단독, 양 fold.        내 예측: **그렇다** (위 수치의 재확인)
    P2  w* > 0.375, 양 fold.                   내 예측: **그렇다**
    P3  조건 ①②③ 통과.                        내 예측: **그렇다**

    ⚠️ §7.26에서 내 **긍정 예측 두 개가 다 빗나갔다.** 지금까지 7전 7패를 맞힌 것은
      전부 "기각될 것"에 건 것이라 예측력인지 비관인지 구분이 안 된다.
      → 이 예측들은 근거로 쓰지 않는다. 판정은 조건 ①②③이 한다.

실행:  .venv/Scripts/python.exe -u src/ablate28.py 2>&1 | tee experiments/ablate_run28.log
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

import ablate23 as A23
import foldcache
import mlp
from config import (MLP_ALPHA, MLP_SEEDS, MLP_VARIANTS, RECENCY_HALF_LIFE,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from features import fit_artifacts
from foldcache import CACHE, _key
from mlp_infer import encode_apply

THREADS = 4
mlp.THREADS = THREADS

FOLDS = [2023, 2024]
# 배포와 같은 구성. foldcache.compute의 순회 순서와 **같아야** members 배열과 짝이 맞는다
ROSTER = [(v, s) for v, (_, n) in MLP_VARIANTS.items() for s in MLP_SEEDS[:n]]
WS = [0.0, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]
DEPLOY_W = sum(1 for v, _ in ROSTER if v == "noid") / len(ROSTER)


def ck(fold, vname, seed):
    return CACHE / f"mem_{vname}_f{fold}_{_key()}_{seed}.npz"


def members(fold, cached, t0):
    """멤버별 z. fold2024는 캐시에 있고, fold2023은 옛 캐시라 없어서 새로 만든다."""
    mem = cached[fold][3]
    if mem is not None:
        print(f"  fold {fold}: 캐시의 멤버별 z 사용 ({mem.shape})")
        return {k: mem[i] for i, k in enumerate(ROSTER)}

    out, need = {}, []
    for vname, s in ROSTER:
        p = ck(fold, vname, s)
        (out.__setitem__((vname, s), np.load(p)["z"]) if p.exists()
         else need.append((vname, s)))
    if not need:
        print(f"  fold {fold}: 체크포인트 재사용 ({len(out)}멤버)")
        return out

    df = load_csv(TRAIN_CSV)
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
    del df, tr, va, Xtr, Xva
    gc.collect()
    print(f"\n=== fold {fold} 멤버 {len(need)}개 학습 ({time.time()-t0:.0f}s) ===")
    for vname, s in need:
        cfg = MLP_VARIANTS[vname][0]
        m, _ = mlp.train_mlp(An, Ctr, ytr, w, An[:512], Ctr[:512], enc["sizes"],
                             seed=s, verbose=False, **cfg)
        z = mlp.infer(m, B, Cva)
        np.savez_compressed(ck(fold, vname, s), z=z)
        out[(vname, s)] = z
        del m
        print(f"  {vname}_{s}  단독 {mlp.score(z, cached[fold][2]):>8.1f}"
              f"  ({time.time()-t0:.0f}s)")
    del An, Ctr, B, Cva
    gc.collect()
    return out


def main():
    t0 = time.time()
    cached = foldcache.load(FOLDS)
    M = {f: members(f, cached, t0) for f in FOLDS}

    def pool(f, w):
        """noid 쪽에 비중 w. 각 변형 안에서는 시드 평균 (순수 분산 축소)."""
        e = np.mean([M[f][k] for k in ROSTER if k[0] == "emb4"], axis=0)
        n = np.mean([M[f][k] for k in ROSTER if k[0] == "noid"], axis=0)
        return (1 - w) * e + w * n

    def bl(f, zm):
        zl, _, y = cached[f][0], cached[f][1], cached[f][2]
        return mlp.score((1 - MLP_ALPHA) * zl + MLP_ALPHA * zm, y)

    print("\n" + "=" * 92)
    print("1. 변형별 단독 (시드 평균)")
    print("=" * 92)
    for f in FOLDS:
        y = cached[f][2]
        e = np.mean([M[f][k] for k in ROSTER if k[0] == "emb4"], axis=0)
        n = np.mean([M[f][k] for k in ROSTER if k[0] == "noid"], axis=0)
        print(f"  fold{f}   emb4 {mlp.score(e, y):>8.1f}   noid {mlp.score(n, y):>8.1f}"
              f"   차이 {mlp.score(n, y)-mlp.score(e, y):>+7.1f}"
              f"   corr(emb4,noid) {np.corrcoef(e, n)[0,1]:.3f}")

    print("\n" + "=" * 92)
    print(f"2. noid 비중 w 스윕 (배포값 w={DEPLOY_W:.3f} = 8멤버 평평 평균, α={MLP_ALPHA} 고정)")
    print("=" * 92)
    base = {f: bl(f, pool(f, DEPLOY_W)) for f in FOLDS}
    print("  기준  " + "  ".join(f"fold{f} {base[f]:.1f}" for f in FOLDS))
    print(f"\n{'fold':>6} " + " ".join(f"{w:>8.3f}" for w in WS))
    D = {}
    for f in FOLDS:
        D[f] = [bl(f, pool(f, w)) - base[f] for w in WS]
        print(f"{f:>6} " + " ".join(f"{v:>+8.2f}" for v in D[f]))

    print("\n" + "=" * 92)
    print("3. 채택 판정 — 두 fold 모두 이득인 w가 있는가 (조건 ②), 크기비 ≤3 (조건 ③)")
    print("=" * 92)
    both = [i for i in range(len(WS)) if all(D[f][i] > 0 for f in FOLDS)]
    if not both:
        print("  양 fold 공통으로 이득인 w 없음 → 기각")
    else:
        for i in both:
            d = {f: D[f][i] for f in FOLDS}
            mags = sorted(abs(v) for v in d.values())
            ratio = mags[-1] / mags[0] if mags[0] > 1e-9 else float("inf")
            avg = float(np.mean(list(d.values())))
            print(f"  w={WS[i]:<6.3f} " + "  ".join(f"fold{f} {d[f]:+7.2f}" for f in FOLDS)
                  + f"   평균 {avg:+7.2f}   크기비 {ratio:>5.1f}"
                  + f"   → {'채택 후보' if ratio <= 3 else '조건③ 걸림'}")

    print("\n사전 예측 채점")
    ok1 = all(mlp.score(np.mean([M[f][k] for k in ROSTER if k[0] == "noid"], axis=0),
                        cached[f][2])
              > mlp.score(np.mean([M[f][k] for k in ROSTER if k[0] == "emb4"], axis=0),
                          cached[f][2]) for f in FOLDS)
    wstar = {f: WS[int(np.argmax(D[f]))] for f in FOLDS}
    print(f"  P1 noid > emb4 양 fold          → {'적중' if ok1 else '빗나감'}")
    print(f"  P2 w* > {DEPLOY_W:.3f} 양 fold        w* = "
          + ", ".join(f"fold{f} {wstar[f]:.3f}" for f in FOLDS)
          + f"   → {'적중' if all(v > DEPLOY_W for v in wstar.values()) else '빗나감'}")
    print(f"  P3 조건 통과                     → "
          f"{'적중' if any(True for i in both if max(abs(D[f][i]) for f in FOLDS) / max(1e-9, min(abs(D[f][i]) for f in FOLDS)) <= 3) else '빗나감'}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
