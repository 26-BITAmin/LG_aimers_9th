"""MLP activation GELU → ReLU (§7.25·§7.26 후속 — 용량 중립 노브)

동기
    §7.25(깊이)·§7.26(PLE/RFF)이 세운 법칙은 "파라미터를 늘리면 진다"였다.
    activation 교체는 그 법칙의 적용 범위 **밖**이다 — 파라미터 수가 그대로다.
    현행 GELU(approximate="tanh")는 ReLU보다도 매끄럽다. "MLP가 축 정렬
    임계값(계단)을 못 만든다"는 진단이 맞다면 꺾임이 있는 ReLU가 유리할 수
    있고, 지금까지 activation은 한 번도 스윕된 적이 없다.

    ⚠️ 반대 방향 증거: PLE는 임계값을 입력에서 공짜로 줬는데도 −44~−227이었다.
    병목이 날카로움이 아니라 신호 총량(≈0.004 nat)이라는 판정이 이미 있다.

설계
    · 대조군은 재학습하지 않는다 — s23 foldcache의 emb4 멤버 [0:3] (시드 42/7/2024).
    · relu 변형은 배포와 같은 경로(soft label + 보조 헤드 λ=1.0)로 학습한다.
      foldcache.compute와 같은 순서: fold_data → aux_labels → soften → encode.
    · 정합성 게이트: emb4 시드 42를 이 스크립트 경로로 fold2024에서 재학습해
      캐시 멤버[0]과 max|diff| = 0 인지 확인한다. 0이 아니면 아래 비교는 무효.
    · THREADS=4 (캐시를 만든 값), 에폭 9 고정, curve[-1] (valid argmax 금지).

사전 예측 (실행 전에 박아둔다)
    P1  relu 단독 Δ대조 |Δ| < 20 양 fold (사실상 중립)      내 예측: **그렇다**
    P2  풀 편입 β* > 0 양 fold                              내 예측: **아니다 (후보 0)**
    P3  corr(블렌드) 변화 < 0.005 (다양성도 안 생긴다)       내 예측: **그렇다**

    P1이 빗나가고 relu가 +면, "용량 중립 노브는 법칙 밖"이 실증되는 첫 사례다.
    그때는 §7.25 프로토콜대로 풀 핸디캡(P2)으로 채택을 가른다.

실행:  .venv/Scripts/python.exe -u src/ablate43.py 2>&1 | tee experiments/ablate_run43.log
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

import foldcache
import mlp
import modelabel
import soften
from config import (MLP_ALPHA, MLP_AUX_LAMBDA, MLP_SEEDS, MLP_VARIANTS,
                    SOFT_LABEL_F, TRAIN_CSV)
from data import load_csv
from foldcache import CACHE, _key, fold_data

THREADS = 4                      # 캐시를 만든 값
mlp.THREADS = THREADS

FOLDS = [2023, 2024]
SEEDS = MLP_SEEDS[:3]            # (42, 7, 2024) — 캐시 emb4 멤버 [0:3]과 같은 시드
EMB4 = MLP_VARIANTS["emb4"][0]
ARCH = {"relu": dict(EMB4, act="relu")}
BETAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.273, 0.35, 0.50]


def ck(fold, name, seed):
    return CACHE / f"act_{name}_f{fold}_{_key()}_{seed}.npz"


def fold_inputs(df, fold):
    """foldcache.compute와 **같은 순서**로 학습 입력을 만든다."""
    Xtr, ytr, w, Xva, yva, art, tr = fold_data(df, fold)
    aux, cov = modelabel.aux_labels(tr)
    if SOFT_LABEL_F:
        n0 = len(Xtr)
        Xtr, ytr, w, aux, soft_h = soften.soften(tr, Xtr, ytr, w, aux)
        print(f"  soft label: {n0:,} → {len(Xtr):,}행  h={soft_h}")
    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    del Xtr, Xva, tr, art
    gc.collect()
    return dict(A=A, C=C, B=B, Cv=Cv, sizes=sizes, ytr=ytr, w=w, aux=aux)


def train_one(D, cfg, seed):
    _, curve = mlp.train_mlp(D["A"], D["C"], D["ytr"], D["w"], D["B"], D["Cv"],
                             D["sizes"], seed=seed, verbose=False, y_aux=D["aux"],
                             **dict(cfg, aux=MLP_AUX_LAMBDA))
    return curve[-1]


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = foldcache.load(FOLDS)
    Z = {f: (cached[f][0], cached[f][1], cached[f][2]) for f in FOLDS}
    MEM = {f: cached[f][3] for f in FOLDS}
    for f in FOLDS:
        assert MEM[f] is not None, f"fold{f} 캐시에 멤버별 z가 없다 — foldcache를 다시 만들 것"

    chk = CACHE / f"act_gelu42chk_f2024_{_key()}.npz"
    todo = [(f, n, s) for f in FOLDS for n in ARCH for s in SEEDS]
    need_train = [t for t in todo if not ck(*t).exists()] or not chk.exists()
    df = load_csv(TRAIN_CSV) if need_train else None

    got = {}
    for fold in FOLDS:
        D = None
        # ── 정합성: fold2024에서 emb4 시드 42를 이 경로로 재학습 → 멤버[0]과 대조 ──
        if fold == 2024:
            if not chk.exists():
                D = D or fold_inputs(df, fold)
                z = train_one(D, EMB4, 42)
                np.savez_compressed(chk, z=z)
                print(f"  gelu42 재학습 완료  ({time.time()-t0:.0f}s)")
            d = np.abs(np.load(chk)["z"] - MEM[2024][0]).max()
            print(f"[정합성] fold2024 emb4_42 재학습 vs 캐시 멤버[0]  max|diff| = {d:.3e}"
                  f"  {'일치' if d == 0 else '⚠️ 불일치 — 비교 무효'}")
        for name in ARCH:
            for s in SEEDS:
                p = ck(fold, name, s)
                if p.exists():
                    got[(fold, name, s)] = np.load(p)["z"]
                    continue
                if D is None:
                    D = fold_inputs(df, fold)
                    print(f"=== fold {fold} 준비 완료 ({time.time()-t0:.0f}s) ===")
                z = train_one(D, ARCH[name], s)
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

    base = {f: bl(*Z[f]) for f in FOLDS}
    ctl3 = {f: MEM[f][:3].mean(0) for f in FOLDS}          # emb4 3시드 (캐시)

    print("\n" + "=" * 96)
    print("1. 단독 성능 (3시드 평균) — 대조군 emb4ctl은 캐시 멤버 [0:3]")
    print("=" * 96)
    hdr = f"{'변형':>10} " + " ".join(f"{'fold'+str(f):>10}" for f in FOLDS) \
        + " ".join(f"{'Δ대조'+str(f)[-2:]:>10}" for f in FOLDS) \
        + f" {'corr(z_lgb)':>12} {'corr(블렌드)':>13}"
    print(hdr)
    ctl = {f: mlp.score(ctl3[f], Z[f][2]) for f in FOLDS}
    print(f"{'emb4ctl':>10} " + " ".join(f"{ctl[f]:>10.1f}" for f in FOLDS)
          + " ".join(f"{0.0:>+10.1f}" for _ in FOLDS))
    for name in ARCH:
        row = [mlp.score(avg(f, name), Z[f][2]) for f in FOLDS]
        dl = [row[i] - ctl[f] for i, f in enumerate(FOLDS)]
        zl, zm, y = Z[2024]
        cur = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm
        print(f"{name:>10} " + " ".join(f"{v:>10.1f}" for v in row)
              + " ".join(f"{v:>+10.1f}" for v in dl)
              + f" {np.corrcoef(avg(2024, name), zl)[0,1]:>12.3f}"
              + f" {np.corrcoef(avg(2024, name), cur)[0,1]:>13.3f}")
        for s in SEEDS:                                     # 시드 산포도 같이 (P1 판정용)
            per = [mlp.score(got[(f, name, s)], Z[f][2]) for f in FOLDS]
            print(f"{name+'_'+str(s):>10} " + " ".join(f"{v:>10.1f}" for v in per))

    print("\n" + "=" * 96)
    print(f"2. 풀 편입 — z_mlp(8멤버)에 β로 섞는다 (α={MLP_ALPHA} 고정, 기준 대비 Δ)")
    print("=" * 96)
    print(f"  기준  " + "  ".join(f"fold{f} {base[f]:.1f}" for f in FOLDS))
    print(f"\n{'후보':>10} {'fold':>6} " + " ".join(f"{b:>8.3f}" for b in BETAS))
    for name in ARCH:
        for f in FOLDS:
            zl, zm, y = Z[f]
            zn = avg(f, name)
            out = [bl(zl, (1 - b) * zm + b * zn, y) - base[f] for b in BETAS]
            out[0] = 0.0
            print(f"{name:>10} {f:>6} " + " ".join(f"{v:>+8.2f}" for v in out))
    print("\n  판정: β* > 0 이 **양 fold**에서 나와야 후보 (§7.25 프로토콜)")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
