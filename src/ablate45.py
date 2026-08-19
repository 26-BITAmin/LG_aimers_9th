"""보조 헤드 축 교체 — (mid,off) / (rev,off) 2헤드 (§7.74 후속)

동기
    §7.74: off(크게 벗어남) 축의 내용 성분은 +6.1/+10.6으로 **양 fold 양수**인데
    (이 계열 최초), 3헤드째의 정규화 비용(−12.4 f2024)이 다 먹었다. 그리고
    §7.42: mid–rev는 카운트 프로파일 r=+0.990으로 사실상 한 축이다.
    → 헤드를 늘리지 말고 **중복 축 하나를 off로 교체**하면 비용 없이 커버리지만
    넓어질 수 있다. 헤드 수가 같으므로 정규화는 구조적으로 통제된다 (§7.42 bs 설계).

    ⚠️ 반대 방향 전례: §7.42의 bs(같은 헤드 수 교체)도 mr에 졌다 (−1.87/−10.70).
    단 그때 교체한 축은 무작위 수준이었고, off는 내용이 검증돼 있어 전제가 다르다.

후보
    mo = (middle, off)     rev를 버린다 (mid–rev r=.990 → rev가 중복이라는 가설)
    ro = (reverse, off)    mid를 버린다

사전 예측 (실행 전에 박아둔다)
    P1  mo·ro 모두 mr 대비 |Δ| < 20 양 fold (교체로 갈아탈 근거는 안 나온다)
        내 예측: **그렇다** — mid·rev가 "중복"이어도 각자 정의 조각이라
        하나를 버리는 손실이 off의 이득과 상쇄될 것
    P2  |mo − ro| < 10 양 fold (버리는 쪽이 mid든 rev든 차이 없다)
        내 예측: **그렇다** (mid–off −0.973, rev–off −0.986으로 대칭적)
    P3  풀 편입 β* > 0 양 fold                     내 예측: **아니다 (후보 0)**

실행:  .venv/Scripts/python.exe -u src/ablate45.py 2>&1 | tee experiments/ablate_run45.log
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

THREADS = 4
mlp.THREADS = THREADS

FOLDS = [2023, 2024]
SEEDS = MLP_SEEDS[:3]
EMB4 = MLP_VARIANTS["emb4"][0]
AUX3 = ("middle", "reverse", "off")
COLS = {"mo": [0, 2], "ro": [1, 2]}          # aux3에서 뽑는 열
BETAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.273, 0.35, 0.50]


def ck(fold, name, seed):
    return CACHE / f"aux2s_{name}_f{fold}_{_key()}_{seed}.npz"


def fold_inputs(df, fold):
    Xtr, ytr, w, Xva, yva, art, tr = fold_data(df, fold)
    aux, cov = modelabel.aux_labels(tr, modes=AUX3)
    print(f"  보조 라벨(3축) 복원 {cov:.1%}")
    if SOFT_LABEL_F:
        Xtr, ytr, w, aux, _h = soften.soften(tr, Xtr, ytr, w, aux)
    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    del Xtr, Xva, tr, art
    gc.collect()
    return dict(A=A, C=C, B=B, Cv=Cv, sizes=sizes, ytr=ytr, w=w, aux=aux)


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = foldcache.load(FOLDS)
    Z = {f: (cached[f][0], cached[f][1], cached[f][2]) for f in FOLDS}
    MEM = {f: cached[f][3] for f in FOLDS}

    todo = [(f, n, s) for f in FOLDS for n in COLS for s in SEEDS]
    df = load_csv(TRAIN_CSV) if any(not ck(*t).exists() for t in todo) else None

    got = {}
    for fold in FOLDS:
        D = None
        for name in COLS:
            for s in SEEDS:
                p = ck(fold, name, s)
                if p.exists():
                    got[(fold, name, s)] = np.load(p)["z"]
                    continue
                if D is None:
                    D = fold_inputs(df, fold)
                    print(f"=== fold {fold} 준비 완료 ({time.time()-t0:.0f}s) ===")
                _, curve = mlp.train_mlp(D["A"], D["C"], D["ytr"], D["w"],
                                         D["B"], D["Cv"], D["sizes"], seed=s,
                                         verbose=False,
                                         y_aux=D["aux"][:, COLS[name]],
                                         **dict(EMB4, aux=MLP_AUX_LAMBDA))
                z = curve[-1]
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
    ctl = {f: mlp.score(MEM[f][:3].mean(0), Z[f][2]) for f in FOLDS}

    print("\n" + "=" * 96)
    print("1. 단독 성능 (3시드 평균) — 대조군 mr = 캐시 emb4 멤버 [0:3]")
    print("=" * 96)
    print(f"{'변형':>8} " + " ".join(f"{'fold'+str(f):>10}" for f in FOLDS)
          + " ".join(f"{'Δmr'+str(f)[-2:]:>9}" for f in FOLDS)
          + f" {'corr(블렌드24)':>14}")
    print(f"{'mr':>8} " + " ".join(f"{ctl[f]:>10.1f}" for f in FOLDS)
          + " ".join(f"{0.0:>+9.1f}" for _ in FOLDS))
    zl24, zm24, y24 = Z[2024]
    cur24 = (1 - MLP_ALPHA) * zl24 + MLP_ALPHA * zm24
    for name in COLS:
        row = [mlp.score(avg(f, name), Z[f][2]) for f in FOLDS]
        dl = [row[i] - ctl[f] for i, f in enumerate(FOLDS)]
        print(f"{name:>8} " + " ".join(f"{v:>10.1f}" for v in row)
              + " ".join(f"{v:>+9.1f}" for v in dl)
              + f" {np.corrcoef(avg(2024, name), cur24)[0,1]:>14.3f}")
        for s in SEEDS:
            per = [mlp.score(got[(f, name, s)], Z[f][2]) for f in FOLDS]
            print(f"{name+'_'+str(s):>8} " + " ".join(f"{v:>10.1f}" for v in per))

    print("\n" + "=" * 96)
    print(f"2. 풀 편입 — z_mlp(8멤버)에 β로 섞는다 (α={MLP_ALPHA}, 기준 대비 Δ)")
    print("=" * 96)
    print(f"  기준  " + "  ".join(f"fold{f} {base[f]:.1f}" for f in FOLDS))
    print(f"\n{'후보':>8} {'fold':>6} " + " ".join(f"{b:>8.3f}" for b in BETAS))
    for name in COLS:
        for f in FOLDS:
            zl, zm, y = Z[f]
            zn = avg(f, name)
            out = [bl(zl, (1 - b) * zm + b * zn, y) - base[f] for b in BETAS]
            out[0] = 0.0
            print(f"{name:>8} {f:>6} " + " ".join(f"{v:>+8.2f}" for v in out))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
