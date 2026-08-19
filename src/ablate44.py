"""보조 헤드 3축 — 세 번째 실패 모드 `off`(크게 벗어남) 추가 (§7.42 후속)

동기 (사용자 제안: "보조 타깃을 더 바꿔볼 수 있지 않나")
    §7.42의 교훈: 보조 타깃은 독립성이 아니라 **표현 공유**로 고른다. 채택된
    mid·rev는 `control_success` 정의의 부분집합이었고, 기각된 ball/strike는
    다른 함수였다. 그런데 정의의 세 실패 모드 중 **셋째(크게 벗어남)는 한 번도
    헤드가 된 적이 없다** — §7.42 본문이 직접 지적한다: "§7.40의 착상은 mid와
    off 사이의 이야기였는데(카운트 프로파일 r=−0.973), off는 헤드가 아니다."

    ⚠️ 반대 방향 논거도 적어둔다: off = (1−suc)(1−mid)(1−rev)로 세 예측치의
    **결정적 함수**다. 정보로는 0을 더한다. 이득이 있다면 §7.41처럼 순수하게
    표현 조각(gradient가 크게-벗어남 확률을 몸통에 선형 노출시키는 것)에서 온다.

설계 (ablate43 프로토콜)
    · 대조군 `mr` = s23 foldcache의 emb4 멤버 [0:3] (재학습 0회).
    · `mro`  = (middle, reverse, off) 3헤드, 배포 경로(soft label + λ1.0) 그대로.
    · `mrsh` = (middle, reverse, off 행섞음) — 3헤드의 정규화 성분만 (§7.42 rand4 역할).
    · soften은 3열 aux로 한 번만 돌리고, mrsh는 soften 후 off 열을 순열한다.
    · 보조 헤드 Linear는 __init__ 맨 마지막에 생성되므로 (mlp.py §7.42 수정)
      헤드 폭 2→3이어도 임베딩·몸통 초기값은 비트 단위로 같다.

사전 예측 (실행 전에 박아둔다)
    P1  mro − mrsh 단독 > 0 양 fold (진짜 축 > 무작위 축. §7.42의 mrbs−rand4 전례)
        내 예측: **그렇다**
    P2  mro − mr 단독 |Δ| < 20 양 fold (off는 결정적 함수라 순증이 얇다)
        내 예측: **그렇다 (중립)**
    P3  풀 편입 β* > 0 양 fold                        내 예측: **아니다 (후보 0)**

    P2가 빗나가고 mro가 +20을 넘기면, §7.42가 남긴 "더 나은 축을 찾으면
    4헤드(이제 3헤드)로 갈 수 있다"의 첫 실증이다 → ablate41 규모(8멤버 풀)로 승격.

실행:  .venv/Scripts/python.exe -u src/ablate44.py 2>&1 | tee experiments/ablate_run44.log
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
from config import (MLP_ALPHA, MLP_AUX_LAMBDA, MLP_SEEDS, MLP_VARIANTS, SEED,
                    SOFT_LABEL_F, TRAIN_CSV)
from data import load_csv
from foldcache import CACHE, _key, fold_data

THREADS = 4
mlp.THREADS = THREADS

FOLDS = [2023, 2024]
SEEDS = MLP_SEEDS[:3]
EMB4 = MLP_VARIANTS["emb4"][0]
AUX3 = ("middle", "reverse", "off")
BETAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.273, 0.35, 0.50]


def ck(fold, name, seed):
    return CACHE / f"aux3_{name}_f{fold}_{_key()}_{seed}.npz"


def fold_inputs(df, fold):
    """foldcache.compute와 같은 순서. aux만 3열(mid, rev, off)로 만든다."""
    Xtr, ytr, w, Xva, yva, art, tr = fold_data(df, fold)
    aux, cov = modelabel.aux_labels(tr, modes=AUX3)
    print(f"  보조 라벨(3축) 복원 {cov:.1%}  off 양성률 {aux[:, 2].mean():.4f}")
    if SOFT_LABEL_F:
        n0 = len(Xtr)
        Xtr, ytr, w, aux, soft_h = soften.soften(tr, Xtr, ytr, w, aux)
        print(f"  soft label: {n0:,} → {len(Xtr):,}행")
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

    rng = np.random.default_rng(SEED)
    todo = [(f, n, s) for f in FOLDS for n in ("mro", "mrsh") for s in SEEDS]
    df = load_csv(TRAIN_CSV) if any(not ck(*t).exists() for t in todo) else None

    got = {}
    for fold in FOLDS:
        D = None
        aux_sh = None
        for name in ("mro", "mrsh"):
            for s in SEEDS:
                p = ck(fold, name, s)
                if p.exists():
                    got[(fold, name, s)] = np.load(p)["z"]
                    continue
                if D is None:
                    D = fold_inputs(df, fold)
                    perm = rng.permutation(len(D["aux"]))
                    aux_sh = D["aux"].copy()
                    aux_sh[:, 2] = D["aux"][perm, 2]
                    print(f"=== fold {fold} 준비 완료 ({time.time()-t0:.0f}s) ===")
                ya = D["aux"] if name == "mro" else aux_sh
                _, curve = mlp.train_mlp(D["A"], D["C"], D["ytr"], D["w"],
                                         D["B"], D["Cv"], D["sizes"], seed=s,
                                         verbose=False, y_aux=ya,
                                         **dict(EMB4, aux=MLP_AUX_LAMBDA))
                z = curve[-1]
                np.savez_compressed(p, z=z)
                got[(fold, name, s)] = z
                print(f"  {name}_{s}  단독 {mlp.score(z, Z[fold][2]):>8.1f}"
                      f"  ({time.time()-t0:.0f}s)")
        del D, aux_sh
        gc.collect()
    del df
    gc.collect()

    def avg(fold, name):
        return np.mean([got[(fold, name, s)] for s in SEEDS], axis=0)

    def bl(zl, zm, y, a=MLP_ALPHA):
        return mlp.score((1 - a) * zl + a * zm, y)

    base = {f: bl(*Z[f]) for f in FOLDS}
    ctl3 = {f: MEM[f][:3].mean(0) for f in FOLDS}
    ctl = {f: mlp.score(ctl3[f], Z[f][2]) for f in FOLDS}

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
    for name in ("mro", "mrsh"):
        row = [mlp.score(avg(f, name), Z[f][2]) for f in FOLDS]
        dl = [row[i] - ctl[f] for i, f in enumerate(FOLDS)]
        print(f"{name:>8} " + " ".join(f"{v:>10.1f}" for v in row)
              + " ".join(f"{v:>+9.1f}" for v in dl)
              + f" {np.corrcoef(avg(2024, name), cur24)[0,1]:>14.3f}")
        for s in SEEDS:
            per = [mlp.score(got[(f, name, s)], Z[f][2]) for f in FOLDS]
            print(f"{name+'_'+str(s):>8} " + " ".join(f"{v:>10.1f}" for v in per))
    d23 = [mlp.score(avg(f, "mro"), Z[f][2]) - mlp.score(avg(f, "mrsh"), Z[f][2])
           for f in FOLDS]
    print(f"\n  mro − mrsh (진짜 축 − 무작위 축):  "
          + "  ".join(f"fold{f} {v:+.1f}" for f, v in zip(FOLDS, d23)))

    print("\n" + "=" * 96)
    print(f"2. 풀 편입 — z_mlp(8멤버)에 β로 섞는다 (α={MLP_ALPHA}, 기준 대비 Δ)")
    print("=" * 96)
    print(f"  기준  " + "  ".join(f"fold{f} {base[f]:.1f}" for f in FOLDS))
    print(f"\n{'후보':>8} {'fold':>6} " + " ".join(f"{b:>8.3f}" for b in BETAS))
    for name in ("mro",):
        for f in FOLDS:
            zl, zm, y = Z[f]
            zn = avg(f, name)
            out = [bl(zl, (1 - b) * zm + b * zn, y) - base[f] for b in BETAS]
            out[0] = 0.0
            print(f"{name:>8} {f:>6} " + " ".join(f"{v:>+8.2f}" for v in out))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
