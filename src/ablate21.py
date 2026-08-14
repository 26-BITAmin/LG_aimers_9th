"""21차 실험 — **제3의 함수족(FM)이 뚫리는가.**

지금까지 전이된 것과 안 된 것의 경계가 뚜렷하다:

    새 함수족(MLP 도입)        홀드아웃 +11.4 → LB **+29.5**
    같은 함수족 확장(멤버 3→8)          +4.1 →     +0.24
    트리 안에서의 변경 (알고리즘·손실·하이퍼파라미터)  전부 기각

앙상블에 있는 귀납 편향은 아직 둘뿐이다 — 축 정렬 계단함수(트리)와
조밀 비선형(MLP). FM은 셋째다: **모든 쌍 상호작용을 저랭크 내적으로 명시**한다.

판정은 이미 확립된 진단으로 한다:
  · **현재 블렌드와의 로짓 상관.** 0.98 이상이면 실력이 비슷해도 얻을 게 없다
    (CatBoost 0.985 → +3.5). MLP는 0.93~0.95에서 +29.5였다.
  · 부호 일치 + 크기비 3배 이내 (조건 ③).

k=0은 상호작용 항이 사라져 **정규화 선형모형**과 같다. 공짜로 딸려오는
대조군이라 같이 잰다 — 이득이 상호작용에서 오는지 그냥 다른 함수족이라
오는지 구분된다.

`foldcache` 덕에 z_lgb·z_mlp는 학습 없이 나온다. 여기서 학습하는 건 FM뿐이다.

실행:  .venv/Scripts/python.exe -u src/ablate21.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import fm
import foldcache
import mlp
from config import MLP_ALPHA, TRAIN_CSV
from data import load_csv
from mlp_infer import EMB_COLS

FOLDS = [2023, 2024]
SEEDS = (42, 7)
BETAS = (0.05, 0.10, 0.15, 0.20, 0.30)
ID_COLS = ("pitcher_id", "batter_id")

# 1차 실행은 전부 참패였는데 **설정 버그**였다 (`src/fm_diag.py`).
#   · pitcher_id(712종) x batter_id(743종) 잠재벡터로 조합을 암기한다.
#     FM은 쌍 상호작용을 명시하므로 이 경로가 구조에 박혀 있다.
#     에폭 곡선: 474 → 307 → 241 → 45 → −10 → −57 → −126 → −233 (붕괴)
#     ID를 빼면:  474 → 424 → 467 → 455 → 385 → 438 → 443 → 433 (안정)
#   · 그런데 `curve[-1]`(마지막 에폭)만 봤다. 붕괴한 값을 실력으로 기록한 것이다.
# MLP도 똑같이 임베딩 16차원 때문에 −4670이 나온 적이 있다. 같은 함정이다.
#
# ⚠️ 에폭 수는 **fold를 보고 고르지 않는다.** 설정의 일부로 고정하고 조건 ③으로 판정한다.
CANDS = {
    "선형 k=0":      dict(k=0,  epochs=4, use_ids=False),
    "FM k=2 ep1":    dict(k=2,  epochs=1, use_ids=False),
    "FM k=2 ep4":    dict(k=2,  epochs=4, use_ids=False),
    "FM k=4 ep4":    dict(k=4,  epochs=4, use_ids=False),
    "FM k=2 +ID":    dict(k=2,  epochs=1, use_ids=True),   # ID 포함 대조군
}

fm.THREADS = 6


def verdict(d):
    same = bool(np.all(d > 0) or np.all(d < 0))
    sp = np.abs(d).max() / max(np.abs(d).min(), 1e-9) if same else np.inf
    return same, sp, (same and d.mean() > 0 and sp <= 3)


def main():
    t0 = time.time()
    cache = foldcache.load(FOLDS)
    Z, Y, cand = {}, {}, {}

    df = load_csv(TRAIN_CSV)
    for f in FOLDS:
        print(f"\n=== valid={f} ===")
        z_lgb, z_mlp, y, _ = cache[f]
        Z[f] = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
        Y[f] = y
        print(f"  현재 블렌드 BSS = {mlp.score(Z[f], y):.1f}"
              f"   (lgb {mlp.score(z_lgb, y):.1f} / mlp {mlp.score(z_mlp, y):.1f})")

        Xtr, ytr, w, Xva, yva, art = foldcache.fold_data(df, f)
        A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
        del Xtr, Xva
        print(f"  인코딩 완료  수치 {A.shape[1]}개 + 범주 {len(EMB_COLS)}개"
              f"  ({time.time()-t0:.0f}s)")

        for name, cfg in CANDS.items():
            cfg = dict(cfg)
            keep = ([i for i in range(len(EMB_COLS))] if cfg.pop("use_ids")
                    else [i for i, c in enumerate(EMB_COLS) if c not in ID_COLS])
            ec = [EMB_COLS[i] for i in keep]
            sz = [sizes[i] for i in keep]
            Ck, Cvk = C[:, keep].copy(), Cv[:, keep].copy()
            zs = []
            for s in SEEDS:
                _, curve = fm.train_fm(A, Ck, ytr, w, B, Cvk, sz, ec, seed=s, **cfg)
                zs.append(curve[-1])          # 설정에 박힌 에폭 수의 결과. fold를 보고 고르지 않는다
            z = np.mean(zs, axis=0)
            cand.setdefault(name, {})[f] = z
            print(f"  {name:<12s} 단독 {mlp.score(z, yva):>7.1f}   "
                  f"corr {np.corrcoef(Z[f], z)[0,1]:.4f}   ({time.time()-t0:.0f}s)")

    base = np.array([mlp.score(Z[f], Y[f]) for f in FOLDS])
    print(f"\n기준(현재 블렌드): " + "  ".join(f"{f}={b:.1f}" for f, b in zip(FOLDS, base)))

    print(f"\n{'후보':<12s} {'beta':>5s} " + " ".join(f"{f:>8d}" for f in FOLDS)
          + f" {'기준대비':>9s} {'크기비':>7s} {'corr':>7s}")
    print("-" * 66)
    for n in cand:
        corr = np.mean([np.corrcoef(Z[f], cand[n][f])[0, 1] for f in FOLDS])
        for b in BETAS:
            v = np.array([mlp.score((1 - b) * Z[f] + b * cand[n][f], Y[f]) for f in FOLDS])
            d = v - base
            same, sp, ok = verdict(d)
            print(f"{n:<12s} {b:>5.2f} " + " ".join(f"{x:>8.1f}" for x in v)
                  + f" {d.mean():>+9.1f} "
                  + (f"{sp:>7.1f}" if same else f"{'inf':>7s}")
                  + f" {corr:>7.4f}" + ("  <<< 채택 조건 통과" if ok else ""))

    print("\n  ※ k=0(선형)과 k>0(FM)의 차이가 곧 **쌍 상호작용 항의 기여**다.")
    print("     둘이 비슷하면 이득의 원천은 상호작용이 아니라 그냥 다른 함수족이라는 뜻.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
