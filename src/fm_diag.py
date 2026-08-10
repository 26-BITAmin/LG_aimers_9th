"""FM이 정말 나쁜 함수족인지, 아니면 내 설정이 나쁜 건지 가른다.

`ablate21`에서 FM k=8이 단독 −530.7, k=16이 −1193.4였다. **k가 커질수록 단조롭게
악화**되는 건 함수족의 한계가 아니라 과적합의 서명이다. 같은 함정을 이미 밟았다 —
MLP 1차 시도 −4670의 원인은 임베딩 16차원 + 약한 정규화로 인한 암기였고,
4차원으로 줄여 살아났다.

FM의 잠재벡터 v_i는 pitcher_id·batter_id에 대해 그 임베딩과 **같은 역할**을 한다.
게다가 FM은 쌍 상호작용을 명시하므로 pitcher_id x batter_id라는 최악의 암기 경로가
구조에 박혀 있다.

세 가지를 본다:
  1) **에폭 곡선.** 초반에 정점을 찍고 무너지면 과적합이다 (마지막 에폭만 봤었다).
  2) **k를 줄이면.** k=2,4 에서 회복되면 용량 문제다.
  3) **고카디널리티 ID를 빼면.** 회복되면 범인이 특정된다.

fold 2024만 본다 — fold 2023은 기준 BSS가 40.7뿐이라 진단용으로 부적합하다.

실행:  .venv/Scripts/python.exe -u src/fm_diag.py
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
from config import TRAIN_CSV
from data import load_csv
from mlp_infer import EMB_COLS

FOLD = 2024
ID_COLS = ("pitcher_id", "batter_id")
fm.THREADS = 6


def main():
    t0 = time.time()
    cache = foldcache.load([FOLD])
    z_lgb, z_mlp, _ = cache[FOLD]

    df = load_csv(TRAIN_CSV)
    Xtr, ytr, w, Xva, yva, art = foldcache.fold_data(df, FOLD)
    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    del Xtr, Xva, df
    print(f"수치 {A.shape[1]}개 + 범주 {len(EMB_COLS)}개  ({time.time()-t0:.0f}s)")
    print(f"범주 카디널리티: {dict(zip(EMB_COLS, sizes))}\n")

    keep_all = list(range(len(EMB_COLS)))
    keep_noid = [i for i, c in enumerate(EMB_COLS) if c not in ID_COLS]

    for tag, keep in (("전체 범주", keep_all), ("ID 제외", keep_noid)):
        ec = [EMB_COLS[i] for i in keep]
        sz = [sizes[i] for i in keep]
        Ck, Cvk = C[:, keep].copy(), Cv[:, keep].copy()
        for k in (2, 4, 8):
            _, curve = fm.train_fm(A, Ck, ytr, w, B, Cvk, sz, ec,
                                   seed=42, k=k, epochs=8)
            bs = [mlp.score(z, yva) for z in curve]
            best = int(np.argmax(bs))
            cz = curve[best]
            print(f"{tag:<9s} k={k:<3d} 에폭별 " + " ".join(f"{b:>7.0f}" for b in bs))
            print(f"{'':<9s}      최적 ep{best+1} = {bs[best]:.1f}"
                  f"   마지막 = {bs[-1]:.1f}"
                  f"   corr(blend) = {np.corrcoef(0.6*z_lgb+0.4*z_mlp, cz)[0,1]:.4f}"
                  f"   ({time.time()-t0:.0f}s)")

    print("\n  판정: 곡선이 초반 정점 후 하강 → 과적합(설정 문제).")
    print("        k를 줄이거나 ID를 빼도 안 오르면 → 함수족의 한계(진짜 기각).")


if __name__ == "__main__":
    main()
