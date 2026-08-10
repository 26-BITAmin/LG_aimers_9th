"""약한 멤버를 섞어 얻은 이득이 **다양성인가 수축인가.**

`ablate21`에서 후보·β를 막론하고 fold2023은 항상 오르고 fold2024는 항상 내렸다.
25행 전부. 이 정도로 일관되면 후보의 성질이 아니라 fold의 성질이다.

`ablate20`이 이미 아는 사실: fold2023의 블렌드는 실력 대비 과신이라 c=0.90에서 +87이다.
약한 모델을 섞으면 예측이 중앙으로 수축한다 — fold2023이 원하는 바로 그것이다.

그래서 **분산을 맞춘 뒤에도 이득이 남는지** 본다.

    원본      z
    혼합      z_b = (1-b) z + b z_cand
    등분산    z_b * (std(z) / std(z_b))      ← 수축분을 되돌린 것

등분산 이득이 0에 가까우면 그 이득은 다양성이 아니라 수축이었다. 그 경우
"약한 멤버 추가"는 온도 c로 대체 가능하고, c는 `ablate20`에서 이미 기각됐다.

실행:  .venv/Scripts/python.exe -u src/blend_iso.py
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
BETAS = (0.10, 0.20, 0.30)
ID_COLS = ("pitcher_id", "batter_id")
fm.THREADS = 6


def main():
    t0 = time.time()
    cache = foldcache.load(FOLDS)
    df = load_csv(TRAIN_CSV)

    print(f"\n{'fold':>5s} {'beta':>5s} {'원본Δ':>9s} {'등분산Δ':>9s} "
          f"{'std비':>7s}   해석")
    print("-" * 62)
    for f in FOLDS:
        z_lgb, z_mlp, y = cache[f]
        Z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
        Xtr, ytr, w, Xva, yva, art = foldcache.fold_data(df, f)
        A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
        del Xtr, Xva
        keep = [i for i, c in enumerate(EMB_COLS) if c not in ID_COLS]
        zs = []
        for s in (42, 7):
            _, curve = fm.train_fm(A, C[:, keep].copy(), ytr, w, B, Cv[:, keep].copy(),
                                   [sizes[i] for i in keep], [EMB_COLS[i] for i in keep],
                                   seed=s, k=0, epochs=4)
            zs.append(curve[-1])
        z_lin = np.mean(zs, axis=0)

        base = mlp.score(Z, y)
        s0 = Z.std()
        for b in BETAS:
            zb = (1 - b) * Z + b * z_lin
            raw = mlp.score(zb, y) - base
            iso = mlp.score(zb * (s0 / zb.std()), y) - base
            tag = ("수축이 거의 전부" if abs(raw) > 3 and abs(iso) < abs(raw) * 0.35
                   else "다양성 기여 있음" if iso > 1.0 else "이득 없음")
            print(f"{f:>5d} {b:>5.2f} {raw:>+9.1f} {iso:>+9.1f} "
                  f"{zb.std()/s0:>7.4f}   {tag}")
        print(f"      (fold {f} 완료 {time.time()-t0:.0f}s)")

    print("\n  ※ 등분산Δ가 원본Δ보다 훨씬 작으면 그 이득은 다양성이 아니라 수축이다.")
    print("     수축은 온도 c로 직접 얻을 수 있고, c는 ablate20에서 이미 기각됐다")
    print("     (두 fold가 반대 방향을 원해서).")


if __name__ == "__main__":
    main()
