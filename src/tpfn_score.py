"""저장된 전문가 로짓을 fold에 붙여 채점한다 (§7.103 TabPFN 후속).

왜 따로 있나
    fspec_tfm.py의 전체 하네스는 fold2024 ctx 12,000 F 팔 하나에 ~65분이 걸린다.
    위약은 선택 fold에서 최적 α=0.00을 골랐으므로 홀드아웃 기여가 구성상
    정확히 0.00이라 돌릴 필요가 없다. 그래서 F 팔만 워커로 따로 적합해
    .npy로 남기고, 이 스크립트가 채점만 한다.

사용:
    .venv-gpu/Scripts/python.exe -u src/tpfn_score.py <z.npy> <fold> <alpha>
예:
    ... src/tpfn_score.py C:/.../z24_tpfn_12000.npy 2024 0.80
"""
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")
sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import warnings

warnings.filterwarnings("ignore")

import numpy as np

import foldcache
from config import FSPEC_ALPHA, MLP_ALPHA, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior


def main():
    zpath, fold, alpha = sys.argv[1], sys.argv[2], float(sys.argv[3])
    fold = int(fold) if fold.isdigit() else fold

    z_spec = np.load(zpath)
    df = load_csv(TRAIN_CSV)
    cache = foldcache.load([fold])
    z_lgb, z_mlp, y, _ = cache[fold]
    z_dep = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp

    _, mva = foldcache.SPLITS[fold](df.season.to_numpy(), df.game_month.to_numpy())
    va = df[mva]
    vF = va["game_type"].to_numpy().astype(str) == "F"
    del df, cache, va
    gc.collect()

    assert len(z_spec) == int(vF.sum()), \
        f"길이 불일치: z {len(z_spec)} vs F검증 {int(vF.sum())}"

    def score(a):
        z = z_dep.copy()
        z[vF] = (1 - a) * z[vF] + a * z_spec
        return bss(y, shift_to_prior(1 / (1 + np.exp(-z)), y.mean()))

    base = score(0.0)
    print(f"fold {fold}   MLP_ALPHA={MLP_ALPHA}   F검증 {int(vF.sum()):,}행")
    print(f"  기준(α=0)            {base:.1f}")
    print(f"  선택 α={alpha:.2f} (2023H2에서 고름)  {score(alpha):.1f}   Δ {score(alpha)-base:+.2f}")
    print()
    print("  [참고] α 스윕 — 이건 홀드아웃을 커닝한 값이라 채택 근거가 아니다")
    grid = np.round(np.arange(0.0, 0.96, 0.05), 3)
    sc = [score(a) for a in grid]
    k = int(np.argmax(sc))
    for a, s in zip(grid, sc):
        mark = "  ←최적" if a == grid[k] else ("  ←선택" if abs(a - alpha) < 1e-9 else "")
        if a % 0.1 < 1e-9 or mark:
            print(f"    α {a:.2f}   {s:.1f}   Δ {s-base:+.2f}{mark}")


if __name__ == "__main__":
    main()
