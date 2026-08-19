"""twospeed의 영가설 — "그냥 전체를 줄이면 되는 것 아닌가".

세 가지를 같은 캐시 z 위에서 비교한다.
  ① 전역 temperature       z' = T·z            (구조 없음)
  ② 투수간만 조정          z' = z̄_j·T + (z−z̄_j)
  ③ 두 속도 (a=1.4,s=0.7)  twospeed 채택 후보
①이 ③만큼 준다면 구조 이야기는 필요 없다. 죽은 축 영가설의 자리다.

실행:  LGA_MODEL_DIR=submit/model_tempnull .venv/Scripts/python.exe -u src/tempnull.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache
from clockaxis import gmean, qbucket
from config import TRAIN_CSV
from data import load_csv
from rookie import FOLDS, deployed_pred
from validation import _logit, bss, shift_to_prior

TG = [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1]


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    for fold in FOLDS:
        va, y, p = deployed_pred(df, s, fold, cached)
        z = _logit(p)
        _, cl = np.unique(va["pitcher_id"].to_numpy(), return_inverse=True)
        an = va["asof_pitcher_n"].to_numpy().astype(float)
        zj = gmean(cl, z)[cl]
        g = cl.astype(np.int64) * 1000 + qbucket(cl, an, 5)
        _, g = np.unique(g, return_inverse=True)
        zb = gmean(g, z)[g]
        base = bss(y, p)

        def d(zz):
            return bss(y, shift_to_prior(1 / (1 + np.exp(-zz)), y.mean())) - base

        print(f"\n=== fold {fold}  기준 {base:.1f} ===")
        print("  ① 전역 temperature   " +
              "  ".join(f"T={t}:{d(t*z):+.1f}" for t in TG))
        print("  ② 투수간만 스케일   " +
              "  ".join(f"T={t}:{d(t*zj + (z-zj)):+.1f}" for t in TG))
        print(f"  ③ 두 속도 a=1.4 s=0.7  {d(zj + 1.4*(zb-zj) + 0.7*(z-zb)):+.1f}")


if __name__ == "__main__":
    main()
