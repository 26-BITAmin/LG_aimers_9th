"""score_diff_home 잔차 구조의 위치 특정 — §7.77 부수 발견의 후속.

사용자 가설: 구조가 0(동점)에 몰려 있다 → is_tied 이진 처리가 유효할 것.
검증 항목:
  ① 정수값별 평균 잔차 프로파일 (블렌드 0.6·lgb+0.4·mlp, 양 fold, t값 포함)
     — 스파이크가 0에 있는가, 축 전체에 퍼져 있는가.
  ② 동점 행을 이닝별로 분해 — 0에는 모든 경기의 1회가 섞인다. "동점 효과"가
     실은 "초반 이닝 효과"일 수 있고, inning은 이미 피처로 들어가 있으므로
     그 경우 새 축이 아니다.
  ③ 양 fold 부호 일치 여부 — 시즌 간 지속성 없는 구조는 후보가 아니다.

실행:  .venv/Scripts/python.exe -u src/tiedprobe.py
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
from config import MLP_ALPHA, TRAIN_CSV
from data import load_csv

FOLDS = [2023, 2024]
CLIP = 6                 # |점수차| 6 이상은 한 셀로 묶는다 (표본 희소)


def profile(v, r, var_row, lo=-CLIP, hi=CLIP):
    vc = np.clip(v, lo, hi)
    print(f"  {'값':>4} {'n':>8} {'r̄':>9} {'t':>7}   δ로짓 환산")
    for val in range(lo, hi + 1):
        m = vc == val
        n = int(m.sum())
        if n == 0:
            continue
        rb = float(r[m].mean())
        se = float(np.sqrt(var_row[m].mean() / n))
        dl = rb / float(var_row[m].mean())        # 로짓 가산으로 환산한 크기
        tag = " ←" if val == 0 else ""
        print(f"  {val:>4} {n:>8,} {rb:>+9.4f} {rb/se:>7.1f}   {dl:>+7.3f}{tag}")


def main():
    cache = foldcache.load(FOLDS, verbose=False)
    df = load_csv(TRAIN_CSV)

    for f in FOLDS:
        z_lgb, z_mlp, y, _ = cache[f]
        _, _, _, Xva, yva, _, _ = foldcache.fold_data(df, f)
        assert np.array_equal(y, yva)

        z = (1.0 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
        p = 1.0 / (1.0 + np.exp(-z))
        r, var_row = y - p, p * (1.0 - p)
        v = Xva["score_diff_home"].to_numpy(dtype=np.float64)
        inn = Xva["inning"].to_numpy(dtype=np.float64)

        print(f"\n=== fold {f}  (블렌드 잔차, n={len(y):,}) ===")
        print("① 점수차별 프로파일")
        profile(v, r, var_row)

        print("② 동점(0) 행의 이닝별 분해")
        m0 = v == 0
        print(f"  {'이닝':>4} {'n':>8} {'r̄':>9} {'t':>7}")
        for i in range(1, 10):
            m = m0 & (np.clip(inn, 1, 9) == i)
            n = int(m.sum())
            if n < 100:
                continue
            rb = float(r[m].mean())
            se = float(np.sqrt(var_row[m].mean() / n))
            print(f"  {i:>4} {n:>8,} {rb:>+9.4f} {rb/se:>7.1f}")

        rb0 = float(r[m0].mean())
        se0 = float(np.sqrt(var_row[m0].mean() / m0.sum()))
        rb1 = float(r[~m0].mean())
        se1 = float(np.sqrt(var_row[~m0].mean() / (~m0).sum()))
        print(f"  동점 전체   r̄ {rb0:+.4f} (t {rb0/se0:+.1f})   "
              f"비동점 전체 r̄ {rb1:+.4f} (t {rb1/se1:+.1f})")


if __name__ == "__main__":
    main()
