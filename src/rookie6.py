"""투수내 신호가 신인에서 0인 이유를 한 걸음 더 — **투수내 캘리브레이션 기울기**.

각 투수 안에서 p와 y를 평균 제거한 뒤
    slope = Σ(p−p̄_j)(y−ȳ_j) / Σ(p−p̄_j)²
slope ≈ 1 이면 투수내 변동이 정확히 실현된다. 0에 가까우면 그 변동은 노이즈다.
sd_within(p)도 같이 본다 — 기울기가 0인데 변동폭까지 크면 Brier를 깎아먹는다.

실행:  .venv/Scripts/python.exe -u src/rookie6.py 2>&1 | tee experiments/rookie6.log
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
from config import TRAIN_CSV
from data import load_csv
from rookie import FOLDS, deployed_pred, prior_seasons

B = 1000
RNG = np.random.default_rng(42)


def slope_parts(cl, ncl, y, p):
    """투수별 Σ(p−p̄)(y−ȳ), Σ(p−p̄)², n — 부트스트랩은 이걸 합치기만 한다."""
    n = np.bincount(cl, minlength=ncl).astype(float)
    sp = np.bincount(cl, weights=p, minlength=ncl)
    sy = np.bincount(cl, weights=y, minlength=ncl)
    spy = np.bincount(cl, weights=p * y, minlength=ncl)
    spp = np.bincount(cl, weights=p * p, minlength=ncl)
    ok = n > 0
    num = np.zeros(ncl); den = np.zeros(ncl)
    num[ok] = spy[ok] - sp[ok] * sy[ok] / n[ok]
    den[ok] = spp[ok] - sp[ok] ** 2 / n[ok]
    return n, num, den


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)

    print(f"{'fold/셀':<16}{'sd_within(p)':>14}{'투수내 기울기':>15}{'±SE':>8}")
    for fold in FOLDS:
        va, y, p = deployed_pred(df, s, fold, cached)
        gt_f = va["game_type"].astype("object").to_numpy() == "F"
        pp = prior_seasons(df, s, fold, "pitcher_id")
        pid = va["pitcher_id"].to_numpy()
        for lab, m in (("R 신인", (~gt_f) & (pp == 0)), ("R 2+시즌", (~gt_f) & (pp >= 2)),
                       ("F 신인", gt_f & (pp == 0)), ("F 2+시즌", gt_f & (pp >= 2))):
            yg, pg = y[m], p[m]
            _, cl = np.unique(pid[m], return_inverse=True)
            ncl = cl.max() + 1
            n, num, den = slope_parts(cl, ncl, yg, pg)
            b0 = num.sum() / den.sum()
            sdw = np.sqrt(den.sum() / n.sum())
            bt = np.empty(B)
            for b in range(B):
                i = RNG.integers(0, ncl, ncl)
                bt[b] = num[i].sum() / den[i].sum()
            print(f"{str(fold)+' '+lab:<16}{sdw:>14.4f}{b0:>15.2f}"
                  f"{np.nanstd(bt, ddof=1):>8.2f}")


if __name__ == "__main__":
    main()
