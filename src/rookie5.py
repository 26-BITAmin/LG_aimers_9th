"""rookie4의 두 축(투수간 상한 · 포착률)에 클러스터 부트스트랩 SE를 붙이고,
fold2023→2024 R 신인 갭 +515를 세 성분으로 배분한다.

  로컬BSS ≈ 상한 × 포착률 + 투수내
  Δ = (상한 변화) + (포착률 변화) + (투수내 변화)

실행:  .venv/Scripts/python.exe -u src/rookie5.py 2>&1 | tee experiments/rookie5.log
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
from rookie4 import MINN, agg, parts

B = 2000
RNG = np.random.default_rng(42)


def cap_capture(n, ys, ps, sq):
    """(투수간 상한, 포착률, 투수간, 투수내). 이항 노이즈를 뺀 진짜 분산이 분모."""
    N = n.sum()
    yb = ys.sum() / N
    ref = yb * (1 - yb)
    _, btw, wit = parts(n, ys, ps, sq)
    k = n >= MINN
    if k.sum() < 5:
        return np.nan, np.nan, btw, wit
    yj, nj = ys[k] / n[k], n[k]
    w = nj / nj.sum()
    mu = (w * yj).sum()
    var_true = max((w * (yj - mu) ** 2).sum() - (w * yj * (1 - yj) / nj).sum(), 1e-9)
    cap = 1e5 * var_true / ref
    return cap, btw / cap, btw, wit


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)

    keep = {}
    print(f"{'fold/셀':<16}{'상한':>9}{'±SE':>8}{'포착률':>9}{'±SE':>8}"
          f"{'투수내':>9}{'±SE':>7}")
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
            n, ys, ps, sq = agg(cl, ncl, yg, pg)
            c0, r0, b0, w0 = cap_capture(n, ys, ps, sq)
            bt = np.empty((B, 3))
            for b in range(B):
                i = RNG.integers(0, ncl, ncl)
                cc, rr, _, ww = cap_capture(n[i], ys[i], ps[i], sq[i])
                bt[b] = (cc, rr, ww)
            se = np.nanstd(bt, axis=0, ddof=1)
            keep[(fold, lab)] = (c0, r0, w0, se)
            print(f"{str(fold)+' '+lab:<16}{c0:>9.0f}{se[0]:>8.0f}"
                  f"{r0*100:>8.0f}%{se[1]*100:>7.0f}%{w0:>9.1f}{se[2]:>7.1f}")

    print("\n[R 신인] fold2023 → fold2024 갭 +514.7의 배분")
    c1, r1, w1, _ = keep[(2023, "R 신인")]
    c2, r2, w2, _ = keep[(2024, "R 신인")]
    a = (c2 - c1) * r1
    b = c2 * (r2 - r1)
    print(f"  ① 상한 증가   {c1:.0f} → {c2:.0f}  (신인 투수단의 실력 분산)  {a:>+8.1f}")
    print(f"  ② 포착률 증가 {r1*100:.0f}% → {r2*100:.0f}%                      {b:>+8.1f}")
    print(f"  ③ 투수내      {w1:+.1f} → {w2:+.1f}                     {w2-w1:>+8.1f}")
    print(f"  합계 {a+b+w2-w1:+.1f}")

    print("\n[투수내] 신인 − 2+시즌 (양수 = 신인이 상황 신호를 더 씀)")
    ms, ss = [], []
    for fold in FOLDS:
        for g in ("R", "F"):
            _, _, wr, sr = keep[(fold, f"{g} 신인")]
            _, _, wv, sv = keep[(fold, f"{g} 2+시즌")]
            d = wr - wv
            e = float(np.hypot(sr[2], sv[2]))
            ms.append(d); ss.append(e)
            print(f"  {fold} {g}: {wr:>7.1f} − {wv:>7.1f} = {d:>+7.1f} ± {e:>5.1f}  (t {d/e:+.1f})")
    ms, ss = np.array(ms), np.array(ss)
    wgt = 1 / ss ** 2
    mu = (ms * wgt).sum() / wgt.sum()
    se = wgt.sum() ** -0.5
    print(f"  결합 (4셀 역분산): {mu:+.1f} ± {se:.1f}  (t {mu/se:+.2f})")


if __name__ == "__main__":
    main()
