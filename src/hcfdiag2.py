"""§7.195b — [B] R-전용 악화의 원인 분리: F 정보 상실 vs 표본 축소 (위약 대조)

사용자 지적: "F 빼면 악화되는 것은 불량 데이터가 의미 있어서가 아니라 그냥
데이터 개수 때문 아니냐." — §7.195 [B] 는 'F 제거'와 '행 10.5% 축소'를 동시에
걸어서 두 원인을 못 가른다. 여기서 분리한다.

설계 (사전 등록)
  · FULL      : clean 전체행 fit (기준)
  · R-ONLY    : F 행 전부 제거 (nF 행 감소)
  · PLACEBO×8 : F 는 유지하고 무작위 행 nF 개 제거 (같은 개수, draw 8)
  판정: R-ONLY 의 전이 Δ 가 PLACEBO 분포 (mean±sd, min~max) 안이면 악화는
  표본 축소로 설명된다 → §7.195 의 "F 행이 과적합을 눌렀다" 문장 철회.
  분포 밖(더 나쁨)이면 F 행 고유의 정보/정칙화 몫이 실재.
  어느 쪽이든 [B] 의 주 판정("R-전용은 개선이 아니다")은 불변.

실행:  .venv/Scripts/python.exe -u src/hcfdiag2.py 2>&1 | tee experiments/hcfdiag2.log
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
import pandas as pd

import deploy29 as D
import hillclimb as H
from hcfdiag import build_groups, subset_delta

FOLDS = (2023, 2024)
STEPS = 60
MARKS = (10, 20, 40, 60)
N_DRAW = 8
RNG_SEED = 7


def run_arm(a, b, S_ref, P_ref, sf, ef, fmask, g_gt_b):
    Za = a["Z"][a["mask"]]
    hist, _ = H.greedy(Za[fmask], a["ax"][fmask], a["yv"][fmask],
                       np.arange(Za.shape[1]), max(MARKS))
    out = []
    for k in MARKS:
        w = hist[k - 1] / k
        p_tr, s_tr = H.eval_w(b, b["Z"], w)
        out.append((s_tr - S_ref[ef],
                    subset_delta(p_tr, P_ref[ef], b["yv"], g_gt_b == 0)))
    return out                                       # [(Δ전체, ΔR행)] × marks


def main():
    t0 = time.time()
    print(f"§7.195b R-전용 악화 원인 분리   지문 {D.KEY}   "
          f"STEPS={STEPS} DRAWS={N_DRAW}")
    df, s, y = D.load_df()
    codes, _ = pd.factorize(df["pitcher_id"].to_numpy())
    ST = {f: H.make_state(df, s, y, f, t0) for f in FOLDS}
    GRP = {}
    for f in FOLDS:
        vi = np.where(s == f)[0]
        GRP[f] = build_groups(df, s, codes, f, vi, ST[f]["mask"])
    del df

    P_ref, S_ref = {}, {}
    for f in FOLDS:
        st = ST[f]
        p, sc = H.eval_w(st, st["Z"], H.baseline_w(st["fam"], "B1"))
        P_ref[f], S_ref[f] = p, sc

    rng = np.random.default_rng(RNG_SEED)
    for sf, ef in ((2023, 2024), (2024, 2023)):
        a, b = ST[sf], ST[ef]
        (g_gt_a, _), _ = GRP[sf]
        (g_gt_b, _), _ = GRP[ef]
        n = len(g_gt_a)
        nF = int((g_gt_a == 1).sum())
        print(f"\nsel={sf}→{ef}   clean {n}행, F {nF} ({nF/n:.3f})"
              f"   ({time.time()-t0:.0f}s)")

        full = run_arm(a, b, S_ref, P_ref, sf, ef,
                       np.ones(n, bool), g_gt_b)
        ronly = run_arm(a, b, S_ref, P_ref, sf, ef,
                        g_gt_a == 0, g_gt_b)
        plc = []
        for d in range(N_DRAW):
            drop = rng.choice(n, nF, replace=False)
            m = np.ones(n, bool)
            m[drop] = False
            plc.append(run_arm(a, b, S_ref, P_ref, sf, ef, m, g_gt_b))
            print(f"  placebo {d+1}/{N_DRAW} 완료 ({time.time()-t0:.0f}s)")
        plc = np.array(plc)                          # (draw, mark, 2)

        for j, tag in ((0, "전이 Δ(전체)"), (1, "전이 Δ(R행)")):
            print(f"  {tag}  k:        " + "".join(f"{k:>9d}" for k in MARKS))
            print("    FULL         " + "".join(f"{v[j]:>+9.2f}" for v in full))
            print("    R-ONLY       " + "".join(f"{v[j]:>+9.2f}" for v in ronly))
            mu, sd = plc[:, :, j].mean(axis=0), plc[:, :, j].std(axis=0)
            lo, hi = plc[:, :, j].min(axis=0), plc[:, :, j].max(axis=0)
            print("    PLACEBO μ    " + "".join(f"{v:>+9.2f}" for v in mu))
            print("    PLACEBO σ    " + "".join(f"{v:>9.2f}" for v in sd))
            print("    PLACEBO 범위 " + "  ".join(
                f"[{l:+.1f},{h:+.1f}]" for l, h in zip(lo, hi)))
            zsc = (np.array([v[j] for v in ronly]) - mu) / np.maximum(sd, 1e-9)
            print("    R-ONLY z     " + "".join(f"{v:>+9.2f}" for v in zsc))

    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
