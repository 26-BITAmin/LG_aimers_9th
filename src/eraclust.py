"""`erashrink.py`의 t를 **투수 클러스터**로 다시 잰다 — 그리고 β를 학습으로 고른다.

왜 다시 재는가
--------------
`resid_t`는 행을 독립으로 본다. 그런데 회귀자 `del_career[p]`는 **투수 안에서 상수**이고
잔차도 투수 안에서 상관돼 있다 (그 시즌 그 투수가 잘했나 못했나). 유효 표본은
25만 행이 아니라 **투수 ~310명**이다. → 순진한 t는 부풀어 있다.

[[report-effect-size-with-paired-se]] 가 적어둔 그대로다. 이번엔 클러스터다.

  ① 투수 클러스터 로버스트 SE (sandwich)
  ② 투수 블록 부트스트랩으로 **BSS 이득의 신뢰구간**
  ③ **β를 fold 밖에서 고를 수 있는가** — fold2023에서 고른 β를 fold2024에 적용.
     이게 되어야 2025에 쓸 β를 정당하게 고른 것이 된다.
     (안 되면 이 축은 §7.22의 "fold 간 외삽 식별 불가"를 다시 밟는 것이다)
  ④ 일반 수축과 구별되는가 — global_temp를 같은 절차로 통과시켜 본다

실행:  .venv/Scripts/python.exe -u src/eraclust.py 2>&1 | tee experiments/eraclust.log
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
from config import CTX_AXES, CTX_K, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from ctxsplit import build_delta, lookup, resid_t
from data import load_csv
from erafit import MARCEL_W, league_rates, pitcher_estimate, season_table
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
K = CTX_K
GRID = np.arange(-1.75, 0.51, 0.05)
N_BOOT = 400


def cluster_t(r, d, g):
    """투수 클러스터 로버스트 t. g = 투수 그룹 인덱스."""
    dc = d - d.mean()
    den = float((dc ** 2).sum())
    if den <= 0:
        return 0.0, 0.0, 0.0
    sl = float((dc * r).sum() / den)
    e = (r - r.mean()) - sl * dc
    # 클러스터별 score 합
    ug, inv = np.unique(g, return_inverse=True)
    sc = np.bincount(inv, weights=dc * e, minlength=len(ug))
    meat = float((sc ** 2).sum())
    se = np.sqrt(meat) / den
    # 순진한 SE (대조)
    se_naive = float(np.sqrt((e ** 2).sum() / (len(r) - 2) / den))
    return sl, (sl / se if se > 0 else 0.0), (sl / se_naive if se_naive > 0 else 0.0)


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    pid = df["pitcher_id"].to_numpy(dtype=np.int64)
    bh = df["batter_hand"].to_numpy(dtype=np.int64)
    b = df["balls_before"].to_numpy(dtype=np.int64)
    st = df["strikes_before"].to_numpy(dtype=np.int64)
    c4 = (b >= 3).astype(np.int64) + 2 * (st >= 2).astype(np.int64)

    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(42)
    store = {}

    for fold in FOLDS:
        tr, va = s < fold, s == fold
        seasons = sorted(int(t) for t in np.unique(s[tr]))
        lg = league_rates(s[tr], y[tr], seasons)
        upit = np.unique(pid[tr])
        ntab, rtab = season_table(pid[tr], s[tr], y[tr], seasons)
        w_mar = [0.0] * len(seasons)
        for i, wv in enumerate(MARCEL_W):
            if len(seasons) - 1 - i >= 0:
                w_mar[len(seasons) - 1 - i] = wv

        d_car, _ = pitcher_estimate(upit, ntab, rtab, seasons, lg,
                                    [1.0] * len(seasons), True)
        d_mar, _ = pitcher_estimate(upit, ntab, rtab, seasons, lg, w_mar, True)
        jv = np.clip(np.searchsorted(upit, pid[va]), 0, len(upit) - 1)
        hit = upit[jv] == pid[va]

        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        z0 = z.copy()
        for name, beta in CTX_AXES:
            code, m = (bh, 8) if name == "hand" else (c4, 4)
            u, d, _ = build_delta(pid[tr], code[tr], y[tr], m, K)
            drow, _ = lookup(u, d, pid[va], code[va], m)
            z0 = z0 + beta * drow
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean())
        base = bss(yv, p0)
        r = yv - p0
        gv = pid[va]

        regs = {"del_career": np.where(hit, d_car[jv], 0.0),
                "del_marcel": np.where(hit, d_mar[jv], 0.0),
                "global_temp": z0 - z0.mean()}

        print("=" * 100)
        print(f"fold {fold}   valid {va.sum():,}   투수 {len(np.unique(gv))}명"
              f"   기준 BSS {base:.1f}")
        print("=" * 100)
        print(f"  {'회귀자':<14}{'기울기':>11}{'t(순진)':>10}{'t(클러스터)':>13}{'부풀림':>9}")
        for tag, v in regs.items():
            sl, tc, tn = cluster_t(r, v, gv)
            print(f"  {tag:<14}{sl:>+11.4f}{tn:>+10.2f}{tc:>+13.2f}"
                  f"{abs(tn / tc) if tc else float('nan'):>9.1f}x")

        # ── β 곡선 + 투수 블록 부트스트랩 ─────────────────────────
        print(f"\n  β 곡선과 투수 블록 부트스트랩 ({N_BOOT}회, 투수 단위 재표집)")
        ug, inv = np.unique(gv, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        starts = np.searchsorted(inv[order], np.arange(len(ug)))
        ends = np.append(starts[1:], len(order))
        idx_by_g = [order[a:b] for a, b in zip(starts, ends)]

        for tag in ("del_career", "del_marcel", "global_temp"):
            v = regs[tag]
            curve = np.array([bss(yv, shift_to_prior(
                1.0 / (1.0 + np.exp(-(z0 + bb * v))), yv.mean())) - base
                for bb in GRID])
            i_star = int(np.argmax(curve))
            b_star, g_star = float(GRID[i_star]), float(curve[i_star])

            # 부트스트랩: 고정 β(=-0.5)에서의 이득 분포
            bfix = -0.5
            pf = 1.0 / (1.0 + np.exp(-(z0 + bfix * v)))
            boot = np.empty(N_BOOT)
            for t_ in range(N_BOOT):
                pick = rng.integers(0, len(ug), len(ug))
                ii = np.concatenate([idx_by_g[q] for q in pick])
                yb = yv[ii]
                boot[t_] = (bss(yb, shift_to_prior(pf[ii], yb.mean()))
                            - bss(yb, shift_to_prior(p0[ii], yb.mean())))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            store.setdefault(tag, {})[fold] = (b_star, g_star, curve,
                                               float(boot.mean()), lo, hi)
            print(f"    {tag:<14} β*={b_star:+.2f}  이득 {g_star:+7.2f}   |"
                  f"  β=-0.50 이득 {boot.mean():+7.2f}  95%CI [{lo:+.2f}, {hi:+.2f}]"
                  f"  {'0 배제' if lo > 0 else '0 포함'}")
        print()

    # ── 교차 fold 이전 — β를 fold 밖에서 고를 수 있는가 ────────────
    print("=" * 100)
    print("교차 fold β 이전 — 한 fold에서 고른 β*를 다른 fold에 적용")
    print("(이게 되어야 2025에 쓸 β를 정당하게 고른 것이 된다. §7.22의 식별 문제)")
    print("=" * 100)
    print(f"  {'회귀자':<14}{'출처':>8}{'β*':>8}{'자기 이득':>11}{'→ 상대 fold 이득':>18}{'최적 대비':>11}")
    for tag, per in store.items():
        for src, dst in ((FOLDS[0], FOLDS[1]), (FOLDS[1], FOLDS[0])):
            b_star = per[src][0]
            curve_d = per[dst][2]
            g_at = float(curve_d[int(np.argmin(np.abs(GRID - b_star)))])
            print(f"  {tag:<14}{src:>8}{b_star:>+8.2f}{per[src][1]:>+11.2f}"
                  f"{g_at:>+18.2f}{g_at / max(per[dst][1], 1e-9):>10.0%}")
    print("\n  '최적 대비' = 상대 fold의 자체 최적 이득 중 몇 %를 건졌는가.")
    print("  100%에 가까우면 β가 fold를 넘어 전이된다는 뜻이다.")


if __name__ == "__main__":
    main()
