"""1단계 — soft label h를 **투수별 reverse 성향**으로 행 단위 세분화 (§7.70 후속)

0단계(softlabel.py)는 시즌 상수 h_s만 썼다: 같은 시즌의 모든 y=1 행이 같은 만큼
깎인다. 그런데 숨은 reverse일 확률은 투수마다 다르다 — reverse 성향이 큰 투수의
y=1이 더 의심스럽다. 이 성향은 §7.29에서 주 신호축으로 확인된 바로 그 값이다.

    h_i = h_s × r_i / r̄_s        (시즌 평균은 h_s로 보존 — 총량 캘리브레이션 유지)
    r_i = 보정(§7.49 D/m) + 스무딩(k=200)된 asof_pitcher_reverse_rate

판정: fold2024에서 Δ(soft2 − soft) ≤ 0 이면 세분화는 죽은 것 — 0단계 유지.

실행:  .venv/Scripts/python.exe -u src/softlabel2.py 2>&1 | tee experiments/softlabel2.log
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

import contam
from config import (DROP_FEATURES, RECENCY_HALF_LIFE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS,
                    CONTAM_AXES)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from softlabel import contam_mask, fit_h, train_eval

FOLDS = [2024, 2023]
SEEDS = (42, 7, 2024)


def prep(df, valid_season):
    tr = df[df.season < valid_season]
    va = df[df.season == valid_season]
    ref = int(tr.season.max())
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=ref)
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    return tr, va, w, X, art


def rev_propensity(tr, art):
    """보정 + 스무딩된 투수 reverse 성향 (행 단위)."""
    adj = contam.adjust(tr, art["contam"], "p")
    raw = tr["asof_pitcher_reverse_rate"].to_numpy(dtype=np.float64)
    n = np.nan_to_num(tr["asof_pitcher_n"].to_numpy(dtype=np.float64))
    prior = float(art["priors"]["asof_pitcher_reverse_rate"])
    r = np.where(np.isfinite(raw), raw, prior) + adj
    return (n * r + 200.0 * prior) / (n + 200.0)


def make_h_rows(tr, ytr, h, r_sm=None):
    """행 단위 h_i. r_sm이 None이면 시즌 상수(0단계와 동일)."""
    cm = contam_mask(tr)
    s = tr["season"].to_numpy()
    h_row = np.zeros(len(tr))
    for yr, v in h.items():
        m = cm & (s == yr) & (ytr == 1)
        if m.sum() == 0:
            continue
        if r_sm is None:
            h_row[m] = v
        else:
            h_row[m] = np.clip(r_sm[m] * (v / max(r_sm[m].mean(), 1e-9)), 0.0, 0.95)
    return h_row


def run_soft(tr, va, w, X, h_row, seeds):
    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    va_f = va["game_type"].to_numpy().astype(str) == "F"
    Xtr = X(tr)
    dup = h_row > 0
    w_soft = w.copy()
    w_soft[dup] *= (1 - h_row[dup])
    Xtr2 = pd.concat([Xtr, Xtr[dup]], ignore_index=True)
    ytr2 = np.concatenate([ytr, np.zeros(int(dup.sum()))])
    w2 = np.concatenate([w_soft, w[dup] * h_row[dup]])
    return train_eval(Xtr2, ytr2, w2, X(va), yva, va_f, seeds)


def main():
    df = load_csv(TRAIN_CSV)
    t0 = time.time()
    res = {}

    for fold in FOLDS:
        print(f"\n=== fold {fold} (train < {fold}) ===")
        tr, va, w, X, art = prep(df, fold)
        ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
        yva = va[TARGET_COL].to_numpy(dtype=np.float64)
        va_f = va["game_type"].to_numpy().astype(str) == "F"
        h = fit_h(df, (df.season < fold).to_numpy())
        r_sm = rev_propensity(tr, art)

        res[("base", fold)] = train_eval(X(tr), ytr, w, X(va), yva, va_f, SEEDS)
        print(f"  base   BSS {res[('base',fold)][:,0].mean():>8.1f}"
              f" ± {res[('base',fold)][:,0].std(ddof=1):.1f}   ({time.time()-t0:.0f}s)")

        for arm, rs in [("soft", None), ("soft2", r_sm)]:
            h_row = make_h_rows(tr, ytr, h, rs)
            m = h_row > 0
            print(f"    {arm}: {m.sum():,}행  h 분포 q10 {np.quantile(h_row[m],.1):.3f}"
                  f"  중앙 {np.median(h_row[m]):.3f}  q90 {np.quantile(h_row[m],.9):.3f}"
                  f"  최대 {h_row[m].max():.3f}")
            res[(arm, fold)] = run_soft(tr, va, w, X, h_row, SEEDS)
            print(f"  {arm:<6} BSS {res[(arm,fold)][:,0].mean():>8.1f}"
                  f" ± {res[(arm,fold)][:,0].std(ddof=1):.1f}"
                  f"   F행 {res[(arm,fold)][:,1].mean():>7.1f}   ({time.time()-t0:.0f}s)")

        d = res[("soft2", fold)][:, 0] - res[("soft", fold)][:, 0]
        print(f"  → Δ(soft2−soft) = {d.mean():+.1f} ± {d.std(ddof=1)/np.sqrt(len(d)):.1f}")
        if fold == 2024 and d.mean() <= 0:
            print("\n  판정: 세분화 이득 없음 — 0단계(시즌 상수) 유지. fold2023 생략.")
            return
        del tr, va, w, X

    print("\n" + "=" * 70)
    print("요약 (Δ vs base, 쌍체)")
    for arm in ["soft", "soft2"]:
        d = [res[(arm, f)][:, 0] - res[("base", f)][:, 0] for f in FOLDS]
        print(f"  {arm:<6}" + "".join(
            f"  f{f} {x.mean():+.1f}±{x.std(ddof=1)/np.sqrt(len(x)):.1f}"
            for x, f in zip(d, FOLDS)))


if __name__ == "__main__":
    main()
