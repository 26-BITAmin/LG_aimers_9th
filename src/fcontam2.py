"""오염 보정을 **피처값 자체**에 넣고 재학습한다 (§7.48 → §7.49)

§7.48이 남긴 문제
    사후 로짓 δ는 오염 투수의 **레벨**만 민다. 그런데 오염이 실제로 하는 일은
    주 신호축(`asof_pitcher_reverse_rate` — §7.29에서 315 중 285)의 **감쇠**다.
    누락분이 섞인 값으로 학습하면 그 구간에서 기울기가 눌린다. δ는 거기 못 닿는다.

보정식
    투구 m번째 행의 `asof_*`는 **직전 m개**의 누적이다. 그중 오염 구간(pre-2023 F)
    투구가 C개 있었다면, 누락된 '반대' 개수는 대략 Σ q_s (s = 그 투구의 시즌).

        q_s = (그 시즌 R의 반대 비율) − (그 시즌 F의 반대 비율)     ← train에서만 추정

    | 시즌 | 2019 | 2020 | 2021 | 2022 |
    | q_s  | .063 | .064 | .170 | .211 |

    (2021~2022가 크다 — 기준이 서서히 안 걸리다 2023에 켜졌다는 §7.48 ①과 정합.)

        reverse_보정 = (reverse_관측 × m + D) / m
        success_보정 = (success_관측 × m − D) / m,     D = Σ q_s

리키지
    D는 **그 행 이전 투구들**의 누적이다. 행 자신의 결과는 안 들어간다 (`asof_*`와 동일).
    2025 추론에서는 오염 투구가 전부 과거에 있으므로 D = (그 투수의 총 오염 누적)이고,
    `pitcher_id → D_total` 은 train에서 fit해 조회만 하는 표다. test 행 간 정보 없음.

    ⚠️ `q_s`도 fold의 train에서만 추정한다 (fold2023은 ≤2022, fold2024는 ≤2023).

변형
    P  : 투수 축만 보정
    PB : 투수 + 타자 축 (`asof_batter_success_rate`도 §10 드리프트 감사에서 2위였다 —
         `break_23` 0.0222. 같은 오염이 타자 쪽으로도 흐른다)

실행:  .venv/Scripts/python.exe -u src/fcontam2.py 2>&1 | tee experiments/fcontam2.log
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

import evaluate
from config import TRAIN_CSV
from data import load_csv
from modelabel import recover

FOLDS = [2023, 2024]
SEEDS = (42, 7, 2024)


def fit_q(df, rev, train_mask):
    """시즌별 '누락 반대 비율' q_s 를 fold의 train에서만 추정한다."""
    s = df["season"].to_numpy()
    g = df["game_type"].to_numpy()
    q = {}
    for yr in range(2019, 2023):
        mR = train_mask & (s == yr) & (g == "R")
        mF = train_mask & (s == yr) & (g == "F")
        if mF.sum() == 0 or mR.sum() == 0:
            continue
        q[yr] = max(0.0, float(rev[mR].mean() - rev[mF].mean()))
    return q


def deficit(df, q, id_col, n_col):
    """각 행에 대해 **직전까지** 누적된 누락 개수 D를 만든다.

    `n_col`(= asof_*_n)로 정렬하면 그 순서가 곧 시간 순이다. 오염 투구의 q를
    누적하되 **한 칸 밀어서** 자기 자신을 빼면 `asof_*`와 같은 시점이 된다.
    """
    s = df["season"].to_numpy()
    g = df["game_type"].to_numpy()
    per_pitch = np.where((s <= 2022) & (g == "F"),
                         np.array([q.get(int(y), 0.0) for y in s]), 0.0)

    key = df[id_col].to_numpy(dtype=np.int64)
    n = df[n_col].to_numpy(dtype=np.float64)
    order = np.lexsort((n, key))
    k_s, v_s = key[order], per_pitch[order]

    cum = np.cumsum(v_s)
    # 같은 선수 블록의 시작에서 offset을 빼고, 자기 자신을 제외한다
    start = np.empty(len(k_s), dtype=bool)
    start[0] = True
    start[1:] = k_s[1:] != k_s[:-1]
    base = np.where(start, cum - v_s, np.nan)
    base = pd.Series(base).ffill().to_numpy()      # 블록 시작 직전까지의 누적
    d_s = (cum - v_s) - base                       # 직전까지 (자기 제외)

    out = np.empty(len(df))
    out[order] = d_s
    return out


def correct(df, q, batter_too):
    """오염 보정된 사본을 만든다 (원본은 건드리지 않는다)."""
    out = df.copy()
    dP = deficit(df, q, "pitcher_id", "asof_pitcher_n")
    m = df["asof_pitcher_n"].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        adj = np.where(m > 0, dP / m, 0.0)
    for col, sign in [("asof_pitcher_reverse_rate", +1.0),
                      ("asof_pitcher_success_rate", -1.0)]:
        v = df[col].to_numpy(dtype=np.float64)
        out[col] = np.clip(v + sign * adj, 0.0, 1.0).astype(df[col].dtype)

    if batter_too:
        dB = deficit(df, q, "batter_id", "asof_batter_n")
        mb = df["asof_batter_n"].to_numpy(dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            adjb = np.where(mb > 0, dB / mb, 0.0)
        col = "asof_batter_success_rate"
        v = df[col].to_numpy(dtype=np.float64)
        out[col] = np.clip(v - adjb, 0.0, 1.0).astype(df[col].dtype)
    return out


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    lab, _, _ = recover(df)
    rev = np.where(np.isnan(lab["reverse"]), 0.0, np.round(lab["reverse"]))

    t0 = time.time()
    res = {}
    for fold in FOLDS:
        tr = s < fold
        q = fit_q(df, rev, tr)
        print(f"\nfold {fold}  q_s = " + "  ".join(f"{y}:{v:.3f}" for y, v in sorted(q.items())))

        variants = {"base": df, "P": correct(df, q, False), "PB": correct(df, q, True)}
        for nm, d in variants.items():
            if nm != "base":
                dd = (d["asof_pitcher_reverse_rate"].to_numpy(dtype=np.float64)
                      - df["asof_pitcher_reverse_rate"].to_numpy(dtype=np.float64))
                v = dd[s == fold]
                print(f"  {nm}: valid행 Δreverse 평균 {v.mean():+.5f}  "
                      f"q90 {np.quantile(v, .9):+.5f}  최대 {v.max():+.5f}")
            r = evaluate.run_fold(d, fold, seeds=SEEDS)
            res[(nm, fold)] = r
            print(f"  {nm:<6} valid={fold}  BSS {r.mean():>8.1f} ± {r.std(ddof=1):.1f}"
                  f"   ({time.time()-t0:.0f}s)")
            del d

    print("\n" + "=" * 78)
    print("요약 (BSS oracle, 3시드)")
    print("=" * 78)
    print(f"  {'변형':<8}" + "".join(f"{f:>12d}" for f in FOLDS))
    for nm in ["base", "P", "PB"]:
        print(f"  {nm:<8}" + "".join(f"{res[(nm,f)].mean():>12.1f}" for f in FOLDS))
    print(f"\n  {'차이(vs base)':<14}" + "".join(f"{f:>10d}" for f in FOLDS) + "   크기비")
    for nm in ["P", "PB"]:
        d = [res[(nm, f)].mean() - res[("base", f)].mean() for f in FOLDS]
        same = (d[0] > 0) == (d[1] > 0)
        ratio = max(abs(d[0]), abs(d[1])) / max(min(abs(d[0]), abs(d[1])), 1e-9)
        print(f"  {nm:<14}" + "".join(f"{x:>+10.1f}" for x in d)
              + f"   {ratio:>5.1f}배 {'(부호 일치)' if same else '(**불일치**)'}")


if __name__ == "__main__":
    main()
