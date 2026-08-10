"""컬럼별 관계 드리프트 감사 — `game_type` 같은 함정이 더 있는가.

지금까지 실제 평가에서 전이된 개선은 `game_type` 제거(+172) 하나뿐이다.
그건 "시즌 중간에 **정의가 바뀐** 컬럼"을 찾아낸 것이었고, gain 순위를 보다 우연히
발견했다. 같은 종류의 함정을 체계적으로 찾는다.

측정 방법
    각 컬럼을 구간(bin)으로 나누고, (시즌, bin)별 양성률을 구한다.
    시즌 전체 평균을 빼서 **상대 양성률**을 만든다 — 리그 전체의 하향 드리프트를 제거해
    "그 컬럼이 타깃과 맺는 관계"만 남긴다.
    이 상대값이 시즌마다 크게 흔들리면 = 관계가 불안정 = 학습한 것이 미래에 안 맞는다.

두 가지 숫자를 본다
    drift_std   시즌 간 상대 양성률의 (표본가중) 표준편차. 클수록 불안정.
    break_23    2019~2022 평균과 2023~2024 평균의 차이. 구조 변화 탐지용.
                (`game_type`이 걸린 지점이 정확히 여기다.)

실행:  .venv/Scripts/python.exe -u src/drift_audit.py
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

from config import TARGET_COL, TRAIN_CSV
from data import load_csv

MIN_BIN = 2000        # 표본이 이보다 적은 (시즌, bin)은 노이즈라 제외
N_QUANTILE = 8


def binned(s):
    """범주형은 그대로, 연속형은 분위 구간으로."""
    if s.dtype.name in ("category", "object") or s.nunique() <= 12:
        return s.astype("object").astype(str)
    q = pd.qcut(s, N_QUANTILE, duplicates="drop", labels=False)
    return q.astype("float").astype("object").astype(str)


def audit(df, col):
    b = binned(df[col])
    g = pd.DataFrame({"season": df.season.to_numpy(), "bin": b.to_numpy(),
                      "y": df[TARGET_COL].to_numpy(dtype=float)})
    tab = g.groupby(["season", "bin"], observed=True).y.agg(["mean", "size"])
    base = g.groupby("season").y.mean()
    tab["rel"] = tab["mean"] - tab.index.get_level_values("season").map(base)
    tab = tab[tab["size"] >= MIN_BIN]
    if len(tab) < 6:
        return None

    rows = []
    for bn, sub in tab.groupby("bin", observed=True):
        if len(sub) < 4:
            continue
        w = sub["size"].to_numpy(dtype=float)
        r = sub["rel"].to_numpy()
        mu = np.average(r, weights=w)
        sd = np.sqrt(np.average((r - mu) ** 2, weights=w))
        seasons = sub.index.get_level_values("season").to_numpy()
        early, late = seasons <= 2022, seasons >= 2023
        brk = (np.average(r[late], weights=w[late]) - np.average(r[early], weights=w[early])
               if early.any() and late.any() else np.nan)
        rows.append((bn, w.sum(), sd, brk))

    if not rows:
        return None
    R = pd.DataFrame(rows, columns=["bin", "n", "sd", "brk"])
    tot = R.n.sum()
    return {
        "col": col,
        "drift_std": float(np.average(R.sd, weights=R.n)),
        "break_23": float(np.average(np.abs(R.brk.fillna(0)), weights=R.n)),
        "worst_bin": R.loc[R.brk.abs().idxmax(), "bin"] if R.brk.notna().any() else "",
        "worst_brk": float(R.brk.abs().max()) if R.brk.notna().any() else np.nan,
        "coverage": float(tot / len(df)),
    }


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    skip = {"row_id", TARGET_COL, "season"}
    cols = [c for c in df.columns if c not in skip]

    out = [r for r in (audit(df, c) for c in cols) if r]
    R = pd.DataFrame(out).sort_values("break_23", ascending=False)

    print(f"컬럼 {len(R)}개 감사 완료  ({time.time()-t0:.0f}s)\n")
    print("=== 2022→2023 구조 변화가 큰 순 (game_type이 걸린 지점) ===")
    print(f"{'컬럼':<40s} {'break_23':>9s} {'drift_std':>10s} {'최악 bin':>10s} {'그 bin 변화':>11s}")
    print("-" * 86)
    for _, r in R.head(15).iterrows():
        print(f"{r.col:<40s} {r.break_23:>9.4f} {r.drift_std:>10.4f} "
              f"{str(r.worst_bin):>10s} {r.worst_brk:>11.4f}")

    print("\n=== 시즌 간 관계 불안정이 큰 순 ===")
    for _, r in R.sort_values("drift_std", ascending=False).head(10).iterrows():
        print(f"{r.col:<40s} drift_std={r.drift_std:.4f}  break_23={r.break_23:.4f}")

    R.to_csv(Path(__file__).parents[1] / "experiments" / "drift_audit.csv", index=False)
    print(f"\n※ break_23이 game_type 수준(0.1 이상)인 컬럼이 있으면 제거를 검토한다.")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
