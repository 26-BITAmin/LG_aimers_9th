"""game_type × season 별 reverse 비율 — 두 추정량을 나란히.

    ① 복원 라벨 (modelabel.recover): 투구별 {0,1}. 그 행의 시즌·game_type에 귀속.
    ② asof 컬럼 평균: 커리어 누적이라 과거 오염이 뒤 시즌으로 번져 들어온다.

실행:  .venv/Scripts/python.exe -u src/revyear.py 2>&1 | tee experiments/revyear.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import modelabel
from config import TARGET_COL, TRAIN_CSV
from data import load_csv

MODES = ["reverse", "middle", "ball", "strike"]


def table(title, seasons, gts, get):
    print(f"\n{title}")
    print("  " + " " * 8 + "".join(f"{s:>9}" for s in seasons))
    for gt in gts:
        print(f"  {gt:<8}" + "".join(f"{get(gt, s):>9}" for s in seasons))


def main():
    df = load_csv(TRAIN_CSV)
    lab, _, _ = modelabel.recover(df)
    s = df["season"].to_numpy()
    gt = df["game_type"].astype(str).to_numpy()
    y = df[TARGET_COL].to_numpy(float)
    seasons = sorted(np.unique(s).tolist())
    gts = sorted(np.unique(gt).tolist())

    print(f"train {df.shape}   game_type {gts}")

    table("[행 수]", seasons, gts,
          lambda g, ss: f"{int(((gt == g) & (s == ss)).sum()):,}")
    table("[control_success 평균]", seasons, gts,
          lambda g, ss: f"{y[(gt == g) & (s == ss)].mean():.3f}")

    for m in MODES:
        v = np.round(lab[m])
        ok = ~np.isnan(v)

        def cell(g, ss, v=v, ok=ok):
            k = ok & (gt == g) & (s == ss)
            return f"{v[k].mean():.3f}" if k.sum() else "     ---"

        table(f"[① 복원 라벨 {m} — 그 해 실제 발생률]", seasons, gts, cell)

    for m in MODES:
        col = f"asof_pitcher_{m}_rate"
        c = df[col].to_numpy(float)
        ok = ~np.isnan(c)

        def cell(g, ss, c=c, ok=ok):
            k = ok & (gt == g) & (s == ss)
            return f"{c[k].mean():.3f}" if k.sum() else "     ---"

        table(f"[② {col} 평균 — 커리어 누적, 오염 이월분 포함]", seasons, gts, cell)

    # 복원 가능 비율 (증분이 잡히는 행)
    okr = ~np.isnan(lab["reverse"])
    table("[복원 가능 행 비율]", seasons, gts,
          lambda g, ss: f"{okr[(gt == g) & (s == ss)].mean():.3f}")


if __name__ == "__main__":
    main()
