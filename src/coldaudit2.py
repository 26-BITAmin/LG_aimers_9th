"""hj03 vs 우리 콜드빌드 — **미지 투수 20% 마스킹** population 에서 비교 (§7.203 프록시와 동형).

§coldaudit(무마스킹 더미)의 결론: 최종 평균이 1.6e-4 안에서 일치 → `PROB_SHIFT`
0.0054 격차는 두 계보 raw 평균차의 상쇄일 뿐, 레벨 문제 없음.

⚠️ 그런데 그 더미는 **미등록 투수가 0%** 다 (2024 행이라 전부 LUT 안에 있다).
   콜드스타트 패치가 만지는 자리가 애초에 없으니 두 수리 방식을 구별할 수 없다.
   §7.203 이 쓴 것과 같은 마스킹(투수 단위, 목표 행비율 0.20, RandomState(0))을 걸어
   그 자리에서 hj03(재학습)과 sy04(행 하나 수술)가 같은 일을 하는지 본다.

마스킹은 `coldbuild.WORKER_MASK` 와 **같은 난수·같은 순서**로 재현한다.

실행: .venv/Scripts/python.exe -u src/coldaudit2.py 2>&1 | tee experiments/coldaudit2.log
"""
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

from config import ROOT

WORK = Path(os.environ.get("COLD_WORK", ROOT / "experiments" / "coldaudit"))
ZIPS = ["hj03_dim8_cold.zip", "sy02_c4.zip", "sy04_cold.zip", "sy05_coldrt.zip"]
UNK = 0.20


def main():
    src = WORK / "data" / "test.csv"
    assert src.exists(), "coldaudit.py 를 먼저 돌려 더미를 만들어야 한다"
    va = pd.read_csv(src)

    # ── coldbuild.WORKER_MASK 와 동일한 선택 절차
    rng = np.random.RandomState(0)
    pids = rng.permutation(np.sort(va["pitcher_id"].unique()))
    cnt = va["pitcher_id"].value_counts()
    take, tot, goal = [], 0, UNK * len(va)
    for pid in pids:
        if tot >= goal:
            break
        take.append(pid)
        tot += int(cnt[pid])
    mask = va["pitcher_id"].isin(take).to_numpy()
    va.loc[mask, "pitcher_id"] = -1
    print(f"마스킹 {mask.sum():,}행 ({mask.mean():.2%})  투수 {len(take)}명 / "
          f"{va['pitcher_id'].nunique()}")

    md = WORK / "masked"
    md.mkdir(exist_ok=True)
    va.to_csv(md / "test.csv", index=False)
    pd.DataFrame({"row_id": va["row_id"], "control_success": 0.5}).to_csv(
        md / "sample_submission.csv", index=False)

    res = {}
    for zp in ZIPS:
        d = WORK / zp.replace(".zip", "")
        for f in ("test.csv", "sample_submission.csv"):
            (d / "data" / f).write_bytes((md / f).read_bytes())
        (d / "output" / "submission.csv").unlink(missing_ok=True)
        t = time.time()
        r = subprocess.run([sys.executable, "script.py"], cwd=d,
                           capture_output=True, text=True)
        out = d / "output" / "submission.csv"
        if not out.exists():
            print(f"  ❌ {zp}: {r.stderr.strip()[-400:]}")
            continue
        p = pd.read_csv(out)["control_success"].to_numpy(dtype=np.float64)
        res[zp] = p
        print(f"  {zp:<22} 전체 {p.mean():.9f}  미지행 {p[mask].mean():.9f}  "
              f"기지행 {p[~mask].mean():.9f}  ({time.time()-t:.0f}s)")

    h = res.get("hj03_dim8_cold.zip")
    if h is None:
        return
    K = 1e5 / (0.4861 * (1 - 0.4861))
    print("\n" + "=" * 84)
    print("챔피언(hj03) 대비 — 마스킹 population")
    print("=" * 84)
    print(f"  {'빌드':<22}{'Δmean 전체':>13}{'레벨비용':>10}"
          f"{'Δmean 미지행':>14}{'mean|Δp| 미지':>14}{'mean|Δp| 기지':>14}{'corr':>9}")
    for zp, p in res.items():
        if zp == "hj03_dim8_cold.zip":
            continue
        dm = p.mean() - h.mean()
        print(f"  {zp:<22}{dm:>+13.6f}{-K*dm**2:>10.2f}"
              f"{p[mask].mean()-h[mask].mean():>+14.6f}"
              f"{np.abs(p-h)[mask].mean():>14.6f}"
              f"{np.abs(p-h)[~mask].mean():>14.6f}"
              f"{np.corrcoef(p, h)[0,1]:>9.5f}")
    print("\n  ※ 기지행 |Δp| 는 콜드스타트와 무관한 상시 차이(hj03 의 MLP 재학습분).")
    print("     미지행 |Δp| 가 그보다 크면 두 수리 방식이 콜드 자리에서 실제로 갈린 것이다.")


if __name__ == "__main__":
    main()
