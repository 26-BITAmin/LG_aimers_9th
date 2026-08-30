"""hj03(챔피언) vs 우리 콜드스타트 빌드 — **같은 더미에서 레벨·예측 실측**.

발견: `predict.py` 차이가 `PROB_SHIFT` 한 줄인데 값이 크게 갈린다.
    sy02_c4  0.005482311   sy04_cold 0.005580239   sy05_coldrt 0.005554188
    hj03     0.000172669   ← 챔피언 (LB 1131.4503)

레벨 항 비용은 K·δ², K = 1e5/(r(1−r)) ≈ 400,320 → δ=0.0054 면 ≈ 11.7 BSS.
단 hj03 은 emb4 MLP 를 **재학습**했으므로(전 배열 상이, corr 0.997) raw 평균이
다를 수 있다. **숫자만 비교하면 안 되고 같은 입력에서 최종 p 를 재야 한다.**

⚠️ `make_dummy_test` 는 `art` 없이 쓰면 평균이 +0.0044 위로 편향된다 (§7.121).
   artifacts.joblib 은 네 zip 이 **바이트 동일**이므로 더미 하나를 공유한다.

실행: .venv/Scripts/python.exe -u src/coldaudit.py 2>&1 | tee experiments/coldaudit.log
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

import joblib
import numpy as np
import pandas as pd

from build_submit import make_dummy_test
from config import ROOT

WORK = Path(os.environ.get("COLD_WORK", ROOT / "experiments" / "coldaudit"))
ZIPS = ["hj03_dim8_cold.zip", "sy02_c4.zip", "sy04_cold.zip", "sy05_coldrt.zip"]
N = 245_789


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    # ── 더미 1개를 만들어 전부 공유 (artifacts 바이트 동일 확인 후)
    art_dir = WORK / "_art"
    art_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT / ZIPS[0]) as z:
        z.extract("model/artifacts.joblib", art_dir)
    art = joblib.load(art_dir / "model" / "artifacts.joblib")["artifacts"]
    t0 = time.time()
    dummy = make_dummy_test(N, art=art)
    print(f"더미 {len(dummy):,}행 생성 ({time.time()-t0:.0f}s)")

    dd = WORK / "data"
    dd.mkdir(exist_ok=True)
    dummy.to_csv(dd / "test.csv", index=False)
    pd.DataFrame({"row_id": dummy["row_id"], "control_success": 0.5}).to_csv(
        dd / "sample_submission.csv", index=False)

    # ── 콜드스타트 노출률: emb0 0번 슬롯에 떨어지는 행 비율
    pid = dummy["pitcher_id"].to_numpy()
    known = set(np.asarray(art["mlp_id_maps"]["pitcher_id"]).tolist()) \
        if "mlp_id_maps" in art else None
    if known is not None:
        cold = float(np.mean([p not in known for p in pid]))
        print(f"더미의 미등록 투수 행 비율 {cold:.4%}")

    res = {}
    for zp in ZIPS:
        d = WORK / zp.replace(".zip", "")
        if not (d / "script.py").exists():
            d.mkdir(exist_ok=True)
            with zipfile.ZipFile(ROOT / zp) as z:
                z.extractall(d)
        (d / "data").mkdir(exist_ok=True)
        for f in ("test.csv", "sample_submission.csv"):
            tgt = d / "data" / f
            if not tgt.exists():
                tgt.write_bytes((dd / f).read_bytes())
        (d / "output").mkdir(exist_ok=True)
        t = time.time()
        r = subprocess.run([sys.executable, "script.py"], cwd=d,
                           capture_output=True, text=True)
        el = time.time() - t
        out = d / "output" / "submission.csv"
        if not out.exists():
            print(f"  ❌ {zp}: {r.stderr.strip()[-400:]}")
            continue
        p = pd.read_csv(out)["control_success"].to_numpy(dtype=np.float64)
        res[zp] = p
        print(f"  {zp:<22} mean {p.mean():.9f}  sd {p.std():.6f}  "
              f"min {p.min():.4f} max {p.max():.4f}  ({el:.0f}s)")

    if "hj03_dim8_cold.zip" not in res:
        return
    h = res["hj03_dim8_cold.zip"]
    K = 1e5 / (0.4861 * (1 - 0.4861))
    print("\n" + "=" * 78)
    print("챔피언(hj03) 대비  — 레벨 비용은 K·(Δmean)², K = %.0f" % K)
    print("=" * 78)
    print(f"  {'빌드':<22}{'Δmean':>13}{'레벨 비용(BSS)':>16}{'corr(p)':>10}{'mean|Δp|':>11}")
    for zp, p in res.items():
        if zp == "hj03_dim8_cold.zip":
            continue
        dm = p.mean() - h.mean()
        print(f"  {zp:<22}{dm:>+13.6f}{-K*dm**2:>16.2f}"
              f"{np.corrcoef(p, h)[0,1]:>10.6f}{np.abs(p-h).mean():>11.6f}")


if __name__ == "__main__":
    main()
