"""임의 zip 의 `PROB_SHIFT` 를 기준 zip 의 예측 평균에 맞춘다.

왜 필요한가
    잔차 층·보정 층을 얹으면 예측 평균이 움직인다. T1 계보는 `logit_delta` 가 동결이라
    그 이동을 되돌리지 않으면 레벨 축이 통째로 어긋난다 — 항등식으로 환산하면
    `Δmean −0.0056` 이 **−35점**, `−0.0122` 가 **−110점**이다.

    우리 빌더(`build_*`)는 변경을 만들면서 재보정하지만, **이미 만들어진 남의 zip**
    에는 쓸 수 없다. 이 도구는 그 경우를 다룬다.

방법
    두 zip 을 2024 프록시로 돌려 평균 차를 재고, 후보의 `PROB_SHIFT` 에 그만큼 더한다.
    바꾸는 것은 `predict.py` 의 상수 하나뿐이다.
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

TRAIN_CSV = "/Users/hyejin/Aimers_9th/data/train.csv"
SHIFT_RE = re.compile(r"(PROB_SHIFT\s*=\s*)([0-9.eE+-]+)")


def run(d, pkl, out):
    code = ("import sys, numpy as np, pandas as pd\n"
            f"sys.path.insert(0, {d!r})\n"
            "from predict import Predictor\n"
            f"df = pd.read_pickle({pkl!r})\n"
            f"p = Predictor(model_dir={os.path.join(d,'model')!r})\n"
            f"np.save({out!r}, np.asarray(p.predict(df), dtype=np.float64))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=d, capture_output=True, text=True,
                       env=dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="4"))
    if r.returncode:
        raise SystemExit(f"[중단] 추론 실패\n{r.stdout[-1500:]}\n{r.stderr[-2000:]}")
    return np.load(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="레벨 기준 zip (LB 를 아는 것)")
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="rl_")
    rd, cd = os.path.join(tmp, "r"), os.path.join(tmp, "c")
    for z, d in ((a.ref, rd), (a.cand, cd)):
        with zipfile.ZipFile(z) as f:
            names = f.namelist() if d == cd else None
            f.extractall(d)
    with zipfile.ZipFile(a.cand) as f:
        names = f.namelist()

    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    df = df[df.season == 2024].drop(columns=["control_success"]).reset_index(drop=True)
    pkl = os.path.join(tmp, "p.pkl"); df.to_pickle(pkl)
    print(f"프록시 2024 {len(df):,}행")

    pr = run(rd, pkl, os.path.join(tmp, "r.npy"))
    pc = run(cd, pkl, os.path.join(tmp, "c.npy"))
    gap = float(pr.mean() - pc.mean())
    print(f"  기준 p̄ {pr.mean():.6f}   후보 p̄ {pc.mean():.6f}   차 {gap:+.6g}")

    pp = os.path.join(cd, "predict.py")
    s0 = open(pp, encoding="utf-8").read()
    old = float(SHIFT_RE.findall(s0)[0][1])
    new = old + gap
    print(f"  PROB_SHIFT {old:.9f} -> {new:.9f}")
    open(pp, "w", encoding="utf-8").write(
        SHIFT_RE.sub(lambda m: f"{m.group(1)}{new:.9f}", s0, count=1))

    pc2 = run(cd, pkl, os.path.join(tmp, "c2.npy"))
    print(f"  재보정 후 p̄ {pc2.mean():.6f}   잔차 {float(pc2.mean()-pr.mean()):+.3g}")
    print(f"  std  기준 {pr.std():.6f}  후보 {pc2.std():.6f}  "
          f"({100*(pc2.std()/pr.std()-1):+.1f}%)   ← 레벨로는 못 고치는 부분")

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            fp = os.path.join(cd, n)
            if os.path.isfile(fp):
                z.write(fp, n)
    print(f"\n출력 {a.out}\n  SHA-256 {hashlib.sha256(open(a.out,'rb').read()).hexdigest()}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
