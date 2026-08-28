"""재학습한 MLP `emb4` 멤버를 챔피언 zip 에 갈아 끼우고 레벨을 재보정한다.

레시피는 원우 `t1cap.py` 와 같다 (§7.182):
    · 교체는 해당 멤버 파일만. 나머지는 챔피언 원본 유지 → LB 변화 = 그 성분분
    · T1 계보는 `logit_delta` 가 동결이라 성분이 바뀌면 레벨이 깨진다.
      2024 전 행 프록시로 평균 이동을 재고 `PROB_SHIFT` 를 (원래값 − Δmean) 으로 패치
    · 챔피언 shift 는 하드코딩하지 않고 zip 의 predict.py 에서 파싱한다
"""
import argparse
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

TRAIN_CSV = "/Users/hyejin/Aimers_9th/data/train.csv"
PROXY_SEASON = 2024
import re
SHIFT_RE = re.compile(r"(PROB_SHIFT\s*=\s*)([0-9.eE+-]+)")


def predict_sub(workdir, proxy, out):
    code = ("import sys, numpy as np, pandas as pd\n"
            f"sys.path.insert(0, {workdir!r})\n"
            "from predict import Predictor\n"
            f"df = pd.read_pickle({proxy!r})\n"
            f"p = Predictor(model_dir={os.path.join(workdir,'model')!r})\n"
            f"np.save({out!r}, np.asarray(p.predict(df), dtype=np.float64))\n")
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="4")
    r = subprocess.run([sys.executable, "-c", code], cwd=workdir, env=env,
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"[중단] 추론 실패\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return np.load(out)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="mlpzip_")
    work, orig = os.path.join(tmp, "w"), os.path.join(tmp, "o")
    with zipfile.ZipFile(a.src) as z:
        names = z.namelist(); z.extractall(work); z.extractall(orig)

    ppath = os.path.join(work, "predict.py")
    src0 = open(ppath, encoding="utf-8").read()
    m = SHIFT_RE.findall(src0)
    assert len(m) == 1, f"PROB_SHIFT 패턴이 {len(m)}개"
    shift0 = float(m[0][1])
    print(f"소스 {os.path.basename(a.src)}   PROB_SHIFT {shift0:.9f}")

    proxy = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    proxy = proxy[proxy["season"] == PROXY_SEASON].drop(columns=["control_success"]).reset_index(drop=True)
    ppk = os.path.join(tmp, "p.pkl"); proxy.to_pickle(ppk)
    print(f"프록시 {PROXY_SEASON} {len(proxy):,}행")

    print("추론 ① 교체 전 …")
    before = predict_sub(work, ppk, os.path.join(tmp, "b.npy"))

    # ── 멤버 교체 ────────────────────────────────────────────
    new = sorted(glob.glob(os.path.join(a.npz_dir, "mlp_emb4_*.npz")))
    assert new, f"npz 가 없다: {a.npz_dir}"
    for f in new:
        dst = os.path.join(work, "model", os.path.basename(f))
        assert os.path.exists(dst), f"챔피언에 없는 멤버다: {os.path.basename(f)}"
        shutil.copyfile(f, dst)
    shp = np.load(new[0])["emb0"].shape
    print(f"교체 {len(new)}개 · pitcher 임베딩 {shp}")

    print("추론 ② 교체 후 (레벨 미보정) …")
    mid = predict_sub(work, ppk, os.path.join(tmp, "m.npy"))
    d = mid - before
    dmean = float(d.mean())
    shift = shift0 - dmean
    print(f"  max|Δp| {np.abs(d).max():.6g} · 평균|Δp| {np.abs(d).mean():.6g} · Δmean {dmean:+.6g}")
    print(f"  PROB_SHIFT {shift0:.9f} -> {shift:.9f}")

    open(ppath, "w", encoding="utf-8").write(SHIFT_RE.sub(
        lambda mm: f"{mm.group(1)}{shift:.9f}", src0))
    print("추론 ③ 재보정 검증 …")
    after = predict_sub(work, ppk, os.path.join(tmp, "a.npy"))
    resid = float((after - before).mean())
    print(f"  패치 후 Δmean {resid:+.3g}  " + ("OK" if abs(resid) < 1e-8 else "⚠️ 1e-8 초과"))

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            fp = os.path.join(work, n)
            if os.path.isfile(fp):
                z.write(fp, n)

    diff = sorted(n for n in names if os.path.isfile(os.path.join(work, n))
                  and open(os.path.join(work, n), "rb").read() != open(os.path.join(orig, n), "rb").read())
    exp = sorted(["predict.py"] + [f"model/{os.path.basename(f)}" for f in new])
    print(f"\n변경 격리 {len(diff)}/{len(names)}: {diff}")
    if diff != exp:
        raise SystemExit(f"[중단] 예상과 다르다. 기대 {exp}")
    print(f"출력 {a.out}\n  SHA-256 {sha(a.out)}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
