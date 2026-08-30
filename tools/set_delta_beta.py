"""배포된 δ 축(`ctx` / `asof`)의 β 를 바꾸고 레벨을 재보정한다.

왜
    δ 축의 최적 β 는 **베이스에 묶여 있다.** `gap_reverse` 가 1115 스택에서 닫혔다가
    1132 스택에서 열려 LB +8.68 을 냈다 (§22-2). 같은 이유로 이미 배포된 축의 β 도
    베이스가 바뀌면 움직인다.

    `hj09` 베이스 재훑기 결과 (양 fold, 격자를 넓혀서):
        hand    배포 1.054  →  꼭짓점 β ≈ 3     2023 +27 / 2024 +50
        count4  배포 1.250  →  꼭짓점 β ≈ 5~6   2023 +68 / 2024 +111

    ⚠️ **꼭짓점으로 바로 가지 않는다.** `hand`·`count4` 는 투수×상황 셀에 붙는 δ 이고
       챔피언이 2023·2024 를 학습했다 — β 를 키우면 그 셀의 **암기를 더 강하게 재생**
       하므로 홀드아웃이 원리적으로 부풀린다. 한 걸음만 옮겨 **배율을 실측**한 뒤
       다음 점을 정한다 (라운드 20 라우팅에서 통한 방식).

⚠️ β 는 배포 함수 위에서 고른 값이어야 한다 (§22-4, 원우 §7.216 의 단위 사고).
   이 도구는 `artifacts.joblib` 의 값을 직접 고치므로 눈금이 자동으로 맞다.
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

import joblib
import numpy as np
import pandas as pd

TRAIN_CSV = "/Users/hyejin/Aimers_9th/data/train.csv"
SHIFT_RE = re.compile(r"(PROB_SHIFT\s*=\s*)([0-9.eE+-]+)")


def predict_sub(w, pkl, out):
    code = ("import sys, numpy as np, pandas as pd\n"
            f"sys.path.insert(0, {w!r})\n"
            "from predict import Predictor\n"
            f"df = pd.read_pickle({pkl!r})\n"
            f"p = Predictor(model_dir={os.path.join(w,'model')!r})\n"
            f"np.save({out!r}, np.asarray(p.predict(df), dtype=np.float64))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=w, capture_output=True, text=True,
                       env=dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="4"))
    if r.returncode:
        raise SystemExit(f"[중단] 추론 실패\n{r.stdout[-1500:]}\n{r.stderr[-2500:]}")
    return np.load(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", required=True,
                    metavar="GROUP:NAME=BETA", help="예: ctx:count4=1.75 (여러 번 가능)")
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="db_")
    w, o = os.path.join(tmp, "w"), os.path.join(tmp, "o")
    with zipfile.ZipFile(a.src) as z:
        names = z.namelist(); z.extractall(w); z.extractall(o)

    bp = os.path.join(w, "model", "artifacts.joblib")
    bundle = joblib.load(bp); art = bundle["artifacts"]
    print(f"소스 {os.path.basename(a.src)}")
    for spec in a.set:
        gn, bv = spec.split("=")
        grp, nm = gn.split(":")
        hit = [x for x in art[grp] if x["name"] == nm]
        if not hit:
            raise SystemExit(f"[중단] {grp}:{nm} 축이 없다. "
                             f"있는 것: {[x['name'] for x in art[grp]]}")
        old = float(hit[0]["beta"]); hit[0]["beta"] = float(bv)
        print(f"  {grp}:{nm}  β {old} -> {float(bv)}")
    joblib.dump(bundle, bp)

    proxy = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    proxy = proxy[proxy.season == 2024].drop(columns=["control_success"]).reset_index(drop=True)
    pk = os.path.join(tmp, "p.pkl"); proxy.to_pickle(pk)

    print("\n추론 ① 변경 전 …")
    before = predict_sub(o, pk, os.path.join(tmp, "b.npy"))
    print("추론 ② 변경 후 (레벨 미보정) …")
    mid = predict_sub(w, pk, os.path.join(tmp, "m.npy"))
    d = mid - before; dm = float(d.mean())
    print(f"  max|Δp| {np.abs(d).max():.6g} · 평균|Δp| {np.abs(d).mean():.6g} · Δmean {dm:+.6g}")

    pp = os.path.join(w, "predict.py")
    s0 = open(pp, encoding="utf-8").read()
    old_s = float(SHIFT_RE.findall(s0)[0][1]); shift = old_s - dm
    print(f"  PROB_SHIFT {old_s:.9f} -> {shift:.9f}")
    open(pp, "w", encoding="utf-8").write(SHIFT_RE.sub(lambda m: f"{m.group(1)}{shift:.9f}", s0, count=1))
    print("추론 ③ 재보정 검증 …")
    after = predict_sub(w, pk, os.path.join(tmp, "a.npy"))
    print(f"  패치 후 Δmean {float((after-before).mean()):+.3g}")

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            fp = os.path.join(w, n)
            if os.path.isfile(fp):
                z.write(fp, n)
    diff = sorted(n for n in names if os.path.isfile(os.path.join(w, n))
                  and open(os.path.join(w, n), "rb").read() != open(os.path.join(o, n), "rb").read())
    print(f"\n변경 격리 {len(diff)}/{len(names)}: {diff}")
    if diff != ["model/artifacts.joblib", "predict.py"]:
        raise SystemExit("[중단] 예상 밖 변경")
    print(f"출력 {a.out}\n  SHA-256 {hashlib.sha256(open(a.out,'rb').read()).hexdigest()}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
