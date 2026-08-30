"""챔피언 zip 에 `asof` δ 축을 하나 추가한다 — 다른 것은 전부 그대로.

⚠️ β 의 눈금
    `features.asof_lookup` 은 **표준화된** 축에 β 를 곱한다:
        gap = (k_szn + S·prior)/(n_szn + S) − prior      v = (gap − mu)/sd
        z += β · v
    측정 스크립트가 원단위 gap 위에서 β 를 골랐다면 그 값을 그대로 적으면 안 된다
    (원우 §7.216: 원단위 −0.60 을 그대로 넣었으면 fold2024 −1954).
    **β 는 배포 함수 위에서 고른 값이어야 한다.**

⚠️ `k_off` 는 축마다 다르다
    `n_off`(시즌 시작 투구 수)는 공통이지만 `k_off`(그 모드의 누적 개수)는 `r_col`
    마다 다르다 — 실측 gap_success 2,740,198 · gap_ball 1,895,279 ·
    gap_reverse 1,116,134. **템플릿 복사는 축을 통째로 틀리게 만든다.**
    그래서 `asofdelta.fit_asof_axes` 로 새로 적합한다 (G0 재현 관문으로 검증).

⚠️ `strength` 는 config 가 아니라 **번들에서 읽는다** (챔피언은 35, config 주석은 200).
"""
import argparse
import copy
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
    ap.add_argument("--work", required=True, help="asofdelta.py 가 있는 디렉터리")
    ap.add_argument("--name", required=True)
    ap.add_argument("--id-col", default="pitcher_id")
    ap.add_argument("--n-col", default="asof_pitcher_n")
    ap.add_argument("--r-col", required=True)
    ap.add_argument("--beta", type=float, required=True)
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="ax_")
    w, o = os.path.join(tmp, "w"), os.path.join(tmp, "o")
    with zipfile.ZipFile(a.src) as z:
        names = z.namelist(); z.extractall(w); z.extractall(o)

    bp = os.path.join(w, "model", "artifacts.joblib")
    bundle = joblib.load(bp)
    art = bundle["artifacts"]
    S = float(art["asof"][0].get("strength", 35.0))
    print(f"소스 {os.path.basename(a.src)}")
    print(f"  기존 축: {[(x['name'], x['beta']) for x in art['asof']]}")
    print(f"  strength = {S}  (번들에서 읽음)")

    sys.path.insert(0, a.work)
    import asofdelta as AD
    AD.ASOF_SPEC[a.name] = dict(id=a.id_col, n_col=a.n_col, r_col=a.r_col)

    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")

    # ── G0 재현 관문 — 기존 축을 다시 만들어 대조한다 ─────────────
    have = [(x["name"], x["beta"]) for x in art["asof"] if x["name"] in AD.ASOF_SPEC]
    if have:
        redo = {x["name"]: x for x in AD.fit_asof_axes(df, have, strength=S, ct=art.get("contam"))}
        print("\nG0 재현 관문:")
        for nm, _ in have:
            d = next(x for x in art["asof"] if x["name"] == nm)
            r = redo[nm]
            ok = (len(r["keys"]) == len(d["keys"]) and np.allclose(r["k_off"], d["k_off"])
                  and abs(r["mu"] - d["mu"]) < 1e-8 and abs(r["sd"] - d["sd"]) < 1e-8)
            print(f"  {nm:<14} 재현 {'일치' if ok else '**불일치**'}")
            if not ok:
                raise SystemExit("[중단] 기존 축을 재현하지 못했다 — 제작 경로가 다르다")

    new = AD.fit_asof_axes(df, [(a.name, a.beta)], strength=S, ct=art.get("contam"))[0]
    print(f"\n새 축 {a.name}  β {a.beta}  keys {len(new['keys'])}  "
          f"mu {new['mu']:.9f}  sd {new['sd']:.9f}  cadj {new.get('cadj')}")
    art["asof"] = list(art["asof"]) + [new]
    joblib.dump(bundle, bp)

    proxy = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    proxy = proxy[proxy.season == 2024].drop(columns=["control_success"]).reset_index(drop=True)
    pk = os.path.join(tmp, "p.pkl"); proxy.to_pickle(pk)

    print("\n추론 ① 축 추가 전 …")
    before = predict_sub(o, pk, os.path.join(tmp, "b.npy"))
    print("추론 ② 축 추가 후 (레벨 미보정) …")
    mid = predict_sub(w, pk, os.path.join(tmp, "m.npy"))
    d = mid - before; dm = float(d.mean())
    print(f"  max|Δp| {np.abs(d).max():.6g} · 평균|Δp| {np.abs(d).mean():.6g} · Δmean {dm:+.6g}")

    pp = os.path.join(w, "predict.py")
    s0 = open(pp, encoding="utf-8").read()
    old = float(SHIFT_RE.findall(s0)[0][1]); shift = old - dm
    print(f"  PROB_SHIFT {old:.9f} -> {shift:.9f}")
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
