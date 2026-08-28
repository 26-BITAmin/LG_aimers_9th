"""미학습 투수 슬롯(`index 0`)을 학습된 임베딩의 평균으로 바꾼다.

근거 (§17-7, 15시드 마스킹 실험)
    임베딩 테이블은 투수 792명 + **미지값 1칸**이고, LUT 를 학습 데이터로 적합했으므로
    `index 0` 에는 학습 중 어떤 행도 매핑되지 않는다 — 사실상 초기값이다.
    그런데 매 시즌 행의 14~22% 가 그 전에 없던 투수다 (5년 평균 18.5%).

        우리 dim8    +9.33  SE 0.49  t 18.94  15/15
        우리 dim12   +2.56  SE 0.91  t  2.81  10/15

⚠️ 레벨 재보정에 왜 마스킹 프록시를 쓰는가
    2024 프록시에는 **미지 투수가 0명**이다 (학습에 2024 가 들어갔다). 그대로 재면
    Δmean = 0 이 나와 재보정이 무의미해진다. 그래서 투수의 일부를 골라 `pitcher_id`
    를 LUT 밖 값으로 바꿔 **실제 test 의 미지값 비율(~20%)을 흉내낸** 뒤 잰다.
    바꾸는 것은 프록시 입력뿐이고 산출물에는 들어가지 않는다.
"""
import argparse
import glob
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
SENTINEL = -1          # LUT 에 없는 값 → encode_apply 가 index 0 으로 보낸다


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
        raise SystemExit(f"[중단] 추론 실패\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return np.load(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--unknown-frac", type=float, default=0.20,
                    help="프록시에서 미지 처리할 행 비율 (실제 test 추정 ~0.20)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="cold_")
    w, o = os.path.join(tmp, "w"), os.path.join(tmp, "o")
    with zipfile.ZipFile(a.src) as z:
        names = z.namelist(); z.extractall(w); z.extractall(o)

    pp = os.path.join(w, "predict.py")
    src0 = open(pp, encoding="utf-8").read()
    m = SHIFT_RE.findall(src0)
    assert len(m) == 1
    shift0 = float(m[0][1])
    print(f"소스 {os.path.basename(a.src)}   PROB_SHIFT {shift0:.9f}")

    # ── 마스킹 프록시: 투수 단위로 골라 목표 행 비율을 맞춘다 ──────
    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    df = df[df["season"] == 2024].drop(columns=["control_success"]).reset_index(drop=True)
    rng = np.random.RandomState(a.seed)
    pids = rng.permutation(df["pitcher_id"].unique())
    cnt = df["pitcher_id"].value_counts()
    take, tot, goal = [], 0, a.unknown_frac * len(df)
    for pid in pids:
        if tot >= goal:
            break
        take.append(pid); tot += cnt[pid]
    mask = df["pitcher_id"].isin(take).to_numpy()
    df.loc[mask, "pitcher_id"] = SENTINEL
    pk = os.path.join(tmp, "p.pkl"); df.to_pickle(pk)
    print(f"프록시 2024 {len(df):,}행 · 미지 처리 {mask.sum():,} ({mask.mean():.2%}, 투수 {len(take)}명)")

    print("추론 ① 패치 전 …")
    before = predict_sub(w, pk, os.path.join(tmp, "b.npy"))

    # ── 패치: index 0 ← 학습된 임베딩의 평균 ─────────────────────
    files = sorted(glob.glob(os.path.join(w, "model", "mlp_emb4_*.npz")))
    assert files, "mlp_emb4_*.npz 가 없다"
    for f in files:
        d = dict(np.load(f))
        e = d["emb0"]
        assert [str(c) for c in d["ecols"]][0] == "pitcher_id", "emb0 가 pitcher_id 가 아니다"
        d["emb0"] = np.vstack([e[1:].mean(0, keepdims=True), e[1:]]).astype(e.dtype)
        np.savez(f, **d)
    print(f"패치 {len(files)}개 · emb0[0] ← mean(emb0[1:])")

    print("추론 ② 패치 후 (레벨 미보정) …")
    mid = predict_sub(w, pk, os.path.join(tmp, "m.npy"))
    d = mid - before
    dm = float(d.mean())
    dm_unknown = float(d[mask].mean())
    print(f"  미지 행 Δp 평균 {dm_unknown:+.6g} · 알려진 행 {float(d[~mask].mean()):+.3g}")
    print(f"  전체 Δmean {dm:+.6g}  (= 미지 행 몫 × {mask.mean():.2%})")
    shift = shift0 - dm
    print(f"  PROB_SHIFT {shift0:.9f} -> {shift:.9f}")
    open(pp, "w", encoding="utf-8").write(SHIFT_RE.sub(lambda mm: f"{mm.group(1)}{shift:.9f}", src0))

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
    exp = sorted(["predict.py"] + [f"model/{os.path.basename(f)}" for f in files])
    print(f"\n변경 격리 {len(diff)}/{len(names)}: {diff}")
    if diff != exp:
        raise SystemExit(f"[중단] 기대 {exp}")
    print(f"출력 {a.out}\n  SHA-256 {hashlib.sha256(open(a.out,'rb').read()).hexdigest()}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
