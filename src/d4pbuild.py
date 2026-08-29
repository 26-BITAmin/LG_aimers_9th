"""§7.217c — `d4p`(depth 4 · plain) 를 챔피언 MLP 풀에 **40% 로 편입**한다.

base: `hj05_r047f330.zip` (사용자 지정 챔피언, LB 1132.2409729548)

■ 근거 (§7.217 ⑤)
β 스윕 생존자 3/17 중 하나이고, **유일하게 내부 최적점**을 갖는다.

    fold2024  β .10 .20 .30 **.40** .50  →  +0.61 +1.03 +1.27 **+1.33** +1.20
    fold2023                              →  +0.67 +1.18 +1.52  +1.71  +1.73

양 fold × 전 β 구간 양수. 단독으로는 f2024 −0.94 인데 **섞으면** 이긴다
([[ensembling-reduces-bias-not-just-variance]]).

■ 왜 이 팔만 배포 가능한가
`export_weights` 가 막던 것은 ① 층 인덱스 하드코딩 ② `Res` 경로 부재였다.
`d2r`·`w64r4` 는 잔차 팔이라 ②까지 뚫어야 하는데 **`d4p` 는 plain 이라 ①만** 뚫으면 된다.
§7.217b 에서 둘 다 일반화하고 하위 호환 관문을 통과시켰다:
zip 의 `mlp_infer.py` 만 새 것으로 갈아끼워 챔피언을 돌린 결과 **`max|Δp| = 0.000e+00`**.

■ 편입 방식 — 멤버를 **더한다** (교체가 아니다)
`mlp_weights` 는 멤버별 가중이므로 β 를 가중으로 표현할 수 있다.

    emb4   (혜진, depth2) 5개 × 0.175·0.6 = 0.105      합 0.525
    emb4d4 (우리, depth4) 5개 × 0.175·0.4 = 0.070      합 0.350
    noid   (혜진)         3개 × 0.0416667            합 0.125   → 총합 1.000

emb4 부분의 40% 가 `d4p` 가 된다 = 하네스의 β=0.40 과 같은 비율.

■ ⚠️ 정직하게 — 이식은 측정한 그 물건이 아니다
하네스의 60% 는 **우리** fold 학습 `d2p` 였는데 여기서는 **혜진** 멤버다 (§7.218:
우리 `train.py` 는 챔피언 MLP 를 재현 못 한다, 132.19%). 즉 계보가 섞인다.
방향은 오히려 강해질 수 있다 — 깊이 다양성에 계보 다양성이 더해지므로 —
but **크기는 전이를 보장하지 않는다.** 계보 비용만 따로 읽고 싶으면 이미 만들어 둔
`hj10_ourmlp4.zip`(전부 우리 d2p) 가 그 대조군이다.

■ 관문
  G1 numpy-torch 대조: 학습 중 `train.py::_fit_one` 이 멤버마다 찍는다 (임계 1e-4)
  G2 하위 호환: §7.217b `src/inferdepth.py` 통과 (max|Δp| = 0)
  G3 shape: `W0..W4` (은닉 4 + 헤드 1) · `W0` 행 = 71 + 32 = 103
  G4 가중 합 = 1.0 · 멤버 수 = 가중 수
  G5 `verify_zip` 6관문 + `metafix` 일치

실행
    .venv/Scripts/python.exe -u src/d4pbuild.py --stage train
    .venv/Scripts/python.exe -u src/d4pbuild.py --stage build
"""
import argparse
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASCII = Path("D:/t1eval")
WORK = ASCII / "D4P"
BASE_ZIP = ROOT / "hj05_r047f330.zip"
OUT_ZIP = ROOT / "hj12_d4p.zip"
MEMDIR = ROOT / "experiments" / "d4p_members"
SEEDS = ("42", "7", "2024", "1234", "31337")
VAR = "emb4d4"                       # 새 변형 이름 (`_variant_of` 는 마지막 _ 로 자른다)
FILES = [f"mlp_{VAR}_{s}.npz" for s in SEEDS]
BETA = 0.40                          # §7.217 ⑤ fold2024 내부 최적점
P_DIM, DEPTH, HID, EPOCHS = 8, 4, 128, 9

_ap = argparse.ArgumentParser()
_ap.add_argument("--stage", required=True, choices=("train", "build"))
ARGS = _ap.parse_args()

os.environ["LGA_MODEL_DIR"] = str(WORK / "model")
os.environ["LGA_MLP_ONLY"] = "1"
os.environ["LGA_MLP_REPRO"] = "0"    # 새 변형이라 대조할 배포 npz 가 없다 (G1 이 대신한다)
os.environ["MLP_DEVICE"] = "cpu"     # §7.218 — 제출물은 CPU 정본으로만

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

NL = chr(10)
SNIPPET = NL.join([
    "import os, sys",
    "os.environ['LGA_MODEL_DIR'] = r'{md}'",
    "os.environ['LGA_MLP_ONLY'] = '1'",
    "os.environ['LGA_MLP_REPRO'] = '0'",
    "os.environ['MLP_DEVICE'] = 'cpu'",
    "sys.path.insert(0, r'{src}')",
    "import config",
    "c, n = config.MLP_VARIANTS['emb4']; c = dict(c)",
    "c['emb_dim'] = dict(c['emb_dim'], pitcher_id={p})",
    "c['depth'] = {d}; c['resid'] = False; c['hid'] = {h}; c['epochs'] = {e}",
    # 기존 변형은 전부 지운다 — 이 실행은 새 변형 5개만 만든다
    "config.MLP_VARIANTS = {{'{v}': (c, 5)}}",
    "import train",
    "cfg = train.MLP_VARIANTS['{v}'][0]",
    "assert (cfg['depth'], cfg['resid'], cfg['hid'], cfg['epochs'])"
    " == ({d}, False, {h}, {e}), cfg",
    "assert cfg['emb_dim']['pitcher_id'] == {p}",
    "train.main()",
])


def cold_patch(p):
    d = dict(np.load(p, allow_pickle=True))
    ec = [str(c) for c in d["ecols"]]
    done = []
    for col in ("pitcher_id", "batter_id"):
        k = f"emb{ec.index(col)}"
        e = d[k]
        d[k] = np.vstack([e[1:].mean(0, keepdims=True), e[1:]]).astype(e.dtype)
        done.append(f"{col} {np.linalg.norm(e[0]):.3f}->{np.linalg.norm(d[k][0]):.3f}")
    np.savez(p, **d)
    return done


def stage_train():
    import subprocess
    t0 = time.time()
    if not (WORK / "predict.py").exists():
        WORK.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(BASE_ZIP) as z:
            z.extractall(WORK)
    md = WORK / "model"
    todo = [f for f in FILES if not (md / f).exists()]
    print(f"§7.217c  d4p 멤버 학습 — depth {DEPTH} · plain · hid {HID} · ep {EPOCHS} · "
          f"pitcher emb {P_DIM}   대상 {len(todo)}/5")
    if todo:
        code = SNIPPET.format(md=md, src=Path(__file__).parent, p=P_DIM,
                              d=DEPTH, h=HID, e=EPOCHS, v=VAR)
        r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        assert r.returncode == 0, f"학습 실패 (rc {r.returncode})"
        print(f"[{time.time()-t0:.0f}s] 학습 완료")

    # ── G3 shape ────────────────────────────────────────────────────
    import joblib
    enc = joblib.load(md / "artifacts.joblib")["mlp_enc"]
    n_num, sizes = len(enc["num_cols"]), list(enc["sizes"])
    for f in FILES:
        with np.load(md / f) as z:
            nW = sum(1 for k in z.files if k.startswith("W"))
            ec = [str(c) for c in z["ecols"]]
            dm = [z[f"emb{i}"].shape[1] for i in range(len(ec))]
            rw = [z[f"emb{i}"].shape[0] for i in range(len(ec))]
            assert nW == DEPTH + 1, f"{f}: W 층 {nW}개 (기대 {DEPTH+1})"
            assert z["W0"].shape == (n_num + sum(dm), HID), f"{f} W0 {z['W0'].shape}"
            assert z[f"W{DEPTH}"].shape == (HID, 1), f"{f} 헤드 {z[f'W{DEPTH}'].shape}"
            assert rw == sizes, f"{f} 임베딩 행 {rw} != 인코더 {sizes}"
            assert dm == [P_DIM] + [4] * 6, f"{f} dims {dm}"
    print(f"  G3 통과 — W0..W{DEPTH} · W0 {(n_num + P_DIM + 24, HID)} · 인코더 {sizes}")

    print("  콜드 슬롯 접기 (emb0[0] · emb1[0])")
    for f in FILES:
        print(f"    {f:<24} " + " · ".join(cold_patch(md / f)))
    MEMDIR.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copy2(md / f, MEMDIR / f)
    print(f"  멤버 사본 -> {MEMDIR}   ({time.time()-t0:.0f}s)")


def stage_build():
    import tempfile

    import joblib
    sys.path.insert(0, str(ROOT / "tools_hj"))
    from rebase_routing import SHIFT_RE, TRAIN_CSV, predict_in_subprocess, sha

    import pandas as pd
    for f in FILES:
        assert (MEMDIR / f).exists(), f"{f} 없음 — --stage train 먼저"

    tmp = tempfile.mkdtemp(prefix="d4p_", dir=str(ASCII))
    work, orig = os.path.join(tmp, "work"), os.path.join(tmp, "orig")
    with zipfile.ZipFile(BASE_ZIP) as z:
        names = z.namelist()
        z.extractall(work)
        z.extractall(orig)

    pp = os.path.join(work, "predict.py")
    src0 = open(pp, encoding="utf-8").read()
    shift0 = float(SHIFT_RE.findall(src0)[0][1])
    ajp = os.path.join(work, "model", "artifacts.joblib")
    bundle = joblib.load(ajp)
    mm, mw = list(bundle["mlp_members"]), [float(v) for v in bundle["mlp_weights"]]
    print(f"베이스 {BASE_ZIP.name}   PROB_SHIFT {shift0:.9f}")
    print(f"  기존 풀 {len(mm)}멤버  가중합 {sum(mw):.6f}")

    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    df = df[df["season"] == 2024].drop(columns=["control_success"]).reset_index(drop=True)
    pq = os.path.join(tmp, "proxy.pkl")
    df.to_pickle(pq)
    print("추론 1/3 편입 전 (= 챔피언) ...")
    before = predict_in_subprocess(work, pq, os.path.join(tmp, "b.npy"))

    # ── 편입: mlp_infer 일반화 + 멤버 5개 추가 + 가중 재배분 ─────────
    shutil.copy2(ROOT / "src" / "mlp_infer.py", os.path.join(work, "mlp_infer.py"))
    for f in FILES:
        shutil.copy2(MEMDIR / f, os.path.join(work, "model", f))
    new_mm, new_mw = [], []
    for tag, w in zip(mm, mw):
        new_mm.append(tag)
        new_mw.append(w * (1 - BETA) if tag.startswith("emb4_") else w)
    ins = len([t for t in mm if t.startswith("emb4_")])
    base_w = [w for t, w in zip(mm, mw) if t.startswith("emb4_")]
    for k, s in enumerate(SEEDS):
        new_mm.insert(ins + k, f"{VAR}_{s}")
        new_mw.insert(ins + k, base_w[k] * BETA)
    assert abs(sum(new_mw) - 1.0) < 1e-12, f"가중합 {sum(new_mw)}"
    assert len(new_mm) == len(new_mw) == len(mm) + 5
    bundle["mlp_members"], bundle["mlp_weights"] = new_mm, new_mw
    joblib.dump(bundle, ajp)
    print(f"  G4 통과 — 풀 {len(new_mm)}멤버 · 가중합 {sum(new_mw):.12f}")
    print(f"    emb4(혜진 d2) {sum(w for t,w in zip(new_mm,new_mw) if t.startswith('emb4_')):.4f}"
          f" · {VAR}(우리 d4p) {sum(w for t,w in zip(new_mm,new_mw) if t.startswith(VAR)):.4f}"
          f" · noid {sum(w for t,w in zip(new_mm,new_mw) if t.startswith('noid')):.4f}")

    print("추론 2/3 편입 후 (레벨 미보정) ...")
    mid = predict_in_subprocess(work, pq, os.path.join(tmp, "m.npy"))
    d_ = mid - before
    dm = float(d_.mean())
    print(f"  corr(챔피언, 새것) {np.corrcoef(before, mid)[0,1]:.6f} · "
          f"평균|Δp| {np.abs(d_).mean():.6g} · max|Δp| {np.abs(d_).max():.6g}")
    print(f"  Var(p) {before.var():.8f} -> {mid.var():.8f} · Δmean {dm:+.9g}")
    assert np.abs(d_).max() > 0, "편입이 반영되지 않았다"
    shift = shift0 - dm
    open(pp, "w", encoding="utf-8").write(
        SHIFT_RE.sub(lambda m: f"{m.group(1)}{shift:.9f}", src0))
    print(f"  PROB_SHIFT {shift0:.9f} -> {shift:.9f}")
    print("추론 3/3 재보정 검증 ...")
    after = predict_in_subprocess(work, pq, os.path.join(tmp, "a.npy"))
    r = float((after - before).mean())
    print(f"  재보정 후 Δmean {r:+.3g}  {'OK' if abs(r) < 1e-9 else '경고'}")
    assert abs(r) < 1e-7

    allnames = list(names) + [f"model/{f}" for f in FILES]
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for n in allnames:
            fp = os.path.join(work, n)
            if os.path.isfile(fp):
                z.write(fp, n)
    diff = sorted(n for n in names if os.path.isfile(os.path.join(work, n))
                  and open(os.path.join(work, n), "rb").read()
                  != open(os.path.join(orig, n), "rb").read())
    exp = sorted(["predict.py", "mlp_infer.py", "model/artifacts.joblib"])
    print(f"\n변경 {len(diff)}/{len(names)}: {diff}")
    print(f"추가 {len(FILES)}: {FILES}")
    assert diff == exp, f"기대 {exp}"
    assert len(OUT_ZIP.name) <= 30
    print(f"출력 {OUT_ZIP}  ({OUT_ZIP.stat().st_size/1e6:.1f} MB, {len(allnames)}파일)")
    print(f"  SHA-256 {sha(OUT_ZIP)}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    {"train": stage_train, "build": stage_build}[ARGS.stage]()
