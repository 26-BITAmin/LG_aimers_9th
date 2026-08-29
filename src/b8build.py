"""§7.218 — 챔피언 위에서 `batter_id` 임베딩 **4 → 8** (대칭 축 회수) + 정직한 대조군.

base: `hj05_r047f330.zip` (사용자 지정 챔피언, LB 1123.7187284865)

■ 왜 이 축인가
챔피언은 `pitcher_id` 임베딩만 4→8 로 넓혔고 (§7.197 — **그 한 줄이 LB +7.44**),
`batter_id` 는 아직 **4** 다. 같은 표현 축인데 한쪽만 올라가 있다.
`hjswap.log` 의 `pitch +0.00 / bat +0.45` 도 같은 비대칭을 다른 채널에서 보여줬다.

■ 콜드 패치와 무엇이 다른가 — **입력이 실제로 바뀐다**
    콜드 패치   `emb1` 의 **0번 행 1개** 값만 바꾼다. 재학습 0회. 입력 차원 불변.
    폭 확대     `emb1` 이 (831,4) → (831,8). W0 이 (103,128) → **(107,128)**.
                → `mlp_emb4_*.npz` 5개를 **재학습**해야 한다.

■ ★★ 재현 관문 **실패** — 설계가 여기서 바뀌었다
우리 `train.py` 가 챔피언의 배포 emb4 를 만든 그 함수인가? 챔피언 dim 그대로
(pitcher 8 · batter 4) 로 `emb4_42` 를 다시 학습해 대조했다. 판정은 비트가 아니라
**예측 차이**다 (§7.167 사전 등록): `max|dz| / sd(z) < 1%`.

    CUDA (.venv-gpu, MLP_DEVICE=cuda)   max|dw| 2.087   max|dz|/sd = 159.27%
    CPU  (.venv, 정본 환경)             max|dw| 2.087   max|dz|/sd = **132.19%**

**디바이스 문제가 아니다.** shape 는 전부 맞는다(emb 행수 793/831 포함) — 그런데
가중치가 초기화부터 다른 스트림이다. §7.206 이 이미 적은 *"hj03 은 우리 빌드의
패치판이 아니다"* 와 같은 사실이고, §7.202 가 통과시킨 0.00% 관문은 `t1work/model` 의
**dim 4** 멤버 대상이었지 챔피언의 dim 8 멤버가 아니었다.

→ emb4 를 재학습하면 변경이 **둘**이 된다: ① 파이프라인 교체(혜진 → 우리)
  ② `batter_id` dim. 대조군 없이 ②만 지으면 교락돼서 아무것도 못 읽는다.
  [[anchor-proposals-need-an-honest-anchor-control]]

■ 그래서 대조군을 같이 짓는다
    `hj10_ourmlp4.zip`  우리 학습 emb4, pitcher 8 / batter **4**  <- 파이프라인 교체만
    `hj11_ourmlp8.zip`  우리 학습 emb4, pitcher 8 / batter **8**  <- 교체 + dim
  `hj11 - hj10` = batter dim 축의 순효과 · `hj10 - 챔피언` = 파이프라인 교체 비용.

■ 콜드 슬롯
새로 학습한 멤버는 `emb0[0]`·`emb1[0]` 이 **둘 다 미학습**이다 (챔피언은 투수 쪽만
패치돼 있다). **두 팔 모두** 둘 다 접는다 — 대조군과 후보가 같은 처리를 받아야
차이가 dim 축만 남는다. §7.208 이 확인한 결함이고, 폭이 넓어질수록 미학습 슬롯의
구멍도 커진다 (§7.202 의 dim8 vs dim12 관찰).

■ ⚠️ 디바이스 — 제출 아티팩트는 **CPU `.venv` 로만** 만든다
`mlp.py:71`: CUDA 는 RNG·누산 순서가 달라 같은 시드로도 다른 가중치를 낸다.
GPU 는 "같은 실행 안의 A/B" 에만 쓴다. §7.202 가 `.venv-gpu` 로 실행하면서도
`MLP_DEVICE` 를 안 켠 이유가 이것이다 (인터프리터가 GPU 인 것과 학습이 GPU 인 것은
다른 얘기다).

■ ⚠️ 판정 한계 — 미리 적는다
§7.198·§7.199: MLP 층 재학습에 붙는 SD(dBSS) 는 4.8~7.1 이다. 이 빌드는 **측정이지
판정이 아니다.** 그리고 이 저장소에는 챔피언 층 홀드아웃이 없다 (§7.207).

실행
    .venv/Scripts/python.exe -u src/b8build.py --stage repro
    .venv/Scripts/python.exe -u src/b8build.py --stage train
    .venv/Scripts/python.exe -u src/b8build.py --stage assemble
"""
import argparse
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASCII = Path("D:/t1eval")                 # LightGBM 이 비ASCII 경로에서 죽는다
BASE_ZIP = ROOT / "hj05_r047f330.zip"
SEEDS = ("42", "7", "2024", "1234", "31337")
EMB4 = [f"mlp_emb4_{s}.npz" for s in SEEDS]
P_DIM, B_DIM_OLD, B_DIM_NEW = 8, 4, 8
MEMDIR = ROOT / "experiments" / "b8_members"
# (출력 zip, 베이스 zip, 멤버 dim) — dim4 는 **정직한 앵커 대조군**이다
OUTS = [("hj10_ourmlp4.zip", BASE_ZIP, B_DIM_OLD),
        ("hj11_ourmlp8.zip", BASE_ZIP, B_DIM_NEW)]

_ap = argparse.ArgumentParser()
_ap.add_argument("--stage", required=True, choices=("repro", "train", "assemble"))
ARGS = _ap.parse_args()

# ⚠️ config 를 import 하기 **전에** 세팅해야 MODEL_DIR / 디바이스가 잡힌다.
os.environ.setdefault("LGA_MODEL_DIR", str(ASCII / "B8" / "model"))
os.environ["LGA_MLP_ONLY"] = "1"
os.environ["MLP_DEVICE"] = "cpu"
os.environ["LGA_MLP_REPRO"] = "1" if ARGS.stage == "repro" else "0"

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

NL = chr(10)


def unpack(dst):
    if not (dst / "predict.py").exists():
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(BASE_ZIP) as z:
            z.extractall(dst)
        print(f"챔피언 해제: {dst}")
    return dst


def dims_of(p):
    with np.load(p) as z:
        ec = [str(c) for c in z["ecols"]]
        return {c: z[f"emb{i}"].shape for i, c in enumerate(ec)}, z["W0"].shape


# ────────────────────────────────────────────────────── stage repro
def stage_repro():
    """챔피언 dim 그대로 emb4_42 를 다시 학습해 배포 npz 와 대조한다."""
    wd = unpack(ASCII / "B8")
    md = wd / "model"
    for f in EMB4:
        s, w0 = dims_of(md / f)
        assert s["pitcher_id"][1] == P_DIM and s["batter_id"][1] == B_DIM_OLD, \
            f"{f} 가 챔피언 dim 이 아니다: {s}"
    print(f"챔피언 emb4 5개 확인 — pitcher {P_DIM} · batter {B_DIM_OLD} · W0 {w0}")
    import config
    cfg, n = config.MLP_VARIANTS["emb4"]
    cfg = dict(cfg)
    cfg["emb_dim"] = dict(cfg["emb_dim"], pitcher_id=P_DIM, batter_id=B_DIM_OLD)
    config.MLP_VARIANTS["emb4"] = (cfg, n)
    import train
    e = train.MLP_VARIANTS["emb4"][0]["emb_dim"]
    assert (e["pitcher_id"], e["batter_id"]) == (P_DIM, B_DIM_OLD)
    print("재현 관문 실행 — 학습 대상 0개여야 한다 (5개 전부 디스크에 있다)")
    try:
        train.main()
    except SystemExit as ex:                 # 비트 불일치면 train.py 가 여기로 온다
        print(f"[재현 관문] {ex}")
        print("  위에 찍힌 `비 = x.xx%` 가 판정이다. 임계 1%.")
        return False
    print("[재현 관문] 전 텐서 비트 동일 — 통과")
    return True


# ────────────────────────────────────────────────────── stage train
def cold_patch_file(p):
    """`emb0[0]`(투수)·`emb1[0]`(타자) 를 학습된 나머지의 중심으로 접는다."""
    d = dict(np.load(p, allow_pickle=True))
    ec = [str(c) for c in d["ecols"]]
    done = []
    for col in ("pitcher_id", "batter_id"):
        k = f"emb{ec.index(col)}"
        e = d[k]
        if np.allclose(e[0], e[1:].mean(0)):
            done.append(f"{col}:이미")
            continue
        d[k] = np.vstack([e[1:].mean(0, keepdims=True), e[1:]]).astype(e.dtype)
        done.append(f"{col} {np.linalg.norm(e[0]):.3f}->"
                    f"{np.linalg.norm(d[k][0]):.3f}")
    np.savez(p, **d)
    return done


TRAIN_SNIPPET = NL.join([
    "import os, sys",
    "os.environ['LGA_MODEL_DIR'] = r'{md}'",
    "os.environ['LGA_MLP_ONLY'] = '1'",
    "os.environ['LGA_MLP_REPRO'] = '0'",
    "os.environ['MLP_DEVICE'] = 'cpu'",
    "sys.path.insert(0, r'{src}')",
    "import config",
    "c, n = config.MLP_VARIANTS['emb4']; c = dict(c)",
    "c['emb_dim'] = dict(c['emb_dim'], pitcher_id={p}, batter_id={b})",
    "config.MLP_VARIANTS['emb4'] = (c, n)",
    "import train",
    "e = train.MLP_VARIANTS['emb4'][0]['emb_dim']",
    "assert (e['pitcher_id'], e['batter_id']) == ({p}, {b}), e",
    "train.main()",
])


def train_dim(batter_dim, t0):
    """`B8_d{dim}` 에서 emb4 5멤버를 학습하고 콜드 슬롯을 접어 MEMDIR 로 복사한다."""
    import hashlib
    import subprocess

    import joblib

    out = MEMDIR / f"d{batter_dim}"
    if all((out / f).exists() for f in EMB4):
        print(f"[batter_id dim {batter_dim}]  MEMDIR 캐시 적중 — 건너뜀 ({out})")
        return
    wd = unpack(ASCII / f"B8_d{batter_dim}")
    md = wd / "model"
    # 🪤 **무조건 지운다.** 예전에는 `dim != batter_dim` 일 때만 지웠는데, dim4 팔에서는
    #    챔피언 멤버가 이미 dim4 라 그 조건이 안 걸려 **혜진 멤버가 대조군으로 들어갔다.**
    #    대조군의 존재 이유는 "우리 파이프라인" 을 고정하는 것이므로 반드시 재학습해야 한다.
    for f in EMB4:
        (md / f).unlink(missing_ok=True)
    todo = [f for f in EMB4 if not (md / f).exists()]
    print(f"[batter_id dim {batter_dim}]  학습 대상 {len(todo)}/5   ({wd})")
    if todo:
        # 하위 프로세스로 돌린다 — config/train 은 import 시점에 MODEL_DIR 을 굳힌다.
        code = TRAIN_SNIPPET.format(md=md, src=Path(__file__).parent,
                                    p=P_DIM, b=batter_dim)
        r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        assert r.returncode == 0, f"dim {batter_dim} 학습 실패 (rc {r.returncode})"
        print(f"[{time.time()-t0:.0f}s] dim {batter_dim} 학습 완료")

    enc = joblib.load(md / "artifacts.joblib")["mlp_enc"]
    n_num, sizes = len(enc["num_cols"]), list(enc["sizes"])
    md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()
    for f in EMB4:
        with np.load(md / f) as z:
            ec = [str(c) for c in z["ecols"]]
            dm = [z[f"emb{i}"].shape[1] for i in range(len(ec))]
            rw = [z[f"emb{i}"].shape[0] for i in range(len(ec))]
            assert dm == [P_DIM, batter_dim] + [4] * 5, f"{f} dims={dm}"
            assert rw == sizes, f"{f} 임베딩 행 {rw} != 인코더 {sizes}"
            assert z["W0"].shape[0] == n_num + sum(dm), f"{f} W0 {z['W0'].shape}"
            assert np.isfinite(z["W0"]).all() and np.isfinite(z["emb1"]).all()
    assert len({md5(md / f) for f in EMB4}) == 5, "멤버가 서로 같다 — 시드 미반영"
    print(f"  인코더 수치 {n_num} · 범주 {sizes} · W0 {dims_of(md / EMB4[0])[1]}")

    print("  콜드 슬롯 접기 (emb0[0] · emb1[0])")
    for f in EMB4:
        print(f"    {f:<22} " + " · ".join(cold_patch_file(md / f)))
    for f in EMB4:
        with np.load(md / f) as z:
            ec = [str(c) for c in z["ecols"]]
            for col in ("pitcher_id", "batter_id"):
                e = z[f"emb{ec.index(col)}"]
                assert np.allclose(e[0], e[1:].mean(0)), f"{f} {col} 패치 실패"
    out.mkdir(parents=True, exist_ok=True)
    for f in EMB4:
        shutil.copy2(md / f, out / f)
    print(f"  멤버 사본 -> {out}")


def stage_train():
    t0 = time.time()
    print("§7.218 — 우리 파이프라인 emb4 재학습 (dim4 대조군 + dim8 후보)")
    print("  ⚠️ 재현 관문 실패(132.19%) 이후 설계다 — 이 멤버들은 챔피언 멤버의")
    print("     '패치판' 이 아니라 다른 학습 산출물이다. 그래서 대조군을 같이 짓는다.")
    for d in (B_DIM_OLD, B_DIM_NEW):
        train_dim(d, t0)
    print(f"총 {time.time()-t0:.0f}s")


# ────────────────────────────────────────────────────── stage assemble
def stage_assemble():
    import tempfile

    sys.path.insert(0, str(ROOT / "tools_hj"))
    from rebase_routing import SHIFT_RE, TRAIN_CSV, predict_in_subprocess, sha

    import pandas as pd
    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    df = df[df["season"] == 2024].drop(columns=["control_success"]).reset_index(drop=True)
    P = {}

    for outname, base, bdim in OUTS:
        src_members = MEMDIR / f"d{bdim}"
        for f in EMB4:
            assert (src_members / f).exists(), f"{f} 없음 — --stage train 먼저"
        sh, w0 = dims_of(src_members / EMB4[0])
        print("=" * 92)
        print(f"{outname}   <- {Path(base).name}   멤버 dim {bdim}")
        print(f"  emb0{sh['pitcher_id']} emb1{sh['batter_id']} W0{w0}")
        print("=" * 92)

        tmp = tempfile.mkdtemp(prefix="b8_", dir=str(ASCII))
        work, orig = os.path.join(tmp, "work"), os.path.join(tmp, "orig")
        with zipfile.ZipFile(base) as z:
            names = z.namelist()
            z.extractall(work)
            z.extractall(orig)
        pp = os.path.join(work, "predict.py")
        src0 = open(pp, encoding="utf-8").read()
        m = SHIFT_RE.findall(src0)
        assert len(m) == 1
        shift0 = float(m[0][1])
        pq = os.path.join(tmp, "proxy.pkl")
        df.to_pickle(pq)

        print(f"  PROB_SHIFT {shift0:.9f}")
        print("  추론 1/3 교체 전 (= 챔피언) ...")
        before = predict_in_subprocess(work, pq, os.path.join(tmp, "b.npy"))
        for f in EMB4:
            shutil.copy2(src_members / f, os.path.join(work, "model", f))
        print("  추론 2/3 교체 후 (레벨 미보정) ...")
        mid = predict_in_subprocess(work, pq, os.path.join(tmp, "m.npy"))
        d_ = mid - before
        dm = float(d_.mean())
        print(f"  corr(챔피언, 새것) {np.corrcoef(before, mid)[0, 1]:.6f} · "
              f"평균|dp| {np.abs(d_).mean():.6g} · max|dp| {np.abs(d_).max():.6g}")
        print(f"  Var(p) {before.var():.8f} -> {mid.var():.8f} · dmean {dm:+.9g}")
        assert np.abs(d_).max() > 0, "교체가 반영되지 않았다"
        shift = shift0 - dm
        open(pp, "w", encoding="utf-8").write(
            SHIFT_RE.sub(lambda mm: f"{mm.group(1)}{shift:.9f}", src0))
        print(f"  PROB_SHIFT -> {shift:.9f}")
        print("  추론 3/3 재보정 검증 ...")
        after = predict_in_subprocess(work, pq, os.path.join(tmp, "a.npy"))
        r = float((after - before).mean())
        print(f"  재보정 후 dmean {r:+.3g}  {'OK' if abs(r) < 1e-9 else '경고'}")
        assert abs(r) < 1e-7
        P[outname] = after

        op = ROOT / outname
        with zipfile.ZipFile(op, "w", zipfile.ZIP_DEFLATED) as z:
            for n in names:
                fp = os.path.join(work, n)
                if os.path.isfile(fp):
                    z.write(fp, n)
        diff = sorted(n for n in names if os.path.isfile(os.path.join(work, n))
                      and open(os.path.join(work, n), "rb").read()
                      != open(os.path.join(orig, n), "rb").read())
        exp = sorted(["predict.py"] + [f"model/{f}" for f in EMB4])
        print(f"  변경 격리  {len(diff)}/{len(names)}: {[Path(x).name for x in diff]}")
        assert diff == exp, f"기대 {exp}"
        assert len(outname) <= 30
        print(f"  출력 {op}")
        print(f"  SHA-256 {sha(op)}")
        shutil.rmtree(tmp, ignore_errors=True)

    if len(P) == 2:
        a, b = [P[k] for k, _b, _d in OUTS]
        print("=" * 92)
        print("dim 축 순효과 (대조군 대비) — 2024 프록시 예측 차")
        print(f"  corr(ourmlp4, ourmlp8) {np.corrcoef(a, b)[0, 1]:.6f} · "
              f"평균|dp| {np.abs(b - a).mean():.6g} · max|dp| {np.abs(b - a).max():.6g}")
        print("  ⚠️ 이건 '얼마나 달라졌나' 이지 '얼마나 좋아졌나' 가 아니다. "
              "2024 는 in-sample 이라 점수로 못 읽는다 (§7.207).")


if __name__ == "__main__":
    {"repro": stage_repro, "train": stage_train, "assemble": stage_assemble}[ARGS.stage]()
