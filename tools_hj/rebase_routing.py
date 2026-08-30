"""R/F 라우팅 계수를 임의의 제출 zip 위에 재적용한다 (재학습 없음).

왜 이 도구인가
    라우팅 축(원우 §7.177·§7.191)은 `predict.py` 두 줄이 전부다. 그런데 원우는
    s45 베이스로, 신영의 C-4(1127.1558)는 s45 베이스로 갈라져 나갔다. 두 이득은
    독립이므로 **같은 베이스 위에 합쳐야** max 채점에서 합산된다.

    ray (혜진 §15-9 방향, 원우 §7.177 파라미터화)
        R = 0.10 + t * 0.075        F = 0.40 + t * 0.10
        t =  0.00  T1 원본          R 0.100000   F 0.400000
        t = -0.25  s41r / s45       R 0.081250   F 0.375000
        t = -0.45  s47              R 0.066250   F 0.355000
        t = -0.70  s49 (미제출)     R 0.047500   F 0.330000

PROB_SHIFT 재보정
    라우팅을 바꾸면 예측 평균이 미세하게 움직인다. 레벨 항(확률공간 아핀)은
    목표 prior 에 맞춰 고정된 값이라 그 이동을 되돌려야 한다. 전체 test.csv 가
    없으므로 **train.csv 2024행(253,507행)을 프록시**로 Δmean 을 재고 그만큼 뺀다
    (원우 §7.191 과 같은 절차 · 같은 프록시).

행 독립성
    이 도구는 zip 안의 텍스트 두 줄과 상수 하나만 바꾼다. 모델 파일과
    features / mlp_infer 는 건드리지 않는다. 재패킹 후 predict.py 하나만
    다른지를 바이트로 확인한다.

사용
    python3 tools/rebase_routing.py --src sy02_c4.zip --t -0.45 --out sy03_r066f355.zip
    python3 tools/rebase_routing.py --src sy02_c4.zip --R 0.0475 --F 0.330 --out sy04.zip
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

TRAIN_CSV = "D:/projects/LG aimers 야구/data/train.csv"   # [로컴] 원우 환경
PROXY_SEASON = 2024

R_RE = re.compile(r"(z\[is_r\]\s*=\s*)([0-9.]+)(\s*\*\s*z\[is_r\]\s*\+\s*)([0-9.]+)(\s*\*\s*zg\[is_r\])")
F_RE = re.compile(r"(z\[is_f\]\s*=\s*)([0-9.]+)(\s*\*\s*z\[is_f\]\s*\+\s*)([0-9.]+)(\s*\*\s*zf\[is_f\])")
SHIFT_RE = re.compile(r"(PROB_SHIFT\s*=\s*)([0-9.eE+-]+)")


def ray(t):
    return 0.10 + t * 0.075, 0.40 + t * 0.10


def read_routing(src):
    """predict.py 에서 현재 (R, F, PROB_SHIFT) 를 읽는다. 각 패턴은 정확히 1개여야 한다."""
    mr, mf, ms = R_RE.findall(src), F_RE.findall(src), SHIFT_RE.findall(src)
    for name, m in (("z[is_r]", mr), ("z[is_f]", mf), ("PROB_SHIFT", ms)):
        if len(m) != 1:
            raise SystemExit(f"[중단] {name} 패턴이 {len(m)}개다 (1개여야 함). predict.py 형식을 확인할 것.")
    return float(mr[0][3]), float(mf[0][3]), float(ms[0][1])


def write_routing(src, R, F, shift):
    src = R_RE.sub(lambda m: f"{m.group(1)}{1.0 - R:.6f}{m.group(3)}{R:.6f}{m.group(5)}", src)
    src = F_RE.sub(lambda m: f"{m.group(1)}{1.0 - F:.6f}{m.group(3)}{F:.6f}{m.group(5)}", src)
    return SHIFT_RE.sub(lambda m: f"{m.group(1)}{shift:.9f}", src)


def load_proxy():
    df = pd.read_csv(TRAIN_CSV)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    df = df[df["season"] == PROXY_SEASON].reset_index(drop=True)
    return df.drop(columns=["control_success"], errors="ignore")


def predict_in_subprocess(workdir, proxy_path, out_npy):
    """예측기를 별도 프로세스에서 돌린다.

    같은 인터프리터에서 predict/features 를 두 번 import 하면 첫 번째 모듈이
    캐시돼 패치가 반영되지 않는다. 또 OpenMP 충돌(혜진 §15-8)도 프로세스
    분리로 피한다.
    """
    code = (
        "import sys, numpy as np, pandas as pd\n"
        f"sys.path.insert(0, {workdir!r})\n"
        "from predict import Predictor\n"
        f"df = pd.read_pickle({proxy_path!r})\n"
        f"p = Predictor(model_dir={os.path.join(workdir, 'model')!r})\n"
        "out = p.predict(df) if hasattr(p, 'predict') else p.predict_proba(df)\n"
        f"np.save({out_npy!r}, np.asarray(out, dtype=np.float64))\n"
    )
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="4")
    r = subprocess.run([sys.executable, "-c", code], cwd=workdir, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"[중단] 예측 실패\n--- stdout ---\n{r.stdout[-3000:]}\n--- stderr ---\n{r.stderr[-3000:]}")
    return np.load(out_npy)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t", type=float)
    ap.add_argument("--R", type=float)
    ap.add_argument("--F", type=float)
    ap.add_argument("--no-recal", action="store_true", help="PROB_SHIFT 재보정을 건너뛴다 (프록시 없이 계수만 바꿀 때)")
    a = ap.parse_args()

    if a.t is not None:
        R, F = ray(a.t)
    elif a.R is not None and a.F is not None:
        R, F = a.R, a.F
    else:
        raise SystemExit("--t 또는 (--R, --F) 중 하나가 필요하다")

    # [로컴] 기본 temp 경로에 한글이 들어 LightGBM 모델 로드가 깨진다 -> ASCII 경로 강제
    import os as _os; _os.makedirs("D:/t1eval", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="rebase_", dir="D:/t1eval")
    work, orig = os.path.join(tmp, "work"), os.path.join(tmp, "orig")
    with zipfile.ZipFile(a.src) as z:
        names = z.namelist()
        z.extractall(work)
        z.extractall(orig)

    ppath = os.path.join(work, "predict.py")
    if not os.path.exists(ppath):
        raise SystemExit(f"[중단] predict.py 가 zip 루트에 없다: {a.src}")
    src0 = open(ppath, encoding="utf-8").read()
    R0, F0, shift0 = read_routing(src0)

    print(f"소스   {os.path.basename(a.src)}")
    print(f"  현재  R {R0:.6f}   F {F0:.6f}   PROB_SHIFT {shift0:.9f}")
    print(f"  목표  R {R:.6f}   F {F:.6f}" + (f"   (t = {a.t})" if a.t is not None else ""))
    if abs(R - R0) < 1e-12 and abs(F - F0) < 1e-12:
        raise SystemExit("[중단] 계수가 이미 목표값이다. 바꿀 것이 없다.")

    shift = shift0
    if not a.no_recal:
        print(f"\n프록시 {PROXY_SEASON}행 적재 …")
        proxy = load_proxy()
        ppq = os.path.join(tmp, "proxy.pkl")
        proxy.to_pickle(ppq)
        print(f"  {len(proxy):,}행 · F 비율 {(proxy['game_type'].astype(str) == 'F').mean():.4f}")

        print("추론 ① 패치 전 …")
        p_before = predict_in_subprocess(work, ppq, os.path.join(tmp, "b.npy"))

        open(ppath, "w", encoding="utf-8").write(write_routing(src0, R, F, shift0))
        print("추론 ② 패치 후 (PROB_SHIFT 미보정) …")
        p_mid = predict_in_subprocess(work, ppq, os.path.join(tmp, "m.npy"))

        d = p_mid - p_before
        dmean = float(d.mean())
        shift = shift0 - dmean
        print(f"\n  max|Δp|   {np.abs(d).max():.6g}")
        print(f"  평균|Δp|  {np.abs(d).mean():.6g}")
        print(f"  Δmean     {dmean:+.6g}")
        print(f"  PROB_SHIFT {shift0:.9f} -> {shift:.9f}")

        open(ppath, "w", encoding="utf-8").write(write_routing(src0, R, F, shift))
        print("추론 ③ 재보정 검증 …")
        p_after = predict_in_subprocess(work, ppq, os.path.join(tmp, "a.npy"))
        resid = float((p_after - p_before).mean())
        print(f"  패치 후 Δmean {resid:+.3g}  " + ("✅" if abs(resid) < 1e-8 else "⚠️ 1e-8 초과"))
        print(f"  최종 max|Δp| {np.abs(p_after - p_before).max():.6g}")
    else:
        open(ppath, "w", encoding="utf-8").write(write_routing(src0, R, F, shift0))
        print("\n[건너뜀] PROB_SHIFT 재보정 없음 — 레벨이 어긋난 채로 나간다")

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            fp = os.path.join(work, n)
            if os.path.isfile(fp):
                z.write(fp, n)

    diff = [n for n in names if os.path.isfile(os.path.join(work, n))
            and open(os.path.join(work, n), "rb").read() != open(os.path.join(orig, n), "rb").read()]
    print(f"\n변경 격리  {len(diff)}/{len(names)}개 파일 상이: {diff}")
    if diff != ["predict.py"]:
        raise SystemExit("[중단] predict.py 외의 파일이 바뀌었다. 재패킹을 신뢰할 수 없다.")
    print(f"출력   {a.out}\n  SHA-256 {sha(a.out)}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
