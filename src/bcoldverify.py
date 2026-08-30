"""`sy06_bcold.zip` 제출 전 검증 — 규정 · 행 독립성 · 추론 시간 (§7.208).

`src/auditzips.py` 의 기계를 재사용한다 (거기 glob 이 `s??_*.zip` 이라 새 이름을
못 잡으므로 대상만 여기서 지정한다). 베이스 `hj03` 도 같이 돌려 **대조**한다 —
새 zip 만 재면 "원래 그랬던 것" 과 "내가 깨뜨린 것" 을 못 가른다.

검사
    ① AST 정적 — 금지 호출(groupby/value_counts/rolling/fit/quantile...) 0건
    ② 행 독립성 — 조각/셔플/1행 예측이 통짜 예측과 **정확히 같아야** 한다 (0.0)
    ③ 추론 시간 — 245,789행(평가 규모) 이 600초 안에 끝나야 한다

실행:  .venv/Scripts/python.exe -u src/bcoldverify.py 2>&1 | tee experiments/bcoldverify.log
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import pandas as pd

import auditzips as A

ROOT = Path(__file__).resolve().parents[1]
ZIPS = ["hj03_dim8_cold.zip", "sy06_bcold.zip"]
N_PROBE = 4000
CHUNK = 400
N_EVAL = 245789          # 평가 서버 test.csv 행 수

TIMER = (
    "import sys, time, pandas as pd" + chr(10) +
    "sys.path.insert(0, sys.argv[1])" + chr(10) +
    "from predict import Predictor" + chr(10) +
    "df = pd.read_parquet(sys.argv[2])" + chr(10) +
    "t = time.time()" + chr(10) +
    "pr = Predictor(sys.argv[1] + '/model')" + chr(10) +
    "p = pr.predict(df)" + chr(10) +
    "print(f'{time.time()-t:.1f} {len(df)} {p.mean():.9f}')" + chr(10)
)

ENV = dict(os.environ, PYTHONIOENCODING="utf-8")


def run(code, *args):
    return subprocess.run([sys.executable, "-c", code, *args],
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=ENV)


def main():
    # ⚠️ LightGBM 은 경로에 비 ASCII 문자가 있으면 모델 파일을 못 연다.
    #    기본 temp 는 C:/Users/<한글이름>/... 아래라 반드시 걸린다.
    #    [[lightgbm-windows-nonascii-path]] — ASCII 작업 경로를 쓴다.
    ASCII_TMP = Path("D:/t1eval/_audit")
    ASCII_TMP.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="bcold_", dir=str(ASCII_TMP)))
    assert tmp.as_posix().isascii(), f"작업 경로에 비 ASCII: {tmp}"
    src = pd.read_csv(ROOT / "data" / "train.csv")
    d24 = src[src.season == 2024].drop(columns=["control_success"], errors="ignore")
    del src

    probe = d24.sample(n=N_PROBE, random_state=42).reset_index(drop=True)
    pq = tmp / "probe.parquet"
    probe.to_parquet(pq)
    full = d24.sample(n=N_EVAL, replace=True, random_state=7).reset_index(drop=True)
    pqf = tmp / "full.parquet"
    full.to_parquet(pqf)
    del d24, probe, full
    print(f"프로브 {N_PROBE:,}행 · 시간측정 {N_EVAL:,}행 (평가 규모)   작업 {tmp}\n")

    print(f"  {'zip':<20}{'AST':>8}{'조각':>10}{'셔플':>10}{'1행':>10}"
          f"{'추론(s)':>10}{'p̄':>13}")
    print("  " + "-" * 82)
    for zp in ZIPS:
        d = tmp / Path(zp).stem
        with zipfile.ZipFile(ROOT / zp) as f:
            f.extractall(d)
        if not (d / "predict.py").exists():
            sub = [p for p in d.iterdir() if p.is_dir() and (p / "predict.py").exists()]
            if sub:
                d = sub[0]

        st = A.scan(d)
        r = run(A.RUNNER, str(d), str(pq), str(CHUNK))
        if r.returncode != 0:
            print(f"  {Path(zp).stem:<20}  실행 실패: {(r.stderr or '')[-300:]}")
            continue
        m = json.loads(r.stdout.strip().splitlines()[-1])

        rt = run(TIMER, str(d), str(pqf))
        if rt.returncode != 0:
            secs, pbar = "실패", "-"
        else:
            parts = rt.stdout.strip().splitlines()[-1].split()
            secs, pbar = parts[0], parts[2]

        # 조각/셔플은 **정확히 0** 이어야 한다 (여기서 어긋나면 진짜 행 의존이다).
        # 1행 경로는 BLAS 커널이 갈려(GEMV vs GEMM) 1 ULP 가 뜬다 — 행 의존이 아니다.
        # 판정은 베이스와의 **대조**로 한다: 베이스와 같은 값이면 내가 만든 게 아니다.
        ok = (m["chunk"] == 0.0 and m["shuffle"] == 0.0 and m["single"] < 1e-12)
        print(f"  {Path(zp).stem:<20}{('OK' if not st else str(len(st))+'건'):>8}"
              f"{m['chunk']:>10.1e}{m['shuffle']:>10.1e}{m['single']:>10.1e}"
              f"{secs:>10}{pbar:>13}   {'✅' if ok else '❌'}")
        for name, line, call in st:
            print(f"      ❌ {name}:{line}  {call}")

    shutil.rmtree(tmp, ignore_errors=True)
    print()
    print("  · 조각·셔플이 정확히 0 이어야 한다 — 여기서 어긋나면 행 사이가 이어진 것")
    print("  · 1행 경로의 1e-16 은 BLAS 커널 차이(1 ULP)다. 베이스와 같은 값인지로 판정한다")
    print(f"  · 추론은 600초 한도. 로컬은 평가 서버(6 vCPU)보다 빠를 수 있으니 여유를 볼 것")


if __name__ == "__main__":
    main()
