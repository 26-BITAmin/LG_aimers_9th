"""§7.191 후속 — s49 후보 빌드: 신챔피언 s47 위에서 **R/F 라우팅 t=−0.70**

§7.191 실측(+0.2172)으로 s45 곡선의 국소 기울기(1.086/단위t)가 s41 곡선(0.988)에서
안 꺾인 것이 확인됐다 — 꼭짓점은 −0.45 보다 한참 아래. 한 구간 + 곡률 가정 재적합:

    C 가정        t*        t=−0.70 한 걸음
    혜진 ×1     −1.328        +0.209
    ×2          −0.839        +0.147
    ×3          −0.676        +0.084     ← 전 가정 양수, §7.177 선정 논리

    ray:  R = 0.10 + t·0.075,  F = 0.40 + t·0.10
    t=−0.45 (s47)   R 0.066250   F 0.355000
    t=−0.70 (이것)  R 0.047500   F 0.330000

기준측 2024 프록시 = rt45_new2.npz (s47 최종 예측, §7.191 빌드 캐시) 재사용.

실행:  .venv/Scripts/python.exe -u src/rt70build.py > experiments/rt70build.log 2>&1
"""
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
ASCII = Path("D:/t1eval")
sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

from config import TRAIN_CSV
from t1core import WORKER

CRLF = chr(13) + chr(10)
LF = chr(10)
BASE_ZIP = ROOT / "s47r066f355.zip"          # 신챔피언 1119.7066843417
OUT = ROOT / "s49r047f330.zip"
REF_DIR = ASCII / "RT45"                     # s47 해제본 (§7.191 빌드 산출)
NEW_DIR = ASCII / "RT70"
REF_NPZ = ROOT / "experiments" / "cache" / "rt45_new2.npz"   # s47 의 2024 프록시

T = -0.70
R_G = 0.10 + T * 0.075
F_Z = 0.40 + T * 0.100
assert abs(R_G - 0.047500) < 1e-12 and abs(F_Z - 0.330000) < 1e-12
SWAPS = [
    ("0.933750", "0.952500"), ("0.066250", "0.047500"),
    ("0.645000", "0.670000"), ("0.355000", "0.330000"),
]
assert abs(float(SWAPS[0][1]) + float(SWAPS[1][1]) - 1.0) < 1e-12
assert abs(float(SWAPS[2][1]) + float(SWAPS[3][1]) - 1.0) < 1e-12


def proxy(dname, tag):
    f = ROOT / "experiments" / "cache" / f"rt70_{tag}.npz"
    if not f.exists():
        wk = ROOT / "experiments" / "cache" / "_rt70_worker.py"
        wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/")),
                      encoding="utf-8", newline="\n")
        r = subprocess.run([sys.executable, str(wk), dname, str(f)],
                           capture_output=True, text=True)
        if not f.exists():
            print(f"  ❌ {dname}: {r.stderr.strip()[-400:]}")
            sys.exit(1)
    return np.load(f)["p"]


def main():
    t0 = time.time()
    assert BASE_ZIP.exists() and REF_NPZ.exists() and (REF_DIR / "predict.py").exists()

    if not (NEW_DIR / "predict.py").exists():
        shutil.copytree(REF_DIR, NEW_DIR, dirs_exist_ok=True)
    src = (NEW_DIR / "predict.py").read_text(encoding="utf-8").replace(CRLF, LF)
    with zipfile.ZipFile(BASE_ZIP) as z:
        zsrc = z.read("predict.py").decode("utf-8").replace(CRLF, LF)
    if src == zsrc:
        for old, new in SWAPS:
            assert src.count(old) == 1, f"{old!r} {src.count(old)}회"
            src = src.replace(old, new)
        (NEW_DIR / "predict.py").write_text(src, encoding="utf-8", newline="\n")
        print("라우팅 패치 완료  R 0.066250→0.047500 · F 0.355000→0.330000")
    else:
        for _, new in SWAPS:
            assert src.count(new) == 1, f"패치 상태 불명: {new!r}"
        print("라우팅 패치 재사용 (이미 적용됨)")

    p_ref = np.load(REF_NPZ)["p"]
    p_new = proxy("RT70", "new")
    assert len(p_new) == len(p_ref)
    dmax = float(np.abs(p_new - p_ref).max())
    dmean = float(p_new.mean() - p_ref.mean())
    assert dmax > 0, "라우팅 패치가 예측에 반영되지 않았다"
    print(f"[{time.time()-t0:.0f}s] 프록시 2024 {len(p_ref):,}행  "
          f"max|Δp|={dmax:.5f}  평균|Δp|={np.abs(p_new-p_ref).mean():.6f}  "
          f"Δmean={dmean:+.9f}")

    src = (NEW_DIR / "predict.py").read_text(encoding="utf-8").replace(CRLF, LF)
    m = re.search(r"PROB_SHIFT\s*=\s*([0-9.eE+-]+)", src)
    shift_old = float(m.group(1))
    shift_new = shift_old - dmean
    src2 = src[:m.start(1)] + f"{shift_new:.9f}" + src[m.end(1):]
    (NEW_DIR / "predict.py").write_text(src2, encoding="utf-8", newline="\n")
    print(f"  PROB_SHIFT {shift_old:.9f} → {shift_new:.9f}")

    (ROOT / "experiments" / "cache" / "rt70_new2.npz").unlink(missing_ok=True)
    p_new2 = proxy("RT70", "new2")
    d2 = float(p_new2.mean() - p_ref.mean())
    print(f"  패치 후 Δmean = {d2:+.12f}  {'✅' if abs(d2) < 1e-9 else '⚠️'}")
    assert abs(d2) < 1e-7, "평균 정렬 실패"

    src_final = (NEW_DIR / "predict.py").read_text(encoding="utf-8").replace(CRLF, LF)
    with zipfile.ZipFile(BASE_ZIP) as zin:
        names = zin.namelist()
        if OUT.exists():
            OUT.unlink()
        with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
            for n in names:
                if n == "predict.py":
                    zout.writestr(n, src_final)
                else:
                    zout.writestr(n, zin.read(n))
    assert len(OUT.name) <= 30
    print(f"\n{OUT.name}  {OUT.stat().st_size/1e6:.1f} MB  "
          f"(파일 {len(names)}개 · 교체 1개 = predict.py)")
    print("기대 (s45 곡선 재적합): C×1/×2/×3 = +0.209 / +0.147 / +0.084")
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
