"""§7.19x — s47 빌드: 챔피언 s45 위에서 **R/F 라우팅 t=−0.45** (§7.177 미완 축 계속)

§7.177 이 s41 위에서 t=−0.25 실측 +0.2469 를 얻고, 역산 곡률 4가정 전부에서
꼭짓점 t* ∈ [−1.01, −0.42] "더 내려갈 여지"를 남겼다. 다음 점으로 사전 지정된
t=−0.45 (네 가정 전부 +0.29~+0.39) 는 s43/s45 로 넘어가며 제출되지 않았다.

이 빌드는 그 점을 **현 챔피언 s45_cap2.zip 위에서** 낸다:

    ray (혜진 §15-9 방향):  R = 0.10 + t·0.075,  F = 0.40 + t·0.10
    t=−0.25 (s45 현재)      R 0.081250  F 0.375000
    t=−0.45 (이 빌드)       R 0.066250  F 0.355000

변경은 predict.py 의 라우팅 계수 2줄 + PROB_SHIFT 재보정뿐이다. 모델 파일
전부 원본 → LB 변화 = 라우팅분 (§7.177 의 귀속 원칙 계승).

⚠️ 기울기는 s41 에서 잰 것이라 s45 에서는 낡았다. 다만 s43(cat12)·s45(leaves×2)
   둘 다 코어 강화 방향이라 최적 라우팅은 코어 몫을 더 키우는(t 더 음수) 쪽으로
   밀렸을 개연성이 높다. 홀드아웃은 이 축에서 무효(§7.174) — LB 만이 계측기.

레벨 재보정 (§7.178 ② — T1 계보 logit_delta 동결)
    기준측 2024 프록시는 §7.183a 가 만든 캐시 t1cap46_ref.npz (= S45REF) 재사용.
    신규측 프록시 Δmean 만큼 PROB_SHIFT 에서 빼 평균을 소수 9자리까지 일치.

실행:  .venv/Scripts/python.exe -u src/rt45build.py > experiments/rt45build.log 2>&1
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
BASE_ZIP = ROOT / "s45_cap2.zip"             # 챔피언 1119.4894859009
OUT = ROOT / "s47r066f355.zip"
REF_DIR = ASCII / "S45REF"                   # §7.183a 가 해제해 둔 챔피언
NEW_DIR = ASCII / "RT45"
REF_NPZ = ROOT / "experiments" / "cache" / "t1cap46_ref.npz"   # §7.183a 캐시 (S45REF 프록시)

# t=−0.45 : 검산 — ray 정의에서 유도
T = -0.45
R_G = 0.10 + T * 0.075       # 일반 LGBM 몫
F_Z = 0.40 + T * 0.100       # 채연 F전문가 몫
assert abs(R_G - 0.066250) < 1e-12 and abs(F_Z - 0.355000) < 1e-12
SWAPS = [  # (old, new) — 각자 predict.py 에 정확히 1회 존재해야 한다
    ("0.918750", "0.933750"), ("0.081250", "0.066250"),
    ("0.625000", "0.645000"), ("0.375000", "0.355000"),
]
assert abs(float(SWAPS[0][1]) + float(SWAPS[1][1]) - 1.0) < 1e-12
assert abs(float(SWAPS[2][1]) + float(SWAPS[3][1]) - 1.0) < 1e-12


def proxy(dname, tag):
    """2024 전 행을 해당 디렉터리의 T1 predict.py 로 예측 (t1core WORKER 재사용)."""
    f = ROOT / "experiments" / "cache" / f"rt45_{tag}.npz"
    if not f.exists():
        wk = ROOT / "experiments" / "cache" / "_rt45_worker.py"
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

    # ── 1. 신규 팔 = S45REF 사본 + 라우팅 2줄 패치 ──────────────────
    if not (NEW_DIR / "predict.py").exists():
        shutil.copytree(REF_DIR, NEW_DIR, dirs_exist_ok=True)
    src = (NEW_DIR / "predict.py").read_text(encoding="utf-8").replace(CRLF, LF)

    # 무결성: 팔 predict 가 zip 의 predict 와 같은 내용에서 출발했는가
    with zipfile.ZipFile(BASE_ZIP) as z:
        zsrc = z.read("predict.py").decode("utf-8").replace(CRLF, LF)
    if src == zsrc:                      # 첫 실행 — 패치 전
        for old, new in SWAPS:
            assert src.count(old) == 1, f"{old!r} {src.count(old)}회"
            src = src.replace(old, new)
        (NEW_DIR / "predict.py").write_text(src, encoding="utf-8", newline="\n")
        print("라우팅 패치 완료  R 0.081250→0.066250 · F 0.375000→0.355000")
    else:                                # 재실행 — 이미 패치됨을 검증
        for _, new in SWAPS:
            assert src.count(new) == 1, f"패치 상태 불명: {new!r}"
        print("라우팅 패치 재사용 (이미 적용됨)")

    # ── 2. 프록시 평균 정렬 → PROB_SHIFT 재보정 ─────────────────────
    p_ref = proxy("S45REF", "ref_chk") if not REF_NPZ.exists() else np.load(REF_NPZ)["p"]
    p_new = proxy("RT45", "new")
    assert len(p_new) == len(p_ref)
    dmax = float(np.abs(p_new - p_ref).max())
    dmean = float(p_new.mean() - p_ref.mean())
    assert dmax > 0, "라우팅 패치가 예측에 반영되지 않았다"
    print(f"[{time.time()-t0:.0f}s] 프록시 2024 {len(p_ref):,}행  "
          f"max|Δp|={dmax:.5f}  평균|Δp|={np.abs(p_new-p_ref).mean():.6f}  "
          f"Δmean={dmean:+.9f}  (방치 시 레벨 비용 {1e5*dmean**2/0.2497:.4f}점)")

    src = (NEW_DIR / "predict.py").read_text(encoding="utf-8").replace(CRLF, LF)
    m = re.search(r"PROB_SHIFT\s*=\s*([0-9.eE+-]+)", src)
    shift_old = float(m.group(1))
    # 재실행 안전: 이미 보정된 상태면 Δmean≈0 이라 재보정이 항등이다
    shift_new = shift_old - dmean
    src2 = src[:m.start(1)] + f"{shift_new:.9f}" + src[m.end(1):]
    (NEW_DIR / "predict.py").write_text(src2, encoding="utf-8", newline="\n")
    print(f"  PROB_SHIFT {shift_old:.9f} → {shift_new:.9f}")

    (ROOT / "experiments" / "cache" / "rt45_new2.npz").unlink(missing_ok=True)
    p_new2 = proxy("RT45", "new2")
    d2 = float(p_new2.mean() - p_ref.mean())
    print(f"  패치 후 Δmean = {d2:+.12f}  {'✅' if abs(d2) < 1e-9 else '⚠️'}")
    assert abs(d2) < 1e-7, "평균 정렬 실패"

    # ── 3. 패키징 — 챔피언 zip 그대로, predict.py 만 교체 ───────────
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
    print("§7.177 예측(단, s41 층 기준): 4곡률 가정 전부 +0.29~+0.39")
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
