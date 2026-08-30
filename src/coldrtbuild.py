"""§7.203b — `sy04_cold` 위에 **R/F 라우팅 t=-0.45** (§7.191) 를 얹는다.

§7.191 은 이 라우팅을 s45 -> s47 단계에서 재 **LB +0.2172** 를 실측했다. 그런데
그 이득은 그 뒤 계보(sy02_c4 -> hj01 -> hj03)에 **한 번도 안 들어갔다** — 혜진
라운드 18 리포트도 "원우 s47 라우팅(+0.2172)과 `s48_catf0` 은 아직 이 계보에
안 들어가 있다" 로 적어놨다. 여기서 회수한다.

    t=-0.25 (계보 현재)     R 0.081250  F 0.375000
    t=-0.45 (이 빌드)       R 0.066250  F 0.355000

왜 이 축은 전이를 믿을 만한가
    라우팅은 **재학습이 없는 결정적 변경**이다. §7.198~§7.199 가 잰 MLP 층 뽑기
    잡음 SD 4.8~7.1 이 여기엔 안 붙는다 — 같은 zip 은 같은 LB 를 낸다. +0.2172 는
    그 베이스에 대한 **참값**이지 추정값이 아니다.
    다만 **베이스가 바뀌면 최적 t 는 움직인다.** 라우팅은 R/F 전문가의 혼합비인데
    MLP 임베딩이 바뀌면 섞이는 대상이 바뀐다. 그래서 크기는 전이 안 되고
    부호만 기대한다. 규모가 ±0.2 라 틀려도 손해가 그만큼이다.
    ([[additive-axis-lb-forecast-splits-gain-and-variance-cost]] — 이 축은 직교
    비용이 거의 없어 순이득이 곧 이득이다.)

베이스: sy04_cold.zip (= sy02_c4 + 콜드스타트, §7.203)
변경:   predict.py 의 라우팅 계수 2쌍 + PROB_SHIFT 재보정. 모델 파일은 전부 원본.

실행:  .venv/Scripts/python.exe -u src/coldrtbuild.py > experiments/coldrtbuild.log 2>&1
"""
import hashlib
import re
import shutil
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

from coldbuild import CACHE, CRLF, LF, proxy

BASE_ZIP = ROOT / "sy04_cold.zip"
OUT = ROOT / "sy05_coldrt.zip"
REF_DIR = ASCII / "COLD"             # sy04_cold 의 작업본 (패치·재보정 완료 상태)
NEW_DIR = ASCII / "COLDRT"

# 상보 쌍 2개 — general/R 게이트와 F 전문가 가중.
# ⚠️ 리터럴 포맷이 계보마다 다르다. s45 계보는 "0.081250", 신영 계보는 "0.08125".
#    src/rt45build.py 의 6자리 리터럴을 그대로 물려썼다가 0회 매칭으로 걸렸다
#    ([[audit-inherited-constants-not-just-registered-gates]]).
#    그래서 상수가 아니라 **줄 전체**를 갈아끼운다 — 포맷에 안 걸리고 1회 매칭이 강제된다.
PAIRS = [
    ("        z[is_r] = 0.91875 * z[is_r] + 0.08125 * zg[is_r]",
     "        z[is_r] = 0.93375 * z[is_r] + 0.06625 * zg[is_r]"),
    ("        z[is_f] = 0.625 * z[is_f] + 0.375 * zf[is_f]",
     "        z[is_f] = 0.645 * z[is_f] + 0.355 * zf[is_f]"),
]


def main():
    t0 = time.time()
    assert BASE_ZIP.exists(), BASE_ZIP

    # -- REF 무결성: COLD 작업본이 정말 sy04_cold 인가 -----------------------
    with zipfile.ZipFile(BASE_ZIP) as z:
        names = z.namelist()
        bad = [n for n in names
               if (REF_DIR / n).is_file() and (REF_DIR / n).read_bytes() != z.read(n)]
        assert not bad, f"REF 가 베이스와 다르다: {bad[:5]}"
    print(f"베이스 {BASE_ZIP.name} · 파일 {len(names)}개 · REF 바이트 일치 OK")

    if not (NEW_DIR / "predict.py").exists():
        shutil.copytree(REF_DIR, NEW_DIR, dirs_exist_ok=True)
        print(f"NEW 복사: {NEW_DIR}")

    src = (NEW_DIR / "predict.py").read_text(encoding="utf-8").replace(CRLF, LF)
    with zipfile.ZipFile(BASE_ZIP) as z:
        zsrc = z.read("predict.py").decode("utf-8").replace(CRLF, LF)
    if src == zsrc:
        for old, new in PAIRS:
            assert src.count(old) == 1, f"'{old}' {src.count(old)}회 (1회여야 한다)"
        for old, new in PAIRS:
            src = src.replace(old, new)
        (NEW_DIR / "predict.py").write_text(src, encoding="utf-8", newline="\n")
        print("라우팅 패치 완료  R 0.081250->0.066250 · F 0.375000->0.355000")
    else:
        assert all(new in src for _, new in PAIRS), "패치 상태 불명"
        print("패치 재사용 (이미 적용됨)")

    # -- 같은 마스킹 프록시 위에서 Δ 측정 ------------------------------------
    p_ref, mask = proxy("COLD", "new2")          # sy04_cold 최종 (재보정 완료)
    p_new, mask2 = proxy("COLDRT", "rt")
    assert np.array_equal(mask, mask2), "두 팔의 마스크가 다르다"
    assert not np.array_equal(p_new, p_ref), "두 팔이 같다 — 캐시가 안 갈렸다"
    dmean = float(p_new.mean() - p_ref.mean())
    nz = float((np.abs(p_new - p_ref) > 0).mean())
    print(f"[{time.time()-t0:.0f}s] 프록시 2024 {len(p_ref):,}행")
    print(f"  max|Δp| {np.abs(p_new-p_ref).max():.6g} · 평균|Δp| {np.abs(p_new-p_ref).mean():.6g}"
          f" · 달라진 행 {nz:.4f}")
    print(f"  Δmean {dmean:+.9f}")

    src = (NEW_DIR / "predict.py").read_text(encoding="utf-8").replace(CRLF, LF)
    m = re.search(r"PROB_SHIFT\s*=\s*([0-9.eE+-]+)", src)
    shift_old = float(m.group(1))
    shift_new = shift_old - dmean
    (NEW_DIR / "predict.py").write_text(
        src[:m.start(1)] + f"{shift_new:.9f}" + src[m.end(1):],
        encoding="utf-8", newline="\n")
    print(f"  PROB_SHIFT {shift_old:.9f} -> {shift_new:.9f}")

    (CACHE / "coldrt_rt2.npz").unlink(missing_ok=True)
    p_new2, _ = proxy("COLDRT", "rt2")
    d2 = float(p_new2.mean() - p_ref.mean())
    print(f"  재보정 후 Δmean = {d2:+.12f}  {'OK' if abs(d2) < 1e-9 else '주의'}")
    assert abs(d2) < 1e-7, "평균 정렬 실패"

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(BASE_ZIP) as zin, \
            zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
        for n in names:
            fp = NEW_DIR / n
            zout.writestr(n, fp.read_bytes() if fp.is_file() else zin.read(n))
        diff = sorted(n for n in names if (NEW_DIR / n).is_file()
                      and (NEW_DIR / n).read_bytes() != zin.read(n))
    print(f"\n변경 격리 {len(diff)}/{len(names)}: {diff}")
    assert diff == ["predict.py"], "라우팅 밖으로 변경이 샜다"
    assert len(OUT.name) <= 30
    print(f"{OUT.name}  {OUT.stat().st_size/1e6:.1f} MB")
    print(f"  SHA-256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
