"""§7.19x — s48 후보 빌드: 챔피언 s45 위에서 **CatBoost α 의 F행 차단** (α_R=.10, α_F=0)

근거 사슬
    §7.145④  (구 스택 s35층, fold-정직)  cat 기여 F 초과분 −0.826 [−1.647, −0.029]
    catrf45   (s45 등가층)               F 초과분 −3.844 [−5.947, −1.922], 게이트 3/3
              단 이 층은 F 기저의 in-sample F전문가(37.5%) 때문에 α_F=0 쪽으로
              편향돼 있어 크기는 못 믿는다 — 방향 재현으로만 읽는다.
    기대 LB   +0 ~ +0.5 (정직한 §7.145④ 크기 기준 보수적으로)

변경: predict.py 의 cat 블렌드 1줄 (전 행 → R 행만) + PROB_SHIFT 재보정.
모델 파일 전부 원본 → LB 변화 = α_F 차단분.

⚠️ 재베이스 (§7.191 실측 후): s47r066f355 이 신기록(1119.7067)으로 챔피언이 되어
   베이스를 s45 → **s47** 로 올렸다. 귀속이 깨끗해지고 max 채점에서 두 이득이
   합산된다. 기준 프록시 = rt45_new2.npz (s47 최종 2024 예측).
   1차 빌드(s45 베이스, CATF0/catf0_* 캐시)는 폐기 — 캐시 접두를 catf0b_ 로 갈랐다
   ([[assert-the-cache-key-actually-split]]).

실행:  .venv/Scripts/python.exe -u src/catf0build.py > experiments/catf0build.log 2>&1
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
BASE_ZIP = ROOT / "s47r066f355.zip"          # 신챔피언 (§7.191)
OUT = ROOT / "s48_catf0.zip"
REF_DIR = ASCII / "RT45"                     # s47 해제본
NEW_DIR = ASCII / "CATF0B"
REF_NPZ = ROOT / "experiments" / "cache" / "rt45_new2.npz"   # s47 의 2024 프록시

OLD = "            z = (1.0 - self.cat_alpha) * z + self.cat_alpha * zc"
NEW = ("            # α_F=0 — cat 은 R 행에만 섞는다 (§7.145④ F 손해 · catrf45 방향 재현)\n"
       "            z[is_r] = ((1.0 - self.cat_alpha) * z[is_r]\n"
       "                       + self.cat_alpha * zc[is_r])")


def proxy(dname, tag):
    f = ROOT / "experiments" / "cache" / f"catf0b_{tag}.npz"
    if not f.exists():
        wk = ROOT / "experiments" / "cache" / "_catf0b_worker.py"
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
        assert src.count(OLD) == 1, f"패치 대상 {src.count(OLD)}회"
        # is_r 이 cat 블렌드 시점에 스코프에 있어야 한다 (predict.py:168)
        assert src.index("is_r = ~is_f") < src.index(OLD)
        src = src.replace(OLD, NEW)
        (NEW_DIR / "predict.py").write_text(src, encoding="utf-8", newline="\n")
        print("cat 블렌드 패치 완료 — 전 행 → R 행만 (α_R=0.10, α_F=0)")
    else:
        assert "z[is_r] = ((1.0 - self.cat_alpha)" in src, "패치 상태 불명"
        print("패치 재사용 (이미 적용됨)")

    p_ref = np.load(REF_NPZ)["p"]
    p_new = proxy("CATF0B", "new")
    assert len(p_new) == len(p_ref)
    dmax = float(np.abs(p_new - p_ref).max())
    dmean = float(p_new.mean() - p_ref.mean())
    assert dmax > 0, "패치가 예측에 반영되지 않았다"
    print(f"[{time.time()-t0:.0f}s] 프록시 2024 {len(p_ref):,}행  "
          f"max|Δp|={dmax:.5f}  평균|Δp|={np.abs(p_new-p_ref).mean():.6f}  "
          f"Δmean={dmean:+.9f}")
    # F행(11.8%)에서만 달라야 한다
    nz = float((np.abs(p_new - p_ref) > 0).mean())
    print(f"  예측이 달라진 행 비율 = {nz:.4f}  (F행 비율 ~0.118 과 일치해야 한다)")
    assert 0.10 < nz < 0.14, "변경이 F 행 밖으로 샜다"

    src = (NEW_DIR / "predict.py").read_text(encoding="utf-8").replace(CRLF, LF)
    m = re.search(r"PROB_SHIFT\s*=\s*([0-9.eE+-]+)", src)
    shift_old = float(m.group(1))
    shift_new = shift_old - dmean
    src2 = src[:m.start(1)] + f"{shift_new:.9f}" + src[m.end(1):]
    (NEW_DIR / "predict.py").write_text(src2, encoding="utf-8", newline="\n")
    print(f"  PROB_SHIFT {shift_old:.9f} → {shift_new:.9f}")

    (ROOT / "experiments" / "cache" / "catf0b_new2.npz").unlink(missing_ok=True)
    p_new2 = proxy("CATF0B", "new2")
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
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
