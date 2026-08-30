"""타자 콜드스타트 — **팀 챔피언 `hj03` 위**에서 `emb1[0]` 를 무게중심으로 (§7.208)

무엇을 고치나
    MLP 임베딩 표의 **0번 행은 "학습 때 못 본 값" 자리**다. 학습 중 gradient 를 한 번도
    받지 않으므로 초기화 상태로 남는다. 배포 zip 자신의 추론 코드가 그렇게 만든다:

        mlp_infer.py:71   C[:, i] = np.where(keys[pos] == v, idx[pos], 0)   # 미지값 -> 0

    §7.203(혜진 라운드 18 · 우리 재현)이 고친 것은 `emb0`(**pitcher_id**) 하나다.
    같은 결함이 `emb1`(**batter_id**) 에 그대로 남아 있다.

왜 batter_id 만 남은 문제인가 (7개 표 중)
    | 표 | 2025 에 미지값이 나타나나 | |
    | emb0 pitcher_id | 예 (신규 투수 19.9%) | ✅ §7.203 에서 패치됨 |
    | emb1 batter_id  | **예 (신규 타자 9.3%)** | ❌ **미패치 ← 이 파일** |
    | emb2/3 team_id  | 아니오 (팀 고정) | 무해 |
    | emb4/5/6 base_state·hand_matchup·count_state | 아니오 (전수 열거) | 무해 |
    → 노름이 커 보여도 그 슬롯을 아무도 안 타면 예측에 영향이 없다. **노출이 판정자다.**

결함 실측 (`hj03`, emb1 = batter_id, dim 4)
    seed      |row0|   |centroid|   사잇각   노름비   학습행 노름 %ile
      42      0.6286      0.0795     92.2°     7.91         1%
       7      2.2474      0.1067     37.3°    21.07        74%
    2024      0.9688      0.0726    112.9°    13.34         9%
    1234      1.5513      0.0787     93.6°    19.72        37%
   31337      1.9744      0.0426     85.1°    46.38        59%
    → §7.203 이 투수 슬롯에서 기술한 *"학습 노름 분포의 1.4~96 백분위를 시드마다 배회"* 와
      같은 모양이다. **미학습 슬롯이 진짜 타자처럼 생긴 가짜다.**
    대조: 패치된 emb0 는 전 시드 노름비 **1.0000** · 사잇각 **0.0°**.

베이스를 hj03 으로 잡는 이유
    사용자 지시("챔피언을 바탕으로"). 그리고 §7.206 이 hj03 의 정체를 확정했다 —
    `sy02_c4` + 패치가 아니라 **emb4 5멤버가 다른 학습 draw** 이고, LGBM/CatBoost/noid/
    artifacts 는 전부 바이트 동일이다. 그 위에 **이 변경 하나만** 얹으면
    LB 차이가 곧 타자 패치의 순효과가 된다.
    ⚠️ 그래서 §7.191 라우팅(t=−0.45, 미회수 +0.2172)은 **일부러 같이 넣지 않는다** —
       넣으면 §7.206 이 hj03 에서 지적한 것과 같은 교락이 재발한다.

마스킹 프록시가 정당한 이유
    2024 프록시에는 미지 타자가 0명이다(학습에 2024 가 들어갔다). 그래서 타자를 골라
    `batter_id = -1` 로 만들어 미지 경로를 태운다. 이때 임베딩만이 아니라 `asof_batter_*`
    조회까지 흔들리지만, **두 팔이 정확히 같은 마스크를 쓰므로 그 몫은 Δp 에서 상쇄된다.**
    살아남는 것은 `emb1[0]` 차이 하나뿐이고, **"알려진 행 Δp = 정확히 0"** 이 그 관문이다.

레벨
    패치는 미지 행의 p 를 움직이므로 전체 평균이 밀린다. `PROB_SHIFT` 를 그만큼 되돌려
    **레벨을 챔피언에 고정**한다 → LB 차이에 refinement 축만 남는다 (§7.203 과 같은 절차).

실행:  .venv/Scripts/python.exe -u src/bcoldbuild.py 2>&1 | tee experiments/bcoldbuild.log
"""
import hashlib
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

CRLF = chr(13) + chr(10)
LF = chr(10)
BASE_ZIP = ROOT / "hj03_dim8_cold.zip"     # 팀 챔피언 LB 1131.4503140506
OUT = ROOT / "sy06_bcold.zip"
REF_DIR = ASCII / "HJ03REF"                # 챔피언 원본 (대조군)
NEW_DIR = ASCII / "BCOLD"                  # 타자 패치 팔
UNK = 0.093                                # 프록시 미지 타자 목표 행 비율 (2024 실측 9.3%)
CACHE = ROOT / "experiments" / "cache"

WORKER_MASK = r"""
import os, sys, numpy as np, pandas as pd
arm, out = sys.argv[1], sys.argv[2]
d = r"D:/t1eval/" + arm
sys.path.insert(0, d)          # 우리 src 를 넣지 않는다 (§7.174 사고)
va = pd.read_csv(r"{csv}")
va = va[va.season == 2024].reset_index(drop=True)
# --- 미지 타자 마스킹: 타자 단위로 골라 목표 행 비율을 맞춘다 ---
rng = np.random.RandomState(0)
bids = rng.permutation(np.sort(va["batter_id"].unique()))
cnt = va["batter_id"].value_counts()
take, tot, goal = [], 0, {unk} * len(va)
for bid in bids:
    if tot >= goal:
        break
    take.append(bid); tot += int(cnt[bid])
mask = va["batter_id"].isin(take).to_numpy()
va.loc[mask, "batter_id"] = -1
from predict import Predictor
import predict as _pm
assert _pm.__file__.replace("\\", "/").startswith(d), "그 zip 의 predict.py 가 아니다"
p = Predictor(model_dir=d + "/model")
np.savez_compressed(out, p=np.asarray(p.predict(va), dtype=np.float64), mask=mask)
"""


def proxy(dname, tag):
    """마스킹 2024 프록시 추론. 캐시 접두 `bcold_` 로 다른 실험과 갈랐다."""
    f = CACHE / f"bcold_{tag}.npz"
    if not f.exists():
        wk = CACHE / "_bcold_worker.py"
        wk.write_text(WORKER_MASK.replace("{csv}", str(TRAIN_CSV).replace("\\", "/"))
                                 .replace("{unk}", repr(UNK)),
                      encoding="utf-8", newline="\n")
        r = subprocess.run([sys.executable, str(wk), dname, str(f)],
                           capture_output=True, text=True)
        if not f.exists():
            print(f"  [실패] {dname}: {r.stdout.strip()[-600:]}\n{r.stderr.strip()[-900:]}")
            sys.exit(1)
    z = np.load(f)
    return z["p"], z["mask"]


def main():
    t0 = time.time()
    assert BASE_ZIP.exists(), BASE_ZIP

    # -- REF 준비 · 무결성: 해제본이 정말 hj03 인가 (바이트 대조) ----------
    with zipfile.ZipFile(BASE_ZIP) as z:
        names = z.namelist()
        if not (REF_DIR / "predict.py").exists():
            REF_DIR.mkdir(parents=True, exist_ok=True)
            z.extractall(REF_DIR)
            print(f"REF 해제: {REF_DIR}")
        bad = [n for n in names
               if (REF_DIR / n).is_file() and (REF_DIR / n).read_bytes() != z.read(n)]
        assert not bad, f"REF 가 베이스와 다르다: {bad[:5]}"
    print(f"베이스 {BASE_ZIP.name} · 파일 {len(names)}개 · REF 바이트 일치 OK")

    # -- 패치 팔 준비 ---------------------------------------------------
    if not (NEW_DIR / "predict.py").exists():
        shutil.copytree(REF_DIR, NEW_DIR, dirs_exist_ok=True)
        print(f"NEW 복사: {NEW_DIR}")

    emb_files = sorted((NEW_DIR / "model").glob("mlp_emb4_*.npz"))
    assert len(emb_files) == 5, f"emb4 npz {len(emb_files)}개"
    patched = 0
    for f in emb_files:
        d = dict(np.load(f))
        ecols = [str(c) for c in d["ecols"]]
        assert ecols[1] == "batter_id", f"{f.name}: emb1 이 batter_id 가 아니다 ({ecols[1]})"
        e = d["emb1"]
        assert e.shape[1] == 4, f"{f.name}: emb1 dim {e.shape[1]} (4 가 아니다)"
        # 투수 슬롯이 이미 패치돼 있어야 한다 — 베이스가 hj03 이라는 확인
        e0 = d["emb0"]
        assert np.allclose(e0[0], e0[1:].mean(0), atol=1e-6), \
            f"{f.name}: emb0[0] 가 무게중심이 아니다 — 베이스가 hj03 이 아닌 듯"
        mu = e[1:].mean(0)
        if np.allclose(e[0], mu):
            continue                                  # 이미 패치됨 (재실행)
        n0, nmu = np.linalg.norm(e[0]), np.linalg.norm(mu)
        pct = float((np.linalg.norm(e[1:], axis=1) < n0).mean())
        print(f"  {f.name}: |emb1[0]| {n0:.4f} (학습 노름 {pct:6.2%} 분위) -> |mean| {nmu:.4f}")
        d["emb1"] = np.vstack([mu[None, :], e[1:]]).astype(e.dtype)
        np.savez_compressed(f, **d)
        patched += 1
    print(f"패치 {patched}/5 · emb1[0] <- mean(emb1[1:])"
          f"{' (재사용)' if patched == 0 else ''}")

    # -- 마스킹 프록시 위에서 Δ 측정 --------------------------------------
    p_ref, mask = proxy("HJ03REF", "ref")
    p_new, mask2 = proxy("BCOLD", "new")
    assert np.array_equal(mask, mask2), "두 팔의 마스크가 다르다"
    assert len(p_new) == len(p_ref)
    assert not np.array_equal(p_new, p_ref), "두 팔이 같다 — 캐시가 안 갈렸다"
    print(f"[{time.time()-t0:.0f}s] 프록시 2024 {len(p_ref):,}행 · "
          f"미지 타자 처리 {mask.sum():,} ({mask.mean():.2%})")

    d = p_new - p_ref
    d_unk, d_kn = d[mask], d[~mask]
    dmean = float(d.mean())
    print(f"  미지 행 Δp  평균 {d_unk.mean():+.6g} · 평균|Δp| {np.abs(d_unk).mean():.6g} "
          f"· max|Δp| {np.abs(d_unk).max():.6g}")
    print(f"  알려진 행 Δp  max|Δp| {np.abs(d_kn).max():.3g}   <- 0 이어야 한다 (기전 격리)")
    assert np.abs(d_kn).max() == 0.0, "패치가 알려진 타자 행까지 건드렸다"
    assert np.abs(d_unk).max() > 0.0, "패치가 예측에 반영되지 않았다"
    print(f"  전체 Δmean {dmean:+.6g}  (= 미지 행 몫 x {mask.mean():.2%})")

    # -- 레벨 재보정 (축 분리: refinement 만 LB 에 남긴다) ------------------
    src = (NEW_DIR / "predict.py").read_text(encoding="utf-8").replace(CRLF, LF)
    m = re.search(r"PROB_SHIFT\s*=\s*([0-9.eE+-]+)", src)
    assert m, "PROB_SHIFT 를 못 찾았다"
    shift_old = float(m.group(1))
    shift_new = shift_old - dmean
    (NEW_DIR / "predict.py").write_text(
        src[:m.start(1)] + f"{shift_new:.9f}" + src[m.end(1):],
        encoding="utf-8", newline="\n")
    print(f"  PROB_SHIFT {shift_old:.9f} -> {shift_new:.9f}")

    (CACHE / "bcold_new2.npz").unlink(missing_ok=True)
    p_new2, _ = proxy("BCOLD", "new2")
    d2 = float(p_new2.mean() - p_ref.mean())
    print(f"  재보정 후 Δmean = {d2:+.12f}  {'OK' if abs(d2) < 1e-9 else '주의'}")
    assert abs(d2) < 1e-7, "평균 정렬 실패"

    # -- 출력 zip · 변경 격리 ---------------------------------------------
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(BASE_ZIP) as zin, \
            zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
        for n in names:
            fp = NEW_DIR / n
            zout.writestr(n, fp.read_bytes() if fp.is_file() else zin.read(n))
        diff = sorted(n for n in names if (NEW_DIR / n).is_file()
                      and (NEW_DIR / n).read_bytes() != zin.read(n))
    exp = sorted(["predict.py"] + [f"model/{f.name}" for f in emb_files])
    print(f"\n변경 격리 {len(diff)}/{len(names)}: {diff}")
    assert diff == exp, f"기대 {exp}"
    assert len(OUT.name) <= 30
    print(f"{OUT.name}  {OUT.stat().st_size/1e6:.1f} MB")
    print(f"  SHA-256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
