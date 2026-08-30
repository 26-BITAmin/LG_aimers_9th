"""§7.203 — 콜드스타트 패치를 **우리 dim8 챔피언(`sy02_c4`)** 위에 얹는다.

혜진 라운드 18 이 `hj00_dim8_control`(미제출) 위에서 이 패치로 팀 신기록
1131.4503 을 냈다. 그 빌드의 한계는 §18-5 가 스스로 적어둔 대로 **베이스의 LB 를
모른다**는 것이다 — 순효과가 [+3.32, +4.29] 구간으로만 말해진다.

우리는 **LB 를 아는 dim8**(`sy02_c4` = 1127.1557560138, 신영 출하본)을 갖고 있다.
같은 패치를 그 위에 얹으면 그 구간이 점추정이 된다.

기전 (실측, `sy02_c4` 의 emb4 npz 5개)
    학습된 투수 792명의 중심   |mean(emb0[1:])| = 0.073 ~ 0.153   <- 사실상 원점
    미지 슬롯                  |emb0[0]|        = 1.29 / 2.40 / 3.68 / 3.90 / 2.17
                               학습된 노름 분포의 1.4 ~ 96 백분위를 시드마다 배회
    -> `index 0` 은 "평균 투수" 가 아니라 **시드마다 다른 임의의 한 투수**다.
      학습이 한 번도 안 건드려 초기값이 남았고, 초기화 스케일이 학습 스케일과
      비슷해서 *진짜 투수처럼 생긴 가짜* 가 됐다. test 행의 약 20% 가 거기로 간다.

미지 비율 감사 (`data/train.csv` 자체 실측 — 물려받지 않았다)
    직전 전체 이력 대비 신규 투수 행 비율: 2020 22.4% · 2021 20.6% · 2022 15.7%
                                           2023 13.8% · **2024 19.9%**
    2025 의 구조적 유사물은 2024 다 (5시즌 이력 -> 다음 시즌). -> UNK = 0.20 채택.

⚠️ 왜 마스킹 프록시인가
    2024 프록시에는 미지 투수가 **0명**이다 (학습에 2024 가 들어갔다). 그대로 재면
    Δmean = 0 이라 레벨 재보정이 무의미해진다. 그래서 투수를 단위로 골라
    `pitcher_id` 를 LUT 밖 값(-1)으로 바꿔 실제 test 의 미지 비율을 흉내낸다.
    조작은 프록시 입력에만 하고 산출물에는 안 들어간다.

    ⚠️ 한계: 마스킹은 MLP 뿐 아니라 트리 채널에도 `pitcher_id = -1` 을 보인다.
    Δ 측정에서는 두 팔이 같은 마스크를 쓰므로 **상쇄**되지만, 미지 행의 기저 p 가
    실제와 조금 달라 dp/dz 를 통해 2차 오차로 남는다. 레벨 항 규모(1e-4)에서
    무시 가능하고, 혜진 빌드와 같은 방식이라 챔피언 교정과 정합적이기도 하다.

실행:  .venv/Scripts/python.exe -u src/coldbuild.py > experiments/coldbuild.log 2>&1
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
BASE_ZIP = ROOT / "sy02_c4.zip"      # LB 1127.1557560138 (신영 C-4, dim 8)
OUT = ROOT / "sy04_cold.zip"
REF_DIR = ASCII / "SY02REF"          # 챔피언 원본 (대조군, 패치 전)
NEW_DIR = ASCII / "COLD"             # 패치 팔
UNK = 0.20                           # 프록시 미지 행 목표 비율
CACHE = ROOT / "experiments" / "cache"

# 마스킹 프록시 워커 — t1core.WORKER 에 투수 마스킹만 더했다.
WORKER_MASK = r"""
import os, sys, numpy as np, pandas as pd
arm, out = sys.argv[1], sys.argv[2]
d = r"D:/t1eval/" + arm
sys.path.insert(0, d)          # 우리 src 를 넣지 않는다 (§7.174 사고)
va = pd.read_csv(r"{csv}")
va = va[va.season == 2024].reset_index(drop=True)
# --- 미지 투수 마스킹: 투수 단위로 골라 목표 행 비율을 맞춘다 ---
rng = np.random.RandomState(0)
pids = rng.permutation(np.sort(va["pitcher_id"].unique()))
cnt = va["pitcher_id"].value_counts()
take, tot, goal = [], 0, {unk} * len(va)
for pid in pids:
    if tot >= goal:
        break
    take.append(pid); tot += int(cnt[pid])
mask = va["pitcher_id"].isin(take).to_numpy()
va.loc[mask, "pitcher_id"] = -1
from predict import Predictor
import predict as _pm
assert _pm.__file__.replace("\\", "/").startswith(d), "T1 predict.py 가 아니다"
p = Predictor(model_dir=d + "/model")
np.savez_compressed(out, p=np.asarray(p.predict(va), dtype=np.float64), mask=mask)
"""


def proxy(dname, tag):
    """마스킹 2024 프록시 추론. 캐시 접두 `cold_` 로 다른 실험과 갈랐다."""
    f = CACHE / f"cold_{tag}.npz"
    if not f.exists():
        wk = CACHE / "_cold_worker.py"
        wk.write_text(WORKER_MASK.replace("{csv}", str(TRAIN_CSV).replace("\\", "/"))
                                 .replace("{unk}", repr(UNK)),
                      encoding="utf-8", newline="\n")
        r = subprocess.run([sys.executable, str(wk), dname, str(f)],
                           capture_output=True, text=True)
        if not f.exists():
            print(f"  [실패] {dname}: {r.stdout.strip()[-600:]}\n{r.stderr.strip()[-800:]}")
            sys.exit(1)
    z = np.load(f)
    return z["p"], z["mask"]


def main():
    t0 = time.time()
    assert BASE_ZIP.exists(), BASE_ZIP

    # -- REF 무결성: 해제본이 정말 sy02_c4 인가 (바이트 대조) --------------
    with zipfile.ZipFile(BASE_ZIP) as z:
        names = z.namelist()
        assert (REF_DIR / "predict.py").exists(), f"{REF_DIR} 해제본이 없다"
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
        e = d["emb0"]
        assert [str(c) for c in d["ecols"]][0] == "pitcher_id", f"{f.name}: emb0 가 pitcher_id 가 아니다"
        assert e.shape[1] == 8, f"{f.name}: dim {e.shape[1]} (8 이 아니다)"
        mu = e[1:].mean(0)
        if np.allclose(e[0], mu):
            continue                                  # 이미 패치됨 (재실행)
        n0, nmu = np.linalg.norm(e[0]), np.linalg.norm(mu)
        pct = float((np.linalg.norm(e[1:], axis=1) < n0).mean())
        print(f"  {f.name}: |emb0[0]| {n0:.4f} (학습 노름 {pct:6.2%} 분위) -> |mean| {nmu:.4f}")
        d["emb0"] = np.vstack([mu[None, :], e[1:]]).astype(e.dtype)
        np.savez_compressed(f, **d)
        patched += 1
    print(f"패치 {patched}/5 · emb0[0] <- mean(emb0[1:])"
          f"{' (재사용)' if patched == 0 else ''}")

    # -- 마스킹 프록시 위에서 Δ 측정 --------------------------------------
    p_ref, mask = proxy("SY02REF", "ref")
    p_new, mask2 = proxy("COLD", "new")
    assert np.array_equal(mask, mask2), "두 팔의 마스크가 다르다"
    assert len(p_new) == len(p_ref)
    assert not np.array_equal(p_new, p_ref), "두 팔이 같다 — 캐시가 안 갈렸다"
    print(f"[{time.time()-t0:.0f}s] 프록시 2024 {len(p_ref):,}행 · "
          f"미지 처리 {mask.sum():,} ({mask.mean():.2%})")

    d = p_new - p_ref
    d_unk, d_kn = d[mask], d[~mask]
    dmean = float(d.mean())
    print(f"  미지 행 Δp  평균 {d_unk.mean():+.6g} · 평균|Δp| {np.abs(d_unk).mean():.6g} "
          f"· max|Δp| {np.abs(d_unk).max():.6g}")
    print(f"  알려진 행 Δp  max|Δp| {np.abs(d_kn).max():.3g}   <- 0 이어야 한다 (기전 격리)")
    assert np.abs(d_kn).max() == 0.0, "패치가 알려진 투수 행까지 건드렸다"
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

    (CACHE / "cold_new2.npz").unlink(missing_ok=True)
    p_new2, _ = proxy("COLD", "new2")
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
