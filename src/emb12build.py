"""§7.201 — emb_dim 사다리 다음 칸: 챔피언 `sy02_c4` 위에서 `pitcher_id` 임베딩 **8→12**

무엇을 바꾸나 (축 하나)
    `mlp_emb4_*.npz` 5개를 `emb_dim.pitcher_id = 12` 로 재학습해 갈아 끼운다.
    나머지 77개 파일(LGBM 20 · CatBoost 12 · mlp_noid 3 · artifacts · team_reference …)
    은 챔피언 zip 에서 **바이트 그대로** 옮긴다. §7.197 이 확인했듯 서빙 경로는
    `meta.json` 이 아니라 npz 의 텐서 shape 를 읽으므로 다른 파일은 손댈 필요가 없다.

왜 우리 파이프라인으로 학습해도 되나 (사전 확인 — `experiments/emb12_repro.log`)
    배포 emb4 멤버는 우리 `train.py` 산출물이 아니다 (T1 `eb7ee215…` vs 우리 `627256…`).
    그래서 `LGA_MODEL_DIR=t1work/model` 로 §7.167 재현 관문을 먼저 돌렸다:

        emb4_42 재학습 vs 배포 npz    max|Δw| 1.073e-06 · max|Δz| 6.56e-07
                                      sd(z) 0.1956 → **비 0.00%**  [사전 등록 임계 1%]

    비트는 갈렸지만(GPU/torch 누적) 판정 기준은 예측 차이이고 임계의 1/1000 이다.
    → 우리 학습 함수 = 배포 emb4 를 만든 함수. dim 12 멤버는 챔피언의 dim 8 멤버와
      **같은 함수의 draw** 다. (train.py 의 관문 자체는 shape 이 달라 여기선 끈다.)

레벨 (§7.178 ②·§7.197 ④-2 계승)
    MLP 가 바뀌면 `p̄` 가 움직인다. 2024 프록시에서 챔피언과 평균을 소수 9자리까지
    맞추도록 `PROB_SHIFT` 를 재유도한다 → LB 차이가 **레벨 항이 아니라 emb_dim 축**
    에서만 오게 만든다. (§7.197 ④-2 의 부모 대비 −5.80e-4 표류는 **일부러 안 고친다**
    — 여기서 같이 고치면 두 변경이 교락된다.)

⚠️ 판정 한계 — 사전 등록
    §7.198·§7.199: MLP 층 재학습에 붙는 SD(Δ) 는 4.8~7.1 BSS 다. 8→12 의 기대 효과가
    4→8(+7.44) 보다 작다면 LB 한 점으로는 **부호도 판정되지 않는다.** 이 빌드는
    "판정" 이 아니라 사다리 다음 점의 **측정**이다.

실행:  .venv-gpu/Scripts/python.exe -u src/emb12build.py 2>&1 | tee experiments/emb12build.log
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
ASCII = Path("D:/t1eval")            # LightGBM 이 비ASCII 경로에서 죽는다 (§7.173 ③)
REF_DIR = ASCII / "SY02REF"          # 챔피언 원본 (대조군)
NEW_DIR = ASCII / "EMB12"            # 신규 팔
BASE_ZIP = ROOT / "sy02_c4.zip"      # LB 1127.15
OUT = ROOT / "sy03_emb12.zip"
CACHE = ROOT / "experiments" / "cache"

NEW_DIM = 12
MLP_SEEDS_EXP = ("42", "7", "2024", "1234", "31337")
EMB4_FILES = [f"mlp_emb4_{s}.npz" for s in MLP_SEEDS_EXP]
NOID_FILES = [f"mlp_noid_{s}.npz" for s in ("42", "7", "2024")]

# 'config' 를 import 하기 **전에** 세팅해야 MODEL_DIR 이 잡힌다.
os.environ["LGA_MODEL_DIR"] = str(NEW_DIR / "model")
os.environ["LGA_MLP_ONLY"] = "1"
os.environ["LGA_MLP_REPRO"] = "0"     # dim 이 달라 비트 대조 불가 (위 로그로 대체)

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import joblib
import numpy as np

from config import TRAIN_CSV
from t1core import WORKER

CRLF, LF = chr(13) + chr(10), chr(10)
OK, BAD = "OK", "MISMATCH"


def emb0_dim(p):
    with np.load(p) as z:
        return int(z["emb0"].shape[1])


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


# ────────────────────────────────────────────────────────────── 1. 팔 준비
def stage_unpack():
    if not (REF_DIR / "predict.py").exists():
        REF_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(BASE_ZIP) as z:
            z.extractall(REF_DIR)
        print(f"REF 해제: {REF_DIR}")
    else:
        print(f"REF 재사용: {REF_DIR}")

    if not (NEW_DIR / "predict.py").exists():
        shutil.copytree(REF_DIR, NEW_DIR, dirs_exist_ok=True)
        print(f"NEW 복사: {NEW_DIR}")

    md = NEW_DIR / "model"
    for f in EMB4_FILES:
        p = md / f
        if p.exists() and emb0_dim(p) != NEW_DIM:
            p.unlink()                       # dim 8 원본 → 지우고 다시 뽑는다
    have = [f for f in EMB4_FILES if (md / f).exists()]
    print(f"  emb4 보유 {len(have)}/5 (dim {NEW_DIM}) · 학습 대상 {5 - len(have)}개")

    # 대조군 무결성 — REF 는 챔피언 그대로여야 한다
    for f in EMB4_FILES:
        assert emb0_dim(REF_DIR / "model" / f) == 8, f"REF {f} 가 dim 8 이 아니다"


# ────────────────────────────────────────────────────────────── 2. 학습
def stage_train(t0):
    md = NEW_DIR / "model"
    todo = [f for f in EMB4_FILES if not (md / f).exists()]
    if todo:
        import config
        cfg, nseed = config.MLP_VARIANTS["emb4"]
        cfg = dict(cfg)
        cfg["emb_dim"] = dict(cfg["emb_dim"], pitcher_id=NEW_DIM)
        config.MLP_VARIANTS["emb4"] = (cfg, nseed)      # train 이 같은 객체를 본다
        print(f"\nMLP_VARIANTS.emb4.emb_dim.pitcher_id = {NEW_DIM} (학습 {len(todo)}개)")
        import train
        assert train.MLP_VARIANTS["emb4"][0]["emb_dim"]["pitcher_id"] == NEW_DIM
        train.main()
        print(f"[{time.time() - t0:.0f}s] 학습 완료")

    # ── 검산: shape · 서빙 인코더 정합 · 나머지 파일 불변 ──
    bundle = joblib.load(md / "artifacts.joblib")
    enc = bundle["mlp_enc"]
    n_num, sizes = len(enc["num_cols"]), list(enc["sizes"])
    print(f"\n서빙 인코더: 수치 {n_num}개 · 범주 크기 {sizes}")
    for f in EMB4_FILES:
        with np.load(md / f) as z:
            k = len(z["ecols"])
            dims = [z[f"emb{i}"].shape[1] for i in range(k)]
            rows = [z[f"emb{i}"].shape[0] for i in range(k)]
            assert dims == [NEW_DIM] + [4] * 6, f"{f} dims={dims}"
            assert rows == sizes, f"{f} 임베딩 행 {rows} != 인코더 {sizes}"
            assert z["W0"].shape[0] == n_num + sum(dims), \
                f"{f} W0 {z['W0'].shape} != {n_num}+{sum(dims)}"
            assert np.isfinite(z["W0"]).all() and np.isfinite(z["emb0"]).all()
            print(f"  {f}  emb0{z['emb0'].shape} W0{z['W0'].shape} "
                  f"|w|={np.abs(z['emb0']).mean():.3f} md5={md5(md / f)[:8]}")
    assert len({md5(md / f) for f in EMB4_FILES}) == 5, "멤버가 서로 같다 — 시드 미반영"
    for f in NOID_FILES + ["artifacts.joblib"]:
        assert md5(md / f) == md5(REF_DIR / "model" / f), f"{f} 가 REF 와 다르다"
    print("  [OK] noid 3개 · artifacts 바이트 동일 (emb4 만 바뀌었다)")


# ────────────────────────────────────────────────────────────── 3. 레벨
def proxy(arm, tag, force=False):
    """2024 전 행을 해당 디렉터리의 predict.py 로 예측 (t1core WORKER 재사용)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"emb12_{tag}.npz"
    if force:
        f.unlink(missing_ok=True)
    if not f.exists():
        wk = CACHE / "_emb12_worker.py"
        wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/")),
                      encoding="utf-8", newline="\n")
        t = time.time()
        # ⚠️ 학습은 `.venv-gpu`(torch)에서 돌지만 추론은 catboost/lightgbm 이 있는
        #    `.venv` 여야 한다. sys.executable 을 그대로 쓰면 catboost 가 없어 죽는다.
        py = ROOT / ".venv" / "Scripts" / "python.exe"
        r = subprocess.run([str(py if py.exists() else sys.executable),
                            str(wk), arm, str(f)], capture_output=True, text=True)
        if not f.exists():
            print(f"  [FAIL] {arm}: {r.stderr.strip()[-500:]}")
            sys.exit(1)
        print(f"  [{arm}] 추론 {time.time() - t:.0f}s (train.csv 로드 포함)")
    return np.load(f)["p"]


def stage_level(t0):
    p_ref = proxy("SY02REF", "ref")
    p_new = proxy("EMB12", "new")
    assert len(p_new) == len(p_ref)
    d = p_new - p_ref
    dmean = float(d.mean())
    print(f"\n[{time.time() - t0:.0f}s] 프록시 2024 {len(p_ref):,}행")
    print(f"  corr(p_ref, p_new) = {np.corrcoef(p_ref, p_new)[0, 1]:.6f}  "
          f"mean|dp| = {np.abs(d).mean():.6f}  max|dp| = {np.abs(d).max():.6f}")
    print(f"  Var(p): {p_ref.var():.8f} -> {p_new.var():.8f}   "
          f"dmean = {dmean:+.9f}  (방치 시 레벨 비용 {1e5 * dmean ** 2 / 0.2497:.4f}점)")
    assert np.abs(d).max() > 0, "두 팔이 같다 — npz 교체가 반영되지 않았다"

    src = (NEW_DIR / "predict.py").read_text(encoding="utf-8").replace(CRLF, LF)
    assert len(re.findall(r"PROB_SHIFT\s*=\s*[0-9.eE+-]+", src)) == 1
    m = re.search(r"PROB_SHIFT\s*=\s*([0-9.eE+-]+)", src)
    old = float(m.group(1))
    new = old - dmean
    (NEW_DIR / "predict.py").write_text(
        src[:m.start(1)] + f"{new:.9f}" + src[m.end(1):],
        encoding="utf-8", newline="\n")
    print(f"  PROB_SHIFT {old:.9f} -> {new:.9f}  (챔피언 레벨에 고정)")

    p2 = proxy("EMB12", "new2", force=True)
    d2 = float(p2.mean() - p_ref.mean())
    print(f"  재보정 후 dmean = {d2:+.12f}  [{OK if abs(d2) < 1e-9 else BAD}]")
    assert abs(d2) < 1e-7, "평균 정렬 실패"
    return p_ref, p2


# ────────────────────────────────────────────────────────────── 4. 패키징
def stage_package():
    md = NEW_DIR / "model"
    final_predict = (NEW_DIR / "predict.py").read_text(encoding="utf-8").replace(CRLF, LF)

    with zipfile.ZipFile(BASE_ZIP) as zin:
        names = zin.namelist()
        meta = json.loads(zin.read("model/meta.json"))
        # §7.193·§7.197 ④-3 — meta.json 이 거짓말하지 않게 한다 (서빙엔 무영향)
        meta["mlp_variants"]["emb4"][0]["emb_dim"]["pitcher_id"] = NEW_DIM
        meta["build_note"] = ("sy03_emb12 = sy02_c4 + emb4 pitcher_id emb_dim 8->12 "
                              "(5 members retrained); PROB_SHIFT re-derived to hold "
                              "the sy02_c4 level on the 2024 proxy")
        meta_txt = json.dumps(meta, indent=2, ensure_ascii=False)

        OUT.unlink(missing_ok=True)
        swapped = 0
        with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
            for n in names:
                if n == "predict.py":
                    zout.writestr(n, final_predict)
                    swapped += 1
                elif n == "model/meta.json":
                    zout.writestr(n, meta_txt)
                    swapped += 1
                elif n.startswith("model/mlp_emb4_"):
                    zout.writestr(n, (md / Path(n).name).read_bytes())
                    swapped += 1
                else:
                    zout.writestr(n, zin.read(n))
    assert swapped == 7, f"교체 {swapped}개 (기대 7 = emb4 5 + predict + meta)"
    assert len(OUT.name) <= 30
    print(f"\n{OUT.name}  {OUT.stat().st_size / 1e6:.1f} MB  "
          f"(파일 {len(names)}개 · 교체 {swapped}개)")
    return names


# ────────────────────────────────────────────────────────────── 5. 규정 검사
def stage_scan():
    from build_submit import leakage_scan
    tmp = CACHE / "_emb12_scan"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(OUT) as z:
        pys = [n for n in z.namelist() if n.endswith(".py")]
        for n in pys:
            d = tmp / n
            d.parent.mkdir(parents=True, exist_ok=True)
            d.write_bytes(z.read(n))
        meta_txt = z.read("model/meta.json").decode("utf-8")
    hits = leakage_scan([tmp / n for n in pys])
    print(f"AST 금지 호출 검사: .py {len(pys)}개 -> {len(hits)}건 "
          + (OK if not hits else f"{BAD} {hits[:5]}"))
    assert not hits
    assert '"pitcher_id": 12' in meta_txt
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    t0 = time.time()
    assert BASE_ZIP.exists()
    stage_unpack()
    stage_train(t0)
    stage_level(t0)
    stage_package()
    stage_scan()
    print(f"\n총 {time.time() - t0:.0f}s — 제출 후보 {OUT}")
    print("기준: sy02_c4 = LB 1127.15. 이 빌드 Δ 에 붙는 잡음 SD ~ 4.8~7.1 (§7.199).")


if __name__ == "__main__":
    main()
