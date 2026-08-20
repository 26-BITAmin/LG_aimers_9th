#!/usr/bin/env python3
"""제출 zip 독립 검증 — **압축을 푼 상태에서** 평가 서버를 흉내낸다.

왜 별도로 필요한가
------------------
`wonwoo:src/build_submit.py`는 `script.py`를 **`submit/` 작업 디렉터리에서**
실행한다 (`subprocess.run(..., cwd=SUBMIT)`). 그래서 **`submit/`에는 있는데
zip 에는 안 들어간 파일**이 있어도 빌드 검사를 통과한다 — 그리고 평가 서버에서만
터진다. 이 스크립트는 zip 을 **빈 임시 디렉터리에 풀어서** 실행하므로 그 구멍을 막는다.

(hyejin `reports/07b` 필수 검사 10번 — "제출 전에 평가 스크립트를 별도 임시
 디렉터리에서 end-to-end 실행한다" — 을 코드로 옮긴 것이다.)

검사 항목
---------
  1. zip 무결성 (CRC)
  2. 구조     최상위 래퍼 폴더 없음 · `model/` `script.py` `requirements.txt` 존재
              · `__MACOSX` / `.DS_Store` 등 숨은 파일 없음 · 파일명 30자 이하
  3. 의존성   requirements.txt 의 패키지가 전부 import 가능한가
  4. 실행     빈 임시 디렉터리에 풀고 `script.py` 를 subprocess 로 실행
  5. 출력     `output/submission.csv` 의 컬럼 · 행 수 · `row_id` 정확 일치
              · [0,1] 범위 · NaN 없음
  6. 행 독립  통짜 vs 셔플 vs 청크 분할 예측이 **비트 단위로 같은가**
              (다르면 test 전체 통계를 쓰고 있다는 뜻 = 대회 규정 위반)

사용법
------
    python tools/verify_zip.py ~/Aimers_9th_champion/p_a40.zip
    python tools/verify_zip.py <zip> --rows 30000     # 기본 20000
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CHAMPION = Path.home() / "Aimers_9th_champion"
REQUIRED = ["script.py", "requirements.txt"]
HIDDEN_MARKERS = ("__MACOSX", ".DS_Store", "Thumbs.db", ".ipynb_checkpoints")
EVAL_ROWS = 245_789          # 실제 평가 데이터 행 수
TIME_LIMIT = 600.0           # 추론 10분


class Fail(Exception):
    pass


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠️  {msg}")


# ── 1~3. 정적 검사 ────────────────────────────────────────────────────
def check_static(zip_path: Path) -> list[str]:
    print("=== 1. zip 무결성 ===")
    if len(zip_path.name) > 30:
        raise Fail(f"파일명 {len(zip_path.name)}자 > 30자 제한: {zip_path.name}")
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            raise Fail(f"CRC 불일치: {bad}")
        names = z.namelist()
    ok(f"CRC 정상 · 구성원 {len(names)}개 · 파일명 {len(zip_path.name)}자")

    print("\n=== 2. 구조 ===")
    for marker in HIDDEN_MARKERS:
        hits = [n for n in names if marker in n]
        if hits:
            raise Fail(f"숨은 파일이 들어 있다 ({marker}): {hits[:5]}")
    ok("숨은 macOS/에디터 파일 없음")

    tops = {n.split("/", 1)[0] for n in names}
    if len(tops) == 1 and not any(n in tops for n in REQUIRED):
        raise Fail(f"최상위에 래퍼 폴더가 하나 있다: {tops}. "
                   "평가 서버는 zip 루트에서 script.py 를 찾는다.")
    for req in REQUIRED:
        if req not in names:
            raise Fail(f"루트에 {req} 가 없다")
    if not any(n.startswith("model/") for n in names):
        raise Fail("model/ 디렉터리가 없다")
    ok(f"루트 구성 정상: {sorted(tops)}")

    n_lgbm = sum(n.startswith("model/lgbm_") for n in names)
    n_mlp = sum(n.startswith("model/mlp_") for n in names)
    ok(f"모델 파일: lgbm {n_lgbm}개 · mlp {n_mlp}개")
    return names


def check_requirements(extracted: Path) -> None:
    print("\n=== 3. 의존성 ===")
    text = (extracted / "requirements.txt").read_text(encoding="utf-8")
    pkgs = [ln.split("==")[0].strip() for ln in text.splitlines()
            if ln.strip() and not ln.startswith("#")]
    alias = {"scikit-learn": "sklearn", "lightgbm": "lightgbm", "pyarrow": "pyarrow"}
    import importlib
    for p in pkgs:
        mod = alias.get(p, p)
        try:
            importlib.import_module(mod)
        except Exception as e:                       # noqa: BLE001
            raise Fail(f"requirements 의 {p} 를 import 할 수 없다: {e}")
    ok(f"{len(pkgs)}개 패키지 import 가능: {', '.join(pkgs)}")
    if "torch" in pkgs:
        raise Fail("requirements 에 torch 가 있다. 추론은 numpy 전용이어야 한다")
    ok("torch 미포함 (추론은 mlp_infer.py 의 numpy 순전파)")


# ── 4~6. 실행 검사 ────────────────────────────────────────────────────
def make_dummy(champion: Path, n: int, seed: int = 0) -> pd.DataFrame:
    """실제 train.csv 에서 표집해 평가 test.csv 의 질감을 흉내낸다."""
    sys.path.insert(0, str(champion / "src"))
    import build_submit                                   # noqa: PLC0415
    return build_submit.make_dummy_test(n)


def run_script(work: Path, test_df: pd.DataFrame, python: str) -> tuple[pd.DataFrame, float]:
    data = work / "data"
    data.mkdir(exist_ok=True)
    test_df.to_csv(data / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame({"row_id": test_df["row_id"], "control_success": 0.5}).to_csv(
        data / "sample_submission.csv", index=False, encoding="utf-8")
    shutil.rmtree(work / "output", ignore_errors=True)

    t0 = time.time()
    r = subprocess.run([python, "script.py"], cwd=work,
                       capture_output=True, text=True, encoding="utf-8")
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-4000:])
        raise Fail(f"script.py 종료코드 {r.returncode}")
    out_path = work / "output" / "submission.csv"
    if not out_path.is_file():
        raise Fail("output/submission.csv 가 생성되지 않았다")
    return pd.read_csv(out_path), elapsed


def check_output(out: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\n=== 5. 출력 형식 ===")
    if list(out.columns) != ["row_id", "control_success"]:
        raise Fail(f"컬럼이 다르다: {list(out.columns)} (기대 ['row_id','control_success'])")
    ok("컬럼 ['row_id', 'control_success']")
    if len(out) != len(test_df):
        raise Fail(f"행 수 {len(out)} != test {len(test_df)}")
    ok(f"행 수 {len(out):,}")
    if not out["row_id"].equals(test_df["row_id"].reset_index(drop=True)):
        if set(out["row_id"]) != set(test_df["row_id"]):
            raise Fail("row_id 집합이 test.csv 와 다르다")
        raise Fail("row_id 순서가 sample_submission 과 다르다")
    ok("row_id 가 sample_submission 과 정확히 일치 (순서 포함)")
    p = out["control_success"]
    if p.isna().any():
        raise Fail(f"NaN {int(p.isna().sum())}개")
    if not p.between(0, 1).all():
        raise Fail(f"[0,1] 범위 밖: min={p.min()} max={p.max()}")
    ok(f"확률 범위 정상  min={p.min():.6f}  mean={p.mean():.6f}  max={p.max():.6f}")
    if p.nunique() < 100:
        warn(f"고유 예측값이 {p.nunique()}개뿐이다 — 상수 예측인지 확인할 것")


def check_row_independence(work: Path, test_df: pd.DataFrame, base: pd.DataFrame,
                           python: str) -> None:
    print("\n=== 6. 행 독립성 (대회 규정) ===")
    ref = base.set_index("row_id")["control_success"]

    shuffled = test_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out_s, _ = run_script(work, shuffled, python)
    d = (out_s.set_index("row_id")["control_success"] - ref).abs().max()
    if d != 0.0:
        raise Fail(f"셔플 후 예측이 달라진다 (최대 {d:.3e}) — test 전체 통계 사용 의심")
    ok(f"셔플 불변  최대 차이 {d:.1e}")

    half = len(test_df) // 2
    parts = []
    for chunk in (test_df.iloc[:half], test_df.iloc[half:]):
        out_c, _ = run_script(work, chunk.reset_index(drop=True), python)
        parts.append(out_c)
    merged = pd.concat(parts).set_index("row_id")["control_success"]
    d = (merged - ref).abs().max()
    if d != 0.0:
        raise Fail(f"청크 분할 예측이 달라진다 (최대 {d:.3e}) — 규정 위반")
    ok(f"청크 분할 불변  최대 차이 {d:.1e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zip", type=Path)
    ap.add_argument("--rows", type=int, default=20_000,
                    help="검증용 더미 행 수 (기본 20000). 실제 평가는 245,789행")
    ap.add_argument("--champion", type=Path, default=DEFAULT_CHAMPION)
    ap.add_argument("--python", default=None,
                    help="script.py 를 실행할 인터프리터 (기본: 챔피언 venv)")
    args = ap.parse_args()

    zip_path = args.zip.resolve()
    champion = args.champion.resolve()
    python = args.python or str(champion / ".venv" / "bin" / "python")
    print(f"검증 대상: {zip_path.name}  ({zip_path.stat().st_size/1e6:.1f} MB)\n")

    try:
        check_static(zip_path)
        test_df = make_dummy(champion, args.rows)

        with tempfile.TemporaryDirectory(prefix="verify_zip_") as tmp:
            work = Path(tmp) / "extracted"
            work.mkdir()
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(work)
            check_requirements(work)

            print(f"\n=== 4. 격리 실행 ({args.rows:,}행) ===")
            print(f"  작업 디렉터리: {work}")
            base, elapsed = run_script(work, test_df, python)
            per_row = elapsed / len(test_df)
            projected = per_row * EVAL_ROWS
            ok(f"정상 종료  {elapsed:.1f}s  →  245,789행 환산 약 {projected:.0f}s "
               f"/ {TIME_LIMIT:.0f}s ({projected/TIME_LIMIT*100:.0f}%)")
            if projected > TIME_LIMIT * 0.8:
                warn("추론 시간 여유가 20% 미만이다")

            check_output(base, test_df)
            check_row_independence(work, test_df, base, python)
    except Fail as e:
        print(f"\n❌ 검증 실패: {e}")
        print("   이 zip 은 제출하지 말 것.")
        return 1

    print(f"\n{'='*54}\n✅ {zip_path.name} — 모든 검사 통과. 제출 가능.\n{'='*54}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
