#!/usr/bin/env python3
"""프로브 zip 빌더 — 챔피언 `build_submit.py`를 안전하게 감싼다.

`wonwoo:src/build_submit.py`를 그대로 돌리면 세 가지가 깨진다. 이 래퍼가 막는다.

1. **`model_dir = SUBMIT / "model"` 이 하드코딩돼 있다** (build_submit.py:99).
   `LGA_MODEL_DIR` 을 봐주지 않으므로, 변형 모델을 `submit/model/` 자리에
   **스테이징**해야 한다.

2. **zip 이 `submit/` 전체를 훑는다** (`SUBMIT.rglob("*")`, build_submit.py:161).
   변형 모델을 `submit/model_a30/` 처럼 옆에 두면 **전부 zip 에 들어간다** —
   용량이 배로 늘고 `predict.py` 는 `./model` 만 보므로 조용히 낭비된다.
   → 변형은 `outputs/probe/` (gitignore 대상) 에 두고 하나씩 스테이징한다.

3. **추적 파일 두 개를 쓴다** — `record_history()` → `submit/history.json`,
   `log_submission()` → `SUBMISSIONS.md`. 둘 다 원우 브랜치 추적 파일이다.
   → 두 함수를 no-op 으로 갈아끼우고, 이력은 hyejin 쪽에 따로 남긴다.

4. **`next_zip_path()` 가 s25/s26/s27 로 자동 번호를 매긴다.**
   원우의 실제 제출 번호와 충돌한다 → 이름을 우리가 정한다.

원우 브랜치 파일은 읽기만 하고 한 글자도 고치지 않는다.

사용법
------
    python tools/build_probe.py --models outputs/probe/model_a40 --name a40_base
    python tools/build_probe.py --models outputs/probe/model_a30 --name a30

근거: DIRECTION.md §6-1 · wonwoo:MODEL.md §7.65
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CHAMPION = Path.home() / "Aimers_9th_champion"
HYEJIN_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = HYEJIN_ROOT / "outputs" / "alpha_probe" / "manifest.jsonl"

MAX_NAME = 30      # 대회 규칙: 제출 파일명 30자 이하


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", type=Path, required=True,
                   help="변형 모델 디렉터리 (챔피언 워크트리 기준 상대경로 가능)")
    p.add_argument("--name", required=True,
                   help="zip 이름의 태그 부분. 최종 파일명은 p_<태그>.zip")
    p.add_argument("--champion", type=Path, default=DEFAULT_CHAMPION)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.champion.resolve()
    src = root / "src"
    if not (src / "build_submit.py").is_file():
        raise SystemExit(f"챔피언 워크트리를 찾을 수 없다: {src}")

    models = args.models if args.models.is_absolute() else (root / args.models)
    if not (models / "artifacts.joblib").is_file():
        raise SystemExit(f"모델 디렉터리가 비어 있다: {models}")

    zip_name = f"p_{args.name}.zip"
    if len(zip_name) > MAX_NAME:
        raise SystemExit(f"파일명 {len(zip_name)}자 > {MAX_NAME}자 제한: {zip_name}")
    zip_path = root / zip_name
    if zip_path.exists():
        raise SystemExit(f"이미 존재한다. 지우거나 다른 --name 을 쓸 것: {zip_path}")

    # ── ① 스테이징: 변형 모델을 하드코딩된 자리로 ──────────────────
    staged = root / "submit" / "model"
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(models, staged)
    alpha = json.loads((staged / "meta.json").read_text(encoding="utf-8")).get("mlp_alpha")
    print(f"스테이징: {models} → submit/model   (meta.json 의 mlp_alpha={alpha})\n")

    # ── ② 변형 디렉터리가 submit/ 안에 남아 있으면 zip 이 통째로 삼킨다 ──
    strays = [p for p in (root / "submit").iterdir()
              if p.is_dir() and p.name.startswith("model") and p.name != "model"]
    if strays:
        raise SystemExit(
            "submit/ 안에 다른 모델 디렉터리가 있다. zip 에 전부 들어간다:\n  "
            + "\n  ".join(str(p) for p in strays)
            + "\n→ outputs/probe/ 로 옮길 것."
        )

    sys.path.insert(0, str(src))
    os.chdir(root)
    import build_submit

    # ── ③ 추적 파일에 쓰는 두 함수를 무력화한다 ──────────────────
    build_submit.record_history = lambda *a, **k: print(
        "  이력 기록 건너뜀 (원우 submit/history.json 을 보호한다)")
    build_submit.log_submission = lambda *a, **k: print(
        "  SUBMISSIONS.md 기록 건너뜀 (원우 추적 파일을 보호한다)")

    # ── ④ 파일명은 우리가 정한다 (s25.. 자동 번호와 충돌 방지) ──────
    build_submit.next_zip_path = lambda meta, suffix="": zip_path

    build_submit.main()

    # ── 사후 검증 ────────────────────────────────────────────────
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    n_lgbm = sum(n.startswith("model/lgbm_") for n in names)
    n_mlp = sum(n.startswith("model/mlp_") for n in names)
    size_mb = zip_path.stat().st_size / 1e6
    problems = []
    if n_lgbm != 20:
        problems.append(f"LGBM 부스터 {n_lgbm}개 (기대 20)")
    if n_mlp != 8:
        problems.append(f"MLP 멤버 {n_mlp}개 (기대 8)")
    if any(n.startswith("model_") or "/model_" in n for n in names):
        problems.append("다른 변형 모델이 zip 에 섞였다")

    print(f"\n=== 검증 ===")
    print(f"  파일명    {zip_path.name}  ({len(zip_path.name)}자 / {MAX_NAME}자)")
    print(f"  크기      {size_mb:.1f} MB")
    print(f"  구성원    {len(names)}개  (lgbm {n_lgbm} · mlp {n_mlp})")
    print(f"  mlp_alpha {alpha}")
    if problems:
        print("  ❌ " + "\n  ❌ ".join(problems))
        raise SystemExit("zip 이 기대와 다르다. 제출하지 말 것.")
    print("  ✅ 이상 없음")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "zip": str(zip_path), "name": zip_name, "mlp_alpha": alpha,
            "models": str(models), "size_mb": round(size_mb, 2),
            "n_lgbm": n_lgbm, "n_mlp": n_mlp, "lb_score": None,
        }, ensure_ascii=False) + "\n")
    print(f"  기록      {MANIFEST}  (lb_score 는 결과 받은 뒤 직접 채울 것)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
