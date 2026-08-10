"""제출 zip 빌드 + 리키지 정적 검사 + 추론 시간 측정.

실행:  .venv/Scripts/python.exe -u src/build_submit.py
산출:  submit.zip  (model/, script.py, features.py, predict.py, requirements.txt)
"""
import ast
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# 콘솔 인코딩이 UTF-8이 아니면(예: Windows CP949) print 한 줄에 프로세스가 죽는다.
# 평가 서버의 로케일을 알 수 없으므로 방어적으로 고정한다.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

from config import ROOT, TEST_CSV, TRAIN_CSV

SRC = ROOT / "src"
SUBMIT = ROOT / "submit"

# src → submit 으로 복사되는 파일 (단일 진실 소스는 항상 src/)
# mlp_infer.py는 **numpy 전용** 순전파다 — torch는 제출물에 들어가지 않는다.
COPY_FILES = ["features.py", "predict.py", "mlp_infer.py"]

# 추론 코드에서 test 데이터프레임에 대해 호출되면 안 되는 것들 (CLAUDE.md §3.2)
FORBIDDEN_CALLS = {
    "groupby", "value_counts", "rolling", "expanding", "cumsum", "cumcount",
    "fit", "fit_transform", "nlargest", "nsmallest", "rank", "duplicated",
}
# 분포 통계 함수도 금지 (test 전체를 보고 계산하는 값)
FORBIDDEN_ATTRS = {"quantile", "describe", "corr", "cov", "mode"}

REQUIREMENTS = """\
lightgbm==4.6.0
numpy==2.2.6
pandas==2.3.3
scipy==1.15.2
joblib==1.4.2
scikit-learn==1.6.1
"""


def leakage_scan(paths):
    """AST로 금지 호출을 찾는다. 문자열 grep이 아니라 실제 호출 노드를 본다."""
    hits = []
    for p in paths:
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_CALLS | FORBIDDEN_ATTRS:
                    hits.append((p.name, node.lineno, node.func.attr))
    return hits


def make_dummy_test(n=245789):
    """평가 규모(245,789행)의 더미 test.

    2024 시즌 행을 쓴다. 2025 test의 피처 분포에 가장 가까우므로
    추론 **시간**뿐 아니라 예측 **평균**도 의미 있는 검사가 된다
    (logit_delta를 2024 행 기준으로 풀었으니 평균이 TARGET_PRIOR_2025 근처여야 한다).
    """
    df = pd.read_csv(TRAIN_CSV)
    df = df[df.season == 2024].drop(columns=["control_success"]).reset_index(drop=True)
    reps = int(np.ceil(n / len(df)))
    out = pd.concat([df] * reps, ignore_index=True).iloc[:n].copy()
    out["season"] = 2025
    out["row_id"] = [f"TEST_{i:06d}" for i in range(n)]
    return out


def main():
    print("=== 1. 리키지 정적 검사 ===")
    targets = [SUBMIT / "script.py"] + [SRC / f for f in COPY_FILES]
    hits = leakage_scan(targets)
    if hits:
        for f, ln, call in hits:
            print(f"  ❌ {f}:{ln}  .{call}()")
        raise SystemExit("추론 경로에 금지 호출이 있다. 제출 중단.")
    print("  ✅ groupby/value_counts/rolling/fit/quantile 등 금지 호출 없음")

    print("\n=== 2. src → submit 복사 ===")
    for f in COPY_FILES:
        shutil.copy2(SRC / f, SUBMIT / f)
        print(f"  {f}")
    (SUBMIT / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")

    model_dir = SUBMIT / "model"
    for f in ["artifacts.joblib", "meta.json"]:
        if not (model_dir / f).exists():
            raise SystemExit(f"model/{f} 없음. 먼저 src/train.py를 실행할 것.")
    boosters = sorted(model_dir.glob("lgbm_*.txt"))
    if not boosters:
        raise SystemExit("model/lgbm_*.txt 없음. 먼저 src/train.py를 실행할 것.")
    print(f"  부스터 {len(boosters)}개: {[p.name for p in boosters]}")
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))

    mlps = sorted(model_dir.glob("mlp_*.npz"))
    if meta.get("use_mlp"):
        if not mlps:
            raise SystemExit("USE_MLP인데 model/mlp_*.npz 없음. src/train.py 재실행.")
        print(f"  MLP {len(mlps)}개: {[p.name for p in mlps]}  "
              f"(alpha={meta['mlp_alpha']})")
    elif mlps:
        raise SystemExit("USE_MLP=False인데 mlp_*.npz가 남아 있다. model/ 정리 필요.")

    print("\n=== 3. 추론 시간 측정 (245,789행 더미) ===")
    dummy_dir = SUBMIT / "data"
    dummy_dir.mkdir(exist_ok=True)
    dummy = make_dummy_test()
    dummy.to_csv(dummy_dir / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame({"row_id": dummy["row_id"], "control_success": 0.5}).to_csv(
        dummy_dir / "sample_submission.csv", index=False, encoding="utf-8")

    t0 = time.time()
    r = subprocess.run([sys.executable, "script.py"], cwd=SUBMIT,
                       capture_output=True, text=True, encoding="utf-8")
    elapsed = time.time() - t0
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("script.py 실행 실패")
    print(f"  ⏱ {elapsed:.1f}s / 600s 한도  ({elapsed/600*100:.1f}%)")
    if elapsed > 480:
        print("  ⚠️ 여유가 20% 미만이다. 트리 수를 줄이는 것을 검토할 것.")

    out = pd.read_csv(SUBMIT / "output" / "submission.csv")
    assert list(out.columns) == ["row_id", "control_success"], out.columns
    assert out["control_success"].between(0, 1).all()
    assert out["control_success"].notna().all()
    print(f"  submission 검증 OK  mean={out['control_success'].mean():.4f}")
    if meta.get("trackman_mode", "none") != "none":
        print("  ⚠️ 위 평균을 목표 prior와 비교하지 말 것. 더미는 2024 행이라 Trackman")
        print("     조회 적중률이 ~99%로 실제 2025(~81%)보다 높고, 커버리지가 높을수록")
        print("     평균이 낮게 나온다. 실제 목표 달성 여부는 train.py의 참조 분포")
        print("     (신인 비율까지 재현) 기준으로 이미 보정돼 있다.")

    print("\n=== 4. zip 패키징 ===")
    shutil.rmtree(dummy_dir, ignore_errors=True)
    shutil.rmtree(SUBMIT / "output", ignore_errors=True)

    # 이 설정에서 실제로 쓰지 않는 아티팩트는 넣지 않는다.
    # (남아 있으면 코드 검증자가 "왜 안 쓰는 테이블이 있나" 하고 의심한다)
    skip = set()
    if meta.get("trackman_mode", "none") == "none":
        skip |= {"trackman_pitcher.csv", "pitcher_id_map.csv"}

    zip_path = next_zip_path(meta)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(SUBMIT.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.name not in skip:
                z.write(p, p.relative_to(SUBMIT).as_posix())
    if skip:
        print(f"  제외: {sorted(skip)}  (TRACKMAN_MODE=none)")
    with zipfile.ZipFile(zip_path) as z:
        for i in z.infolist():
            print(f"  {i.file_size:>10,}  {i.filename}")
    print(f"\n✅ {zip_path.name}  ({zip_path.stat().st_size/1e6:.1f} MB)")

    log_submission(zip_path, meta, float(out["control_success"].mean()), elapsed)


MAX_NAME = 30      # 제출 파일명 길이 제한


def describe(meta):
    """제출물을 구분하는 짧은 태그. 30자 제한이 있으므로 최대한 압축한다.
    전체 설정은 SUBMISSIONS.md에 있고, 파일명은 서로 구분만 되면 된다.

    예: `s06_p475_h5_f51_e20m3.zip`
        prior 0.475 / half_life 0.5 / 피처 51개 / LGBM 20 + MLP 3
    """
    parts = ["p" + f"{meta['target_prior_2025']:.3f}".split(".")[1]]     # 0.475 -> p475
    parts.append("h" + str(meta["recency_half_life"]).replace("0.", "").replace(".", ""))
    parts.append("f" + str(meta["n_features"]))                          # 피처 수
    e = "e" + str(meta.get("n_members", len(meta.get("seeds", []))))
    if meta.get("use_mlp"):
        e += "m" + str(len(meta.get("mlp_members", [])))
    parts.append(e)
    tm = meta.get("trackman_mode", "none")
    if tm != "none":                        # 기본값(끔)일 때는 태그를 생략한다
        parts.insert(0, {"release": "tmR", "release_ext": "tmE"}.get(tm, "tmA"))
    return "_".join(parts)


def next_zip_path(meta, suffix=""):
    """이미 쓴 번호의 **최댓값 + 1**. 개수로 세면 파일을 지우거나 이름을 바꿨을 때
    같은 번호가 다시 나와 기존 제출물을 덮어쓴다."""
    used = [0]
    for p in ROOT.glob("s[0-9][0-9]_*.zip"):
        head = p.stem.split("_")[0][1:]
        if head.isdigit():
            used.append(int(head))
    name = f"s{max(used)+1:02d}_{describe(meta)}{suffix}.zip"
    if len(name) > MAX_NAME:
        name = name[:MAX_NAME - 4] + ".zip"
        print(f"  ⚠️ 파일명이 {MAX_NAME}자를 넘어 잘랐다 → {name}")
    return ROOT / name


def log_submission(zip_path, meta, pred_mean, elapsed):
    """제출 이력을 한 줄 추가한다. LB 점수는 받은 뒤 사람이 채운다."""
    path = ROOT / "SUBMISSIONS.md"
    if not path.exists():
        path.write_text(
            "# 제출 이력\n\n"
            "`LB` 열은 리더보드 점수를 받은 뒤 직접 채운다.\n"
            "**같은 이름으로 제출하지 말 것** — 어느 설정의 점수인지 구분이 안 된다.\n\n"
            "| # | 파일 | Trackman | prior | 피처 | 홀드아웃 BSS(oracle) | LB |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n",
            encoding="utf-8")
    row = (f"| {zip_path.stem.split('_')[0][1:]} | `{zip_path.name}` | "
           f"{meta.get('trackman_mode','none')} | {meta['target_prior_2025']} | "
           f"{meta['n_features']}{'+MLP' if meta.get('use_mlp') else ''} | | |")

    # ⚠️ append하면 표 아래 본문에 줄이 붙는다. **표의 마지막 행 뒤에 끼워 넣는다.**
    lines = path.read_text(encoding="utf-8").split("\n")
    last = max((i for i, l in enumerate(lines)
                if l.startswith("| ") and l[2:4].isdigit()), default=None)
    if last is None:
        print(f"  ⚠️ {path.name}에서 표를 못 찾았다. 수동으로 추가할 것:\n    {row}")
        return
    lines.insert(last + 1, row)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  이력 추가: {path.name} (표 {last+1}번째 행 뒤)")


if __name__ == "__main__":
    main()
