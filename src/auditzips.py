"""제출 zip 전수 감사 — **실제로 제출한 그 파일들**에 대고 행 독립성을 잰다

왜 필요한가
    운영측 공지: "모든 리더보드 제출은 **제출 시점부터** 대회 규칙을 준수해야 하며,
    규칙 위반 제출이 확인되는 경우 실격에 해당할 수 있습니다."

    `src/rowindep.py`는 **현재 작업 트리**의 `submit/`만 본다. 그건 마지막 제출뿐이다.
    과거 제출을 확인하려면 그때 올린 zip 자체를 열어야 한다. 다행히 전부 남아 있다.

    git 이력은 근거가 약하다 — 제출물은 대부분 커밋되지 않은 작업 트리 상태에서
    빌드됐다. zip이 유일한 1차 증거다.

무엇을 검사하는가
    ① 정적: AST로 금지 호출(groupby/value_counts/rolling/fit/quantile...)을 찾는다
    ② 동적: 각 zip **자신의** predict.py + model/ 로
         · 조각으로 나눠 예측 == 통짜 예측
         · 행 순서를 섞어 예측 후 복원 == 통짜 예측
       하나라도 0이 아니면 행 사이가 이어져 있다는 뜻이다.

    ②는 **서브프로세스로 격리**한다. zip마다 predict/features 모듈 이름이 같아서
    한 프로세스에서 연달아 import하면 sys.modules 캐시가 섞인다.

실행:  .venv/Scripts/python.exe -u src/auditzips.py 2>&1 | tee experiments/auditzips.log
"""
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import pandas as pd

from build_submit import FORBIDDEN_ATTRS, FORBIDDEN_CALLS

ROOT = Path(__file__).resolve().parents[1]
N = 4000            # 행 수. 11개 zip × 모델 로딩이라 작게 잡는다
CHUNK = 400

# 각 zip 안에서 실행될 러너. 자기 디렉터리를 sys.path 맨 앞에 넣는다.
RUNNER = r'''
import sys, json
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
from predict import Predictor
df = pd.read_parquet(sys.argv[2])
pr = Predictor(sys.argv[1] + "/model")
whole = pr.predict(df)
n, chunk = len(df), int(sys.argv[3])
parts = np.concatenate([pr.predict(df.iloc[i:i+chunk]) for i in range(0, n, chunk)])
rng = np.random.default_rng(42); perm = rng.permutation(n)
sh = pr.predict(df.iloc[perm].reset_index(drop=True))
inv = np.empty_like(perm); inv[perm] = np.arange(n)
print(json.dumps({
    "mean": float(whole.mean()),
    "chunk": float(np.abs(parts - whole).max()),
    "shuffle": float(np.abs(sh[inv] - whole).max()),
    "single": float(np.abs(np.concatenate([pr.predict(df.iloc[i:i+1])
                                           for i in range(50)]) - whole[:50]).max()),
}))
'''


def scan(d):
    """AST 정적 검사. (파일, 줄, 호출명) 목록."""
    hits = []
    for p in sorted(Path(d).rglob("*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        except SyntaxError as e:
            hits.append((p.name, e.lineno or 0, f"SyntaxError: {e.msg}"))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_CALLS | FORBIDDEN_ATTRS:
                    hits.append((p.name, node.lineno, node.func.attr))
    return hits


def main():
    zips = sorted(ROOT.glob("s??_*.zip"))
    print(f"감사 대상 {len(zips)}개\n")

    # test를 흉내낼 행 (2024 표본, 타깃 제거). parquet로 넘겨 dtype을 보존한다.
    src = pd.read_csv(ROOT / "data" / "train.csv")
    df = (src[src.season == 2024]
          .drop(columns=["control_success"], errors="ignore")
          .sample(n=N, random_state=42).reset_index(drop=True))
    del src
    tmp = Path(tempfile.mkdtemp(prefix="audit_"))
    pq = tmp / "probe.parquet"
    df.to_parquet(pq)
    print(f"검사 행 {len(df):,}개 (2024 표본, 타깃 제거)")
    print(f"작업 폴더 {tmp}\n")
    print(f"  {'제출':<34}{'정적':>6}{'조각':>11}{'셔플':>11}{'1행':>11}{'평균':>9}")
    print("  " + "-" * 82)

    bad, rows = [], []
    for z in zips:
        d = tmp / z.stem
        with zipfile.ZipFile(z) as f:
            f.extractall(d)
        # zip 루트에 script.py가 없으면 한 겹 더 들어가 있는 것이다
        if not (d / "predict.py").exists():
            sub = [p for p in d.iterdir() if p.is_dir() and (p / "predict.py").exists()]
            if sub:
                d = sub[0]

        st = scan(d)
        st_s = "✅" if not st else f"❌{len(st)}"
        if st:
            bad.append((z.stem, "정적", st))

        r = subprocess.run([sys.executable, "-c", RUNNER, str(d), str(pq), str(CHUNK)],
                           capture_output=True, text=True,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        if r.returncode != 0:
            print(f"  {z.stem:<34}{st_s:>6}   실행 실패: "
                  f"{(r.stderr or '').strip().splitlines()[-1][:44] if r.stderr else '?'}")
            bad.append((z.stem, "실행실패", (r.stderr or "")[-800:]))
            shutil.rmtree(d, ignore_errors=True)
            continue

        m = json.loads(r.stdout.strip().splitlines()[-1])
        ok = all(m[k] == 0.0 for k in ("chunk", "shuffle", "single"))
        print(f"  {z.stem:<34}{st_s:>6}{m['chunk']:>11.3e}{m['shuffle']:>11.3e}"
              f"{m['single']:>11.3e}{m['mean']:>9.4f}   {'✅' if ok else '❌'}")
        rows.append((z.stem, m, ok))
        if not ok:
            bad.append((z.stem, "행독립성", m))
        shutil.rmtree(d, ignore_errors=True)      # zip 11개 × 18MB — 바로 지운다

    print("\n" + "=" * 86)
    if bad:
        print(f"❌ 문제 {len(bad)}건")
        for name, kind, detail in bad:
            print(f"  · {name}  [{kind}]  {detail}")
        raise SystemExit(1)
    print(f"✅ 제출 zip {len(rows)}개 전부 통과 — 정적 검사 무결, "
          f"행 독립성 세 검사 모두 정확히 0")
    print("   각 zip 자신의 predict.py + model/ 로 쟀다 (현재 작업 트리가 아니라).")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
