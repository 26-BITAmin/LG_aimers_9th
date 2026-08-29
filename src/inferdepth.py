"""§7.217b — `mlp_infer` 임의 깊이 일반화의 **하위 호환 관문**.

무엇을 바꿨나 (§7.217 ⑤ 의 `d4p` 를 배포 가능하게)
    `mlp.export_weights`   `len(net)==9` assert → 블록 4개 단위로 일반화 (plain 전용)
    `mlp_infer.MLPPredictor`  `W0/W1/W2` 하드코딩 → npz 가 정하는 층 수 루프

왜 관문이 필요한가
    `mlp_infer.py` 는 **zip 안 서빙 코드**이고 8멤버 전부가 이 파일을 탄다. 새 팔을
    위해 고친 코드가 기존 8멤버의 순전파를 1 ULP 라도 바꾸면, LB 변화가 새 팔 때문인지
    코드 변경 때문인지 영영 못 가른다 — 그게 이 저장소가 반복해서 데인 자리다.

    → 챔피언 zip 의 `mlp_infer.py` **한 파일만** 새 것으로 갈아끼우고 2024 전 행을
      예측해 원본과 대조한다. 임계는 **`max|Δp| == 0`** (근사 아님).

    ⚠️ `team_reference/mlp_infer.py`(채연 계보 compact/general 뷰)는 **건드리지 않는다.**
       구조가 다르고 이 변경과 무관하다.

실행: .venv/Scripts/python.exe -u src/inferdepth.py 2>&1 | tee experiments/inferdepth.log
"""
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "tools_hj"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

from rebase_routing import TRAIN_CSV, predict_in_subprocess

BASE = ROOT / "hj05_r047f330.zip"
NEW_INFER = ROOT / "src" / "mlp_infer.py"


def main():
    os.makedirs("D:/t1eval", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="infd_", dir="D:/t1eval")
    old, new = os.path.join(tmp, "old"), os.path.join(tmp, "new")
    with zipfile.ZipFile(BASE) as z:
        z.extractall(old)
        z.extractall(new)

    a = open(os.path.join(old, "mlp_infer.py"), "rb").read()
    b = NEW_INFER.read_bytes()
    print(f"베이스 {BASE.name}")
    print(f"  zip 안 mlp_infer.py {len(a):,}B  vs  src/mlp_infer.py {len(b):,}B  "
          f"{'동일' if a == b else '다름 (기대대로)'}")
    shutil.copy2(NEW_INFER, os.path.join(new, "mlp_infer.py"))

    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    df = df[df["season"] == 2024].drop(columns=["control_success"]).reset_index(drop=True)
    pq = os.path.join(tmp, "proxy.pkl")
    df.to_pickle(pq)
    print(f"  프록시 2024 {len(df):,}행")

    print("추론 1/2 원본 mlp_infer ...")
    p0 = predict_in_subprocess(old, pq, os.path.join(tmp, "a.npy"))
    print("추론 2/2 일반화 mlp_infer ...")
    p1 = predict_in_subprocess(new, pq, os.path.join(tmp, "b.npy"))

    d = float(np.abs(p1 - p0).max())
    print(f"\n[하위 호환 관문]  max|Δp| = {d:.3e}   "
          f"{'통과 — 비트 동일' if d == 0.0 else '실패'}")
    print(f"  p̄ {p0.mean():.9f} -> {p1.mean():.9f}")
    shutil.rmtree(tmp, ignore_errors=True)
    return d == 0.0


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
