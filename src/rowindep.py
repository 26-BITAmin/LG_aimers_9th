"""행 독립성 최종 확인 — **실제 제출 아티팩트**에 대고 잰다

대회 규정상 각 행은 완전히 독립적으로 예측돼야 한다 (CLAUDE.md §3.2).
정적 AST 스캔(`build_submit.py` 1단계)은 `groupby`/`quantile` 같은 **호출 이름**만
본다. 실제로 행 사이가 이어지는지는 **동적으로** 확인해야 한다.

`src/`가 아니라 `submit/`을 임포트한다 — 평가 서버가 실행할 바로 그 코드다.
(둘이 같은지는 build_submit이 복사로 보장하지만, 여기서는 실물을 쓴다.)

  ① 조각으로 나눠 예측 == 통짜 예측
     행 사이를 잇는 계산이 하나라도 있으면 조각 경계에서 값이 달라진다.
  ② 행 순서를 섞어 예측 후 복원 == 통짜 예측
     행 순서에 의존하는 계산(rolling/cumsum/정렬 기반)을 잡는다.

둘 다 `0.000e+00`이어야 한다. **부동소수점 오차조차 허용하지 않는다** — 진짜로
행 단위 독립이면 같은 입력에 같은 부동소수점 연산이 일어나므로 비트가 같다.

실행:  .venv/Scripts/python.exe -u src/rowindep.py 2>&1 | tee experiments/rowindep_s12.log
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "submit"))          # ← src가 아니라 submit

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

from predict import Predictor                     # submit/predict.py

N = 20000
CHUNK = 1000
SEED = 42


def main():
    src = pd.read_csv(ROOT / "data" / "train.csv")
    # 2024 행에서 표본을 뽑아 test를 흉내낸다 (타깃 컬럼은 떨어뜨린다)
    df = (src[src.season == 2024]
          .drop(columns=["control_success"], errors="ignore")
          .sample(n=N, random_state=SEED)
          .reset_index(drop=True))
    print(f"제출 아티팩트: {ROOT / 'submit' / 'model'}")
    print(f"검사 행 {len(df):,}개 (2024 표본, 타깃 제거)\n")

    pr = Predictor(str(ROOT / "submit" / "model"))
    whole = pr.predict(df)
    print(f"통짜 예측   평균 {whole.mean():.6f}  최소 {whole.min():.4f}"
          f"  최대 {whole.max():.4f}")
    print(f"보정축 적중 " + "  ".join(f"{k} {v:.1%}" for k, v in pr.ctx_hits.items()))

    fail = []

    # ① 조각
    parts = [pr.predict(df.iloc[i:i + CHUNK]) for i in range(0, len(df), CHUNK)]
    d1 = float(np.abs(np.concatenate(parts) - whole).max())
    print(f"\n① 조각 {CHUNK}행 × {len(parts)}개 vs 통짜   max|diff| = {d1:.3e}"
          f"   {'✅' if d1 == 0.0 else '❌'}")
    if d1 != 0.0:
        fail.append("조각")

    # ② 셔플 후 복원
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(df))
    shuffled = pr.predict(df.iloc[perm].reset_index(drop=True))
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(df))
    d2 = float(np.abs(shuffled[inv] - whole).max())
    print(f"② 행 순서 셔플 후 복원 vs 통짜        max|diff| = {d2:.3e}"
          f"   {'✅' if d2 == 0.0 else '❌'}")
    if d2 != 0.0:
        fail.append("셔플")

    # ③ 단일 행 — 극단적 경계 조건 (조각 크기 1)
    single = np.concatenate([pr.predict(df.iloc[i:i + 1]) for i in range(200)])
    d3 = float(np.abs(single - whole[:200]).max())
    print(f"③ 1행씩 200회 vs 통짜                 max|diff| = {d3:.3e}"
          f"   {'✅' if d3 == 0.0 else '❌'}")
    if d3 != 0.0:
        fail.append("단일행")

    print("\n" + "=" * 60)
    if fail:
        raise SystemExit(f"❌ 행 독립성 위반: {fail}")
    print("✅ 행 독립성 확인. 제출 아티팩트가 각 행을 독립적으로 예측한다.")


if __name__ == "__main__":
    main()
