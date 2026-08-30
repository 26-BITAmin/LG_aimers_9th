"""평가 서버 엔트리포인트.

계약:
    입력  ./data/test.csv, ./data/sample_submission.csv
    출력  ./output/submission.csv   (row_id, control_success)

리키지 자가검증: 이 스크립트와 predict.py / features.py 어디에도
test 데이터프레임에 대한 groupby / value_counts / rolling / fit 호출이 없다.
행 사이를 잇는 모든 값(리그 평균, 범주 코드, prior shift delta)은
학습 시점에 계산되어 model/artifacts.joblib에 동결돼 있다.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 콘솔 인코딩이 UTF-8이 아니면(예: Windows CP949) print 한 줄에 프로세스가 죽는다.
# 평가 서버의 로케일을 알 수 없으므로 방어적으로 고정한다.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import pandas as pd

from predict import Predictor

ID_COL = "row_id"
TARGET_COL = "control_success"

TEST_PATH = "./data/test.csv"
SAMPLE_SUB_PATH = "./data/sample_submission.csv"
MODEL_DIR = "./model"
OUT_PATH = "./output/submission.csv"


def main():
    t0 = time.time()

    print("Load model...")
    predictor = Predictor(MODEL_DIR)
    print(f"  features={len(predictor.cols)}  logit_delta={predictor.delta:+.5f}")

    print("Load test data...")
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    sub = pd.read_csv(SAMPLE_SUB_PATH, encoding="utf-8-sig")
    if ID_COL not in test.columns:
        raise ValueError(f"test에 {ID_COL} 없음: {list(test.columns)[:5]}")
    print(f"  test={len(test)}  submission={len(sub)}")

    print("Predict...")
    preds = predictor.predict(test)
    print(f"  mean={preds.mean():.4f}  min={preds.min():.4f}  max={preds.max():.4f}")
    if getattr(predictor, "tm_match_rate", None) is not None:
        # 실제 2025 test의 기대 적중률은 80% 안팎이다 (신인 투수는 매핑에 없다).
        # 로컬 더미(2024 행)로 재면 과대평가되므로 그 숫자를 믿지 말 것.
        print(f"  trackman 조회 적중률 {predictor.tm_match_rate:.1%}"
              f"{'   ⚠️ 비정상적으로 낮다' if predictor.tm_match_rate < 0.5 else ''}")

    # sample_submission의 row_id 순서를 기준으로 병합 (베이스라인과 동일 패턴)
    pred_map = dict(zip(test[ID_COL].tolist(), preds.tolist()))
    values = [pred_map.get(rid, cur) for rid, cur in zip(sub[ID_COL], sub[TARGET_COL])]
    n_missing = sum(1 for rid in sub[ID_COL] if rid not in pred_map)
    if n_missing:
        print(f"  경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    sub.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved: {OUT_PATH} (rows={len(sub)})  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
