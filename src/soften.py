"""2019–22 F 오염 y=1 라벨의 soft label 보정 (§7.70) — 학습 경로 **공용** 단일 진실 소스.

무엇을 고치나
    2019–22 F는 reverse 검출 실패로 실패가 성공으로 기록됐다. 오염은 단측이다 —
    y_obs=0 행은 무결하고(몰림/벗어남/잡힌 reverse), **y_obs=1 행만** 숨은
    reverse가 섞여 있다. 그 확률은 시즌 상수로 추정된다:

        h_s = P(숨은 reverse | y_obs=1) = q_s / p̄_s
        q_s = (그 시즌 R 반대율) − (그 시즌 F 반대율)     ← §7.49 contam과 같은 정의
        p̄_s = 그 시즌 F의 관측 성공률                     ← 둘 다 넘긴 df에서만 추정

구현 — binary 손실을 유지하는 행 복제 트릭
    라벨 p인 행 = (y=1, w·p) + (y=0, w·(1−p)). soft target cross-entropy와
    수학적으로 동치이고, LGBM(binary)과 MLP(BCEWithLogits × 행 가중) 모두에
    같은 방식으로 적용된다. 투수별 세분화는 부호 불일치로 기각됐다 (§7.70 1단계).

경로
    train.py(배포)와 verify_pipeline.py(검증)가 **같은 함수를 호출한다.**
    추론 경로에는 존재하지 않는다 — 라벨이 없는 곳에 보정할 것도 없다.
    제출 zip에도 들어가지 않는다 (학습 전용).
"""
import numpy as np
import pandas as pd

from config import MLP_AUX_MODES, TARGET_COL
from modelabel import recover


def fit_h(df):
    """시즌별 h_s. 넘긴 df(= 그 경로의 학습 데이터)에서만 추정한다."""
    lab, _, _ = recover(df)
    rev = np.where(np.isnan(lab["reverse"]), 0.0, np.round(lab["reverse"]))
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    s = df["season"].to_numpy()
    g = df["game_type"].to_numpy().astype(str)
    h = {}
    for yr in range(2019, 2023):
        mR = (s == yr) & (g == "R")
        mF = (s == yr) & (g == "F")
        if not mF.any() or not mR.any():
            continue
        q = max(0.0, float(rev[mR].mean() - rev[mF].mean()))
        h[int(yr)] = min(q / max(float(y[mF].mean()), 1e-9), 0.95)
    return h


def soften(df, X, y, w, aux=None):
    """(X, y, w[, aux])에 복제 행을 붙여 soft target CE와 동치로 만든다.

    ⚠️ X·y·w·aux는 전부 df와 **행이 정렬**돼 있어야 한다 (같은 순서, 같은 길이).
    반환: X2, y2, w2, aux2(입력이 None이면 None), h(기록용)
    """
    assert len(X) == len(df) == len(y) == len(w), "행 정렬이 깨졌다"
    h = fit_h(df)
    s = df["season"].to_numpy()
    g = df["game_type"].to_numpy().astype(str)
    h_row = np.zeros(len(df))
    for yr, v in h.items():
        h_row[(s == yr) & (g == "F")] = v
    dup = (h_row > 0) & (y == 1)

    X2 = pd.concat([X, X[dup]], ignore_index=True)
    y2 = np.concatenate([y, np.zeros(int(dup.sum()))])
    w2 = np.asarray(w, dtype=np.float64).copy()
    w2[dup] *= (1 - h_row[dup])
    w2 = np.concatenate([w2, np.asarray(w)[dup] * h_row[dup]])

    aux2 = None
    if aux is not None:
        # 복제 행은 "숨은 reverse였다"는 가설 그 자체다 → reverse=1, 나머지 0.
        # 축 인덱스는 이름으로 찾는다 — 순서를 가정하면 §7.36 사고의 재판이다.
        a_dup = np.zeros_like(aux[dup])
        if "reverse" in MLP_AUX_MODES:
            a_dup[:, tuple(MLP_AUX_MODES).index("reverse")] = 1.0
        aux2 = np.concatenate([aux, a_dup])
    return X2, y2, w2, aux2, h
