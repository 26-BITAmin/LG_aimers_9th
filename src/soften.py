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
    같은 방식으로 적용된다. 투수별 세분화는 부호 불일치로 기각됐고 (§7.70 1단계),
    R 학습 r(x) 행 단위 재배분도 배포 등가 하네스에서 기각됐다 (§7.179 — 시즌
    상수가 실측으로 최종).

★ h 의 **크기**는 §7.70 에서 한 번도 스윕되지 않았다 (§7.161).
    q_s 는 "F 의 진짜 reverse 성향 = R 의 성향" 을 가정한다. 2023+ 실측은
    F(.300/.288) > R(.250/.240) 이라 그 가정이 h 를 **과소추정**할 수 있다.
    `SOFT_LABEL_BETA` 가 그 크기를 조절하는 유일한 노브다. 기본 1.0 = 항등.

    mode "lin"    h ← clip(β·h, 0, 0.95)          단순. β∈[0.8,1.6] 에서 포화 없음.
    mode "logit"  h ← σ(β·logit(h))               (0,1) 에 자연히 갇힘. β=1 항등.
    mode "target" h ← 1 − target_s/p̄_s            추세 목표에 직접 앉힌다.
                  target_s = p_anchor + drift·(anchor_season − s)
                  ⚠️ df 에 2023+ F 가 있어야 정의된다 (fold2023 에서는 불가).

경로
    train.py(배포)와 verify_pipeline.py(검증)가 **같은 함수를 호출한다.**
    추론 경로에는 존재하지 않는다 — 라벨이 없는 곳에 보정할 것도 없다.
    제출 zip에도 들어가지 않는다 (학습 전용).
"""
import numpy as np
import pandas as pd

from config import (MLP_AUX_MODES, SOFT_LABEL_BETA, SOFT_LABEL_BETA_MODE,
                    SOFT_LABEL_TARGET_DRIFT, TARGET_COL)
from modelabel import recover

H_CAP = 0.95


def _scale(h_raw, beta, mode, pbar, df=None):
    """h_raw(시즌→h) 에 공통 배율을 건다. beta=1.0 이면 어느 모드든 항등이어야 한다."""
    if mode == "lin":
        return {s: float(np.clip(beta * v, 0.0, H_CAP)) for s, v in h_raw.items()}

    if mode == "logit":
        out = {}
        for s, v in h_raw.items():
            if v <= 0.0:
                out[s] = 0.0                       # logit 미정의 — 0 은 0 으로 둔다
                continue
            z = np.log(v / (1.0 - min(v, 1 - 1e-9)))
            out[s] = float(np.clip(1.0 / (1.0 + np.exp(-beta * z)), 0.0, H_CAP))
        return out

    if mode == "target":
        # 깨끗한 구간(2023+)의 F 성공률을 앵커로, 거기서 추세로 외삽한 목표에 앉힌다.
        assert df is not None, "target 모드는 df 가 필요하다"
        s_all = df["season"].to_numpy()
        g_all = df["game_type"].to_numpy().astype(str)
        y_all = df[TARGET_COL].to_numpy(dtype=np.float64)
        clean = [yr for yr in range(2023, 2031) if ((s_all == yr) & (g_all == "F")).any()]
        if not clean:
            raise ValueError("target 모드: df 에 2023+ F 행이 없다 — 앵커를 만들 수 없다")
        a_yr = min(clean)                          # 가장 이른 깨끗한 시즌 = 외삽 거리 최소
        p_a = float(y_all[(s_all == a_yr) & (g_all == "F")].mean())
        out = {}
        for s, _ in h_raw.items():
            tgt = p_a + SOFT_LABEL_TARGET_DRIFT * (a_yr - s)
            out[s] = float(np.clip(1.0 - tgt / max(pbar[s], 1e-9), 0.0, H_CAP))
        return out

    raise ValueError(f"모르는 mode: {mode}")


def fit_h(df, beta=None, mode=None):
    """시즌별 h_s. 넘긴 df(= 그 경로의 학습 데이터)에서만 추정한다.

    beta/mode 를 넘기지 않으면 config 값을 쓴다 (배포 경로).
    """
    beta = SOFT_LABEL_BETA if beta is None else float(beta)
    mode = SOFT_LABEL_BETA_MODE if mode is None else str(mode)

    lab, _, _ = recover(df)
    rev = np.where(np.isnan(lab["reverse"]), 0.0, np.round(lab["reverse"]))
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    s = df["season"].to_numpy()
    g = df["game_type"].to_numpy().astype(str)
    h_raw, pbar = {}, {}
    for yr in range(2019, 2023):
        mR = (s == yr) & (g == "R")
        mF = (s == yr) & (g == "F")
        if not mF.any() or not mR.any():
            continue
        q = max(0.0, float(rev[mR].mean() - rev[mF].mean()))
        pbar[int(yr)] = float(y[mF].mean())
        h_raw[int(yr)] = min(q / max(pbar[int(yr)], 1e-9), H_CAP)

    if beta == 1.0 and mode in ("lin", "logit"):
        return h_raw                               # 배포 기본값 — 비트 하나 안 바뀐다
    return _scale(h_raw, beta, mode, pbar, df=df)


def soften(df, X, y, w, aux=None, beta=None, mode=None):
    """(X, y, w[, aux])에 복제 행을 붙여 soft target CE와 동치로 만든다.

    ⚠️ X·y·w·aux는 전부 df와 **행이 정렬**돼 있어야 한다 (같은 순서, 같은 길이).
    반환: X2, y2, w2, aux2(입력이 None이면 None), h(기록용)
    """
    assert len(X) == len(df) == len(y) == len(w), "행 정렬이 깨졌다"
    h = fit_h(df, beta=beta, mode=mode)
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
