"""pre-2023 F 오염 보정 — 학습/추론 **공용** 단일 진실 소스 (§7.48·§7.49)

무엇을 고치나
    2019~2022 F 구간은 "포수 요구 반대" 기준이 걸리지 않은 채 라벨링됐다
    (F의 reverse .048 → 2023에 .300). 운영측은 과거를 재계산하지 않았고,
    `asof_*`는 커리어 누적이라 그 누락분이 **2025 행까지 따라온다**.

보정식
    투구 m번째 행의 `asof_*`는 직전 m개의 누적이다. 그 m개 중 오염 구간 투구가
    있었다면 누락된 '반대' 개수는 D = Σ q_s 이고,

        reverse_보정 = reverse_관측 + D/m
        success_보정 = success_관측 − D/m

    q_s = (그 시즌 R의 반대 비율) − (그 시즌 F의 반대 비율).   **train에서만 추정.**

왜 lookup 구조가 이런가
    D는 "직전 m개 중"의 누적이므로 **행마다 다르다.** 학습 행은 커리어 중간에
    있어서 D가 총량보다 작고, 2025 추론 행은 오염이 전부 과거라 D = 총량이다.
    두 경우를 **같은 코드**로 처리하려고 (선수, 오염투구의 n위치) 정렬 배열에
    `searchsorted`를 건다. 학습·검증·추론 세 경로가 같은 함수를 통과한다
    (§7.36 사고의 유일한 구조적 예방책).

리키지
    D에는 그 행 **이전** 투구만 들어간다 (`n < m`). 행 자신의 결과는 안 쓴다.
    test에는 `pitcher_id`/`batter_id`와 `asof_*_n`으로 조회만 한다 —
    test 행끼리 아무것도 공유하지 않으므로 CLAUDE.md §3.2를 위반하지 않는다.
"""
import numpy as np

KEY = 10_000_000            # pid * KEY + n. asof_n 최대치보다 충분히 크다


def _lookup(keys, cum0, pid, n):
    """행마다 D = (그 선수의 n보다 앞선 오염 투구들의 Σq)."""
    if len(keys) == 0:
        return np.zeros(len(pid))
    base = pid.astype(np.int64) * KEY
    start = np.searchsorted(keys, base, side="left")
    idx = np.searchsorted(keys, base + n.astype(np.int64), side="left")
    return cum0[idx] - cum0[start]


def adjust(df, art, axis):
    """행별 보정량 D/m. `axis`는 'p'(투수) 또는 'b'(타자). 표본 0이면 0."""
    if not art:
        return np.zeros(len(df))
    id_col, n_col = (("pitcher_id", "asof_pitcher_n") if axis == "p"
                     else ("batter_id", "asof_batter_n"))
    pid = df[id_col].to_numpy(dtype=np.int64)
    n = df[n_col].to_numpy(dtype=np.float64)
    d = _lookup(art[f"{axis}_keys"], art[f"{axis}_cum"], pid,
                np.nan_to_num(n).astype(np.int64))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, d / np.maximum(n, 1.0), 0.0)
