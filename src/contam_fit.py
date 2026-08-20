"""오염 보정표 **fit** — 학습 전용 (추론 경로에 들어가지 않는다).

`build_submit.py`의 AST 스캐너는 `cumsum` 같은 누적 연산을 금지한다.
그건 test 행끼리 정보를 흘리는 전형적 통로이기 때문이다. 여기 있는 누적은
**train 라벨**에 대한 것이라 무해하지만, 스캐너가 파일 단위로 보므로
추론용 `contam.py`와 물리적으로 분리해 둔다 — 제출물에는 fit 코드가 없다.
"""
import numpy as np

from contam import KEY


def _build_axis(pid, n, per_pitch_q):
    """오염 투구만 (선수, n) 정렬해 누적 q를 만든다. 조회는 searchsorted 2회."""
    m = per_pitch_q > 0
    keys = pid[m].astype(np.int64) * KEY + n[m].astype(np.int64)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    cum0 = np.concatenate([[0.0], np.cumsum(per_pitch_q[m][order])])
    return keys, cum0


def fit_contam(train_df, rev_label):
    """train에서만 호출. q_s 추정 + 조회 배열 생성.

    rev_label: (N,) 투구별 '반대' 라벨 (0/1). `modelabel.recover`가 복원한 것.
    """
    s = train_df["season"].to_numpy(dtype=np.int64)
    g = train_df["game_type"].astype("object").to_numpy()
    rev = np.asarray(rev_label, dtype=np.float64)

    q = {}
    for yr in range(2019, 2023):
        mR = (s == yr) & (g == "R")
        mF = (s == yr) & (g == "F")
        if mR.sum() == 0 or mF.sum() == 0:
            continue
        q[yr] = max(0.0, float(rev[mR].mean() - rev[mF].mean()))

    qv = np.zeros(len(train_df))
    dirty = (s <= 2022) & (g == "F")
    if dirty.any():
        qv[dirty] = np.array([q.get(int(y), 0.0) for y in s[dirty]])

    art = {"q": {int(k): float(v) for k, v in q.items()}}
    for axis, id_col, n_col in [("p", "pitcher_id", "asof_pitcher_n"),
                                ("b", "batter_id", "asof_batter_n")]:
        keys, cum0 = _build_axis(train_df[id_col].to_numpy(dtype=np.int64),
                                 train_df[n_col].to_numpy(dtype=np.int64), qv)
        art[f"{axis}_keys"], art[f"{axis}_cum"] = keys, cum0
    return art


