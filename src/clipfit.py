"""꼬리 winsorize 경계를 학습 데이터에서 뽑는다. **학습 전용 — 제출 zip에 안 들어간다.**

왜 여기 있나
    `features.py`는 제출 zip에 복사되고 `build_submit.py`의 AST 스캐너가
    `.quantile()` 호출을 금지한다. 경계 계산은 학습 시점 fit이므로 규정상 문제가
    없지만, **스캐너를 약화시키는 대신 fit을 추론 경로 밖으로 옮긴다.**
    `features.build_features`는 얼어붙은 숫자 두 개로 `np.clip`만 한다.

무엇을 자르나 (`experiments/eda_tail.log`)
    시즌간 산포:  중간 빈 0.007  vs  꼬리 빈 0.023  (3.2배)
    배포 모델의 꼬리 5% 손실:
        fold2023  batter 상위5%  실제 .4767 → 예측 .6380   레벨손실 10,412
        fold2024  batter 상위5%  실제 .4710 → 예측 .5167   레벨손실    837
    양 fold에서 같은 부호로 잃는다. 관계는 중간에서 안정적이고 **꼬리만 배신한다.**
"""
import numpy as np

# 꼬리가 불안정한 것으로 측정된 컬럼 (eda_tail.py Q1: 배율 2.3~3.5x)
CLIP_COLS = [
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
]


def fit_clip(train_df, q, cols=None, weights=None):
    """train에서만 호출. {컬럼: (lo, hi)} 를 돌려준다.

    q=0.02 면 양끝 2%를 각각 2%/98% 분위값으로 눌러 앉힌다. 값을 버리는 게 아니라
    **레버리지를 자르는 것**이다 — 행 수는 그대로고 극단값만 경계로 이동한다.

    가중치를 주면 최근 시즌 분포에 맞춘 경계가 나온다. 학습 가중이 hl=0.5로
    사실상 직전 시즌이므로 그쪽에 맞추는 편이 추론 시점 분포에 가깝다.
    """
    out = {}
    for c in (CLIP_COLS if cols is None else cols):
        if c not in train_df.columns:
            continue
        v = train_df[c].to_numpy(dtype=np.float64)
        m = np.isfinite(v)
        if m.sum() < 1000:
            continue
        if weights is None:
            lo, hi = np.quantile(v[m], [q, 1.0 - q])
        else:
            lo, hi = _wquantile(v[m], np.asarray(weights, dtype=np.float64)[m],
                                [q, 1.0 - q])
        out[c] = (float(lo), float(hi))
    return out


def _wquantile(v, w, qs):
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    cw = (cw - 0.5 * w) / cw[-1]
    return np.interp(qs, cw, v)
