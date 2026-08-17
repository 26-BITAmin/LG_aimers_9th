"""투수 × 상황 편차 δ 를 학습 데이터에서 만든다 (사후 로짓 보정항)

**학습 전용 — 제출 zip에 안 들어간다.** 이유는 두 가지다.
  ① 타깃을 쓴다. 추론 경로에 타깃을 읽는 코드가 있으면 안 된다.
  ② `build_submit.py`의 AST 스캐너가 도는 파일에는 fit 계열을 두지 않는다.
추론 쪽은 `features.logit_adjust`가 얼어붙은 배열을 searchsorted로 조회만 한다.

정의 (축 이름 c, 상황 코드 함수는 `features.CTX_CODERS[c]`)
    rate_pc = 투수 p가 상황 c에서 낸 제구 성공률
    rate_p  = 투수 p 전체 성공률
    δ(p,c)  = n_pc/(n_pc + k) · (rate_pc − rate_p)

    **자기 평균으로부터의 편차**이므로 리그 레벨 드리프트에 자동으로 중립이다.
    상황의 주효과(손잡이·카운트)는 이미 모델 피처로 들어가 있고, 여기서 더하는
    것은 **투수별 편차만**이다. 대조군(리그 평균 상황효과)이 0인 것으로 확인했다.

리키지
    train에서만 fit → test에 조회. CLAUDE.md가 명시적으로 허용하는 형태다
    ("train에서 fit, test에서 transform"). 모델을 δ로 재학습하지 않고 **사후에
    로짓에 더하므로** 학습셋 타깃 리크 문제가 아예 생기지 않는다.
    조회 키에 `season`이 없다 — §6 버그 2(참조 분포 불일치)가 성립하지 않는 이유다.

채택 근거
    hand    §7.36  잔차 t +7.3/+7.2   BSS +22.04/+20.03   →  LB +5.93
    count4  §7.37  잔차 t +4.6/+5.7   BSS  +7.49/ +9.74 (손잡이 δ 위 증분)
"""
import numpy as np

from features import CTX_CODERS


def fit_ctx(train_df, target_col, axes, k, weights=None):
    """axes: [(축이름, β), ...] → [{"name", "beta", "keys", "delta"}, ...]

    `keys`는 정렬된 int64 (searchsorted 전제), `delta`는 float64.
    m(코드 개수)은 `features.CTX_CODERS`가 정하므로 여기서 따로 저장하지 않는다 —
    두 곳에 두면 어긋난다.

    `weights`: 행 가중 (없으면 균등 = 종전과 비트 동일). §7.61에서 넣었다 —
        균등으로 두면 **δ 표의 질량 66%가 2019~2022**에 있고, 그 시즌들은 모델
        학습에서 가중 6%밖에 안 갖는다. 게다가 pre-2023 F 105,308행(7.1%)은
        라벨 성공률이 +0.23 편향돼 있다 (2022 F .709 vs R .504 → 2023 F .473).
        → δ 표가 모델이 무시하는 시즌, 그중 오염 구간까지 균등하게 반영한다.

    수축 분모의 n은 **유효 표본 수** (Σw)²/Σw² 를 쓴다. Σw를 그대로 쓰면
    가중이 작은 옛 행이 많은 셀에서 수축이 과하게 세진다.
    """
    y = train_df[target_col].to_numpy(dtype=np.float64)
    pid = train_df["pitcher_id"].to_numpy(dtype=np.int64)
    w = (np.ones(len(y)) if weights is None
         else np.asarray(weights, dtype=np.float64))

    upid, invp = np.unique(pid, return_inverse=True)
    sw_p = np.bincount(invp, weights=w)
    rate_p = np.bincount(invp, weights=w * y) / np.maximum(sw_p, 1e-12)

    out = []
    for name, beta in axes:
        if not beta:
            continue
        code, m = CTX_CODERS[name](train_df)
        key = pid * m + code
        u, inv = np.unique(key, return_inverse=True)     # u는 정렬돼 있다
        sw = np.bincount(inv, weights=w)
        sw2 = np.bincount(inv, weights=w * w)
        n_eff = np.where(sw2 > 0, sw * sw / np.maximum(sw2, 1e-12), 0.0)
        t = np.bincount(inv, weights=w * y)
        rp = rate_p[np.searchsorted(upid, u // m)]       # 각 셀이 속한 투수의 전체 성공률
        delta = (n_eff / (n_eff + k)) * (t / np.maximum(sw, 1e-12) - rp)
        out.append({"name": name, "beta": float(beta),
                    "keys": u, "delta": delta.astype(np.float64)})
    return out


def summary(ctx):
    """학습 로그용 한 줄씩."""
    return [f"{a['name']}: 셀 {len(a['keys']):,}개  δ std={a['delta'].std():.5f}"
            f"  β={a['beta']}" for a in ctx]
