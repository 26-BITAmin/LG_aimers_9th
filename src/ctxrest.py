"""§7.118-C — δ 스크린에 오르지 않았던 pre-pitch 축 4종의 마감 전수 (신규 축 아님 선언용).

§11 의 "투구 직전 상태 컬럼은 전부 한 번씩 넣어봤다"는 엄밀히 과장이다 —
ctxsplit 9축 + ctxscreen 4축 대조 결과 다음 칸은 측정 기록이 없다:
    outs3     아웃카운트 (0/1/2)
    risp2     득점권 (2루 or 3루 주자)
    bopen2    베이스오픈 득점권 (득점권 & 1루 비어있음) — 고의 볼질/피해가는 투구가
              제구 실패 정의 2·3항과 직결되는 유일한 실기전 후보
    sdiff5    점수차 버킷 (투수팀 기준, ≤−3 / −2·−1 / 0 / +1·+2 / ≥+3)

측정은 ctxsplit 프로토콜을 s29 배포 층 위에서 재현한다:
    T1  δ표의 시즌 간 지속 — train 적합 δ vs 검증시즌 자체 적합 δ 의 셀 상관
    T2  잔차 회귀 t (OLS + 투수 CR0)
    T3  β 스윕 ΔBSS + **리그 상황효과 대조군** (주효과 재발견 감별)

사전 등록: 사망 예상 지점은 T1 (runner_on_1b 전례 .111/.037 — 스플릿 자체가
시즌을 못 넘는다). 인접 축 전멸(1루·any주자·li·이닝·홈원정) 전력상 기대 ≈0.
목적은 발견이 아니라 "전수" 선언을 측정으로 참으로 만드는 것.

실행:  .venv/Scripts/python.exe -u src/ctxrest.py 2>&1 | tee experiments/ctxrest.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import deploy29 as D
from config import CTX_HALF_LIFE, CTX_K, TARGET_COL
from ctxsplit import resid_t
from data import recency_weight
from tmslot import resid_t_cr0

BETAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
MIN_CELL = 100


def ax_outs3(df):
    return np.clip(df["outs_before"].to_numpy(np.int64), 0, 2), 3


def ax_risp2(df):
    r = ((df["runner_on_2b"].to_numpy(np.int64) > 0)
         | (df["runner_on_3b"].to_numpy(np.int64) > 0))
    return r.astype(np.int64), 2


def ax_bopen2(df):
    r = ((df["runner_on_2b"].to_numpy(np.int64) > 0)
         | (df["runner_on_3b"].to_numpy(np.int64) > 0)) \
        & (df["runner_on_1b"].to_numpy(np.int64) == 0)
    return r.astype(np.int64), 2


def ax_sdiff5(df):
    sd = df["score_diff_pitcher_team"].to_numpy(np.int64)
    b = np.select([sd <= -3, sd <= -1, sd == 0, sd <= 2], [0, 1, 2, 3], default=4)
    return b.astype(np.int64), 5


AXES = {"outs3": ax_outs3, "risp2": ax_risp2, "bopen2": ax_bopen2, "sdiff5": ax_sdiff5}


def fit_delta(dfp, coder, k, weights=None):
    """ctxfit.fit_ctx 동형 — coder 함수를 직접 받는 판. (keys, delta, g_c, m)"""
    y = dfp[TARGET_COL].to_numpy(np.float64)
    pid = dfp["pitcher_id"].to_numpy(np.int64)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, np.float64)
    code, m = coder(dfp)

    swc = np.bincount(code, weights=w, minlength=m)
    g_c = (np.bincount(code, weights=w * y, minlength=m)
           / np.maximum(swc, 1e-12)) - (w * y).sum() / w.sum()

    upid, invp = np.unique(pid, return_inverse=True)
    rate_p = (np.bincount(invp, weights=w * y)
              / np.maximum(np.bincount(invp, weights=w), 1e-12))
    key = pid * m + code
    u, inv = np.unique(key, return_inverse=True)
    sw = np.bincount(inv, weights=w)
    sw2 = np.bincount(inv, weights=w * w)
    n_eff = np.where(sw2 > 0, sw * sw / np.maximum(sw2, 1e-12), 0.0)
    t = np.bincount(inv, weights=w * y)
    rp = rate_p[np.searchsorted(upid, u // m)]
    delta = (n_eff / (n_eff + k)) * (t / np.maximum(sw, 1e-12) - rp)
    return u, delta, n_eff, g_c, m


def lookup(dfp, coder, keys, delta, m):
    code, _ = coder(dfp)
    kv = dfp["pitcher_id"].to_numpy(np.int64) * m + code
    j = np.clip(np.searchsorted(keys, kv), 0, len(keys) - 1)
    return np.where(keys[j] == kv, delta[j], 0.0), code


def main():
    df, s, y = D.load_df()
    print(f"train {df.shape}   K={CTX_K}   MIN_CELL={MIN_CELL}\n")

    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        tr = df[s < fold]
        w = recency_weight(tr["season"].to_numpy(), CTX_HALF_LIFE,
                           ref_season=int(s[s < fold].max()))
        print("=" * 96)
        print(f"fold {fold}   기준 BSS {st['base']:.1f}")
        print("=" * 96)
        print(f"  {'축':<9}{'T1 corr':>9}{'셀수':>7}{'T2 OLS':>9}{'CR0':>7}"
              f"{'리그t':>7}   T3 (β: " + " ".join(f"{b}" for b in BETAS) + ") · 리그 대조")

        for name, coder in AXES.items():
            u_tr, d_tr, n_tr, g_c, m = fit_delta(tr, coder, CTX_K, w)
            # T1 — 검증 시즌 자체 적합(무수축 rate 차)과의 셀 상관
            dv = df[s == fold]
            u_va, d_va, n_va, _, _ = fit_delta(dv, coder, 0.0)   # k=0 = 원시 편차
            common, ia, ib = np.intersect1d(u_tr, u_va, return_indices=True)
            okm = (n_tr[ia] >= MIN_CELL) & (n_va[ib] >= MIN_CELL)
            t1 = (np.corrcoef(d_tr[ia][okm], d_va[ib][okm])[0, 1]
                  if okm.sum() >= 30 else np.nan)

            # T2 — 잔차 회귀
            v, code = lookup(st["d"], coder, u_tr, d_tr, m)
            _, t_ols = resid_t(st["r"], v)
            _, t_cr0 = resid_t_cr0(st["r"], v, st["cl"], st["n_id"])
            v_lg = g_c[code]
            _, t_lg = resid_t_cr0(st["r"], v_lg, st["cl"], st["n_id"])

            # T3 — β 스윕 (투수별 / 리그 대조)
            row = [D.rescore(st, b * v) for b in BETAS]
            row_lg = [D.rescore(st, b * v_lg) for b in BETAS]
            print(f"  {name:<9}{t1:>9.3f}{int(okm.sum()):>7}{t_ols:>+9.2f}{t_cr0:>+7.2f}"
                  f"{t_lg:>+7.2f}   " + " ".join(f"{g:>+6.2f}" for g in row)
                  + "  |  " + " ".join(f"{g:>+6.2f}" for g in row_lg))

    print("\n" + "=" * 96)
    print("판정: T1 낮으면(전례 .11/.04) 즉사. T2 양 fold CR0 |t|>2 & 리그 대조 초과 &")
    print("T3 이득 > 시드 노이즈(±3.3)여야 후보. 사전 예측: 전 축 T1 사망 또는 노이즈 대역.")


if __name__ == "__main__":
    main()
