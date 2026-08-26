"""§7.118-B — δ 저표본 셀을 0 대신 **리그 상황효과로** 수축하면 나아지는가 (RESUME #5).

현재 δ(p,c) = w_cell·(rate_pc − rate_p) 는 저표본 셀에서 0 으로 수축한다 —
신인·저경력 투수는 상황 보정을 사실상 못 받는다. 제안: 0 대신 그 상황의
**리그 평균 편차 g_c** 로 수축한다:
    δ_grp(p,c) = w·(rate_pc − rate_p) + (1−w)·g_c
               = w·(rate_pc − rate_p − g_c) + g_c        ← 미등록 셀은 g_c 를 받는다

사망 예상 지점 (사전 등록)
    §7.36 대조군이 이미 답을 반쯤 줬다: 리그 평균 플래툰(β1.5)은 −6.73/+0.10 —
    상황 주효과는 **모델 피처에 이미 있다.** g_c 성분이 이득을 내면 그건 새 정보가
    아니라 주효과 재주입일 가능성이 크므로, **리그 단독 팔(g_c만)** 을 반드시
    나란히 잰다. 살 길은 하나: 이득이 저표본 행 부분집합에 국한되고 리그 단독
    팔과 구별될 때만. 관문: 저n 행 한정 Δ 양 fold 양수 & 리그 팔 초과 &
    전체 행 순액 비음수. 사전 예측: 리그 팔과 구별 불가로 죽는다.

실행:  .venv/Scripts/python.exe -u src/ctxgshrink.py 2>&1 | tee experiments/ctxgshrink.log
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
from config import CTX_AXES, CTX_HALF_LIFE, CTX_K, TARGET_COL
from data import recency_weight
from features import CTX_CODERS, ctx_lookup

DEP_BETA = dict(CTX_AXES)
BETAS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.875, 2.5]


def fit_group(train_df, name, k, weights):
    """δ_grp 구성 요소: 셀 δ'(= w·(rate_pc−rate_p−g_c)) 표 + g_c 벡터 + n_eff 표."""
    y = train_df[TARGET_COL].to_numpy(dtype=np.float64)
    pid = train_df["pitcher_id"].to_numpy(dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)

    code, m = CTX_CODERS[name](train_df)
    # 리그 상황효과 g_c (가중)
    swc = np.bincount(code, weights=w, minlength=m)
    g_c = (np.bincount(code, weights=w * y, minlength=m)
           / np.maximum(swc, 1e-12)) - (w * y).sum() / w.sum()

    upid, invp = np.unique(pid, return_inverse=True)
    sw_p = np.bincount(invp, weights=w)
    rate_p = np.bincount(invp, weights=w * y) / np.maximum(sw_p, 1e-12)

    key = pid * m + code
    u, inv = np.unique(key, return_inverse=True)
    sw = np.bincount(inv, weights=w)
    sw2 = np.bincount(inv, weights=w * w)
    n_eff = np.where(sw2 > 0, sw * sw / np.maximum(sw2, 1e-12), 0.0)
    t = np.bincount(inv, weights=w * y)
    rc = t / np.maximum(sw, 1e-12)
    rp = rate_p[np.searchsorted(upid, u // m)]
    shrink = n_eff / (n_eff + k)
    dprime = shrink * (rc - rp - g_c[u % m])
    return ({"name": name, "beta": 1.0, "keys": u, "delta": dprime.astype(np.float64)},
            g_c, {"name": name, "beta": 1.0, "keys": u, "delta": n_eff.astype(np.float64)},
            m)


def main():
    df, s, y = D.load_df()
    print(f"train {df.shape}   K={CTX_K}   배포 β {DEP_BETA}\n")

    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        tr = df[s < fold]
        w = recency_weight(tr["season"].to_numpy(), CTX_HALF_LIFE,
                           ref_season=int(s[s < fold].max()))
        print("=" * 92)
        print(f"fold {fold}   기준 BSS {st['base']:.1f}")
        print("=" * 92)

        for name in ("hand", "count4"):
            axp, g_c, axn, m = fit_group(tr, name, CTX_K, w)
            code, _ = CTX_CODERS[name](st["d"])
            xj = st["Xax"][:, st["names"].index(name)]        # 배포 확률차 δ
            drop = -DEP_BETA[name] * xj

            v_grp = ctx_lookup(st["d"], axp) + g_c[code]      # δ_grp 행 값
            v_lg = g_c[code]                                  # 리그 단독 팔
            n_eff_row = ctx_lookup(st["d"], axn)              # 미등록 셀 → 0
            lown = n_eff_row < CTX_K                          # w<0.5 = 저표본 행

            print(f"\n  {name}  (저n 행 {lown.mean():.1%}, g_c 범위 "
                  f"[{g_c.min():+.4f}, {g_c.max():+.4f}])")
            print("    팔        " + " ".join(f"β{b:<6}" for b in BETAS))
            for label, v in (("grp", v_grp), ("리그단독", v_lg)):
                row = [D.rescore(st, drop + b * v) for b in BETAS]
                print(f"    {label:<9} " + " ".join(f"{g:>+7.2f}" for g in row))

            # 저n 부분집합 관점: 배포 대비 grp 의 행별 Brier 개선을 저n/고n 으로 분해
            b_dep = DEP_BETA[name]
            for label, v in (("grp", v_grp), ("리그단독", v_lg)):
                p_var = 1 / (1 + np.exp(-(st["z0"] + drop + b_dep * v)))
                from validation import shift_to_prior
                p_var = shift_to_prior(p_var, st["yv"].mean())
                di = (st["p0"] - st["yv"]) ** 2 - (p_var - st["yv"]) ** 2
                n_all = len(di)
                dl = di[lown].sum() / n_all * 1e5 / (st["yv"].mean() * (1 - st["yv"].mean()))
                dh = di[~lown].sum() / n_all * 1e5 / (st["yv"].mean() * (1 - st["yv"].mean()))
                print(f"    β={b_dep} {label:<9} Δ분해: 저n {dl:+.2f}  고n {dh:+.2f}"
                      f"  (합 {dl+dh:+.2f})")

    print("\n" + "=" * 92)
    print("판정: grp 이 리그단독 팔과 구별되고 저n 성분이 양 fold 양수일 때만 후보.")
    print("사전 예측: 리그 팔과 구별 불가 — 상황 주효과는 이미 피처에 있다 (§7.36 대조군).")


if __name__ == "__main__":
    main()
