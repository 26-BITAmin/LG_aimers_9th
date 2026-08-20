# -*- coding: utf-8 -*-
"""§7.98 검증 — 유입 구성 신호인가, 평균회귀 인공물인가.

turnover4가 얻은 r(pred_shift, actual_dp)=+0.317 은 기계적으로 생길 수 있다:
  pred_shift = w_in·(phat_in − phat_stay) ∝ −phat_stay
  actual_dp  = p_{t+1} − p_t              ∝ −p_t
  그리고 phat_stay ≈ p_t.  → 유입 정보가 0이어도 양의 상관.

위약: 유입 선수의 정체를 지우고 전원 리그 평균으로 대체 (규모 w_in과 잔류
성적 phat_stay만 남김). 위약이 같은 r을 주면 신호는 유입이 아니다.

결정적 검정: 팀 수준 예측에서 phat_stay(잔류만) 위에 유입 정보가 얹히는가.
"""
import numpy as np
import pandas as pd

DATA = "data/train.csv"
K_EB = 200.0
N_PERM = 5000
SEED = 42


def build(df):
    ps = (df.groupby(["season", "pitcher_id"])
            .agg(n=("y", "size"), p=("y", "mean"),
                 team=("pitcher_team_id", lambda s: s.mode().iloc[0]))
            .reset_index())
    seasons = sorted(ps["season"].unique())
    rows = []
    for tp, t in zip(seasons[:-1], seasons[1:]):
        a = ps[ps["season"] == tp].set_index("pitcher_id")
        lg_prev = df.loc[df["season"] == tp, "y"].mean()
        phat_prev = (a["p"] * a["n"] + lg_prev * K_EB) / (a["n"] + K_EB)

        cur = df[df["season"] == t].copy()
        cur["phat"] = cur["pitcher_id"].map(phat_prev)
        cur["prev_team"] = cur["pitcher_id"].map(a["team"])
        prev_p = df[df["season"] == tp].groupby("pitcher_team_id")["y"].mean()

        for team, cu in cur.groupby("pitcher_team_id"):
            if team not in prev_p.index:
                continue
            stay = (cu["prev_team"] == team).to_numpy()
            rookie = cu["phat"].isna().to_numpy()
            trans = (~stay) & (~rookie)
            inflow = ~stay
            if stay.sum() == 0 or inflow.sum() == 0:
                continue

            phat = cu["phat"].to_numpy().copy()
            phat[rookie] = lg_prev
            w_in = inflow.mean()
            phat_stay = phat[stay].mean()

            # 실제: 유입 선수의 직전 성적 사용 (이적은 옛 팀 성적)
            phat_in = phat[inflow].mean()
            # 위약: 유입 선수 정체 삭제 — 전원 리그 평균
            phat_in_pl = lg_prev
            # 이적만 실제, 신인은 리그평균 (이미 그러함) → 이적 정보의 순수 기여
            phat_in_tr = (phat[trans].sum() + lg_prev * rookie.sum()) / inflow.sum()

            rows.append(dict(
                pair=f"{tp}->{t}", team=team, n=len(cu),
                w_in=w_in, trans_share=trans.mean(), rookie_share=rookie.mean(),
                phat_stay=phat_stay, prev_level=prev_p[team],
                pred_shift=w_in * (phat_in - phat_stay),
                placebo_shift=w_in * (phat_in_pl - phat_stay),
                trans_shift=w_in * (phat_in_tr - phat_stay),
                phat_team=w_in * phat_in + (1 - w_in) * phat_stay,
                actual_level=cu["y"].mean(),
                actual_dp=cu["y"].mean() - prev_p[team],
            ))
    return pd.DataFrame(rows)


def center(d, cols):
    out = d.copy()
    for c in cols:
        out[c] = d[c] - d.groupby("pair")[c].transform("mean")
    return out


def perm_r(d, x, y, rng):
    obs = np.corrcoef(d[x], d[y])[0, 1]
    pos = {ix: i for i, ix in enumerate(d.index)}
    gidx = [np.array([pos[i] for i in g.index]) for _, g in d.groupby("pair")]
    xv, yv = d[x].to_numpy(), d[y].to_numpy()
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        yp = yv.copy()
        for gi in gidx:
            yp[gi] = yv[rng.permutation(gi)]
        null[k] = np.corrcoef(xv, yp)[0, 1]
    return obs, null.std(), (np.abs(null) >= abs(obs)).mean()


def ols(y, X):
    """반환: 계수, SE, R²"""
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c) for c in X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    dof = len(y) - X.shape[1]
    s2 = (r ** 2).sum() / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, se, r2


def main():
    usecols = ["season", "game_type", "pitcher_id", "pitcher_team_id", "control_success"]
    df = pd.read_csv(DATA, usecols=usecols).rename(columns={"control_success": "y"})
    df = df[df["game_type"] == "R"]
    d = build(df)
    cols = ["w_in", "phat_stay", "prev_level", "pred_shift", "placebo_shift",
            "trans_shift", "phat_team", "actual_level", "actual_dp"]
    dc = center(d, cols)
    rng = np.random.default_rng(SEED)

    print(f"팀-시즌쌍 {len(d)}개 (시즌 쌍 내 중심화)\n")
    print("[A] 위약 대조 — 유입 선수의 '정체'가 신호를 만드나")
    for x, lab in [("pred_shift", "실제 (신인+이적 직전 성적)"),
                   ("placebo_shift", "위약 (유입 전원 리그평균 = 정체 삭제)"),
                   ("trans_shift", "이적만 실제 / 신인 평균")]:
        r, s0, p = perm_r(dc, x, "actual_dp", rng)
        print(f"  {lab:38s} r={r:+.3f}  z={r/s0:+.2f}  p={p:.3f}")
    print(f"  corr(pred_shift, placebo_shift) = "
          f"{np.corrcoef(dc['pred_shift'], dc['placebo_shift'])[0,1]:+.3f}")

    print("\n[B] 평균회귀 확인")
    for x, y in [("phat_stay", "prev_level"), ("prev_level", "actual_dp"),
                 ("phat_stay", "actual_dp")]:
        print(f"  corr({x:11s}, {y:12s}) = {np.corrcoef(dc[x], dc[y])[0,1]:+.3f}")

    print("\n[C] 결정적 검정 — 팀 수준 예측에서 유입 정보가 잔류 위에 얹히나")
    y = dc["actual_level"].to_numpy()
    b1, se1, r2_1 = ols(y, [dc["phat_stay"]])
    print(f"  ① actual_level ~ phat_stay            R²={r2_1:.3f}  "
          f"b={b1[1]:+.2f}±{se1[1]:.2f}")
    b2, se2, r2_2 = ols(y, [dc["phat_stay"], dc["pred_shift"]])
    print(f"  ② + pred_shift (유입 정보)             R²={r2_2:.3f}  "
          f"b_shift={b2[2]:+.2f}±{se2[2]:.2f}  t={b2[2]/se2[2]:+.1f}")
    b3, se3, r2_3 = ols(y, [dc["phat_stay"], dc["placebo_shift"]])
    print(f"  ③ + placebo_shift (위약)              R²={r2_3:.3f}  "
          f"b_shift={b3[2]:+.2f}±{se3[2]:.2f}  t={b3[2]/se3[2]:+.1f}")
    print(f"  → 유입 정보의 순수 증분 R² = {r2_2 - r2_1:+.3f} "
          f"(위약 {r2_3 - r2_1:+.3f})")

    print("\n[D] 이적 정보만의 증분 (신인은 사전 정보가 원래 없다)")
    b4, se4, r2_4 = ols(y, [dc["phat_stay"], dc["w_in"], dc["trans_shift"]])
    print(f"  actual_level ~ phat_stay + w_in + trans_shift  R²={r2_4:.3f}  "
          f"b_trans={b4[3]:+.2f}±{se4[3]:.2f}  t={b4[3]/se4[3]:+.1f}")

    print("\n[E] 시즌 쌍별 부호 일치 (관계의 지속성 — 점수가 아니라 관계에)")
    for x in ["pred_shift", "placebo_shift"]:
        rs = [np.corrcoef(g[x], g["actual_dp"])[0, 1] for _, g in dc.groupby("pair")]
        print(f"  {x:14s} " + " ".join(f"{v:+.2f}" for v in rs) +
              f"   양수 {sum(v>0 for v in rs)}/5")

    d.to_csv("experiments/turnover_placebo.csv", index=False)
    print("\nsaved: experiments/turnover_placebo.csv")


if __name__ == "__main__":
    main()
