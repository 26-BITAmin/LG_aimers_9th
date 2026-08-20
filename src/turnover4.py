# -*- coding: utf-8 -*-
"""유입(신인+이적) 합산 구성 변화 vs 팀 수준 (§7.95~97 후속).

§7.96의 entry 항은 신규 선수의 *그 시즌* 성적을 썼다 — 사후 분해지 예측이 아니다.
여기서는 **직전 시즌 정보만으로** 팀의 구성 기반 예측 수준을 만든다:

  p̂_team,t+1 = Σ_g w_g · p̂_g       (w = t+1 실현 투구 비중)
    잔류  p̂ = 그 투수의 t 시즌 성적 (EB 수축, K=200)
    이적  p̂ = 그 투수의 t 시즌 성적 (= 옛 팀에서의 성적, 동일 수축)
    신인  p̂ = t 시즌 리그 평균 (사전 정보 없음; §7.95가 신인≈잔류를 확인)

리그 공통 드리프트는 **시즌 쌍 내 팀 간 중심화**로 제거한다 → 순수 횡단면 질문.
영가설은 시즌 쌍 내 팀 라벨 순열 (죽은 축 대신 정확한 순열 영분포).
"""
import numpy as np
import pandas as pd

DATA = "data/train.csv"
K_EB = 200.0          # 배포 K (§7.90)
N_PERM = 5000
SEED = 42


def build(df, rookie_offset=0.0):
    ps = (df.groupby(["season", "pitcher_id"])
            .agg(n=("y", "size"), p=("y", "mean"),
                 team=("pitcher_team_id", lambda s: s.mode().iloc[0]))
            .reset_index())
    seasons = sorted(ps["season"].unique())
    rows = []
    for tp, t in zip(seasons[:-1], seasons[1:]):
        a = ps[ps["season"] == tp].set_index("pitcher_id")
        lg_prev = df.loc[df["season"] == tp, "y"].mean()
        # EB 수축: 직전 시즌 성적을 그 시즌 리그 평균으로
        phat_prev = (a["p"] * a["n"] + lg_prev * K_EB) / (a["n"] + K_EB)
        prev_team = a["team"]

        cur = df[df["season"] == t].copy()
        cur["phat"] = cur["pitcher_id"].map(phat_prev)
        cur["prev_team"] = cur["pitcher_id"].map(prev_team)
        prev_p = df[df["season"] == tp].groupby("pitcher_team_id")["y"].mean()

        for team, cu in cur.groupby("pitcher_team_id"):
            if team not in prev_p.index:
                continue
            stay = (cu["prev_team"] == team).to_numpy()
            rookie = cu["phat"].isna().to_numpy()            # 직전 시즌 리그 미등장
            trans = ~stay & ~rookie
            inflow = ~stay                                    # 신인 + 이적 (합산)
            if stay.sum() == 0 or inflow.sum() == 0:
                continue

            phat = cu["phat"].to_numpy().copy()
            phat[rookie] = lg_prev + rookie_offset            # 신인 사전값

            w_in = inflow.mean()
            phat_stay = phat[stay].mean()
            phat_in = phat[inflow].mean()

            rows.append(dict(
                pair=f"{tp}->{t}", team=team, n=len(cu),
                inflow_share=w_in,
                rookie_share=rookie.mean(), trans_share=trans.mean(),
                phat_team=w_in * phat_in + (1 - w_in) * phat_stay,
                # 구성 채널이 예측하는 이동 (직전 정보만)
                pred_shift=w_in * (phat_in - phat_stay),
                actual_level=cu["y"].mean(),
                actual_dp=cu["y"].mean() - prev_p[team],
            ))
    return pd.DataFrame(rows)


def center(d, cols):
    """시즌 쌍 내 팀 간 중심화 — 리그 드리프트/시즌 효과 제거."""
    out = d.copy()
    for c in cols:
        out[c] = d[c] - d.groupby("pair")[c].transform("mean")
    return out


def perm_test(d, xcol, ycol, rng):
    """시즌 쌍 내 팀 라벨 순열 영분포."""
    obs = np.corrcoef(d[xcol], d[ycol])[0, 1]
    groups = [g.index.to_numpy() for _, g in d.groupby("pair")]
    y = d[ycol].to_numpy()
    x = d[xcol].to_numpy()
    pos = {ix: i for i, ix in enumerate(d.index)}
    gidx = [np.array([pos[i] for i in g]) for g in groups]
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        yp = y.copy()
        for gi in gidx:
            yp[gi] = y[rng.permutation(gi)]
        null[k] = np.corrcoef(x, yp)[0, 1]
    p = (np.abs(null) >= abs(obs)).mean()
    return obs, null.std(), p


def main():
    usecols = ["season", "game_type", "pitcher_id", "pitcher_team_id", "control_success"]
    df = pd.read_csv(DATA, usecols=usecols).rename(columns={"control_success": "y"})
    df = df[df["game_type"] == "R"]

    d = build(df)
    print(f"팀-시즌쌍 {len(d)}개\n")

    print("[유입 = 신인 + 이적, 합산 규모 (t+1 투구 비중)]")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(d.groupby("pair")[["inflow_share", "rookie_share", "trans_share"]]
              .agg(["min", "median", "max"]).to_string())

    cols = ["inflow_share", "phat_team", "pred_shift", "actual_level", "actual_dp"]
    dc = center(d, cols)
    rng = np.random.default_rng(SEED)

    print("\n[상관 — 시즌 쌍 내 중심화 후 풀링, 순열 검정]")
    tests = [
        ("pred_shift", "actual_dp", "구성 예측 이동 → 실제 수준 변화 ★ 핵심"),
        ("phat_team", "actual_level", "구성 예측 수준 → 실제 수준"),
        ("inflow_share", "actual_dp", "유입 규모 → 수준 변화"),
        ("inflow_share", "actual_level", "유입 규모 → 수준"),
    ]
    for x, y, lab in tests:
        r, s0, p = perm_test(dc, x, y, rng)
        print(f"  {lab:34s} r={r:+.3f}  영분포SD {s0:.3f}  z={r/s0:+.2f}  p={p:.3f}")

    print("\n[구성 채널이 실제 이동의 몇 %를 설명하나 — 회귀 기울기]")
    x, y = dc["pred_shift"], dc["actual_dp"]
    b = (x * y).sum() / (x * x).sum()
    resid = y - b * x
    se = np.sqrt((resid ** 2).sum() / (len(x) - 1) / (x * x).sum())
    print(f"  기울기 {b:+.2f} ± {se:.2f} (t {b/se:+.1f})   ※ 1.0이면 구성이 그대로 전이")
    print(f"  pred_shift 산포 {x.std()*100:.2f}%p vs actual_dp 산포 {y.std()*100:.2f}%p"
          f"  (비 {x.std()/y.std():.2f})")

    print("\n[신인 사전값 민감도 — rookie_offset 스윕]")
    for off in [-0.02, -0.01, 0.0, +0.01]:
        dd = center(build(df, rookie_offset=off), cols)
        r = np.corrcoef(dd["pred_shift"], dd["actual_dp"])[0, 1]
        print(f"  offset {off:+.2f}: r(pred_shift, actual_dp) = {r:+.3f}")

    d.to_csv("experiments/turnover_combined.csv", index=False)
    print("\nsaved: experiments/turnover_combined.csv")


if __name__ == "__main__":
    main()
