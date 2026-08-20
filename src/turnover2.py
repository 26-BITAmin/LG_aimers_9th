# -*- coding: utf-8 -*-
"""팀별 로스터 회전 — 유입 규모의 팀 간 편차와 팀 레벨 변화의 관계.

§7.95(리그 합계: within 지배)의 팀 단위 후속. 팀 레벨에서는
  - "팀 신규" = 지난 시즌 그 팀에 없던 투수 (리그 신인 + 트레이드/이적 포함)
  - "리그 신규" = 지난 시즌 리그 전체에 없던 투수
를 구분한다. 팀마다 유입 비중이 다르므로, 유입 비중이 팀 Δp를 설명하는지 본다.
"""
import numpy as np
import pandas as pd

DATA = "data/train.csv"


def team_decompose(df):
    """시즌 쌍 × 팀 단위 정확 분해. 반환: 팀-시즌쌍 행 단위 DataFrame."""
    seasons = sorted(df["season"].unique())
    rows = []
    for tp, t in zip(seasons[:-1], seasons[1:]):
        prev = df[df["season"] == tp]
        cur = df[df["season"] == t]
        league_prev_ids = set(prev["pitcher_id"].unique())
        for team in sorted(set(cur["pitcher_team_id"]) & set(prev["pitcher_team_id"])):
            pv = prev[prev["pitcher_team_id"] == team]
            cu = cur[cur["pitcher_team_id"] == team]
            prev_ids = set(pv["pitcher_id"].unique())
            cur_stay = cu["pitcher_id"].isin(prev_ids)       # 팀 기준 잔류
            prev_stay = pv["pitcher_id"].isin(set(cu["pitcher_id"].unique()))
            new_mask = ~cur_stay
            league_new = new_mask & ~cu["pitcher_id"].isin(league_prev_ids)

            p_t, p_t1 = pv["y"].mean(), cu["y"].mean()
            s_t, s_t1 = prev_stay.mean(), cur_stay.mean()
            p_stay_t = pv.loc[prev_stay, "y"].mean() if prev_stay.any() else np.nan
            p_stay_t1 = cu.loc[cur_stay, "y"].mean() if cur_stay.any() else np.nan
            p_exit = pv.loc[~prev_stay, "y"].mean() if (~prev_stay).any() else np.nan
            p_new = cu.loc[new_mask, "y"].mean() if new_mask.any() else np.nan

            within = p_stay_t1 - p_stay_t
            entry = (1 - s_t1) * (p_new - p_stay_t1) if new_mask.any() else 0.0
            exit_ = -(1 - s_t) * (p_exit - p_stay_t) if (~prev_stay).any() else 0.0
            rows.append(dict(
                pair=f"{tp}->{t}", team=team, n=len(cu),
                dp=p_t1 - p_t, within=within, entry=entry, exit=exit_,
                team_new_share=new_mask.mean(),
                league_new_share=league_new.mean(),
                transfer_share=new_mask.mean() - league_new.mean(),
                p_new=p_new, p_stay_t1=p_stay_t1,
            ))
    return pd.DataFrame(rows)


def main():
    usecols = ["season", "game_type", "pitcher_id", "pitcher_team_id", "control_success"]
    df = pd.read_csv(DATA, usecols=usecols)
    df = df.rename(columns={"control_success": "y"})
    df = df[df["game_type"] == "R"]
    print(f"R only: {len(df):,} rows, 팀 수 = {df['pitcher_team_id'].nunique()}")

    td = team_decompose(df)

    # 1) 팀 간 유입 비중 편차
    print("\n[1] 팀 신규 투구 비중의 팀 간 분포 (시즌 쌍별)")
    g = td.groupby("pair").agg(
        teams=("team", "size"),
        new_min=("team_new_share", "min"), new_med=("team_new_share", "median"),
        new_max=("team_new_share", "max"),
        lgnew_med=("league_new_share", "median"),
        transfer_med=("transfer_share", "median"),
    )
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(g.to_string())

    # 2) 팀 레벨에서도 within이 지배하나
    print("\n[2] 팀-시즌쌍 단위 분해 항 크기 (|평균|과 산포, %p)")
    for c in ["dp", "within", "entry", "exit"]:
        v = td[c] * 100
        print(f"  {c:7s} mean|x|={v.abs().mean():5.2f}  std={v.std():5.2f}  "
              f"[{v.min():+6.2f}, {v.max():+6.2f}]")

    # 3) 유입 비중이 팀 Δp / entry를 설명하나 (팀-시즌쌍 풀링, n 가중)
    print("\n[3] 상관 (팀-시즌쌍 풀링, 투구수 가중)")
    w = td["n"]
    def wcorr(a, b):
        a, b = td[a], td[b]
        am, bm = np.average(a, weights=w), np.average(b, weights=w)
        cov = np.average((a - am) * (b - bm), weights=w)
        return cov / np.sqrt(np.average((a - am) ** 2, weights=w)
                             * np.average((b - bm) ** 2, weights=w))
    for x, y in [("team_new_share", "dp"), ("team_new_share", "entry"),
                 ("league_new_share", "dp"), ("transfer_share", "dp"),
                 ("team_new_share", "within")]:
        print(f"  corr({x:16s}, {y:6s}) = {wcorr(x, y):+.3f}")

    # 4) 유입 비중 상/하위 팀의 다음 시즌 Δp 대비 (시즌쌍 내 중앙값 기준 이분)
    print("\n[4] 시즌쌍 내 유입 비중 상위 절반 vs 하위 절반의 Δp (%p, n 가중)")
    rows = []
    for pair, sub in td.groupby("pair"):
        med = sub["team_new_share"].median()
        hi, lo = sub[sub["team_new_share"] > med], sub[sub["team_new_share"] <= med]
        rows.append(dict(
            pair=pair,
            dp_hi=100 * np.average(hi["dp"], weights=hi["n"]),
            dp_lo=100 * np.average(lo["dp"], weights=lo["n"]),
        ))
    r4 = pd.DataFrame(rows)
    r4["gap"] = r4["dp_hi"] - r4["dp_lo"]
    with pd.option_context("display.float_format", "{:+.2f}".format):
        print(r4.to_string(index=False))
    print(f"  gap 평균 {r4['gap'].mean():+.2f}%p")

    td.to_csv("experiments/turnover_team.csv", index=False)
    print("\nsaved: experiments/turnover_team.csv")


if __name__ == "__main__":
    main()
