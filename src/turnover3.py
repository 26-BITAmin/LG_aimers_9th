# -*- coding: utf-8 -*-
"""이적 투수 분리 분석 (§7.95/7.96 후속).

① 팀 entry 효과를 리그 신인 vs 이적으로 분해
② 이적 투수의 시즌 간 제구율 변화 vs 같은 팀 잔류 투수 (쌍체 대조)
   — 신호가 있으면 test에서 pitcher_id→마지막 팀 조회로 합법 피처가 된다.

팀 귀속은 시즌 내 최빈 팀(modal team). 시즌 중 트레이드는 최빈 팀으로 뭉갠다.
"""
import numpy as np
import pandas as pd

DATA = "data/train.csv"


def main():
    usecols = ["season", "game_type", "pitcher_id", "pitcher_team_id", "control_success"]
    df = pd.read_csv(DATA, usecols=usecols)
    df = df.rename(columns={"control_success": "y"})
    df = df[df["game_type"] == "R"]

    # 투수-시즌 요약: 최빈 팀, 투구수, 제구율
    ps = (df.groupby(["season", "pitcher_id"])
            .agg(n=("y", "size"), p=("y", "mean"),
                 team=("pitcher_team_id", lambda s: s.mode().iloc[0]),
                 nteam=("pitcher_team_id", "nunique"))
            .reset_index())
    multi = ps.groupby("season")["nteam"].apply(lambda s: (s > 1).mean())
    print("시즌 내 복수 팀 투수 비율:",
          " ".join(f"{s}:{v:.1%}" for s, v in multi.items()))

    seasons = sorted(ps["season"].unique())
    print("\n[②] 이적 투수 vs 같은 팀 잔류 투수 — 시즌 간 Δp (쌍체, 투구수 가중)")
    print(f"{'pair':>10s} {'n_mv':>5s} {'pitch%':>7s} {'Δ_mover':>8s} {'Δ_stay':>8s} "
          f"{'차이':>7s} {'SE':>6s} {'t':>6s}")
    agg_rows = []
    for tp, t in zip(seasons[:-1], seasons[1:]):
        a = ps[ps["season"] == tp].set_index("pitcher_id")
        b = ps[ps["season"] == t].set_index("pitcher_id")
        both = a.join(b, lsuffix="_0", rsuffix="_1", how="inner")
        both["d"] = both["p_1"] - both["p_0"]
        both["w"] = np.minimum(both["n_0"], both["n_1"])
        mv = both[both["team_0"] != both["team_1"]]
        st = both[both["team_0"] == both["team_1"]]

        def wmean_se(g):
            m = np.average(g["d"], weights=g["w"])
            v = np.average((g["d"] - m) ** 2, weights=g["w"])
            neff = g["w"].sum() ** 2 / (g["w"] ** 2).sum()
            return m, np.sqrt(v / neff)

        dm, sem = wmean_se(mv)
        ds, ses = wmean_se(st)
        diff = dm - ds
        se = np.sqrt(sem ** 2 + ses ** 2)
        pitch_share = b.loc[b.index.isin(mv.index), "n"].sum() / b["n"].sum()
        print(f"{tp}->{t:>4d} {len(mv):5d} {pitch_share:7.1%} {dm*100:+8.2f} "
              f"{ds*100:+8.2f} {diff*100:+7.2f} {se*100:6.2f} {diff/se:+6.1f}")
        agg_rows.append(dict(diff=diff, se=se, n1=mv["n_1"].sum()))

    ag = pd.DataFrame(agg_rows)
    ivw = (ag["diff"] / ag["se"] ** 2).sum() / (1 / ag["se"] ** 2).sum()
    ivse = np.sqrt(1 / (1 / ag["se"] ** 2).sum())
    print(f"\n  역분산 가중 평균 차이 = {ivw*100:+.2f}%p  SE {ivse*100:.2f}  "
          f"t {ivw/ivse:+.1f}")

    # ① 팀 entry 효과의 신인/이적 분해: entry = Σ_g share_g (p_g − p_stay)
    print("\n[①] 팀 entry 효과 분해 — 리그 신인 vs 이적 (팀-시즌쌍 50개 풀링, %p)")
    rows = []
    for tp, t in zip(seasons[:-1], seasons[1:]):
        prev_league = set(ps.loc[ps["season"] == tp, "pitcher_id"])
        prev_team = ps[ps["season"] == tp].set_index("pitcher_id")["team"]
        cur = df[df["season"] == t].copy()
        cur["prev_team"] = cur["pitcher_id"].map(prev_team)
        for team, cu in cur.groupby("pitcher_team_id"):
            stay = cu["prev_team"] == team
            rookie = ~cu["pitcher_id"].isin(prev_league)
            trans = ~stay & ~rookie
            if not stay.any():
                continue
            p_stay = cu.loc[stay, "y"].mean()
            e_rk = rookie.mean() * (cu.loc[rookie, "y"].mean() - p_stay) if rookie.any() else 0
            e_tr = trans.mean() * (cu.loc[trans, "y"].mean() - p_stay) if trans.any() else 0
            rows.append(dict(pair=f"{tp}->{t}", team=team, n=len(cu),
                             rookie_share=rookie.mean(), trans_share=trans.mean(),
                             e_rookie=e_rk, e_trans=e_tr))
    r = pd.DataFrame(rows)
    for c in ["rookie_share", "trans_share"]:
        print(f"  {c:13s} 중앙값 {r[c].median():.1%}  최대 {r[c].max():.1%}")
    for c in ["e_rookie", "e_trans"]:
        v = r[c] * 100
        print(f"  {c:13s} mean|x|={v.abs().mean():5.2f}%p  std={v.std():5.2f}  "
              f"[{v.min():+6.2f}, {v.max():+6.2f}]")
    # 이적 entry의 팀 지속성
    for c in ["e_trans", "trans_share"]:
        pv = r.pivot(index="team", columns="pair", values=c)
        pairs = sorted(r["pair"].unique())
        cs = [pv[a].corr(pv[b]) for a, b in zip(pairs[:-1], pairs[1:])]
        print(f"  {c:13s} 팀 자기상관 " +
              " ".join(f"{x:+.2f}" for x in cs) + f"  평균 {np.nanmean(cs):+.2f}")


if __name__ == "__main__":
    main()
