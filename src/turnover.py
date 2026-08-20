# -*- coding: utf-8 -*-
"""로스터 회전(신규 유입/이탈)이 시즌 평균 제구율 변화에 기여하는 크기 분해.

각 인접 시즌 쌍 (t, t+1)에 대해 정확 분해:
  Δp = Δp_stay                                   (잔류 투수의 within 변화)
     + (1 - s'_{t+1}) (p_new  - p_stay,t+1)       (유입 효과)
     - (1 - s_t     ) (p_exit - p_stay,t  )       (이탈 효과)
s = 잔류 투수의 투구 비중. 세 항의 합 = Δp (항등식).
"""
import numpy as np
import pandas as pd

DATA = "data/train.csv"


def decompose(df, id_col):
    seasons = sorted(df["season"].unique())
    rows = []
    seen_before = set()
    for i, t in enumerate(seasons):
        cur_ids = set(df.loc[df["season"] == t, id_col].unique())
        # 커리어 신규(이전 어느 시즌에도 없음) 비중/성적
        cur = df[df["season"] == t]
        is_career_new = ~cur[id_col].isin(seen_before)
        career_new_share = is_career_new.mean()
        career_new_rate = cur.loc[is_career_new, "y"].mean() if is_career_new.any() else np.nan
        rows_career = dict(season=t, career_new_share=career_new_share,
                           career_new_rate=career_new_rate,
                           overall_rate=cur["y"].mean(), n=len(cur))
        if i > 0:
            tp = seasons[i - 1]
            prev = df[df["season"] == tp]
            prev_ids = set(prev[id_col].unique())
            stay_ids = cur_ids & prev_ids

            prev_stay = prev[id_col].isin(stay_ids)
            cur_stay = cur[id_col].isin(stay_ids)

            p_t, p_t1 = prev["y"].mean(), cur["y"].mean()
            s_t = prev_stay.mean()          # t 시즌에서 (다음 시즌에도 남는) 투수의 투구 비중
            s_t1 = cur_stay.mean()          # t+1 시즌에서 잔류 투수의 투구 비중
            p_stay_t = prev.loc[prev_stay, "y"].mean()
            p_stay_t1 = cur.loc[cur_stay, "y"].mean()
            p_exit = prev.loc[~prev_stay, "y"].mean()
            p_new = cur.loc[~cur_stay, "y"].mean()

            within = p_stay_t1 - p_stay_t
            entry = (1 - s_t1) * (p_new - p_stay_t1)
            exit_ = -(1 - s_t) * (p_exit - p_stay_t)
            rows_career.update(dict(
                pair=f"{tp}->{t}", dp=p_t1 - p_t,
                within=within, entry=entry, exit=exit_,
                resid=(p_t1 - p_t) - (within + entry + exit_),
                new_share=1 - s_t1, exit_share=1 - s_t,
                p_new=p_new, p_exit=p_exit,
                p_stay_t=p_stay_t, p_stay_t1=p_stay_t1,
            ))
        rows.append(rows_career)
        seen_before |= cur_ids
    return pd.DataFrame(rows)


def report(df, id_col, label):
    print(f"\n{'='*90}\n[{label}] 축={id_col}")
    out = decompose(df, id_col)
    pair = out.dropna(subset=["pair"])
    cols = ["pair", "dp", "within", "entry", "exit", "new_share", "exit_share",
            "p_new", "p_exit", "p_stay_t", "p_stay_t1"]
    with pd.option_context("display.width", 200, "display.float_format", "{:+.4f}".format):
        print(pair[cols].to_string(index=False))
    print("\n커리어 신규(이전 시즌 전부 미등장) — 2019는 전원 신규라 제외")
    cn = out[out["season"] > out["season"].min()]
    with pd.option_context("display.float_format", "{:+.4f}".format):
        print(cn[["season", "career_new_share", "career_new_rate", "overall_rate"]]
              .to_string(index=False))
    return out


def main():
    usecols = ["season", "game_type", "pitcher_id", "batter_id", "control_success"]
    df = pd.read_csv(DATA, usecols=usecols)
    df = df.rename(columns={"control_success": "y"})
    print(f"train: {len(df):,} rows, seasons {sorted(df['season'].unique())}")

    r = df[df["game_type"] == "R"]
    report(r, "pitcher_id", "R only (권장: F 라벨 정의 변경 배제)")
    report(df, "pitcher_id", "ALL (R+F)")
    report(r, "batter_id", "R only")


if __name__ == "__main__":
    main()
