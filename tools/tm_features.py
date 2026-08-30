"""트랙맨 투수-시즌 집계 → as-of 피처. **우리 하니스로 직접 만든다.**

왜 다시 만드는가
    원우 §7.39 가 시험한 것은 **릴리스포인트 산포 계열**(`relpoint_disp` · `rels_std` ·
    `relh_std` · `ext_std`)뿐이다. §7.116a E1 이 그 산포들의 인접시즌 상관을 재서
    `rels_std 0.225` · `relpoint_disp 0.326` — **형질이 아니라 상태** 라고 확정했다.

    그런데 **물리량 수준**(구속·회전수·무브먼트·익스텐션의 평균)은 그 표에도 §7.39
    표에도 없다. E1 이 형질로 판정한 것은 `relh_mean`/`rels_mean`(0.98) ·
    `angle`/`abs_rels`/`velo_gap`(0.93/0.87/0.75) 이다. **구속·회전수 평균 자체는
    피처로 시험된 기록이 없다.**

리키지
    시즌 S 행의 피처 = **시즌 < S 의 트랙맨 집계**. 그 투수의 현재/미래 투구를 안 본다.
    매핑은 `control_success` 를 전혀 쓰지 않는다 (구조적 대응, `trackman_map.py`).
    트랙맨은 2019~2024 뿐이므로 2025(test) 행은 ≤2024 집계를 받는다 — 동결 상수다.
"""
import numpy as np
import pandas as pd

PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]
TM_CSV = "/Users/hyejin/Aimers_9th/data/trackman_history.csv"


def load_tm(map_csv):
    m = pd.read_csv(map_csv)
    col = [c for c in m.columns if "trackman" in c.lower()][0]
    pcol = [c for c in m.columns if c != col][0]
    tm = pd.read_csv(TM_CSV, encoding="utf-8-sig",
                     usecols=["season", "pitcher_trackman_id", "pitch_type_group"] + PHYS)
    tm.columns = [c.lstrip("﻿") for c in tm.columns]
    tm = tm.merge(m.rename(columns={col: "pitcher_trackman_id", pcol: "pitcher_id"}),
                  on="pitcher_trackman_id", how="inner")
    return tm


def season_agg(tm):
    """(pitcher_id, season) 별 집계 — 평균 · 표준편차 · 구종 믹스 · 표본수."""
    g = tm.groupby(["pitcher_id", "season"])
    out = g[PHYS].mean().add_prefix("tm_m_")
    out = out.join(g[PHYS].std().add_prefix("tm_s_"))
    out["tm_n"] = g.size()
    mix = (tm.assign(v=1).pivot_table(index=["pitcher_id", "season"],
                                      columns="pitch_type_group", values="v",
                                      aggfunc="sum", fill_value=0))
    mix = mix.div(mix.sum(1), axis=0).add_prefix("tm_mix_")
    return out.join(mix).reset_index()


def asof(agg, seasons):
    """시즌 S 에 대해 **S 미만** 누적 가중평균(투구수 가중)을 만든다."""
    rows = []
    cols = [c for c in agg.columns if c.startswith("tm_") and c != "tm_n"]
    for s in seasons:
        h = agg[agg.season < s]
        if h.empty:
            continue
        w = h["tm_n"].to_numpy(float)
        d = h.groupby("pitcher_id").apply(
            lambda x: pd.Series({c: np.average(x[c], weights=x["tm_n"])
                                 if np.isfinite(x[c]).all() else np.nan for c in cols}
                                | {"tm_hist_n": x["tm_n"].sum()}),
            include_groups=False)
        d["season"] = s
        rows.append(d.reset_index())
    return pd.concat(rows, ignore_index=True)
