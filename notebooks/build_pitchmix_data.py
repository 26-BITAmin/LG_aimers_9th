#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""투수별 구종 시계열 + 최근 창(1년/6개월/1개월) 집계 → JSON (HTML 탐색기용)."""
from __future__ import annotations
import json, sys
import numpy as np
import pandas as pd

CANON = {"fastball": "Fastball", "four-seam": "Fastball", "sinker": "Sinker",
         "cutter": "Cutter", "slider": "Slider", "sweeper": "Slider",
         "curveball": "Curveball", "slurve": "Curveball",
         "changeup": "ChangeUp", "splitter": "Splitter"}
ORDER = ["Fastball", "Slider", "Curveball", "ChangeUp", "Splitter", "Sinker", "Cutter"]
WINDOWS = [("1y", 365), ("6m", 182), ("1m", 30)]
PWINDOWS = [50, 100, 150, 200]     # 투구 수 기준 창


def parse_dates(s: pd.Series) -> pd.Series:
    """2019–21은 MM/DD/YYYY, 2022–24는 YYYY-MM-DD 로 포맷이 섞여 있다."""
    s = s.astype(str)
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}$")
    d = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    d[iso] = pd.to_datetime(s[iso], format="%Y-%m-%d", errors="coerce")
    d[~iso] = pd.to_datetime(s[~iso], format="%m/%d/%Y", errors="coerce")
    return d


def main(src: str, out: str, min_pitch: int = 30) -> None:
    tm = pd.read_csv(src, usecols=["pitcher_trackman_id", "game_date", "season",
                                   "tagged_pitch_type", "pitcher_hand", "pitcher_team",
                                   "trackman_game_id", "pitch_no", "inning", "top_bottom"])
    tm["d"] = parse_dates(tm["game_date"])
    tm["pt"] = tm["tagged_pitch_type"].astype(str).str.strip().str.lower().map(CANON)
    n_drop = int(tm["pt"].isna().sum())
    tm = tm[tm["pt"].notna() & tm["d"].notna()].copy()
    tm["ti"] = pd.Categorical(tm["pt"], categories=ORDER, ordered=True).codes

    ym = tm["d"].dt.to_period("M")
    months = [str(p) for p in sorted(ym.unique())]
    mi = {m: i for i, m in enumerate(months)}
    tm["mi"] = ym.astype(str).map(mi).astype("int32")
    tm["si"] = tm["season"].astype("int32")
    tm["cm"] = tm["d"].dt.month.astype("int32")          # 달력 월(3~11)

    data_max = tm["d"].max()
    tot = tm.groupby("pitcher_trackman_id").size()
    keep = tot[tot >= min_pitch].index
    tm = tm[tm["pitcher_trackman_id"].isin(keep)]
    print(f"투수 {len(keep):,}명 / 투구 {len(tm):,} · 구종 미분류 {n_drop:,} 제외 "
          f"· 월 버킷 {len(months)} ({months[0]}~{months[-1]}) · 데이터 마지막 {data_max.date()}")

    meta = (tm.sort_values("d").groupby("pitcher_trackman_id")
            .agg(hand=("pitcher_hand", "last"), team=("pitcher_team", "last"),
                 first=("d", "min"), last=("d", "max"), n=("ti", "size")))
    seasons = sorted(tm["si"].unique().tolist())
    cmonths = sorted(tm["cm"].unique().tolist())

    def window_counts(sub: pd.DataFrame, base: pd.Timestamp) -> dict:
        out = {}
        for name, days in WINDOWS:
            lo = base - pd.Timedelta(days=days)
            w = sub[(sub["d"] > lo) & (sub["d"] <= base)]
            c = np.bincount(w["ti"].to_numpy(), minlength=len(ORDER))
            out[name] = {"c": [int(v) for v in c],
                         "from": str(lo.date()), "to": str(base.date())}
        return out

    # 이동 누적(각 시점의 직전 N일)을 그리려면 커리어 전 기간의 일 단위 해상도가 필요하다.
    # 연-월 / 시즌 / 달력월 집계는 이 일별 배열에서 클라이언트가 직접 만든다(중복 제거).
    EPOCH = pd.Timestamp("2019-01-01")
    tm["day"] = (tm["d"] - EPOCH).dt.days.astype("int32")

    # ── 선발 판정 ────────────────────────────────────────────────────────
    # 각 (경기, 수비쪽)에서 1회에 나온 투수 중 pitch_no 가 가장 작은 사람이 선발이다.
    # 검증: 5,980경기에서 11,903건(경기당 1.99, 정상값 2) · 같은 경기 중복 0건 ·
    #       등판당 투구 수 중앙값 선발 86구 vs 불펜 16구로 뚜렷이 갈린다.
    i1 = tm[tm["inning"] == 1]
    st_idx = i1.groupby(["trackman_game_id", "top_bottom"])["pitch_no"].idxmin()
    st = tm.loc[st_idx]
    starts = (st.groupby(["pitcher_trackman_id", "day"]).size()
              .rename("k").reset_index())
    print(f"선발 등판 {len(st):,}건 / {tm['trackman_game_id'].nunique():,}경기")
    starts_by_pid = {pid: [[int(d), int(k)] for d, k in zip(g["day"], g["k"])]
                     for pid, g in starts.groupby("pitcher_trackman_id")}

    # 투구 수 기준 창(최근 50/100/150/200구)은 하루를 쪼개야 하므로 투구 순서가 필요하다.
    # pitch_no 로 경기 내 순서가 잡히고(결측 0건), 같은 날 2경기는 104건뿐이라
    # (day, trackman_game_id, pitch_no) 정렬이면 순서가 확정된다.
    tm = tm.sort_values(["pitcher_trackman_id", "day", "trackman_game_id", "pitch_no"])

    pitchers = {}
    for pid, sub in tm.groupby("pitcher_trackman_id", sort=False):
        m = meta.loc[pid]
        dg = (sub.groupby(["day", "ti"]).size().unstack(fill_value=0)
              .reindex(columns=range(len(ORDER)), fill_value=0))
        # 같은 날 2경기(더블헤더) 등판이 56,634건 중 104건 있으므로 날짜가 아니라
        # trackman_game_id 를 세어야 '경기 수'가 정확해진다.
        gd = sub.groupby("day")["trackman_game_id"].nunique()
        pitchers[str(pid)] = {
            "hand": str(m["hand"]), "team": str(m["team"]), "n": int(m["n"]),
            "first": str(m["first"].date()), "last": str(m["last"].date()),
            "lastDay": int((m["last"] - EPOCH).days),
            # [day, c0..c6, 경기 수]
            "daily": [[int(d)] + [int(v) for v in row] + [int(gd.loc[d])]
                      for d, row in dg.iterrows()],
            # 구종 코드(0~6) 시퀀스 — daily 의 날짜별 합계가 이 문자열의 경계를 준다
            "seq": "".join(map(str, sub["ti"].to_numpy())),
            "starts": starts_by_pid.get(pid, []),
            "win": {"data": window_counts(sub, data_max),
                    "self": window_counts(sub, m["last"])},
        }

    order = [str(i) for i in meta.sort_values("n", ascending=False).index]
    payload = {
        "types": ORDER,
        "dataMax": str(data_max.date()), "order": order, "pitchers": pitchers,
        "epoch": "2019-01-01",
        "dataMaxDay": int((data_max - pd.Timestamp("2019-01-01")).days),
        "nDrop": n_drop, "minPitch": min_pitch,
        "windows": [{"key": k, "days": d} for k, d in WINDOWS],
        "pwindows": PWINDOWS,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
    import os
    print(f"저장: {out}  ({os.path.getsize(out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
