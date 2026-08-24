"""§7.160 2차 패스 — **저카디널리티 교호** (탐색적 지위, `resmax.py` 후속).

사전 등록 문구는 "**플래그된 블록의 쌍**만"이었고 플래그가 하나뿐이라 그 트리거는
**미충족**이다. 트리거를 바꾸지 않는다 — 대신 장부의 **한계 조항**("한계 지도는
교호를 놓친다")이 주는 별도 근거로, **탐색적 지위의 소규모 고정 목록**을 돌린다.

목록 (사전 고정, 넷):  카운트×apn5 · F/R×apn5 · 카운트×F/R · **위약 요일×apn5**

### ★ 위약 설계 — 교호에는 "죽은 축과의 교차"가 옳은 영가설이다

교차 블록의 C 는 **두 주효과 + 교호**를 전부 담는다. 그래서 `C(카운트×apn5)` 가
크다고 교호가 있는 것이 아니다 — apn5 주효과만으로도 커진다.

`요일×apn5` 가 정확히 그 영가설이다: **살아 있는 축(apn5)을 죽은 축(요일, §7.78)과
교차**한 것이라 "apn5 주효과 + 교차로 인한 노이즈"만 담고 교호는 0 이다.
→ 판독은 `C(교차) > C(요일×apn5)` 이고, 급별 경기분할 바닥은 보조로만 본다.

실행:  .venv/Scripts/python.exe -u src/resmax3.py 2>&1 | tee experiments/resmax3.log
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np
import pandas as pd

import deploy29 as D
from calslope import build_layers
from calsub import EDGE_SEASON
from predict import PROB_CENTER, PROB_SCALE
from resmax import three_cols
from resmax2 import mg_stats
from validation import shift_to_prior


def main():
    t0 = time.time()
    df, s, y = D.load_df()
    apn_edges = np.quantile(df.loc[s < EDGE_SEASON, "asof_pitcher_n"].to_numpy(float),
                            [.2, .4, .6, .8])
    pairk = (np.minimum(df.pitcher_team_id, df.batter_team_id).astype(np.int64) * 100
             + np.maximum(df.pitcher_team_id, df.batter_team_id))
    key = df.season.astype(np.int64) * 10000 + df.game_month * 100 + df.game_dayofweek
    half = df.inning.astype(np.int64) * 2 + (df.top_bottom == "B")
    game = (((pairk != pairk.shift()) | (key != key.shift())
             | (half < half.shift())).cumsum()).to_numpy()

    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        yv, mm = st["yv"], st["mm"]
        ax = st["Xax"] @ st["dep"]
        _, zb = build_layers(df, s, fold, mm)[-1]
        pb = shift_to_prior(1 / (1 + np.exp(-(zb + ax))), yv.mean())
        f = shift_to_prior(PROB_CENTER + PROB_SCALE * (pb - PROB_CENTER), yv.mean())
        e = yv - f
        d = df.iloc[st["vi"][mm]]
        cl = pd.factorize(game[st["vi"][mm]])[0]
        K = 1e5 / (yv.mean() * (1 - yv.mean()))
        rng = np.random.default_rng(20260824)

        cnt = (d.balls_before * 3 + d.strikes_before).to_numpy().astype(np.int64)
        apn = np.digitize(d.asof_pitcher_n.to_numpy(float), apn_edges).astype(np.int64)
        fr = (d.game_type.to_numpy().astype(str) == "F").astype(np.int64)
        dow = pd.factorize(d.game_dayofweek)[0].astype(np.int64)

        BL = [("카운트 (주효과)", cnt),
              ("apn5 (주효과)", apn),
              ("F/R (주효과)", fr),
              ("카운트×apn5", cnt * 5 + apn),
              ("F/R×apn5", fr * 5 + apn),
              ("카운트×F/R", cnt * 2 + fr),
              ("★위약 요일×apn5", dow * 5 + apn)]

        print(f"\n{'='*94}\nfold{fold}   {len(yv):,}행 · 경기 {cl.max()+1:,}\n{'='*94}")
        print(f"  {'블록':<20s}{'G':>5s}{'m_g 중앙':>9s}{'m<5':>7s}"
              f"{'A 천장':>9s}{'B 표현':>9s}{'C 헤드룸':>10s}{'C(CR0)':>9s}")
        out = {}
        for name, codes in BL:
            codes = codes - codes.min()
            ng = int(codes.max()) + 1
            med, mn, frac = mg_stats(codes, cl, ng)
            (A, Bv, C), (_, _, Cc), G = three_cols(codes, ng, cl, (yv, f, e), rng)
            out[name] = C * K
            print(f"  {name:<20s}{G:>5d}{med:>9d}{frac:>7.1%}"
                  f"{A*K:>9.1f}{Bv*K:>9.1f}{C*K:>10.1f}{Cc*K:>9.1f}"
                  f"   ({time.time()-t0:.0f}s)")

        pl = out["★위약 요일×apn5"]
        print(f"\n  판독 — 위약(요일×apn5) C = {pl:+.1f} 이 이 계열의 바닥이다")
        for nmb in ("카운트×apn5", "F/R×apn5", "카운트×F/R"):
            base = pl if "apn5" in nmb else max(out["카운트 (주효과)"], out["F/R (주효과)"])
            tag = "위약(요일×apn5)" if "apn5" in nmb else "주효과 최대"
            print(f"    {nmb:<14s} C {out[nmb]:>7.1f}  vs {tag} {base:>7.1f}"
                  f"   {'🚩플래그' if out[nmb] > base else '—'}")


if __name__ == "__main__":
    main()
