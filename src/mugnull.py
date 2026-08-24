"""§7.155 부속 — 서브그룹 후보의 **재사용 가능한 문턱**을 만든다.

§7.155 에서 요일 위약이 |t| = 1.77 로 진짜 후보 전부를 이겼다. 그런데 **위약 하나의
t 는 그 자체가 표집 노이즈**다. 무작위 분할을 여러 번 돌려 `max|t|` 의 영분포를
만들어 두면, 앞으로 이 잔차에서 나오는 모든 서브그룹 후보에 같은 문턱을 쓸 수 있다.

### 영가설을 두 종류로 낸다 — 이게 핵심이다

  ① **무작위 행 분할** — "구조가 전혀 없다"의 바닥.
  ② **무작위 경기 분할** — 그룹 안에 현실적인 상관이 있다 (같은 경기 = 같은 투수·
     같은 날). 진짜 후보(hand·mix·team)는 전부 ② 쪽 성질을 가지므로 **②가 옳은 영가설**이다.

①만 쓰면 문턱이 낮게 잡혀 위약을 통과시킨다 — §7.45 의 요일 +22.8 이 그 실패였다.

실행:  .venv/Scripts/python.exe -u src/mugnull.py 2>&1 | tee experiments/mugnull.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np
import pandas as pd

import deploy29 as D
from calslope import EPS, _cluster_cov, _glm, build_layers
from calsub import EDGE_SEASON, make_buckets
from config import RECENCY_HALF_LIFE, SMOOTH_K_PITCHER
from data import recency_weight
from predict import PROB_CENTER, PROB_SCALE
from validation import shift_to_prior

REF_SEASON = 2024
RATE, NCOL, K = "asof_pitcher_success_rate", "asof_pitcher_n", SMOOTH_K_PITCHER
NG, NDRAW = 7, 25


def t_for(codes_tr, codes_va, tr, wtr, mu, yv, off, Dm, shave, n, cl, n_id):
    """그 분할로 μ_g 를 적합해 Δ_g 를 만들고 3항 회귀의 클러스터 t 를 낸다."""
    v = tr[RATE].to_numpy(dtype=np.float64)
    fin = np.isfinite(v)
    mug = np.full(NG, mu)
    for g in range(NG):
        s = fin & (codes_tr == g)
        if s.sum() >= 2000:
            mug[g] = float(np.average(v[s], weights=wtr[s]))
    dg = (mug[codes_va] - mu) * K / (n + K)
    if np.std(dg) < 1e-12:
        return np.nan, float(np.abs(mug - mu).max())
    X = np.column_stack([np.ones(len(yv)), Dm, shave, dg])
    b, m3, H = _glm(X, yv, offset=off)
    V, _ = _cluster_cov(X, yv, m3, H, cl, n_id)
    return float(b[-1] / np.sqrt(V[-1, -1])), float(np.abs(mug - mu).max())


def main():
    df, s, y = D.load_df()
    gt = df["game_type"].to_numpy()
    clean_all = ~((gt == "F") & (s < 2023))
    edges = np.quantile(df.loc[s < EDGE_SEASON, NCOL].to_numpy(float), [.2, .4, .6, .8])
    # 경기 경계 복원 — trackman_map 과 같은 규칙 (§4)
    pair = (np.minimum(df.pitcher_team_id, df.batter_team_id).astype(np.int64) * 100
            + np.maximum(df.pitcher_team_id, df.batter_team_id))
    key = df.season.astype(np.int64) * 10000 + df.game_month * 100 + df.game_dayofweek
    half = df.inning.astype(np.int64) * 2 + (df.top_bottom == "B")
    game = (((pair != pair.shift()) | (key != key.shift())
             | (half < half.shift())).cumsum()).to_numpy()

    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        yv, cl, n_id, mm = st["yv"], st["cl"], st["n_id"], st["mm"]
        ax = st["Xax"] @ st["dep"]
        ybar = yv.mean()
        _, zb = build_layers(df, s, fold, mm)[-1]
        pb = shift_to_prior(1 / (1 + np.exp(-(zb + ax))), ybar)
        p = shift_to_prior(PROB_CENTER + PROB_SCALE * (pb - PROB_CENTER), ybar)
        off = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))

        d = df.iloc[st["vi"][mm]]
        n = d[NCOL].to_numpy(dtype=np.float64)
        r = d[RATE].to_numpy(dtype=np.float64)
        tr = df[(s < fold) & clean_all]
        wtr = recency_weight(tr["season"].to_numpy(), RECENCY_HALF_LIFE,
                             ref_season=REF_SEASON)
        v = tr[RATE].to_numpy(dtype=np.float64)
        fin = np.isfinite(v)
        mu = float(np.average(v[fin], weights=wtr[fin]))
        f = (n * np.nan_to_num(r, nan=mu) + K * mu) / (n + K)
        shave = (f - mu) * K / (n + K)
        bk, _ = make_buckets(df, s, fold, mm, "apn5", edges)
        Dm = np.column_stack([(bk == g).astype(np.float64) for g in range(1, 5)])

        g_tr = game[(s < fold) & clean_all]
        g_va = game[st["vi"][mm]]
        rng = np.random.default_rng(20260824)
        res = {"행 무작위": [], "경기 무작위": []}
        spread = {"행 무작위": [], "경기 무작위": []}
        for i in range(NDRAW):
            ctr = rng.integers(0, NG, len(tr))
            cva = rng.integers(0, NG, len(yv))
            t_, sp = t_for(ctr, cva, tr, wtr, mu, yv, off, Dm, shave, n, cl, n_id)
            res["행 무작위"].append(abs(t_)); spread["행 무작위"].append(sp)

            gid = np.unique(np.concatenate([g_tr, g_va]))
            lab = rng.integers(0, NG, len(gid))
            mp = pd.Series(lab, index=gid)
            t_, sp = t_for(mp.reindex(g_tr).to_numpy(), mp.reindex(g_va).to_numpy(),
                           tr, wtr, mu, yv, off, Dm, shave, n, cl, n_id)
            res["경기 무작위"].append(abs(t_)); spread["경기 무작위"].append(sp)

        print(f"\nfold {fold}   무작위 {NG}분할 × {NDRAW}회   |t| 영분포")
        print(f"  {'영가설':<12s}{'중앙':>8s}{'90%':>8s}{'95%':>8s}{'최대':>8s}"
              f"{'|μ_g−μ| 중앙':>14s}")
        for k_, vlist in res.items():
            a = np.array(vlist, dtype=float)
            a = a[np.isfinite(a)]
            print(f"  {k_:<12s}{np.median(a):>8.2f}{np.percentile(a,90):>8.2f}"
                  f"{np.percentile(a,95):>8.2f}{a.max():>8.2f}"
                  f"{np.median(spread[k_]):>14.5f}")
        print(f"  참고 — §7.155 실측: 요일 위약 |t| = "
              f"{1.77 if fold == 2024 else 0.54:.2f} · 진짜 후보 최대 "
              f"{0.25 if fold == 2024 else 1.14:.2f}")


if __name__ == "__main__":
    main()
