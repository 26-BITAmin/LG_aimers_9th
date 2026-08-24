"""§7.160 정정 — 엔티티 블록의 **위약 바닥을 다시 만든다** (`resmax.py` 후속).

`resmax.py` 의 `수백~수천 급` 바닥(G=400 무작위 경기 분할)이 **무효**다.

    위약 C 중앙 = +302.6 / +247.5

노이즈 차감이 옳다면 위약은 0 근처여야 한다. 원인은 신호가 아니라 **내 노이즈
추정의 붕괴**다: G=400 이면 그룹당 경기가 **2개**뿐이고, `m=2` 에서 부트스트랩
분산은 `(m−1)/m · s²/m = s²/4` 로 참값 `s²/2` 의 절반이라 노이즈가 절반만 차감된다.

**그런데 진짜 엔티티 블록은 `m_g` 가 수십이다** — 부트스트랩이 멀쩡하다.
같은 자로 잰 것이 잘못이었다.

### 옳은 엔티티 위약 — 등판 순열 (기하 보존)

`(경기, 엔티티)` **등판 목록**을 만들고 그 위에서 **엔티티 라벨만 순열**한다.

  · 각 가짜 엔티티는 실제 엔티티와 **똑같은 등판 수**를 갖는다 (라벨 다중집합 보존)
  · 여러 경기에 걸친 다중-경기 구조가 그대로 유지된다 → `m_g` 분포가 같다
  · 그런데 **경기를 가로지르는 정체는 파괴된다** — 가짜 투수는 매 등판마다 다른
    실제 투수의 행을 받는다

즉 "이 축에 지속적 개체 신호가 없다면 C 는 얼마인가"의 정직한 영가설이고,
`resmax.py` 의 경기 분할과 달리 **클러스터 기하가 실제 블록과 같다.**

실행:  .venv/Scripts/python.exe -u src/resmax2.py 2>&1 | tee experiments/resmax2.log
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
from predict import PROB_CENTER, PROB_SCALE
from resmax import three_cols
from validation import shift_to_prior

N_PLACEBO = 25
ENT = [("투수 정체", "pitcher_id"), ("타자 정체", "batter_id"),
       ("투수팀", "pitcher_team_id"), ("타자팀", "batter_team_id")]


def mg_stats(codes, cl, ng):
    """그룹당 클러스터(경기) 수 분포 — 부트스트랩이 성립하는지의 진단."""
    key = codes.astype(np.int64) * (cl.max() + 1) + cl
    uk = np.unique(key)
    g = uk // (cl.max() + 1)
    m = np.bincount(g, minlength=ng)
    m = m[m > 0]
    return int(np.median(m)), int(m.min()), float((m < 5).mean())


def main():
    t0 = time.time()
    df, s, y = D.load_df()
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

        print(f"\n{'='*100}\nfold{fold}   {len(yv):,}행 · 경기 {cl.max()+1:,}\n{'='*100}")
        print(f"  {'블록':<12s}{'G':>5s}{'m_g 중앙':>9s}{'m_g 최소':>9s}"
              f"{'m<5 비율':>9s}{'실측 C':>9s}{'★등판순열 바닥(95%)':>21s}{'판정':>10s}")
        for name, col in ENT:
            codes = pd.factorize(d[col])[0]
            ng = int(codes.max()) + 1
            med, mn, frac = mg_stats(codes, cl, ng)
            (_, _, C), _, G = three_cols(codes, ng, cl, (yv, f, e), rng)
            C *= K

            # ── 등판 순열 위약 ──────────────────────────────────────
            app_key = codes.astype(np.int64) * (cl.max() + 1) + cl
            uk, inv = np.unique(app_key, return_inverse=True)
            app_ent = (uk // (cl.max() + 1)).astype(np.int64)     # 등판별 실제 엔티티
            vals = []
            for _ in range(N_PLACEBO):
                perm = rng.permutation(app_ent)                   # 라벨만 순열
                cd = perm[inv]
                (_, _, Cp), _, _ = three_cols(cd, ng, cl, (yv, f, e), rng, B=400)
                vals.append(Cp * K)
            fl = float(np.percentile(vals, 95))
            mark = "🚩플래그" if C > fl else "—"
            print(f"  {name:<12s}{G:>5d}{med:>9d}{mn:>9d}{frac:>9.1%}"
                  f"{C:>9.1f}{fl:>21.1f}{mark:>10s}   ({time.time()-t0:.0f}s)")
            print(f"     위약 분포: 중앙 {np.median(vals):+.1f}  "
                  f"최소 {min(vals):+.1f}  최대 {max(vals):+.1f}")


if __name__ == "__main__":
    main()
