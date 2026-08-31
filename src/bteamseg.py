"""§7.160 의 유일한 플래그 `batter_team` 을 §7.125 의 transfer 기계에 건다 (`src/bteamseg.py`).

§7.160 장부: `타자팀` C f2023 43.0(바닥 11.0) / f2024 11.5(바닥 7.9) — 16블록 중
**양 fold 바닥 위 유일**. 그런데 §7.125 의 19축 목록에 `pitcher_team` 은 있고
**`batter_team` 이 없다.** C 는 *한계 상한*이고 transfer 는 *실현값*이라 둘은 다른 수다.
재학습 0회로 잴 수 있는데 안 쟀다.

§7.221 이 찾은 결함을 그대로 반영한다:
  · 위약은 **타자** 쪽으로 (투수 위약은 타자 축에 기하가 안 맞는다)
  · fold2023 은 `clean` 마스크가 prev(F 0%) 와 valid(F 있음) 를 갈라놔 해석 불가 →
    **F 몫 표를 같이 찍고 fold2024 를 주 판단 근거로 읽는다**
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np                                              # noqa: E402

import deploy29 as D                                            # noqa: E402
from bexpseg import transfer                                    # noqa: E402
from config import SEED, TARGET_COL                             # noqa: E402

B_DRAW = 200


def main():
    df, s, y = D.load_df()
    gt = df["game_type"].to_numpy()
    clean = ~((gt == "F") & (s < 2023))
    rng = np.random.default_rng(SEED)

    for fold in (2023, 2024):
        st = D.state(df, s, y, fold)
        d = st["d"]
        prev = df[(s == fold - 1) & clean]
        y_prev = prev[TARGET_COL].to_numpy(dtype=np.float64)
        print("\n" + "=" * 92)
        print(f"fold {fold}   n={len(st['yv']):,}   기준 BSS {st['base']:.2f}   "
              f"참조행(={fold-1}) {len(prev):,}")

        for col, ent in (("batter_team_id", "batter_id"),
                         ("pitcher_team_id", "pitcher_id")):
            u = np.unique(np.concatenate([d[col].to_numpy(), prev[col].to_numpy()]))
            lut = {v: i for i, v in enumerate(u)}
            gv = np.array([lut[v] for v in d[col].to_numpy()])
            gp = np.array([lut[v] for v in prev[col].to_numpy()])
            G = len(u)
            (a1, a5), w, pg, rg, rp = transfer(st, gv, gp, y_prev, G)
            print(f"\n  {col}  G={G}   transfer α=1.0 {a1:+8.2f}   α=0.5 {a5:+8.2f}")

            # 기하 정합 위약 — 같은 엔티티를 G 개 그룹으로 무작위 영구 분할
            eid = d[ent].to_numpy().astype(np.int64)
            eidp = prev[ent].to_numpy().astype(np.int64)
            ue = np.unique(np.concatenate([eid, eidp]))
            draws = []
            for _ in range(B_DRAW):
                lab = rng.integers(0, G, size=len(ue))
                m = dict(zip(ue, lab))
                (x1,), *_ = transfer(st, np.array([m[e] for e in eid]),
                                     np.array([m[e] for e in eidp]),
                                     y_prev, G, alphas=(1.0,))
                draws.append(x1)
            dr = np.array(draws)
            lo, md, hi = np.percentile(dr, [2.5, 50, 97.5])
            print(f"    위약 [{ent} 무작위 {G}분할, B={B_DRAW}]  중앙 {md:+8.2f}"
                  f"   95% [{lo:+.2f}, {hi:+.2f}]")
            print(f"    → 후보 − 위약중앙 = {a1 - md:+8.2f}"
                  f"   위약 초과 분위 {100 * (dr < a1).mean():.1f}%")

            # F 몫 정합 (§7.221) — 상위 4개 그룹만
            gtv, gtp = d["game_type"].to_numpy(), prev["game_type"].to_numpy()
            big = np.argsort(-np.bincount(gv, minlength=G))[:4]
            fv = [(gtv[gv == i] == "F").mean() for i in big]
            fp = [(gtp[gp == i] == "F").mean() if (gp == i).any() else np.nan
                  for i in big]
            print("    F몫 valid/prev (최대 4그룹)  "
                  + "  ".join(f"{a:.3f}/{b:.3f}" for a, b in zip(fv, fp)))


if __name__ == "__main__":
    main()
