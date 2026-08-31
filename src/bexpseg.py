"""§7.125 `batter_exp` 재개봉 — **타자 축을 투수 위약으로 판정했다** (`src/bexpseg.py`).

§7.125 는 19축 전수 스캔에서 `batter_exp`(= `asof_batter_n` 5분위) 를
f2023 −20.08 / f2024 +3.96 으로 재고 **"두 fold 양수" 규칙에 걸어 기각**했다.
그 표의 G=5 위약(`pid5`)은 **−26.4 / −29.4** 다 — 영가설이 0 이 아니다.
그리고 위약 셋이 전부 투수/달력이라 **타자 축의 바닥을 한 번도 안 쟀다**
(§7.57 이 "타자 축에는 타자 대조군" 이라고 먼저 써 놓은 규칙을 어겼다).

여기서 세 가지를 잰다 — 전부 재학습 0회.

  A. 재현        segscan 의 transfer 팔을 그대로 (−20.08 / +3.96 이 나와야 한다)
  B. 타자 위약   bid5 · bid2 · brand5(타자당 동결 난수 5분위, B draw) 를 추가
  C. 기하 진단   prev/valid 의 빈 구성·레벨이 어긋나는가 (누적 카운터 축의 함정)
  D. 안정성      빈별 실제 성공률이 시즌 간 표류인가 진동인가 (라벨만, 6시즌)

transfer 팔 (segscan 과 동일):
    bh_g = p̄_g(valid) − r̄_g(prev) ,  dev = bh − Σ w·bh ,  p = p0 − α·dev[g]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

import deploy29 as D                                            # noqa: E402
from config import SEED, TARGET_COL                             # noqa: E402
from validation import bss, shift_to_prior                      # noqa: E402

B_DRAW = 200
ALPHAS = (1.0, 0.5)


def qcodes(v, edges):
    return np.searchsorted(edges, v, side="right")


def edges_for(train_lt, col):
    v = train_lt[col].to_numpy(dtype=np.float64)
    return np.quantile(v[np.isfinite(v)], [0.2, 0.4, 0.6, 0.8])


def transfer(st, g_val, g_prev, y_prev, G, alphas=ALPHAS, reshift=False):
    """§7.125 transfer 팔. reshift=True 면 레벨을 다시 풀어 refinement 만 남긴다."""
    p0, yv, n = st["p0"], st["yv"], len(st["yv"])
    cntv = np.bincount(g_val, minlength=G)
    w = cntv / n
    pg = np.bincount(g_val, weights=p0, minlength=G) / np.maximum(cntv, 1)
    rg = np.bincount(g_val, weights=yv, minlength=G) / np.maximum(cntv, 1)
    cp = np.bincount(g_prev, minlength=G)
    rp = np.where(cp > 0,
                  np.bincount(g_prev, weights=y_prev, minlength=G) / np.maximum(cp, 1),
                  rg)
    bh = pg - rp
    dev = bh - np.sum(w * bh)
    out = []
    for a in alphas:
        p = np.clip(p0 - a * dev[g_val], 1e-6, 1 - 1e-6)
        if reshift:
            p = shift_to_prior(p, yv.mean())
        out.append(bss(yv, p) - st["base"])
    return out, w, pg, rg, rp


def main():
    df, s, y = D.load_df()
    gt = df["game_type"].to_numpy()
    clean = ~((gt == "F") & (s < 2023))
    rng = np.random.default_rng(SEED)

    # ── D. 안정성 (라벨만, 모델 무관) ────────────────────────────────
    print("=" * 92)
    print("D. 빈별 실제 성공률 — 시즌 간 표류인가 진동인가 (clean 행, 라벨만)")
    print("   빈 경계는 각 fold 의 `train < fold` 에서 얼린다 (segscan 규약)")
    for fold in (2023, 2024):
        ed = edges_for(df[s < fold], "asof_batter_n")
        print(f"\n  [edges from <{fold}]  {np.round(ed, 1)}")
        print(f"  {'시즌':>6}" + "".join(f"{f'bin{i}':>9}" for i in range(5))
              + f"{'전역':>9}   빈별 편차(× 1e3)")
        for ss in range(2019, 2025):
            m = (s == ss) & clean
            gg = qcodes(np.nan_to_num(df.loc[m, "asof_batter_n"]
                                      .to_numpy(dtype=np.float64), nan=-1e18), ed)
            yy = y[m]
            c = np.bincount(gg, minlength=5)
            r = np.bincount(gg, weights=yy, minlength=5) / np.maximum(c, 1)
            gl = yy.mean()
            dv = (r - gl) * 1e3
            print(f"  {ss:>6}" + "".join(f"{r[i]:>9.4f}" for i in range(5))
                  + f"{gl:>9.4f}   " + " ".join(f"{v:+6.1f}" for v in dv))

    # ── A~C. fold 별 ────────────────────────────────────────────────
    for fold in (2023, 2024):
        st = D.state(df, s, y, fold)
        d = st["d"]
        prev = df[(s == fold - 1) & clean]
        y_prev = prev[TARGET_COL].to_numpy(dtype=np.float64)
        ed = edges_for(df[s < fold], "asof_batter_n")

        def code(frame):
            return qcodes(np.nan_to_num(frame["asof_batter_n"]
                                        .to_numpy(dtype=np.float64), nan=-1e18), ed)

        print("\n" + "=" * 92)
        print(f"fold {fold}   n={len(st['yv']):,}   기준 BSS {st['base']:.2f}   "
              f"참조행(={fold-1}) {len(prev):,}")

        gv, gp = code(d), code(prev)
        (a1, a5), w, pg, rg, rp = transfer(st, gv, gp, y_prev, 5)
        (r1, r5), *_ = transfer(st, gv, gp, y_prev, 5, reshift=True)
        print(f"\nA. 재현  batter_exp  α=1.0 {a1:+7.2f}   α=0.5 {a5:+7.2f}"
              f"   (§7.125: f2023 −20.08 / f2024 +3.96)")
        print(f"   레벨 재보정 후(refinement 만)  α=1.0 {r1:+7.2f}   α=0.5 {r5:+7.2f}")

        print("\nC. 기하 — prev 와 valid 에서 같은 빈이 같은 모집단인가")
        cv = np.bincount(gv, minlength=5) / len(gv)
        cp_ = np.bincount(gp, minlength=5) / len(gp)
        print(f"   {'':<14}" + "".join(f"{f'bin{i}':>10}" for i in range(5)))
        print(f"   {'질량 valid':<14}" + "".join(f"{v:>10.3f}" for v in cv))
        print(f"   {'질량 prev':<14}" + "".join(f"{v:>10.3f}" for v in cp_))
        print(f"   {'p̄ valid':<14}" + "".join(f"{v:>10.4f}" for v in pg))
        print(f"   {'r prev':<14}" + "".join(f"{v:>10.4f}" for v in rp))
        print(f"   {'r valid(오라클)':<14}" + "".join(f"{v:>10.4f}" for v in rg))
        print(f"   {'bh=p̄−r_prev':<14}" + "".join(f"{v:>10.4f}" for v in (pg - rp)))
        print(f"   {'진짜 편향':<14}" + "".join(f"{v:>10.4f}" for v in (pg - rg)))

        print("\nB. 위약 — 타자 쪽 바닥 (§7.125 는 투수·달력만 썼다)")
        bid = d["batter_id"].to_numpy().astype(np.int64)
        bidp = prev["batter_id"].to_numpy().astype(np.int64)
        pid = d["pitcher_id"].to_numpy().astype(np.int64)
        pidp = prev["pitcher_id"].to_numpy().astype(np.int64)
        for nm, gv2, gp2, G in (("pid5 (§7.125 재현)", pid % 5, pidp % 5, 5),
                                ("pid2 (§7.125 재현)", pid % 2, pidp % 2, 2),
                                ("bid5 [타자]", bid % 5, bidp % 5, 5),
                                ("bid2 [타자]", bid % 2, bidp % 2, 2)):
            (x1, x5), *_ = transfer(st, gv2, gp2, y_prev, G)
            print(f"   {nm:<22} G={G}  α=1.0 {x1:+8.2f}   α=0.5 {x5:+8.2f}")

        # brand5 — 타자당 동결 난수 5분위: 엔티티 영구성 + 등질량 기하를 동시에 맞춘다
        ub = np.unique(np.concatenate([bid, bidp]))
        draws = []
        for _ in range(B_DRAW):
            sc = rng.random(len(ub))
            lut = dict(zip(ub, np.searchsorted(np.quantile(sc, [.2, .4, .6, .8]), sc,
                                               side="right")))
            gv2 = np.array([lut[b] for b in bid])
            gp2 = np.array([lut[b] for b in bidp])
            (x1,), *_ = transfer(st, gv2, gp2, y_prev, 5, alphas=(1.0,))
            draws.append(x1)
        dr = np.array(draws)
        lo, md, hi = np.percentile(dr, [2.5, 50, 97.5])
        print(f"   {'brand5 [타자 동결난수]':<22} G=5  α=1.0 중앙 {md:+8.2f}   "
              f"95% [{lo:+.2f}, {hi:+.2f}]   B={B_DRAW}")
        print(f"   → batter_exp − brand5중앙 = {a1 - md:+8.2f}   "
              f"위약 초과 분위 {100 * (dr < a1).mean():.1f}%")


        # E. F 비대칭 — prev 는 R 전용인데 valid 는 R+F 다
        print()
        print("E. F 비대칭 진단 (clean = pre-2023 F 제외 → prev 와 valid 의 구성이 다르다)")
        gtv = d["game_type"].to_numpy()
        gtp = prev["game_type"].to_numpy()
        print(f"   {'':<16}" + "".join(f"{f'bin{i}':>10}" for i in range(5)))
        print(f"   {'F몫 valid':<16}"
              + "".join(f"{(gtv[gv == i] == 'F').mean():>10.3f}" for i in range(5)))
        print(f"   {'F몫 prev':<16}"
              + "".join(f"{(gtp[gp == i] == 'F').mean():>10.3f}" for i in range(5)))
        mv, mp = gtv == "R", gtp == "R"
        cvR = np.maximum(np.bincount(gv[mv], minlength=5), 1)
        cpR = np.bincount(gp[mp], minlength=5)
        rvR = np.bincount(gv[mv], weights=st["yv"][mv], minlength=5) / cvR
        rpR = np.where(cpR > 0, np.bincount(gp[mp], weights=y_prev[mp], minlength=5)
                       / np.maximum(cpR, 1), 0.0)
        print(f"   {'r valid R행':<16}" + "".join(f"{v:>10.4f}" for v in rvR))
        print(f"   {'r prev R행':<16}" + "".join(f"{v:>10.4f}" for v in rpR))

        # 수리 팔 — prev·valid 를 둘 다 R 행 기준으로 맞춘다 (평가는 전체 행)
        cntv = np.bincount(gv, minlength=5)
        wv = cntv / len(gv)
        pgv = np.bincount(gv, weights=st["p0"], minlength=5) / np.maximum(cntv, 1)
        bhR = pgv - rpR
        devR = bhR - np.sum(wv * bhR)
        for a in ALPHAS:
            pp = np.clip(st["p0"] - a * devR[gv], 1e-6, 1 - 1e-6)
            print(f"   [수리] prev 도 R전용   α={a}   Δ {bss(st['yv'], pp) - st['base']:+8.2f}")


if __name__ == "__main__":
    main()
