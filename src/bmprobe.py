"""타자 매핑 전제 검정 — 코드를 쓰기 전에 PA 경계가 복원되는지부터 잰다.

투수 매핑(`trackman_map.py`)은 "하프이닝별 **투구 수**"를 지문으로 썼다.
타자에는 그 신호가 약하다 — 타자당 3~5구라 카운트가 서로 충돌한다.
대신 타자에는 더 강한 축이 있다: **타석(PA) 순서**.

    main      PA 첫 구 = (balls_before == 0) & (strikes_before == 0)
    trackman  PA 첫 구 = (pitch_of_pa == 1)

하프이닝 안에서 PA 시퀀스를 위치로 맞추면 타순이 그대로 정렬 키가 된다.
단, 이게 성립하려면 아래 네 가지가 참이어야 한다. 전부 여기서 잰다.

    A. main 행이 하프이닝 안에서 투구 순으로 정렬돼 있는가
       → (0,0)로 자른 PA 안에서 볼/스트라이크가 단조 증가해야 한다
    B. PA 길이 분포가 야구 상식과 맞는가 (1~10구, 최빈 1~4)
    C. 매칭된 경기의 하프이닝에서 양쪽 PA 개수가 일치하는가  ← 핵심
    D. batter_hand 코드 대응 (main 2 == "Right"인가)

실행:  .venv/Scripts/python.exe -u src/bmprobe.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

from config import DATA_DIR
from trackman_map import load_main, load_tm, profiles, match_games

MAIN_EXTRA = ["balls_before", "strikes_before", "batter_id", "batter_hand"]
TM_EXTRA = ["pitch_of_pa", "batter_trackman_id", "batter_hand", "pitch_no"]


def main():
    t0 = time.time()

    # trackman_map.load_main/load_tm은 필요한 컬럼만 읽는다.
    # 여기서는 PA 판정용 컬럼이 더 필요하므로 따로 읽어 붙인다.
    print("Loading ...")
    m = load_main(MAIN_EXTRA)
    t = load_tm(TM_EXTRA)
    print(f"  main {m.shape}  trackman {t.shape}   ({time.time()-t0:.0f}s)")
    print(f"  main batter_id 고유 {m.batter_id.nunique():,}명 / "
          f"trackman batter_trackman_id 고유 {t.batter_trackman_id.nunique():,}명")
    inter = set(m.batter_id.unique()) & set(t.batter_trackman_id.dropna().astype(int).unique())
    print(f"  ID 교집합 {len(inter)}개   ← 투수와 마찬가지로 0이어야 정상")

    # ── A. main 행 순서가 투구 순인가 ────────────────────────────
    print("\n=== A. main 행 순서 검정 (PA 안에서 카운트가 단조 증가하는가) ===")
    is_pa_start = (m.balls_before == 0) & (m.strikes_before == 0)
    m["pa"] = is_pa_start.cumsum().astype(np.int32)
    # PA 안에서 balls+strikes는 매 구 정확히 1씩 늘어야 한다 (파울은 strikes 유지 →
    # 그 경우 합이 그대로다). 따라서 증분은 0 또는 1이고 음수는 나오면 안 된다.
    tot = (m.balls_before + m.strikes_before).to_numpy()
    same_pa = (m.pa.to_numpy()[1:] == m.pa.to_numpy()[:-1])
    d = tot[1:] - tot[:-1]
    within = d[same_pa]
    print(f"  PA 내부 인접쌍 {len(within):,}개")
    print(f"    증분 <0 (순서 위반): {(within < 0).sum():,}  ({(within<0).mean():.4%})")
    print(f"    증분 == 0 (파울):    {(within == 0).sum():,}  ({(within==0).mean():.2%})")
    print(f"    증분 == 1 (정상):    {(within == 1).sum():,}  ({(within==1).mean():.2%})")
    print(f"    증분 >1 (건너뜀):    {(within > 1).sum():,}  ({(within>1).mean():.4%})")

    # PA가 하프이닝 경계를 넘지 않는가 (넘으면 행 순서나 PA 판정이 틀린 것)
    span = m.groupby("pa", observed=True).agg(nhalf=("half", "nunique"),
                                              ngame=("game", "nunique"))
    print(f"  PA {len(span):,}개 중 하프이닝을 걸친 것 {int((span.nhalf > 1).sum()):,}, "
          f"경기를 걸친 것 {int((span.ngame > 1).sum()):,}")

    # ── B. PA 길이 분포 ──────────────────────────────────────────
    print("\n=== B. PA 길이 분포 ===")
    mlen = m.groupby("pa", observed=True).size()
    tlen = t.groupby((t.pitch_of_pa == 1).cumsum(), observed=True).size()
    print("  len :  " + "".join(f"{k:>7d}" for k in range(1, 11)))
    print("  main:  " + "".join(f"{mlen.eq(k).mean():>7.1%}" for k in range(1, 11)))
    print("  tm  :  " + "".join(f"{tlen.eq(k).mean():>7.1%}" for k in range(1, 11)))
    print(f"  평균 길이  main {mlen.mean():.3f}  trackman {tlen.mean():.3f}")

    # 한 PA는 타자 1명이어야 한다
    nb = m.groupby("pa", observed=True).batter_id.nunique()
    print(f"  PA 안에 타자가 2명 이상인 경우: {int((nb > 1).sum()):,} / {len(nb):,}")

    # ── C. 매칭된 하프이닝의 PA 개수 일치 ────────────────────────
    print("\n=== C. 경기 매칭 후 하프이닝별 PA 개수 일치율  ← 핵심 ===")
    mmeta, mprof = profiles(m)
    tmeta, tprof = profiles(t)
    pairs, costs = match_games(mmeta, mprof, tmeta, tprof)
    exact = {k: v[0] for k, v in pairs.items() if v[1] == 0}
    print(f"  경기 매칭 {len(pairs):,}쌍, 그중 L1=0 정밀집합 {len(exact):,}쌍")

    mpa = (m.assign(s=is_pa_start).groupby(["game", "half"], observed=True).s.sum()
             .rename("n_pa"))
    tpa = (t.assign(s=(t.pitch_of_pa == 1)).groupby(["game", "half"], observed=True).s.sum()
             .rename("n_pa"))
    mm = mpa.reset_index()
    mm = mm[mm.game.isin(exact)]
    mm["tgame"] = mm.game.map(exact)
    j = mm.merge(tpa.reset_index().rename(columns={"game": "tgame", "n_pa": "n_pa_t"}),
                 on=["tgame", "half"], how="left")
    j["n_pa_t"] = j.n_pa_t.fillna(-1)
    ok = j.n_pa == j.n_pa_t
    print(f"  대조 가능한 하프이닝 {len(j):,}개")
    print(f"    PA 개수 완전 일치: {int(ok.sum()):,}  ({ok.mean():.2%})")
    diff = (j.n_pa - j.n_pa_t)[j.n_pa_t >= 0]
    print(f"    차이 분포: " + "  ".join(
        f"{k:+d}:{(diff == k).mean():.2%}" for k in (-2, -1, 0, 1, 2)))
    print(f"    trackman 쪽 하프이닝 자체가 없음: {int((j.n_pa_t < 0).sum()):,}")
    cov = j.loc[ok].n_pa.sum() / j.n_pa.sum()
    print(f"  → PA 개수가 일치하는 하프이닝이 담는 main PA 비중: {cov:.2%}")

    # ── D. batter_hand 코드 대응 ─────────────────────────────────
    print("\n=== D. batter_hand 코드 ===")
    print("  main  :", m.batter_hand.value_counts(normalize=True).round(4).to_dict())
    print("  tm    :", t.batter_hand.value_counts(normalize=True).round(4).to_dict())
    print("  ※ 투수는 main 2 == 'Right'였다. 타자는 좌타 비율이 높아 분포로 바로 읽힌다.")
    print("  ※ 스위치히터가 있으면 타자별 최빈값이 상대 투수에 따라 갈린다 →")
    print("     손잡이 제약을 걸 때 주의해야 한다. 타자별 hand 고유값 개수:")
    hu = m.groupby("batter_id").batter_hand.nunique()
    print(f"     main 타자 {len(hu):,}명 중 hand가 2개 이상인 타자 {int((hu > 1).sum()):,}명 "
          f"({(hu > 1).mean():.1%})")

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
