"""Trackman 매핑 신뢰도 — §7.9가 적어둔 재시도 절차의 1단계

§7.9의 진단 (2025 LB −13.6의 원인)
    "매핑 검증은 시즌 홀드아웃으로 했는데, 그 대상은 **양쪽 기간에 모두 등장한
     장기 재직 투수 322명**이었다. 최근 데뷔 투수는 투표에 쓸 경기가 적어 오매핑
     위험이 크고, **오매핑은 결측보다 나쁘다 — 남의 릴리스포인트를 갖게 된다.**
     그리고 2025 활동 투수는 최근 데뷔 쪽에 치우친다.
     → 다시 시도한다면 **투수별 경기 수에 따른 매핑 신뢰도를 먼저 측정하고,
       신뢰도가 낮은 투수는 아예 결측 처리하는 것이 순서다.**"

🪤 시즌 분할 검증의 구조적 결함
    2019~2022 매핑과 2023~2024 매핑을 대조하면 **양쪽에 다 나온 투수만** 검증 대상이
    된다. 그건 정의상 장기 재직자다. **검증하고 싶은 대상(단기 투수)이 표본에서
    자동으로 빠지는 설계**였고, 그래서 322/322 = 100%가 나왔다.
    100%는 매핑이 완벽하다는 뜻이 아니라 **쉬운 문제만 풀었다**는 뜻이다.

이 파일의 수정
    **경기 단위로 무작위 분할한다.** 그러면 경기 3개짜리 투수도 양쪽에 나뉘어
    검증 대상이 된다. 시즌 경계가 아니라 경기 경계로 자르는 것뿐인데,
    검증 가능 모집단이 322명에서 전체로 바뀐다.
    (매핑은 타깃을 전혀 쓰지 않으므로 이 분할에 리키지 문제가 없다.)

재는 것
    ① 경기 단위 2분할 일치율 — **경력 길이 구간별로** 본다
    ② 투표 신뢰도(득표 점유율 · 총 득표)가 일치율을 예측하는가
    ③ 신뢰도 임계를 걸면 커버리지를 얼마나 잃고 정확도를 얼마나 얻는가

⚠️ 여기까지는 **매핑 품질 측정**이다. Trackman 피처를 되살리는 결정이 아니다.
   ②가 예측력이 없으면 게이트를 걸 방법이 없고, 그러면 §7.9의 재시도 절차가
   1단계에서 막힌다 — 그것도 결론이다.

실행:  .venv/Scripts/python.exe -u src/tmconf.py 2>&1 | tee experiments/tmconf.log
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

from trackman_map import (GAME_COST_MAX, load_main, load_tm, match_games,
                          pitcher_votes, profiles, solve_mapping)

N_SPLIT = 4          # 무작위 2분할을 몇 번 반복할 것인가 (시드마다 다른 분할)
SEED = 42
CAREER_BINS = [0, 200, 1000, 3000, 10000, 10**9]
CAREER_LAB = ["~200투구", "200~1k", "1k~3k", "3k~10k", "10k+"]


def confidence(m, t, vote):
    """pitcher_id별 매핑 신뢰도. 타깃을 쓰지 않는 순수 구조 지표다.

      share = 최다 득표 / 그 투수의 총 득표      (경쟁자가 있으면 낮다)
      votes = 최다 득표 수                        (경기가 적으면 낮다)
    """
    tot, top, arg = {}, {}, {}
    for (a, b), n in vote.items():
        tot[a] = tot.get(a, 0.0) + n
        if n > top.get(a, -1):
            top[a], arg[a] = n, b
    rows = [(a, arg[a], top[a], tot[a], top[a] / tot[a]) for a in tot if tot[a] > 0]
    return pd.DataFrame(rows, columns=["pitcher_id", "tm_id", "votes",
                                       "votes_tot", "share"]).set_index("pitcher_id")


def main():
    t0 = time.time()
    m, t = load_main(), load_tm()
    mmeta, mprof = profiles(m)
    tmeta, tprof = profiles(t)
    pairs, _ = match_games(mmeta, mprof, tmeta, tprof)
    exact = {k: v[0] for k, v in pairs.items() if v[1] == 0}
    print(f"main {m.shape}  trackman {t.shape}   정밀 경기 {len(exact):,}쌍"
          f"   ({time.time()-t0:.0f}s)")

    npitch = m.groupby("pitcher_id").size().rename("main_n")
    ngame = m.groupby("pitcher_id").game.nunique().rename("main_g")
    career = pd.concat([npitch, ngame], axis=1)
    career["bin"] = pd.cut(career.main_n, CAREER_BINS, labels=CAREER_LAB)

    full_vote = pitcher_votes(m, t, exact)
    conf = confidence(m, t, full_vote)
    full_map, _ = solve_mapping(m, t, full_vote)
    print(f"전체 매핑 {len(full_map)}명 / main 투수 {len(npitch)}명"
          f"   투구 커버리지 {npitch.reindex(list(full_map)).sum()/npitch.sum():.1%}\n")

    # ── ① 시즌 분할(옛 방식) vs 경기 분할(새 방식) ─────────────────
    print("=" * 96)
    print("① 검증 설계 비교 — 무엇이 검증 대상이 되는가")
    print("=" * 96)
    early = {k: v for k, v in exact.items() if mmeta.loc[k, "season"] <= 2022}
    late = {k: v for k, v in exact.items() if mmeta.loc[k, "season"] >= 2023}
    me = solve_mapping(m, t, pitcher_votes(m, t, early))[0]
    ml = solve_mapping(m, t, pitcher_votes(m, t, late))[0]
    both_s = sorted(set(me) & set(ml))
    ag_s = sum(me[k] == ml[k] for k in both_s)
    print(f"  [옛] 시즌 분할  검증 가능 {len(both_s)}명  일치 {ag_s}/{len(both_s)}"
          f" = {ag_s/max(len(both_s),1):.2%}")
    print(f"       그 {len(both_s)}명의 경력 분포 — **장기 재직자만 남는다**:")
    print("       " + career.loc[both_s, "bin"].value_counts()
          .reindex(CAREER_LAB).to_string().replace("\n", "\n       "))

    keys = np.array(sorted(exact))
    rng = np.random.default_rng(SEED)
    agree_cnt, seen_cnt = {}, {}
    tot_ag = tot_n = 0
    for s in range(N_SPLIT):
        mask = rng.random(len(keys)) < 0.5
        A = {k: exact[k] for k in keys[mask]}
        B = {k: exact[k] for k in keys[~mask]}
        ma = solve_mapping(m, t, pitcher_votes(m, t, A))[0]
        mb = solve_mapping(m, t, pitcher_votes(m, t, B))[0]
        both = set(ma) & set(mb)
        for k in both:
            seen_cnt[k] = seen_cnt.get(k, 0) + 1
            agree_cnt[k] = agree_cnt.get(k, 0) + int(ma[k] == mb[k])
        tot_ag += sum(ma[k] == mb[k] for k in both)
        tot_n += len(both)
    print(f"\n  [새] 경기 분할 ×{N_SPLIT}  검증 가능 {len(seen_cnt)}명"
          f"  일치 {tot_ag}/{tot_n} = {tot_ag/max(tot_n,1):.2%}")

    # ── ② 경력 길이별 일치율 ───────────────────────────────────────
    print("\n" + "=" * 96)
    print("② 경력 길이별 일치율 — 오매핑이 정말 단기 투수에 몰리는가")
    print("=" * 96)
    df = pd.DataFrame({"ag": pd.Series(agree_cnt), "n": pd.Series(seen_cnt)})
    df = df.join(career)
    g = df.groupby("bin", observed=False).agg(투수=("n", "size"), 검증=("n", "sum"),
                                              일치=("ag", "sum"))
    g["일치율"] = g.일치 / g.검증.clip(lower=1)
    g["투구비중"] = [career.loc[df[df.bin == b].index, "main_n"].sum() / npitch.sum()
                  if (df.bin == b).any() else 0.0 for b in g.index]
    print(g.to_string())

    # ── ③ 신뢰도가 일치율을 예측하는가 ─────────────────────────────
    print("\n" + "=" * 96)
    print("③ 투표 신뢰도가 오매핑을 예측하는가 — 게이트를 걸 수 있는지의 조건")
    print("=" * 96)
    j = df.join(conf, how="inner")
    j["ok"] = j.ag / j.n
    for col, cuts in (("share", [0, .5, .8, .95, .999, 1.001]),
                      ("votes", [0, 10, 50, 200, 1000, 10**9])):
        j["b"] = pd.cut(j[col], cuts)
        gg = j.groupby("b", observed=False).agg(투수=("ok", "size"), 검증=("n", "sum"),
                                                일치=("ag", "sum"))
        gg["일치율"] = gg.일치 / gg.검증.clip(lower=1)
        print(f"\n  [{col}]")
        print("  " + gg.to_string().replace("\n", "\n  "))
    print(f"\n  corr(share, 일치율) {j.share.corr(j.ok):+.3f}"
          f"   corr(log votes, 일치율) {np.log1p(j.votes).corr(j.ok):+.3f}")

    # ── ④ 게이트를 걸면 얼마를 잃고 얼마를 얻나 ───────────────────
    print("\n" + "=" * 96)
    print("④ 게이트 곡선 — 신뢰도 임계별 (커버리지, 검증 일치율)")
    print("=" * 96)
    print(f"  {'임계(votes)':>12}{'남는 투수':>10}{'투구 커버리지':>14}{'검증 일치율':>12}")
    for thr in (0, 3, 10, 30, 100, 300, 1000):
        keep = j[j.votes >= thr]
        cov = npitch.reindex(keep.index).sum() / npitch.sum()
        acc = keep.ag.sum() / max(keep.n.sum(), 1)
        print(f"  {thr:>12}{len(keep):>10}{cov:>14.1%}{acc:>12.2%}")

    print("\n  ⚠️ 이 표는 **매핑 품질**이다. Trackman 피처가 2025에서 이득인지와는 다른 문제다.")
    print("     여기서 게이트가 성립해야 §7.9의 재시도가 2단계로 넘어간다.")


if __name__ == "__main__":
    main()
