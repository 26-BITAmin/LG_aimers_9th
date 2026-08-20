"""타자 매핑 채택 게이트(`MIN_VOTES` · `MIN_SHARE`) 스윕.

§7.9가 남긴 원칙: **오매핑은 결측보다 나쁘다.** 매핑이 없으면 피처가 결측이 되고
모델이 그렇게 다루지만, 잘못 매핑되면 **남의 기록을 자기 것으로 갖게 된다.**
따라서 게이트는 "커버리지 최대"가 아니라 그 비대칭을 반영해 고른다.

두 숫자를 함께 본다:
    커버리지  — 매핑이 담는 main 투구 비중 (놓치는 양)
    일치율    — 경기 분할 홀드아웃 (틀리는 양)

실행:  .venv/Scripts/python.exe -u src/bmgate.py
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

import batter_map as B
from trackman_map import profiles, match_games

VOTES = [3, 10, 20, 50, 100]
SHARES = [0.5, 0.7, 0.9, 0.95]


def main():
    t0 = time.time()
    m, t, mpa, tpa = B.load_pa_tables()
    mmeta, mprof = profiles(m)
    tmeta, tprof = profiles(t)
    pairs, _ = match_games(mmeta, mprof, tmeta, tprof)
    exact = {k: v[0] for k, v in pairs.items() if v[1] == 0}

    m_ids = sorted(m.batter_id.unique())
    t_ids = sorted(t.batter_trackman_id.dropna().unique().astype(np.int64))
    ms, mh = B.hand_shares(m.batter_hand.to_numpy(), m.batter_id.to_numpy(), 2)
    tt = t.dropna(subset=["batter_trackman_id"])
    ts, th = B.hand_shares(tt.batter_hand.to_numpy(),
                           tt.batter_trackman_id.astype(np.int64).to_numpy(), "Right")
    mh = mh[[list(ms.index).index(i) for i in m_ids]]
    th = th[[list(ts.index).index(i) for i in t_ids]]
    npitch = m.groupby("batter_id").size()

    # 투표는 게이트와 무관하므로 세 번만 만들고 재사용한다.
    gl = sorted(exact)
    v_all = B.batter_votes(mpa, tpa, exact)[0]
    v_a = B.batter_votes(mpa, tpa, {k: exact[k] for k in gl[0::2]})[0]
    v_b = B.batter_votes(mpa, tpa, {k: exact[k] for k in gl[1::2]})[0]
    print(f"투표 준비 완료 ({time.time()-t0:.0f}s)\n")

    # ⚠️ 홀드아웃은 경기의 절반씩만 쓰므로 득표도 대략 1/2이다. 양쪽에 같은
    #    min_votes를 걸면 **총 2×min_votes 이상인 타자만** 검증 대상이 되고,
    #    배포 매핑에 실제로 들어가는 min_votes ~ 2×min_votes 구간이 표본에서 빠진다.
    #    §7.39에서 시즌 분할이 장기 재직자만 검증했던 것과 같은 모양의 구멍이다.
    #
    #    → 검증은 **배포될 집합 그대로**에 대해 한다. 홀드아웃 매핑은 게이트를
    #      최소로 두고 만들고(min_votes=3), 비교 대상만 전체 게이트가 채택한
    #      타자로 제한한다. 그러면 "이 게이트로 내보낼 매핑이 얼마나 틀리는가"를
    #      직접 답한다.
    ma0 = B.solve_mapping(v_a, m_ids, t_ids, mh, th, 3, 0.5)[0]
    mb0 = B.solve_mapping(v_b, m_ids, t_ids, mh, th, 3, 0.5)[0]
    both0 = set(ma0) & set(mb0)

    print("  [naive]  홀드아웃에도 같은 게이트 → 검증 대상이 장기 타자로 쏠린다")
    print("  [정직]   홀드아웃 게이트는 최소, 비교 대상만 배포 집합으로 제한\n")
    print(f"{'min_v':>5} {'min_sh':>6} | {'매핑':>4} {'커버리지':>8} | "
          f"{'naive n':>7} {'일치':>7} | {'정직 n':>6} {'일치':>7} {'불일치':>5} {'놓침':>5}")
    print("-" * 84)

    for mv in VOTES:
        for sh in SHARES:
            full = B.solve_mapping(v_all, m_ids, t_ids, mh, th, mv, sh)[0]
            cov = npitch.reindex(list(full)).sum() / npitch.sum()

            ma = B.solve_mapping(v_a, m_ids, t_ids, mh, th, mv, sh)[0]
            mb = B.solve_mapping(v_b, m_ids, t_ids, mh, th, mv, sh)[0]
            bn = set(ma) & set(mb)
            nbad = sum(ma[k] != mb[k] for k in bn)

            # 정직한 검증: 배포 집합 ∩ 양쪽 홀드아웃에서 모두 매핑된 타자
            sel = both0 & set(full)
            hbad = sum(ma0[k] != mb0[k] for k in sel)
            miss = len(full) - len(sel)     # 배포되지만 검증 못 한 타자 수
            print(f"{mv:>5} {sh:>6.2f} | {len(full):>4} {cov:>7.2%} | "
                  f"{len(bn):>7} {(len(bn)-nbad)/max(len(bn),1):>6.2%} | "
                  f"{len(sel):>6} {(len(sel)-hbad)/max(len(sel),1):>6.2%} {hbad:>5} {miss:>5}")
        print()

    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
