"""Trackman ↔ main **타자** ID 매핑.

투수 매핑(`trackman_map.py`)의 1~3단계(경기 복원 → 지문 매칭 → 팀 일관성)를 그대로
재사용하고, **4단계(투표)만 타자용으로 바꾼다.**

왜 바꿔야 하나
    투수 투표는 "하프이닝별 **투구 수**"를 대조한다. 투수는 한 하프이닝에 15~30구를
    던지므로 그 숫자가 거의 고유하다. 타자는 3~5구씩이라 카운트가 서로 충돌한다.

대신 타자에는 더 강한 축이 있다 — **타석(PA) 순서**. 타순은 정렬 그 자체다.

    main      PA 첫 구 = (balls_before == 0) & (strikes_before == 0)
    trackman  PA 첫 구 = (pitch_of_pa == 1)

전제 검정 완료 (`src/bmprobe.py`, experiments/bmprobe.log)
    A. main 행이 하프이닝 안에서 투구 순  — PA 내부 카운트 순서 위반 18/1,094,096
    B. PA 평균 길이 main 3.872 vs trackman 3.846
    C. **매칭된 하프이닝의 PA 개수 일치율 99.98%** (60,028/60,038)   ← 핵심
    D. batter_hand 코드는 투수와 동일 (main 2 == "Right")

검증은 **경기 분할**로 한다. 시즌 분할이 아니다 — §7.39에서 시즌 분할이 양쪽 시즌에
모두 나오는 장기 재직자만 검증 대상으로 삼는다는 것이 드러났다. 검증하고 싶은 대상
(단기·신인)이 표본에서 자동으로 빠지는 설계였다. 경기로 자르면 경기 3개짜리 타자도
양쪽에 나뉜다. (매핑은 타깃을 안 쓰므로 이 분할에 리키지가 없다.)

리키지: 이 매핑은 `control_success`를 전혀 쓰지 않는다. 순수하게 구조적 대응이다.

실행:  .venv/Scripts/python.exe -u src/batter_map.py
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
from scipy.optimize import linear_sum_assignment

from config import DATA_DIR, MODEL_DIR
from trackman_map import load_main, load_tm, profiles, match_games

MAIN_EXTRA = ["balls_before", "strikes_before", "batter_id", "batter_hand"]
TM_EXTRA = ["pitch_of_pa", "batter_trackman_id", "batter_hand", "pitch_no"]

# ── 채택 게이트 (`src/bmgate.py` 스윕, experiments/bmgate.log) ──────────
#
# §7.9의 원칙: **오매핑은 결측보다 나쁘다.** 매핑이 없으면 피처가 결측이 되고 모델이
# 그렇게 다루지만, 잘못 매핑되면 남의 기록을 자기 것으로 갖게 된다. 게이트는
# "커버리지 최대"가 아니라 그 비대칭을 반영해 고른다.
#
#   게이트            매핑   커버리지   불일치   검증 못 한 타자
#   votes≥3  sh≥0.5    752    99.92%       4          74
#   votes≥3  sh≥0.7    747    99.67%       0          74
#   **votes≥10**       660    99.66%       0          14   ← 채택
#   votes≥20           584    99.18%       0           0
#
# `votes≥10`이 `sh≥0.7`을 **지배한다** — 커버리지가 사실상 같은데(99.66 vs 99.67)
# 미검증 타자가 74 → 14명으로 준다. `min_share`는 득표 분포가 이봉이라 거의 놀고 있다.
#
# votes≥20이면 미검증이 0이 되지만 커버리지를 0.48%p 더 낸다. 그 구간(10~19표)은
# **불일치가 관측되지 않았고 관측할 표본도 없는** 회색지대다. 판단이 갈릴 수 있는
# 지점이라 아래 `main()`이 두 값을 모두 출력한다.
MIN_VOTES = 10
MIN_SHARE = 0.5


# ─────────────────────────────────────────────────────────────────
#  손잡이 제약  ← 여기가 이번 매핑의 유일한 설계 결정이다
# ─────────────────────────────────────────────────────────────────
def hand_compatible(m_hand_share, t_hand_share):
    """(main 타자 N명) × (trackman 타자 M명) 손잡이 호환 불린 행렬을 만든다.

    투수 매핑은 최빈값으로 하드 제약을 걸었다 (`main 2 == "Right"`).
    타자에는 **스위치히터**가 있어서 그게 그대로 통하지 않는다. 실측:

        main     최빈 점유율 <1.00  32명 (투구 4.50%)   <0.90  15명 (1.65%)
        trackman 최빈 점유율 <1.00  95명 (투구 13.30%)  <0.90  21명 (1.62%)

    → 최빈 점유율 0.90 아래 14~15명이 진짜 스위치히터고, 그 위 노이즈성 전환은
      데이터 오류 수준이다. 스위치히터에는 `batter_id 23413`(7,433투구) 같은
      주전이 포함된다. 하드 제약은 이들을 통째로 깨뜨린다.

    ❌ **제약을 걸지 않는다.** "쓸까 말까"를 재는 대신 **실제 실패 사례에 걸어봤고,
       판별력이 0이었다.** 경기 분할 홀드아웃의 불일치 5건 전부:

           batter_id   투구   main 우타율   후보A   후보B    손잡이로 구분되나
              23798     24        1.00      1.00    1.00    아니오
              23799     31        1.00      1.00    1.00    아니오
              23802     26        1.00      1.00    1.00    아니오
              23805     43        0.05      0.00    0.00    아니오
              23810     38        0.05      0.00    0.00    아니오

       실패는 24~43투구짜리 단역 타자에서 일어나는데(ID가 연번인 걸 보면 같은 시즌
       콜업 무리다), **거기서는 후보들이 애초에 같은 손잡이다.** 제약을 걸면 이 5건은
       그대로 두고 스위치히터 15명(투구 1.65%)만 잃는다. 순손실이다.

       > 제약의 가치는 평균 판별력이 아니라 **실패 지점에서의 판별력**이다.
         투수에서 통했다는 것이 타자에서 통한다는 근거가 아니다.

    Parameters
    ----------
    m_hand_share : (N, 2) float ndarray
        main 타자별 [우타 비율, 좌타 비율]. `batter_hand == 2`가 우타다.
    t_hand_share : (M, 2) float ndarray
        trackman 타자별 ["Right" 비율, "Left" 비율].

    Returns
    -------
    (N, M) bool ndarray — True면 이 쌍의 투표를 살린다.
    """
    return np.ones((len(m_hand_share), len(t_hand_share)), dtype=bool)


def load_pa_tables():
    """main / trackman을 각각 **PA 단위** 테이블로 압축한다.

    반환 컬럼: game, half, k (하프이닝 안 PA 순번), batter (ID)
    행 단위 루프 없이 전부 벡터화한다.
    """
    m = load_main(MAIN_EXTRA)
    t = load_tm(TM_EXTRA)

    # main: 행이 이미 투구 순이므로 PA 첫 구만 뽑으면 그 순서가 타순이다.
    ms = m[(m.balls_before == 0) & (m.strikes_before == 0)].copy()
    ms["k"] = ms.groupby(["game", "half"], observed=True).cumcount().astype(np.int16)
    mpa = ms[["game", "half", "k", "batter_id"]].rename(columns={"batter_id": "batter"})

    # trackman: 파일 순서를 믿지 않고 pitch_no로 명시 정렬한다.
    ts = t[t.pitch_of_pa == 1].dropna(subset=["batter_trackman_id"]).copy()
    ts = ts.sort_values(["game", "half", "pitch_no"], kind="mergesort")
    ts["k"] = ts.groupby(["game", "half"], observed=True).cumcount().astype(np.int16)
    ts["batter_trackman_id"] = ts.batter_trackman_id.astype(np.int64)
    tpa = ts[["game", "half", "k", "batter_trackman_id"]].rename(
        columns={"batter_trackman_id": "batter"})

    return m, t, mpa, tpa


def batter_votes(mpa, tpa, game_pairs):
    """매칭된 경기의 하프이닝 안에서 PA를 **순번으로** 정렬해 투표한다.

    PA 개수가 다른 하프이닝은 통째로 버린다 (전체의 0.02%). 개수가 어긋난 채로
    순번을 맞추면 그 지점 이후가 전부 한 칸씩 밀려 조직적 오매핑이 된다 —
    소수의 결측보다 훨씬 나쁘다.
    """
    a = mpa[mpa.game.isin(game_pairs)].copy()
    a["tgame"] = a.game.map(game_pairs)

    na = a.groupby(["tgame", "half"], observed=True).size().rename("na")
    nb = tpa.groupby(["game", "half"], observed=True).size().rename("nb")
    cnt = pd.concat([na, nb.rename_axis(["tgame", "half"])], axis=1)
    keep = cnt[(cnt.na == cnt.nb)].index          # PA 개수가 일치하는 하프이닝만

    a = a.set_index(["tgame", "half"]).loc[a.set_index(["tgame", "half"]).index.isin(keep)]
    a = a.reset_index()
    b = tpa.rename(columns={"game": "tgame", "batter": "tbatter"})

    j = a.merge(b, on=["tgame", "half", "k"], how="inner")
    vote = j.groupby(["batter", "tbatter"], observed=True).size()
    return vote, len(keep)


def solve_mapping(vote, m_ids, t_ids, m_hand, t_hand,
                  min_votes=MIN_VOTES, min_share=MIN_SHARE):
    """전역 헝가리안 배정 + 손잡이 호환 + 득표 임계."""
    mi = {v: i for i, v in enumerate(m_ids)}
    ti = {v: i for i, v in enumerate(t_ids)}
    V = np.zeros((len(m_ids), len(t_ids)), dtype=np.float64)
    idx = vote.index.to_frame(index=False)
    for a, b, n in zip(idx.batter, idx.tbatter, vote.to_numpy()):
        if a in mi and b in ti:
            V[mi[a], ti[b]] = n

    ok = hand_compatible(m_hand, t_hand)
    V = np.where(ok, V, 0.0)

    r, c = linear_sum_assignment(-V)
    row_tot = V.sum(axis=1)
    mapping, votes = {}, {}
    for a, b in zip(r, c):
        n = V[a, b]
        if n >= min_votes and row_tot[a] > 0 and n / row_tot[a] >= min_share:
            mapping[m_ids[a]] = t_ids[b]
            votes[m_ids[a]] = (n, row_tot[a])
    return mapping, votes


def hand_shares(labels, groups, right_value):
    """선수별 [우타 비율, 좌타 비율] (N, 2) 행렬."""
    df = pd.DataFrame({"g": groups, "is_r": (labels == right_value).astype(np.float64)})
    s = df.groupby("g").is_r.mean()
    return s, np.column_stack([s.to_numpy(), 1.0 - s.to_numpy()])


def main():
    t0 = time.time()
    print("Loading ...")
    m, t, mpa, tpa = load_pa_tables()
    print(f"  main {m.shape}  trackman {t.shape}   ({time.time()-t0:.0f}s)")
    print(f"  PA 단위: main {len(mpa):,}  trackman {len(tpa):,}")
    print(f"  타자 고유: main {m.batter_id.nunique():,}명  trackman {t.batter_trackman_id.nunique():,}명")

    print("\n=== 1~2. 경기 복원 + 지문 매칭 (trackman_map과 동일) ===")
    mmeta, mprof = profiles(m)
    tmeta, tprof = profiles(t)
    pairs, costs = match_games(mmeta, mprof, tmeta, tprof)
    exact = {k: v[0] for k, v in pairs.items() if v[1] == 0}
    print(f"  경기 매칭 {len(pairs):,}쌍   정밀집합(L1=0) {len(exact):,}쌍")

    print("\n=== 3. 타자 투표 (하프이닝 안 PA 순번 정렬) ===")
    vote, n_half = batter_votes(mpa, tpa, exact)
    print(f"  사용한 하프이닝 {n_half:,}개")
    print(f"  투표 총량 {int(vote.sum()):,}  (batter_id, trackman_id) 조합 {len(vote):,}개")

    m_ids = sorted(m.batter_id.unique())
    t_ids = sorted(t.batter_trackman_id.dropna().unique().astype(np.int64))
    ms, m_hand = hand_shares(m.batter_hand.to_numpy(), m.batter_id.to_numpy(), 2)
    tt = t.dropna(subset=["batter_trackman_id"])
    ts, t_hand = hand_shares(tt.batter_hand.to_numpy(),
                             tt.batter_trackman_id.astype(np.int64).to_numpy(), "Right")
    m_hand = m_hand[[list(ms.index).index(i) for i in m_ids]]
    t_hand = t_hand[[list(ts.index).index(i) for i in t_ids]]

    print("\n=== 4. 매핑 ===")
    mapping, votes = solve_mapping(vote, m_ids, t_ids, m_hand, t_hand)
    npa = mpa.groupby("batter").size()
    cov = npa.reindex(list(mapping)).sum() / npa.sum()
    npitch = m.groupby("batter_id").size()
    pcov = npitch.reindex(list(mapping)).sum() / npitch.sum()
    print(f"  매핑된 타자 {len(mapping):,}명 / main 타자 {len(m_ids):,}명")
    print(f"  main PA 커버리지 {cov:.2%}   투구 커버리지 {pcov:.2%}")
    sh = np.array([votes[k][0] / votes[k][1] for k in mapping])
    print(f"  득표 점유율 분위: {np.round(np.quantile(sh, [0.01,0.05,0.25,0.5]), 4)}")
    print(f"  점유율 <0.9인 매핑 {int((sh < 0.9).sum())}건, <0.7 {int((sh < 0.7).sum())}건")

    print("\n=== 5. 검증 — **경기 분할** 홀드아웃 (시즌 분할 아님, §7.39) ===")
    gl = sorted(exact)
    ea = {k: exact[k] for k in gl[0::2]}
    eb = {k: exact[k] for k in gl[1::2]}
    # ⚠️ 홀드아웃 게이트는 **최소로** 둔다. 양쪽에 배포와 같은 min_votes를 걸면
    #    득표가 절반이라 "총 2×min_votes 이상"인 타자만 검증 대상이 되고,
    #    정작 배포되는 경계 구간이 표본에서 빠진다 (§7.39와 같은 모양의 구멍).
    ma = solve_mapping(batter_votes(mpa, tpa, ea)[0], m_ids, t_ids, m_hand, t_hand, 3, 0.5)[0]
    mb = solve_mapping(batter_votes(mpa, tpa, eb)[0], m_ids, t_ids, m_hand, t_hand, 3, 0.5)[0]
    both = set(ma) & set(mb)
    sel = both & set(mapping)                      # 배포될 집합만 본다
    bad = [k for k in sel if ma[k] != mb[k]]
    print(f"  A {len(ma)}명 / B {len(mb)}명 / 양쪽 모두 {len(both)}명")
    print(f"  배포 집합 {len(mapping)}명 중 검증 가능 {len(sel)}명 "
          f"(검증 못 함 {len(mapping)-len(sel)}명 — 경기 절반에만 등장)")
    print(f"  **일치 {len(sel)-len(bad)}/{len(sel)} = {(len(sel)-len(bad))/max(len(sel),1):.2%}**")
    if bad:
        print(f"  불일치: {bad}")

    print("\n  경력 구간별 (검증하려던 대상이 표본에 남아 있는지 확인):")
    for lo, hi, lab in [(0, 200, "~200"), (200, 1000, "200~1k"), (1000, 3000, "1k~3k"),
                        (3000, 10000, "3k~10k"), (10000, 10**9, "10k+")]:
        s = [k for k in sel if lo <= npitch.get(k, 0) < hi]
        if s:
            a = sum(ma[k] == mb[k] for k in s)
            print(f"    {lab:>7s}  {len(s):>4d}명  {a/len(s):>7.2%}")

    print("\n=== 6. 정합성 — 매핑된 쌍의 PA 수 비교 ===")
    tnpa = tpa.groupby("batter").size()
    ratio = (tnpa.reindex([mapping[k] for k in mapping]).to_numpy()
             / np.maximum(npa.reindex(list(mapping)).to_numpy(), 1))
    print(f"  tm/main PA 비율 분위: {np.round(np.nanquantile(ratio, [0.1,0.25,0.5,0.75,0.9]), 2)}")
    print("  (trackman에 퓨처스가 포함되므로 1.0 이상이 정상)")

    # 득표를 함께 저장한다 — 소비 측이 재실행 없이 게이트를 다시 걸 수 있고,
    # 신뢰도 가중이나 마스킹에 그대로 쓸 수 있다. (매핑만 담으면 그게 불가능하다.)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / "batter_id_map.csv"
    pd.DataFrame({
        "batter_id": list(mapping),
        "batter_trackman_id": [mapping[k] for k in mapping],
        "votes": [int(votes[k][0]) for k in mapping],
        "vote_share": [round(votes[k][0] / votes[k][1], 4) for k in mapping],
        "main_pitches": [int(npitch.get(k, 0)) for k in mapping],
    }).to_csv(out, index=False)
    print(f"\n  저장: {out}  (게이트 votes>={MIN_VOTES}, share>={MIN_SHARE})")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
