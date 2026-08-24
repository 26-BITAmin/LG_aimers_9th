"""주자 유무 δ가 왜 +0.20에 그치는가 — 분산 분해 진단 (§7.37 후속).

§7.37은 T1 corr .111/.037로 축을 기각했다. 판정은 맞았지만 **왜 그 값인가**는
남겨뒀다. T1은 두 가지를 섞는다:
    ⓐ 추정 노이즈  — 셀당 표본이 적어 δ̂이 흔들린다        → 표본을 늘리면 산다
    ⓑ 비지속       — 진짜 효과가 해마다 다른 모양이다      → 영원히 안 산다
처방이 정반대이므로 갈라야 한다.

측정 (전부 이항 대비, 셀당 MIN_N=100):
    L  리그 주효과   조건별 성공률 차. 0에 가까우면 δ는 순수 개인차 게임이다.
    S  관측 산포     투수별 δ̂의 표준편차 (축소 없음)
    N  노이즈 산포   이항 분산에서 오는 기대 산포 (적률법)
    T  진짜 산포     sqrt(max(0, S² − N²))          ← 이게 축의 실제 물건 크기
    R  신뢰도        T²/S²   한 시즌 안에서 δ̂이 얼마나 믿을 만한가
    P  시즌 간 상관  corr(δ̂_2023, δ̂_2024)
    ρ  탈감쇠 지속   P / R    진짜 효과 자체가 해를 넘는가 (1이면 완전 지속)

대조군: hand(채택·배포, T1 .47), count4 성분 2개(채택·배포, T1 .30/.40).
같은 잣대로 돌려야 "주자가 작다"가 무엇 대비 작은지 말할 수 있다.

실행:  .venv/Scripts/python.exe -u src/runnerwhy.py 2>&1 | tee experiments/runnerwhy.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

from config import TARGET_COL, TRAIN_CSV

MIN_N = 100          # ctxsplit과 동일
SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]

USE = ["season", "pitcher_id", TARGET_COL, "runner_on_1b", "num_runners_on",
       "runner_on_2b", "runner_on_3b", "base_state", "batter_hand",
       "balls_before", "strikes_before"]


def axes(df):
    """(이름, 마스크1, 마스크0, 기전) — 이항 대비만."""
    b = df["balls_before"].to_numpy(np.int64)
    s = df["strikes_before"].to_numpy(np.int64)
    r1 = df["runner_on_1b"].to_numpy(np.int64)
    nr = df["num_runners_on"].to_numpy(np.int64)
    r2 = df["runner_on_2b"].to_numpy(np.int64)
    r3 = df["runner_on_3b"].to_numpy(np.int64)
    bh = df["batter_hand"].to_numpy(np.int64)
    bh_lo = np.unique(bh)[0]
    return [
        ("hand  [배포]",      bh == bh_lo,        bh != bh_lo,   "좌우 매치업"),
        ("3볼   [배포·성분]", b >= 3,             b < 3,         "볼넷 회피 압박"),
        ("2스트 [배포·성분]", s >= 2,             s < 2,         "유인구"),
        ("runner_on_1b",      r1 == 1,            r1 == 0,       "세트포지션"),
        ("주자유무",          nr > 0,             nr == 0,       "주자 있음 전반"),
        ("득점권",            (r2 + r3) > 0,      nr == 0,       "실점 부담"),
        ("만루",              nr == 3,            nr == 0,       "최대 압박"),
    ]


def cell_table(y, pid, m1, m0):
    """투수별 (n1, k1, n0, k0)."""
    order = np.argsort(pid, kind="stable")
    p_s, y_s, m1_s, m0_s = pid[order], y[order], m1[order], m0[order]
    uniq, start = np.unique(p_s, return_index=True)
    n1 = np.add.reduceat(m1_s.astype(np.int64), start)
    k1 = np.add.reduceat((m1_s * y_s).astype(np.int64), start)
    n0 = np.add.reduceat(m0_s.astype(np.int64), start)
    k0 = np.add.reduceat((m0_s * y_s).astype(np.int64), start)
    return uniq, n1, k1, n0, k0


def decompose(uniq, n1, k1, n0, k0):
    ok = (n1 >= MIN_N) & (n0 >= MIN_N)
    if ok.sum() < 30:
        return None
    u, n1, k1, n0, k0 = uniq[ok], n1[ok], k1[ok], n0[ok], k0[ok]
    p1, p0 = k1 / n1, k0 / n0
    d = p1 - p0
    # 가중치: 대비의 정보량 (역분산)
    var_i = p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0
    S2 = d.var(ddof=1)                 # 관측 산포²
    N2 = var_i.mean()                  # 노이즈 산포² (적률법)
    T2 = max(0.0, S2 - N2)
    return dict(u=u, d=d, n=len(u), S=np.sqrt(S2), N=np.sqrt(N2),
                T=np.sqrt(T2), R=T2 / S2 if S2 > 0 else 0.0)


def main():
    df = pd.read_csv(TRAIN_CSV, usecols=USE, encoding="utf-8-sig")
    y_all = df[TARGET_COL].to_numpy(np.int64)
    print(f"train {df.shape}   전체 성공률 {y_all.mean():.4f}\n")

    # ── L: 리그 주효과 (시즌별) ──────────────────────────────
    print("=" * 96)
    print("L. 리그 주효과 — 조건별 성공률 차 (조건1 − 조건0), 시즌별")
    print("=" * 96)
    print(f"{'축':<20s}" + "".join(f"{s:>9d}" for s in SEASONS) + f"{'전체':>10s}{'노출%':>8s}")
    for name, m1, m0, _ in axes(df):
        row = []
        for s in SEASONS:
            sm = (df["season"].to_numpy() == s)
            a, b_ = y_all[sm & m1], y_all[sm & m0]
            row.append(a.mean() - b_.mean() if len(a) and len(b_) else np.nan)
        tot = y_all[m1].mean() - y_all[m0].mean()
        print(f"{name:<20s}" + "".join(f"{v:>+9.4f}" for v in row)
              + f"{tot:>+10.4f}{100*m1.mean():>7.1f}%")

    # ── S/N/T/R + 시즌 간 지속 ───────────────────────────────
    print("\n" + "=" * 96)
    print("δ 분산 분해 — 투수별 (조건1 − 조건0), 셀당 n>=100")
    print("=" * 96)
    print(f"{'축':<20s}{'시즌':>6s}{'투수':>6s}{'S 관측':>9s}{'N 노이즈':>10s}"
          f"{'T 진짜':>9s}{'R 신뢰도':>10s}   {'P 23-24':>8s}{'ρ 탈감쇠':>9s}")
    for name, m1, m0, mech in axes(df):
        res, keep = {}, {}
        for s in (2023, 2024):
            sm = (df["season"].to_numpy() == s)
            r = decompose(*cell_table(y_all[sm], df["pitcher_id"].to_numpy()[sm],
                                      m1[sm], m0[sm]))
            res[s] = r
            if r:
                keep[s] = dict(zip(r["u"], r["d"]))
        both = sorted(set(keep.get(2023, {})) & set(keep.get(2024, {})))
        if len(both) >= 30:
            a = np.array([keep[2023][p] for p in both])
            b_ = np.array([keep[2024][p] for p in both])
            P = np.corrcoef(a, b_)[0, 1]
            Rg = np.sqrt(max(1e-9, res[2023]["R"] * res[2024]["R"]))
            rho = P / Rg
        else:
            P, rho = np.nan, np.nan
        for s in (2023, 2024):
            r = res[s]
            if r is None:
                print(f"{name:<20s}{s:>6d}{'—':>6s}   (셀 부족)")
                continue
            tail = (f"   {P:>+8.3f}{rho:>+9.3f}" if s == 2024 else "")
            print(f"{name:<20s}{s:>6d}{r['n']:>6d}{r['S']:>9.4f}{r['N']:>10.4f}"
                  f"{r['T']:>9.4f}{r['R']:>10.3f}{tail}")
        print(f"{'':<20s}{'짝both':>6s}{len(both):>6d}   ← 두 시즌 모두 n>=100 인 투수  ({mech})")
    print("\n해석: T가 축의 실제 물건 크기다. 이득은 대략 T²·ρ 에 비례한다.")
    print("      R이 낮으면 표본 문제(→ 축소·통합으로 회복 가능),")
    print("      ρ가 낮으면 효과 자체가 해를 못 넘는다(→ 회복 불가).")


if __name__ == "__main__":
    main()
