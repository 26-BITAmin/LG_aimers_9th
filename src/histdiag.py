"""§7.130 진단 — 결측 현황 · 신인 테이블 제안의 학습 없는 판정 · 이력 깊이 분포.

`histdepth.py`(재학습 판정)의 **앞단**이다. 여기 숫자는 전부 학습 0초로 나오고,
MODEL.md §7.130 에 인용된 값의 재현 스크립트다.

무엇을 답하나
    (1) 지금 결측이 실제로 얼마나 있고 누구인가
    (2) 사용자 제안 — "결측 이전 시즌에 등장했던 **신인들 통계**로 채우자" 의 크기
        원시 비교와 **(시즌 x game_type) 층화** 비교를 나란히 낸다.
        [[raw-transfer-confounds-level-with-structure]] 가 이 자리의 함정이다.
    (3) 그 조사에서 나온 표적 — prev{1,3,5} 의 **구조적 동일값** 비율

⚠️ (2) 에서 가중 방식이 결론을 바꾼다. 신인 1명당 1표로 세면 짧은(대개 나쁜)
   데뷔 등판이 과대대표된다. 실제로 채워질 것은 **행**이므로 데뷔경기 행 수로
   가중해야 한다. 두 값을 모두 낸다 — 안 내면 어느 쪽을 봤는지 나중에 모른다.

실행:  .venv/Scripts/python.exe -u src/histdiag.py 2>&1 | tee experiments/histdiag.log
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

from config import TRAIN_CSV

P1 = "asof_pitcher_prev1_game_success_rate"
P3 = "asof_pitcher_prev3_game_success_rate"
P5 = "asof_pitcher_prev5_game_success_rate"
M1 = "asof_pitcher_prev1_game_middle_rate"
Y = "control_success"


def sec(t):
    print(f"\n{'='*88}\n{t}\n{'='*88}")


def main():
    d = pd.read_csv(TRAIN_CSV).sort_values("row_id")
    n = len(d)

    sec("(1) 결측 현황")
    s = d.isna().sum()
    s = s[s > 0].sort_values(ascending=False)
    for k, v in s.items():
        print(f"   {k:45s} {v:9d}  {v/n:7.3%}")
    print(f"   {'(그 외 컬럼 전부)':45s} {0:9d}  {0.0:7.3%}")

    mp = d["asof_pitcher_success_rate"].isna()
    mv = d[P1].isna()
    print(f"\n   결측 블록이 3 덩어리다 -> features.py 의 miss 플래그도 3 개다.")
    print(f"   asof_pitcher_* 결측 : {int(mp.sum()):6d} 행 / 투수 {d.pitcher_id[mp].nunique():4d} 명"
          f"  asof_n 중앙 {d.asof_pitcher_n[mp].median()}")
    print(f"   prev*        결측 : {int(mv.sum()):6d} 행 / 투수 {d.pitcher_id[mv].nunique():4d} 명"
          f"  asof_n 중앙 {d.asof_pitcher_n[mv].median()}")
    print(f"   -> 같은 투수 집합이다. prev 결측 = 그 투수의 **데뷔 경기 전체**.")

    sec("(2a) 신인 제안 — 결측행의 실현 y (원시)")
    print(f"   전체 y        = {d[Y].mean():.5f}")
    print(f"   결측행 y      = {d[Y][mv].mean():.5f}   차 {d[Y][mv].mean()-d[Y].mean():+.5f}")
    print(f"   -> 원시로는 '신인이 더 잘한다'로 읽힌다. 아래에서 갈린다.")

    print(f"\n   시즌 x game_type 층별 (Δ = 신인 - 비신인, 같은 층 안에서)")
    print(f"   {'시즌':<6}{'gt':<4}{'신인행':>8}{'신인수':>8}{'y(비신인)':>12}"
          f"{'y(신인)':>10}{'Δ':>10}{'SE':>9}")
    for (sz, g), gg in d.groupby(["season", "game_type"]):
        m = gg[P1].isna()
        if m.sum() == 0:
            continue
        a, b = gg[Y][m], gg[Y][~m]
        dl = a.mean() - b.mean()
        se = np.sqrt(a.var() / len(a) + b.var() / len(b))
        print(f"   {sz:<6}{g:<4}{int(m.sum()):>8}{gg.pitcher_id[m].nunique():>8}"
              f"{b.mean():>12.4f}{a.mean():>10.4f}{dl:>+10.4f}{se:>9.4f}")

    def pooled(sub, lab):
        """층(시즌 x gt) 고정효과 안에서의 신인-비신인 차, 역분산 가중."""
        num = den = 0.0
        for _, gg in sub.groupby(["season", "game_type"]):
            m = gg[P1].isna()
            if m.sum() < 50 or (~m).sum() < 50:
                continue
            a, b = gg[Y][m], gg[Y][~m]
            v = a.var() / len(a) + b.var() / len(b)
            num += (a.mean() - b.mean()) / v
            den += 1 / v
        est, se = num / den, den ** -0.5
        print(f"   {lab:26s} Δ = {est:+.5f} ± {se:.5f}   t = {est/se:+.2f}")

    print(f"\n   층 고정효과 결합")
    pooled(d, "전체")
    pooled(d[d.game_type == "R"], "R만")
    pooled(d[d.game_type == "F"], "F만")
    pooled(d[d.season >= 2023], "2023+ (오염 F 제외 구간)")
    pooled(d[(d.season >= 2023) & (d.game_type == "R")], "2023+ R만")
    print(f"   -> 원시 +0.027 이 층화 후 -0.003 ± 0.003. 전량 층 구성비였다.")

    sec("(2b) 신인 제안 — '갓 만들어진' 신인 컬럼값 (제안의 정확한 형태)")
    deb = d[mv].groupby("pitcher_id").agg(GT=("game_type", "first"),
                                          SZN=("season", "first"),
                                          NR=("row_id", "size"))
    first = d[~mv].groupby("pitcher_id")[[P1, P3, P5, M1]].first()
    r = deb.join(first, how="inner")
    print(f"   신인 {len(r)} 명 (데뷔 + 다음 등판 모두 존재)")
    print(f"   ⚠️ 첫 값에서 prev1 == prev3 == prev5 인 비율: "
          f"{float((r[P1] == r[P3]).mean() * (r[P3] == r[P5]).mean()):.3f} "
          f"— 완료 경기가 1 개라 셋이 같은 경기다. (3) 의 발단.")

    base = d[~mv].groupby(["season", "game_type"])[[P1, M1]].mean()
    for col in (P1, M1):
        prior = d[col].mean()
        print(f"\n   [{col}]   현재 prior(전체 행 평균) = {prior:.5f}")
        print(f"   {'층':<10}{'신인':>6}{'행':>8}{'n_eff':>8}"
              f"{'vs prior':>11}{'vs 자기 층':>13}{'SE':>9}{'t':>7}")
        for g in ("F", "R"):
            for lab, sub in (("전체", r[r["GT"] == g]),
                             ("2023+", r[(r["GT"] == g) & (r["SZN"] >= 2023)])):
                v = sub[col].to_numpy()
                w = sub["NR"].to_numpy(float)
                b = base.loc[list(zip(sub["SZN"], sub["GT"])), col].to_numpy()
                ok = np.isfinite(v) & np.isfinite(b)
                v, w, b = v[ok], w[ok], b[ok]
                d_pr = np.average(v, weights=w) - prior
                d_st = np.average(v - b, weights=w)
                neff = w.sum() ** 2 / (w ** 2).sum()
                se = np.sqrt(np.average((v - b - d_st) ** 2, weights=w) / neff)
                print(f"   {g+' '+lab:<10}{len(v):>6}{int(w.sum()):>8}{neff:>8.1f}"
                      f"{d_pr:>+11.5f}{d_st:>+13.5f}{se:>9.5f}{d_st/se:>+7.2f}")
    print(f"\n   -> 'vs prior' 는 크지만 그중 대부분이 층 레벨이다. 'vs 자기 층'(=신인성)은")
    print(f"      전부 1.6 시그마 미만이고, 오염 F 를 빼면 n_eff 가 40 대로 떨어진다.")
    print(f"      F 는 전체(+) 와 2023+(-) 의 **부호가 반대**다 — train 전체로 표를")
    print(f"      적합하면 2025 에 정확히 반대 부호를 넣게 된다.")
    print(f"   -> 상한: b <= 0.009, f = 0.014  ->  f*b^2 = 1.1e-6  ->  +0.46 BSS. 기각.")

    sec("(3) 표적 — prev{1,3,5} 의 구조적 동일값 (= 이력 깊이)")
    ok = ~mv
    e13 = (d[P1] == d[P3]) & ok
    e35 = (d[P3] == d[P5]) & ok
    d0 = mv
    d1 = ok & (d[P1] == d[P3]) & (d[P3] == d[P5])
    d3 = ok & ~d1 & (d[P3] == d[P5])
    print(f"   p1 == p3            {int(e13.sum()):8d}  {e13.mean():7.3%}")
    print(f"   p3 == p5            {int(e35.sum()):8d}  {e35.mean():7.3%}")
    print(f"\n   hist_depth 분포")
    for lab, m in (("0  이력 없음", d0), ("1  완료 1 경기", d1),
                   ("2  완료 2~3", d3), ("3  완료 4+", ok & ~d1 & ~d3)):
        print(f"     depth={lab:16s} {int(m.sum()):8d}  {m.mean():7.3%}")
    print(f"\n   -> form_p3_minus_p5 이 구조적 고정값인 행 = depth 0+1+2 = "
          f"{float((d0|d1|d3).mean()):.3%}")
    print(f"      miss_prev_game 플래그가 덮는 것은 depth 0 ({d0.mean():.3%}) 뿐이다.")
    print(f"      **결측 표현이 이진인데 실제 조건은 서수다.** 표적 행 수가 4 배.")

    print(f"\n   깊이별 y (⚠️ 시즌 교란 있음 — 2019 는 depth<=2 비율이 14% 이고 y 도 최고다)")
    for lab, m in (("0", d0), ("1", d1), ("2", d3), ("3", ok & ~d1 & ~d3)):
        print(f"     depth={lab}  y = {d[Y][m].mean():.5f}  (n={int(m.sum())})")
    print(f"   -> 레벨에서 벌자는 게 아니다. 퇴화된 form_* 을 진짜 0 과 구별시키자는 것.")


if __name__ == "__main__":
    main()
