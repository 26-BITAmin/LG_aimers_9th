"""Trackman 재구축 4단계 — **조회 커버리지 감사. 2025 에 실제로 몇 %가 값을 받나**

사용자 질문: *"매핑이 어떻게 쓰이는 거지? 2025 는 trackman 데이터 없어서 매핑 못하는 거 알지?"*

답의 구조
    매핑은 **빌드 타임 자산**이다. 추론에는 쓰이지 않는다.
        빌드   trackman_history(pitcher_trackman_id) --매핑--> pitcher_id 로 재키잉
               → (pitcher_id, season) → 요약값 테이블로 얼림
        추론   test 행의 pitcher_id 로 (pitcher_id, 2025) 조회 → 얼린 값
    2025 에 매핑할 것은 없다. `season=2025` 키는 "2019~2024 전부를 쓴 창" 이라는 뜻일 뿐이고,
    제출물에 trackman ID 는 한 개도 들어가지 않는다.

    ★ 그런데 질문이 겨냥한 진짜 문제는 **조회 적중률**이다.
      2019~2024 에 없던 투수는 2025 에 값이 없다. 그리고 그 비율이 매년 커진다.
      앞 실험 실측: fold2023 86.6% → fold2024 80.3%. **창이 길어지는데 적중률이 떨어진다.**

이 파일이 재는 것
    C1  변형별 · fold 별 조회 적중률 — BSS 표를 읽을 때 필요한 교란 통제
        (재구축이 `MIN_N=100` 필터를 넣어 변형마다 커버리지가 다르다. `recent` 는 특히 낮다)
    C2  미적중의 구성 분해 — 신규 투수 / 미매핑 / 소표본 중 무엇이 범인인가
    C3  2025 투영 — 구조적 유사물(각 시즌의 '직전 이력 없는 투수 행 비율')로 외삽
    C4  **적중 행만의 BSS 기여 상한** — 커버리지가 이득을 얼마나 희석하는가

⚠️ 여기서 아무것도 기각하지 않는다. 3단계 BSS 표를 **읽는 법**을 만든다.

실행:  .venv/Scripts/python.exe -u src/tmr4cov.py 2>&1 | tee experiments/tmr4cov.log
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

from config import DATA_DIR, ROOT, TRAIN_CSV
from data import load_csv

TMR_DIR = ROOT / "experiments" / "tmr"
MAP_CSV = ROOT / "model_s28" / "pitcher_id_map.csv"
OLD_TM = ROOT / "model_s28" / "trackman_pitcher.csv"
VARIANTS = ["deploy", "within", "drift", "regime", "recent"]
SEASONS = list(range(2019, 2025))


def hr(t):
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    pid = df["pitcher_id"].to_numpy(dtype=np.int64)

    # ── C1. 변형별 · 시즌별 조회 적중률 ────────────────────────
    hr("C1. 조회 적중률 — 변형별 × 시즌별 (행 기준)")
    print("  배포 피처는 (pitcher_id, season) 키다. 그 시즌 행이 값을 받는 비율.")
    tabs = {}
    for v in VARIANTS:
        p = TMR_DIR / f"tm_{v}.csv"
        if not p.exists():
            continue
        t = pd.read_csv(p)
        tabs[v] = {int(se): set(g.pitcher_id.astype(np.int64))
                   for se, g in t.groupby("season")}
    if OLD_TM.exists():
        t = pd.read_csv(OLD_TM)
        tabs["기존(s28)"] = {int(se): set(g.pitcher_id.astype(np.int64))
                             for se, g in t.groupby("season")}

    names = list(tabs)
    print(f"\n  {'시즌':<8}{'행 수':>11}" + "".join(f"{n:>12}" for n in names))
    for se in range(2020, 2025):
        m = s == se
        row = []
        for n in names:
            ids = tabs[n].get(se, set())
            row.append(np.isin(pid[m], list(ids)).mean() if ids else 0.0)
        print(f"  {se:<8}{int(m.sum()):>11,}" + "".join(f"{x:>11.1%} " for x in row))

    # 2025 는 main 데이터가 없다 — 테이블에 든 투수 수만 본다
    print(f"\n  {'2025 (test)':<8}{'—':>11}"
          + "".join(f"{len(tabs[n].get(2025, set())):>11,} " for n in names))
    print("  ↑ 2025 행은 우리에게 없다. 위 숫자는 **테이블에 값이 있는 투수 수**다.")

    # ── C2. 미적중 분해 ────────────────────────────────────────
    hr("C2. 미적중의 정체 — 신규 투수 / 미매핑 / 소표본")
    mp = pd.read_csv(MAP_CSV)
    mapped = set(mp.pitcher_id.astype(np.int64))
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv",
                     usecols=["season", "pitcher_trackman_id"])
    t2m = dict(zip(mp.pitcher_trackman_id, mp.pitcher_id))
    tm["pitcher_id"] = tm.pitcher_trackman_id.map(t2m)
    tm = tm.dropna(subset=["pitcher_id"])
    tm["pitcher_id"] = tm.pitcher_id.astype(np.int64)
    tm_n = tm.groupby(["pitcher_id", "season"]).size()

    print(f"  {'시즌':<7}{'행':>10}{'적중':>8}{'미적중':>8}"
          f"{'①신규(main 이력無)':>20}{'②미매핑':>10}{'③소표본':>10}")
    for se in range(2020, 2025):
        m = s == se
        prev = set(pid[s < se])                       # main 기준 직전 이력
        have = set(tm_n[tm_n.index.get_level_values("season") < se]
                   .index.get_level_values("pitcher_id"))
        big = set(tm_n[(tm_n.index.get_level_values("season") < se) & (tm_n >= 100)]
                  .index.get_level_values("pitcher_id"))
        pv = pid[m]
        hit = np.isin(pv, list(big))
        new = ~np.isin(pv, list(prev))
        unmapped = np.isin(pv, list(prev)) & ~np.isin(pv, list(mapped))
        small = np.isin(pv, list(have)) & ~np.isin(pv, list(big))
        print(f"  {se:<7}{int(m.sum()):>10,}{hit.mean():>8.1%}{1-hit.mean():>8.1%}"
              f"{new.mean():>19.1%}{unmapped.mean():>10.1%}{small.mean():>10.1%}")

    # ── C3. 2025 투영 ──────────────────────────────────────────
    hr("C3. 2025 투영 — 신규 투수 비율의 시계열로 외삽")
    print("  §7.203 이 같은 축을 다른 목적(콜드스타트)으로 이미 쟀다: 2024 신규 19.9%.")
    rates = []
    for se in range(2020, 2025):
        m = s == se
        prev = set(pid[s < se])
        r = float((~np.isin(pid[m], list(prev))).mean())
        rates.append((se, r))
        print(f"    {se}  직전 이력 없는 투수 행 {r:>6.1%}")
    x = np.array([a for a, _ in rates], dtype=float)
    y = np.array([b for _, b in rates])
    sl, ic = np.polyfit(x, y, 1)
    print(f"\n    선형 추세 기울기 {sl:+.4f}/년   2025 외삽 **{sl*2025+ic:.1%}**")
    print(f"    → 2025 조회 적중률 대략 **{1-(sl*2025+ic):.0%} 안팎** (신규 투수는 정의상 미적중)")
    print("    ⚠️ 2점이 아니라 5점 추세지만 외삽은 외삽이다. 구간으로 읽을 것.")

    # ── C4. 희석 계수 ──────────────────────────────────────────
    hr("C4. 커버리지가 이득을 얼마나 희석하나")
    print("  trackman 이 적중 행에서만 일한다면, 전체 BSS 이득 ≈ (적중률) × (적중 행 이득).")
    print("  홀드아웃과 test 의 적중률이 다르면 **홀드아웃 이득은 test 를 과대평가한다.**")
    for v in names:
        c23 = np.isin(pid[s == 2023], list(tabs[v].get(2023, set()))).mean()
        c24 = np.isin(pid[s == 2024], list(tabs[v].get(2024, set()))).mean()
        p25 = 1 - (sl * 2025 + ic)
        print(f"    {v:<12} f2023 {c23:>6.1%}  f2024 {c24:>6.1%}   "
              f"→ 2025 대비 희석비 {p25/max(c24,1e-9):>5.2f}× (f2024 기준)")
    print("\n  ★ 희석비 < 1 이면 홀드아웃 이득이 test 에서 그만큼 줄어든다는 뜻이다.")
    print("  ⚠️ 이건 '적중 행에서만 일한다'는 가정 위의 상한 계산이다. 실제로는 tm_matched")
    print("     플래그 자체가 피처라 미적중 행도 (약하게) 영향을 받는다.")


if __name__ == "__main__":
    main()
