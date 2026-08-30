"""Trackman 재구축 1단계 — **원본 감사. 물려받은 것을 하나도 믿지 않는다**

사용자 지시: *"이전에 trackman 가지고 했던 것에 대해 신뢰를 많이 하지 말고 전부 다시 해.
기각 함부로 하지 말고, 무조건 holdout 2023 2024 결과를 보여줘."*

물려받지 않는 것 (전부 이 단계에서 다시 잰다)
    ✗ §7.9 의 "릴리스포인트 산포 10개" 선택
    ✗ §7.116a E1 의 형질/상태 분류표
    ✗ `trackman_pitcher.csv` 의 값
    ✗ §7.39 의 부호 반전 해석
    △ `pitcher_id_map.csv` 는 재검증 후 조건부 사용 (매핑 재구축은 5단계 비용이라
      먼저 **매핑을 쓰지 않는 독립 신호**로 검증한다)

이 단계가 겨냥하는 구조적 결함 후보 2개 (코드 읽고 지목, 사전 등록)
    D1  `trackman_features.py::_finish` 의 std 는 **시즌 ≤ t−1 을 한 덩어리로 합친
        pooled 산포**다. 시즌 내 일관성이 아니라 **시즌 간 표류가 섞여 있다.**
        팔각도를 바꾼 투수는 매 시즌 완벽히 일관돼도 큰 값을 받는다.
    D2  창 길이가 **target season 마다 다르다** (2020=1시즌 … 2025=6시즌).
        시즌 간 표류가 누적되므로 같은 투수의 값이 해마다 **기계적으로 커진다.**
        → E1 이 "인접 시즌 상관 0.225 → 상태다" 라고 읽은 것이 신호의 성질이 아니라
          **창 길이 아티팩트**일 수 있다. 그렇다면 §7.39·§7.116a 의 폐쇄 근거가 흔들린다.

    D3  측정 레짐 단절. §7.116a 가 각주로만 남긴 것 — relh_std 쌍별 상관이
        0.45/0.43/0.39 → **0.77/0.76** 으로 2022 이후 급등. 레짐이 갈리면 2019~2021 을
        같은 덩어리에 넣는 것 자체가 오염이다.

측정
    A. 원본 시즌별 규모·결측·측정값 분포 (레짐 단절 직접 확인)
    B. 매핑 독립 재검증 — **팀 이력 일치** (매핑 구축에 쓰이지 않은 신호)
    C. D1/D2 실측 — 시즌내 산포 vs pooled 산포, 창 길이 팽창
    D. 형질성 재측정 — **같은 창 길이**로 맞춘 뒤 인접 시즌 상관 (E1 재검)

실행:  .venv/Scripts/python.exe -u src/tmr1audit.py 2>&1 | tee experiments/tmr1audit.log
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

from config import DATA_DIR, MODEL_DIR, TRAIN_CSV

MEAS = ["rel_height", "rel_side", "extension", "rel_speed", "spin_rate",
        "induced_vert_break", "horz_break"]
MAP_CSV = MODEL_DIR.parent / "model_s28" / "pitcher_id_map.csv"
if not MAP_CSV.exists():
    MAP_CSV = Path(__file__).parent.parent / "model_s28" / "pitcher_id_map.csv"


def hr(t):
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


def main():
    t0 = time.time()

    # ── A. 원본 규모·결측·레짐 ────────────────────────────────
    use = ["season", "pitcher_trackman_id", "pitch_type_group", "tagged_pitch_type",
           "pitcher_team", "trackman_game_id"] + MEAS
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv", usecols=use)
    print(f"trackman_history {tm.shape}   ({time.time()-t0:.0f}s)")

    hr("A. 시즌별 규모 · 측정 결측률")
    g = tm.groupby("season")
    tab = pd.DataFrame({
        "행": g.size(),
        "투수": g.pitcher_trackman_id.nunique(),
        "경기": g.trackman_game_id.nunique(),
    })
    for c in MEAS:
        tab[f"{c[:6]}결측%"] = (g[c].apply(lambda s: s.isna().mean()) * 100).round(1)
    print(tab.to_string())

    hr("A2. 측정값 분포의 시즌별 변화 — 레짐 단절 직접 확인 (D3)")
    print("  전 투수 풀링 평균 / 표준편차. 시즌 사이에 계단이 있으면 레짐이 갈린 것이다.")
    for c in MEAS:
        m = g[c].mean()
        s = g[c].std()
        print(f"\n  {c}")
        print("    mean " + "  ".join(f"{y}:{m.get(y, np.nan):>8.3f}" for y in sorted(m.index)))
        print("    std  " + "  ".join(f"{y}:{s.get(y, np.nan):>8.3f}" for y in sorted(s.index)))

    # ── B. 매핑 독립 재검증 — 팀 이력 ─────────────────────────
    hr("B. 매핑 독립 재검증 — 투수별 **시즌 팀 이력** 일치 (매핑 구축에 안 쓰인 신호)")
    if not MAP_CSV.exists():
        print(f"  ⚠️ 매핑 파일 없음: {MAP_CSV}  — B 생략")
    else:
        mp = pd.read_csv(MAP_CSV)
        print(f"  매핑 {len(mp):,}쌍   {MAP_CSV}")
        m2t = dict(zip(mp.pitcher_trackman_id, mp.pitcher_id))

        main_df = pd.read_csv(TRAIN_CSV, usecols=["season", "pitcher_id", "pitcher_team_id"])
        # 투수-시즌 주팀 (최다 투구 팀)
        mt = (main_df.groupby(["pitcher_id", "season"]).pitcher_team_id
              .agg(lambda s: s.value_counts().index[0]).rename("main_team"))
        tt = tm.assign(pitcher_id=tm.pitcher_trackman_id.map(m2t)).dropna(subset=["pitcher_id"])
        tt["pitcher_id"] = tt.pitcher_id.astype(np.int64)
        tt = (tt.groupby(["pitcher_id", "season"]).pitcher_team
              .agg(lambda s: s.value_counts().index[0]).rename("tm_team"))
        j = pd.concat([mt, tt], axis=1).dropna()
        print(f"  대조 가능 (투수, 시즌) 쌍 {len(j):,}")

        # team_id ↔ team 문자열 대응을 **데이터에서 추정**한 뒤 일치율을 본다
        ct = pd.crosstab(j.main_team, j.tm_team)
        best = ct.idxmax(axis=1)
        hitn = sum(ct.loc[i, best[i]] for i in ct.index)
        print(f"  team_id → team 문자열 최빈 대응으로 잰 일치율 "
              f"**{hitn/len(j):.2%}** ({hitn:,}/{len(j):,})")
        print("  대응표:")
        for i in sorted(ct.index):
            row = ct.loc[i]
            print(f"    team_id {i:>3}  → {best[i]:<10} "
                  f"{row[best[i]]:>5,}/{row.sum():>5,} = {row[best[i]]/row.sum():>6.1%}")
        print("\n  ★ 이 신호는 매핑 알고리즘(하프이닝 투구수 지문 + 헝가리안)에 주어진 적이 없다.")
        print("    일치율이 높으면 매핑은 독립적으로 확인된 것이다.")

    # ── C. D1/D2 실측 ─────────────────────────────────────────
    hr("C. D1 · D2 실측 — pooled 산포는 시즌내 일관성이 아니다")
    tm["pitcher_id"] = tm.pitcher_trackman_id.map(
        dict(zip(mp.pitcher_trackman_id, mp.pitcher_id))) if MAP_CSV.exists() else np.nan
    tmm = tm.dropna(subset=["pitcher_id"]).copy()
    tmm["pitcher_id"] = tmm.pitcher_id.astype(np.int64)

    col = "rel_height"
    by = tmm.groupby(["pitcher_id", "season"])[col]
    ps = pd.DataFrame({"n": by.count(), "mean": by.mean(), "std": by.std()}).dropna()
    ps = ps[ps.n >= 200]
    print(f"  대상: (투수, 시즌) n≥200 인 {len(ps):,} 쌍   컬럼 {col}")

    print("\n  C1. 시즌내 산포 vs 시즌간 평균 표류 — 어느 쪽이 큰가")
    piv = ps.reset_index()
    wi = piv.groupby("pitcher_id")["std"].mean().rename("시즌내std_평균")
    be = piv.groupby("pitcher_id")["mean"].std().rename("시즌간mean_표류")
    cnt = piv.groupby("pitcher_id").size().rename("시즌수")
    cc = pd.concat([wi, be, cnt], axis=1)
    cc = cc[cc.시즌수 >= 3].dropna()
    print(f"    2시즌 이상 투수 {len(cc):,}명")
    print(f"    시즌내 std 평균      중앙값 {cc.시즌내std_평균.median():.4f}")
    print(f"    시즌간 mean 표류     중앙값 {cc.시즌간mean_표류.median():.4f}")
    ratio = (cc.시즌간mean_표류 / cc.시즌내std_평균).median()
    print(f"    → 표류/시즌내 비율   중앙값 **{ratio:.3f}**")
    print(f"    pooled std ≈ sqrt(시즌내² + 표류²) 이므로 이 비율이 클수록 "
          f"pooled 는 시즌내 일관성을 못 잰다")

    print("\n  C2. 창 길이 팽창 (D2) — 같은 투수, target season 별 pooled std")
    print("      배포 피처는 시즌 ≤ t−1 을 전부 합친다. 창이 길어지면 표류가 누적된다.")
    rows = []
    for target in range(2020, 2026):
        past = [s for s in range(2019, 2025) if s <= target - 1]
        sub = tmm[tmm.season.isin(past)]
        gg = sub.groupby("pitcher_id")[col]
        pooled = gg.std()
        n = gg.count()
        keep = pooled[n >= 500]
        rows.append((target, len(past), len(keep), float(keep.median())))
    print(f"    {'target':>7}{'창시즌수':>9}{'투수':>7}{'pooled std 중앙값':>19}")
    for t_, k, nn, md in rows:
        print(f"    {t_:>7}{k:>9}{nn:>7}{md:>19.4f}")
    base = rows[0][3]
    print(f"    → 2020(창 1시즌) 대비 2025(창 6시즌) "
          f"**{rows[-1][3]/base:.3f}배**  (1.0 이면 팽창 없음)")

    # ── D. 형질성 재측정 — 창 길이를 맞춘 뒤 ──────────────────
    hr("D. 형질성 재측정 (E1 재검) — 창 길이를 **1시즌으로 고정**하면 달라지나")
    print("  E1 은 배포 피처(누적창)로 인접 시즌 상관을 쟀다. 창 길이가 매년 다르면")
    print("  그 상관은 신호의 형질성이 아니라 창 아티팩트를 잰 것일 수 있다.")
    print(f"\n  {'컬럼':<20}{'누적창 ρ':>10}{'단일시즌 ρ':>12}{'쌍수':>7}   판정")

    for c in MEAS + ["_disp"]:
        if c == "_disp":
            sub = tmm.groupby(["pitcher_id", "season"])[["rel_height", "rel_side"]]
            d1 = sub.std()
            v_single = np.sqrt(d1.rel_height ** 2 + d1.rel_side ** 2).rename("v")
            nn = tmm.groupby(["pitcher_id", "season"]).size().rename("n")
            single = pd.concat([v_single, nn], axis=1)
            label = "relpoint_disp"
        else:
            gg = tmm.groupby(["pitcher_id", "season"])[c]
            single = pd.DataFrame({"v": gg.std(), "n": gg.count()})
            label = f"{c}_std"
        single = single[single.n >= 200].dropna()

        # 단일 시즌 창: (투수, 시즌) → (투수, 시즌+1) 상관
        s = single.reset_index()
        a = s.rename(columns={"v": "v0"})[["pitcher_id", "season", "v0"]]
        b = s.rename(columns={"v": "v1"})[["pitcher_id", "season", "v1"]]
        b["season"] = b.season - 1
        pair = a.merge(b, on=["pitcher_id", "season"])
        rho_single = pair.v0.corr(pair.v1) if len(pair) >= 30 else np.nan

        # 누적창: 배포 피처와 같은 방식 (시즌 ≤ t−1 pooled)
        accs = []
        for target in range(2021, 2026):
            past = [x for x in range(2019, 2025) if x <= target - 1]
            sb = tmm[tmm.season.isin(past)]
            if c == "_disp":
                q = sb.groupby("pitcher_id")[["rel_height", "rel_side"]].std()
                v = np.sqrt(q.rel_height ** 2 + q.rel_side ** 2)
                n = sb.groupby("pitcher_id").size()
            else:
                v = sb.groupby("pitcher_id")[c].std()
                n = sb.groupby("pitcher_id")[c].count()
            accs.append(pd.DataFrame({"pitcher_id": v.index, "season": target,
                                      "v": v.values, "n": n.reindex(v.index).values}))
        acc = pd.concat(accs).dropna()
        acc = acc[acc.n >= 200]
        a2 = acc.rename(columns={"v": "v0"})[["pitcher_id", "season", "v0"]]
        b2 = acc.rename(columns={"v": "v1"})[["pitcher_id", "season", "v1"]]
        b2["season"] = b2.season - 1
        p2 = a2.merge(b2, on=["pitcher_id", "season"])
        rho_acc = p2.v0.corr(p2.v1) if len(p2) >= 30 else np.nan

        flag = ""
        if np.isfinite(rho_single) and np.isfinite(rho_acc):
            if rho_single > rho_acc + 0.10:
                flag = "★ 단일창이 더 형질적 — E1 결론 흔들림"
            elif rho_acc > rho_single + 0.10:
                flag = "누적창이 더 안정 (표본 효과)"
        print(f"  {label:<20}{rho_acc:>10.3f}{rho_single:>12.3f}{len(pair):>7,}   {flag}")

    print("\n  ⚠️ 여기서 무엇을 기각하지 않는다. 다음 단계(재구축)의 설계 근거만 만든다.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
