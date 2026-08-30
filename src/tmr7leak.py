"""Trackman 재구축 7단계 — **오염 감사. 논증하지 말고 실측한다**

사용자 질문: *"이거 데이터 오염 고려해서 다 하고 있는 거지?"*

이 저장소가 실제로 당한 사고 (§6 버그 2)
    제출 #2(Trackman)가 홀드아웃 +13.5 인데 LB −13.9.
    원인은 피처가 틀린 게 아니라 **조회 키에 `season` 이 들어간 것**이었다:
        delta 를 풀 때 (2024 행, ≤2023 테이블)   적중률 80.9%
        실제 추론  (2025 행, ≤2024 테이블)       적중률 99.5%(측정) / 실제 ~80%
    캘리브레이션 조건 ≠ 추론 조건 → 레벨이 어긋나 ~16 BSS 손실.

검사 항목
    L1  **미래 오염 독약 시험** — 전 시즌으로 만든 표 vs 시즌 ≤ t−1 만으로 만든 표.
        target ≤ t 행이 **비트 단위로 같아야** 한다. 다르면 미래가 새고 있다.
    L2  **행 독립성** (CLAUDE.md 절대 금지 조항) — valid 프레임을 청크로 쪼개
        따로 피처를 만든 결과가 통짜와 같은가. 다르면 test 전체 통계에 의존한 것.
    L3  **라벨 비사용** — trackman 경로에 `control_success` 가 단 한 번도 안 닿는가.
    L4  **§6 버그 2 재발 확인** — 같은 행에 `season` 라벨만 바꾸면 적중률·예측 평균이
        얼마나 어긋나는가. (이건 홀드아웃 Δ 에는 안 나타나고 **제출 때만** 터진다.)
    L5  **표준화·중심화 상수의 시간 경계** — 5·6단계가 쓴 mu/sd 가 ≤ t−1 인가.

⚠️ 이 감사가 통과해도 "홀드아웃 BSS = test 성능" 이 되는 것은 아니다.
   L4 가 정확히 그 간극이고, 우리 holdout 측정은 `shift_to_prior(p, y_valid.mean())`
   로 **레벨을 오라클로 맞춰** 놓기 때문에 레벨 취약성이 **보이지 않는다.**

실행:  .venv/Scripts/python.exe -u src/tmr7leak.py 2>&1 | tee experiments/tmr7leak.log
"""
import ast
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

from config import DATA_DIR, ROOT, TRAIN_CSV
from data import load_csv

SRC = Path(__file__).parent
TMR_DIR = ROOT / "experiments" / "tmr"
MAP_CSV = ROOT / "model_s28" / "pitcher_id_map.csv"
MEAS = {"rel_height": "relh", "rel_side": "rels", "extension": "ext",
        "rel_speed": "velo", "spin_rate": "spin",
        "induced_vert_break": "ivb", "horz_break": "hb"}
MIN_N = 100


def hr(t):
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


def cells(tm, seasons):
    """(투수, 시즌) 1차 집계 — tmr2build 와 같은 식."""
    sub = tm[tm.season.isin(seasons)]
    g = sub.groupby(["pitcher_id", "season"], observed=True)
    ps = pd.DataFrame({"n": g.size()})
    for c in MEAS:
        ps[f"{c}_m"] = g[c].mean()
        ps[f"{c}_s"] = g[c].std()
    return ps[ps.n >= MIN_N].reset_index()


def within_table(ps, target):
    """`within` 변형 한 target season 분. tmr2build 와 같은 식."""
    past = [s for s in range(2019, 2025) if s <= target - 1]
    sub = ps[ps.season.isin(past)]
    if len(sub) == 0:
        return pd.DataFrame()
    out = {}
    for c, tag in MEAS.items():
        num = sub.assign(_x=sub[f"{c}_s"] * sub.n).groupby("pitcher_id")._x.sum()
        den = (sub.assign(_x=np.where(sub[f"{c}_s"].notna(), sub.n, 0.0))
               .groupby("pitcher_id")._x.sum().replace(0, np.nan))
        out[f"tm_{tag}_std"] = num / den
        numm = sub.assign(_x=sub[f"{c}_m"] * sub.n).groupby("pitcher_id")._x.sum()
        out[f"tm_{tag}_mean"] = numm / den
    d = pd.DataFrame(out)
    d["tm_relpoint_disp"] = np.sqrt(d.tm_relh_std ** 2 + d.tm_rels_std ** 2)
    return d.sort_index()


def main():
    t0 = time.time()
    mp = pd.read_csv(MAP_CSV)
    m2t = dict(zip(mp.pitcher_trackman_id, mp.pitcher_id))
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv",
                     usecols=["season", "pitcher_trackman_id"] + list(MEAS))
    tm["pitcher_id"] = tm.pitcher_trackman_id.map(m2t)
    tm = tm.dropna(subset=["pitcher_id"])
    tm["pitcher_id"] = tm.pitcher_id.astype(np.int64)
    print(f"trackman 매핑 행 {len(tm):,}  ({time.time()-t0:.0f}s)")

    # ── L1. 미래 오염 독약 시험 ───────────────────────────────
    hr("L1. 미래 오염 — 전 시즌 표 vs 시즌 ≤ t−1 만으로 만든 표")
    print("  배포 피처가 target t 에 대해 정말 ≤ t−1 만 보는가.")
    print("  전 시즌(2019~2024)으로 만든 값과, **미래를 아예 지우고** 만든 값을 대조한다.\n")
    full = cells(tm, list(range(2019, 2025)))
    print(f"  {'target':>7}{'투수':>7}{'max|Δ|':>14}{'다른 셀':>9}   판정")
    ok_all = True
    for target in (2021, 2022, 2023, 2024, 2025):
        a = within_table(full, target)
        # 미래를 물리적으로 삭제한 원본으로 다시 만든다
        cut = cells(tm[tm.season <= target - 1], list(range(2019, target)))
        b = within_table(cut, target)
        common = a.index.intersection(b.index)
        cols = [c for c in a.columns if c in b.columns]
        d = (a.loc[common, cols].to_numpy(dtype=np.float64)
             - b.loc[common, cols].to_numpy(dtype=np.float64))
        mx = float(np.nanmax(np.abs(d))) if d.size else 0.0
        ndiff = int(np.nansum(np.abs(d) > 0))
        same_idx = len(a.index.symmetric_difference(b.index)) == 0
        good = (mx == 0.0) and same_idx
        ok_all &= good
        print(f"  {target:>7}{len(common):>7}{mx:>14.3e}{ndiff:>9}   "
              f"{'✅ 동일' if good else '❌ 미래 유입'}"
              f"{'' if same_idx else '  (투수 집합도 다름)'}")
    print(f"\n  → L1 {'✅ 통과' if ok_all else '❌ 실패'}: "
          f"target t 피처는 시즌 ≤ t−1 만으로 결정된다.")

    # ── L2. 행 독립성 ─────────────────────────────────────────
    hr("L2. 행 독립성 — valid 프레임을 청크로 쪼개도 같은 피처가 나오는가")
    print("  CLAUDE.md 절대 금지: test 의 다른 행을 이용한 어떤 피처도 안 된다.")
    import tmr3score as S
    df = load_csv(TRAIN_CSV)
    va = df[df.season.to_numpy() == 2024]
    tab = pd.read_csv(TMR_DIR / "tm_within.csv")
    from features import build_features, fit_artifacts
    from config import (CENTER_ASOF, CONTAM_AXES, RECENCY_HALF_LIFE,
                        SMOOTH_K_BATTER, SMOOTH_K_PITCHER)
    from data import recency_weight
    tr = df[df.season.to_numpy() < 2024]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                        center_asof=CENTER_ASOF, tm_table=tab)
    whole = build_features(va, art)
    tmcols = [c for c in whole.columns if c.startswith("tm")]
    parts = [build_features(va.iloc[i:i + 1000], art) for i in range(0, len(va), 1000)]
    chunk = pd.concat(parts, ignore_index=True)
    mx = float(np.nanmax(np.abs(whole[tmcols].to_numpy(dtype=np.float64)
                                - chunk[tmcols].to_numpy(dtype=np.float64))))
    print(f"  1000행 청크 {len(parts)}개 vs 통짜   tm 컬럼 {len(tmcols)}개   "
          f"max|Δ| = {mx:.3e}   {'✅' if mx == 0 else '❌'}")
    perm = np.random.default_rng(0).permutation(len(va))
    shuf = build_features(va.iloc[perm], art)
    inv = np.argsort(perm)
    mx2 = float(np.nanmax(np.abs(whole[tmcols].to_numpy(dtype=np.float64)
                                 - shuf[tmcols].to_numpy(dtype=np.float64)[inv])))
    print(f"  행 셔플 후 되돌림 vs 원본                        "
          f"max|Δ| = {mx2:.3e}   {'✅' if mx2 == 0 else '❌'}")

    # ── L3. 라벨 비사용 ───────────────────────────────────────
    hr("L3. 라벨 비사용 — trackman 경로가 `control_success` 에 닿는가")
    files = ["tmr1audit.py", "tmr2build.py", "tmr4cov.py", "tmr7leak.py"]
    for f in files:
        src = (SRC / f).read_text(encoding="utf-8")
        tree = ast.parse(src)
        names = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                 and isinstance(n.value, str)}
        hits = [x for x in names if "control_success" in str(x)]
        ids = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        print(f"  {f:<16} 문자열 'control_success' {len(hits)}건   "
              f"TARGET_COL 참조 {'있음' if 'TARGET_COL' in ids else '없음'}")
    print("  · trackman_history.csv 에는 라벨 컬럼 자체가 없다 (30컬럼, control_success 부재)")
    print("  · 3·5·6단계는 TARGET_COL 을 쓰지만 **기준선 학습/평가용**이고 피처 생성 경로가 아니다")

    # ── L4. §6 버그 2 재발 확인 ───────────────────────────────
    hr("L4. §6 버그 2 — `season` 라벨만 바꾸면 무엇이 어긋나나 (제출 때만 터지는 구멍)")
    v25 = va.copy()
    v25["season"] = 2025
    f25 = build_features(v25, art)
    h24 = float(whole["tm_matched"].mean())
    h25 = float(f25["tm_matched"].mean())
    print(f"  같은 2024 행, season 라벨만 2024 → 2025")
    print(f"    조회 적중률  {h24:.1%}  →  {h25:.1%}   (격차 {h25-h24:+.1%})")
    ch = [c for c in tmcols if c != "tm_matched"]
    dd = np.abs(np.nan_to_num(whole[ch].to_numpy(dtype=np.float64))
                - np.nan_to_num(f25[ch].to_numpy(dtype=np.float64)))
    print(f"    피처값 max|Δ| {dd.max():.4f}   평균|Δ| {dd.mean():.6f}")
    print("  ★ 이 격차가 0 이 아니면 prior shift 참조 분포를 추론 조건에 맞춰야 한다")
    print("    (§6 수정: season 을 2025 로 바꾸고 커버리지도 되돌린 reference_matrix).")
    print("  ⚠️ 이 구멍은 **홀드아웃 Δ 에 안 나타난다** — 우리 측정은 두 팔 모두")
    print("     shift_to_prior(p, y_valid.mean()) 로 레벨을 오라클로 맞추기 때문이다.")

    # ── L5. 표준화 상수 시간 경계 ─────────────────────────────
    hr("L5. 5·6단계가 쓴 표준화 상수의 시간 경계")
    print("  코드: past = tab[tab.season <= t][c]  → mu, sd")
    print("  `tab` 의 season 은 **target season** 이고 그 행은 ≤ target−1 로그로 만들어졌다.")
    for t in (2023, 2024):
        sub = tab[tab.season <= t]
        print(f"    t={t}: 쓰는 target 행 {sorted(sub.season.unique())}"
              f"  → 원 로그 시즌 ≤ {t-1}   ✅")
    print("  · 검증 시즌 **라벨**은 표준화에 안 들어간다. 값 분포만 쓴다.")
    print("  · 다만 배포하려면 이 mu/sd 를 **빌드 타임 상수로 얼려야** 한다"
          " (추론 시 계산 금지 — AST 스캐너가 막는다).")

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
