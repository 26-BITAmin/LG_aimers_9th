"""EDA — 데이터 구성과 신호 지도. 이 저장소에 처음 만드는 EDA 산출물이다.

왜 지금 하는가
    §7.33까지 33개 실험이 전부 기각됐다. 노브를 더 돌리는 대신 **데이터가 실제로
    어떻게 생겼는지**를 본다. 지금까지 측정된 적 없는 것 다섯 가지:

    A. 데이터 구성 프로파일 (결측·카디널리티·시즌 구성)
    B. **실패 모드 항등식** — 타깃은 실패 3종의 합성인데 `asof_*`에 그 3종이
       따로 들어 있다. 관계가 성립하는지 본 적이 없다
    C. **컬럼별 단독 신호량** — 각 컬럼 하나만으로 낼 수 있는 BSS oracle.
       "신호가 어디 있는지"를 대회 지표 단위로 직접 읽는 유일한 방법이다
    D. 시즌 드리프트 (분포 이동 + 신호 이동)
    E. cold-start 실체 — 신규 선수 비율과 그 구간의 점수
    F. **배포 모델 잔차 지도** — 어느 세그먼트에서 점수를 잃고 있나 (캐시 사용)

측정 규약
    · 모든 신호량은 **시즌 홀드아웃**이다 (train ≤ t-1 → valid t). 랜덤 분할 금지.
    · **BSS oracle**로 보고한다 — 예측 평균을 검증셋 실제 평균으로 이동시킨 뒤의
      점수. 레벨 성분을 빼서 순수 refinement만 남긴다 (MODEL.md §2 규약).
    · 빈 평균은 경험적 베이즈로 스무딩한다 (k=200). 안 하면 희소 빈이 신호를 위조한다.

실행:  .venv/Scripts/python.exe -u src/eda.py 2>&1 | tee experiments/eda.log
산출:  experiments/eda.json  (시각화가 이걸 읽는다)
"""
import gc
import json
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

import foldcache
from config import EXP_DIR, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

SMOOTH_K = 200.0          # 빈 평균 스무딩. 희소 빈이 가짜 신호를 만드는 것을 막는다
N_QBINS = 32
MAX_LEVELS = 40           # 고유값이 이보다 적으면 분위수 대신 값 그대로 쓴다

OUT = {}


def sig(v_tr, y_tr, v_va, y_va):
    """컬럼 하나만으로 낼 수 있는 BSS oracle. train에서 빈 평균을 fit,
    valid에 적용 → 오라클 레벨 이동 → BSS.

    반환: (BSS oracle, 빈 수, valid 커버리지)
    """
    fin_tr = np.isfinite(v_tr)
    fin_va = np.isfinite(v_va)
    if fin_tr.sum() < 1000:
        return None

    u = np.unique(v_tr[fin_tr])
    if len(u) <= MAX_LEVELS:
        edges = u
        b_tr = np.searchsorted(edges, v_tr, side="left")
        b_va = np.searchsorted(edges, v_va, side="left")
    else:
        qs = np.linspace(0, 1, N_QBINS + 1)[1:-1]
        edges = np.unique(np.quantile(v_tr[fin_tr], qs))
        b_tr = np.searchsorted(edges, v_tr, side="right")
        b_va = np.searchsorted(edges, v_va, side="right")
    nb = len(edges) + 1

    # 결측은 **자기 빈**을 갖는다 — 결측 자체가 신호다 (cold-start)
    b_tr = np.where(fin_tr, b_tr, nb)
    b_va = np.where(fin_va, b_va, nb)
    b_tr = np.clip(b_tr, 0, nb)
    b_va = np.clip(b_va, 0, nb)

    prior = float(y_tr.mean())
    cnt = np.bincount(b_tr, minlength=nb + 1).astype(np.float64)
    tot = np.bincount(b_tr, weights=y_tr, minlength=nb + 1)
    rate = (tot + SMOOTH_K * prior) / (cnt + SMOOTH_K)

    p = rate[b_va]
    return (float(bss(y_va, shift_to_prior(p, y_va.mean()))), int((cnt > 0).sum()),
            float(fin_va.mean()))


def numeric(df, col):
    """category/object 컬럼도 숫자 코드로 눌러서 돌려준다."""
    s = df[col]
    if str(s.dtype) in ("category", "object"):
        return s.astype("category").cat.codes.to_numpy(dtype=np.float64)
    return s.to_numpy(dtype=np.float64)


def main():
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    df = load_csv(TRAIN_CSV)
    season = df["season"].to_numpy()
    y_all = df[TARGET_COL].to_numpy(dtype=np.float64)
    print(f"train {df.shape}  ({time.time()-t0:.0f}s)")

    # ══ A. 데이터 구성 ═════════════════════════════════════════════
    print("\n" + "=" * 88)
    print("A. 데이터 구성")
    print("=" * 88)

    seasons = sorted(set(season.tolist()))
    comp = []
    for s in seasons:
        m = season == s
        comp.append({"season": int(s), "n": int(m.sum()),
                     "rate": float(y_all[m].mean()),
                     "pitchers": int(df.loc[m, "pitcher_id"].nunique()),
                     "batters": int(df.loc[m, "batter_id"].nunique()),
                     "miss_pitcher_asof": float(
                         df.loc[m, "asof_pitcher_success_rate"].isna().mean()),
                     "miss_prev_game": float(
                         df.loc[m, "asof_pitcher_prev1_game_success_rate"].isna().mean())})
    OUT["composition"] = comp
    print(f"{'시즌':>6} {'행수':>10} {'양성률':>8} {'투수':>6} {'타자':>6}"
          f" {'asof결측':>9} {'prev결측':>9}")
    for r in comp:
        print(f"{r['season']:>6} {r['n']:>10,} {r['rate']:>8.4f} {r['pitchers']:>6}"
              f" {r['batters']:>6} {r['miss_pitcher_asof']:>9.2%}"
              f" {r['miss_prev_game']:>9.2%}")

    cols = [c for c in df.columns if c not in ("row_id", TARGET_COL)]
    prof = []
    for c in cols:
        v = df[c]
        prof.append({"col": c, "dtype": str(v.dtype), "nunique": int(v.nunique()),
                     "miss": float(v.isna().mean())})
    OUT["profile"] = prof
    miss = [p for p in prof if p["miss"] > 0]
    print(f"\n결측이 있는 컬럼 {len(miss)}개 / 전체 {len(prof)}개")
    for p in sorted(miss, key=lambda x: -x["miss"]):
        print(f"  {p['col']:<45} {p['miss']:>7.2%}")

    # ══ B. 실패 모드 항등식 ════════════════════════════════════════
    print("\n" + "=" * 88)
    print("B. 실패 모드 — 타깃은 실패 3종의 합성이다. asof에 그 3종이 따로 있다")
    print("=" * 88)
    a = {c: df[c].to_numpy(dtype=np.float64) for c in
         ("asof_pitcher_success_rate", "asof_pitcher_middle_rate",
          "asof_pitcher_reverse_rate", "asof_pitcher_ball_rate",
          "asof_pitcher_strike_rate")}
    m = np.isfinite(np.stack(list(a.values()))).all(0)
    fail = 1.0 - a["asof_pitcher_success_rate"][m]
    comps = {k.replace("asof_pitcher_", "").replace("_rate", ""): a[k][m]
             for k in ("asof_pitcher_middle_rate", "asof_pitcher_reverse_rate",
                       "asof_pitcher_ball_rate", "asof_pitcher_strike_rate")}
    ident = {}
    print(f"  유효행 {m.sum():,}   1−success 평균 {fail.mean():.4f}")
    for k, v in comps.items():
        c_ = float(np.corrcoef(fail, v)[0, 1])
        ident[k] = {"mean": float(v.mean()), "corr_with_fail": c_}
        print(f"  {k:<10} 평균 {v.mean():.4f}   corr(1−success) {c_:>+.4f}")
    mr = comps["middle"] + comps["reverse"]
    ident["middle_plus_reverse"] = {
        "mean": float(mr.mean()),
        "corr_with_fail": float(np.corrcoef(fail, mr)[0, 1]),
        "mean_resid": float((fail - mr).mean()),
        "std_resid": float((fail - mr).std())}
    print(f"  middle+reverse 평균 {mr.mean():.4f}   corr {ident['middle_plus_reverse']['corr_with_fail']:>+.4f}"
          f"   잔차(1−success − 합) 평균 {ident['middle_plus_reverse']['mean_resid']:+.4f}"
          f" ± {ident['middle_plus_reverse']['std_resid']:.4f}")
    print(f"  ball+strike 평균 {(comps['ball']+comps['strike']).mean():.4f}"
          f"   ← 1.0에 가까우면 이 둘은 판정 분해라 실패 3종과 다른 축이다")
    OUT["failure_modes"] = ident
    del a, comps
    gc.collect()

    # ══ C. 컬럼별 단독 신호량 ══════════════════════════════════════
    print("\n" + "=" * 88)
    print("C. 컬럼별 단독 신호량 — BSS oracle (시즌 홀드아웃, 빈 평균 k=200 스무딩)")
    print("=" * 88)
    print("   각 컬럼 하나만으로 낼 수 있는 점수다. 배포 시스템은 fold2024에서 약 770.")
    rows = []
    for c in cols:
        if c == "season":
            continue
        v = numeric(df, c)
        r = {}
        for va in (2023, 2024):
            tr_m, va_m = season < va, season == va
            g = sig(v[tr_m], y_all[tr_m], v[va_m], y_all[va_m])
            if g is None:
                continue
            r[va] = g
        if 2024 not in r:
            continue
        rows.append({"col": c, "bss2023": r.get(2023, (None,))[0],
                     "bss2024": r[2024][0], "bins": r[2024][1],
                     "cover": r[2024][2]})
    rows.sort(key=lambda x: -(x["bss2024"] or -1e9))
    OUT["univariate"] = rows
    print(f"\n{'컬럼':<45} {'BSS2023':>9} {'BSS2024':>9} {'빈':>5} {'커버':>7}")
    for r in rows:
        b23 = "     —   " if r["bss2023"] is None else f"{r['bss2023']:>9.1f}"
        print(f"{r['col']:<45} {b23} {r['bss2024']:>9.1f} {r['bins']:>5}"
              f" {r['cover']:>7.1%}")

    # ══ D. 시즌 드리프트 ═══════════════════════════════════════════
    print("\n" + "=" * 88)
    print("D. 드리프트 — 신호량이 시즌마다 얼마나 흔들리나 (상위 컬럼)")
    print("=" * 88)
    top = [r["col"] for r in rows[:12]]
    drift = {}
    for c in top:
        v = numeric(df, c)
        series = []
        for va in (2021, 2022, 2023, 2024):
            tr_m, va_m = season < va, season == va
            g = sig(v[tr_m], y_all[tr_m], v[va_m], y_all[va_m])
            series.append(None if g is None else g[0])
        drift[c] = series
        print(f"{c:<45} " + " ".join(
            "     —  " if x is None else f"{x:>8.1f}" for x in series))
    OUT["drift"] = drift
    print("   (열 = valid 2021 / 2022 / 2023 / 2024)")

    # ══ E. cold-start 실체 ═════════════════════════════════════════
    print("\n" + "=" * 88)
    print("E. cold-start — 검증 시즌에 처음 나오는 선수의 비중과 그 구간 점수")
    print("=" * 88)
    cold = []
    for va in (2023, 2024):
        tr_m, va_m = season < va, season == va
        known_p = set(df.loc[tr_m, "pitcher_id"].unique().tolist())
        known_b = set(df.loc[tr_m, "batter_id"].unique().tolist())
        pn = df.loc[va_m, "pitcher_id"].isin(known_p).to_numpy()
        bn = df.loc[va_m, "batter_id"].isin(known_b).to_numpy()
        yv = y_all[va_m]
        cold.append({"valid": va,
                     "new_pitcher_share": float((~pn).mean()),
                     "new_batter_share": float((~bn).mean()),
                     "rate_known_pitcher": float(yv[pn].mean()),
                     "rate_new_pitcher": float(yv[~pn].mean()) if (~pn).any() else None,
                     "miss_asof_share": float(
                         df.loc[va_m, "asof_pitcher_success_rate"].isna().mean())})
        r = cold[-1]
        print(f"  valid {va}: 신규 투수 투구 {r['new_pitcher_share']:.2%}"
              f" / 신규 타자 {r['new_batter_share']:.2%}"
              f" | 양성률 기존 {r['rate_known_pitcher']:.4f}"
              f" vs 신규 {r['rate_new_pitcher']:.4f}"
              f" | asof 결측 {r['miss_asof_share']:.2%}")
    OUT["cold_start"] = cold

    # ══ F. 배포 모델 잔차 지도 ═════════════════════════════════════
    print("\n" + "=" * 88)
    print("F. 잔차 지도 — 배포 모델(트리 앙상블 + MLP 블렌드)이 어디서 점수를 잃나")
    print("=" * 88)
    cached = foldcache.load([2024], verbose=False)
    z_lgb, z_mlp, yv = cached[2024][0], cached[2024][1], cached[2024][2]
    z = 0.6 * z_lgb + 0.4 * z_mlp
    p = shift_to_prior(1.0 / (1.0 + np.exp(-z)), yv.mean())
    va_m = season == 2024
    d24 = df.loc[va_m].reset_index(drop=True)
    assert len(d24) == len(yv), f"정렬 불일치 {len(d24)} vs {len(yv)}"
    print(f"  fold2024 전체 BSS oracle = {bss(yv, p):.1f}   n={len(yv):,}")

    # 신뢰도 곡선
    qs = np.quantile(p, np.linspace(0, 1, 21))
    qs = np.unique(qs)
    b = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, len(qs) - 2)
    rel = []
    for i in range(len(qs) - 1):
        m_ = b == i
        if m_.sum() < 100:
            continue
        rel.append({"pred": float(p[m_].mean()), "actual": float(yv[m_].mean()),
                    "n": int(m_.sum())})
    OUT["reliability"] = rel
    gap = max(abs(r["pred"] - r["actual"]) for r in rel)
    print(f"  신뢰도 곡선 {len(rel)}개 빈, 최대 |예측−실제| = {gap:.4f}")

    # 세그먼트별 손익 — 이 세그먼트를 그 세그먼트의 실제 평균으로만 예측했을 때
    # 대비 모델이 얼마나 더 벌었나 (= 세그먼트 내부 refinement)
    ref = yv.mean() * (1 - yv.mean())
    segs = {
        "count_state": (d24["balls_before"].to_numpy() * 3
                        + d24["strikes_before"].to_numpy()),
        "inning": np.clip(d24["inning"].to_numpy(), 1, 10),
        "base_state": d24["base_state"].astype("category").cat.codes.to_numpy(),
        "hand_matchup": (d24["pitcher_hand"].to_numpy() * 4
                         + d24["batter_hand"].to_numpy()),
        "game_month": d24["game_month"].to_numpy(),
        "asof_pitcher_n_q": np.clip(np.searchsorted(
            np.quantile(d24["asof_pitcher_n"].to_numpy(dtype=np.float64),
                        np.linspace(0, 1, 9)[1:-1]),
            d24["asof_pitcher_n"].to_numpy(dtype=np.float64)), 0, 7),
        "miss_pitcher_asof": d24["asof_pitcher_success_rate"].isna()
                             .to_numpy().astype(int),
    }
    seg_out = {}
    for name, key in segs.items():
        rowsg = []
        for k in np.unique(key):
            m_ = key == k
            n_ = int(m_.sum())
            if n_ < 500:
                continue
            ys, ps = yv[m_], p[m_]
            # 세그먼트 내부 refinement: 세그먼트 평균 상수 대비 이득
            bs_model = float(((ps - ys) ** 2).mean())
            bs_seg = float(ys.mean() * (1 - ys.mean()))
            rowsg.append({"level": float(k), "n": n_, "rate": float(ys.mean()),
                          "pred": float(ps.mean()),
                          "refine": 100000.0 * (bs_seg - bs_model) / ref,
                          "level_err": 100000.0 * ((ps.mean() - ys.mean()) ** 2) / ref})
        seg_out[name] = rowsg
        print(f"\n  [{name}]  {'수준':>8} {'n':>9} {'실제':>8} {'예측':>8}"
              f" {'내부refine':>11} {'레벨손실':>9}")
        for r in sorted(rowsg, key=lambda x: -x["refine"])[:12]:
            print(f"           {r['level']:>8.0f} {r['n']:>9,} {r['rate']:>8.4f}"
                  f" {r['pred']:>8.4f} {r['refine']:>11.1f} {r['level_err']:>9.1f}")
    OUT["segments"] = seg_out

    path = EXP_DIR / "eda.json"
    path.write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {path}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
