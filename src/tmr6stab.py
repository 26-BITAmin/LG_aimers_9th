"""Trackman 재구축 6단계 — **기울기 시계열의 안정 성분** (§7.204c 와 같은 기계)

5단계가 준 것
    4시즌 사후 β 스윕 — **50개 컬럼 중 4시즌 전부 양수인 것 0개.**
    다만 사용자 지시가 "기각 함부로 하지 말 것" 이므로 BSS 부호로 닫지 않는다.
    BSS 부호는 β 격자·시즌 난이도에 민감하다 (base BSS 가 2022 1218.5 / 2023 676.6 로 2배 차).

이 단계가 묻는 것 — 부호가 아니라 **관계 자체**
    각 시즌 t 에서 배포 잔차에 컬럼을 회귀한 기울기 β_t (투수 클러스터 CR0 SE).
    4점 시계열에 **안정 성분**이 있나:
        평균  β̄ = Σw·β_t / Σw      (w = 1/SE²)
        t     β̄ / SE(β̄)            — 0 을 배제하나
        Q     Σw(β_t − β̄)²          — df 3, 7.81 이 5% 과분산선
    안정 성분이 없으면 (t 작거나 Q 과분산) **얼린 계수 하나가 못 나른다.**

    ★ 부호 일관성만으로는 증거가 안 된다: 4시즌 부호가 우연히 다 같을 확률 2/16=12.5%,
      50컬럼이면 기대 6.25개. 실제 관측 7개 — **정확히 우연 수준**이다.

영가설 — 죽은 축
    같은 컬럼의 **투수 배정을 셔플**한 위약. 값 분포와 잔차 구조는 그대로 두고
    투수↔값 연결만 끊는다. 위약이 같은 코드 경로로 같은 통계를 받는다.
    NULL_DRAWS 번 뽑아 |t| 의 분포를 낸다.

⚠️ 이건 **선형 가산 관계**만 본다. 3단계(재학습)는 트리가 비선형·상호작용까지 쓴다.
   둘을 **같이** 읽어야 한다. 어느 쪽도 단독 기각 근거가 아니다.

실행:  .venv/Scripts/python.exe -u src/tmr6stab.py 2>&1 | tee experiments/tmr6stab.log
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

import ctxfit
import foldcache
from config import CTX_AXES, CTX_K, MLP_ALPHA, ROOT, SEED, TARGET_COL, TRAIN_CSV
from data import load_csv
from features import logit_adjust
from validation import bss, shift_to_prior

SEASONS = [2021, 2022, 2023, 2024]
TMR_DIR = ROOT / "experiments" / "tmr"
NULL_DRAWS = 30
Q95 = 7.81          # chi2(3) 95%


def cluster_slope(r, d, g):
    dc = d - d.mean()
    den = float((dc ** 2).sum())
    if den <= 0:
        return 0.0, np.inf
    sl = float((dc * r).sum() / den)
    e = (r - r.mean()) - sl * dc
    h = dc * e
    _, inv = np.unique(g, return_inverse=True)
    G = inv.max() + 1
    sg = np.bincount(inv, weights=h, minlength=G)
    meat = float((sg ** 2).sum()) * G / max(G - 1, 1)
    return sl, float(np.sqrt(meat) / den)


def pool(slopes, ses):
    a, e = np.asarray(slopes, float), np.asarray(ses, float)
    ok = np.isfinite(a) & np.isfinite(e) & (e > 0)
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan
    a, e = a[ok], e[ok]
    w = 1.0 / e ** 2
    m = float((w * a).sum() / w.sum())
    sem = float(np.sqrt(1.0 / w.sum()))
    Q = float((w * (a - m) ** 2).sum())
    return m, sem, Q


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(SEASONS, verbose=False)
    rng = np.random.default_rng(SEED)

    base = {}
    for t in SEASONS:
        tr, va = s < t, s == t
        d = df[va]
        yv = cached[t][2]
        z = (1 - MLP_ALPHA) * cached[t][0] + MLP_ALPHA * cached[t][1]
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        adj, _ = logit_adjust(d, {"ctx": ctx})
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-(z + adj))), yv.mean())
        base[t] = dict(r=yv - p0, pid=d["pitcher_id"].to_numpy(dtype=np.int64))
    print(f"시즌 {SEASONS}   배포 잔차 준비 완료\n")

    rows = []
    for v in ("deploy", "within", "drift", "regime", "recent", "both"):
        p = TMR_DIR / f"tm_{v}.csv"
        if not p.exists():
            continue
        tab = pd.read_csv(p)
        cols = [c for c in tab.columns if c not in ("pitcher_id", "season", "tm_n")]
        for c in cols:
            sl, se = [], []
            for t in SEASONS:
                B = base[t]
                sub = tab[tab.season == t].set_index("pitcher_id")
                if c not in sub.columns:
                    sl.append(np.nan); se.append(np.nan); continue
                val = sub[c].reindex(B["pid"]).to_numpy(dtype=np.float64)
                past = tab[tab.season <= t][c].to_numpy(dtype=np.float64)
                mu, sd = np.nanmean(past), np.nanstd(past)
                vz = np.where(np.isnan(val), 0.0, (val - mu) / (sd if sd > 0 else 1.0))
                a, b = cluster_slope(B["r"], vz, B["pid"])
                sl.append(a); se.append(b)
            m, sem, Q = pool(sl, se)
            if not np.isfinite(m):
                continue
            sgn = "".join("+" if x > 0 else "−" for x in sl)
            rows.append((v, c, m, sem, m / sem if sem else np.nan, Q, sgn, sl))

    # ── 영가설: 투수 배정 셔플 위약 ────────────────────────────
    print("영가설 구축 — 투수↔값 연결만 끊은 셔플 위약 (같은 코드 경로)")
    nulls = []
    tab = pd.read_csv(TMR_DIR / "tm_within.csv")
    probe = [c for c in ("tm_hb_std", "tm_ivb_std", "tm_relpoint_disp") if c in tab.columns]
    for c in probe:
        for _ in range(NULL_DRAWS):
            sl, se = [], []
            for t in SEASONS:
                B = base[t]
                sub = tab[tab.season == t].set_index("pitcher_id")
                vals = sub[c].to_numpy(dtype=np.float64)
                ids = sub.index.to_numpy()
                perm = rng.permutation(len(vals))          # 투수 배정만 셔플
                shuf = pd.Series(vals[perm], index=ids)
                val = shuf.reindex(B["pid"]).to_numpy(dtype=np.float64)
                mu, sd = np.nanmean(vals), np.nanstd(vals)
                vz = np.where(np.isnan(val), 0.0, (val - mu) / (sd if sd > 0 else 1.0))
                a, b = cluster_slope(B["r"], vz, B["pid"])
                sl.append(a); se.append(b)
            m, sem, Q = pool(sl, se)
            if np.isfinite(m) and sem:
                nulls.append(abs(m / sem))
    nulls = np.array(nulls)
    print(f"  위약 {len(nulls)}개   |t| 중앙 {np.median(nulls):.2f}"
          f"  90% {np.quantile(nulls,0.9):.2f}  최대 {nulls.max():.2f}\n")

    rows.sort(key=lambda r: -abs(r[4]))
    print("=" * 108)
    print("기울기 시계열 안정 성분 — |t| 상위 20 (전체 {}컬럼)".format(len(rows)))
    print("=" * 108)
    print(f"  {'변형':<9}{'컬럼':<20}{'β̄':>10}{'SE':>9}{'t':>7}{'Q':>7}"
          f"{'부호':>7}{'위약초과':>9}   시즌별 β")
    for v, c, m, sem, t_, Q, sgn, sl in rows[:20]:
        over = "✓" if abs(t_) > nulls.max() else ("~" if abs(t_) > np.quantile(nulls, .9) else "")
        qf = "★과분산" if Q > Q95 else ""
        print(f"  {v:<9}{c:<20}{m:>+10.5f}{sem:>9.5f}{t_:>7.2f}{Q:>7.2f}"
              f"{sgn:>7}{over:>9}   " + " ".join(f"{x:+.4f}" for x in sl) + f"  {qf}")

    print("\n" + "=" * 108)
    print("읽는 법")
    print("=" * 108)
    print(f"  · |t| 가 위약 최대({nulls.max():.2f})를 못 넘으면 → 그 컬럼의 안정 성분은")
    print("    다중검정 잡음과 구분되지 않는다.")
    print(f"  · Q > {Q95} 면 과분산 — 시즌마다 관계가 달라서 **평균이 대표값이 아니다.**")
    print("    (§7.204c 가 `gap_strike` 를 이걸로 닫았다: Q 9.36)")
    print("  ⚠️ 선형 가산만 본다. 3단계(재학습)와 **같이** 읽는다. 단독 기각 근거 아님.")


if __name__ == "__main__":
    main()
