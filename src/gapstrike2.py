"""§7.204b — `gap_strike` 부호 반전이 **진짜 분열인가, 잡음 두 개인가**.

§7.204 는 배포층에서 tB = −3.73 (f2023) / +2.96 (f2024) 를 얻고 "폐쇄 유지" 로 읽었다.
그런데 그 t 는 `ctxsplit.resid_t` 의 **행 단위 SE** 다.

    se = sqrt( Σe² / (n−2) / Σd_c² )        n = 245,525 (f2023) · 253,507 (f2024)

⚠️ `gap_strike` 는 **투수-시즌 단위 양**이다 — `(올해 스트라이크율 − 커리어)` 라
   한 투수의 한 시즌 안에서 천천히만 움직인다. 잔차 e 도 투수 안에서 상관된다.
   그러면 행 단위 SE 는 유효 표본을 **투수 수(수백)가 아니라 행 수(수십만)** 로 세고,
   t 가 √(행/투수) ~ 20배 대역까지 부풀 수 있다.

   → **부호 반전이 "진짜 분열" 이려면 두 fold 각각이 자기 잡음을 이겨야 한다**
     ([[sign-isnt-evidence-when-the-interval-spans-zero]] · §7.200 의 그 지적).
     행 단위 t 로는 그 조건을 검사할 수 없다.

이 스크립트가 하는 것 — 같은 기울기에 **투수 클러스터 robust SE** 를 붙인다.

    slope = Σ(d_c · r) / Σ(d_c²)
    CR-SE = sqrt( Σ_g (Σ_{i∈g} d_c,i · e_i)² ) / Σ(d_c²) · G/(G−1)

    g = pitcher_id (축이 정의되는 단위). ([[weight-estimates-by-the-unit-theyll-be-applied-to]])

그리고 §7.200 D군과의 구별을 명시적으로 만든다.

    D군(채택해서 이긴 것들)은 **이득 구간이 한쪽 fold 에서 0 을 품었다** —
    즉 애초에 분열이 아니었다. `gap_strike` 가 D군이려면 클러스터 구간이 0 을 품어야 하고,
    C군(진짜 분열)이려면 두 fold 구간이 **각각** 0 을 배제해야 한다.

기준선은 §7.204 B팔(실제 배포: ctx + asof 3축, `gap_ball` 포함)을 그대로 쓴다.

실행:  .venv/Scripts/python.exe -u src/gapstrike2.py > experiments/gapstrike2.log 2>&1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import asofdelta
import foldcache
from config import SEED, TARGET_COL, TRAIN_CSV
from ctxsplit import resid_t
from data import load_csv
from gapstrike import (DEPLOYED_BASE, EXTRA_SPEC, FOLDS, GAP_BALL, baseline)
from mixresid import GAP_SPEC, build


def cluster_t(r, d, g):
    """기울기와 클러스터 robust t. g = 클러스터 라벨 (정수 배열)."""
    dc = d - d.mean()
    den = float((dc ** 2).sum())
    if den <= 0:
        return 0.0, 0.0, 0.0, 0
    sl = float((dc * r).sum() / den)
    e = (r - r.mean()) - sl * dc
    h = dc * e
    # 클러스터별 합 — bincount 로 벡터화
    _, inv = np.unique(g, return_inverse=True)
    G = inv.max() + 1
    sg = np.bincount(inv, weights=h, minlength=G)
    meat = float((sg ** 2).sum()) * G / max(G - 1, 1)
    se = float(np.sqrt(meat) / den)
    return sl, (sl / se if se > 0 else 0.0), se, G


def main():
    for k, v in EXTRA_SPEC.items():
        asofdelta.ASOF_SPEC.setdefault(k, v)

    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(SEED)

    print("§7.204b  gap_strike — 행 단위 t vs 투수 클러스터 t")
    print("  기준선 = §7.204 B팔 (실제 배포: ctx 2축 + asof gap_success/gap_bsuccess/gap_ball)\n")

    rows = {}
    for fold in FOLDS:
        tr, va = s < fold, s == fold
        d = df[va].copy()
        yv = cached[fold][2]
        pid = d["pitcher_id"].to_numpy(dtype=np.int64)
        bid = d["batter_id"].to_numpy(dtype=np.int64)

        art = asofdelta.fit_offsets(df[tr])
        league = float(df.loc[tr, TARGET_COL].mean())
        priors = {c: float(np.nanmean(df.loc[tr, c].to_numpy(dtype=np.float64)))
                  for _, c, _ in GAP_SPEC}
        feat = build(d, art, league, priors)
        dsh = d.copy()
        dsh["pitcher_id"] = rng.permutation(pid)
        dsh["batter_id"] = rng.permutation(bid)
        fsh = build(dsh, art, league, priors)

        rB, _ = baseline(df, tr, d, yv, cached[fold], DEPLOYED_BASE + [GAP_BALL])

        x = feat["gap_strike"]
        xs = fsh["gap_strike"]
        _, t_row = resid_t(rB, x)
        sl, t_cl, se_cl, G = cluster_t(rB, x, pid)
        _, tc_row = resid_t(rB, xs)
        _, tc_cl, _, _ = cluster_t(rB, xs, pid)

        # 축이 투수 안에서 얼마나 안 움직이는가 (= 유효 표본이 왜 작은가)
        _, inv = np.unique(pid, return_inverse=True)
        gm = np.bincount(inv, weights=x, minlength=inv.max() + 1) / np.bincount(inv, minlength=inv.max() + 1)
        within = float(np.sqrt(np.mean((x - gm[inv]) ** 2)))
        between = float(gm[np.unique(inv)].std())
        icc = between ** 2 / (between ** 2 + within ** 2)

        lo, hi = sl - 1.96 * se_cl, sl + 1.96 * se_cl
        rows[fold] = (sl, se_cl, t_row, t_cl, lo, hi)
        print("=" * 84)
        print(f"fold {fold}   행 {len(d):,}   투수 {G}명   행/투수 {len(d)/G:.0f}")
        print("=" * 84)
        print(f"  축 분산 구조   투수간 sd {between:.6f} · 투수내 sd {within:.6f} · **ICC {icc:.3f}**")
        print(f"  기울기         {sl:+.5f}")
        print(f"  행 단위 t      {t_row:+.2f}      (대조군 {tc_row:+.2f})")
        print(f"  클러스터 t     {t_cl:+.2f}      (대조군 {tc_cl:+.2f})     SE {se_cl:.5f}")
        print(f"  t 부풀림 배율  {abs(t_row)/max(abs(t_cl), 1e-9):.1f}배")
        print(f"  95% 구간       [{lo:+.5f}, {hi:+.5f}]   0 포함 {'예' if lo < 0 < hi else '아니오'}")

    print("\n" + "=" * 84)
    print("판정")
    print("=" * 84)
    a, b = rows[FOLDS[0]], rows[FOLDS[1]]
    print(f"  {'fold':<8}{'기울기':>12}{'클러스터 t':>13}{'95% 구간':>30}{'0 배제':>9}")
    for f in FOLDS:
        sl, se, tr_, tc, lo, hi = rows[f]
        print(f"  {f:<8}{sl:>+12.5f}{tc:>13.2f}   [{lo:+.5f}, {hi:+.5f}]{'예' if not (lo < 0 < hi) else '아니오':>9}")
    excl = [f for f in FOLDS if not (rows[f][4] < 0 < rows[f][5])]
    print(f"\n  0 을 배제하는 fold: {excl if excl else '없음'}")
    if len(excl) == 2:
        print("  -> **진짜 분열**(§7.200 C군). 두 fold 가 각각 자기 잡음을 이기고 반대를 가리킨다. 폐쇄.")
    elif len(excl) <= 1:
        print("  -> **분열이 아니다**(§7.200 D군 자리). 한쪽 이상이 무정보다.")
        print("     §7.201: 이 자리에서 옳은 행동은 기각이 아니라 사전확률·비용으로 결정하는 것.")
    # 두 fold 기울기 차이가 유의한가 (독립 fold 이므로 SE 합성)
    dsl = b[0] - a[0]
    dse = float(np.sqrt(a[1] ** 2 + b[1] ** 2))
    print(f"\n  기울기 차 (f2024 − f2023) = {dsl:+.5f} ± {dse:.5f}   z = {dsl/dse:+.2f}"
          f"   {'유의' if abs(dsl/dse) > 1.96 else '유의하지 않다'}")


if __name__ == "__main__":
    main()
