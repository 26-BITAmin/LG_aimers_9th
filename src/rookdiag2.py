"""rookdiag 후속 — 세 가지를 쌍체로 못 박는다.

① 투수 **간** 캘리브레이션 기울기의 신인−2+ 차. rookdiag 는 코호트별 CI 만 냈고
   두 CI 가 겹치는지로 읽으면 안 된다 (memory: "효과 크기에는 쌍체 SE를 붙일 것").
   같은 부트스트랩 뽑기 안에서 두 기울기를 동시에 계산해 차의 구간을 낸다.
② 압축비(예측 스프레드 / 실현 스프레드)의 신인−2+ 차 — 같은 방식.
③ **교란 분리**: 신인의 편향이 '경력' 때문인가 'asof 표본 수가 적어서'인가.
   같은 asof_pitcher_n 버킷 안에서 신인 vs 2+ 를 맞대면 경력 고유분만 남는다.

실행:  .venv/Scripts/python.exe -u src/rookdiag2.py 2>&1 | tee experiments/rookdiag2.log
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

import rookdiag as R
from config import TARGET_COL, TRAIN_CSV
from data import load_csv

B = 2000
RNG = np.random.default_rng(7)


def pitcher_agg(g, y, p, n_id):
    n = np.bincount(g, minlength=n_id)
    sy = np.bincount(g, weights=y, minlength=n_id)
    sp = np.bincount(g, weights=p, minlength=n_id)
    m = n > 0
    return n[m], sy[m] / n[m], sp[m] / n[m]


def slope_of(n, pb, yb):
    P = np.average(pb, weights=n)
    Y = np.average(yb, weights=n)
    d = np.sum(n * (pb - P) ** 2)
    return np.sum(n * (pb - P) * (yb - Y)) / d if d > 0 else np.nan


def paired_slope(gA, yA, pA, gB, yB, pB, n_id):
    """두 코호트의 투수간 기울기를 같은 뽑기에서 계산해 차의 구간을 낸다."""
    a = pitcher_agg(gA, yA, pA, n_id)
    b = pitcher_agg(gB, yB, pB, n_id)
    d = np.empty(B)
    sa = np.empty(B)
    sb = np.empty(B)
    for i in range(B):
        ka = RNG.integers(0, len(a[0]), len(a[0]))
        kb = RNG.integers(0, len(b[0]), len(b[0]))
        s1 = slope_of(a[0][ka], a[2][ka], a[1][ka])
        s2 = slope_of(b[0][kb], b[2][kb], b[1][kb])
        sa[i], sb[i], d[i] = s1, s2, s1 - s2
    return (slope_of(a[0], a[2], a[1]), slope_of(b[0], b[2], b[1]),
            d.mean(), np.percentile(d, [2.5, 97.5]), d.mean() / d.std())


def paired_spread(gA, yA, pA, gB, yB, pB, n_id, q=(0.1, 0.9)):
    """압축비 = (예측 분위폭)/(실현 분위폭). 투수 단위, n 가중 없이 분위."""
    a = pitcher_agg(gA, yA, pA, n_id)
    b = pitcher_agg(gB, yB, pB, n_id)

    def ratio(n, yb, pb, k):
        yq = np.quantile(yb[k], q)
        pq = np.quantile(pb[k], q)
        return (pq[1] - pq[0]) / (yq[1] - yq[0])

    d = np.empty(B)
    for i in range(B):
        ka = RNG.integers(0, len(a[0]), len(a[0]))
        kb = RNG.integers(0, len(b[0]), len(b[0]))
        d[i] = ratio(*a, ka) - ratio(*b, kb)
    ra = ratio(a[0], a[1], a[2], np.arange(len(a[0])))
    rb = ratio(b[0], b[1], b[2], np.arange(len(b[0])))
    return ra, rb, d.mean(), np.percentile(d, [2.5, 97.5]), d.mean() / d.std()


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()

    for fold in R.FOLDS:
        va, y, p, _ = R.build(fold, df, s)
        pp = R.prior_seasons(df, s, fold, "pitcher_id")
        an = va["asof_pitcher_n"].to_numpy(dtype=np.float64)
        g, uniq = pd.factorize(va["pitcher_id"].to_numpy())
        n_id = len(uniq)
        rook, vet = pp == 0, pp >= 2

        print("\n" + "=" * 96)
        print(f"fold {fold}")
        print("=" * 96)

        s1, s2, mu, ci, t = paired_slope(g[rook], y[rook], p[rook],
                                         g[vet], y[vet], p[vet], n_id)
        print(f"\n  ① 투수간 기울기   신인 {s1:.3f}   2+ {s2:.3f}"
              f"   차 {mu:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  t {t:+.1f}")
        print("     >1 = 투수 간 실력차를 압축해서 예측한다는 뜻")

        ra, rb, mu, ci, t = paired_spread(g[rook], y[rook], p[rook],
                                          g[vet], y[vet], p[vet], n_id)
        print(f"\n  ② 압축비(p10~p90)  신인 {ra:.3f}   2+ {rb:.3f}"
              f"   차 {mu:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  t {t:+.1f}")

        print("\n  ③ 같은 asof_pitcher_n 버킷 안에서 신인 vs 2+ (경력 고유분)")
        edges = [500, 1000, 2000, 4000, 8000]
        labs = ["<500", "500~999", "1k~2k", "2k~4k", "4k~8k", "8k+"]
        bn = np.where(np.isnan(an), -1, np.digitize(an, edges))
        print(f"    {'asof_n':<10}{'신인 n':>9}{'신인 ȳ':>9}{'신인 편향':>11}"
              f"{'2+ n':>9}{'2+ ȳ':>9}{'2+ 편향':>10}{'Δ편향':>10}{'95% CI':>24}")
        for i, lab in enumerate(labs):
            m1, m2 = rook & (bn == i), vet & (bn == i)
            if m1.sum() < 300 or m2.sum() < 300:
                print(f"    {lab:<10}{m1.sum():>9,}{'—':>9}{'—':>11}{m2.sum():>9,}")
                continue
            b1 = (p[m1] - y[m1]).mean()
            b2 = (p[m2] - y[m2]).mean()
            mu, ci, t = R.cl_boot(g, n_id, y, p, m1, m2, "bias")
            print(f"    {lab:<10}{m1.sum():>9,}{y[m1].mean():>9.4f}{b1:>+11.4f}"
                  f"{m2.sum():>9,}{y[m2].mean():>9.4f}{b2:>+10.4f}{mu:>+10.4f}"
                  f"{f'[{ci[0]:+.4f}, {ci[1]:+.4f}]':>24}")


if __name__ == "__main__":
    main()
