"""§7.167 정정 — 두 가지 오류를 고치고 매칭을 제대로 다시 한다.

오류 ①  `rookdiag.py` §5 의 버킷 라벨이 한 칸씩 밀렸다.
        `np.digitize(an, edges) + 1` 인데 라벨은 0 부터 붙였다. 최상위 구간
        (10,000+, 행의 5~7%)은 아예 출력되지 않았다. 숫자는 맞고 이름이 틀렸다.

오류 ②  '2+시즌'을 **등장한 시즌 수**로 정의했다. 1이닝 콜업도 한 시즌이다.
        그래서 '통산 500구 미만인 2+시즌 투수' 45명이 나왔는데, 이들은 베테랑이
        아니라 4년간 총 186구(중앙값) 던진 주변부 투수다.
        → 저표본 칸의 매칭은 "신인 vs 베테랑"이 아니라 "신인 vs 주변부"였다.

무엇을 다시 묻나
    경력 축과 표본 수 축은 **저표본 구간에서 구조적으로 분리 불가**다
    (통산 기록이 얇은 진짜 베테랑은 존재할 수 없다). 그래서 분리 가능한
    구간이 어디까지인지부터 재고, **그 안에서만** n-표준화 대비를 낸다.
    덮지 못하는 신인 행 비중을 같이 보고해야 결론의 사정거리가 정해진다.

실행:  .venv/Scripts/python.exe -u src/rookdiag5.py 2>&1 | tee experiments/rookdiag5.log
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
from config import TRAIN_CSV
from data import load_csv

B = 2000
RNG = np.random.default_rng(11)
EDGES = [1, 100, 500, 2000, 10000]
LABS = ["0 (기록 없음)", "1~99", "100~499", "500~1,999", "2,000~9,999", "10,000+"]
# 매칭용 촘촘한 격자
FINE = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000]
MIN_CELL = 200


def bucket(an, edges):
    """NaN 은 -1. 그 외는 digitize 그대로 (라벨 인덱스와 1:1)."""
    return np.where(np.isnan(an), -1, np.digitize(an, edges))


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()

    for fold in R.FOLDS:
        va, y, p, _ = R.build(fold, df, s)
        pp = R.prior_seasons(df, s, fold, "pitcher_id")
        an = va["asof_pitcher_n"].to_numpy(dtype=np.float64)
        pid = va["pitcher_id"].to_numpy()
        g, uniq = pd.factorize(pid)
        n_id = len(uniq)
        rook, vet = pp == 0, pp >= 2
        b = bucket(an, EDGES)

        print("\n" + "=" * 98)
        print(f"fold {fold}  n={len(y):,}   [정정판]")
        print("=" * 98)

        print(f"\n  §A  asof_pitcher_n 버킷 — **라벨 정정**. 편향 = 예측평균 − 실제평균")
        print(f"    {'통산 기록':<16}{'전체 n':>10}{'비중':>7}{'편향':>9}"
              f"{'|  신인 n':>11}{'편향':>9}{'|  2+ n':>10}{'편향':>9}")
        for i, lab in enumerate(LABS):
            m = b == i
            if m.sum() == 0:
                continue
            mr, mv = m & rook, m & vet
            f_r = f"{(p[mr]-y[mr]).mean():+.4f}" if mr.sum() >= MIN_CELL else "—"
            f_v = f"{(p[mv]-y[mv]).mean():+.4f}" if mv.sum() >= MIN_CELL else "—"
            print(f"    {lab:<16}{m.sum():>10,}{m.mean()*100:>6.1f}%"
                  f"{(p[m]-y[m]).mean():>+9.4f}{mr.sum():>11,}{f_r:>9}"
                  f"{mv.sum():>10,}{f_v:>9}")
        tot = sum((b == i).sum() for i in range(len(LABS))) + (b == -1).sum()
        assert tot == len(y), (tot, len(y))
        print(f"    (합계 검증 {tot:,} = 전체 {len(y):,} ✓  — 구판은 최상위 구간이 누락됐다)")

        print(f"\n  §B  '2+시즌인데 통산 기록이 얇은' 집단의 정체")
        low = vet & (an < 500)
        ids = np.unique(pid[low])
        prior_tot = df.loc[s < fold].groupby("pitcher_id", observed=True).size()
        pt = prior_tot.reindex(ids).fillna(0)
        print(f"    행 {low.sum():,} · 투수 {len(ids)}명")
        print(f"    이들이 직전 시즌들에 던진 총 투구수: 중앙값 {pt.median():,.0f}구 "
              f"· 최대 {pt.max():,.0f}구")
        print(f"    (참고: 풀타임 선발 1시즌 ≈ 2,500~3,000구)")
        print(f"    → 베테랑이 아니라 **여러 해에 걸쳐 조금씩 던진 주변부 투수**다.")

        print(f"\n  §C  겹침 구간에서만 n-표준화 대비 (양쪽 다 {MIN_CELL}행 이상인 칸만)")
        fb = bucket(an, FINE)
        cells, wr = [], []
        for i in range(len(FINE) + 1):
            m1, m2 = rook & (fb == i), vet & (fb == i)
            if m1.sum() >= MIN_CELL and m2.sum() >= MIN_CELL:
                cells.append(i)
                wr.append(m1.sum())
        wr = np.array(wr, dtype=float)
        wr /= wr.sum()
        lo = ["0~99"] + [f"{FINE[j-1]:,}~{FINE[j]-1:,}" for j in range(1, len(FINE))] \
             + [f"{FINE[-1]:,}+"]
        print(f"    {'통산 기록':<16}{'신인 n':>9}{'신인 편향':>11}"
              f"{'2+ n':>9}{'2+ 편향':>10}{'Δ':>10}")
        for i, w in zip(cells, wr):
            m1, m2 = rook & (fb == i), vet & (fb == i)
            b1, b2 = (p[m1] - y[m1]).mean(), (p[m2] - y[m2]).mean()
            print(f"    {lo[i]:<16}{m1.sum():>9,}{b1:>+11.4f}{m2.sum():>9,}"
                  f"{b2:>+10.4f}{b1-b2:>+10.4f}")
        cov = sum((rook & (fb == i)).sum() for i in cells) / rook.sum()
        print(f"\n    겹침 구간이 덮는 신인 행: {cov*100:.1f}%"
              f"   (나머지 {100-cov*100:.1f}% 는 대조군 자체가 없다)")

        # 클러스터 부트스트랩 — n-표준화 Δ편향
        cl = g
        out = np.empty(B)
        pre = []
        for i in cells:
            m1, m2 = rook & (fb == i), vet & (fb == i)
            pre.append((np.bincount(cl, weights=m1, minlength=n_id),
                        np.bincount(cl, weights=m1 * (p - y), minlength=n_id),
                        np.bincount(cl, weights=m2, minlength=n_id),
                        np.bincount(cl, weights=m2 * (p - y), minlength=n_id)))
        for t in range(B):
            k = RNG.integers(0, n_id, n_id)
            acc, wsum = 0.0, 0.0
            for w, (n1, s1, n2, s2) in zip(wr, pre):
                a1, a2 = n1[k].sum(), n2[k].sum()
                if a1 == 0 or a2 == 0:
                    continue
                acc += w * (s1[k].sum() / a1 - s2[k].sum() / a2)
                wsum += w
            out[t] = acc / wsum if wsum > 0 else np.nan
        o = out[~np.isnan(out)]
        ci = np.percentile(o, [2.5, 97.5])
        point = sum(w * ((p[rook & (fb == i)] - y[rook & (fb == i)]).mean()
                         - (p[vet & (fb == i)] - y[vet & (fb == i)]).mean())
                    for i, w in zip(cells, wr))
        print(f"    n-표준화 Δ편향 (신인−2+)  {point:+.4f}"
              f"  [{ci[0]:+.4f}, {ci[1]:+.4f}]  t {o.mean()/o.std():+.1f}")
        raw = (p[rook] - y[rook]).mean() - (p[vet] - y[vet]).mean()
        print(f"    표준화 전 Δ편향           {raw:+.4f}"
              f"   → 표본 수를 맞추면 {abs(point)/abs(raw)*100:.0f}% 만 남는다"
              if raw != 0 else "")


if __name__ == "__main__":
    main()
