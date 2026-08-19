"""투수내 예측 변동의 '시계'는 달력인가 누적 투구수인가 (사용자 질문).

투수 평균을 뺀 예측 변동 안에는 "모델이 이 투수를 알아가는 과정"이 들어 있다.
그 진행을 재는 축 후보가 둘이다:
    ① game_month        — 달력. 4월/5월/…
    ② asof_pitcher_n    — 그 투수가 지금까지 던진 누적 투구수 (경력 누적)

둘은 한 투수 안에서 같이 증가하므로 상관이 높다. 가르는 지점은 **신인 대 베테랑**:
    신인   4월 n=0 → 9월 n=900       두 축이 같이 움직인다 (구분 어려움)
    베테랑 4월 n=12,000 → 9월 13,500  달력은 똑같이 흐르지만 n은 비율상 거의 안 움직인다
따라서 "n이 진짜 시계"라면 **설명력이 신인에서 크고 베테랑에서 작아야** 한다.
"달력이 시계"라면 두 집단에서 비슷해야 한다.

각 축에 대해 두 가지를 낸다 (투수별 평균 제거 후):
    설명 비율 = 그 축의 버킷 평균이 설명하는 투수내 예측 분산 비율
    성분 기울기 = Σ(p̄_b−p̄_j)(y−ȳ_j) / Σ(p̄_b−p̄_j)²   1이면 그 성분이 정확

실행:  LGA_MODEL_DIR=submit/model_clockaxis .venv/Scripts/python.exe -u src/clockaxis.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache
from config import TRAIN_CSV
from data import load_csv
from rookie import FOLDS, deployed_pred, prior_seasons

B = 1000
RNG = np.random.default_rng(42)


def gmean(g, v):
    K = g.max() + 1
    n = np.bincount(g, minlength=K)
    return np.bincount(g, weights=v, minlength=K) / np.maximum(n, 1)


def comp(cl, bkt, y, p):
    """축 성분의 (설명 분산, 기울기 분자·분모) — 투수 평균 제거 기준."""
    pj, yj = gmean(cl, p)[cl], gmean(cl, y)[cl]
    g = cl.astype(np.int64) * 1000 + bkt
    _, g = np.unique(g, return_inverse=True)
    pb, yb = gmean(g, p)[g], gmean(g, y)[g]
    dpb = pb - pj                      # 축이 만든 성분
    var_w = ((p - pj) ** 2).mean()
    return (dpb ** 2).mean() / var_w, dpb * (y - yj), dpb ** 2, var_w


def se_slope(cl, num, den):
    C = cl.max() + 1
    n = np.bincount(cl, weights=num, minlength=C)
    d = np.bincount(cl, weights=den, minlength=C)
    out = np.empty(B)
    for b in range(B):
        i = RNG.integers(0, C, C)
        out[b] = n[i].sum() / d[i].sum()
    return float(np.nanstd(out, ddof=1))


def qbucket(cl, v, k=5):
    """투수별 v의 k분위 버킷."""
    out = np.zeros(len(v), dtype=np.int64)
    for j in range(cl.max() + 1):
        m = cl == j
        if m.sum() < k * 2:
            continue
        vv = v[m]
        out[m] = np.clip(np.searchsorted(np.quantile(vv, np.arange(1, k) / k), vv), 0, k - 1)
    return out


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)

    for fold in FOLDS:
        va, y, p = deployed_pred(df, s, fold, cached)
        gt_f = va["game_type"].astype("object").to_numpy() == "F"
        pp = prior_seasons(df, s, fold, "pitcher_id")
        pid = va["pitcher_id"].to_numpy()
        mon = va["game_month"].to_numpy().astype(np.int64)
        an = va["asof_pitcher_n"].to_numpy().astype(float)

        print(f"\n=== fold {fold} · 정규시즌(R) ===")
        print(f"  {'집단':<10}{'축':<16}{'설명비율':>9}{'성분 기울기':>13}{'±SE':>8}")
        for lab, m in (("신인", (~gt_f) & (pp == 0)), ("베테랑", (~gt_f) & (pp >= 2))):
            yg, pg = y[m], p[m]
            _, cl = np.unique(pid[m], return_inverse=True)
            for axname, bkt in (("달력(월)", mon[m] - mon[m].min()),
                                ("누적 투구수 5분위", qbucket(cl, an[m], 5)),
                                ("누적 투구수 10분위", qbucket(cl, an[m], 10))):
                sh, num, den, _ = comp(cl, bkt, yg, pg)
                sl = num.sum() / den.sum()
                print(f"  {lab:<10}{axname:<16}{sh*100:>8.1f}%{sl:>13.2f}"
                      f"{se_slope(cl, num, den):>8.2f}")

            # 두 축을 같이 넣었을 때 — 달력이 n 위에 뭘 더 얹나
            g2 = qbucket(cl, an[m], 5) * 100 + (mon[m] - mon[m].min())
            _, g2 = np.unique(g2, return_inverse=True)
            sh, num, den, _ = comp(cl, g2, yg, pg)
            print(f"  {lab:<10}{'n5분위 × 월':<16}{sh*100:>8.1f}%"
                  f"{num.sum()/den.sum():>13.2f}{se_slope(cl, num, den):>8.2f}")


if __name__ == "__main__":
    main()
