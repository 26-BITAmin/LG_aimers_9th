"""§7.79 투수내 기울기를 **눈으로 보이게** + 원인 절반 가르기.

(A) 시각 확인
    투수별 평균을 뺀 뒤, "모델이 평균보다 얼마나 높게 봤나"(d_p)로 10분위를 나누고
    각 칸에서 실제 성공률 편차(d_y)를 본다. 기울기가 1이면 두 열이 같아야 한다.

(B) 흔들림의 출처 가르기
    투수내 예측 변동은 두 종류가 섞여 있다.
      ① 시즌 진행에 따른 asof 누적치의 이동 (신인은 시즌 초 리그평균 → 후반 개인기록)
      ② 같은 시점에서의 상황 차이 (볼카운트·주자·타자)
    투수 × (그 투수의 asof_n 5분위)로 평균을 한 번 더 빼면 ①이 지워지고 ②만 남는다.
    신인 기울기가 그때 회복되면 범인은 ①이다.

실행:  .venv/Scripts/python.exe -u src/rookie7.py 2>&1 | tee experiments/rookie7.log
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


def demean(g, v):
    """그룹 평균을 뺀다. g는 0..K-1 정수 코드."""
    K = g.max() + 1
    n = np.bincount(g, minlength=K)
    m = np.bincount(g, weights=v, minlength=K) / np.maximum(n, 1)
    return v - m[g]


def slope(g, y, p):
    """그룹 평균 제거 후 원점 통과 회귀 기울기 + 투수 클러스터 부트스트랩 SE."""
    dp, dy = demean(g, p), demean(g, y)
    K = g.max() + 1
    num = np.bincount(g, weights=dp * dy, minlength=K)
    den = np.bincount(g, weights=dp * dp, minlength=K)
    return num, den, num.sum() / den.sum()


def boot_se(num, den, cl_of_grp):
    """그룹(투수×구간)을 투수 단위로 묶어 재표집한다 — 같은 투수는 함께 움직인다."""
    C = cl_of_grp.max() + 1
    nn = np.bincount(cl_of_grp, weights=num, minlength=C)
    dd = np.bincount(cl_of_grp, weights=den, minlength=C)
    out = np.empty(B)
    for b in range(B):
        i = RNG.integers(0, C, C)
        out[b] = nn[i].sum() / dd[i].sum()
    return float(np.nanstd(out, ddof=1))


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)

    for fold in FOLDS:
        va, y, p = deployed_pred(df, s, fold, cached)
        gt_f = va["game_type"].astype("object").to_numpy() == "F"
        pp = prior_seasons(df, s, fold, "pitcher_id")
        pid = va["pitcher_id"].to_numpy()
        an = va["asof_pitcher_n"].to_numpy().astype(float)

        print("\n" + "=" * 74)
        print(f"fold {fold} · 정규시즌(R)")
        for lab, m in (("신인", (~gt_f) & (pp == 0)), ("베테랑(2+시즌)", (~gt_f) & (pp >= 2))):
            yg, pg, ag = y[m], p[m], an[m]
            _, cl = np.unique(pid[m], return_inverse=True)

            # (A) 10분위 표 — 투수 평균만 제거
            dp, dy = demean(cl, pg), demean(cl, yg)
            q = np.quantile(dp, np.linspace(0, 1, 11))
            q[0], q[-1] = -np.inf, np.inf
            b = np.digitize(dp, q[1:-1])
            print(f"\n  [{lab}]  n={m.sum():,}  투수 {cl.max()+1}명")
            print(f"    {'분위':>4}{'모델이 본 편차 d_p':>20}{'실제 편차 d_y':>16}{'비율':>8}")
            for k in range(10):
                sel = b == k
                a_, c_ = dp[sel].mean(), dy[sel].mean()
                print(f"    {k+1:>4}{a_:>+20.4f}{c_:>+16.4f}{c_/a_:>8.2f}")

            # (B) asof_n 5분위까지 제거 → 시즌 내 드리프트 성분 소거
            rk = np.zeros(len(ag), dtype=np.int64)
            for j in range(cl.max() + 1):
                sel = cl == j
                if sel.sum() < 10:
                    continue
                v = ag[sel]
                rk[sel] = np.clip(np.searchsorted(np.quantile(v, [.2, .4, .6, .8]), v), 0, 4)
            g2 = cl * 5 + rk
            _, g2 = np.unique(g2, return_inverse=True)
            cl_of_g2 = np.zeros(g2.max() + 1, dtype=np.int64)
            cl_of_g2[g2] = cl

            n1, d1, s1 = slope(cl, yg, pg)
            n2, d2, s2 = slope(g2, yg, pg)
            e1 = boot_se(n1, d1, np.arange(len(n1)))
            e2 = boot_se(n2, d2, cl_of_g2)
            print(f"    투수 평균만 제거          기울기 {s1:.2f} ± {e1:.2f}"
                  f"   (변동폭 sd {np.sqrt(d1.sum()/m.sum()):.4f})")
            print(f"    + 시즌 구간 평균도 제거   기울기 {s2:.2f} ± {e2:.2f}"
                  f"   (변동폭 sd {np.sqrt(d2.sum()/m.sum()):.4f})")


if __name__ == "__main__":
    main()
