"""§7.78 후속 2 — rookie3의 투수간/투수내 분해를 **공정 비교**로 바꾼다.

rookie3 결과
    로컬BSS = 투수간 + 투수내
    2023 R 신인   404 =  417 + (−12)      2024 R 신인   919 = 909 + 10
    2023 R 2+     815 =  735 +  80        2024 R 2+     839 = 626 + 213

원시 투수간 값은 **비교 불가**다. 셀마다 투수들의 진짜 실력 분산이 다르면
"잘 골랐나"와 "고를 게 많았나"가 섞인다 ([[raw-transfer-confounds-level-with-structure]]).
분모를 만든다:
    Var_true(θ_j) = Var_obs(ȳ_j) − E[ȳ_j(1−ȳ_j)/n_j]     (이항 노이즈 제거)
    투수간 상한   = 1e5 · Var_true / ref                   (각 투수의 진짜 θ를 알 때)
    포착률        = 투수간 BSS / 상한
투수내는 클러스터 부트스트랩 SE를 붙인다 — 이게 유일하게 fold 간 부호가
일관된 항이라 진짜인지부터 확인해야 한다.

실행:  .venv/Scripts/python.exe -u src/rookie4.py 2>&1 | tee experiments/rookie4.log
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
MINN = 30          # 투수간 통계는 표본 30구 미만 투수를 빼고 본다 (노이즈 지배)


def agg(cl, ncl, y, p):
    n = np.bincount(cl, minlength=ncl).astype(float)
    ys = np.bincount(cl, weights=y, minlength=ncl)
    ps = np.bincount(cl, weights=p, minlength=ncl)
    sq = np.bincount(cl, weights=(p - y) ** 2, minlength=ncl)
    return n, ys, ps, sq


def parts(n, ys, ps, sq):
    """(로컬BSS, 투수간, 투수내). 부트스트랩에서 재호출되므로 순수 함수로 둔다."""
    N = n.sum()
    yb = ys.sum() / N
    ref = yb * (1 - yb)
    full = 1e5 * (1 - (sq.sum() / N) / ref)
    # 투수간: 예측을 p̄_j로 대체했을 때. Σ_j n_j·E[(p̄_j − y)²]
    #   = Σ_j [ n_j·p̄_j² − 2·p̄_j·ys_j + ys_j ]   (y∈{0,1} → y²=y)
    ok = n > 0
    pbar = np.divide(ps, n, out=np.zeros_like(ps), where=ok)
    bs_btw = (n * pbar ** 2 - 2 * pbar * ys + ys).sum() / N
    btw = 1e5 * (1 - bs_btw / ref)
    return full, btw, full - btw


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)

    print(f"{'fold/셀':<16}{'투수':>5}{'행/투수':>8}{'sd(ȳ_j)':>9}{'sd(θ)':>8}"
          f"{'sd(p̄_j)':>9}{'corr':>7}{'투수간':>9}{'상한':>9}{'포착률':>8}"
          f"{'투수내':>9}{'±SE':>7}")
    for fold in FOLDS:
        va, y, p = deployed_pred(df, s, fold, cached)
        gt_f = va["game_type"].astype("object").to_numpy() == "F"
        pp = prior_seasons(df, s, fold, "pitcher_id")
        pid = va["pitcher_id"].to_numpy()

        for lab, m in (("R 신인", (~gt_f) & (pp == 0)), ("R 2+시즌", (~gt_f) & (pp >= 2)),
                       ("F 신인", gt_f & (pp == 0)), ("F 2+시즌", gt_f & (pp >= 2))):
            yg, pg = y[m], p[m]
            _, cl = np.unique(pid[m], return_inverse=True)
            ncl = cl.max() + 1
            n, ys, ps, sq = agg(cl, ncl, yg, pg)
            full, btw, wit = parts(n, ys, ps, sq)

            # 투수간 상한 — 표본 30구 이상 투수만, 행 수 가중
            k = n >= MINN
            yj, pj, nj = ys[k] / n[k], ps[k] / n[k], n[k]
            w = nj / nj.sum()
            mu = (w * yj).sum()
            var_obs = (w * (yj - mu) ** 2).sum()
            var_noise = (w * yj * (1 - yj) / nj).sum()
            var_true = max(var_obs - var_noise, 1e-9)
            yb = ys.sum() / n.sum()
            ref = yb * (1 - yb)
            cap = 1e5 * var_true / ref
            mp = (w * pj).sum()
            cov = (w * (yj - mu) * (pj - mp)).sum()
            sdp = np.sqrt((w * (pj - mp) ** 2).sum())
            corr = cov / (np.sqrt(var_obs) * sdp + 1e-12)

            wb = np.empty(B)
            for b in range(B):
                i = RNG.integers(0, ncl, ncl)
                wb[b] = parts(n[i], ys[i], ps[i], sq[i])[2]

            print(f"{str(fold)+' '+lab:<16}{ncl:>5}{n.mean():>8.0f}"
                  f"{np.sqrt(var_obs):>9.4f}{np.sqrt(var_true):>8.4f}{sdp:>9.4f}"
                  f"{corr:>7.2f}{btw:>9.1f}{cap:>9.1f}{btw/cap*100:>7.0f}%"
                  f"{wit:>9.1f}{np.nanstd(wb, ddof=1):>7.1f}")


if __name__ == "__main__":
    main()
