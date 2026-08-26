"""연도별 지속성 — 투수간 캘리브레이션 기울기의 신인−2+ 차가 4시즌 내내 있는가.

rookdiag2 ①이 두 fold 에서 0을 배제했다. 두 점은 추세가 아니다
(memory: "홀드아웃 이득은 신호와 시즌 국한 오차를 섞는다" — 관계의 시즌 간
지속성을 따로 재라). foldcache 는 2021·2022 도 갖고 있다.

⚠️ **층이 다르다.** 2021·2022 에는 init_score·CatBoost·fspec·δ축 캐시가 없다.
   네 시즌을 같은 잣대로 보려면 전부 **z_lgb + z_mlp 블렌드**(배포 이전 층)에서
   재야 한다. 절대 BSS 는 챔피언보다 낮지만, 여기서 묻는 것은 점수가 아니라
   **기울기라는 관계**다.

실행:  .venv/Scripts/python.exe -u src/rookdiag3.py 2>&1 | tee experiments/rookdiag3.log
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
import rookdiag2 as R2
from config import MLP_ALPHA, MLP_VARIANT_WEIGHT, MLP_VARIANTS, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

SEASONS = [2021, 2022, 2023, 2024]


def light_pred(fold, df, s):
    """배포 이전 층: 0.25·z_lgb + 0.75·z_mlp(변형가중) → 오라클 prior shift."""
    y = df.loc[s == fold, TARGET_COL].to_numpy(dtype=np.float64)
    with np.load(R.CACHE / f"fold{fold}_{R.KEY}.npz") as d:
        assert np.allclose(d["y"], y)
        z_lgb, mem = d["z_lgb"], d["members"]
    i, zs = 0, {}
    for v, (_, n) in MLP_VARIANTS.items():
        zs[v] = np.mean(mem[i:i + n], axis=0)
        i += n
    z_mlp = sum(MLP_VARIANT_WEIGHT[v] * z for v, z in zs.items())
    z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
    return y, shift_to_prior(1 / (1 + np.exp(-z)), y.mean())


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    print("=" * 100)
    print("연도별 — 투수간 캘리브레이션 기울기 (배포 이전 층, 네 시즌 동일 잣대)")
    print("=" * 100)
    print(f"\n  {'시즌':<7}{'BSS':>8}{'신인수':>8}{'2+수':>7}{'신인 기울기':>13}"
          f"{'2+ 기울기':>11}{'차':>9}{'95% CI':>22}{'t':>6}")
    for fold in SEASONS:
        y, p = light_pred(fold, df, s)
        va = df[s == fold]
        pp = R.prior_seasons(df, s, fold, "pitcher_id")
        g, uniq = pd.factorize(va["pitcher_id"].to_numpy())
        n_id = len(uniq)
        rook, vet = pp == 0, pp >= 2
        s1, s2, mu, ci, t = R2.paired_slope(g[rook], y[rook], p[rook],
                                            g[vet], y[vet], p[vet], n_id)
        nk = len(np.unique(g[rook]))
        nv = len(np.unique(g[vet]))
        print(f"  {fold:<7}{bss(y, p):>8.1f}{nk:>8,}{nv:>7,}{s1:>13.3f}{s2:>11.3f}"
              f"{mu:>+9.3f}{f'[{ci[0]:+.3f}, {ci[1]:+.3f}]':>22}{t:>+6.1f}")
    print("\n  기울기 >1 = 투수 간 실력차를 압축해 예측한다는 뜻.")
    print("  차 >0 = 신인에서 그 압축이 더 심하다.")


if __name__ == "__main__":
    main()
