"""§7.140 후속 — 12멤버 CatBoost 블렌드 Δ 의 오차막대와 추론 시간.

catens 가 양 fold 양수 칸을 만들었다 (w=.10~.20). 그러나 크기가 작다
(f2023 +0.3~+0.8 / f2024 +3.1~+4.7). **0 과 구별되는가**가 채택을 가른다.
z 는 전부 캐시돼 있으므로 학습 0회.

  1. 투수 클러스터 부트스트랩 (복원추출 400회) — 효과가 투수 단위이므로
  2. 멤버 leave-one-out — 특정 변형 하나가 결과를 만들고 있는가
  3. 추론 시간 — 제출 한도 10분 / 245,789행. CatBoost 12개를 더 얹을 수 있는가

실행:  .venv/Scripts/python.exe -u src/catboot.py 2>&1 | tee experiments/catboot.log
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass
import numpy as np
import deploy29 as D
from catens import SEEDS, VARIANTS, WEIGHTS
from config import MLP_ALPHA
from validation import bss, shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
B = 400


def main():
    df, s, y = D.load_df()
    rng = np.random.default_rng(42)
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        with np.load(CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
            z_mlp = d0["z_mlp"]
        z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
        mm, yv, dep, cl, n_id = st["mm"], st["yv"], st["dep"], st["cl"], st["n_id"]
        Xa = np.zeros((len(z_lgb), st["Xax"].shape[1])); Xa[mm] = st["Xax"]
        z_blend = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
        ax = Xa[mm] @ dep
        ybar = yv.mean(); ref = ybar * (1 - ybar)

        def p_of(zb):
            return shift_to_prior(1 / (1 + np.exp(-(zb[mm] + ax))), ybar)

        Z = {(v, sd): np.load(CACHE / f"catens_fold{fold}_{v}_{sd}.npz")["z"]
             for v in VARIANTS for sd in SEEDS}
        ens = np.mean(list(Z.values()), axis=0)
        p0 = p_of(z_blend)
        base = bss(yv, p0)
        e0 = (p0 - yv) ** 2
        n = np.bincount(cl, minlength=n_id); s0 = np.bincount(cl, weights=e0, minlength=n_id)
        seen = np.where(n > 0)[0]
        print(f"\nfold {fold}  기준 {base:.1f}  투수 {len(seen)}명")
        print(f"  {'w':>6s}{'Δ':>9s}{'2.5%':>9s}{'97.5%':>9s}{'P(Δ>0)':>9s}")
        for w in WEIGHTS:
            p1 = p_of((1 - w) * z_blend + w * ens)
            d = bss(yv, p1) - base
            s1 = np.bincount(cl, weights=(p1 - yv) ** 2, minlength=n_id)
            draws = np.empty(B)
            for b in range(B):
                pick = rng.choice(seen, size=len(seen), replace=True)
                N = n[pick].sum()
                draws[b] = 1e5 * (s0[pick].sum() - s1[pick].sum()) / (N * ref)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            print(f"  {w:>6.2f}{d:>+9.2f}{lo:>+9.2f}{hi:>+9.2f}{np.mean(draws>0):>9.2f}")
        # 멤버 LOO — 변형 하나를 빼면 어떻게 되는가 (w=0.15 고정)
        print(f"  ── 변형 leave-one-out (w=0.15) ──")
        w = 0.15
        full = bss(yv, p_of((1 - w) * z_blend + w * ens)) - base
        print(f"    전체 4변형          Δ {full:+.2f}")
        for drop in VARIANTS:
            sub = np.mean([v for k, v in Z.items() if k[0] != drop], axis=0)
            d = bss(yv, p_of((1 - w) * z_blend + w * sub)) - base
            print(f"    −{drop:<12s} (3변형) Δ {d:+.2f}   ({d-full:+.2f})")


if __name__ == "__main__":
    main()
