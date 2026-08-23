"""**멤버를 2배로 늘렸을 때의 draw 분포** — §7.128 이 놓친 것 (`MODEL.md §7.130`).

§7.128 은 `BSS(k) = 927.42 − 25.26/k` 의 **평균 곡선**을 읽고 `k=8→16` 을 `+1.58`
로 예측했다. 실측은 `s32 = 1104.9203080936` 으로 s30 대비 **−2.30** 이다.

`E[BS(2k)] < E[BS(k)]` 는 여전히 참이다. 틀린 것은 **"부호가 보장된다"** 는 문장이다
— 보장되는 것은 **기댓값의 부호**고, 한 번의 실현은 draw 분산을 탄다.
그리고 **draw 는 245,789행이 아니라 시드 8개다.** 행 평균이 이 분산을 안 줄인다.

fold 멤버 8개를 겹치지 않는 두 4-집합으로 갈라 `BSS(A)` vs `BSS(A∪B)` 를 재면
**"멤버 2배"의 draw 분포**를 직접 얻는다 (35가지 분할 전부).
"""
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import deploy29 as D                                            # noqa: E402
from config import ASOF_AXES, CTX_AXES, MLP_ALPHA               # noqa: E402
from validation import bss, shift_to_prior                      # noqa: E402


def main(fold=2024):
    df, s, y = D.load_df()
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    vi = np.where(s == fold)[0]
    with np.load(D.CACHE / f"dep29_fold{fold}_{D.KEY}.npz", allow_pickle=True) as z:
        Xax = z["Xax"]
    with np.load(D.CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
        M, ya = d0["members"], d0["y"]
    z_lgb = np.load(D.CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
    gt = df["game_type"].to_numpy()
    mm = (~((gt == "F") & (s < 2023)))[vi]
    adj = (Xax @ dep)[mm]
    M, z_lgb, yv = M[:, mm], z_lgb[mm], ya[mm]
    r = float(yv.mean())

    def sc(idx):
        z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * M[list(idx)].mean(axis=0) + adj
        p = 1 / (1 + np.exp(-z))
        return bss(yv, shift_to_prior(np.clip(p, 1e-6, 1 - 1e-6), r))

    all8 = tuple(range(8))
    gains = []
    for A in combinations(all8, 4):
        gains.append(sc(all8) - sc(A))          # 4 -> 8 (A 에 나머지 4개를 더한다)
    g = np.array(gains)
    print(f"fold {fold}   '멤버 4 → 8' 의 draw 분포  (35가지 분할 전부)")
    print(f"  평균 {g.mean():+.2f}   sd {g.std(ddof=1):.2f}   "
          f"최소 {g.min():+.2f}   최대 {g.max():+.2f}")
    print(f"  음수인 분할: {(g < 0).sum()}/{len(g)}  ({(g<0).mean():.0%})")
    print(f"  이론 평균 (25.26/4 − 25.26/8) = {25.26/4 - 25.26/8:+.2f}")

    # 8 -> 16 으로 스케일: 평균은 1/2, sd 는 대략 1/√2 (독립 draw 가 하나 더 붙는다)
    print(f"\n  8 → 16 으로 외삽:  평균 {g.mean()/2:+.2f}   sd 약 {g.std(ddof=1)/np.sqrt(2):.2f}")
    obs = 1104.9203080936 - 1107.2194255322
    z_ = (obs - g.mean() / 2) / (g.std(ddof=1) / np.sqrt(2))
    print(f"  실측 s32−s30 = {obs:+.2f}   ->   z = {z_:+.2f}σ")
    print(f"  P(단일 실현이 음수) ≈ {(g < 0).mean():.0%} (4→8 기준, 8→16 은 더 높다)")


if __name__ == "__main__":
    main()
