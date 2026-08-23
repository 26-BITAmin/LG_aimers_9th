"""로짓 블렌드 vs **확률 블렌드** — §7.19→라운드2 의 재적용 (`MODEL.md §7.127`).

배포는 로짓공간에서 섞는다:  `z = (1−α)·z_lgb + α·z_mlp`,  α = 0.75.
확률공간에서 섞으면 `p = (1−a)·p_lgb + a·p_mlp` 이고 **p 가 a 에 선형**이라
Brier 가 a 에 대해 **정확히 2차**다 — 닫힌 해가 있다:

    a* = Σ(p_lgb − y)(p_lgb − p_mlp) / Σ(p_lgb − p_mlp)²

`§7.19` 는 **로짓공간** 스케일 `z' = c·z` 로 재서 두 fold 가 반대를 가리켜 기각했고,
같은 축을 **확률공간**에서 다시 물었더니 열렸다 (라운드 2, +3.09). 매개변수화가
바뀌면 목적함수의 모양이 바뀐다. 블렌드도 같은 구조다.

두 fold 에서 나란히 잰다. **두 fold 가 같은 방향이어야 후보다.**
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import deploy29 as D                                            # noqa: E402
from config import ASOF_AXES, CTX_AXES, MLP_ALPHA               # noqa: E402
from validation import bss, shift_to_prior                      # noqa: E402
import jointbeta as J                                           # noqa: E402


def parts(df, s, y, fold):
    """z_lgb · z_mlp · δ축 기여를 clean 행 기준으로 돌려준다."""
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    vi = np.where(s == fold)[0]
    with np.load(D.CACHE / f"dep29_fold{fold}_{D.KEY}.npz", allow_pickle=True) as z:
        Xax = z["Xax"]
    with np.load(D.CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
        z_mlp = d0["z_mlp"]
    z_lgb = np.load(D.CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
    gt = df["game_type"].to_numpy()
    mm = (~((gt == "F") & (s < 2023)))[vi]
    adj = (Xax @ dep)[mm]
    return z_lgb[mm] + adj, z_mlp[mm] + adj, y[vi][mm]


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    df, s, y = D.load_df()
    print(f"배포 블렌드: 로짓공간, α = {MLP_ALPHA}\n")
    print(f"  {'fold':>6s}{'로짓 α=0.75':>13s}{'로짓 oracle':>13s}{'확률 a=0.75':>13s}"
          f"{'확률 oracle':>13s}{'a*':>8s}{'확률−로짓':>11s}")
    out = {}
    for fold in D.FOLDS:
        zl, zm, yv = parts(df, s, y, fold)
        r = float(yv.mean()); K = 1e5 / (r * (1 - r))
        pl, pm = sig(zl), sig(zm)

        def sc(p):
            return bss(yv, shift_to_prior(np.clip(p, 1e-6, 1 - 1e-6), r))

        base = sc(sig((1 - MLP_ALPHA) * zl + MLP_ALPHA * zm))
        gridz = np.linspace(0.30, 1.00, 71)
        bz = max(sc(sig((1 - a) * zl + a * zm)) for a in gridz)
        pb = sc((1 - MLP_ALPHA) * pl + MLP_ALPHA * pm)
        d = pl - pm
        astar = float(np.sum((pl - yv) * d) / np.sum(d * d))     # 닫힌 해
        bp = sc((1 - astar) * pl + astar * pm)
        out[fold] = (base, bz, pb, bp, astar)
        print(f"  {fold:>6d}{base:>13.2f}{bz:>13.2f}{pb:>13.2f}{bp:>13.2f}"
              f"{astar:>8.3f}{bp-base:>+11.2f}")

    print()
    d23, d24 = out[2023][3] - out[2023][0], out[2024][3] - out[2024][0]
    print(f"확률 블렌드(각 fold 최적 a*) − 배포 로짓 블렌드:  "
          f"f2023 {d23:+.2f}   f2024 {d24:+.2f}")
    print(f"  a* 두 fold: {out[2023][4]:.4f} / {out[2024][4]:.4f}   "
          f"차 {abs(out[2023][4]-out[2024][4]):.4f}")
    # 전이: f2023 의 a* 를 f2024 에 적용
    zl, zm, yv = parts(df, s, y, 2024)
    r = float(yv.mean()); pl, pm = sig(zl), sig(zm)
    a23 = out[2023][4]
    p = (1 - a23) * pl + a23 * pm
    tr = bss(yv, shift_to_prior(np.clip(p, 1e-6, 1 - 1e-6), r)) - out[2024][0]
    print(f"\n  ★ 전이 (f2023 의 a*={a23:.4f} 를 f2024 에 적용): {tr:+.2f} BSS")


if __name__ == "__main__":
    main()
