"""시드 수 → BSS 곡선을 **fold 라벨로 직접** 잰다 (`MODEL.md §7.128`).

`src/seednoise.py` 는 라벨 없이 `이득 = K·Var(앙상블 평균)` 로 +4.83 을 계산했다.
그런데 §7.9 는 `MLP 3→8` 에서 **홀드아웃 +4.1 → LB +0.24 (배율 0.06)** 를 기록했다.
계산값이 홀드아웃 쪽 자릿수라 **검증 없이 믿으면 안 된다.**

`fold2024_{KEY}.npz` 에 개별 MLP 멤버 8개의 로짓이 라벨과 함께 있다.
부분집합을 뽑아 `BSS(k)` 를 직접 재고 `BSS(k) = BSS_∞ − C/k` 를 적합한다.
1/k 법칙이 맞으면 계산이 옳고, 안 맞으면 어디서 깨지는지가 드러난다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import deploy29 as D                                            # noqa: E402
from config import ASOF_AXES, CTX_AXES, MLP_ALPHA               # noqa: E402
from validation import bss, shift_to_prior                      # noqa: E402


def main(fold=2024, draws=60, seed=0):
    df, s, y = D.load_df()
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    vi = np.where(s == fold)[0]
    with np.load(D.CACHE / f"dep29_fold{fold}_{D.KEY}.npz", allow_pickle=True) as z:
        Xax = z["Xax"]
    with np.load(D.CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
        M, yv_all = d0["members"], d0["y"]
    z_lgb = np.load(D.CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
    gt = df["game_type"].to_numpy()
    mm = (~((gt == "F") & (s < 2023)))[vi]
    adj = (Xax @ dep)[mm]
    M = M[:, mm]; z_lgb = z_lgb[mm]; yv = yv_all[mm]
    r = float(yv.mean()); K = 1e5 / (r * (1 - r))
    nm = M.shape[0]

    def score(zm):
        z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * zm + adj
        p = 1 / (1 + np.exp(-z))
        return bss(yv, shift_to_prior(np.clip(p, 1e-6, 1 - 1e-6), r))

    rng = np.random.default_rng(seed)
    print(f"fold {fold}  n={len(yv):,}  MLP 멤버 {nm}개  α={MLP_ALPHA}  K={K:,.0f}")
    print(f"  {'k':>3s}{'BSS 평균':>12s}{'sd':>9s}{'1/k':>9s}   (부분집합 {draws}회)")
    ks, ms = [], []
    for k in range(1, nm + 1):
        vals = []
        n_draw = 1 if k == nm else draws
        for _ in range(n_draw):
            idx = rng.choice(nm, k, replace=False)
            vals.append(score(M[idx].mean(axis=0)))
        v = np.array(vals)
        ks.append(k); ms.append(v.mean())
        print(f"  {k:>3d}{v.mean():>12.2f}{v.std():>9.2f}{1/k:>9.4f}")

    ks, ms = np.array(ks, float), np.array(ms)
    A = np.stack([np.ones_like(ks), -1 / ks], 1)
    inf_, C = np.linalg.lstsq(A, ms, rcond=None)[0]
    pred = A @ np.array([inf_, C])
    print(f"\n  적합  BSS(k) = {inf_:.2f} − {C:.2f}/k     "
          f"최대잔차 {np.abs(pred-ms).max():.3f}")
    print(f"  BSS(8)  실측 {ms[-1]:.2f}   적합 {inf_-C/8:.2f}")
    print(f"  ★ k=8 -> ∞ 남은 헤드룸 = {C/8:+.2f} BSS")
    print(f"    k=8 -> 16          = {C/8-C/16:+.2f} BSS")
    print(f"    k=8 -> 24          = {C/8-C/24:+.2f} BSS")
    print(f"\n  [대조] seednoise.py 의 라벨없는 계산: k=8 -> ∞ 가 +4.83, k=8->16 이 +2.42")


if __name__ == "__main__":
    main()
