"""CatBoost 이득의 F/R 분해 — 병행 세션의 frsplit 진단을 **채택 후보**에 적용.

동료가 기각 건에서 F 집중을 찾았다(중심화 손해의 45~56%가 F 행 10~12%에서 나온다).
같은 검사를 내가 **채택하려는** 축에 안 하는 것은 일관성이 없다.

기각 건과 채택 건의 함의가 다르다:
    기각 건  F 집중 -> s35 층에서 부호가 바뀔 수 있었음 (그러나 fold2023 미정의로 포기)
    채택 건  F 집중 -> **취약성**이다. 이득이 11.8% 행에 얹혀 있고 fspec 과 자리를 다투며,
             2025 의 F 비율이 다르면 전이가 흔들린다. 균등하면 그 걱정이 없다.

측정 (학습 0회, 캐시만): s35 등가 층에서 w=0.10 을 얹었을 때
    행별 Brier 감소분을 game_type 으로 쪼갠다.
    배율 = (F 가 가져가는 이득 비율) / (F 노출 비율).  1 이면 균등.
대조: fold2023 은 s29 층(fspec 없음)이라 같은 배율을 s29 층에서도 찍어 비교한다.

실행:  .venv/Scripts/python.exe -u src/catfr.py 2>&1 | tee experiments/catfr.log
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import numpy as np
import deploy29 as D
from catens import SEEDS, VARIANTS
from config import FSPEC_ALPHA, MLP_ALPHA
from validation import bss, shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
W = 0.10


def decomp(yv, p0, p1, isf, label):
    """행별 Brier 감소분을 F/R 로 쪼갠다."""
    d = (p0 - yv) ** 2 - (p1 - yv) ** 2          # 양수 = 개선
    tot = d.sum()
    fshare = d[isf].sum() / tot if tot != 0 else np.nan
    expo = isf.mean()
    print(f"    {label:<16s} 전체이득 {tot:+.3f}  F기여율 {100*fshare:5.1f}%"
          f"  F노출율 {100*expo:5.1f}%  **배율 {fshare/expo:+.2f}**")
    return fshare / expo


def main():
    df, s, y = D.load_df()
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        with np.load(CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
            z_mlp = d0["z_mlp"]
        z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
        mm, yv, dep = st["mm"], st["yv"], st["dep"]
        Xa = np.zeros((len(z_lgb), st["Xax"].shape[1])); Xa[mm] = st["Xax"]
        ax = Xa[mm] @ dep; ybar = yv.mean()
        isf = st["d"]["game_type"].to_numpy().astype(str) == "F"
        ens = np.mean([np.load(CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz")["z"]
                       for v in VARIANTS for sd in SEEDS], axis=0)

        def pv(zb):
            return shift_to_prior(1 / (1 + np.exp(-(zb[mm] + ax))), ybar)

        z29 = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
        layers = [("s29 층", z29)]
        fs = CACHE / f"s35_fspec_fold{fold}.npz"
        if fs.exists():
            z35 = z29.copy()
            vf = (df[s == fold]["game_type"].to_numpy().astype(str) == "F")
            z35[vf] = (1 - FSPEC_ALPHA) * z29[vf] + FSPEC_ALPHA * np.load(fs)["z"][vf]
            layers.append(("s35 등가 층", z35))
        print(f"\nfold {fold}   clean {len(yv):,}행 · F {isf.sum():,} ({isf.mean():.1%})")
        for nm, zb in layers:
            p0, p1 = pv(zb), pv((1 - W) * zb + W * ens)
            print(f"  ── {nm}  기준 {bss(yv,p0):.1f} -> {bss(yv,p1):.1f} "
                  f"({bss(yv,p1)-bss(yv,p0):+.2f}) ──")
            decomp(yv, p0, p1, isf, f"CatBoost w={W}")
    print("\n  배율 1 근처 = 균등 분산 (전이 걱정 없음)")
    print("  배율 >> 1  = F 행에 얹힌 이득 → fspec 과 경합 · 2025 F 비율에 민감 → 취약")


if __name__ == "__main__":
    main()
