"""catfr 정정 — **비율 대신 F/R 각각의 ΔBSS 를 자기 오차막대와 함께** 잰다.

병행 세션이 자기 배율의 분모(ΔBSS)가 노이즈였음을 찾아 정정했다. 같은 검사를
내 것에 걸면 **내 fold2023 배율도 못 쓴다** — Δ +1.48 [−1.81, +5.00] 로 0 을 포함한다.
(저장소 교훈: "비율 조건에는 노이즈 게이트가 먼저다. 분모가 노이즈면 1.03 도 5.1 도 무의미.")

그리고 fold2024 는 전체 Δ 구간이 0 을 제외하지만, 그것이 **F/R 분해의 오차막대는
아니다** — 부분합은 자기 분산을 따로 갖는다. 비율을 아예 버리고 **부분군별 ΔBSS 를
그 부분군 안에서** 재고 각각 부트스트랩한다. 해석이 직접적이고 분모 폭발도 없다.

    Δ_F = (F 행만으로 계산한 BSS)_후보 − (F 행만)_기준
    Δ_R = 같은 것을 R 행으로

실행:  .venv/Scripts/python.exe -u src/catfr2.py 2>&1 | tee experiments/catfr2.log
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import numpy as np
import deploy29 as D
from catens import SEEDS, VARIANTS
from config import FSPEC_ALPHA, MLP_ALPHA
from validation import shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
W, B = 0.10, 400


def sub_bss_delta(yv, p0, p1, m, cl, n_id, rng):
    """부분군 m 안에서의 ΔBSS 와 투수 클러스터 부트스트랩 구간."""
    yb = yv[m]; ybar = yb.mean(); ref = ybar * (1 - ybar)
    e0 = (p0[m] - yb) ** 2; e1 = (p1[m] - yb) ** 2
    c = cl[m]
    n = np.bincount(c, minlength=n_id)
    s0 = np.bincount(c, weights=e0, minlength=n_id)
    s1 = np.bincount(c, weights=e1, minlength=n_id)
    seen = np.where(n > 0)[0]
    d = 1e5 * (s0.sum() - s1.sum()) / (n.sum() * ref)
    draws = np.empty(B)
    for b in range(B):
        pick = rng.choice(seen, size=len(seen), replace=True)
        draws[b] = 1e5 * (s0[pick].sum() - s1[pick].sum()) / (n[pick].sum() * ref)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return d, lo, hi, float(np.mean(draws > 0)), int(m.sum()), len(seen)


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
        print(f"\nfold {fold}   clean {len(yv):,} · F {isf.sum():,} ({isf.mean():.1%})")
        for nm, zb in layers:
            p0, p1 = pv(zb), pv((1 - W) * zb + W * ens)
            print(f"  ── {nm} ──")
            print(f"    {'부분군':<8s}{'행':>9s}{'투수':>6s}{'ΔBSS':>9s}"
                  f"{'2.5%':>9s}{'97.5%':>9s}{'P(Δ>0)':>9s}")
            for lab, m in (("전체", np.ones(len(yv), bool)), ("R", ~isf), ("F", isf)):
                d, lo, hi, p, nr, np_ = sub_bss_delta(yv, p0, p1, m, cl, n_id, rng)
                print(f"    {lab:<8s}{nr:>9,d}{np_:>6d}{d:>+9.2f}{lo:>+9.2f}"
                      f"{hi:>+9.2f}{p:>9.2f}")
    print("\n  주: 부분군 ΔBSS 는 그 부분군의 ref(=ȳ(1−ȳ))로 정규화한 값이라")
    print("      전체 ΔBSS 의 가중합이 아니다. 부호와 유의성만 읽는다.")


if __name__ == "__main__":
    main()
