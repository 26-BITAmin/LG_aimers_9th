"""catfr 완결 — **가법 분해에 오차막대**를 붙여 "F 집중"을 확정/기각한다.

병행 세션 지적: `frsplit` 의 분해와 `catfr2.sub_bss_delta` 는 **다른 물건**이다.
    가법 분해   dF = −1e5·Σdi[F]/n/bsref,  분모가 전체 것 -> **dF + dR = dTotal 항등식**
                "전체 이득 중 얼마가 F 행에서 왔나" 에 답한다. 몫 해석이 된다.
    부분군 지표  부분군의 ref=ȳ_g(1−ȳ_g) 로 재정규화 -> 가법 아님. 부호·유의성만.
내가 "가중합이 아니다"라고 경고한 것은 **내 방법**에만 해당하는데 그쪽 표에 적용했다.

가법 분해라 부트스트랩이 더 간단하다 — 같은 클러스터 뽑기 안에서 Σdi[F] 만 같이
재면 된다. 그리고 **균등 기대값과 직접 비교**할 수 있다:
    균등이면  dF_expected = dTotal x (그 뽑기의 F 행 비율)
    검정량    dF − dF_expected    구간이 0 을 배제하면 **집중이 확정**된다.
이것이 §7.145 에서 "점추정만 보고 한 말"이라고 정정한 그 진술의 결판이다.

실행:  .venv/Scripts/python.exe -u src/catfr3.py 2>&1 | tee experiments/catfr3.log
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
W, B = 0.10, 4000


def main():
    df, s, y = D.load_df()
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        with np.load(CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
            z_mlp = d0["z_mlp"]
        z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
        mm, yv, dep, cl, n_id = st["mm"], st["yv"], st["dep"], st["cl"], st["n_id"]
        Xa = np.zeros((len(z_lgb), st["Xax"].shape[1])); Xa[mm] = st["Xax"]
        ax = Xa[mm] @ dep; ybar = yv.mean(); bsref = ybar * (1 - ybar)
        isf = st["d"]["game_type"].to_numpy().astype(str) == "F"
        ens = np.mean([np.load(CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz")["z"]
                       for v in VARIANTS for sd in SEEDS], axis=0)
        pv = lambda zb: shift_to_prior(1 / (1 + np.exp(-(zb[mm] + ax))), ybar)

        z29 = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
        layers = [("s29 층", z29)]
        fs = CACHE / f"s35_fspec_fold{fold}.npz"
        if fs.exists():
            z35 = z29.copy()
            vf = (df[s == fold]["game_type"].to_numpy().astype(str) == "F")
            z35[vf] = (1 - FSPEC_ALPHA) * z29[vf] + FSPEC_ALPHA * np.load(fs)["z"][vf]
            layers.append(("s35 등가 층", z35))

        print(f"\nfold {fold}   F {isf.sum():,}/{len(yv):,} ({isf.mean():.1%})")
        for nm, zb in layers:
            p0, p1 = pv(zb), pv((1 - W) * zb + W * ens)
            di = (p1 - yv) ** 2 - (p0 - yv) ** 2          # 음수 = 개선
            # 클러스터별 부분합 (전체 / F / 행수 / F행수)
            g_all = np.bincount(cl, weights=di, minlength=n_id)
            g_f = np.bincount(cl, weights=di * isf, minlength=n_id)
            n_all = np.bincount(cl, minlength=n_id)
            n_f = np.bincount(cl, weights=isf.astype(float), minlength=n_id)
            seen = np.where(n_all > 0)[0]
            k = -1e5 / bsref
            dT = k * g_all.sum() / n_all.sum()
            dF = k * g_f.sum() / n_all.sum()
            exp_F = dT * (n_f.sum() / n_all.sum())        # 균등이면 F 의 몫
            rng = np.random.default_rng(42)
            dd = np.empty(B)
            for b in range(B):
                pk = rng.choice(seen, size=len(seen), replace=True)
                N = n_all[pk].sum()
                t = k * g_all[pk].sum() / N
                f = k * g_f[pk].sum() / N
                dd[b] = f - t * (n_f[pk].sum() / N)       # 초과분
            lo, hi = np.percentile(dd, [2.5, 97.5])
            verdict = "집중 확정" if hi < 0 or lo > 0 else "**미판정** (0 포함)"
            print(f"  ── {nm} ──")
            print(f"    dTotal {dT:+7.3f} = dF {dF:+7.3f} + dR {dT-dF:+7.3f}"
                  f"   (항등식 오차 {abs(dT-(dF+(dT-dF))):.1e})")
            print(f"    균등 기대 dF {exp_F:+7.3f}   초과분 {dF-exp_F:+7.3f}"
                  f"   95% [{lo:+.3f}, {hi:+.3f}]   -> {verdict}")
    print("\n  초과분 = dF − dTotal x (F 행 비율). 0 이면 균등 분산이다.")
    print("  이 구간이 0 을 배제해야 '이득/손해가 F 에 몰렸다'가 확정된다.")


if __name__ == "__main__":
    main()
