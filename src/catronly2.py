"""§7.149 후속 — **후보 − 위약을 같은 부트스트랩 뽑기 안에서** 쌍체로 잰다.

catronly 가 두 질문에 서로 다른 답을 냈다:
    "위약과 다른가"   z = +2.68, 위약 12개 최대를 초과      -> F 고유성 확립
    "0 과 다른가"     +0.686 [−0.254, +1.687] P=0.92        -> 미달
두 질문이 다르고, 첫째는 **쌍체로 재면 훨씬 좁아진다** (공유 노이즈가 상쇄된다).
저장소 교훈: "쌍체 비교는 클러스터 상관을 흡수한다."

같은 투수 리샘플 안에서 Δ_F 와 Δ_위약 을 **동시에** 계산해 차이의 구간을 낸다.

실행:  .venv/Scripts/python.exe -u src/catronly2.py 2>&1 | tee experiments/catronly2.log
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import numpy as np
import deploy29 as D
from catens import SEEDS, VARIANTS
from catronly import arms
from config import FSPEC_ALPHA, MLP_ALPHA
from validation import bss, shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
W, B = 0.10, 8000


def main():
    df, s, y = D.load_df()
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        with np.load(CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
            z_mlp = d0["z_mlp"]
        z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
        mm, yv, dep, cl, n_id = st["mm"], st["yv"], st["dep"], st["cl"], st["n_id"]
        Xa = np.zeros((len(z_lgb), st["Xax"].shape[1])); Xa[mm] = st["Xax"]
        ax = Xa[mm] @ dep; ybar = yv.mean(); ref = ybar * (1 - ybar)
        dd = st["d"]; isf = dd["game_type"].to_numpy().astype(str) == "F"
        ens = np.mean([np.load(CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz")["z"]
                       for v in VARIANTS for sd in SEEDS], axis=0)
        z29 = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
        fs = CACHE / f"s35_fspec_fold{fold}.npz"
        if fs.exists():
            zb = z29.copy()
            vf = (df[s == fold]["game_type"].to_numpy().astype(str) == "F")
            zb[vf] = (1 - FSPEC_ALPHA) * z29[vf] + FSPEC_ALPHA * np.load(fs)["z"][vf]
            layer = "s35 등가 층"
        else:
            zb, layer = z29, "s29 층 (fspec 없음)"
        zc_all = (1 - W) * zb + W * ens
        p_base = shift_to_prior(1 / (1 + np.exp(-(zc_all[mm] + ax))), ybar)
        n = np.bincount(cl, minlength=n_id)
        S = {"base": np.bincount(cl, weights=(p_base - yv) ** 2, minlength=n_id)}
        names = []
        for name, mask, _ in arms(dd, isf):
            z = zc_all.copy(); z[mm] = np.where(mask, zb[mm], zc_all[mm])
            p = shift_to_prior(1 / (1 + np.exp(-(z[mm] + ax))), ybar)
            S[name] = np.bincount(cl, weights=(p - yv) ** 2, minlength=n_id)
            names.append(name)
        seen = np.where(n > 0)[0]
        cand = names[0]
        pls = [x for x in names if x.startswith("위약")]
        rng = np.random.default_rng(42)
        dcand = np.empty(B); dmean = np.empty(B); dmax = np.empty(B)
        for b in range(B):
            pk = rng.choice(seen, size=len(seen), replace=True)
            N = n[pk].sum(); k = 1e5 / (N * ref)
            b0 = S["base"][pk].sum()
            dc = k * (b0 - S[cand][pk].sum())
            dp = np.array([k * (b0 - S[p][pk].sum()) for p in pls])
            dcand[b] = dc; dmean[b] = dc - dp.mean(); dmax[b] = dc - dp.max()
        print(f"\nfold {fold} · {layer}")
        for lab, arr in (("후보 − 0", dcand), ("후보 − 위약평균", dmean),
                         ("후보 − 위약최대", dmax)):
            lo, hi = np.percentile(arr, [2.5, 97.5])
            v = "0 배제 ✅" if lo > 0 or hi < 0 else "0 포함"
            print(f"  {lab:<16s}{arr.mean():>+8.3f}  [{lo:+.3f}, {hi:+.3f}]"
                  f"  P(>0)={np.mean(arr>0):.3f}  {v}")
    print("\n  '후보 − 0' 은 배포 대비 실이득, '후보 − 위약' 은 F 고유성이다.")
    print("  둘은 다른 질문이고, 채택에는 **둘 다** 필요하다.")


if __name__ == "__main__":
    main()
