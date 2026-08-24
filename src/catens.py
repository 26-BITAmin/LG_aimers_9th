"""§7.140 — 사용자 제안: **CatBoost 도 LGBM 처럼 많이 두고 앙상블하면 되지 않나.**

배포 LGBM 은 시드 3개가 아니라 **4변형 x 3시드 = 12개**다 (ENSEMBLE_VARIANTS).
§7.139/§7.140 의 CatBoost 는 **시드 3개**뿐이었다. 하이퍼파라미터 다양성은 안 줬다.

구분해야 할 두 가지 (이게 이 실험의 전부다)
    시드 분산   같은 설정, 다른 난수 → N개 평균이 1/sqrt(N) 로 줄인다.  **앙상블이 고친다.**
    fold 발산   같은 설정이 2023 과 2024 에서 **반대로** 움직인다 → train/valid 시즌
                관계가 바뀐 것. 멤버를 늘려도 두 fold 가 같은 방향이 되지는 않는다.
                **앙상블이 못 고친다** (하이퍼파라미터 평균은 그 축 위의 보간일 뿐이고,
                 개별이 엇갈리면 조합도 엇갈린다).

관측된 것: gpu_none 블렌드Δ(w=.30) 이 fold2023 −4.4 / fold2024 +6.4 로 부호가 갈렸다.
          이게 시드 분산이면 12개로 늘리면 붙는다. fold 발산이면 안 붙는다.
**단정하지 않고 잰다.**

측정
    멤버 12개(4변형 x 3시드)를 개별 저장 → 다음을 분해한다
      · 멤버 간 solo BSS 산포 (시드 축 / 변형 축 따로)
      · 앙상블 크기 1 / 3(시드만) / 12(변형x시드) 별 블렌드 Δ, 양 fold
      · **핵심 판정: 멤버를 늘리면 fold 부호 불일치가 줄어드는가**

변형 (LGBM ENSEMBLE_VARIANTS 의 CatBoost 대응. 구조적으로 다르게 잡는다)
    base       Lossguide max_leaves=31
    wide       Lossguide max_leaves=127, min_data_in_leaf=8000
    sym        SymmetricTree depth=6      <- min_data_in_leaf 는 조용히 무시되므로 안 준다
    depthwise  Depthwise    depth=8

실행:  .venv/Scripts/python.exe -u src/catens.py 2>&1 | tee experiments/catens.log
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
from catboost import CatBoostClassifier, Pool

import deploy29 as D
from catgpu import ITERS, prep
from config import MLP_ALPHA
from validation import bss, shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEEDS = (42, 7, 2024)
WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.30)
COMMON = dict(learning_rate=0.03, l2_leaf_reg=30.0, border_count=128,
              loss_function="Logloss", verbose=0, allow_writing_files=False,
              task_type="GPU", devices="0")
VARIANTS = {
    "base":      dict(grow_policy="Lossguide", max_leaves=31, min_data_in_leaf=2000),
    "wide":      dict(grow_policy="Lossguide", max_leaves=127, min_data_in_leaf=8000),
    "sym":       dict(grow_policy="SymmetricTree", depth=6),
    "depthwise": dict(grow_policy="Depthwise", depth=8, min_data_in_leaf=2000),
}


def member(Xtr, Xva, y2, w2, init_tr, var, sd, fold, t0):
    pth = CACHE / f"catens_fold{fold}_{var}_{sd}.npz"
    if pth.exists():
        return np.load(pth)["z"]
    m = CatBoostClassifier(iterations=ITERS, random_seed=sd,
                           **COMMON, **VARIANTS[var])
    m.fit(Pool(Xtr, y2, weight=w2, baseline=init_tr))
    z = m.predict(Pool(Xva), prediction_type="RawFormulaVal")
    del m
    np.savez_compressed(pth, z=z)
    print(f"    fold {fold} [{var}/{sd}] ({time.time() - t0:.0f}s)")
    return z


def main():
    t0 = time.time()
    df, s, y = D.load_df()
    summary = {}
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        with np.load(CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
            z_mlp = d0["z_mlp"]
        z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
        mm, yv, dep = st["mm"], st["yv"], st["dep"]
        Xa = np.zeros((len(z_lgb), st["Xax"].shape[1]))
        Xa[mm] = st["Xax"]
        z_blend = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp

        def sc(zb, solo=False):
            z = zb[mm] + (0.0 if solo else Xa[mm] @ dep)
            return bss(yv, shift_to_prior(1 / (1 + np.exp(-z)), yv.mean()))

        base = sc(z_blend)
        assert abs(base - D.EXPECT_BASE[fold]) < 0.3
        print(f"\nfold {fold}  앵커 OK {base:.1f}  ·  LGBM solo {sc(z_lgb, solo=True):.1f}")
        Xtr, Xva, y2, w2, init_tr = prep(df, fold)

        Z = {}
        for var in VARIANTS:
            for sd in SEEDS:
                Z[(var, sd)] = member(Xtr, Xva, y2, w2, init_tr, var, sd, fold, t0)

        solos = {k: sc(v, solo=True) for k, v in Z.items()}
        print(f"  멤버 solo BSS ({len(solos)}개)")
        for var in VARIANTS:
            vals = [solos[(var, sd)] for sd in SEEDS]
            print(f"    {var:<10s} " + "  ".join(f"{v:7.1f}" for v in vals)
                  + f"   시드 sd {np.std(vals):5.1f}")
        vmeans = [np.mean([solos[(var, sd)] for sd in SEEDS]) for var in VARIANTS]
        print(f"    변형 평균 간 sd {np.std(vmeans):5.1f}   (시드 sd 평균 "
              f"{np.mean([np.std([solos[(v,s2)] for s2 in SEEDS]) for v in VARIANTS]):.1f})")

        ens = {
            "1개 (base/42)":   Z[("base", 42)],
            "3개 (시드만)":    np.mean([Z[("base", sd)] for sd in SEEDS], axis=0),
            "12개 (변형x시드)": np.mean(list(Z.values()), axis=0),
        }
        for nm, zc in ens.items():
            ds = [sc((1 - w) * z_blend + w * zc) - base for w in WEIGHTS]
            summary[(fold, nm)] = (sc(zc, solo=True),
                                   np.corrcoef(zc[mm], z_blend[mm])[0, 1], ds)
            print(f"  {nm:<16s} solo {sc(zc, solo=True):7.1f}"
                  f"  corr(blend) {summary[(fold, nm)][1]:.3f}"
                  f"  Δ: " + " ".join(f"{w:.2f}:{d:+6.1f}" for w, d in zip(WEIGHTS, ds)))

    print("\n" + "=" * 100)
    print("판정 — 멤버를 늘리면 fold 부호 불일치가 줄어드는가")
    print("=" * 100)
    print(f"  {'앙상블':<18s}{'w':>6s}{'Δ2023':>10s}{'Δ2024':>10s}{'부호일치':>10s}")
    for nm in ("1개 (base/42)", "3개 (시드만)", "12개 (변형x시드)"):
        for i, w in enumerate(WEIGHTS):
            d23 = summary[(2023, nm)][2][i]
            d24 = summary[(2024, nm)][2][i]
            ok = "O" if (d23 > 0) == (d24 > 0) else "X"
            mark = "  <<<" if (ok == "O" and d23 > 0 and d24 > 0) else ""
            print(f"  {nm:<18s}{w:>6.2f}{d23:>+10.1f}{d24:>+10.1f}{ok:>10s}{mark}")
    print("\n  시드 분산이면 12개에서 두 fold 가 붙는다. fold 발산이면 안 붙는다.")
    print("  채택 후보는 '양 fold 양수'가 뜨는 칸뿐이고, 거기서 시드 노이즈(4~9)를 넘어야 한다.")


if __name__ == "__main__":
    main()
