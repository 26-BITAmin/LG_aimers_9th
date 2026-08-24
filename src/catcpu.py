"""§7.139 후속2 — 대조군이 안 죽은 것이 **fold 탓인가 백엔드 탓인가**.

catinit 이 init_score 가설을 기각했다. allid-none 격차는 fold2023 -34, fold2024 +15~+29
로 **fold 가 부호를 가른다**. 그런데 §7.12 의 -101.4 는 **fold2024 에서** 측정됐다
(ablate15 주석: "§7.12 2x2 결과 (valid=2024, lgbm 741.9 기준)"). 같은 fold, 반대 부호다.

의심: CPU -> GPU. CatBoost GPU 는 CTR 기본 설정이 CPU 와 다르고 rsm 도 뺐다.
      즉 내 대조군이 §7.12 와 **같은 조작이 아닐 수** 있다.

fold2024 에서 CPU(§7.12 설정: rsm=0.8) 로 none vs allid 를 다시 잰다. 2시드.
  · CPU 에서 allid-none 이 **크게 음수**면 -> 백엔드가 원인. GPU 결과는 선행 연구와
    비교 불가이고, GPU 로 재려면 눈금을 GPU 안에서 새로 만들어야 한다.
  · CPU 에서도 양수면 -> 백엔드가 아니라 파이프라인 변경(soften/contam/피처)이 원인.
    그러면 §7.12 의 결론이 현 파이프라인에서 뒤집힌 것이고, 그 자체가 발견이다.

실행:  .venv/Scripts/python.exe -u src/catcpu.py 2>&1 | tee experiments/catcpu.log
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass
import numpy as np
from catboost import CatBoostClassifier, Pool
import deploy29 as D
from catgpu import ITERS, LOWCARD, prep
from config import MLP_ALPHA
from validation import bss, shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
ALLID = LOWCARD + ["pitcher_id", "batter_id"]
# §7.12 CPU 설정 그대로 — rsm=0.8 포함 (GPU 에서 못 쓰던 그것)
CPU_PARAMS = dict(learning_rate=0.03, l2_leaf_reg=30.0, border_count=128, rsm=0.8,
                  grow_policy="Lossguide", max_leaves=31, min_data_in_leaf=2000,
                  loss_function="Logloss", verbose=0, allow_writing_files=False,
                  task_type="CPU", thread_count=6)
FOLD = 2024


def main():
    t0 = time.time()
    df, s, y = D.load_df()
    st = D.state(df, s, y, FOLD)
    with np.load(CACHE / f"fold{FOLD}_{D.KEY}.npz") as d0:
        z_mlp = d0["z_mlp"]
    z_lgb = np.load(CACHE / f"initdep_fold{FOLD}_ctr0.npz")["z"]
    mm, yv, dep = st["mm"], st["yv"], st["dep"]
    Xa = np.zeros((len(z_lgb), st["Xax"].shape[1])); Xa[mm] = st["Xax"]
    z_blend = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp

    def sc(zb, solo=False):
        z = zb[mm] + (0.0 if solo else Xa[mm] @ dep)
        return bss(yv, shift_to_prior(1 / (1 + np.exp(-z)), yv.mean()))

    base, solo_lgb = sc(z_blend), sc(z_lgb, solo=True)
    assert abs(base - D.EXPECT_BASE[FOLD]) < 0.3
    print(f"fold {FOLD} 앵커 OK {base:.1f} · LGBM solo {solo_lgb:.1f}")
    Xtr, Xva, y2, w2, init_tr = prep(df, FOLD)
    print(f"X {Xtr.shape} 준비 ({time.time()-t0:.0f}s)  — CPU, rsm=0.8, 2시드\n")

    out = {}
    for tag, cats in (("none", []), ("allid", ALLID)):
        Xa_tr, Xa_va = Xtr.copy(), Xva.copy()
        idx = None
        if cats:
            for c in cats:
                Xa_tr[c] = Xa_tr[c].to_numpy(np.float64).round().astype(np.int64).astype(str)
                Xa_va[c] = Xa_va[c].to_numpy(np.float64).round().astype(np.int64).astype(str)
            idx = [Xa_tr.columns.get_loc(c) for c in cats]
        z, n = np.zeros(len(Xa_va)), 0
        for sd in (42, 7):
            m = CatBoostClassifier(iterations=ITERS, random_seed=sd, **CPU_PARAMS)
            m.fit(Pool(Xa_tr, y2, weight=w2, baseline=init_tr, cat_features=idx))
            z += m.predict(Pool(Xa_va, cat_features=idx), prediction_type="RawFormulaVal")
            n += 1; del m
        z /= n
        sol = sc(z, solo=True)
        out[tag] = sol
        print(f"  CPU [{tag}] solo {sol:8.1f}  핸디캡 {sol-solo_lgb:+8.1f}"
              f"  블렌드Δ(w=.30) {sc(0.7*z_blend+0.3*z)-base:+7.1f}  ({time.time()-t0:.0f}s)")

    d = out["allid"] - out["none"]
    print(f"\n  CPU  allid − none = {d:+.1f}")
    print(f"  GPU  allid − none = +14.9  (catinit, 같은 fold·같은 init)")
    print(f"\n  판정: CPU 가 크게 음수면 백엔드가 원인 → GPU 결과는 선행 연구와 비교 불가.")
    print(f"        CPU 도 양수면 파이프라인 변경이 원인 → §7.12 결론이 뒤집힌 것.")


if __name__ == "__main__":
    main()
