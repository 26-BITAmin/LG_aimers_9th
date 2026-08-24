"""§7.141 — §7.140 의 12멤버 CatBoost 앙상블을 **CPU 로 재학습·재검증**한다.

이유는 성능이 아니라 **재현성**이다. CatBoost GPU 는 하드웨어 의존이라 코드 검증
(9/11)에서 심사 측이 다른 장비로 재학습하면 다른 모델이 된다 (CPU-GPU 대조
corr 0.9957 — §7.139 catcpu 에서 확인). Private Score 재현을 요구하는 대회다.

**순수 백엔드 교체다.** 변형 정의·시드·iteration 전부 §7.140 과 동일하고
`task_type` 만 CPU 다. `rsm=0.8`(§7.12 튜닝값)은 **넣지 않는다** — 넣으면 백엔드와
rsm 이 동시에 바뀌어 "GPU 결과가 CPU 에서 살아남는가"를 못 본다.
`thread_count=6` 은 ablate15 에 기록된 값 그대로 — 제출 학습 코드가 검증한 것과
비트 단위로 같아야 한다.

채택 기준은 §7.140 과 **동일하게 미리 박는다** (결과를 보고 완화하지 않는다)
   (1) 양 fold 양수인 w 가 존재
   (2) fold2024 부트스트랩 구간이 0 을 배제
   (3) fold2023 구간이 0 을 유의하게 밑돌지 않을 것
   (4) 추론 10분 한도 — 이미 통과 (12개 4.3s)
   w 는 **0.10 으로 사전 고정**한다. 고원(0.05~0.20)의 낮은 쪽이고, fold 결과를
   보고 최대점을 고르면 배포 규약대로 부풀려진다.

사전 예측: GPU 와 corr 0.9957 이므로 Δ 도 비슷할 것이다 (f2023 +0.8 / f2024 +3.1
부근). 크게 다르면 §7.140 의 결론이 백엔드 특이적이었다는 뜻이고, 그러면 기각한다.

실행:  .venv/Scripts/python.exe -u src/catcpuens.py 2>&1 | tee experiments/catcpuens.log
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
from catens import SEEDS, VARIANTS, WEIGHTS
from catgpu import ITERS, prep
from config import MLP_ALPHA
from validation import bss, shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
B = 400
W_FIXED = 0.10          # 사전 고정. fold 를 보고 고르지 않는다
CPU_COMMON = dict(learning_rate=0.03, l2_leaf_reg=30.0, border_count=128,
                  loss_function="Logloss", verbose=0, allow_writing_files=False,
                  task_type="CPU", thread_count=6)


def member(Xtr, Xva, y2, w2, init_tr, var, sd, fold, t0):
    pth = CACHE / f"catcpu_fold{fold}_{var}_{sd}.npz"
    if pth.exists():
        return np.load(pth)["z"]
    m = CatBoostClassifier(iterations=ITERS, random_seed=sd,
                           **CPU_COMMON, **VARIANTS[var])
    m.fit(Pool(Xtr, y2, weight=w2, baseline=init_tr))
    z = m.predict(Pool(Xva), prediction_type="RawFormulaVal")
    del m
    np.savez_compressed(pth, z=z)
    print(f"    fold {fold} [{var}/{sd}] ({time.time() - t0:.0f}s)")
    return z


def main():
    t0 = time.time()
    df, s, y = D.load_df()
    rng = np.random.default_rng(42)
    summary = {}
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        with np.load(CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
            z_mlp = d0["z_mlp"]
        z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
        mm, yv, dep, cl, n_id = st["mm"], st["yv"], st["dep"], st["cl"], st["n_id"]
        Xa = np.zeros((len(z_lgb), st["Xax"].shape[1]))
        Xa[mm] = st["Xax"]
        z_blend = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
        ax = Xa[mm] @ dep
        ybar = yv.mean()
        ref = ybar * (1 - ybar)

        def p_of(zb):
            return shift_to_prior(1 / (1 + np.exp(-(zb[mm] + ax))), ybar)

        def solo(zb):
            return bss(yv, shift_to_prior(1 / (1 + np.exp(-zb[mm])), ybar))

        base = bss(yv, p_of(z_blend))
        assert abs(base - D.EXPECT_BASE[fold]) < 0.3, f"앵커 실패 {base:.1f}"
        print(f"\nfold {fold}  앵커 OK {base:.1f}  ·  LGBM solo {solo(z_lgb):.1f}")
        Xtr, Xva, y2, w2, init_tr = prep(df, fold)
        print(f"    X {Xtr.shape}  CPU 6스레드 · 12멤버 학습 시작 ({time.time()-t0:.0f}s)")

        Z = {(v, sd): member(Xtr, Xva, y2, w2, init_tr, v, sd, fold, t0)
             for v in VARIANTS for sd in SEEDS}
        ens = np.mean(list(Z.values()), axis=0)

        # GPU 멤버와의 대조 (같은 fold·같은 변형·같은 시드)
        gz = {k: np.load(CACHE / f"catens_fold{fold}_{k[0]}_{k[1]}.npz")["z"]
              for k in Z if (CACHE / f"catens_fold{fold}_{k[0]}_{k[1]}.npz").exists()}
        if gz:
            gens = np.mean(list(gz.values()), axis=0)
            print(f"    corr(CPU앙상블, GPU앙상블) = {np.corrcoef(ens, gens)[0,1]:.5f}"
                  f"   GPU solo {solo(gens):.1f}")

        print(f"    멤버 solo BSS")
        for v in VARIANTS:
            vals = [solo(Z[(v, sd)]) for sd in SEEDS]
            print(f"      {v:<10s} " + "  ".join(f"{x:7.1f}" for x in vals)
                  + f"   시드 sd {np.std(vals):5.1f}")
        s_ens = solo(ens)
        print(f"    12멤버 앙상블 solo {s_ens:.1f}  (핸디캡 {s_ens - solo(z_lgb):+.1f})"
              f"  corr(blend) {np.corrcoef(ens[mm], z_blend[mm])[0,1]:.3f}")

        p0 = p_of(z_blend)
        e0 = (p0 - yv) ** 2
        n = np.bincount(cl, minlength=n_id)
        s0 = np.bincount(cl, weights=e0, minlength=n_id)
        seen = np.where(n > 0)[0]
        print(f"    {'w':>6s}{'Δ':>9s}{'2.5%':>9s}{'97.5%':>9s}{'P(Δ>0)':>9s}")
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
            summary[(fold, w)] = (d, lo, hi, float(np.mean(draws > 0)))
            mark = "  <<< 사전 고정 w" if abs(w - W_FIXED) < 1e-9 else ""
            print(f"    {w:>6.2f}{d:>+9.2f}{lo:>+9.2f}{hi:>+9.2f}"
                  f"{np.mean(draws>0):>9.2f}{mark}")

    print("\n" + "=" * 92)
    print(f"판정 — CPU 12멤버, 사전 고정 w={W_FIXED}")
    print("=" * 92)
    print(f"  {'w':>6s}{'Δ2023':>10s}{'[95% 구간]':>22s}{'Δ2024':>10s}{'[95% 구간]':>22s}{'부호':>6s}")
    for w in WEIGHTS:
        d23, l23, h23, _ = summary[(2023, w)]
        d24, l24, h24, _ = summary[(2024, w)]
        ok = "O" if (d23 > 0 and d24 > 0) else "X"
        print(f"  {w:>6.2f}{d23:>+10.2f}{f'[{l23:+.2f}, {h23:+.2f}]':>22s}"
              f"{d24:>+10.2f}{f'[{l24:+.2f}, {h24:+.2f}]':>22s}{ok:>6s}")
    d23 = summary[(2023, W_FIXED)]
    d24 = summary[(2024, W_FIXED)]
    print(f"\n  사전 고정 w={W_FIXED}:  f2023 {d23[0]:+.2f} [{d23[1]:+.2f}, {d23[2]:+.2f}]"
          f"   f2024 {d24[0]:+.2f} [{d24[1]:+.2f}, {d24[2]:+.2f}]")
    print("  기준 (§7.140 과 동일, 미리 박음):")
    print("    (1) 양 fold 양수  (2) f2024 구간이 0 배제  (3) f2023 구간이 0 을 유의하게 밑돌지 않음")
    print(f"  GPU 참조: f2023 +0.80 [-2.97, +4.55]   f2024 +3.10 [+0.47, +5.64]")


if __name__ == "__main__":
    main()
