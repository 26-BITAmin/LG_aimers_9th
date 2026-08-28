"""§7.190d — 마지막 미측정 채널: **CatBoost 가 mf_ 컬럼을 받아 재학습**되면 다른가

배경 (사용자: "LGBM 말고 CatBoost 도 따라 나오는데 그 부분은 어떻게 했지 → 해보자")
    §7.190c 는 LGBM 채널만 재학습했다. 배포에서는 CatBoost(w=0.10)도 같은 X 를
    받으므로 이 채널이 미측정으로 남아 있었다. 학습기 특이 반응 전례
    ([[augmentation-targets-learners-not-just-rows]] — 같은 조작 LGBM +8.5 / MLP −12.1)
    가 있으므로 산수("0.10 가중 · corr .958 채널로는 구제 불가")로 닫지 않고 실측한다.

설계 — 배포 CatBoost fold 레시피(§7.141 catcpuens) 를 비트 단위로 따른다
    데이터   catgpu.prep (build_features + soften + init_score(level_init))
    멤버     CPU 4변형(base/wide/sym/depthwise) × 3시드(42/7/2024) × 250 iter,
             thread_count 6, baseline=init_tr — catcpu_fold* 캐시와 같은 코드 경로
    MASS     + mf_ 3컬럼 (massfeat3.add_cols 재사용 — train 적합 z 상수 차분)
    PLB      + b쪽 행 순열 컬럼 (같은 함수, arm="PLB")

평가 (사전 등록) — **배포층**에서 잰다 ([[run-diagnostics-on-the-deployed-layer]])
    p_dep(ens) = shift(affine(shift(σ( (1−w)·z_blend + w·ens + Xax@β ))))
    앵커: 캐시 ens0 (배포 12멤버) 로 f2023 812.6 / f2024 929.9 재현 (±0.15, 즉사 assert)
    판정: Δ(MASS−PLB) at **사전 고정 w=0.10**, 투수 클러스터 쌍체 부트스트랩 B=2000,
          채택 후보 = 양 fold 양수 & P≥0.9. w 격자 {.05,.10,.15,.20,.30} 는 참고
          (재평형 반문 커버). CatBoost 단독(solo) Δ 와 mf_ 중요도는 진단 병기.
    미달 → CatBoost 채널 폐쇄, §7.190 완결 (프로브·층분해·LGBM 학습·CatBoost 학습 4중).

실행:  .venv/Scripts/python.exe -u src/massfeat4.py 2>&1 | tee experiments/massfeat4.log
"""
import gc
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
from catens import SEEDS, VARIANTS
from catgpu import ITERS, prep
from config import MLP_ALPHA
from massfeat3 import FAMILY, add_cols
from predict import PROB_CENTER, PROB_SCALE
from validation import bss, shift_to_prior

CACHE = D.CACHE
B_BOOT = 2000
W_FIXED = 0.10
W_GRID = (0.05, 0.10, 0.15, 0.20, 0.30)
EXPECT_DEP = {2023: 812.6, 2024: 929.9}      # resprobe 실측 — 배포층 앵커
CPU_COMMON = dict(learning_rate=0.03, l2_leaf_reg=30.0, border_count=128,
                  loss_function="Logloss", verbose=0, allow_writing_files=False,
                  task_type="CPU", thread_count=6)
MF = [nm for nm, _a, _b in FAMILY]


def member(Xtr, Xva, y2, w2, init_tr, arm, var, sd, fold, t0):
    pth = CACHE / f"massfeat4_fold{fold}_{arm}_{var}_{sd}.npz"
    if pth.exists():
        with np.load(pth, allow_pickle=True) as z:
            return z["z"], z["imp"].item()
    m = CatBoostClassifier(iterations=ITERS, random_seed=sd,
                           **CPU_COMMON, **VARIANTS[var])
    m.fit(Pool(Xtr, y2, weight=w2, baseline=init_tr))
    z = m.predict(Pool(Xva), prediction_type="RawFormulaVal")
    try:
        fi = dict(zip(m.feature_names_, m.get_feature_importance()))
        imp = {nm: float(fi.get(nm, 0.0)) for nm in MF}
    except Exception:
        imp = {}
    del m
    np.savez_compressed(pth, z=z, imp=np.array(imp, dtype=object))
    print(f"    fold {fold} [{arm} {var}/{sd}] ({time.time() - t0:.0f}s)")
    return z, imp


def main():
    t0 = time.time()
    df, s, y = D.load_df()
    rng_boot = np.random.default_rng(42)
    verdict, table = {}, []
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        with np.load(CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
            z_mlp = d0["z_mlp"]
        z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
        mm, yv, cl, n_id = st["mm"], st["yv"], st["cl"], st["n_id"]
        ax = st["Xax"] @ st["dep"]
        zb = ((1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp)[mm]
        ybar = yv.mean()
        ref = ybar * (1 - ybar)

        def dep_p(ens, w=W_FIXED):
            zc = (1 - w) * zb + w * ens + ax
            pb = shift_to_prior(1 / (1 + np.exp(-zc)), ybar)
            return shift_to_prior(PROB_CENTER + PROB_SCALE * (pb - PROB_CENTER), ybar)

        def solo(ens):
            return bss(yv, shift_to_prior(1 / (1 + np.exp(-ens)), ybar))

        ens0 = np.mean([np.load(CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz")["z"]
                        for v in VARIANTS for sd in SEEDS], axis=0)[mm]
        base = bss(yv, dep_p(ens0))
        assert abs(base - EXPECT_DEP[fold]) < 0.15, f"앵커 실패 {base:.2f}"
        print(f"\nfold {fold}  배포층 앵커 OK {base:.1f}  ·  캐시 cat solo {solo(ens0):.1f}")

        Xtr, Xva, y2, w2, init_tr = prep(df, fold)
        print(f"    X {Xtr.shape} 준비 ({time.time()-t0:.0f}s)")
        ENS, IMP = {}, {}
        for arm in ("MASS", "PLB"):
            Xt, Xv = add_cols(Xtr, Xva, arm, np.random.default_rng(3000 + fold))
            zs, imps = [], []
            for var in VARIANTS:
                for sd in SEEDS:
                    z_, imp_ = member(Xt, Xv, y2, w2, init_tr, arm, var, sd, fold, t0)
                    zs.append(z_)
                    imps.append(imp_)
            ENS[arm] = np.mean(zs, axis=0)[mm]
            IMP[arm] = {nm: float(np.mean([i.get(nm, 0.0) for i in imps if i]))
                        for nm in MF}
            del Xt, Xv
            gc.collect()
        del Xtr, Xva
        gc.collect()

        print(f"    mf_ 중요도(PredictionValuesChange 평균) "
              f"MASS {IMP['MASS']}  PLB {IMP['PLB']}")
        print(f"    cat solo: 캐시 {solo(ens0):7.1f}  MASS {solo(ENS['MASS']):7.1f}"
              f"  PLB {solo(ENS['PLB']):7.1f}"
              f"   corr(MASS,캐시) {np.corrcoef(ENS['MASS'], ens0)[0,1]:.4f}"
              f"   corr(MASS,PLB) {np.corrcoef(ENS['MASS'], ENS['PLB'])[0,1]:.4f}")

        n = np.bincount(cl, minlength=n_id)
        seen = np.flatnonzero(n > 0)
        picks = [rng_boot.choice(seen, size=len(seen), replace=True)
                 for _ in range(B_BOOT)]

        def paired(pA, pB):
            SA = np.bincount(cl, weights=(pA - yv) ** 2, minlength=n_id)
            SB = np.bincount(cl, weights=(pB - yv) ** 2, minlength=n_id)
            return np.array([1e5 * (SB[pk].sum() - SA[pk].sum())
                             / (n[pk].sum() * ref) for pk in picks])

        # CatBoost 단독 Δ (진단)
        Ds = paired(shift_to_prior(1/(1+np.exp(-ENS["MASS"])), ybar),
                    shift_to_prior(1/(1+np.exp(-ENS["PLB"])), ybar))
        lo, hi = np.percentile(Ds, [2.5, 97.5])
        print(f"    ① cat 단독 Δ(MASS−PLB) {solo(ENS['MASS'])-solo(ENS['PLB']):+8.2f}"
              f"  [{lo:+7.2f}, {hi:+7.2f}]  P {np.mean(Ds > 0):.3f}")

        print(f"    ② 배포층 Δ(MASS−PLB) — w 격자 (판정은 w={W_FIXED})")
        for w in W_GRID:
            pA, pB = dep_p(ENS["MASS"], w), dep_p(ENS["PLB"], w)
            pt = bss(yv, pA) - bss(yv, pB)
            Dw = paired(pA, pB)
            lo, hi = np.percentile(Dw, [2.5, 97.5])
            pos = float(np.mean(Dw > 0))
            mark = " ← 사전 고정 w" if abs(w - W_FIXED) < 1e-9 else ""
            print(f"      w {w:.2f}  Δ {pt:+7.2f}  [{lo:+7.2f}, {hi:+7.2f}]"
                  f"  P {pos:.3f}{mark}")
            table.append((fold, w, pt, lo, hi, pos))
            if abs(w - W_FIXED) < 1e-9:
                verdict[fold] = (pt > 0, pos)

    print("\n" + "=" * 96)
    print(f"판정 (사전 등록: 배포층 Δ(MASS−PLB), w={W_FIXED} 고정, 양 fold 양수 & P≥0.9)")
    for fold, (pos_sign, p) in verdict.items():
        print(f"  fold{fold}: {'양수' if pos_sign else '음수'} · P {p:.3f}")
    ok = all(v[0] and v[1] >= 0.9 for v in verdict.values())
    print("→ " + ("**후보 생존** — 단 컬럼 경로 LB 전례(§7.56 4전 4패) 명기, LB 슬롯 별도 논의"
                  if ok else
                  "미달 — CatBoost 채널 폐쇄. §7.190 4중 완결 (프로브·층분해·LGBM·CatBoost)"))
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
