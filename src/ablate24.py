"""`RECENCY_HALF_LIFE` 다중 fold 재검증 — 배포 설정에서 유일하게 검증 안 된 축

동기 (MODEL.md §7.22 부산 소득)
    배포값 `hl=0.5`는 §7.2에서 **valid=2024 단일 fold · 단일 LightGBM · MLP 도입 전**에
    골라진 뒤 한 번도 재검증되지 않았다. `ablate19`의 트리 재튜닝에도 없었다.
    §7.18이 "트리 축은 전부 닫혔다"고 했지만 **hl은 트리 축이 아니라 데이터 가중 축**이다.

    그리고 §7.22가 이 값이 위험하다는 직접 증거를 냈다:

        검증 2024   유효표본    lgb     블렌드정점
        W2021       322,948    381.6      425.4     ← 2시즌 더 낡았는데 85점 이긴다
        W2022       328,209    323.6      340.3

    hl=0.5는 유효표본을 ~1.33시즌으로 만든다 (1 + .25 + .0625 + … ). 즉 모델을
    **직전 한 시즌에 건다.** 2021 지배 모델이 2022 지배 모델보다 2024를 잘 맞혔다 —
    **시즌은 교환 가능하지 않다.** 배포는 2024 지배이고, 2024가 2022 같은 이상 시즌이면
    그 손해를 그대로 뒤집어쓴다. hl을 키우는 것은 그 위험에 대한 헤지다.

        hl 0.5 → 유효 1.33시즌 | hl 1.0 → 1.94 | hl 2.0 → 2.81

공짜 보너스 — 교차 조합
    hl마다 z_lgb·z_mlp를 따로 얼려두면 **학습 없이** 다음을 전부 잴 수 있다:
      · 동종 블렌드 (같은 hl의 lgb+mlp)
      · **이종 블렌드** (lgb는 hl=0.5, mlp는 hl=2.0 …)
      · 패밀리 내 다중 hl 평균 (lgb를 hl 0.5·2.0으로 두 벌 학습해 평균)
    이건 정보 추가가 아니라 **학습 분포 다양성**이다. Trackman식 실패 모드
    (외부 데이터와 타깃의 관계가 2025에서 달라짐)가 구조적으로 없다.

사전 예측 (실행 전에 박아둔다)
    P1  fold2024는 여전히 hl=0.5를 고른다.   내 예측: **그렇다**
        (§7.2가 이 fold에서 0.5>0.7>1.0>1.5를 이미 봤고, 2023은 2024를 예측하기에
         좋은 지배 시즌이었다)
    P2  fold2023은 더 큰 hl을 고른다.        내 예측: **그렇다**
        (지배 시즌 2022가 §7.22에서 나쁜 지배 시즌으로 드러났다 → 헤지가 이득)
    P3  따라서 두 fold의 부호가 엇갈려 조건 ②에서 기각된다.  내 예측: **그렇다**

    즉 나는 **hl 변경이 채택되지 않을 것**으로 본다. 그래도 돌리는 이유: P1·P2가 맞으면
    "hl=0.5는 그것을 고른 fold에서만 최적"이 실측되고, 배포값의 근거가 한 fold짜리라는
    것이 기록으로 남는다. 그리고 **교차 조합은 별개 축이라 여기서 기각돼도 살 수 있다.**

    ⚠️ 통제: hl 외에는 전부 고정한다 (`N_ROUNDS=320`도). hl=2.0은 유효 표본이 2배라
    라운드를 늘리는 게 유리할 수 있지만, 두 개를 같이 바꾸면 귀속이 흐려진다.
    hl=2.0이 지면 "저학습 때문일 수 있다"를 결과에 같이 적는다.

실행:  .venv/Scripts/python.exe -u src/ablate24.py 2>&1 | tee experiments/ablate_run24.log
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

import lightgbm as lgb
import numpy as np

import ablate23 as A
import mlp
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA, MLP_SEEDS,
                    MLP_VARIANTS, N_ROUNDS, RECENCY_HALF_LIFE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from features import fit_artifacts
from foldcache import CACHE, LGB_SEEDS, _key
from mlp_infer import encode_apply

THREADS = 5                      # 남은 외부 잡 하나(≈0.85코어)와 나눠 쓴다
mlp.THREADS = THREADS

HLS = [0.5, 1.0, 2.0]
FOLDS = [2023, 2024]
BASE_HL = 0.5                    # 배포값. 비교 기준


def ck(fold, hl, kind, tag):
    return CACHE / f"hl{hl}_f{fold}_{_key()}_{kind}_{tag}.npz"


def run(df, fold, hl, t0):
    """학습 < fold, 검증 == fold. hl만 바꾼다."""
    season = df.season.to_numpy()
    tr, va = df[season < fold], df[season == fold]
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)

    # 배포값이면 foldcache가 이미 갖고 있다 (같은 지문·같은 구성)
    if hl == RECENCY_HALF_LIFE:
        import foldcache
        d = foldcache.load([fold], verbose=True)
        # 캐시의 y를 그대로 쓴다. 내가 만든 yva와 행 순서가 어긋나면 조용히 틀린다
        assert len(d[fold][2]) == len(yva) and abs(d[fold][2].mean() - yva.mean()) < 1e-12
        return d[fold][0], d[fold][1], d[fold][2]

    todo = [(k, t) for k in ("lgb", "mlp") for t in A.member_list(k)]
    have = all(ck(fold, hl, k, t).exists() for k, t in todo)

    w = recency_weight(tr.season.to_numpy(), hl, ref_season=fold - 1)
    if have:
        print(f"\n=== fold {fold}  hl={hl}  체크포인트 재사용 ===")
        Xtr = ytr = Xva = art = None
    else:
        art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                            k_batter=SMOOTH_K_BATTER)
        Xtr, Xva = A.featurize(tr, art), A.featurize(va, art)
        ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
        print(f"\n=== fold {fold}  hl={hl}  n={len(Xtr):,}"
              f"  유효표본 {w.sum():,.0f} ({w.sum()/len(yva):.2f}시즌) ===")
    del tr, va
    gc.collect()

    z_lgb, n = np.zeros(len(yva)), 0
    for vname, over in ENSEMBLE_VARIANTS.items():
        for s in LGB_SEEDS:
            p = ck(fold, hl, "lgb", f"{vname}_{s}")
            if p.exists():
                z_lgb += np.load(p)["z"]
            else:
                par = dict(FINAL_LGB_PARAMS, **over, seed=s, bagging_seed=s,
                           feature_fraction_seed=s, data_random_seed=s,
                           num_threads=THREADS)
                bst = lgb.train(par, lgb.Dataset(Xtr, ytr, weight=w),
                                num_boost_round=N_ROUNDS)
                z = bst.predict(Xva, raw_score=True)
                np.savez_compressed(p, z=z)
                z_lgb += z
                del bst, z
                print(f"  lgb {vname}_{s}  ({time.time()-t0:.0f}s)")
            n += 1
    z_lgb /= n

    z_mlp, m_n = np.zeros(len(yva)), 0
    A_, Ctr, B, Cva = None, None, None, None
    for vname, (cfg, nseed) in MLP_VARIANTS.items():
        for s in MLP_SEEDS[:nseed]:
            p = ck(fold, hl, "mlp", f"{vname}_{s}")
            if p.exists():
                z_mlp += np.load(p)["z"]
            else:
                if A_ is None:               # 인코딩은 실제로 학습할 때만 한다
                    enc = mlp.fit_encoder(Xtr, art)
                    A_, Ctr = encode_apply(Xtr, enc)
                    B, Cva = encode_apply(Xva, enc)
                    sizes = enc["sizes"]
                m, _ = mlp.train_mlp(A_, Ctr, ytr, w, A_[:512], Ctr[:512], sizes,
                                     seed=s, verbose=False, **cfg)
                z = mlp.infer(m, B, Cva)
                np.savez_compressed(p, z=z)
                z_mlp += z
                del m, z
                print(f"  mlp {vname}_{s}  ({time.time()-t0:.0f}s)")
            m_n += 1
    z_mlp /= m_n

    del Xtr, Xva, A_, Ctr, B, Cva
    gc.collect()
    return z_lgb, z_mlp, yva


def blend(zl, zm, y, a=MLP_ALPHA):
    return mlp.score((1 - a) * zl + a * zm, y)


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    df = load_csv(TRAIN_CSV)
    Z = {}
    for fold in FOLDS:
        for hl in HLS:
            Z[(fold, hl)] = run(df, fold, hl, t0)
    del df
    gc.collect()

    print("\n" + "=" * 96)
    print(f"{'fold':>6} {'hl':>5} {'유효시즌':>8} {'lgb':>9} {'mlp':>9} "
          f"{'블렌드@.40':>11} {'α*':>7} {'블렌드@α*':>11}")
    print("=" * 96)
    eff = {0.5: 1.33, 1.0: 1.94, 2.0: 2.81}
    cur = {}
    for fold in FOLDS:
        for hl in HLS:
            zl, zm, y = Z[(fold, hl)]
            bss, err, ref = A.alpha_curve(zl, zm, y)
            st = A.refine(bss)
            cur[(fold, hl)] = (bss, err, ref, st)
            print(f"{fold:>6} {hl:>5.1f} {eff[hl]:>8.2f} {mlp.score(zl, y):>9.1f} "
                  f"{mlp.score(zm, y):>9.1f} {blend(zl, zm, y):>11.1f} "
                  f"{st:>7.3f} {max(bss):>11.1f}")

    # ── 채택 판정 (기준 = hl 0.5, 배포 α) ────────────────────────────
    print("\n" + "=" * 96)
    print(f"채택 판정 — 기준 hl={BASE_HL}, α={MLP_ALPHA} 고정 (배포 설정)")
    print("=" * 96)
    base = {f: blend(*Z[(f, BASE_HL)]) for f in FOLDS}
    print(f"  기준  " + "  ".join(f"fold{f} {base[f]:.1f}" for f in FOLDS))
    for hl in HLS:
        if hl == BASE_HL:
            continue
        d = {f: blend(*Z[(f, hl)]) - base[f] for f in FOLDS}
        sign = len({np.sign(v) for v in d.values()}) == 1
        mags = sorted(abs(v) for v in d.values())
        ratio = mags[-1] / mags[0] if mags[0] > 1e-9 else float("inf")
        ok = sign and ratio <= 3 and np.mean(list(d.values())) > 0
        print(f"  hl={hl:<4} " + "  ".join(f"Δfold{f} {d[f]:+8.1f}" for f in FOLDS)
              + f"   평균 {np.mean(list(d.values())):+8.1f}"
              + f"   부호일치 {'O' if sign else 'X'}   크기비 {ratio:>6.1f}"
              + f"   → {'채택 후보' if ok else '기각'}")

    # ── 교차 조합 (학습 없이) ───────────────────────────────────────
    print("\n" + "=" * 96)
    print("학습 분포 다양성 — 학습 없이 z만 조합. 정보는 그대로고 학습 분포만 다르다")
    print("=" * 96)
    combos = {}
    for hl in HLS:
        combos[f"동종 hl={hl}"] = ({hl: 1.0}, {hl: 1.0})
    combos["이종 lgb.5 / mlp2.0"] = ({0.5: 1.0}, {2.0: 1.0})
    combos["이종 lgb.5 / mlp1.0"] = ({0.5: 1.0}, {1.0: 1.0})
    combos["이종 lgb2.0 / mlp.5"] = ({2.0: 1.0}, {0.5: 1.0})
    combos["lgb 평균(.5,2.0) / mlp.5"] = ({0.5: .5, 2.0: .5}, {0.5: 1.0})
    combos["lgb.5 / mlp 평균(.5,2.0)"] = ({0.5: 1.0}, {0.5: .5, 2.0: .5})
    combos["양쪽 평균(.5,2.0)"] = ({0.5: .5, 2.0: .5}, {0.5: .5, 2.0: .5})
    combos["양쪽 평균(.5,1,2)"] = ({0.5: 1/3, 1.0: 1/3, 2.0: 1/3},) * 2

    print(f"{'조합':>26} " + " ".join(f"{'fold'+str(f):>10}" for f in FOLDS)
          + f" {'평균':>9} {'기준대비':>9} {'부호':>5} {'크기비':>7}")
    base_avg = np.mean([base[f] for f in FOLDS])
    for name, (wl, wm) in combos.items():
        vals = {}
        for f in FOLDS:
            y = Z[(f, BASE_HL)][2]
            zl = sum(c * Z[(f, h)][0] for h, c in wl.items())
            zm = sum(c * Z[(f, h)][1] for h, c in wm.items())
            vals[f] = blend(zl, zm, y)
        d = {f: vals[f] - base[f] for f in FOLDS}
        sign = len({np.sign(v) for v in d.values()}) == 1
        mags = sorted(abs(v) for v in d.values())
        ratio = mags[-1] / mags[0] if mags[0] > 1e-9 else float("inf")
        print(f"{name:>26} " + " ".join(f"{vals[f]:>10.1f}" for f in FOLDS)
              + f" {np.mean(list(vals.values())):>9.1f}"
              + f" {np.mean(list(vals.values()))-base_avg:>+9.1f}"
              + f" {'O' if sign else 'X':>5} {ratio:>7.1f}")

    print("\n  조건 ①효과>노이즈 ②부호일치 ③크기비≤3 ④+100규모 — ④는 이 프로젝트에서")
    print("  이미 완화됐다(MLP는 +11~28로 채택). ①②③을 본다.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
