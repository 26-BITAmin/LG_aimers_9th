"""§7.150 — 중심화 + **부호 있는 제곱** `sign(c)·c²` (`lag5sq`). 대조군은 캐시 재사용.

사용자 제안: *"센터링만 하는데 여기에 100을 곱하고 그러고 제곱까지"* → 제곱이 부호를
접는다는 지적에 → *"부호가 문제라면 마지막에 부호를 붙이면 되는거지"*.

두 성분을 분리해 두면 실험할 것이 하나만 남는다.

**① ×100 은 무연산이다 (실험 불필요).**
   LGBM 은 `x ≤ t` 비교라 강단조 변환에 불변. MLP 는 표준화가 `(100x−mean)/std` 에서
   100 을 정확히 상쇄한다. `sign(kc)(kc)² = k³ sign(c)c²` 도 마찬가지로 k³ 이 상쇄된다.

**② 부호 있는 제곱은 정보를 안 버린다 — 그래서 트리에도 무연산이다.**
   `f(x)=x|x|` 는 강단조(f'=2|x|≥0, 단사)라 **분할 집합이 동일**하다.
   → **LGBM 예측이 `lag5` 와 같아야 한다. 아래 verify 가 그걸 assert 한다**
     (같지 않으면 배선 버그다. 이 실험의 하네스 검증이자 명제의 실측이다).
   → 움직일 수 있는 것은 **MLP 뿐**이다. 표준화 뒤에도 꼬리가 중심 대비 늘어난다.

즉 이 실험의 실제 질문은 **"5개 중심화 입력의 꼬리를 늘리면 MLP 가 좋아지는가"** 다.

사전 근거 (§7.150 잔차 진단 `src/sqdiag.py`, 학습 0회):
    단순 제곱 c² 은 배포 잔차에 신호가 없다 — 5컬럼 전부 in-sample 최적 β 에서도
    Δ=0.0, t −2.11~+0.34, 전이 전부 0.0. |c| 는 f2023 에서 +10.2 였으나 전이 −9.2.
    → **가법적 곡률 항으로서는 죽었다.** 다만 그 진단은 MLP 입력 기하를 못 본다.
      이 실험이 그 남은 통로를 잰다.

팔 (LGBM init_score ON = s29 배포 상태)
    A 현 배포     캐시            중심화 없음
    lag5         캐시 (§7.119)   중심화만 (−11.3 / −3.1)
    lag5sq       **신규**        중심화 + sign(c)·c²

읽는 법: lag5sq − lag5 가 **제곱만의 효과**다 (중심화는 양쪽에 있다). 이게 이 실험의
판정 대상이고, lag5sq − A 는 참고용(중심화 손해가 섞여 있다).

채택 기준(§7.111 개정): 어느 fold 도 0 을 유의하게 밑돌지 않을 것 + fold2024 양수.
검정력: 시드 노이즈 std 4~9 → |Δ| ≲ 10 인 대비는 "미판정" 으로 보고한다.

실행:  .venv/Scripts/python.exe -u src/asofctr8.py 2>&1 | tee experiments/asofctr8.log
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

import foldcache as _fc

_fc.CENTER_ASOF = False
KEY = _fc._key()

import lightgbm as lgb                                              # noqa: E402
import pandas as pd                                                 # noqa: E402
import features as F                                                # noqa: E402
import mlp                                                          # noqa: E402
import modelabel                                                    # noqa: E402
import soften                                                       # noqa: E402
import jointbeta as J                                               # noqa: E402
from config import (ASOF_AXES, CONTAM_AXES, CTX_AXES, DROP_FEATURES,            # noqa: E402
                    ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA,
                    MLP_AUX_LAMBDA, MLP_SEEDS, MLP_VARIANTS, N_ROUNDS,
                    RECENCY_HALF_LIFE, SEED_ENSEMBLE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight                           # noqa: E402
from features import build_features, fit_artifacts                  # noqa: E402
from initfix3 import level_init                                     # noqa: E402
from validation import bss, shift_to_prior                          # noqa: E402

FOLDS = [2023, 2024]
LGB_SEEDS = SEED_ENSEMBLE[:3]
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED = 42
VCOL = "asof_pitcher_success_rate"


def verify(art, Xtr, tr, mode):
    ac = art["asof_center"]
    cols = list(ac["table"])
    assert set(cols) == set(F.CENTER_SUBSET5), f"대상 목록 불일치: {cols}"
    assert all(art["priors"][c] == 0.0 for c in cols), "priors 가 0 이 아니다"
    v = Xtr[VCOL].to_numpy(dtype=np.float64)
    fin = np.isfinite(v)
    # 부호 있는 제곱이면 값의 스케일이 c 의 제곱 대역이어야 한다 (c ~ 0.03 → ~1e-3)
    print(f"    · {mode} {VCOL}: sd {v[fin].std():.6f}  "
          f"음수비 {(v[fin] < 0).mean():.3f}  |max| {np.abs(v[fin]).max():.6f}")
    s2 = tr["season"].to_numpy()
    g2 = tr["game_type"].astype(str).to_numpy()
    m19 = (s2 == 2019) & (g2 == "R") & fin
    assert abs(v[m19].mean()) < 0.02, "중심화가 안 걸렸다"


def compute(df, fold, mode, t0):
    pth = CACHE / f"ctr8_fold{fold}_{mode}.npz"
    if pth.exists():
        d = np.load(pth)
        print(f"    캐시 적중 {pth.name}")
        return d["z_lgb"], d["z_mlp"]

    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                        center_asof=mode)
    print(f"    fold {fold} [{mode}] 아티팩트 완료 ({time.time()-t0:.0f}s)")

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xtr, Xva = X(tr), X(va)
    verify(art, Xtr, tr, mode)
    ss = tr["season"].to_numpy(dtype=np.int64)
    gg = tr["game_type"].astype("object").to_numpy().astype(str)
    aux, _ = modelabel.aux_labels(tr)
    h = soften.fit_h(tr)
    h_row = np.zeros(len(tr))
    for yr, v in h.items():
        h_row[(ss == yr) & (gg == "F")] = v
    dup = (h_row > 0) & (y_tr == 1)
    Xtr, y2, w2, aux2, _ = soften.soften(tr, Xtr, y_tr, w, aux)
    init_tr = level_init(np.concatenate([ss, ss[dup]]),
                         np.concatenate([gg, gg[dup]]), y2, w2)
    assert len(init_tr) == len(Xtr), "정렬 깨짐"

    z_lgb, n = np.zeros(len(Xva)), 0
    for over in ENSEMBLE_VARIANTS.values():
        for sd in LGB_SEEDS:
            p = dict(FINAL_LGB_PARAMS, **over, seed=sd, bagging_seed=sd,
                     feature_fraction_seed=sd, data_random_seed=sd)
            bst = lgb.train(p, lgb.Dataset(Xtr, y2, weight=w2, init_score=init_tr),
                            num_boost_round=N_ROUNDS)
            z_lgb += bst.predict(Xva, raw_score=True)
            n += 1
            del bst
    z_lgb /= n
    print(f"    fold {fold} [{mode}] lgbm {n}개 ({time.time()-t0:.0f}s)")

    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    zs = []
    for vname, (cfg, nseed) in MLP_VARIANTS.items():
        for sd in MLP_SEEDS[:nseed]:
            _, curve = mlp.train_mlp(A, C, y2, w2, B, Cv, sizes, seed=sd,
                                     verbose=False, y_aux=aux2,
                                     **dict(cfg, aux=MLP_AUX_LAMBDA))
            zs.append(curve[-1])
    z_mlp = np.mean(zs, axis=0)
    print(f"    fold {fold} [{mode}] mlp {len(zs)}개 ({time.time()-t0:.0f}s)")
    np.savez_compressed(pth, z_lgb=z_lgb, z_mlp=z_mlp)
    return z_lgb, z_mlp


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    codes, uniq = pd.factorize(df["pitcher_id"].to_numpy())
    n_id = len(uniq)
    mclean = ~((gt == "F") & (s < 2023))
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    print(f"train {df.shape}   α={MLP_ALPHA}  init_score=ON   팔: lag5sq (대조 lag5 캐시)\n")

    labs = ["A 현 배포", "lag5", "lag5sq"]
    R, KEEP, LGBDIFF = {}, {}, {}
    for fold in FOLDS:
        Xax, _, _, _ = J.axis_matrix(df, s, fold)
        vi = np.where(s == fold)[0]
        mm = mclean[vi]
        yv = y[vi]
        with np.load(CACHE / f"rounds_fold{fold}_r640.npz") as d:
            zl_a = d["k320"]
        with np.load(CACHE / f"fold{fold}_{KEY}.npz") as d:
            zm_a = d["z_mlp"]
            assert np.allclose(d["y"], yv)
        with np.load(CACHE / f"ctr7_fold{fold}_lag5.npz") as d:   # §7.119 캐시 재사용
            zl_c, zm_c = d["z_lgb"], d["z_mlp"]
        arms = {"A 현 배포": (zl_a, zm_a), "lag5": (zl_c, zm_c),
                "lag5sq": compute(df, fold, "lag5sq", t0)}

        # ★ 명제의 실측 — 강단조 변환이므로 LGBM 이 (거의) 같아야 한다
        d_lgb = np.abs(arms["lag5sq"][0] - zl_c)
        LGBDIFF[fold] = (float(d_lgb.max()), float(np.corrcoef(
            arms["lag5sq"][0], zl_c)[0, 1]), float(np.std(zl_c)))
        for lab, (zl, zm) in arms.items():
            z = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm + Xax @ dep
            p = shift_to_prior(1 / (1 + np.exp(-z[mm])), yv[mm].mean())
            R[(fold, lab)] = bss(yv[mm], p)
            KEEP[(fold, lab)] = (yv[mm], z[mm], codes[vi][mm])
        print(f"  fold {fold}: " + "  ".join(f"{k} {R[(fold,k)]:.1f}" for k in labs))

    print(f"\n{'='*88}\n█ 판정 — 배포 층 (α={MLP_ALPHA} + δ축 + prior shift, clean 행)\n{'='*88}")
    print(f"  {'팔':<12}" + "".join(f"{f'fold {f}':>11}" for f in FOLDS)
          + "".join(f"{f'Δ vs A ({f})':>15}" for f in FOLDS))
    for lab in labs:
        print(f"  {lab:<12}" + "".join(f"{R[(f,lab)]:>11.1f}" for f in FOLDS)
              + "".join(f"{R[(f,lab)]-R[(f,'A 현 배포')]:>+15.1f}" for f in FOLDS))

    print(f"\n  ★ 명제 검증 — `sign(c)·c²` 은 강단조라 **트리에 무연산**이어야 한다")
    for f in FOLDS:
        mx, cr, sdz = LGBDIFF[f]
        print(f"     fold {f}: LGBM 로짓 최대차 {mx:.3e}  corr {cr:.8f}"
              f"  (참고: z 의 sd {sdz:.4f})")
    print(f"     → 최대차가 sd 대비 무시할 수준이면 명제 확인. 아니면 배선 버그다.")

    def boot(f, lab1, lab0):
        yv, z1, cf = KEEP[(f, lab1)]
        z0 = KEEP[(f, lab0)][1]
        order = np.argsort(cf, kind="stable")
        bnd = np.searchsorted(cf[order], np.arange(n_id + 1))
        seen = np.where(np.diff(bnd) > 0)[0]
        bs = []
        for _ in range(200):
            idx = np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                                  rng.choice(seen, size=len(seen), replace=True)])
            yy = yv[idx]
            bs.append(bss(yy, shift_to_prior(1 / (1 + np.exp(-z1[idx])), yy.mean()))
                      - bss(yy, shift_to_prior(1 / (1 + np.exp(-z0[idx])), yy.mean())))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return R[(f, lab1)] - R[(f, lab0)], lo, hi

    print(f"\n  █ 판정 대비 — 팔 쌍 직접 부트스트랩 (투수 복원추출 200회)")
    for name, a, b in [("**제곱만의 효과** (lag5sq−lag5)", "lag5sq", "lag5"),
                       ("총합 (lag5sq−A)", "lag5sq", "A 현 배포"),
                       ("참고: 중심화만 (lag5−A)", "lag5", "A 현 배포")]:
        cells = [f"{d:+6.1f} [{lo:+6.1f},{hi:+6.1f}]"
                 for d, lo, hi in (boot(f, a, b) for f in FOLDS)]
        print(f"     {name:<32}" + "".join(f"{c:>24}" for c in cells))
    print(f"     (시드 노이즈 std 4~9 → |Δ| ≲ 10 이면 미판정)")


if __name__ == "__main__":
    main()
