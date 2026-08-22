"""§7.115 — **결함을 고친** 중심화를 배포 층에서 다시 (사용자 지적으로 재조사).

§7.106 의 기각은 **결함 있는 구현**에 대한 측정이었다. 사용자가 "μ_global 을 왜 더했나,
더하면 안 되는 것 아닌가" 라고 물어 재조사했고 결함 **둘**이 나왔다.

  ① **더한 상수가 틀렸다.** 스무딩 prior(recency 가중 .52686) 대신 비가중 전체 평균
     (.54018)을 더했다. 근거는 "트리에는 상수항이 무의미" 였는데, 이 컬럼은 트리로 바로
     안 가고 **먼저 `_smooth` 를 통과한다.** 스무딩은 평행이동에 불변이 아니다 —
     `prior` 는 고정인데 `v` 만 밀면 그 간격(+0.0133)만큼 **저표본 행만** 끌려간다.
  ② **표를 원본 컬럼에서 적합했다.** 중심화는 오염보정·클리핑 **뒤**에 걸리는데,
     오염보정은 2023·2024 행에도 걸린다 (그 투수의 커리어에 pre-2023 F 가 섞여 있다).
     보정량이 누적 F 노출(≈n)의 함수라 잔차가 정확히 **신인–베테랑 축**에 남았다.

고친 뒤 실측 (fold2024 학습셋):
    center=False  시즌별(값−prior)  2019 +0.0286 … 2023 −0.0162   산포 0.045
    center=True   시즌별(값−prior)  2019 +0.0000 … 2023 +0.0000   **완전 정렬**
                  valid2024(외삽)   +0.0026

⚠️ 그래도 신인–베테랑 격차는 +0.0080 → +0.0182 로 커진다. 이건 결함이 아니라
   **중심화의 본질**이다 (§7.106 ⑦) — 틀린 축(시즌)을 정렬하면서 진짜 축(n)의 교란을
   오히려 드러낸다. 전에는 2019 신인(리그가 높아 값이 큼)과 2024 베테랑이 상쇄돼
   격차가 뭉개져 있었는데, 시즌을 정렬하니 신인은 전부 아래로 베테랑은 전부 위로 쌓인다.
   → 그래도 **예측이 아니라 측정으로** 확인한다.

팔 (LGBM 은 둘 다 init_score ON = s29 배포 상태)
    A   현 배포        z_lgb = rounds 캐시 k320,  z_mlp = 배포 캐시
    C2  고친 중심화    LGBM·MLP 둘 다 center_asof=True 로 재학습

실행:  .venv/Scripts/python.exe -u src/asofctr4.py 2>&1 | tee experiments/asofctr4.log
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


def compute(df, fold, t0):
    pth = CACHE / f"ctr4_fold{fold}.npz"
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
                        center_asof=True)
    print(f"    fold {fold} 아티팩트 완료 (2패스 중심화표) ({time.time()-t0:.0f}s)")

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xtr, Xva = X(tr), X(va)
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
    print(f"    fold {fold} lgbm {n}개 ({time.time()-t0:.0f}s)")

    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    zs = []
    for vname, (cfg, nseed) in MLP_VARIANTS.items():
        for sd in MLP_SEEDS[:nseed]:
            _, curve = mlp.train_mlp(A, C, y2, w2, B, Cv, sizes, seed=sd,
                                     verbose=False, y_aux=aux2,
                                     **dict(cfg, aux=MLP_AUX_LAMBDA))
            zs.append(curve[-1])
    z_mlp = np.mean(zs, axis=0)
    print(f"    fold {fold} mlp {len(zs)}개 ({time.time()-t0:.0f}s)")
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
    print(f"train {df.shape}   α={MLP_ALPHA}  init_score=ON  중심화=고친 버전\n")

    R, KEEP = {}, {}
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
        zl_c, zm_c = compute(df, fold, t0)
        for lab, (zl, zm) in (("A 현 배포", (zl_a, zm_a)),
                              ("C2 고친 중심화", (zl_c, zm_c))):
            z = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm + Xax @ dep
            p = shift_to_prior(1 / (1 + np.exp(-z[mm])), yv[mm].mean())
            R[(fold, lab)] = bss(yv[mm], p)
            KEEP[(fold, lab)] = (yv[mm], z[mm], codes[vi][mm])
        print(f"  fold {fold}: A {R[(fold,'A 현 배포')]:.1f}   "
              f"C2 {R[(fold,'C2 고친 중심화')]:.1f}")

    print(f"\n{'='*80}\n█ 판정 — 배포 층 (α={MLP_ALPHA} + δ축 + prior shift, clean 행)\n{'='*80}")
    print(f"  {'팔':<18}" + "".join(f"{f'fold {f}':>11}" for f in FOLDS)
          + "".join(f"{f'Δ ({f})':>12}" for f in FOLDS))
    for lab in ("A 현 배포", "C2 고친 중심화"):
        print(f"  {lab:<18}" + "".join(f"{R[(f,lab)]:>11.1f}" for f in FOLDS)
              + "".join(f"{R[(f,lab)]-R[(f,'A 현 배포')]:>+12.1f}" for f in FOLDS))

    print(f"\n  클러스터 부트스트랩 (투수 복원추출 200회)")
    for f in FOLDS:
        yv, z1, cf = KEEP[(f, "C2 고친 중심화")]
        z0 = KEEP[(f, "A 현 배포")][1]
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
        print(f"     fold {f}: {R[(f,'C2 고친 중심화')]-R[(f,'A 현 배포')]:+6.1f}"
              f"  [{lo:+6.1f}, {hi:+6.1f}]")

    print(f"\n  ★ 대조 — **결함 있던** 중심화의 기록")
    print(f"     §7.106 (init 없음):   −4.5 / −8.2")
    print(f"     §7.111 (init 위에서): −4.1 / −12.3   ← 이번 C2 와 같은 조건")
    print(f"\n  판정 기준(§7.111 개정): 어느 fold 도 0 을 유의하게 밑돌지 않을 것 + fold2024 양수.")
    print(f"  ⚠️ 재학습 축이라 시드 노이즈(std 4~9)가 낀다.")


if __name__ == "__main__":
    main()
