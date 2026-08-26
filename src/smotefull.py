"""§7.162b — 노이즈 증강을 **s37 전체 스택** 위에서 다시 잰다 (사용자 선택)

§7.162 의 하네스는 LGBM 3시드 + fspec 뿐이었다. s37 배포 스택은:

    z = (1−0.75)·z_lgb(4변형×시드) + 0.75·z_mlp(emb4×5 + noid×3)
      → F행에 fspec α=0.458
      → 전 행에 (1−0.10)·z + 0.10·z_cat(4변형×3시드)
      → δ 보정 (ctx 2축 + asof 2축)
      → prior shift → PROB_CENTER/SCALE

즉 **LGBM 은 최종 로짓의 22.5%** 이고, 내가 뺀 MLP 8 + CatBoost 12 = 앙상블 멤버
20개는 노이즈 증강과 **같은 종류의 정규화**다. §7.161 이 남긴 사전 등록 항목이
정확히 이 경우를 가리킨다 — *새 정규화/다양성 후보는 주효과를 재기 전에 배포
스택의 기존 정규화 전부(앙상블 포함)와의 2×2를 먼저 설계한다.*

팔 3개 (fold2024 = 배포 레짐)
    base            현 배포
    SMOTE r=0.40    pre-2023 F & y=0 에 13,357 행 보간   ← 사용자 원안
    PLACEBO r=0.40  같은 수를 무결한 pre-2023 R & y=0 에  ← §7.162 에서 후보보다 높았다

**층별로 찍는다** — LGBM만 / +MLP / +fspec / +CatBoost / +δ.
효과가 어느 층에서 희석·중복되는지가 이 실험의 답이다.

⚠️ 합성 행의 보조 라벨(aux)은 **씨앗 행의 것을 상속**한다. 보간된 행의 진짜
   몰림/반대는 정의되지 않는다 — 씨앗 상속이 가장 덜 임의적인 선택이다.

실행:  .venv/Scripts/python.exe -u src/smotefull.py 2>&1 | tee experiments/smotefull.log
"""
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

import jointbeta as J
import mlp
import modelabel
import soften
from config import (ASOF_AXES, CAT_ALPHA, CAT_ITERS, CAT_PARAMS, CAT_SEEDS,
                    CAT_VARIANTS, CTX_AXES, ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS,
                    FSPEC_ALPHA, MLP_ALPHA, MLP_AUX_LAMBDA, MLP_AUX_MODES,
                    MLP_SEEDS, MLP_VARIANTS, N_ROUNDS, SEED_ENSEMBLE, TARGET_COL,
                    TRAIN_CSV)
from data import load_csv
from hbeta2 import season_gt_level, train_fspec
from predict import PROB_CENTER, PROB_SCALE
from smotenc import n_needed, prep, smote, target_mask
from validation import bss, shift_to_prior

FOLD = 2024
LGB_SEEDS = SEED_ENSEMBLE[:3]          # foldcache 규약 (4변형 × 3시드 = 12멤버)
RATIO = 0.40
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
TAG = "sfull"


def build(tr, Xtr, w, kind, rng):
    """(X2, y2, w2, aux2, init, n_syn). 순서 = [원본][soften 복제][합성]."""
    y = tr[TARGET_COL].to_numpy(dtype=np.float64)
    s = tr["season"].to_numpy(dtype=np.int64)
    g = tr["game_type"].astype("object").to_numpy().astype(str)
    aux, cov = modelabel.aux_labels(tr)

    # ── ① soften (배포와 같은 순서 — 원본 행에만 걸린다) ──────────────
    X1, y1, w1, aux1, h = soften.soften(tr, Xtr, y, np.asarray(w, np.float64), aux)
    hr = np.zeros(len(tr))
    for yr, v in h.items():
        hr[(s == yr) & (g == "F")] = v
    dm = (hr > 0) & (y == 1)
    ss1 = np.concatenate([s, s[dm]])
    gt1 = np.concatenate([g, g[dm]])

    # ── ② 증강 (합성 행은 y=0 이라 soften 과 겹치지 않는다) ───────────
    n_syn = 0
    if kind != "base":
        nF1 = int(((s <= 2022) & (g == "F") & (y == 1)).sum())
        nF0 = int(((s <= 2022) & (g == "F") & (y == 0)).sum())
        n_syn = n_needed(nF1, nF0, RATIO)
        if kind == "PLACEBO":
            mF, mR = target_mask(tr, "F"), target_mask(tr, "R")
            m = np.zeros(len(tr), bool)
            for yr in range(2019, 2023):
                cand = np.flatnonzero(mR & (s == yr))
                m[rng.choice(cand, size=min(len(cand), int((mF & (s == yr)).sum())),
                             replace=False)] = True
        else:
            m = target_mask(tr, "F")
        sub = Xtr[m].to_numpy(dtype=np.float64)
        new, seed = smote(sub, n_syn, rng)
        src = np.flatnonzero(m)[seed]
        import pandas as pd
        X1 = pd.concat([X1, pd.DataFrame(new.astype(np.float32), columns=Xtr.columns)],
                       ignore_index=True)
        y1 = np.concatenate([y1, np.zeros(n_syn)])
        w1 = np.concatenate([w1, np.asarray(w, np.float64)[src]])
        aux1 = np.concatenate([aux1, aux[src]])      # 씨앗 상속 (docstring 참조)
        ss1 = np.concatenate([ss1, s[src]])
        gt1 = np.concatenate([gt1, g[src]])

    init = season_gt_level(ss1, gt1, y1, w1)
    assert len(X1) == len(y1) == len(w1) == len(aux1) == len(init), "정렬이 깨졌다"
    return X1, y1, w1, aux1, init, n_syn


def members(kind, X2, y2, w2, aux2, init, Xva, art, t0):
    """LGBM 12 + MLP 8 + CatBoost 12. 멤버 단위로 캐시한다."""
    def cp(k):
        return CACHE / f"{TAG}_f{FOLD}_{kind}_{k}.npz"

    # LGBM
    if cp("lgb").exists():
        z_lgb = np.load(cp("lgb"))["z"]
        print(f"    lgbm 재사용 ({time.time()-t0:.0f}s)")
    else:
        z_lgb, n = np.zeros(len(Xva)), 0
        for over in ENSEMBLE_VARIANTS.values():
            for sd in LGB_SEEDS:
                p = dict(FINAL_LGB_PARAMS, **over, seed=sd, bagging_seed=sd,
                         feature_fraction_seed=sd, data_random_seed=sd)
                b = lgb.train(p, lgb.Dataset(X2, y2, weight=w2, init_score=init),
                              num_boost_round=N_ROUNDS)
                z_lgb += b.predict(Xva, raw_score=True)
                n += 1
                del b
        z_lgb /= n
        np.savez_compressed(cp("lgb"), z=z_lgb)
        print(f"    lgbm {n}개 ({time.time()-t0:.0f}s)")

    # MLP
    zs = []
    A, C, B, Cv, _, sizes = mlp.encode(X2, Xva, art)
    for vn, (cfg, ns) in MLP_VARIANTS.items():
        for sd in MLP_SEEDS[:ns]:
            f = cp(f"mlp_{vn}_{sd}")
            if f.exists():
                zs.append(np.load(f)["z"])
                continue
            _, curve = mlp.train_mlp(A, C, y2, w2, B, Cv, sizes, seed=sd,
                                     verbose=False, y_aux=aux2,
                                     **dict(cfg, aux=MLP_AUX_LAMBDA))
            zs.append(curve[-1])
            np.savez_compressed(f, z=zs[-1])
            print(f"    mlp {vn}_{sd} ({time.time()-t0:.0f}s)")
    z_mlp = np.mean(zs, axis=0)
    del A, C, B, Cv

    # CatBoost
    if cp("cat").exists():
        z_cat = np.load(cp("cat"))["z"]
        print(f"    cat 재사용 ({time.time()-t0:.0f}s)")
    else:
        from catboost import CatBoostClassifier, Pool
        pool = Pool(X2, y2, weight=w2, baseline=init)
        ref = Pool(Xva)
        z_cat, n = np.zeros(len(Xva)), 0
        for var, over in CAT_VARIANTS.items():
            for sd in CAT_SEEDS:
                m = CatBoostClassifier(iterations=CAT_ITERS, random_seed=sd,
                                       **CAT_PARAMS, **over)
                m.fit(pool)
                z_cat += m.predict(ref, prediction_type="RawFormulaVal")
                n += 1
                del m
        z_cat /= n
        np.savez_compressed(cp("cat"), z=z_cat)
        print(f"    catboost {n}개 ({time.time()-t0:.0f}s)")
    return z_lgb, z_mlp, z_cat


def layers(z_lgb, z_mlp, z_cat, z_fs, isF, ax, yva):
    """층을 하나씩 얹으며 BSS 를 찍는다. 순서는 train.py 배포 순서와 같다."""
    yb = yva.mean()

    def sc(z, with_ax=False, with_ps=False):
        zz = z + (ax if with_ax else 0.0)
        p = shift_to_prior(1 / (1 + np.exp(-zz)), yb)
        if with_ps:
            p = shift_to_prior(PROB_CENTER + PROB_SCALE * (p - PROB_CENTER), yb)
        return bss(yva, p)

    out = {"① LGBM만": sc(z_lgb)}
    z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
    out["② +MLP(α.75)"] = sc(z)
    z = z.copy()
    z[isF] = (1 - FSPEC_ALPHA) * z[isF] + FSPEC_ALPHA * z_fs[isF]
    out["③ +fspec"] = sc(z)
    z = (1 - CAT_ALPHA) * z + CAT_ALPHA * z_cat
    out["④ +CatBoost"] = sc(z)
    out["⑤ +δ·스케일 (배포)"] = sc(z, with_ax=True, with_ps=True)
    return out


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    tr, va, w, X, art = prep(df, FOLD)
    Xtr, Xva = X(tr), X(va)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    isF = va["game_type"].to_numpy().astype(str) == "F"
    z_fs, n_fs = train_fspec(tr, X, Xva)

    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    # ⚠️ axis_matrix 는 **검증 fold 행만** 돌려준다 (gmix.py:63 과 같은 규약).
    #    df 전체 길이로 다시 인덱싱하면 안 된다.
    Xax, _, _, _ = J.axis_matrix(df, df["season"].to_numpy(), FOLD)
    ax = Xax @ dep
    assert len(ax) == len(Xva), "δ축 길이가 검증행과 다르다"
    print(f"준비 완료 ({time.time()-t0:.0f}s)  fspec {n_fs:,}행  δ축 {len(dep)}개 "
          f"(기여 평균 {ax.mean():+.5f})")

    R = {}
    for kind in ("base", "SMOTE", "PLACEBO"):
        print(f"\n  [{kind}]")
        rng = np.random.default_rng(777)
        X2, y2, w2, aux2, init, ns = build(tr, Xtr, w, kind, rng)
        print(f"    행 {len(Xtr):,} → {len(X2):,}  (합성 {ns:,})")
        zl, zm, zc = members(kind, X2, y2, w2, aux2, init, Xva, art, t0)
        R[kind] = layers(zl, zm, zc, z_fs, isF, ax, yva)
        del X2, y2, w2, aux2, init

    keys = list(R["base"])
    bar = "=" * 84
    print(f"\n{bar}\n층별 BSS (fold{FOLD}) — Δ 는 base 대비\n{bar}")
    print("  %-22s %10s %12s %10s %12s %10s"
          % ("층", "base", "SMOTE", "Δ", "PLACEBO", "Δ"))
    for k in keys:
        print("  %-22s %10.1f %12.1f %+10.1f %12.1f %+10.1f"
              % (k, R["base"][k], R["SMOTE"][k], R["SMOTE"][k] - R["base"][k],
                 R["PLACEBO"][k], R["PLACEBO"][k] - R["base"][k]))
    print(f"\n  ⑤ 가 배포층이다. ① 에서의 이득이 ⑤ 까지 얼마나 남는지가 이 실험의 답.")
    print(f"  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
