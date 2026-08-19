"""h 깊이 판정의 블렌드 재측정 — LGBM 단독 기각이 블렌드에서도 성립하는가

qref/qref2/linefill의 기각은 전부 LGBM 단독 하네스였다. 그런데 MLP는 soft
label 수혜가 LGBM의 2.4배(+41.9 vs +17.7) — 잔여 편향이 더 크다면 MLP의
최적 h는 더 깊을 수 있고, 배포물은 0.6·LGBM + 0.4·MLP 블렌드다.
LGBM −3.4여도 MLP가 +8 이상이면 블렌드 부호가 뒤집힌다.

팔: 배포 h  vs  g×0.5 (기각들 중 F행 봉우리 +8.2였던 최유망 변형)
각 팔에서 LGBM 3시드 + MLP(emb4+aux) 3시드 → 시드별 블렌드 로짓 → 쌍체.

실행:  .venv/Scripts/python.exe -u src/blendh.py 2>&1 | tee experiments/blendh.log
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
import pandas as pd

import mlp
import modelabel
import soften as soften_mod
from config import (FINAL_LGB_PARAMS, MLP_ALPHA, MLP_AUX_LAMBDA, MLP_VARIANTS,
                    N_ROUNDS, TARGET_COL, TRAIN_CSV)
from data import load_csv
from qref import fit_h_g
from softlabel import prep_fold
from validation import bss, brier, shift_to_prior

FOLD = 2024
SEEDS = (42, 7, 2024)
CFG = dict(MLP_VARIANTS["emb4"][0], aux=MLP_AUX_LAMBDA)


def sub(yva, p, m):
    ref = yva[m].mean() * (1 - yva[m].mean())
    return 1e5 * (1 - brier(yva[m], p[m]) / ref)


def main():
    df = load_csv(TRAIN_CSV)
    t0 = time.time()
    tmask = (df.season < FOLD).to_numpy()
    tr, va, w0, X = prep_fold(df, FOLD)
    # prep_fold의 art를 다시 얻을 수 없으므로 인코딩용으로 X만 쓰고,
    # mlp.encode에 필요한 art는 fit_artifacts를 중복 호출하는 대신
    # prep_fold를 고치지 않고 priors만 필요한 fit_encoder 경로를 위해 재계산한다.
    from config import (CONTAM_AXES, RECENCY_HALF_LIFE, SMOOTH_K_BATTER,
                        SMOOTH_K_PITCHER)
    from data import recency_weight
    from features import fit_artifacts
    art = fit_artifacts(tr, weights=w0, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    va_f = va["game_type"].to_numpy().astype(str) == "F"
    ytr0 = tr[TARGET_COL].to_numpy(dtype=np.float64)
    aux0, _ = modelabel.aux_labels(tr)
    Xtr0, Xva = X(tr), X(va)
    s_tr = tr["season"].to_numpy()
    g_tr = tr["game_type"].to_numpy().astype(str)

    res = {}
    for name, mult in [("배포", 0.0), ("g×0.5", 0.5)]:
        h, g = fit_h_g(df, tmask, mult)
        h_row = np.zeros(len(tr))
        for yr, v in h.items():
            h_row[(s_tr == yr) & (g_tr == "F")] = v
        dup = (h_row > 0) & (ytr0 == 1)
        w_soft = np.asarray(w0, dtype=np.float64).copy()
        w_soft[dup] *= (1 - h_row[dup])
        X2 = pd.concat([Xtr0, Xtr0[dup]], ignore_index=True)
        y2 = np.concatenate([ytr0, np.zeros(int(dup.sum()))])
        w2 = np.concatenate([w_soft, np.asarray(w0)[dup] * h_row[dup]])
        a_dup = np.zeros_like(aux0[dup])
        from config import MLP_AUX_MODES
        a_dup[:, tuple(MLP_AUX_MODES).index("reverse")] = 1.0   # soften.py와 동일 규약
        aux2 = np.concatenate([aux0, a_dup])
        print(f"\n=== {name}:  h = "
              + "  ".join(f"{yr}:{v:.3f}" for yr, v in sorted(h.items())))

        z_l, z_m = [], []
        for s in SEEDS:
            p = dict(FINAL_LGB_PARAMS, seed=s, bagging_seed=s,
                     feature_fraction_seed=s, data_random_seed=s)
            bst = lgb.train(p, lgb.Dataset(X2, y2, weight=w2),
                            num_boost_round=N_ROUNDS)
            z_l.append(bst.predict(Xva, raw_score=True))
            del bst
        print(f"  LGBM 3시드 완료  ({time.time()-t0:.0f}s)")

        A, C, B, Cv, _, sizes = mlp.encode(X2, Xva, art)
        for s in SEEDS:
            m, _ = mlp.train_mlp(A, C, y2, w2, B, Cv, sizes, seed=s,
                                 verbose=False, y_aux=aux2, **CFG)
            z_m.append(mlp.infer(m, B, Cv))
            del m
        del A, C, B, Cv
        print(f"  MLP 3시드 완료  ({time.time()-t0:.0f}s)")

        vals = []
        for zl, zm in zip(z_l, z_m):
            zb = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm
            for tag, z in [("lgb", zl), ("mlp", zm), ("blend", zb)]:
                p = shift_to_prior(1 / (1 + np.exp(-z)), yva.mean())
                vals.append((tag, bss(yva, p), sub(yva, p, va_f), sub(yva, p, ~va_f)))
        res[name] = vals
        for tag in ("lgb", "mlp", "blend"):
            v = np.array([[a, b, c] for t, a, b, c in vals if t == tag])
            print(f"  {tag:<6} BSS {v[:,0].mean():>8.1f} ± {v[:,0].std(ddof=1):.1f}"
                  f"   F행 {v[:,1].mean():>7.1f}   R행 {v[:,2].mean():>7.1f}")

    print("\n" + "=" * 64)
    print("쌍체 Δ(g×0.5 − 배포)")
    for tag in ("lgb", "mlp", "blend"):
        a = np.array([[x, y_, z] for t, x, y_, z in res["g×0.5"] if t == tag])
        b = np.array([[x, y_, z] for t, x, y_, z in res["배포"] if t == tag])
        d = a - b
        print(f"  {tag:<6} 전체 {d[:,0].mean():+.1f}±{d[:,0].std(ddof=1)/np.sqrt(3):.1f}"
              f"   F행 {d[:,1].mean():+.1f}   R행 {d[:,2].mean():+.1f}")


if __name__ == "__main__":
    main()
