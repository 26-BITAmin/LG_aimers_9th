"""§7.187 — GNN 채널 ①②③ 재검증 · **학습 경로(holdout)** — 사전 등록

사용자: *"잔차로 확인하지 말고, 상호작용 그런 요소들도 있으니까 holdout 으로 확인해."*

§7.183 잔차 프로브는 학습이 없다 → 기존 피처와의 **상호작용으로만** 발현되는 신호를
원리적으로 놓칠 수 있다 ([[placebo-pass-doesnt-mean-incremental]] 의 쌍대 — 프로브
실패도 증분 부재를 보장하지 않는다). 여기서는 그래프 전파 컬럼 3개를 배포 LGBM
코어의 피처로 **추가·재학습**해 트리가 상호작용을 형성할 기회를 준다
(예: "상대가 쉬웠던 투수일수록 asof 성공률을 깎아 읽는" 분기).

팔 (fold2023·fold2024, foldcache compute() 와 같은 경로: fold_data → aux → soften
    → 12부스터(4변형×3시드) — BASE 는 캐시 재사용이라 재학습 0)
    BASE  캐시 z_lgb (fold{f}_{KEY}.npz)
    GNN   같은 경로 + 컬럼 3개: p1_opp / b1_opp / p2_coh (§7.183 과 동일 구성 —
          fold 학습 시즌만, LOO, hl0.5, k50. 표는 train 에 적합, train/va 양쪽 사상)
    PLB   같은 경로 + **순열 컬럼 3개** (§7.183 위약 표 — 기하 보존·정체 파괴).
          ⚠️ 이 대조군이 판정자다: 컬럼 추가 자체가 feature_fraction 추첨을 바꿔
          시드 노이즈(std 4~9 BSS)를 만든다. GNN−BASE 가 아니라 **GNN−PLB** 를 본다.

평가 (사전 등록)
    z_lgb 단독 Δ + 블렌드 프록시(0.25·z_lgb + 0.75·z_mlp캐시) Δ.
    투수 클러스터 부트스트랩 쌍체 B=2000. 새 컬럼 split 수(실제 사용량) 병기.
    판정: 채널 실재 = 블렌드 Δ(GNN − PLB) 가 **양 fold 양수 & P ≥ 0.9**.
    통과 시에도 컬럼 경로의 LB 전례(§7.56 — 4전 4패, 분할 전이 배율 68.6)를
    장부에 명기하고 전달 형태를 다시 설계한다. 미달이면 ①②③ 폐쇄 확정.

실행:  .venv/Scripts/python.exe -u src/gnncol.py 2>&1 | tee experiments/gnncol.log
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
import pandas as pd

import foldcache as fc
import gnnprobe as G
import modelabel
import soften
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA, N_ROUNDS,
                    TARGET_COL, TRAIN_CSV)
from data import load_csv
from validation import bss, shift_to_prior

CACHE = fc.CACHE
KEY = fc._key()
SEEDS3 = fc.LGB_SEEDS
B_BOOT = 2000
COLS = {"GNN": ("p1_opp", "b1_opp", "p2_coh"),
        "PLB": ("p1_plb", "b1_plb", "p2_plb")}


def add_cols(X, tabs, d, names):
    """엔티티 표를 행에 사상해 X 에 붙인다 (미등장 → NaN, LGBM 이 자체 처리)."""
    X = X.copy()
    pid, bid = d.pitcher_id.to_numpy(), d.batter_id.to_numpy()
    for nm in names:
        axis, tab = tabs[nm]
        keys = pid if axis == "p" else bid
        X["g_" + nm] = pd.Series(keys).map(tab).to_numpy(dtype=np.float32)
    return X


def train_arm(arm, fold, df, tabs, t0):
    f = CACHE / f"gnncol_fold{fold}_{arm}_{KEY}.npz"
    if f.exists():
        with np.load(f) as z:
            print(f"  {arm} fold{fold} 재사용")
            return z["z"], dict(zip(z["snames"], z["splits"]))
    Xtr, ytr, w, Xva, yva, art, tr = fc.fold_data(df, fold)
    mtr, mva = fc.SPLITS[fold](df.season.to_numpy(), df.game_month.to_numpy())
    va = df[mva]
    Xtr = add_cols(Xtr, tabs, tr, COLS[arm])
    Xva = add_cols(Xva, tabs, va, COLS[arm])
    cov = [float(np.isfinite(Xva["g_" + nm]).mean()) for nm in COLS[arm]]
    aux, _ = modelabel.aux_labels(tr)
    Xtr, ytr, w, aux, _h = soften.soften(tr, Xtr, ytr, w, aux)
    print(f"  [{arm} fold{fold}] 학습 {len(Xtr):,}행 · 피처 {Xtr.shape[1]}"
          f" · va 커버리지 {['%.1f%%' % (c*100) for c in cov]}")
    z, n, splits = np.zeros(len(Xva)), 0, {}
    for vname, over in ENSEMBLE_VARIANTS.items():
        for s in SEEDS3:
            p = dict(FINAL_LGB_PARAMS, **over, seed=s, bagging_seed=s,
                     feature_fraction_seed=s, data_random_seed=s)
            bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w),
                            num_boost_round=N_ROUNDS)
            z += bst.predict(Xva, raw_score=True)
            n += 1
            for nm, sp in zip(bst.feature_name(),
                              bst.feature_importance("split")):
                if nm.startswith("g_"):
                    splits[nm] = splits.get(nm, 0) + int(sp)
            del bst
        print(f"    {vname} ({time.time()-t0:.0f}s)")
    z /= n
    np.savez_compressed(f, z=z,
                        snames=np.array(list(splits), dtype=object),
                        splits=np.array(list(splits.values())))
    del Xtr, Xva
    gc.collect()
    return z, splits


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    base = fc.load([2023, 2024], verbose=True)

    Z, SP, Y, CL = {}, {}, {}, {}
    for fold in (2023, 2024):
        rng = np.random.default_rng(1000 + fold)      # §7.183 과 같은 시드·같은 표
        tabs = G.graph_tables(df, s, fold, rng)
        for arm in ("GNN", "PLB"):
            Z[(arm, fold)], SP[(arm, fold)] = train_arm(arm, fold, df, tabs, t0)
        _, mva = fc.SPLITS[fold](s, df.game_month.to_numpy())
        va = df[mva]
        Y[fold] = va[TARGET_COL].to_numpy(dtype=np.float64)
        CL[fold] = pd.factorize(va["pitcher_id"].to_numpy())[0]
    del df
    gc.collect()

    def sc(z, y):
        return bss(y, shift_to_prior(1 / (1 + np.exp(-z)), y.mean()))

    print("\n" + "=" * 94)
    print("§7.187 GNN 채널 ①②③ 학습 경로 재검증 · 12부스터 · soften · B=%d" % B_BOOT)
    print("=" * 94)
    verdict = {}
    for fold in (2023, 2024):
        zl, zm, y = base[fold][0], base[fold][1], base[fold][2]
        assert np.allclose(y, Y[fold]), "캐시 y 불일치"
        cl, n_id = CL[fold], CL[fold].max() + 1
        arms = {"BASE": zl, "GNN": Z[("GNN", fold)], "PLB": Z[("PLB", fold)]}
        print(f"\n  ── fold {fold} ──   (컬럼 split 수: "
              f"GNN {SP[('GNN', fold)]}  PLB {SP[('PLB', fold)]})")
        print("  %-6s %12s %12s" % ("팔", "LGBM 단독", "블렌드(.75 MLP)"))
        P_l, P_b = {}, {}
        for a, z in arms.items():
            P_l[a] = sc(z, y)
            P_b[a] = sc((1 - MLP_ALPHA) * z + MLP_ALPHA * zm, y)
            print("  %-6s %12.1f %12.1f" % (a, P_l[a], P_b[a]))

        # 쌍체 부트스트랩: 같은 클러스터 draw 로 GNN−PLB, GNN−BASE
        rng = np.random.default_rng(42)
        pb = {a: shift_to_prior(1 / (1 + np.exp(-((1 - MLP_ALPHA) * z
                                                  + MLP_ALPHA * zm))), y.mean())
              for a, z in arms.items()}
        ref = y.mean() * (1 - y.mean())
        n = np.bincount(cl, minlength=n_id)
        S = {a: np.bincount(cl, weights=(p - y) ** 2, minlength=n_id)
             for a, p in pb.items()}
        seen = np.flatnonzero(n > 0)
        D_gp, D_gb = np.empty(B_BOOT), np.empty(B_BOOT)
        for i in range(B_BOOT):
            pick = rng.choice(seen, size=len(seen), replace=True)
            npick = n[pick].sum() * ref
            D_gp[i] = 1e5 * (S["PLB"][pick].sum() - S["GNN"][pick].sum()) / npick
            D_gb[i] = 1e5 * (S["BASE"][pick].sum() - S["GNN"][pick].sum()) / npick
        for nm, D, pt in (("GNN−PLB (판정)", D_gp, P_b["GNN"] - P_b["PLB"]),
                          ("GNN−BASE (참고)", D_gb, P_b["GNN"] - P_b["BASE"])):
            lo, hi = np.percentile(D, [2.5, 97.5])
            print(f"  블렌드 Δ {nm:<16s} {pt:+8.2f}  [{lo:+7.2f}, {hi:+7.2f}]"
                  f"  P {np.mean(D > 0):.3f}")
        verdict[fold] = (P_b["GNN"] - P_b["PLB"] > 0, float(np.mean(D_gp > 0)))

    ok = all(v[0] and v[1] >= 0.9 for v in verdict.values())
    print("\n" + "=" * 94)
    print("판정 (사전 등록: 블렌드 GNN−PLB 양 fold 양수 & P≥0.9)")
    for fold, (pos, p) in verdict.items():
        print(f"  fold{fold}: {'양수' if pos else '음수'} · P {p:.3f}")
    if ok:
        print("→ **채널 실재** — 단, 컬럼 경로 LB 전례는 4전 4패(§7.56). 전달 형태 재설계 후 논의.")
    else:
        print("→ 학습 경로에서도 미달 — GNN 채널 ①②③ 폐쇄 확정 (잔차·학습 양 경로 일치).")
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
