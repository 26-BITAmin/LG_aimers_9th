"""§7.190c — 사용자 반문 2: "컬럼이 추가되면 blend 자체(비중)도 변할 수 있지 않나"

기전 (사전 등록)
    §7.190b 의 (0.25)² 상한은 **블렌드 비중 고정** 가정의 1차 근사다. LGBM 팔이
    좋아지면 최적 α 가 LGBM 쪽으로 움직여 상한을 넘을 수 있다 — 프로브로는 못 닫는다.
    → 학습 경로에서 α 를 격자로 풀어 잰다. 어떤 α 에서도 Δ(MASS−PLB) 가 안 뜨면
    재평형 채널까지 포함해 폐쇄다 (§7.188 kNN 의 "전 α 음수" 폐쇄와 같은 형식).

팔 (fold2023·fold2024 — §7.187 gnncol 하네스 그대로: fc.fold_data → aux → soften
    → 12부스터(4변형×3시드). BASE 는 fc.load 캐시 재사용, 재학습 0)
    MASS  §7.190b L층 상위 가족 3컬럼 추가 (train 적합 z 상수로 표준화한 차분 —
          순수 행 함수라 배포 가능):
            mf_cd_hs   z(count_diff) − z(hand_same)                          S_L 3.48
            mf_cd_p5m  z(count_diff) − z(asof_pitcher_prev5_game_middle_rate) S_L 3.39
            mf_bm_p5s  z(asof_batter_middle_rate_sm) − z(asof_pitcher_prev5_game_success_rate) S_L 3.13
    PLB   같은 구성, b쪽 컬럼을 split 안에서 행 순열 (marginal 보존·정체 파괴 —
          컬럼 추가 자체의 feature_fraction 추첨 효과를 통제. 판정자는 MASS−PLB)

평가 (사전 등록)
    ① LGBM 단독 Δ(MASS−PLB) — 가족이 학습되는가 (진단)
    ② 블렌드 Δ(MASS−PLB), α_mlp ∈ {0.85, 0.80, 0.75, 0.70, 0.65, 0.60} 격자 —
       투수 클러스터 쌍체 부트스트랩 B=2000.
    채택 후보 조건: 어떤 사전 등록 α 에서 **양 fold 양수 & P≥0.9** (α 6개는 강상관 —
    유효 검정 ~2개). 미달이면 §7.190 폐쇄가 재평형 채널까지 확정.
    통과해도 컬럼 경로 LB 전례(§7.56 4전 4패)를 명기하고 LB 슬롯은 별도 논의.

앵커 (하네스 검증): BASE 블렌드(α .75) f2023 645.1 / f2024 838.3 (§7.187 실측)

실행:  .venv/Scripts/python.exe -u src/massfeat3.py 2>&1 | tee experiments/massfeat3.log
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
import modelabel
import soften
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, N_ROUNDS,
                    TARGET_COL, TRAIN_CSV)
from data import load_csv
from validation import bss, shift_to_prior

CACHE = fc.CACHE
KEY = fc._key()
SEEDS3 = fc.LGB_SEEDS
B_BOOT = 2000
ALPHAS = (0.85, 0.80, 0.75, 0.70, 0.65, 0.60)
EXPECT_BLEND75 = {2023: 645.1, 2024: 838.3}          # §7.187 실측 앵커

# (새 컬럼 이름, a, b) — 차분 z(a) − z(b)
FAMILY = [
    ("mf_cd_hs",  "count_diff", "hand_same"),
    ("mf_cd_p5m", "count_diff", "asof_pitcher_prev5_game_middle_rate"),
    ("mf_bm_p5s", "asof_batter_middle_rate_sm",
     "asof_pitcher_prev5_game_success_rate"),
]


def zfit(Xtr, col):
    v = Xtr[col].to_numpy(dtype=np.float64)
    m = np.isfinite(v)
    mu = v[m].mean()
    sd = v[m].std()
    return mu, (sd if sd > 0 else 1.0)


def add_cols(Xtr, Xva, arm, rng):
    """train 적합 z 상수로 차분 3컬럼. PLB 는 b쪽을 split 안 행 순열."""
    Xtr, Xva = Xtr.copy(), Xva.copy()
    for nm, a, b in FAMILY:
        mua, sda = zfit(Xtr, a)
        mub, sdb = zfit(Xtr, b)
        for X in (Xtr, Xva):
            va = (X[a].to_numpy(dtype=np.float64) - mua) / sda
            vb = (X[b].to_numpy(dtype=np.float64) - mub) / sdb
            if arm == "PLB":
                vb = vb[rng.permutation(len(vb))]
            X[nm] = (va - vb).astype(np.float32)
    return Xtr, Xva


def train_arm(arm, fold, df, t0):
    f = CACHE / f"massfeat3_fold{fold}_{arm}_{KEY}.npz"
    if f.exists():
        with np.load(f) as z:
            print(f"  {arm} fold{fold} 재사용")
            return z["z"], dict(zip(z["snames"], z["splits"]))
    Xtr, ytr, w, Xva, yva, art, tr = fc.fold_data(df, fold)
    rng = np.random.default_rng(2000 + fold)         # 팔 공통 draw — 쌍체성 무관(순열은 PLB만)
    Xtr, Xva = add_cols(Xtr, Xva, arm, rng)
    aux, _ = modelabel.aux_labels(tr)
    Xtr, ytr, w, aux, _h = soften.soften(tr, Xtr, ytr, w, aux)
    print(f"  [{arm} fold{fold}] 학습 {len(Xtr):,}행 · 피처 {Xtr.shape[1]}")
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
                if nm.startswith("mf_"):
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
        for arm in ("MASS", "PLB"):
            Z[(arm, fold)], SP[(arm, fold)] = train_arm(arm, fold, df, t0)
        _, mva = fc.SPLITS[fold](s, df.game_month.to_numpy())
        va = df[mva]
        Y[fold] = va[TARGET_COL].to_numpy(dtype=np.float64)
        CL[fold] = pd.factorize(va["pitcher_id"].to_numpy())[0]
    del df
    gc.collect()

    def sc(z, y):
        return bss(y, shift_to_prior(1 / (1 + np.exp(-z)), y.mean()))

    print("\n" + "=" * 100)
    print("§7.190c 학습 경로 + α 격자 · 12부스터 · soften · B=%d" % B_BOOT)
    print("=" * 100)
    verdict = {}
    for fold in (2023, 2024):
        zl, zm, y = base[fold][0], base[fold][1], base[fold][2]
        assert np.allclose(y, Y[fold]), "캐시 y 불일치"
        b75 = sc(0.25 * zl + 0.75 * zm, y)
        assert abs(b75 - EXPECT_BLEND75[fold]) < 0.5, \
            f"앵커 실패 — BASE 블렌드 {b75:.1f} != {EXPECT_BLEND75[fold]}"
        cl, n_id = CL[fold], CL[fold].max() + 1
        ref = y.mean() * (1 - y.mean())
        n = np.bincount(cl, minlength=n_id)
        seen = np.flatnonzero(n > 0)
        rng = np.random.default_rng(42)
        picks = [rng.choice(seen, size=len(seen), replace=True)
                 for _ in range(B_BOOT)]

        def paired(pA, pB):
            SA = np.bincount(cl, weights=(pA - y) ** 2, minlength=n_id)
            SB = np.bincount(cl, weights=(pB - y) ** 2, minlength=n_id)
            D = np.empty(B_BOOT)
            for i, pk in enumerate(picks):
                D[i] = 1e5 * (SB[pk].sum() - SA[pk].sum()) / (n[pk].sum() * ref)
            return D

        print(f"\n  ── fold {fold} ──   (mf_ split 수: MASS {SP[('MASS', fold)]}"
              f"  PLB {SP[('PLB', fold)]})")
        print("  BASE 블렌드(α .75) %.1f — 앵커 %.1f OK" % (b75, EXPECT_BLEND75[fold]))
        pl = {a: shift_to_prior(1 / (1 + np.exp(-Z[(a, fold)])), y.mean())
              for a in ("MASS", "PLB")}
        for a in ("MASS", "PLB"):
            print("  %-6s LGBM 단독 %10.1f" % (a, sc(Z[(a, fold)], y)))
        D = paired(pl["MASS"], pl["PLB"])
        lo, hi = np.percentile(D, [2.5, 97.5])
        print(f"  ① LGBM 단독 Δ(MASS−PLB) {sc(Z[('MASS', fold)], y) - sc(Z[('PLB', fold)], y):+8.2f}"
              f"  [{lo:+7.2f}, {hi:+7.2f}]  P {np.mean(D > 0):.3f}")

        print("  ② 블렌드 Δ(MASS−PLB) — α_mlp 격자")
        best = (-1e9, None)
        for al in ALPHAS:
            pA = shift_to_prior(1 / (1 + np.exp(-((1 - al) * Z[("MASS", fold)]
                                                  + al * zm))), y.mean())
            pB = shift_to_prior(1 / (1 + np.exp(-((1 - al) * Z[("PLB", fold)]
                                                  + al * zm))), y.mean())
            pt = bss(y, pA) - bss(y, pB)
            D = paired(pA, pB)
            lo, hi = np.percentile(D, [2.5, 97.5])
            pos = float(np.mean(D > 0))
            mark = " ← 배포 α" if al == 0.75 else ""
            print(f"    α {al:.2f}  Δ {pt:+7.2f}  [{lo:+7.2f}, {hi:+7.2f}]"
                  f"  P {pos:.3f}{mark}")
            if pos > best[0]:
                best = (pos, al, pt)
        verdict[fold] = best

    print("\n" + "=" * 100)
    print("판정 (사전 등록: 어떤 α 에서든 블렌드 Δ(MASS−PLB) 양 fold 양수 & P≥0.9)")
    for fold, (pos, al, pt) in verdict.items():
        print(f"  fold{fold}: 최선 α {al:.2f}  Δ {pt:+.2f}  P {pos:.3f}")
    ok = all(v[0] >= 0.9 and v[2] > 0 for v in verdict.values())
    print("→ " + ("**후보 생존** — 단 컬럼 경로 LB 전례 4전 4패(§7.56) 명기, LB 슬롯 별도 논의"
                  if ok else
                  "미달 — §7.190 폐쇄가 블렌드 재평형 채널까지 확정"))
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
