"""§7.101 9차 — 공정한 대결 (사용자 지적: "설정이 기존 LGBM하고 다른 게 뭐지")

지금까지의 비교는 세 군데에서 LGBM에 유리했다.

    ①  학습 행     LGBM 25,686 전부   vs  TabPFN 10,000 서브샘플
                   (원본 피처 기준 ctx10k 363.3 → ctx전체 409.5, +46.2로 이미 확인)
    ②  앙상블      LGBM 시드 3개 평균 vs  TabPFN n_estimators=1
                   (TabICL에서 est1 → est8이 +406을 줬다. 같은 축을 TabPFN엔 안 썼다)
    ③  NaN         LGBM 네이티브 처리 vs  트랜스포머만 중앙값 대치
                   (§7.92 — 결측은 이 데이터에서 정보다)

그리고 **TabPFN의 최선 조합이 한 번도 실행된 적이 없다**:
    전체 컨텍스트 + ID 드랍 + 위생 + est8.
    측정된 건 (전체·원본·est1) 409.5 와 (10k·드랍·est1) 415.6 뿐 — 장점의 교집합이 공백.

TabPFN의 n_estimators는 서로 다른 전처리/순열 구성의 앙상블이라 LGBM 시드 평균과
같은 역할을 한다. est8 한 번이 시드 3개 평균의 정당한 대응물이다.

③은 이 프로브에서 **대치 없이** 두는 팔로 확인한다 — TabPFN은 NaN을 네이티브로
받으므로(문서상) 대치가 오히려 손해였을 수 있다.

팔 (전부 8,000행 · 같은 행 · 쌍체)
    LGBM F전문가 raw 3시드          기준 (426.8)
    LGBM F전문가 drop 3시드         6차 최고 (438.3) — 이것도 같이 놓는다
    TabPFN 전체ctx·드랍·위생·est1   ← ①만 고침
    TabPFN 전체ctx·드랍·위생·est4
    TabPFN 전체ctx·드랍·위생·est8   ← ①②를 다 고친 진짜 최선
    TabPFN 전체ctx·드랍·NaN유지·est8 ← ③까지 고침

실행:  .venv-gpu/Scripts/python.exe -u src/fairfight.py 2>&1 | tee experiments/fairfight.log
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
import truststore

truststore.inject_into_ssl()

from config import (CONTAM_AXES, DROP_FEATURES, RECENCY_HALF_LIFE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import auc, bss, shift_to_prior

ACC_N = 8000
SEEDS = (42, 7, 2024)
BIG_COLS = ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n")
ID_COLS = ("pitcher_id", "batter_id")

SMALL_LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15,
                 min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7,
                 bagging_freq=1, lambda_l2=10.0, max_bin=127, num_threads=6,
                 verbosity=-1)

# (이름, n_estimators, NaN 대치 여부)  — 전부 전체 컨텍스트 · ID 드랍 · log1p
PFN_ARMS = [("TabPFN 전체ctx est1", 1, True),
            ("TabPFN 전체ctx est4", 4, True),
            ("TabPFN 전체ctx est8 ★", 8, True),
            ("TabPFN 전체ctx est8 NaN유지", 8, False)]


def main():
    from tabpfn import TabPFNClassifier

    df = load_csv(TRAIN_CSV)
    tr = df[df.season < 2024]
    va = df[df.season == 2024]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=2023)
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ID_COLS:
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    trF = tr[(tr.season == 2023) & (tr.game_type.astype(str) == "F")]
    vaF = va[va.game_type.astype(str) == "F"]
    acc_i = np.random.default_rng(42).permutation(len(vaF))[:ACC_N]

    xtr = X(trF); cols = list(xtr.columns)
    Xtr = xtr.to_numpy(dtype=np.float32)
    Xq = X(vaF).iloc[acc_i].to_numpy(dtype=np.float32)
    ytr = trF[TARGET_COL].to_numpy(dtype=np.int64)
    ya = vaF[TARGET_COL].to_numpy(dtype=np.float64)[acc_i]
    ref = ya.mean() * (1 - ya.mean())

    keep = [i for i, c in enumerate(cols) if c not in ID_COLS]
    big = [j for j, i in enumerate(keep) if cols[i] in BIG_COLS]
    med = np.nanmedian(Xtr[:, keep], axis=0)

    def prep(Z, impute):
        W = Z[:, keep].copy()
        W[:, big] = np.log1p(np.maximum(np.nan_to_num(W[:, big], nan=0.0), 0.0))
        if impute:
            bad = ~np.isfinite(W)
            if bad.any():
                W[bad] = np.take(med, np.nonzero(bad)[1])
        return W

    month = trF["game_month"].to_numpy(dtype=np.float64)
    cut = np.quantile(month, 0.8); itr, iva = month <= cut, month > cut

    def lgb_run(Ztr, Zq):
        ps = []
        for s in SEEDS:
            prm = dict(SMALL_LGB, seed=s, bagging_seed=s, feature_fraction_seed=s,
                       data_random_seed=s)
            b0 = lgb.train(prm, lgb.Dataset(Ztr[itr], ytr[itr].astype(np.float64)),
                           num_boost_round=3000,
                           valid_sets=[lgb.Dataset(Ztr[iva], ytr[iva].astype(np.float64))],
                           callbacks=[lgb.early_stopping(100, verbose=False)])
            b = lgb.train(prm, lgb.Dataset(Ztr, ytr.astype(np.float64)),
                          num_boost_round=max(50, b0.best_iteration))
            ps.append(1 / (1 + np.exp(-b.predict(Zq, raw_score=True))))
        return np.mean(ps, axis=0)

    base = shift_to_prior(lgb_run(Xtr, Xq), ya.mean())
    drop_p = shift_to_prior(lgb_run(Xtr[:, keep], Xq[:, keep]), ya.mean())

    def row(name, p, el=None):
        d = (base - ya) ** 2 - (p - ya) ** 2
        m = 1e5 * d.mean() / ref
        se = 1e5 * d.std(ddof=1) / np.sqrt(len(d)) / ref
        t = f"{m/se:.2f}" if se > 0 else ""
        print(f"  {name:<30}{bss(ya, p):>9.1f}{auc(ya, p):>8.4f}{p.std():>9.4f}"
              f"{m:>+11.1f}{se:>8.1f}{t:>7}{(f'{el:.0f}s' if el else ''):>8}")

    print(f"  {'팔':<30}{'subBSS':>9}{'AUC':>8}{'pred_sd':>9}{'Δ vs 기준':>11}{'SE':>8}{'t':>7}{'소요':>8}")
    print(f"  {'LGBM raw 3시드 (기준)':<30}{bss(ya, base):>9.1f}{auc(ya, base):>8.4f}"
          f"{base.std():>9.4f}{'—':>11}")
    row("LGBM drop 3시드", drop_p)

    for name, n_est, impute in PFN_ARMS:
        Ztr, Zq = prep(Xtr, impute), prep(Xq, impute)
        t0 = time.time()
        try:
            c = TabPFNClassifier(device="cuda", n_estimators=n_est, random_state=42,
                                 inference_precision="autocast",
                                 ignore_pretraining_limits=True)
            c.fit(Ztr, ytr)                       # ← 전체 25,686행. 서브샘플 없음
            p = c.predict_proba(Zq)[:, 1]
        except Exception as e:
            print(f"  {name:<30}  FAIL {type(e).__name__}: {str(e)[:44]}")
            continue
        row(name, shift_to_prior(p, ya.mean()), time.time() - t0)
        del c


if __name__ == "__main__":
    main()
