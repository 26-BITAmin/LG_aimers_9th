"""§7.101 8차 — TabICL을 제대로 돌린 적이 있나 (사용자 지적)

지적
    "6번에서 TabICL은 안 해봄?"  — 맞다. raw/drop은 3·4차에서 봤지만 emb는 빠졌고,
    더 큰 구멍은 **TabICL을 한 번도 기본 설정으로 안 돌렸다**는 것이다.

    TabICLClassifier 기본값 n_estimators = 8.  내가 돌린 건 est1(속도용)과 est4뿐.
    앙상블을 1/8로 깎아놓고 "성능이 안 난다"고 판정한 셈이다.

그리고 지금까지 Brier만 봤다 — 두 고장을 구분 못 하는 자다
    순위가 나쁜 것       →  AUC가 0.5 근처
    확률이 과신된 것     →  AUC는 멀쩡한데 Brier만 폭락
    오라클 시프트는 **평균만** 맞추고 분포 폭은 안 건드리므로, TabICL이 확률을 0/1
    쪽으로 밀어놨다면 순위가 정상이어도 subBSS가 음수로 간다. 처방이 완전히 다르다
    (전자는 닫는다, 후자는 softmax_temperature로 고친다).
    → 이 프로브는 subBSS · **AUC** · **예측 표준편차**를 같이 찍는다.

팔 (전부 ctx10k · ID 드랍 · 위생 — 각 학습기의 4·6차 최적 형태)
    LGBM raw (기준) · TabPFN est1 (415.6 재현) ·
    TabICL est1 / est4 / **est8(기본값)** / est8+temp1.5 / ctx전체 est8

실행:  .venv-gpu/Scripts/python.exe -u src/iclfix.py 2>&1 | tee experiments/iclfix.log
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

# (이름, 종류, 컨텍스트, n_est, softmax_temperature)
ARMS = [
    ("TabPFN est1 (4차 재현)", "pfn", 10_000, 1, None),
    ("TabICL est1", "icl", 10_000, 1, 0.9),
    ("TabICL est4", "icl", 10_000, 4, 0.9),
    ("TabICL est8 ← 기본값", "icl", 10_000, 8, 0.9),
    ("TabICL est8 temp1.5", "icl", 10_000, 8, 1.5),
    ("TabICL ctx전체 est8", "icl", None, 8, 0.9),
]


def main():
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
    rng = np.random.default_rng(42)
    acc_i = rng.permutation(len(vaF))[:ACC_N]

    xtr = X(trF); cols = list(xtr.columns)
    Xtr = xtr.to_numpy(dtype=np.float32)
    Xq = X(vaF).iloc[acc_i].to_numpy(dtype=np.float32)
    ytr = trF[TARGET_COL].to_numpy(dtype=np.int64)
    ya = vaF[TARGET_COL].to_numpy(dtype=np.float64)[acc_i]
    ref = ya.mean() * (1 - ya.mean())

    keep = [i for i, c in enumerate(cols) if c not in ID_COLS]
    big = [j for j, i in enumerate(keep) if cols[i] in BIG_COLS]
    med = np.nanmedian(Xtr[:, keep], axis=0)

    def hyg(Z):
        W = Z[:, keep].copy()
        W[:, big] = np.log1p(np.maximum(np.nan_to_num(W[:, big], nan=0.0), 0.0))
        bad = ~np.isfinite(W)
        if bad.any():
            W[bad] = np.take(med, np.nonzero(bad)[1])
        return W

    Htr, Hq = hyg(Xtr), hyg(Xq)

    # ── 기준: LGBM F전문가 (원본 피처) ──────────────────────
    month = trF["game_month"].to_numpy(dtype=np.float64)
    cut = np.quantile(month, 0.8); itr, iva = month <= cut, month > cut
    ps = []
    for s in SEEDS:
        prm = dict(SMALL_LGB, seed=s, bagging_seed=s, feature_fraction_seed=s,
                   data_random_seed=s)
        b0 = lgb.train(prm, lgb.Dataset(Xtr[itr], ytr[itr].astype(np.float64)),
                       num_boost_round=3000,
                       valid_sets=[lgb.Dataset(Xtr[iva], ytr[iva].astype(np.float64))],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        b = lgb.train(prm, lgb.Dataset(Xtr, ytr.astype(np.float64)),
                      num_boost_round=max(50, b0.best_iteration))
        ps.append(1 / (1 + np.exp(-b.predict(Xq, raw_score=True))))
    base_raw = np.mean(ps, axis=0)
    base = shift_to_prior(base_raw, ya.mean())

    print(f"  {'팔':<26}{'subBSS':>9}{'AUC':>8}{'pred_sd':>9}"
          f"{'Δ vs LGBM':>12}{'SE':>8}{'t':>7}{'소요':>7}")
    print(f"  {'LGBM F전문가 (기준)':<26}{bss(ya, base):>9.1f}{auc(ya, base):>8.4f}"
          f"{base.std():>9.4f}{'—':>12}{'':>8}{'':>7}{'':>7}")

    for name, kind, ctx, n_est, temp in ARMS:
        Ztr, Zq = Htr, Hq
        if ctx is not None:
            i = np.random.default_rng(42).choice(len(Ztr), size=ctx, replace=False)
            Ztr, ysub = Ztr[i], ytr[i]
        else:
            ysub = ytr
        t0 = time.time()
        try:
            if kind == "pfn":
                from tabpfn import TabPFNClassifier
                c = TabPFNClassifier(device="cuda", n_estimators=n_est, random_state=42,
                                     inference_precision="autocast",
                                     ignore_pretraining_limits=True)
            else:
                from tabicl import TabICLClassifier
                c = TabICLClassifier(device="cuda", n_estimators=n_est, random_state=42,
                                     softmax_temperature=temp, verbose=False)
            c.fit(Ztr, ysub)
            p = c.predict_proba(Zq)[:, 1]
        except Exception as e:
            print(f"  {name:<26}  FAIL {type(e).__name__}: {str(e)[:48]}")
            continue
        el = time.time() - t0
        pv = shift_to_prior(p, ya.mean())
        d = (base - ya) ** 2 - (pv - ya) ** 2
        m = 1e5 * d.mean() / ref
        se = 1e5 * d.std(ddof=1) / np.sqrt(len(d)) / ref
        print(f"  {name:<26}{bss(ya, pv):>9.1f}{auc(ya, pv):>8.4f}{pv.std():>9.4f}"
              f"{m:>+12.1f}{se:>8.1f}{m/se:>7.2f}{el:>6.0f}s")
        del c


if __name__ == "__main__":
    main()
