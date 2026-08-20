"""§7.101 4차 — 닫기 전 확인 둘: 전체 컨텍스트 헤드룸 · 피처 위생

3차에서 나온 것
    속도  TabPFN ctx5k est1 = 9.08ms/행 → 263s. **예산 통과.**
    정확도 전부 LGBM F전문가(426.8)에게 패배. TabPFN ctx10k −63.5±91.7, TabICL −978±228.

닫기 전에 둘을 확인해야 한다.

  ① 컨텍스트가 불공평했다
     TabPFN subBSS가 컨텍스트에 단조 증가한다 (5k 296.1 → 10k 363.3). LGBM은
     25,686행을 **전부** 쓴 값이 426.8이다. 유일하게 이길 수 있는 팔(전체 컨텍스트)이
     하필 2단계에서 시간 때문에 생략됐다. 예산과 무관한 헤드룸 질문이니 한 번은 잰다.

  ② TabICL −551.4는 기준선(0)보다 나쁘다 — 내 설정을 먼저 의심한다
     피처 점검: 76열 중 16열에 NaN(6,944개), 그리고 asof_pitcher_n(10,015)·
     asof_batter_n(10,879)·asof_pitcher_pitchmix_n(10,015) 세 열만 4자릿수인데
     나머지는 대부분 [0,1] 비율이다. 거기에 pitcher_id/batter_id가 2만대 생정수로
     들어간다. TabICL은 outlier_threshold=4.0 자체 전처리를 쓰므로 이 스케일 차이에
     취약할 수 있다. LGBM은 분기로 격리하니 같은 입력에도 멀쩡하다.

     정제(clean) = ID 2열 드랍 + 카운트 3열 log1p + NaN을 컨텍스트 중앙값으로 대치
     → 두 학습기에 **똑같이** 적용해서 쌍체로 본다. 한쪽만 고치면 비교가 깨진다.

실행:  .venv-gpu/Scripts/python.exe -u src/icl_probe2.py 2>&1 | tee experiments/icl_probe2.log
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
import torch
import truststore

truststore.inject_into_ssl()

from config import (CONTAM_AXES, DROP_FEATURES, RECENCY_HALF_LIFE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import bss, shift_to_prior

ACC_N = 8000
SEEDS = (42, 7, 2024)
BIG_COLS = ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n")
ID_COLS = ("pitcher_id", "batter_id")

# (이름, 종류, 컨텍스트, n_est, 정제여부)   ※ 무거운 팔을 뒤에 둔다
ARMS = [
    ("TabICL ctx전체 est4 · 원본", "icl", None, 4, False),
    ("TabICL ctx전체 est4 · 정제", "icl", None, 4, True),
    ("TabICL ctx10k  est1 · 정제", "icl", 10_000, 1, True),
    ("TabPFN ctx10k  est1 · 원본", "pfn", 10_000, 1, False),
    ("TabPFN ctx10k  est1 · 정제", "pfn", 10_000, 1, True),
    ("TabPFN ctx전체 est1 · 원본", "pfn", None, 1, False),   # ← ① 헤드룸. ~18분
]

SMALL_LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15,
                 min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7,
                 bagging_freq=1, lambda_l2=10.0, max_bin=127, num_threads=6,
                 verbosity=-1)


def make_clf(kind, n_est):
    if kind == "pfn":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device="cuda", n_estimators=n_est,
                                inference_precision="autocast", random_state=42,
                                ignore_pretraining_limits=True)
    from tabicl import TabICLClassifier
    return TabICLClassifier(device="cuda", n_estimators=n_est, random_state=42,
                            kv_cache=False, verbose=False)


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
    xtr = X(trF)
    cols = list(xtr.columns)
    Xtr = xtr.to_numpy(dtype=np.float32)
    ytr = trF[TARGET_COL].to_numpy(dtype=np.int64)
    XvaF = X(vaF).to_numpy(dtype=np.float32)
    yvaF = vaF[TARGET_COL].to_numpy(dtype=np.float64)

    keep = [i for i, c in enumerate(cols) if c not in ID_COLS]
    big = [j for j, i in enumerate(keep) if cols[i] in BIG_COLS]
    med = np.nanmedian(Xtr[:, keep], axis=0)          # 대치값은 **학습셋에서만**

    def clean(Z):
        Z = Z[:, keep].copy()
        Z[:, big] = np.log1p(np.maximum(np.nan_to_num(Z[:, big], nan=0.0), 0.0))
        bad = ~np.isfinite(Z)
        if bad.any():
            Z[bad] = np.take(med, np.nonzero(bad)[1])
        return Z

    rng = np.random.default_rng(42)
    acc_i = rng.permutation(len(XvaF))[:ACC_N]
    ya = yvaF[acc_i]
    print(f"컨텍스트 {Xtr.shape} → 정제 후 {clean(Xtr[:5]).shape[1]}열   평가 {ACC_N:,}행")

    # ── LGBM F전문가 기준선 (원본 피처. 이게 현행 후보다) ──
    t0 = time.time()
    month = trF["game_month"].to_numpy(dtype=np.float64)
    cut = np.quantile(month, 0.8)
    itr, iva = month <= cut, month > cut
    pl = []
    for s in SEEDS:
        prm = dict(SMALL_LGB, seed=s, bagging_seed=s, feature_fraction_seed=s,
                   data_random_seed=s)
        b0 = lgb.train(prm, lgb.Dataset(Xtr[itr], ytr[itr].astype(np.float64)),
                       num_boost_round=3000,
                       valid_sets=[lgb.Dataset(Xtr[iva], ytr[iva].astype(np.float64))],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        b = lgb.train(prm, lgb.Dataset(Xtr, ytr.astype(np.float64)),
                      num_boost_round=max(50, b0.best_iteration))
        pl.append(1 / (1 + np.exp(-b.predict(XvaF[acc_i], raw_score=True))))
    pL = shift_to_prior(np.mean(pl, axis=0), ya.mean())
    ref = ya.mean() * (1 - ya.mean())
    print(f"  [기준] LGBM F전문가 (원본 피처)  subBSS {bss(ya, pL):>8.1f}  ({time.time()-t0:.0f}s)\n")

    print(f"  {'설정':<28}{'subBSS':>9}{'Δ vs LGBM':>12}{'쌍체SE':>9}{'t':>7}{'소요':>8}")
    for name, kind, ctx, n_est, do_clean in ARMS:
        Xc, yc = Xtr, ytr
        Xq = XvaF[acc_i]
        if ctx is not None:
            i = np.random.default_rng(42).choice(len(Xtr), size=ctx, replace=False)
            Xc, yc = Xtr[i], ytr[i]
        if do_clean:
            Xc, Xq = clean(Xc), clean(Xq)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.time()
            clf = make_clf(kind, n_est)
            clf.fit(Xc, yc)
            p = clf.predict_proba(Xq)[:, 1]
            el = time.time() - t0
        except Exception as e:
            print(f"  {name:<28}  FAIL {type(e).__name__}: {str(e)[:50]}")
            continue
        ps = shift_to_prior(p, ya.mean())
        d = (pL - ya) ** 2 - (ps - ya) ** 2            # 양수 = 모델이 LGBM보다 낫다
        m, se = 1e5 * d.mean() / ref, 1e5 * d.std(ddof=1) / np.sqrt(len(d)) / ref
        print(f"  {name:<28}{bss(ya, ps):>9.1f}{m:>+12.1f}{se:>9.1f}{m/se:>7.2f}{el:>7.0f}s")
        del clf


if __name__ == "__main__":
    main()
