"""TabPFN v2 vs TabICL v2 — 추론 예산 통과 여부 + 정확도 (§7.101 3차 프로브)

앞선 두 프로브에서 배운 것
    1차: TabPFN 기본값(ctx 25,686 · est4)은 220ms/행. 상한은 19.8ms/행 → 11배 초과.
    2차: `fit_mode="fit_with_cache"`는 KV를 통째로 물리화한다 — ctx5k에 15GB,
         ctx10k에 30GB 요구. 8GB OOM이고 L4 22.4GB로도 ctx10k는 못 올린다. 기본값을 쓴다.
    2차: **1,500행 subBSS는 노이즈가 ±2,600**이라 정확도 비교에 못 쓴다
         (LGBM 전문가가 −11.2로 나왔지만 30,010행 전체에선 537.1).

그래서 이 프로브는 두 단계로 나눈다
    1단계 속도   400행이면 ms/행은 충분히 잰다. 여기서 예산 통과 여부만 판정.
    2단계 정확도 통과한 설정만 8,000행. 절대 subBSS가 아니라 **LGBM 전문가와의
                 쌍체 ΔBrier**로 본다 — 두 예측이 비슷하면 차이의 SE가 훨씬 작다.

라이선스 (전부 게이팅 없음, 계정 불필요)
    TabPFN v2  Prior-Labs/TabPFN-v2-clf      Prior Labs License 1.1 = Apache 2.0 + 표시
    TabICL v2  jingang/TabICL (기본 체크포인트) BSD-3-Clause

실행:  .venv-gpu/Scripts/python.exe -u src/icl_probe.py 2>&1 | tee experiments/icl_probe.log
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

SPEED_N = 400
ACC_N = 8000
TEST_F_ROWS = 29_000
BUDGET_S = 573.0
ACC_MAX_S = 420.0          # 2단계에 이보다 오래 걸릴 설정은 정확도 측정 생략
SEEDS = (42, 7, 2024)

# (이름, 종류, 컨텍스트, n_estimators)
CONFIGS = [
    ("TabICL  ctx10k  est1", "icl", 10_000, 1),
    ("TabICL  ctx전체 est1", "icl", None, 1),
    ("TabICL  ctx전체 est4", "icl", None, 4),
    ("TabPFN  ctx5k   est1", "pfn", 5_000, 1),
    ("TabPFN  ctx10k  est1", "pfn", 10_000, 1),
    ("TabPFN  ctx전체 est1", "pfn", None, 1),
]

SMALL_LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15,
                 min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7,
                 bagging_freq=1, lambda_l2=10.0, max_bin=127, num_threads=6,
                 verbosity=-1)


def make_clf(kind, ctx, n_est):
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
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    trF = tr[(tr.season == 2023) & (tr.game_type.astype(str) == "F")]
    vaF = va[va.game_type.astype(str) == "F"]
    Xtr = X(trF).to_numpy(dtype=np.float32)
    ytr = trF[TARGET_COL].to_numpy(dtype=np.int64)
    XvaF = X(vaF).to_numpy(dtype=np.float32)
    yvaF = vaF[TARGET_COL].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(42)
    order = rng.permutation(len(XvaF))
    acc_i = order[:ACC_N]                 # 정확도용 고정 행
    spd_i = order[:SPEED_N]               # 속도용 (정확도 집합의 앞부분)
    ya = yvaF[acc_i]
    print(f"컨텍스트 풀 {Xtr.shape}   속도 {SPEED_N}행 / 정확도 {ACC_N}행")
    print(f"예산 {BUDGET_S:.0f}s ÷ {TEST_F_ROWS:,}행 = {BUDGET_S/TEST_F_ROWS*1e3:.1f} ms/행\n")

    # ── LGBM F전문가 기준선 (정확도 행 전체, 학습셋 내부 early stop) ──
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
    pL = np.mean(pl, axis=0)
    pL_s = shift_to_prior(pL, ya.mean())
    print(f"  [기준] LGBM F전문가  subBSS {bss(ya, pL_s):>8.1f}  ({time.time()-t0:.0f}s)\n")

    # ── 1단계: 속도 ────────────────────────────────────────
    print(f"  {'설정':<24}{'fit':>7}{'ms/행':>9}{'test환산':>10}{'판정':>8}{'VRAM':>8}")
    speed = {}
    for name, kind, ctx, n_est in CONFIGS:
        Xc, yc = (Xtr, ytr) if ctx is None else _sub(Xtr, ytr, ctx, rng)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.time(); clf = make_clf(kind, ctx, n_est); clf.fit(Xc, yc)
            t_fit = time.time() - t0
            t0 = time.time(); clf.predict_proba(XvaF[spd_i]); t_pred = time.time() - t0
        except Exception as e:
            print(f"  {name:<24}  FAIL {type(e).__name__}: {str(e)[:52]}")
            continue
        ms = t_pred / SPEED_N * 1e3
        est = ms * TEST_F_ROWS / 1e3
        speed[name] = (ms, est, kind, ctx, n_est)
        print(f"  {name:<24}{t_fit:>6.1f}s{ms:>9.2f}{est:>9.0f}s"
              f"{('OK' if est <= BUDGET_S else f'{est/BUDGET_S:.1f}x'):>8}"
              f"{torch.cuda.max_memory_allocated()/1e9:>7.1f}G")
        del clf

    # ── 2단계: 정확도 (감당 가능한 것만) ────────────────────
    print(f"\n  2단계 — {ACC_N:,}행. LGBM 전문가와 **쌍체** 비교")
    print(f"  {'설정':<24}{'subBSS':>9}{'Δ vs LGBM':>12}{'쌍체SE':>9}{'소요':>8}")
    for name, (ms, est, kind, ctx, n_est) in sorted(speed.items(), key=lambda kv: kv[1][0]):
        if ms * ACC_N / 1e3 > ACC_MAX_S:
            print(f"  {name:<24}  (생략 — {ms*ACC_N/1e3:.0f}s 예상)")
            continue
        Xc, yc = (Xtr, ytr) if ctx is None else _sub(Xtr, ytr, ctx, np.random.default_rng(42))
        torch.cuda.empty_cache()
        try:
            t0 = time.time()
            clf = make_clf(kind, ctx, n_est); clf.fit(Xc, yc)
            p = clf.predict_proba(XvaF[acc_i])[:, 1]
            el = time.time() - t0
        except Exception as e:
            print(f"  {name:<24}  FAIL {type(e).__name__}: {str(e)[:52]}")
            continue
        ps = shift_to_prior(p, ya.mean())
        d = (pL_s - ya) ** 2 - (ps - ya) ** 2          # 양수 = 모델이 LGBM보다 낫다
        ref = ya.mean() * (1 - ya.mean())
        print(f"  {name:<24}{bss(ya, ps):>9.1f}{1e5*d.mean()/ref:>+12.1f}"
              f"{1e5*d.std(ddof=1)/np.sqrt(len(d))/ref:>9.1f}{el:>7.0f}s")
        del clf


def _sub(X, y, n, rng):
    i = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return X[i], y[i]


if __name__ == "__main__":
    main()
