"""TabPFN v2 — 추론 예산 안에 들어가는 설정 탐색 (§7.101 2차 프로브)

1차 프로브 결과: 컨텍스트 25,686 · n_estimators=4 기본값에서 **220ms/행 초과**.
필요 상한은 19.6ms/행이다 —

    전체 추론 예산 600s  −  현행 파이프라인 ~27s  =  573s
    test의 F 추정 29,000행                        →  19.6 ms/행

손잡이 넷을 조합해 그 아래로 내려가는지, 내려가면 정확도를 얼마나 잃는지 잰다.
    · 컨텍스트 서브샘플 (25,686 → 10,000 / 5,000)  ← 사전학습 한도 안으로
    · n_estimators (기본 4 → 1~2)
    · fit_mode="fit_with_cache"  ← 고정 컨텍스트의 KV를 1회만 계산
    · inference_precision="autocast" (half)

정확도는 **고정된 1,500행**에서만 비교한다 (팔 간 쌍체). LGBM F전문가를 같은
행에서 재학습해 대조군으로 둔다 — 결정에 필요한 비교는 "TabPFN vs LGBM 전문가"다.

⚠️ 로컬은 RTX 5060 8GB, 평가서버는 L4 22.4GB다. 8GB에서는 memory_saving_mode가
   잘게 쪼개므로 여기 숫자는 **비관적**이다. 다만 11배 격차를 메울 정도는 아니다.

실행:  .venv-gpu/Scripts/python.exe -u src/tabpfn_probe2.py 2>&1 | tee experiments/tabpfn_probe2.log
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

truststore.inject_into_ssl()          # 로컬 CA 가로채기 우회 (probe 주석 참조)

from config import (CONTAM_AXES, DROP_FEATURES, RECENCY_HALF_LIFE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import bss, shift_to_prior

N_PROBE = 1500            # 속도·정확도 공용 고정 행수
TEST_F_ROWS = 29_000      # test의 F 추정치 (245,789 × 11.8%)
BUDGET_S = 573.0          # 600s − 현행 파이프라인 27s
SKIP_AFTER_S = 300.0      # 한 팔이 이보다 오래 걸리면 이후 팔은 건너뛴다

# (이름, 컨텍스트 크기, n_estimators, fit_mode, precision)
CONFIGS = [
    ("ctx5k  · est1 · cache · half", 5_000, 1, "fit_with_cache", "autocast"),
    ("ctx10k · est1 · cache · half", 10_000, 1, "fit_with_cache", "autocast"),
    ("ctx10k · est2 · cache · half", 10_000, 2, "fit_with_cache", "autocast"),
    ("ctx전체 · est1 · cache · half", None, 1, "fit_with_cache", "autocast"),
]

SMALL_LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15,
                 min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7,
                 bagging_freq=1, lambda_l2=10.0, max_bin=127, num_threads=6,
                 verbosity=-1)


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
    pi = rng.choice(len(XvaF), size=N_PROBE, replace=False)   # 고정 평가 행
    Xp, yp = XvaF[pi], yvaF[pi]
    print(f"컨텍스트 풀 {Xtr.shape}   평가 고정 {N_PROBE}행 (양성률 {yp.mean():.4f})")
    print(f"예산: {BUDGET_S:.0f}s / {TEST_F_ROWS:,}행 = {BUDGET_S/TEST_F_ROWS*1e3:.1f} ms/행\n")

    # ── 대조군: LGBM F전문가 (같은 1,500행) ──────────────────
    t0 = time.time()
    ps = []
    for s in (42, 7, 2024):
        prm = dict(SMALL_LGB, seed=s, bagging_seed=s, feature_fraction_seed=s,
                   data_random_seed=s)
        b = lgb.train(prm, lgb.Dataset(Xtr, ytr.astype(np.float64)), num_boost_round=56)
        ps.append(1 / (1 + np.exp(-b.predict(Xp, raw_score=True))))
    lgb_bss = np.mean([bss(yp, shift_to_prior(p, yp.mean())) for p in ps])
    print(f"  [대조] LGBM F전문가 3시드     subBSS {lgb_bss:>8.1f}   ({time.time()-t0:.1f}s)\n")

    print(f"  {'설정':<30}{'fit':>7}{'ms/행':>9}{'test환산':>10}{'예산':>7}{'subBSS':>9}")
    for name, ctx, n_est, mode, prec in CONFIGS:
        if ctx is None:
            Xc, yc = Xtr, ytr
        else:
            idx = rng.choice(len(Xtr), size=min(ctx, len(Xtr)), replace=False)
            Xc, yc = Xtr[idx], ytr[idx]
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.time()
            clf = TabPFNClassifier(device="cuda", n_estimators=n_est, fit_mode=mode,
                                   inference_precision=prec, random_state=42,
                                   ignore_pretraining_limits=(ctx is None or ctx > 10_000))
            clf.fit(Xc, yc)
            t_fit = time.time() - t0
            t0 = time.time()
            p = clf.predict_proba(Xp)[:, 1]
            t_pred = time.time() - t0
        except Exception as e:
            print(f"  {name:<30}  FAIL {type(e).__name__}: {str(e)[:60]}")
            continue
        ms = t_pred / N_PROBE * 1e3
        est = ms * TEST_F_ROWS / 1e3
        ok = "OK" if est <= BUDGET_S else f"{est/BUDGET_S:.1f}x"
        b = bss(yp, shift_to_prior(p, yp.mean()))
        print(f"  {name:<30}{t_fit:>6.1f}s{ms:>9.1f}{est:>9.0f}s{ok:>7}{b:>9.1f}"
              f"   VRAM {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
        del clf
        if t_pred > SKIP_AFTER_S:
            print("  → 이 팔이 너무 느리다. 남은(더 무거운) 팔은 건너뛴다.")
            break


if __name__ == "__main__":
    main()
