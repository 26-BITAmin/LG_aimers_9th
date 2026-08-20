"""TabPFN 타당성 프로브 — 돌아가는가 / 얼마나 걸리는가 (§7.101 사전 점검)

측정할 것 세 가지. 본 실험을 짜기 전에 이것부터.
    1. fold2024의 F 전문가 학습셋(2023 F = 25,686행)이 컨텍스트로 들어가는가
       (사전학습 한도 10k 초과 → ignore_pretraining_limits 필요)
    2. 검증 F 30,010행 추론에 몇 초 걸리는가 → 제출 10분 예산 환산
    3. 8.55GB VRAM에서 OOM 없이 도는가

실행:  .venv-gpu/Scripts/python.exe -u src/tabpfn_probe.py 2>&1 | tee experiments/tabpfn_probe.log
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
import torch
import truststore

# 이 PC는 백신/기업 CA가 TLS를 가로채고 있어 OpenSSL 기본 검증이 깨진다
# ("Basic Constraints of CA cert not marked critical"). truststore는 Windows
# 검증기를 쓰게 해서 이를 우회한다. tabpfn은 이 실패를 "gated repo"로 오보한다.
truststore.inject_into_ssl()

from config import (CONTAM_AXES, DROP_FEATURES, RECENCY_HALF_LIFE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import bss, brier, shift_to_prior

N_PROBE = 3000          # 추론 타이밍용 부분집합 (전체는 이 값으로 환산)


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
    Xva = X(vaF).to_numpy(dtype=np.float32)
    yva = vaF[TARGET_COL].to_numpy(dtype=np.float64)
    print(f"학습 컨텍스트 {Xtr.shape}  검증 {Xva.shape}  양성률 {ytr.mean():.4f}/{yva.mean():.4f}")

    t0 = time.time()
    clf = TabPFNClassifier(device="cuda", ignore_pretraining_limits=True,
                           random_state=42)
    clf.fit(Xtr, ytr)
    t_fit = time.time() - t0
    print(f"fit  {t_fit:.1f}s   n_estimators={getattr(clf, 'n_estimators_', '?')}")
    print(f"  VRAM peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    t0 = time.time()
    p = clf.predict_proba(Xva[:N_PROBE])[:, 1]
    t_pred = time.time() - t0
    per_row = t_pred / N_PROBE
    print(f"predict {N_PROBE}행 {t_pred:.1f}s  →  행당 {per_row*1e3:.2f}ms")
    print(f"  VRAM peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"  환산: 검증 F 전체 {len(Xva):,}행 = {per_row*len(Xva):.0f}s")
    print(f"  환산: test F 추정 27,000행 = {per_row*27000:.0f}s  (제출 예산 600s)")

    ps = shift_to_prior(p, yva[:N_PROBE].mean())
    print(f"\n  [참고] 프로브 {N_PROBE}행 subBSS(오라클 시프트) = {bss(yva[:N_PROBE], ps):.1f}")
    print(f"         raw pred mean {p.mean():.4f} vs true {yva[:N_PROBE].mean():.4f}")


if __name__ == "__main__":
    main()
