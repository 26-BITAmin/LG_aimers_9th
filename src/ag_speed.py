"""AutoGluon 기반모델 — 컨텍스트 사다리 · 속도 · VRAM 상한 (§7.103 사전 점검 2/2)

§7.103 사전 점검 1/2(`src/ag_probe.py`)에서 확인된 것:
    · Mitra 학습·추론 경로 동작함 (fit+pred 226s, 가중치 다운로드 포함)
    · 실측 RAM +2.08GB — AutoGluon 추정 7.168GB는 3.4배 보수적이었다
    · VRAM 4.86GB @ ctx 3,000 (단, 40컬럼 합성 데이터 기준)
    · TabPFN-Mix는 메모리 가드에 막힘 → ag.max_memory_usage_ratio=12로 해제

이 스크립트가 답할 것 — 본 실험을 짤 수 있는가
    · ms/행 — fold2024 F행 30,010개 / fold2023H2 11,151개를 추론할 시간
    · **컨텍스트 상한** — 25,686행을 다 못 보면 몇 행까지 보는가

앞 버전 대비 고친 것
    ① 합성 40컬럼 → **실제 피처 행렬(78~79컬럼)**. 폭이 VRAM을 좌우한다
    ② 두 점(3k/10k) → **사다리**. 터지는 지점이 아니라 되는 상한을 찾는다
    ③ 백그라운드 tail 버퍼 의존 → experiments/ag_speed.log에 직접 기록.
       세션이 끊겨도 남는다 (2026-08-20에 이것 때문에 25분치를 잃었다)
    ④ LOKY_MAX_CPU_COUNT 고정 — Windows 11 26200에는 wmic.exe가 없어서
       joblib이 매 호출마다 트레이스백을 뱉는다. 그 소음이 ③의 원인이었다

실행:
    .venv-gpu/Scripts/python.exe -u src/ag_speed.py
"""
import gc
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")   # ④ wmic 부재 우회

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import truststore

truststore.inject_into_ssl()

import foldcache
from config import FSPEC_MIN_SEASON, SEED, TRAIN_CSV
from data import load_csv

LOG = Path(__file__).resolve().parents[1] / "experiments" / "ag_speed.log"
TMP = "C:/Users/강원우/AppData/Local/Temp"

N_PRED = 2000                                   # ms/행 측정용 고정 표본
CONTEXTS = [3000, 6000, 10000, 16000, 25686]    # 마지막이 F 학습행 전체
MODELS = ["MITRA", "TABPFNMIX"]
TARGET_ROWS = {"fold2024": 30010, "fold2023H2": 11151}


def emit(msg=""):
    """stdout과 로그 파일에 동시에. 세션이 끊겨도 파일은 남는다."""
    print(msg)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def pick_context(train_df, k, rng):
    """25,686행 중 k행을 골라 Mitra/TabPFN-Mix의 컨텍스트로 쓴다.

    ⚠️ 속도·VRAM 측정에는 **어느 행을 고르든 무관하다** — 그래서 여기서는
       무작위로 둔다. 본 실험에서는 이 선택이 성능을 좌우하므로 별도 결정이 필요하다.
       (TODO: 아래 '컨텍스트 선택' 논의 참조)
    """
    if k >= len(train_df):
        return train_df
    idx = rng.choice(len(train_df), size=k, replace=False)
    return train_df.iloc[np.sort(idx)].reset_index(drop=True)


def main():
    import torch
    from autogluon.tabular import TabularPredictor

    emit("=" * 78)
    emit(f"cuda={torch.cuda.is_available()}  "
         f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    free, total = torch.cuda.mem_get_info()
    emit(f"VRAM 전체 {total/1e9:.1f}GB  가용 {free/1e9:.1f}GB")

    df = load_csv(TRAIN_CSV)
    Xtr, ytr, _, Xva, yva, _, _ = foldcache.fold_data(df, 2024)
    mtr, mva = foldcache.SPLITS[2024](df.season.to_numpy(), df.game_month.to_numpy())
    tr, va = df[mtr], df[mva]

    mF = ((tr["game_type"].to_numpy().astype(str) == "F")
          & (tr["season"].to_numpy() >= FSPEC_MIN_SEASON))
    vF = va["game_type"].to_numpy().astype(str) == "F"

    full = Xtr[mF].reset_index(drop=True).copy()
    full["__y"] = ytr[mF].astype(int)
    query = Xva[vF].reset_index(drop=True).head(N_PRED)
    n_target = int(vF.sum())
    emit(f"F 학습 {full.shape}  추론 표본 {query.shape}  "
         f"(전체 추론 대상 {n_target:,}행)")

    # train.csv 전체 프레임(1.47M x 49)은 여기서 역할이 끝났다. F 부분집합만 16MB다.
    # 가용 RAM이 2GB 남짓이라 이걸 안 버리면 Mitra 적재 중 스와핑에 들어가고,
    # 그러면 측정하려는 대상(ms/행)이 바로 그 스와핑에 오염된다.
    del df, Xtr, Xva, ytr, yva, tr, va, mtr, mva, mF, vF
    gc.collect()
    import psutil
    emit(f"프레임 해제 후 가용 RAM {psutil.virtual_memory().available/1e9:.1f}GB")
    emit("")

    rng = np.random.default_rng(SEED)

    for name in MODELS:
        emit(f"=== {name} ===")
        for ctx in CONTEXTS:
            ctx_df = pick_context(full, ctx, rng)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                t0 = time.time()
                pred = TabularPredictor(
                    label="__y", problem_type="binary", eval_metric="log_loss",
                    verbosity=0, path=f"{TMP}/agsp_{name}_{ctx}")
                pred.fit(ctx_df,
                         hyperparameters={name: {"ag.max_memory_usage_ratio": 12.0}},
                         fit_weighted_ensemble=False, num_bag_folds=0,
                         num_stack_levels=0, ag_args_fit={"num_gpus": 1})
                t_fit = time.time() - t0

                t0 = time.time()
                _ = pred.predict_proba(query)[1].to_numpy()
                t_pred = time.time() - t0
            except Exception as e:
                emit(f"  ctx {ctx:>6,}  FAIL {type(e).__name__}: {str(e)[:120]}")
                torch.cuda.empty_cache()
                break                      # 사다리는 단조 — 터지면 위쪽도 터진다
            ms = t_pred / len(query) * 1e3
            vram = torch.cuda.max_memory_allocated() / 1e9
            proj = "  ".join(f"{k} {ms*v/1e3:,.0f}s" for k, v in TARGET_ROWS.items())
            emit(f"  ctx {ctx:>6,}  fit {t_fit:>5.0f}s  "
                 f"predict {ms:>6.1f}ms/행  VRAM {vram:>4.1f}GB  → {proj}")
        emit("")


if __name__ == "__main__":
    main()
