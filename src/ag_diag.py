"""가드 진단 — verbosity를 올려 실패의 진짜 문구를 잡는다. (§7.103 사전 점검 보조)

ag_speed.py가 verbosity=0으로 돌아 "No models were trained successfully"만 남겼다.
그건 AutoGluon이 모델을 건너뛴 뒤 던지는 껍데기 예외이고, 진짜 이유는 그 위에 찍힌다.
(2026-08-20 세션에서 이미 한 번 겪은 함정인데 그대로 반복했다.)

겸사겸사 컨텍스트 선택 결정에 필요한 사실 하나도 같이 찍는다 — F 학습행의 시즌 구성.
"""
import gc
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")
sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psutil
import truststore

truststore.inject_into_ssl()

import foldcache
from config import FSPEC_MIN_SEASON, SEED, TRAIN_CSV
from data import load_csv

TMP = "C:/Users/강원우/AppData/Local/Temp"
CASES = [("MITRA", 3000), ("TABPFNMIX", 6000)]


def main():
    from autogluon.tabular import TabularPredictor

    df = load_csv(TRAIN_CSV)
    Xtr, ytr, _, Xva, yva, _, _ = foldcache.fold_data(df, 2024)
    mtr, _ = foldcache.SPLITS[2024](df.season.to_numpy(), df.game_month.to_numpy())
    tr = df[mtr]
    mF = ((tr["game_type"].to_numpy().astype(str) == "F")
          & (tr["season"].to_numpy() >= FSPEC_MIN_SEASON))
    full = Xtr[mF].reset_index(drop=True).copy()
    full["__y"] = ytr[mF].astype(int)

    seasons = tr["season"].to_numpy()[mF]
    vc = pd.Series(seasons).value_counts().sort_index()
    print("SEASONS F 학습행 시즌 구성: " + "  ".join(f"{s}:{c:,}" for s, c in vc.items()))

    del df, Xtr, Xva, ytr, yva, tr, mtr, mF
    gc.collect()
    print(f"RAMFREE 가용 {psutil.virtual_memory().available/1e9:.2f}GB")

    rng = np.random.default_rng(SEED)
    for name, ctx in CASES:
        idx = np.sort(rng.choice(len(full), size=ctx, replace=False))
        sub = full.iloc[idx].reset_index(drop=True)
        print(f"\n===== {name} ctx={ctx} =====")
        try:
            p = TabularPredictor(label="__y", problem_type="binary",
                                 eval_metric="log_loss", verbosity=2,
                                 path=f"{TMP}/agdiag_{name}_{ctx}")
            p.fit(sub, hyperparameters={name: {"ag.max_memory_usage_ratio": 12.0}},
                  fit_weighted_ensemble=False, num_bag_folds=0,
                  num_stack_levels=0, ag_args_fit={"num_gpus": 1})
            print("  RESULT OK " + str(p.model_names()))
        except Exception as e:
            print(f"  RESULT FAIL {type(e).__name__}")


if __name__ == "__main__":
    main()
