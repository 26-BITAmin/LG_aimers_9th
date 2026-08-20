"""AutoGluon 기반모델 타당성 프로브 — API·속도 (§7.103 사전 점검)

왜 이 둘인가 (사용자 지적 두 가지를 반영)
    ① 라이선스: Mitra·TabPFN-Mix는 **Apache 2.0**이다. 출력물 사용에 제한이 없으므로
       증류가 가능하고, 그러면 추론 경로에 torch가 안 들어간다 — §7.101에서 TabPFN을
       닫은 이유(3.5GB 설치 위험)가 사라진다.
    ② 해상도: §7.101의 ±90은 내가 **8,000행으로 자른 결과**다. fold2024의 F 행
       30,010개를 다 쓰면 ±46이고, fold2023H2의 11,151행이 두 번째 창이 된다.
       해상도 부족을 이유로 닫는 건 자초한 것을 근거로 삼는 것이었다.

이 프로브가 답할 것 — 본 실험을 짜기 전에
    · AutoGluon 단일 모델을 우리 행렬로 직접 학습·예측하는 경로가 되는가
    · GPU를 쓰는가
    · 25,686행 컨텍스트 → 30,010행 추론이 현실적인 시간인가

실행:  .venv-gpu/Scripts/python.exe -u src/ag_probe.py 2>&1 | tee experiments/ag_probe.log
"""
import sys
import time
import warnings
from pathlib import Path

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
from config import FSPEC_MIN_SEASON, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import auc, bss, shift_to_prior

N_PRED = 3000        # 속도 측정용. 전체는 이 값으로 환산한다
MODELS = ["MITRA", "TABPFNMIX"]


def main():
    import torch
    from autogluon.tabular import TabularPredictor

    print(f"cuda={torch.cuda.is_available()}  "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    df = load_csv(TRAIN_CSV)
    Xtr, ytr, _, Xva, yva, _, _ = foldcache.fold_data(df, 2024)
    mtr, mva = foldcache.SPLITS[2024](df.season.to_numpy(), df.game_month.to_numpy())
    tr, va = df[mtr], df[mva]

    mF = ((tr["game_type"].to_numpy().astype(str) == "F")
          & (tr["season"].to_numpy() >= FSPEC_MIN_SEASON))
    vF = va["game_type"].to_numpy().astype(str) == "F"

    train_df = Xtr[mF].reset_index(drop=True).copy()
    train_df["__y"] = ytr[mF].astype(int)
    test_df = Xva[vF].reset_index(drop=True)
    yv = yva[vF]
    print(f"학습 {train_df.shape}  추론 {test_df.shape}  양성률 "
          f"{train_df['__y'].mean():.4f}/{yv.mean():.4f}")

    for name in MODELS:
        print(f"\n=== {name} ===")
        t0 = time.time()
        try:
            pred = TabularPredictor(label="__y", problem_type="binary",
                                    eval_metric="log_loss", verbosity=0,
                                    path=f"C:/Users/강원우/AppData/Local/Temp/ag_{name}")
            pred.fit(train_df, hyperparameters={name: {}},
                     fit_weighted_ensemble=False, num_bag_folds=0,
                     num_stack_levels=0, ag_args_fit={"num_gpus": 1})
            t_fit = time.time() - t0
            print(f"  fit {t_fit:.0f}s   모델 {pred.model_names()}")
            t0 = time.time()
            p = pred.predict_proba(test_df.head(N_PRED))[1].to_numpy()
            t_pred = time.time() - t0
        except Exception as e:
            print(f"  FAIL {type(e).__name__}: {str(e)[:200]}")
            continue
        ms = t_pred / N_PRED * 1e3
        ps = shift_to_prior(p, yv[:N_PRED].mean())
        print(f"  predict {N_PRED}행 {t_pred:.0f}s = {ms:.1f}ms/행"
              f"   → 전체 {len(test_df):,}행 환산 {ms*len(test_df)/1e3:.0f}s")
        print(f"  [참고] {N_PRED}행 subBSS {bss(yv[:N_PRED], ps):.1f}  "
              f"AUC {auc(yv[:N_PRED], ps):.4f}")


if __name__ == "__main__":
    main()
