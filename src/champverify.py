"""§7.212b — 내 레시피가 챔피언과 같은지 **hj03 파일에 직접 대조**한다.

사용자 지시: *"그냥 hj03 파일을 직접 확인해서 이번에는 맞는지 검사해봐."*

여태 나는 계보 문서(SUBMISSIONS·§7.175·§7.182)를 읽고 *"챔피언 코어는 leaves×2 + SMOTE"*
라고 **추론**했다. 그건 검증이 아니다. 여기서는 **같은 레시피로 다시 학습해
hj03 의 실제 부스터와 예측을 맞대본다.**

절차 (`t1cap.main` 의 코어 재학습 구간을 그대로)
    art  = hj03/model/artifacts.joblib
    cols = hj03/model/lgbm_base_42.txt 의 feature_name()
    X    = **hj03 자신의 features.py** 로 생성 (`t1core.t1_features`)
    학습 = 전 시즌 2019~2024 · recency(ref 2024) · `t1core.build(..., aug=True, rng 777)`
    파라미터 = FINAL_LGB_PARAMS + ENSEMBLE_VARIANTS[v] · num_leaves ×2 · 320 rounds

판정
    corr(z_mine, z_hj03) 와 max|Δz|. 같은 시드·같은 데이터·같은 파라미터면
    LightGBM 은 결정적이므로 **일치해야 한다.**
    어긋나면 어디서 갈리는지가 곧 내가 놓친 파이프라인 항목이다.

멤버 2개(base_42 · wide_42)만 본다 — 레시피가 맞으면 나머지는 같은 함수다.

실행: .venv/Scripts/python.exe -u src/champverify.py 2>&1 | tee experiments/champverify.log
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import joblib
import lightgbm as lgb
import numpy as np

import t1core
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, N_ROUNDS,
                    RECENCY_HALF_LIFE, TRAIN_CSV)
from data import load_csv, recency_weight

HJ = Path("D:/t1eval/HJ03")
REF_SEASON = 2024
CHECK = [("base", 42), ("wide", 42)]
NS = 200_000            # 대조용 표본 행 수


def leaf2(v):
    nl = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v]).get("num_leaves", 31)
    return int(round(nl * 2.0))


def main():
    assert (HJ / "model" / "lgbm_base_42.txt").exists(), f"hj03 미추출: {HJ}"
    t1core.T1 = HJ                      # t1_features 가 hj03 의 features.py 를 쓰게
    t0 = time.time()

    art = joblib.load(HJ / "model" / "artifacts.joblib")["artifacts"]
    cols = lgb.Booster(model_str=(HJ / "model" / "lgbm_base_42.txt")
                       .read_text(encoding="utf-8")).feature_name()
    print(f"hj03 코어 피처 {len(cols)}개")

    df = load_csv(TRAIN_CSV)            # 전 시즌 2019~2024
    X = t1core.t1_features(df, art, cols)
    w = recency_weight(df.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=REF_SEASON)
    print(f"피처 {X.shape}  ({time.time()-t0:.0f}s)")

    X2, y2, w2, init, k = t1core.build(df, X, w, True, np.random.default_rng(777))
    print(f"soften+SMOTE {len(df):,} → {len(X2):,} (합성 {k:,})  ({time.time()-t0:.0f}s)")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(NS, len(X)), replace=False)
    Xs = X.iloc[idx]
    del X

    print(f"\n{'멤버':<12}{'corr':>12}{'max|Δz|':>12}{'sd(Δz)':>11}{'sd(z_hj)':>11}  판정")
    for v, sd in CHECK:
        p = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v], seed=sd, bagging_seed=sd,
                 feature_fraction_seed=sd, data_random_seed=sd)
        p["num_leaves"] = leaf2(v)
        b = lgb.train(p, lgb.Dataset(X2, y2, weight=w2, init_score=init),
                      num_boost_round=N_ROUNDS)
        zm = b.predict(Xs, raw_score=True)
        ref = lgb.Booster(model_str=(HJ / "model" / f"lgbm_{v}_{sd}.txt")
                          .read_text(encoding="utf-8"))
        zh = ref.predict(Xs, raw_score=True)
        c = float(np.corrcoef(zm, zh)[0, 1])
        d = np.abs(zm - zh)
        ok = "✅ 일치" if d.max() < 1e-6 else ("🔶 근사" if c > 0.99 else "❌ 불일치")
        print(f"{v}_{sd:<8}{c:>12.8f}{d.max():>12.3e}{(zm-zh).std():>11.5f}"
              f"{zh.std():>11.5f}  {ok}  ({time.time()-t0:.0f}s)")
        print(f"    트리 {b.num_trees()} vs {ref.num_trees()}"
              f" · leaves 요청 {p['num_leaves']}")
        del b, ref


if __name__ == "__main__":
    main()
