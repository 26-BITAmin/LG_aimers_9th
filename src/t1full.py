"""§7.175b — 전 시즌 코어 재학습: **재현 확인 + 제출본** (사용자 승인)

§7.175 는 `<2024` 코어로 증강 Δ **+1.29 [+0.48, +2.09] P=0.998** 을 얻었다.
스택은 T1 그대로였고(74개 중 54개 원본, 바뀐 건 코어 lgbm 20개뿐) 두 팔이 같은
코어 계보라 **증강 효과 자체는 깨끗**하다. 그러나 제출하려면 코어를 우리 것으로
바꿔야 하고, 그 교체분은 미측정이다:

    LB Δ  =  (우리 코어 − 혜진 코어)  +  (증강 +1.29)
                ↑ 미측정, 부호 불명       ↑ 측정됨

⚠️ 그 첫 항은 **홀드아웃으로 못 잰다** — 혜진 코어는 2024 를 학습했고 우리 `<2024`
   코어는 안 했다. 2024 에서 붙이면 혜진이 무조건 이긴다 (§7.174 와 같은 함정).

그래서 **점수가 아니라 재현도**로 판정한다. 전 시즌(2019~2024)으로 증강 없이
학습하면 T1 출하 코어와 **같은 데이터·같은 설정**이 된다. 예측 상관이 0.999 대면
"재현했다" 는 뜻이고 교체 위험이 사실상 0 이다.

팔
    E_full_base  전 시즌 · 증강 없음   → T1 출하본과 상관 비교 (재현 확인)
    F_full_aug   전 시즌 · 증강 ON     → 제출 후보

재현 검증 3종
    ① soft_h 가 T1 meta.json 의 값과 일치하는가 (전 시즌에서 추정)
    ② 피처 78개 이름·순서가 부스터 기대치와 일치하는가
    ③ E_full_base 예측 vs T1 출하본 예측 상관

실행:  .venv/Scripts/python.exe -u src/t1full.py 2>&1 | tee experiments/t1full.log
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
T1 = ROOT / "t1work"
ASCII = Path("D:/t1eval")
sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

import soften
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, N_ROUNDS,
                    RECENCY_HALF_LIFE, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from hbeta2 import season_gt_level
from t1core import DROP, RATIO, SEEDS, VARIANTS, WORKER, build, t1_features

ARMS = (("E_full_base", False), ("F_full_aug", True))
REF_SEASON = 2024


def main():
    t0 = time.time()
    bundle = joblib.load(T1 / "model" / "artifacts.joblib")
    art = bundle["artifacts"]
    cols = lgb.Booster(model_str=(T1 / "model" / "lgbm_base_42.txt")
                       .read_text(encoding="utf-8")).feature_name()
    meta = json.load(open(T1 / "model" / "meta.json", encoding="utf-8"))

    df = load_csv(TRAIN_CSV)                       # ← 전 시즌 2019~2024
    print(f"학습 행 {len(df):,} (T1 meta: {meta['train_rows']:,})  "
          f"일치={len(df) == meta['train_rows']}")

    # ── 재현 검증 ① soft_h ────────────────────────────────────────────
    h_ours = soften.fit_h(df)
    h_t1 = {int(k): v for k, v in meta["soft_h"].items()}
    print("\n재현① soft_h (전 시즌 추정)")
    ok = True
    for yr in sorted(h_t1):
        d = abs(h_ours.get(yr, 0) - h_t1[yr])
        ok &= d < 1e-6
        print(f"    {yr}  우리 {h_ours.get(yr, float('nan')):.12f}  "
              f"T1 {h_t1[yr]:.12f}  |Δ|={d:.2e}")
    print(f"    → {'✅ 일치 (파이프라인 동일)' if ok else '❌ 불일치'}")

    X = t1_features(df, art, cols)
    print(f"\n재현② 피처 {X.shape} · 순서 일치 ✅  ({time.time()-t0:.0f}s)")
    w = recency_weight(df.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=REF_SEASON)

    for arm, aug in ARMS:
        dst = ASCII / arm
        if (dst / "model" / f"lgbm_lowff_{SEEDS[-1]}.txt").exists():
            print(f"  {arm} 재사용")
            continue
        shutil.copytree(T1, dst, dirs_exist_ok=True)
        rng = np.random.default_rng(777)
        X2, y2, w2, init, k = build(df, X, w, aug, rng)
        print(f"  [{arm}] 행 {len(X):,} → {len(X2):,} (합성 {k:,})")
        for v in VARIANTS:
            for sd in SEEDS:
                p = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v], seed=sd,
                         bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
                b = lgb.train(p, lgb.Dataset(X2, y2, weight=w2, init_score=init),
                              num_boost_round=N_ROUNDS)
                (dst / "model" / f"lgbm_{v}_{sd}.txt").write_text(
                    b.model_to_string(), encoding="utf-8", newline="\n")   # LF (§7.173③)
                del b
            print(f"    {v} {len(SEEDS)}시드 ({time.time()-t0:.0f}s)")
        del X2, y2, w2, init
    del X

    # ── 재현 검증 ③ — T1 출하본과 예측 상관 ──────────────────────────
    wk = ROOT / "experiments" / "cache" / "_t1full_worker.py"
    wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/")),
                  encoding="utf-8", newline="\n")
    P = {}
    for arm in ("T1_orig", "E_full_base", "F_full_aug"):
        f = ROOT / "experiments" / "cache" / f"t1full_{arm}.npz"
        if not f.exists():
            r = subprocess.run([sys.executable, str(wk), arm, str(f)],
                               capture_output=True, text=True)
            if not f.exists():
                print(f"  ❌ {arm}: {r.stderr.strip()[-300:]}")
                return
        P[arm] = np.load(f)["p"]
        print(f"  {arm} 예측 mean={P[arm].mean():.5f} std={P[arm].std():.5f}"
              f"  ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 78)
    print("재현③ — T1 출하본 대비 예측 상관 (점수가 아니라 재현도로 판정한다)")
    print("=" * 78)
    for arm in ("E_full_base", "F_full_aug"):
        c = float(np.corrcoef(P[arm], P["T1_orig"])[0, 1])
        mad = float(np.abs(P[arm] - P["T1_orig"]).mean())
        mx = float(np.abs(P[arm] - P["T1_orig"]).max())
        print(f"  {arm:<14} corr {c:.6f}  평균|Δp| {mad:.5f}  최대|Δp| {mx:.5f}")
    c2 = float(np.corrcoef(P["F_full_aug"], P["E_full_base"])[0, 1])
    print(f"  (증강 팔 vs 무증강 팔  corr {c2:.6f} — 1.0 이면 증강이 안 걸린 것)")
    assert c2 < 1.0, "두 팔이 같다 — 증강 미적용"
    print("\n  판정: corr ≥ 0.999 면 코어 재현 성공 → 교체 위험 ≈ 0,")
    print("        LB 변화가 곧 증강분(+1.29 홀드아웃)이 된다.")
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
