"""12차 실험 — LightGBM 내부 다양성 앙상블.

11차 결론: 다양화 이득은 실재한다(lgbm+xgb 블렌드가 두 점수의 단순 평균보다 +15.5).
          하지만 xgb/cat은 단독 성능이 45~214점 낮아 그 핸디캡을 못 이긴다.

→ **핸디캡 없는 다양성**을 쓴다. 같은 LightGBM의 서로 다른 설정끼리 섞는다.
   설정이 다르면 오차 패턴도 다르므로 탈상관 이득은 남고, 단독 성능은 비슷하다.
   덤으로 20배 빠르다 (fold당 80초 vs CatBoost 20분).

레짐이 같은 fold(2023, 2024)에서만 판정한다.

실행:  .venv/Scripts/python.exe -u src/ablate11.py
"""
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import lightgbm as lgb
import numpy as np

from config import (DROP_FEATURES, FINAL_LGB_PARAMS, N_ROUNDS, RECENCY_HALF_LIFE,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV,
                    USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
SEEDS = (42, 7, 2024)


def P(**kw):
    return dict(FINAL_LGB_PARAMS, **kw)


# 단독 성능은 비슷하되 오차 패턴이 다르도록 축을 다르게 흔든다
VARIANTS = {
    "base":     (P(), N_ROUNDS),                                    # 31/2000/ff.8/lr.015
    "wide":     (P(num_leaves=127, min_data_in_leaf=8000), N_ROUNDS),
    "extra":    (P(extra_trees=True, lambda_l2=10.0), N_ROUNDS),    # 분할점 무작위화
    "lowff":    (P(feature_fraction=0.45, bagging_fraction=0.6), N_ROUNDS),
    "goss":     (P(data_sample_strategy="goss", bagging_freq=0), N_ROUNDS),
    "slow":     (P(learning_rate=0.008), N_ROUNDS * 2),
}


def fold_logits(df, valid_season, seeds=SEEDS):
    tr = df[df.season < valid_season]
    va = df[df.season == valid_season]
    ref = int(tr.season.max())
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=ref)
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xtr, Xva = X(tr), X(va)
    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    dtr = lgb.Dataset(Xtr, ytr, weight=w, free_raw_data=False)

    out, t0 = {}, time.time()
    for name, (params, rounds) in VARIANTS.items():
        Z = []
        for s in seeds:
            p = dict(params, seed=s, bagging_seed=s,
                     feature_fraction_seed=s, data_random_seed=s)
            bst = lgb.train(p, dtr, num_boost_round=rounds)
            Z.append(bst.predict(Xva, raw_score=True))
        out[name] = np.mean(Z, axis=0)
        p_ = shift_to_prior(1 / (1 + np.exp(-out[name])), yva.mean())
        print(f"    {name:<7s} BSS={bss(yva, p_):>8.1f}   ({time.time()-t0:.0f}s)")
    return out, yva


def score(z, y):
    return bss(y, shift_to_prior(1 / (1 + np.exp(-z)), y.mean()))


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    Zs, ys = {}, {}
    for f in FOLDS:
        print(f"\n=== valid={f} ===")
        Zs[f], ys[f] = fold_logits(df, f)

    names = list(VARIANTS)
    combos = {n: (n,) for n in names}
    for r in (2, 3, 4):
        for c in combinations(names, r):
            if "base" in c:                       # 기준 모델을 포함하는 조합만
                combos["+".join(c)] = c
    combos["ALL"] = tuple(names)

    rows = []
    for label, c in combos.items():
        v = np.array([score(np.mean([Zs[f][m] for m in c], axis=0), ys[f]) for f in FOLDS])
        rows.append((label, v))
    base = dict(rows)["base"]

    rows.sort(key=lambda r: -r[1].mean())
    print(f"\n{'조합':<34s} {'2023':>9s} {'2024':>9s} {'평균':>9s} {'기준대비':>9s} "
          f"{'부호':>5s} {'크기비':>7s}")
    print("-" * 88)
    for label, v in rows[:18]:
        d = v - base
        same = bool(np.all(d > 0) or np.all(d < 0))
        sp = np.abs(d).max() / max(np.abs(d).min(), 1e-9) if same else np.inf
        mark = "" if label == "base" else ("  <<<" if same and d.mean() > 0 and sp <= 3 else "")
        print(f"{label:<34s} {v[0]:>9.1f} {v[1]:>9.1f} {v.mean():>9.1f} {d.mean():>+9.1f} "
              f"{'O' if same else 'X':>5s} {sp:>7.1f}{mark}")

    print("\n  <<< = 채택 조건 통과 (부호 일치 + fold간 크기 3배 이내 + 개선)")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
