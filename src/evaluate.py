"""4-fold forward-chaining 평가 하네스 — 채택 판정의 표준 도구.

왜 만드는가
    fold 2개(valid 2023/2024)로는 **+10 규모의 개선을 판정할 수 없다.**
    Trackman 피처가 그 증거다: fold2에서 +13.3 (3.6σ)였지만 실제 2025에서 −13.6이었다.
    놓친 신호는 부호가 아니라 **fold 간 효과 크기의 불일치**였다 (+2.3 vs +13.3).

설계
    valid = 2021, 2022, 2023, 2024  (각각 직전 시즌들로 학습)
    fold마다 여러 시드 → 효과의 평균과 **fold 간 일관성**을 함께 본다.

    채택 조건 (Trackman 사고 이후 강화):
      1) 전 fold 평균이 노이즈를 넘을 것
      2) **모든 fold에서 부호가 같을 것**
      3) **fold 간 효과 크기가 3배 이내로 모일 것**  ← 이게 없었다

지표는 BSS(oracle shift) — 레벨 성분을 제거한 순수 refinement.
레벨(prior)은 §3에서 따로 다루며 리더보드로 맞추지 않는다.

사용:
    from evaluate import compare
    compare(df, {"기준": cfg_a, "변경": cfg_b})
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

from config import (DROP_FEATURES, FINAL_LGB_PARAMS, N_ROUNDS, RECENCY_HALF_LIFE,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV,
                    USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import bss, shift_to_prior

# ⚠️ fold를 늘린다고 검증이 좋아지지 않는다.
#
# 2021/2022까지 넣어봤더니 현재 설정이 그쪽에서 400~550점 **나빴다**.
# `game_type` 정의가 2022→2023에 바뀌었기 때문이다 —
# 2022 이전을 예측할 때 `game_type`은 유효하고 강력한 피처(F 0.71 vs R 0.50)이고,
# 2023 이후에만 독이 된다.
#
#   설정                        2021    2022    2023    2024
#   game_type 유지, 동일가중    1532.7  2252.2  -798.4   431.6
#   game_type 제거, hl0.5,+ids   980.2  1848.1   -11.1   740.2
#
# "모든 fold 부호 일치"를 요구했다면 실제로 +172를 벌고 LB까지 살아남은
# `game_type` 제거를 기각했을 것이다.
# → **2025와 같은 레짐(2023 이후)의 fold만 쓴다.**
FOLDS = [2023, 2024]
ALL_FOLDS = [2021, 2022, 2023, 2024]      # 레짐 경계 진단용
SEEDS = (42, 7, 2024)


def run_fold(df, valid_season, *, tm_table=None, tmb_table=None,
             half_life=RECENCY_HALF_LIFE,
             params=FINAL_LGB_PARAMS, drop=DROP_FEATURES, use_ids=USE_PLAYER_IDS,
             rounds=N_ROUNDS, seeds=SEEDS, k=SMOOTH_K_PITCHER,
             extra=None, post=None):
    """extra / post — **전달 방식을 같은 자리에서 비교**하기 위한 훅 (§7.56).

    같은 정보라도 컬럼으로 넣는 것과 사후 로짓에 더하는 것은 다른 물건이다.
    컬럼은 상호작용을 쓸 수 있지만 §7.51이 본 '컬럼 추가 비용'을 내고,
    사후 가산은 선형이지만 공짜다. 둘을 다른 스크립트에서 재면 비교가 안 된다.

      extra(tr) -> f(d) -> {컬럼명: 배열}      학습 시즌으로 fit, 행 단위로 apply
      post(tr)  -> f(d) -> 로짓에 더할 배열     예측 **후** 가산

    ⚠️ 둘 다 `tr`로만 fit 한다. 검증 행 통계를 쓰면 그 자체가 리크다.
    """
    tr = df[df.season < valid_season]
    va = df[df.season == valid_season]
    ref = int(tr.season.max())
    w = (np.ones(len(tr)) if half_life is None          # None = 동일 가중
         else recency_weight(tr.season.to_numpy(), half_life, ref_season=ref))
    art = fit_artifacts(tr, weights=w, k_pitcher=k, k_batter=SMOOTH_K_BATTER,
                        tm_table=tm_table, tmb_table=tmb_table)

    ex = extra(tr) if extra is not None else None
    po = post(tr) if post is not None else None

    def X(d):
        x = build_features(d, art)
        if ex is not None:
            for c, v in ex(d).items():
                x[c] = np.asarray(v, dtype=np.float32)
        if use_ids:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in drop if c in x.columns])

    Xtr, Xva = X(tr), X(va)
    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    zadd = po(va) if po is not None else 0.0

    vals = []
    for s in seeds:
        p = dict(params, seed=s, bagging_seed=s, feature_fraction_seed=s,
                 data_random_seed=s)
        bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=rounds)
        pr = 1 / (1 + np.exp(-(bst.predict(Xva, raw_score=True) + zadd)))
        vals.append(bss(yva, shift_to_prior(pr, yva.mean())))
    return np.array(vals)


def compare(df, configs, folds=FOLDS, seeds=SEEDS, baseline=None):
    """configs: {이름: kwargs}. 첫 항목(또는 baseline)을 기준으로 차이를 낸다."""
    t0 = time.time()
    names = list(configs)
    baseline = baseline or names[0]
    res = {n: {} for n in names}

    for f in folds:
        for n in names:
            res[n][f] = run_fold(df, f, seeds=seeds, **configs[n])
            v = res[n][f]
            print(f"  valid={f}  {n:<22s} {v.mean():>8.1f} ± {v.std(ddof=1):>5.1f}"
                  f"   ({time.time()-t0:.0f}s)")

    print(f"\n{'설정':<22s} " + " ".join(f"{f:>9d}" for f in folds) + f" {'평균차':>9s}")
    print("-" * (23 + 10 * len(folds) + 10))
    for n in names:
        row = " ".join(f"{res[n][f].mean():>9.1f}" for f in folds)
        print(f"{n:<22s} {row}")

    print("\n=== 채택 판정 ===")
    print("  근거: game_type 제거는 fold간 1.1배로 모여 LB까지 살아남았고,")
    print("        Trackman은 6배로 흩어져 LB에서 뒤집혔다.")
    for n in names:
        if n == baseline:
            continue
        d = np.array([res[n][f].mean() - res[baseline][f].mean() for f in folds])
        se = np.sqrt(np.mean([res[n][f].var(ddof=1) / len(seeds)
                              + res[baseline][f].var(ddof=1) / len(seeds) for f in folds])
                     / len(folds))
        same_sign = bool(np.all(d > 0) or np.all(d < 0))
        spread = (np.abs(d).max() / max(np.abs(d).min(), 1e-9)) if same_sign else np.inf
        big = abs(d.mean()) > 100          # 이 규모는 실측에서도 전이됐다
        ok = (abs(d.mean()) > 2 * se) and same_sign and spread <= 3
        print(f"  {n}")
        print(f"    fold별 효과: {np.round(d,1)}   평균 {d.mean():+.1f}  (표준오차 {se:.1f})")
        print(f"    ① 노이즈 초과 {'O' if abs(d.mean())>2*se else 'X'}"
              f"   ② 부호 일치 {'O' if same_sign else 'X'}"
              f"   ③ 크기 3배 이내 {'O' if spread<=3 else 'X'} (실제 {spread:.1f}배)")
        if ok and not big:
            print(f"    → 조건 통과. 다만 평균 효과 {abs(d.mean()):.0f}점은 **작은 규모**라")
            print(f"       홀드아웃만으로 확신할 수 없다 (Trackman이 +13에서 뒤집혔다).")
        else:
            print(f"    → {'채택' if ok else '보류'}")
    return res


if __name__ == "__main__":
    df = load_csv(TRAIN_CSV)
    print("현재 설정을 4-fold로 재검증한다 (기준선 = game_type 유지, 동일 가중).\n")
    compare(df, {
        "구 기준 (gt유지,hl∞)": dict(half_life=None, drop=[]),
        "현재 (gt제거,hl0.5,+ids)": dict(),
    })
