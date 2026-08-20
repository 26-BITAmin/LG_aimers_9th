"""13차 실험 — CatBoost 재조사 (11차의 -214는 설정 버그가 아니었나).

배경
    11차에서 CatBoost 단독 BSS가 LightGBM 대비 -214였다. 그때는 "이 문제에 안 맞는다"로
    결론냈는데, **같은 피처 위 GBDT 패밀리 격차가 -214가 나올 수는 없다.** 보통 ±30이다.
    이 크기는 패밀리 차이가 아니라 설정 버그의 지문이다.

확인된 버그
    `SymmetricTree`(CatBoost 기본 grow_policy)에서 `min_data_in_leaf`는 **조용히 무시된다.**
    `get_all_params()`는 2000이라고 답하지만 실제 예측은 `min_data_in_leaf=1`과 비트 단위로
    같다. Lossguide / Depthwise에서만 적용된다.
    → 나는 LightGBM과 같은 정규화를 걸었다고 믿었지만 안 걸려 있었다.

남은 용의자
    (a) `cat_features`의 target statistic.
        `pitcher_id` CTR은 2019~2023을 풀링한 타깃 평균이다. 시즌 base rate가
        .565 -> .486로 흐르므로 CTR은 "실력"이 아니라 "어느 시대에 던졌나"를 인코딩한다.
        게다가 recency weight(2019행 가중치 0.004)를 CTR 계산이 존중하는지 불확실하다.
        → Trackman을 죽인 것과 같은 실패 모드: 시간에 따라 타깃과의 관계가 변하는 피처.
    (b) 학습량.
        LGBM은 lr 0.015 x 320 = 4.8, 내 CatBoost는 0.03 x 800 = 24 (5배).
        LGBM은 lr을 0.03->0.015로 **줄여서** +11을 얻은 문제다. CatBoost는 튜닝한 적이 없다.

설계
    `cat_features` on/off  x  `SymmetricTree` / `Lossguide` 의 2x2.
    각 셀마다 **iteration별 BSS 곡선**을 찍는다. 곡선 하나로 (b)까지 같이 답이 나온다:
      - 정점이 앞쪽(200~400)에 있으면 -> 과적합이 원인
      - 1200까지 단조 증가인데도 571 근처면 -> 설정 문제가 아니다

    ⚠️ 여기서 valid로 최적 iteration을 고르는 것은 **진단이므로** 허용한다
       ("내 설정이 망가졌었나?"를 묻는 중). 채택은 별도로 2 fold x 3 시드로 판정한다.

실행:  .venv/Scripts/python.exe -u src/ablate13.py
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

from config import (DROP_FEATURES, FINAL_LGB_PARAMS, N_ROUNDS, RECENCY_HALF_LIFE,
                    SEED, SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV,
                    USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import bss, shift_to_prior

VALID = 2024                  # 진단은 1 fold면 충분하다 (571 -> 740이면 결정적)
ITERS = 1200
EVAL_PERIOD = 50
CAT_ID_COLS = ["pitcher_id", "batter_id"]

# 2x2.  depth는 정보량을 맞춘다: Symmetric depth6 = 64리프, Lossguide는 LGBM과 같은 31리프.
CONFIGS = {
    "sym_ctr  (11차 원본)": dict(grow_policy="SymmetricTree", depth=6, cat=True),
    "sym_num  (ID 수치)": dict(grow_policy="SymmetricTree", depth=6, cat=False),
    "loss_ctr (31리프)": dict(grow_policy="Lossguide", max_leaves=31, depth=16,
                             min_data_in_leaf=2000, cat=True),
    "loss_num (31리프)": dict(grow_policy="Lossguide", max_leaves=31, depth=16,
                             min_data_in_leaf=2000, cat=False),
}


def fold_matrices(df, valid_season):
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

    return (X(tr), tr[TARGET_COL].to_numpy(dtype=np.float64), w,
            X(va), va[TARGET_COL].to_numpy(dtype=np.float64))


def score(z, y):
    """레벨 성분을 제거한 순수 refinement (MODEL.md 표준 지표)."""
    return bss(y, shift_to_prior(1 / (1 + np.exp(-z)), y.mean()))


def run_lgbm(Xtr, ytr, w, Xva, yva):
    import lightgbm as lgb
    p = dict(FINAL_LGB_PARAMS, seed=SEED, bagging_seed=SEED,
             feature_fraction_seed=SEED, data_random_seed=SEED)
    bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=N_ROUNDS)
    return score(bst.predict(Xva, raw_score=True), yva)


def run_cat(Xtr, ytr, w, Xva, yva, *, cat, **over):
    from catboost import CatBoostClassifier, Pool
    Xtr, Xva = Xtr.copy(), Xva.copy()
    idx = None
    if cat:
        cols = [c for c in CAT_ID_COLS if c in Xtr.columns]
        for c in cols:
            Xtr[c] = Xtr[c].astype(np.int64).astype(str)
            Xva[c] = Xva[c].astype(np.int64).astype(str)
        idx = [Xtr.columns.get_loc(c) for c in cols]

    m = CatBoostClassifier(iterations=ITERS, learning_rate=0.03, l2_leaf_reg=30.0,
                           border_count=128, rsm=0.8, random_seed=SEED,
                           loss_function="Logloss", verbose=0, thread_count=6,
                           allow_writing_files=False, **over)
    m.fit(Pool(Xtr, ytr, weight=w, cat_features=idx))

    pool_va = Pool(Xva, cat_features=idx)
    curve = []
    for i, z in enumerate(m.staged_predict(pool_va, prediction_type="RawFormulaVal",
                                           eval_period=EVAL_PERIOD)):
        curve.append(((i + 1) * EVAL_PERIOD, score(np.asarray(z, dtype=np.float64), yva)))
    return curve


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    Xtr, ytr, w, Xva, yva = fold_matrices(df, VALID)
    print(f"valid={VALID}  train={Xtr.shape}  valid={Xva.shape}  ({time.time()-t0:.0f}s)\n")

    base = run_lgbm(Xtr, ytr, w, Xva, yva)
    print(f"기준 lgbm (1시드, {N_ROUNDS}라운드)  BSS = {base:.1f}"
          f"   ({time.time()-t0:.0f}s)\n")

    best = {}
    for name, cfg in CONFIGS.items():
        curve = run_cat(Xtr, ytr, w, Xva, yva, **cfg)
        it, v = max(curve, key=lambda t: t[1])
        best[name] = (it, v)
        at800 = dict(curve).get(800)     # 11차가 쓴 지점 — 그때와 직접 비교용
        print(f"--- {name} ---   ({time.time()-t0:.0f}s)")
        print("    " + "  ".join(f"{i}:{s:.0f}" for i, s in curve))
        tail = "" if at800 is None else f"   [11차 지점 iter800 = {at800:.1f}]"
        print(f"    정점 iter={it}  BSS={v:.1f}   (기준 대비 {v-base:+.1f}){tail}\n")

    print(f"{'설정':<22s} {'정점iter':>8s} {'BSS':>9s} {'lgbm대비':>9s}")
    print("-" * 52)
    print(f"{'lgbm (기준)':<22s} {N_ROUNDS:>8d} {base:>9.1f} {0.0:>+9.1f}")
    for name, (it, v) in sorted(best.items(), key=lambda kv: -kv[1][1]):
        print(f"{name:<22s} {it:>8d} {v:>9.1f} {v-base:>+9.1f}")

    print("\n해석 가이드:")
    print("  · 어떤 셀이든 lgbm 대비 -50 이내면 → 11차의 -214는 설정 버그였다.")
    print("    그러면 CatBoost는 앙상블 멤버 후보가 된다 (분산 축소 = 전이되는 계열).")
    print("  · sym_num >> sym_ctr 이면  → 범인은 cat_features의 target statistic.")
    print("  · loss_* >> sym_*   이면  → 범인은 symmetric tree / 정규화 미적용.")
    print("  · 정점이 앞쪽이면        → 범인은 학습량(0.03x800이 과했다).")
    print("  · 넷 다 -150 이하면      → 설정이 아니라 진짜 패밀리 부적합. 인텔은 무관한 정보.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
