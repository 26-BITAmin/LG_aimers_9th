"""실패 모드 분해 학습 — 단일 이진 vs 3-헤드 (`src/modelabel.py`가 연 문)

착상 (§7.40)
    `control_success`는 세 실패의 **합집합**이고, 그 셋은 카운트에 따라 **반대로**
    움직인다. `modelabel.py`가 투구별 모드 라벨을 복원했고 (success 라벨과 100% 정합),
    카운트별 구성이 이렇다:

        카운트   성공     몰림     반대     벗어남
        0-2    0.5184  0.1179  0.1624  **0.2012**
        3-1    0.5045  0.1763  0.2755  **0.0437**
        변동폭  1.07배  1.5배   1.7배   **4.6배**

    성공률은 1.07배밖에 안 움직이는데 벗어남은 4.6배 움직인다. **상쇄된다.**
    단일 이진 분류기는 노이즈 낀 라벨에서 그 미세한 순효과를 직접 배워야 한다.
    분해하면 크고 뚜렷한 성분을 각각 배워서 합친다 — 정보가 아니라 **추정**의 문제다.

측정
    A  단일       success 하나로 학습                     ← 현재 파이프라인
    B  3분해      mid/rev/off 각각 학습 → 1−Σ 로 결합
    C  4헤드 평균  A와 B를 로짓 공간에서 반반

    전부 **같은 피처·같은 하이퍼파라미터·같은 시드**다. 달라지는 것은 타깃뿐이다.
    평가는 BSS oracle (검증셋 실제 평균으로 레벨 이동) — 레벨이 아니라 판별력만 본다.

⚠️ 라벨 복원은 **학습 전용**이다. 추론에는 복원 라벨이 없고 필요도 없다.
   복원 라벨로 **피처를 만들면 안 된다** — test에는 후속 투구가 없어 재현 불가다.

실행:  .venv/Scripts/python.exe -u src/ablate35.py 2>&1 | tee experiments/ablate_run35.log
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
                    SEED_ENSEMBLE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL,
                    TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from modelabel import recover
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
SEEDS = SEED_ENSEMBLE[:3]
THREADS = 6
EPS = 1e-6


def featurize(d, art):
    x = build_features(d, art)
    if USE_PLAYER_IDS:
        for c in ("pitcher_id", "batter_id"):
            x[c] = d[c].to_numpy(dtype=np.float32)
    return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])


def fit_one(X, y, w, seed, Xv):
    par = dict(FINAL_LGB_PARAMS, seed=seed, bagging_seed=seed,
               feature_fraction_seed=seed, data_random_seed=seed,
               num_threads=THREADS)
    bst = lgb.train(par, lgb.Dataset(X, y, weight=w), num_boost_round=N_ROUNDS)
    return bst.predict(Xv, raw_score=True)


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    season = df["season"].to_numpy()
    y_all = df[TARGET_COL].to_numpy(dtype=np.float64)

    print("모드 라벨 복원 ...")
    lab, _, _ = recover(df)
    mid = np.round(lab["middle"])
    rev = np.round(lab["reverse"])
    suc = np.round(lab["success"])
    # 겹침이 있어 off가 음수로 나올 수 있다 → 0으로 자른다 (실패의 7.16%)
    off = np.clip(1.0 - suc - mid - rev, 0.0, 1.0)
    okl = ~(np.isnan(mid) | np.isnan(rev) | np.isnan(suc))
    print(f"  복원 {okl.sum():,} / {len(df):,} = {okl.mean():.1%}"
          f"   ({time.time()-t0:.0f}s)")
    print(f"  전체 비율  mid {np.nanmean(mid[okl]):.4f}  rev {np.nanmean(rev[okl]):.4f}"
          f"  off {np.nanmean(off[okl]):.4f}")

    for fold in FOLDS:
        tr = (season < fold) & okl        # 모드 라벨이 있는 행만 학습에 쓴다
        va = season == fold
        d_tr, d_va = df[tr], df[va]
        w = recency_weight(d_tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=fold - 1)
        art = fit_artifacts(d_tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                            k_batter=SMOOTH_K_BATTER)
        X, Xv = featurize(d_tr, art), featurize(d_va, art)
        yv = y_all[va]

        print("\n" + "=" * 86)
        print(f"fold {fold}   train {tr.sum():,} → valid {va.sum():,}"
              f"   피처 {X.shape[1]}개   ({time.time()-t0:.0f}s)")
        print("=" * 86)

        rows = {"A 단일": [], "B 3분해": [], "C 반반": []}
        for seed in SEEDS:
            # ── A: 현재 파이프라인 ──────────────────────────────
            zA = fit_one(X, y_all[tr], w, seed, Xv)
            pA = 1.0 / (1.0 + np.exp(-zA))

            # ── B: mid/rev/off 각각 → 1−Σ ──────────────────────
            ps = {}
            for name, lb in (("mid", mid), ("rev", rev), ("off", off)):
                z = fit_one(X, lb[tr], w, seed, Xv)
                ps[name] = 1.0 / (1.0 + np.exp(-z))
            pB = np.clip(1.0 - (ps["mid"] + ps["rev"] + ps["off"]), EPS, 1 - EPS)

            # ── C: 로짓 공간 반반 ──────────────────────────────
            zB = np.log(pB / (1 - pB))
            pC = 1.0 / (1.0 + np.exp(-(0.5 * zA + 0.5 * zB)))

            for nm, p in (("A 단일", pA), ("B 3분해", pB), ("C 반반", pC)):
                rows[nm].append(bss(yv, shift_to_prior(p, yv.mean())))
            print(f"  seed {seed:>5}   " + "   ".join(
                f"{nm} {rows[nm][-1]:>8.1f}" for nm in rows)
                + f"   ({time.time()-t0:.0f}s)")

        print(f"\n  {'설정':<10}{'평균':>10}{'std':>8}{'vs A':>10}")
        base = np.mean(rows["A 단일"])
        for nm, v in rows.items():
            v = np.array(v)
            print(f"  {nm:<10}{v.mean():>10.1f}{v.std():>8.2f}"
                  f"{v.mean()-base:>+10.2f}")

    print(f"\n총 {time.time()-t0:.0f}s")
    print("\n⚠️ B가 A보다 낮아도 결론이 아니다 — 결합 규칙(1−Σ)이 겹침을 무시한다.")
    print("   그 경우 다음 단계는 **다중 과제 학습**(공유 표현 + 보조 헤드)이다.")
    print("   결합 규칙이 아니라 표현을 공유하는 쪽이라 겹침 문제가 사라진다.")


if __name__ == "__main__":
    main()
