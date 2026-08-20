"""실패 모드 **다중분류** — §7.40의 올바른 형태

§7.40(=ablate35)에서 무엇이 틀렸나
    `1 − (P_mid + P_rev + P_off)` 로 결합했다. 두 가지를 어긴다.
      ① 겹침을 무시한다 — 실패의 7.16%가 몰림이면서 반대다. 과대추정량이 **행마다 다르다**
      ② 세 모델의 추정 오차를 그대로 더한다
    결과: fold2024 −82.6, fold2023 −306. **착상이 아니라 산수가 틀렸다.**

올바른 형태 — 배타적 5클래스 softmax
    0 성공 .5237  |  1 몰림만 .1155  |  2 반대만 .1949  |  3 몰림+반대 .0341  |  4 벗어남 .1317
    합이 정확히 1.0이다. `P(성공) = softmax₀` 이므로 **결합 규칙이 아예 필요 없다.**

왜 이길 수 있나 (정보가 아니라 추정의 문제다)
    카운트별로 성공률은 1.07배밖에 안 움직이는데 벗어남은 **4.6배** 움직인다
    (3-1에서 .044, 0-2에서 .201). 몰림·반대는 반대 방향이라 **서로 상쇄된다.**
    이진 분류기는 노이즈 낀 라벨에서 그 미세한 순효과를 직접 배워야 한다.
    다중분류는 크고 뚜렷한 성분을 각각 배우고 softmax가 합친다.

    ⚠️ 정보량은 같다. 이기더라도 그것은 **추정 효율**의 이득이지 새 신호가 아니다.
       그러므로 크게 이길 이유도 없다 — 수십 점이면 성공이다.

라운드 수 교란
    다중분류는 반복마다 클래스 수만큼 트리를 만든다. 같은 N_ROUNDS라도 학습량이
    다르므로 **640라운드로 한 번 학습해 320·640 두 지점에서 평가**한다
    (`num_iteration`). 라운드가 모자라서 진 것인지 아닌지를 가른다.

⚠️ 라벨 복원은 학습 전용이다. 추론에는 없고 필요도 없다. 복원 라벨로 피처를
   만들면 안 된다 — test에는 후속 투구가 없어 재현 불가다.

실행:  .venv/Scripts/python.exe -u src/ablate36.py 2>&1 | tee experiments/ablate_run36.log
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
MC_ROUNDS = 2 * N_ROUNDS         # 640. 320 지점도 같이 평가한다
EPS = 1e-9


def make_classes(df):
    """배타적 5클래스. 성공 행은 정의상 mid=rev=0이다 (modelabel.py가 확인)."""
    lab, _, _ = recover(df)
    mid, rev, suc = (np.round(lab[k]) for k in ("middle", "reverse", "success"))
    ok = ~(np.isnan(mid) | np.isnan(rev) | np.isnan(suc))
    cls = np.full(len(df), -1, dtype=np.int64)
    m, r, s = mid, rev, suc
    with np.errstate(invalid="ignore"):
        cls = np.where(s == 1, 0,
              np.where((m == 1) & (r == 0), 1,
              np.where((m == 0) & (r == 1), 2,
              np.where((m == 1) & (r == 1), 3, 4))))
    cls = np.where(ok, cls, -1)
    return cls, ok


def featurize(d, art):
    x = build_features(d, art)
    if USE_PLAYER_IDS:
        for c in ("pitcher_id", "batter_id"):
            x[c] = d[c].to_numpy(dtype=np.float32)
    return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    season = df["season"].to_numpy()
    y_all = df[TARGET_COL].to_numpy(dtype=np.float64)
    cls, okl = make_classes(df)
    print(f"5클래스 분포: " + "  ".join(
        f"{i}:{np.mean(cls[okl] == i):.4f}" for i in range(5))
        + f"   복원 {okl.mean():.1%}   ({time.time()-t0:.0f}s)")

    for fold in FOLDS:
        tr = (season < fold) & okl
        va = season == fold
        d_tr, d_va = df[tr], df[va]
        w = recency_weight(d_tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=fold - 1)
        art = fit_artifacts(d_tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                            k_batter=SMOOTH_K_BATTER)
        X, Xv = featurize(d_tr, art), featurize(d_va, art)
        yv = y_all[va]

        print("\n" + "=" * 84)
        print(f"fold {fold}   train {tr.sum():,} → valid {va.sum():,}"
              f"   ({time.time()-t0:.0f}s)")
        print("=" * 84)

        rows = {k: [] for k in ("A 이진", "B 다중320", "B 다중640", "C 반반")}
        for seed in SEEDS:
            sp = dict(seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
                      data_random_seed=seed, num_threads=THREADS)

            # ── A: 현재 파이프라인 ──────────────────────────────
            bA = lgb.train(dict(FINAL_LGB_PARAMS, **sp),
                           lgb.Dataset(X, y_all[tr], weight=w),
                           num_boost_round=N_ROUNDS)
            zA = bA.predict(Xv, raw_score=True)
            pA = 1.0 / (1.0 + np.exp(-zA))

            # ── B: 배타적 5클래스 softmax ──────────────────────
            par = dict(FINAL_LGB_PARAMS, **sp)
            par.pop("objective", None)
            par.pop("metric", None)
            par.update(objective="multiclass", num_class=5, metric="multi_logloss")
            bB = lgb.train(par, lgb.Dataset(X, cls[tr], weight=w),
                           num_boost_round=MC_ROUNDS)
            got = {}
            for tag, it in (("B 다중320", N_ROUNDS), ("B 다중640", MC_ROUNDS)):
                p = bB.predict(Xv, num_iteration=it)[:, 0]   # 클래스 0 = 성공
                got[tag] = np.clip(p, EPS, 1 - EPS)

            # ── C: 로짓 공간 반반 (640 쪽과) ───────────────────
            pb = got["B 다중640"]
            zB = np.log(pb / (1 - pb))
            pC = 1.0 / (1.0 + np.exp(-(0.5 * zA + 0.5 * zB)))

            for nm, p in (("A 이진", pA), ("B 다중320", got["B 다중320"]),
                          ("B 다중640", got["B 다중640"]), ("C 반반", pC)):
                rows[nm].append(bss(yv, shift_to_prior(p, yv.mean())))
            print(f"  seed {seed:>5}  " + "  ".join(
                f"{nm} {rows[nm][-1]:>7.1f}" for nm in rows)
                + f"   ({time.time()-t0:.0f}s)")

        print(f"\n  {'설정':<12}{'평균':>9}{'std':>7}{'vs A':>9}")
        base = np.mean(rows["A 이진"])
        for nm, v in rows.items():
            v = np.array(v)
            print(f"  {nm:<12}{v.mean():>9.1f}{v.std():>7.2f}{v.mean()-base:>+9.2f}")

    print(f"\n총 {time.time()-t0:.0f}s")
    print("\n판정: 양 fold에서 +이고 크기비 3배 이내여야 후보다 (src/decide.py).")
    print("      B가 320<640이면 라운드 부족이므로 더 늘려볼 값이 있다.")


if __name__ == "__main__":
    main()
