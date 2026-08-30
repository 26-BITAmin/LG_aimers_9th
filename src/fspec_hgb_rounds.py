"""§7.219b — §7.219 의 fold 역전이 **시즌 때문인가 라운드 때문인가**.

base: foldcache (우리 fold 하네스). §7.219 와 같은 층·같은 캐시.

무엇이 문제였나
    §7.219 에서 H(HGB) 와 L(LGBM) 의 우열이 fold 간에 완전히 뒤집혔다
    (F행 BSS, a=0.30):  2023H2  L 52.0 vs H 88.1     2024  L 614.1 vs H 610.3

    그런데 두 fold 는 **같은 조건에서 비교되지 않았다.**
        fold 2023H2   L [200,200,200]   H [200,200,200]   <- 전부 nr=200 폴백
        fold 2024     L [50,57,60]      H [50,50,50]      <- 내부 분할로 선택

    2023H2 의 깨끗한 F 학습행은 2023-03~06 의 14,535행뿐이라 `_inner_split` 의
    75% 분위 컷이 검증쪽을 비우고(`iva.sum()=0`), 라운드 선택이 죽어 `nr=200`
    폴백으로 떨어진다. `fspec_verify.py` 에서 물려받은 구조다.
    [[audit-inherited-constants-not-just-registered-gates]]

    그래서 두 fold 는 **학습 깊이가 4배 다르다.** 그리고 HGB 에 없는 노브가 하필
    `bagging_fraction` 인데, 행 배깅의 값어치는 **길게 학습할수록 커진다** —
    짧게 끊으면 배깅이 할 일이 별로 없다. 즉 "200 라운드에서 HGB 승 / 55 라운드에서
    HGB 패" 가 시즌이 아니라 **라운드의 함수**일 수 있다.

이 스크립트가 하는 것
    라운드 선택을 **끄고** nr ∈ {50, 100, 200, 400} 를 두 팔에 똑같이 강제한 뒤,
    두 fold 에서 같은 격자를 그린다. 읽는 법:

      · H−L 부호가 **라운드**를 따라 뒤집히고 fold 안에서 같은 방향으로 움직이면
        -> §7.219 의 fold 역전은 **아티팩트**다. 시즌 증거가 아니다.
      · H−L 부호가 fold 를 따라만 갈리고 라운드에는 둔감하면
        -> 역전은 **진짜 시즌 효과**다. 그때 비로소 전이 위험을 논할 수 있다.

    corr(L,H) 도 라운드별로 같이 찍는다. 다양성이 깊이의 함수인지 보려는 것이다.

⚠️ 여기서 고른 라운드로 배포값을 정하지 않는다. 이건 **진단**이지 튜닝이 아니다.
   격자 최대점을 배포로 가져가면 그게 곧 오라클 선택이다.

실행:
    .venv/Scripts/python.exe -u src/fspec_hgb_rounds.py 2>&1 | tee experiments/fspec_hgb_rounds.log
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import foldcache
from config import MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from fspec_hgb import (A_DEPLOY, FSPEC_MIN_SEASON, HGB_PARAMS, SEEDS, SMALL_LGB)
from validation import bss, shift_to_prior

FOLDS = ["2023H2", 2024]
ROUNDS = [50, 100, 200, 400]


def fixed_lgb(Xf, yf, Xv, nr):
    zs = []
    for s in SEEDS:
        prm = dict(SMALL_LGB, seed=s, bagging_seed=s, feature_fraction_seed=s,
                   data_random_seed=s)
        b = lgb.train(prm, lgb.Dataset(Xf, yf), num_boost_round=nr)
        zs.append(b.predict(Xv, raw_score=True))
    return np.mean(zs, axis=0)


def fixed_hgb(Xf, yf, Xv, nr):
    yi = yf.astype(np.int8)
    zs = []
    for s in SEEDS:
        m = HistGradientBoostingClassifier(**HGB_PARAMS, max_iter=nr, random_state=s)
        m.fit(Xf, yi)
        zs.append(m.decision_function(Xv).ravel())
    return np.mean(zs, axis=0)


def main():
    df = load_csv(TRAIN_CSV)
    print("=" * 82)
    print("  §7.219b  fold 역전은 시즌 때문인가 라운드 때문인가")
    print("  base: foldcache (우리 fold 하네스) — §7.219 와 같은 층")
    print("=" * 82)
    cache = foldcache.load(FOLDS)
    t0 = time.time()

    for f in FOLDS:
        z_lgb, z_mlp, y, _ = cache[f]
        z_dep = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp

        mtr, mva = foldcache.SPLITS[f](df.season.to_numpy(), df.game_month.to_numpy())
        tr, va = df[mtr], df[mva]
        assert len(va) == len(y) and np.allclose(va[TARGET_COL].to_numpy(), y), \
            "행 정렬 불일치"
        Xtr, ytr, _, Xva, yva, _, _ = foldcache.fold_data(df, f)

        gtr = tr["game_type"].to_numpy().astype(str)
        vF = va["game_type"].to_numpy().astype(str) == "F"
        mF = (gtr == "F") & (tr["season"].to_numpy() >= FSPEC_MIN_SEASON)

        Xf = Xtr[mF].to_numpy(dtype=np.float32)
        yf = ytr[mF]
        Xv = Xva.to_numpy(dtype=np.float32)
        Xf[np.isinf(Xf)] = np.nan
        Xv[np.isinf(Xv)] = np.nan

        yF = yva[vF]

        def scoreF(zspec, a=A_DEPLOY):
            zf = (1 - a) * z_dep[vF] + a * zspec[vF]
            return bss(yF, shift_to_prior(1 / (1 + np.exp(-zf)), yF.mean()))

        base = bss(yF, shift_to_prior(1 / (1 + np.exp(-z_dep[vF])), yF.mean()))
        print(f"\n  fold {f}   전문가 학습 {int(mF.sum()):,}행 · 검증 F {int(vF.sum()):,}행"
              f" · a={A_DEPLOY} 고정 · 전문가 없음일 때 F행 BSS {base:.1f}")
        print(f"  {'nr':>6}{'F행 L':>10}{'F행 H':>10}{'H - L':>10}"
              f"{'corr(L,H)':>12}{'sd L':>9}{'sd H':>9}")
        for nr in ROUNDS:
            zL = fixed_lgb(Xf, yf, Xv, nr)
            zH = fixed_hgb(Xf, yf, Xv, nr)
            sL, sH = scoreF(zL), scoreF(zH)
            c = float(np.corrcoef(zL[vF], zH[vF])[0, 1])
            print(f"  {nr:>6}{sL:>10.1f}{sH:>10.1f}{sH-sL:>+10.1f}{c:>12.4f}"
                  f"{zL[vF].std():>9.4f}{zH[vF].std():>9.4f}"
                  f"   ({time.time()-t0:.0f}s)")

    print("\n  읽는 법")
    print("    H−L 이 **라운드**를 따라 부호를 바꾸고 두 fold 에서 같은 방향으로")
    print("    움직이면 → §7.219 의 fold 역전은 아티팩트다 (라운드 폴백 200 vs 55).")
    print("    H−L 이 **fold** 를 따라서만 갈리면 → 진짜 시즌 효과다.")


if __name__ == "__main__":
    main()
