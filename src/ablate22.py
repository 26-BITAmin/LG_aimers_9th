"""22차 실험 — **새 fold 설계를 검증한다.** (fold를 바꾸는 것은 판정 기준을 바꾸는 일이다)

`foldsurvey.py`가 밝힌 것:
    2021  973.2   2022 1849.2   **2023  -10.2**   2024  741.9
    2023 후반기 676.5           2024 후반기 656.1
2022→2023 구조적 단절 때문에 fold2023의 모델은 std(z)=0.2088로 **자신 있게 틀린다.**
그래서 fold2023은 "실력이 느는가"가 아니라 "망가진 모델이 얼마나 구제되는가"를 쟀다.

새 구성: **2023H2(676.5) + 2024(741.9)** — 검증 구간이 겹치지 않고, 둘 다 단절 이후이며,
신호 크기가 비슷하다(기존 40.7 대 769.6은 19배 차이라 크기비 검사가 성립하지 않았다).

**새 설계가 옳다는 것을 따로 보여야 한다.** 결과를 아는 세 변경으로 잰다.

⚠️ **예측을 먼저 박는다** (결과를 보고 성공 기준을 맞추지 않기 위해):

  ① α .30→.40   구CV +1.7  LB **+6.7**
     예측: 새 CV가 +1.7보다 **크게** 나온다. 구 CV는 분해능(±1)이 효과보다 작아
           애초에 못 짚었다. 방향이라도 또렷해지면 성과다.

  ② MLP 3→8     구CV +4.1  LB **+0.24**
     예측: 새 CV가 **작아진다**. 구 CV의 과대평가에 fold2023 기여가 컸을 것이다.

  ③ 피처 51개   구CV +13.3 LB **−2.13**
     예측: **새 CV도 양수로 나온다. 즉 이 실패는 fold 재설계로 안 고쳐진다.**
           fold2024 단독이 이미 +15.2였다 — fold 노이즈가 아니라 2025 전이 문제다.
           ③이 예측대로면 (b)의 한계가 확정된다: fold 재설계는 조건 ③의 검정력을
           올릴 뿐, **정보를 삭감하는 변경의 전이 실패는 못 막는다.**
           (그건 조건 ③이 이미 4.9로 경고했고, 내가 무시했던 것이다.)

①②는 캐시에서 즉시 나온다. ③은 DROP51로 LGBM 앙상블을 다시 학습해야 한다
(s05→s06 비교가 MLP 이전이었으므로 여기서도 LGBM만 쓴다 — 같은 조건으로 맞춘다).

실행:  .venv/Scripts/python.exe -u src/ablate22.py [--skip-features]
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

import foldcache
import mlp
from ablate14 import cols
from ablate20 import se_bss
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA, N_ROUNDS,
                    TRAIN_CSV)
from data import load_csv

NEW = ["2023H2", 2024]
OLD_CV = {"α .30→.40": 1.7, "MLP 3→8": 4.1, "피처 51개": 13.3}
LB = {"α .30→.40": 6.7, "MLP 3→8": 0.24, "피처 51개": -2.13}
DROP51 = cols("달력", "점수", "결측", "구종", "승부처",
              "폼차분", "실패구성", "표본수", "매치업")


def report(name, per_fold, folds):
    d = np.array([per_fold[f] for f in folds])
    same = bool(np.all(d > 0) or np.all(d < 0))
    sp = np.abs(d).max() / max(np.abs(d).min(), 1e-9) if same else np.inf
    ok = same and d.mean() > 0 and sp <= 3
    print(f"{name:<12s} " + " ".join(f"{f}={per_fold[f]:>+8.2f}" for f in folds)
          + f"   평균 {d.mean():>+7.2f}   크기비 "
          + (f"{sp:>5.1f}" if same else "  inf")
          + f"   구CV {OLD_CV[name]:>+6.1f}   LB {LB[name]:>+6.2f}"
          + ("   [채택]" if ok else "   [기각]"))
    return d.mean(), ok


def main():
    t0 = time.time()
    cache = foldcache.load(NEW)
    Z, Y, res = {}, {}, {}

    print()
    for f in NEW:
        z_lgb, z_mlp, y, mem = cache[f]
        Z[f], Y[f] = (z_lgb, z_mlp), y
        base = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
        print(f"fold {str(f):<7s} n={len(y):>8,}  lgb {mlp.score(z_lgb, y):>7.1f}"
              f"  mlp {mlp.score(z_mlp, y):>7.1f}  blend {mlp.score(base, y):>7.1f}"
              f"  SE {se_bss(base, base * 0.999, y) * 1000:>5.2f}")
        if mem is None:
            print("  ⚠️ 이 캐시에는 멤버별 z가 없다 — ②는 건너뛴다 (캐시 재생성 필요)")

    print(f"\n{'변경':<12s} " + " " * 34 + "판정")
    print("-" * 96)

    # ① α  — 캐시에서 즉시
    per = {}
    for f in NEW:
        zl, zm = Z[f]
        per[f] = (mlp.score(0.60 * zl + 0.40 * zm, Y[f])
                  - mlp.score(0.70 * zl + 0.30 * zm, Y[f]))
    res["α .30→.40"] = report("α .30→.40", per, NEW)

    # ② MLP 멤버 3→8 — 멤버별 z가 있어야 한다. s07은 emb4 3시드였다
    if all(cache[f][3] is not None for f in NEW):
        per = {}
        for f in NEW:
            zl, _, y, mem = cache[f]
            z3, z8 = mem[:3].mean(axis=0), mem.mean(axis=0)
            per[f] = (mlp.score(0.60 * zl + 0.40 * z8, y)
                      - mlp.score(0.60 * zl + 0.40 * z3, y))
        res["MLP 3→8"] = report("MLP 3→8", per, NEW)

    # ③ 피처 51개 — 재학습이 필요하다 (LGBM 앙상블만, s05→s06과 같은 조건)
    if "--skip-features" not in sys.argv:
        df = load_csv(TRAIN_CSV)
        per = {}
        for f in NEW:
            Xtr, ytr, w, Xva, yva, _ = foldcache.fold_data(df, f)
            out = {}
            for tag, drop in (("80", []), ("51", DROP51)):
                cs = [c for c in Xtr.columns if c not in drop]
                z, n = np.zeros(len(Xva)), 0
                for over in ENSEMBLE_VARIANTS.values():
                    for s in foldcache.LGB_SEEDS:
                        p = dict(FINAL_LGB_PARAMS, **over, seed=s, bagging_seed=s,
                                 feature_fraction_seed=s, data_random_seed=s)
                        bst = lgb.train(p, lgb.Dataset(Xtr[cs], ytr, weight=w),
                                        num_boost_round=N_ROUNDS)
                        z += bst.predict(Xva[cs], raw_score=True)
                        n += 1
                        del bst
                out[tag] = mlp.score(z / n, yva)
                print(f"  fold {f} 피처 {tag}개({len(cs)}) = {out[tag]:.1f}"
                      f"   ({time.time()-t0:.0f}s)")
            per[f] = out["51"] - out["80"]
        res["피처 51개"] = report("피처 51개", per, NEW)

    print("\n  판정:")
    for k, (m, ok) in res.items():
        agree = "부호 일치" if np.sign(m) == np.sign(LB[k]) else "**부호 불일치**"
        print(f"    {k:<12s} 새CV {m:>+7.2f}  vs  LB {LB[k]:>+6.2f}   {agree}")
    print("\n  ③이 여전히 양수면 fold 재설계로는 정보 삭감의 전이 실패를 못 막는다는 뜻이다.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
