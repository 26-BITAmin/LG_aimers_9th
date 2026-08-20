"""`feature_importance('split'/'gain')`이 **판별 도구가 되는가** (사용자가 가져온 제안 검증).

받은 제안
    booster.feature_importance('split')   # g의 스플릿 횟수
    booster.feature_importance('gain')    # g의 gain 총합

API는 맞다 (LightGBM `Booster.feature_importance(importance_type)`, 기본 'split').
저장소도 이미 쓴다 — `cv.py:112`, `ablate7.py:63`. 다만 스니펫대로면 **전 피처 배열**이
나오므로 `feature_name()`으로 인덱싱해야 한다 (저장소는 `pd.Series(..., index=cols)`).

⛔ 진짜 질문은 API가 아니라 **판별력**이다. importance는 **학습 데이터의 트리 구조**에서
   계산된다. "모델이 이 피처를 많이 쓴다"만 말하고 "2025에서 도움이 된다"는 말하지 않는다.

이 저장소에는 그 반례가 이미 측정돼 있다 (§7.56)
    `gap_success` + `n_szn` 컬럼:  홀드아웃 **+73.1**  →  LB **+0.44**   배율 **68.6**
    importance는 이 컬럼에 크게 붙었을 것이다. 그런데 전이는 1/69였다.

여기서 재는 것 — **신호가 0인 컬럼에 importance가 얼마나 붙는가**
    `col1n`       gap_success + n_szn                      (진짜 신호)
    `col1n_shuf`  ★ 대조군 — 시즌 안에서 pitcher_id 재배정  (**신호 0. 구성상 보장**)

    셔플 컬럼의 importance가 진짜 컬럼과 비슷하면
    → importance는 **신호와 노이즈조차 구별하지 못한다.** 전이는 말할 것도 없다.
    훨씬 낮으면 → 최소한 "신호 있음/없음"은 가른다 (전이 판별은 여전히 못 한다).

    ⚠️ 판정 기준을 미리 못 박는다: **셔플/진짜 importance 비가 0.3 미만**이어야
       "신호 유무는 가른다"고 인정한다. 홀드아웃 BSS 차이를 같이 찍어 대조한다.

⚠️ LGBM 단독 하네스다 (§7.51·§7.56과 같은 자리). 절대값이 아니라 대비를 본다.

실행:  .venv/Scripts/python.exe -u src/impcheck.py 2>&1 | tee experiments/impcheck.log
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
import pandas as pd

import foldcache
from asofcol import make_extra
from config import FINAL_LGB_PARAMS, N_ROUNDS, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

FOLD = 2024
SEEDS = [42, 7, 2024]
NEW = ["gap_success", "n_szn"]


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    Xtr, ytr, w, Xva, yva, art, tr = foldcache.fold_data(df, FOLD)
    print(f"fold {FOLD}  train {Xtr.shape}  valid {Xva.shape}  ({time.time()-t0:.0f}s)")

    out = {}
    for tag, shuf in (("col1n", False), ("col1n_shuf", True)):
        fit = make_extra(NEW, shuffle=shuf)(tr)
        A, B = Xtr.copy(), Xva.copy()
        for c, v in fit(tr).items():
            A[c] = v.astype(np.float32)
        for c, v in fit(df[df.season == FOLD]).items():
            B[c] = v.astype(np.float32)
        cols = list(A.columns)

        sp, gn, sc = [], [], []
        for s in SEEDS:
            p = dict(FINAL_LGB_PARAMS, seed=s, bagging_seed=s,
                     feature_fraction_seed=s, data_random_seed=s)
            bst = lgb.train(p, lgb.Dataset(A, ytr, weight=w),
                            num_boost_round=N_ROUNDS)
            sp.append(pd.Series(bst.feature_importance("split"), index=cols))
            gn.append(pd.Series(bst.feature_importance("gain"), index=cols))
            z = bst.predict(B, raw_score=True)
            sc.append(bss(yva, shift_to_prior(1 / (1 + np.exp(-z)), yva.mean())))
            del bst
        out[tag] = dict(split=pd.concat(sp, axis=1).mean(axis=1),
                        gain=pd.concat(gn, axis=1).mean(axis=1),
                        bss=np.array(sc))
        print(f"  {tag:<12} BSS {out[tag]['bss'].mean():>8.1f}"
              f" ± {out[tag]['bss'].std(ddof=1):>5.1f}   ({time.time()-t0:.0f}s)")

    # 기준선 (새 컬럼 없음)
    sc0 = []
    for s in SEEDS:
        p = dict(FINAL_LGB_PARAMS, seed=s, bagging_seed=s,
                 feature_fraction_seed=s, data_random_seed=s)
        bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=N_ROUNDS)
        z = bst.predict(Xva, raw_score=True)
        sc0.append(bss(yva, shift_to_prior(1 / (1 + np.exp(-z)), yva.mean())))
        del bst
    sc0 = np.array(sc0)
    print(f"  {'base':<12} BSS {sc0.mean():>8.1f} ± {sc0.std(ddof=1):>5.1f}"
          f"   ({time.time()-t0:.0f}s)")

    n_feat = len(out["col1n"]["split"])
    print(f"\n{'='*82}\n■ 새 컬럼의 importance — 진짜 신호 vs **신호 0 대조군**\n{'='*82}")
    print(f"  전체 피처 {n_feat}개. 순위는 1위가 가장 큼.\n")
    for kind in ("split", "gain"):
        print(f"  ── importance_type = '{kind}' " + "─" * 46)
        print(f"    {'컬럼':<16}{'col1n':>14}{'순위':>7}{'col1n_shuf':>14}{'순위':>7}"
              f"{'셔플/진짜':>11}")
        for c in NEW:
            a, b = out["col1n"][kind], out["col1n_shuf"][kind]
            ra = int((a > a[c]).sum()) + 1
            rb = int((b > b[c]).sum()) + 1
            print(f"    {c:<16}{a[c]:>14,.0f}{ra:>7}{b[c]:>14,.0f}{rb:>7}"
                  f"{b[c]/max(a[c],1e-9):>11.2f}")
        print()

    print(f"{'='*82}\n■ 판정\n{'='*82}")
    d1 = out["col1n"]["bss"].mean() - sc0.mean()
    d2 = out["col1n_shuf"]["bss"].mean() - sc0.mean()
    print(f"  홀드아웃 BSS   base {sc0.mean():.1f}"
          f"   col1n {out['col1n']['bss'].mean():.1f} ({d1:+.1f})"
          f"   shuf {out['col1n_shuf']['bss'].mean():.1f} ({d2:+.1f})")
    print(f"  → 신호 유무의 **정답**: col1n {d1:+.1f} vs shuf {d2:+.1f}"
          f"   (차이 {d1-d2:+.1f})")
    for kind in ("split", "gain"):
        a, b = out["col1n"][kind], out["col1n_shuf"][kind]
        r = sum(b[c] for c in NEW) / max(sum(a[c] for c in NEW), 1e-9)
        print(f"  → '{kind}' 셔플/진짜 비 = {r:.2f}"
              f"   {'✅ 신호 유무는 가른다 (<0.3)' if r < 0.3 else '❌ 신호와 노이즈를 구별 못 한다'}")
    print("\n  ⛔ 어느 쪽이든 **전이 판별은 못 한다** — §7.56에서 이 컬럼 조합은")
    print("     홀드아웃 +73.1 / LB +0.44 (배율 68.6)였다. importance는 그때도 컸다.")


if __name__ == "__main__":
    main()
