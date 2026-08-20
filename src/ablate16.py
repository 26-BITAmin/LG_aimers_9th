"""16차 실험 — 피처 제거 이득이 **앙상블 위에서도 남는가**.

`ablate14`에서 29개 제거(80→51)가 단일 모델 기준 +45.1 (t=4.7, 부호 일치)로 나왔다.
그대로 채택하고 싶지만, 바로 직전 `ablate15`에서 정확히 이 함정을 밟았다:

    CatBoost 단독 핸디캡 -26.7 (vs 단일 lgbm)  →  앙상블에 얹으니 -0.6 ~ -5.5
    이유: 각 패밀리를 먼저 평균내면 노이즈가 상쇄되고 공유 신호만 남아
          패밀리 간 상관이 0.985로 올라간다. 단일끼리는 달라 보여도 앙상블끼리는 같다.

**피처 제거도 앙상블도 둘 다 분산 축소다.** 겹칠 수 있다.
그러므로 배포와 같은 구조(ENSEMBLE_VARIANTS x 시드, 로짓 평균) 위에서 다시 잰다.

한 스크립트 안에서 80개 / 51개를 **같은 시드로** 돌려 짝지은 비교를 만든다.
(ablate15의 lgb_ens 값을 가져다 쓰면 시드·환경이 어긋날 수 있어 다시 돌린다.)

실행:  .venv/Scripts/python.exe -u src/ablate16.py
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

from ablate14 import G, cols, fold_data
from config import ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, N_ROUNDS, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
SEEDS = (42, 7, 2024)

DROP51 = cols("달력", "점수", "결측", "구종", "승부처",
              "폼차분", "실패구성", "표본수", "매치업")
DROP63 = cols("달력", "점수", "결측", "구종", "승부처")

CONFIGS = {"80개 (현행)": [], "63개": DROP63, "51개": DROP51}


def score(z, y):
    return bss(y, shift_to_prior(1 / (1 + np.exp(-z)), y.mean()))


def ens_member_logits(Xtr, ytr, w, Xva, t0, tag):
    """배포와 같은 구조. 멤버별 로짓을 따로 돌려준다 (앙상블 크기 효과도 보려고)."""
    zs = []
    for vname, over in ENSEMBLE_VARIANTS.items():
        for s in SEEDS:
            p = dict(FINAL_LGB_PARAMS, **over, seed=s, bagging_seed=s,
                     feature_fraction_seed=s, data_random_seed=s)
            bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=N_ROUNDS)
            zs.append(bst.predict(Xva, raw_score=True))
    print(f"    {tag}: {len(zs)}부스터 완료  ({time.time()-t0:.0f}s)")
    return np.array(zs)


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    res = {k: {} for k in CONFIGS}
    solo = {k: {} for k in CONFIGS}

    for f in FOLDS:
        print(f"\n=== valid={f} ===")
        Xtr, ytr, w, Xva, yva = fold_data(df, f)
        for name, drop in CONFIGS.items():
            keep = [c for c in Xtr.columns if c not in set(drop)]
            Z = ens_member_logits(Xtr[keep], ytr, w, Xva[keep], t0, f"{name}({len(keep)})")
            res[name][f] = score(Z.mean(axis=0), yva)
            solo[name][f] = float(np.mean([score(z, yva) for z in Z]))
            print(f"      앙상블 {res[name][f]:>8.1f}   (단독 평균 {solo[name][f]:>8.1f}"
                  f"  앙상블 이득 {res[name][f]-solo[name][f]:>+6.1f})")

    base = "80개 (현행)"
    print(f"\n{'설정':<12s} " + " ".join(f"{f:>9d}" for f in FOLDS)
          + f" {'평균':>9s} {'기준대비':>9s} {'크기비':>7s}")
    print("-" * 60)
    for name in CONFIGS:
        v = np.array([res[name][f] for f in FOLDS])
        d = v - np.array([res[base][f] for f in FOLDS])
        same = bool(np.all(d > 0) or np.all(d < 0))
        sp = np.abs(d).max() / max(np.abs(d).min(), 1e-9) if same else np.inf
        tail = "" if name == base else (f" {d.mean():>+9.1f} "
                                        + (f"{sp:>7.1f}" if same else f"{'inf':>7s}"))
        print(f"{name:<12s} " + " ".join(f"{x:>9.1f}" for x in v) + f" {v.mean():>9.1f}{tail}")

    print("\n=== 단독 vs 앙상블: 이득이 겹치는가 ===")
    print("  피처 제거 이득이 앙상블 위에서 줄어들면 두 분산 축소가 겹친다는 뜻이다.")
    print(f"  {'설정':<12s} {'단독 기준대비':>14s} {'앙상블 기준대비':>16s} {'잔존율':>8s}")
    for name in CONFIGS:
        if name == base:
            continue
        ds = np.mean([solo[name][f] - solo[base][f] for f in FOLDS])
        de = np.mean([res[name][f] - res[base][f] for f in FOLDS])
        print(f"  {name:<12s} {ds:>+14.1f} {de:>+16.1f} {de/ds*100 if ds else 0:>7.0f}%")

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
