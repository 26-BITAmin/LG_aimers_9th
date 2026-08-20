"""§7.92·§7.93 드랍 팔을 **배포 최종층**에서 재판정 (사용자: "블렌드에서 끝내지
말고 추가 로짓 쪽도").

블렌드 판정(§7.92 −1.7/−3.1, §7.93 −5.6/−4.1)은 사후 로짓 보정 **이전**이다.
배포는 z_blend + δ_hand + δ_count4 + gap_success + gap_bsuccess → prior shift
(PIPELINE §0)까지가 한 판이고, δ축은 두 팔에 **공통**(train 라벨에서만 fit,
피처와 무관)이지만 시그모이드가 비선형이라 층이 바뀌면 쌍체 Δ가 움직일 수
있다 — jointbeta 계열(§7.83~)에서 층이 결론을 바꾼 전례를 확인하는 절차.

재학습 없음: drop 팔 z는 §7.92(cache/featdrop)·§7.93(cache/featdrop2)의
part 파일을 읽는다. ctl은 foldcache 캐시.

실행:  .venv/Scripts/python.exe -u src/featdrop3.py 2>&1 | tee experiments/featdrop3.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import asofdelta
import ctxfit
import features as F
import foldcache
from config import (ASOF_AXES, ASOF_STRENGTH, CONTAM_AXES, CTX_AXES, CTX_HALF_LIFE,
                    CTX_K, MLP_ALPHA, MLP_SEEDS, MLP_VARIANTS, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from mlp import score
from validation import _sigmoid, shift_to_prior

FOLDS = [2023, 2024]
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
ARMS = {"§7.92 miss3+top_bottom": "featdrop", "§7.93 is_pitcher_home": "featdrop2"}


def paired_p(pA, pB, y):
    """확률로 이미 만들어진 두 팔의 쌍체 Δ(BSS, B−A) ± SE."""
    d = (pA - y) ** 2 - (pB - y) ** 2
    ref = y.mean() * (1 - y.mean())
    k = 1e5 / ref
    return k * d.mean(), k * d.std(ddof=1) / np.sqrt(len(d))


def drop_blend(sub, f):
    """part 파일에서 drop 팔의 블렌드 로짓을 복원한다."""
    z_lgb = np.load(CACHE / sub / f"fold{f}_lgb.npz")["z"]
    zs = [np.load(CACHE / sub / f"fold{f}_mlp_{v}_{s}.npz")["z"]
          for v, (_, n) in MLP_VARIANTS.items() for s in MLP_SEEDS[:n]]
    return (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * np.mean(zs, axis=0), len(zs)


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)

    for f in FOLDS:
        tr, va = df[s < f], df[s == f].copy()
        y = cached[f][2]
        z_ctl = (1 - MLP_ALPHA) * cached[f][0] + MLP_ALPHA * cached[f][1]

        # 배포와 같은 사후 보정 (rookie.deployed_pred와 동일 구성, 두 팔 공통)
        ref = int(tr["season"].max())
        ctx = ctxfit.fit_ctx(tr, TARGET_COL, CTX_AXES, CTX_K,
                             weights=recency_weight(s[s < f], CTX_HALF_LIFE,
                                                    ref_season=ref))
        art = F.fit_artifacts(tr, ctx=ctx, contam_axes=CONTAM_AXES)
        art["asof"] = asofdelta.fit_asof_axes(tr, ASOF_AXES, ASOF_STRENGTH,
                                              ct=art.get("contam"))
        adj, _ = F.logit_adjust(va, art)

        print(f"\nfold {f}  n={len(y):,}  |adj| 평균 {np.abs(adj).mean():.4f}")
        for name, sub in ARMS.items():
            z_drop, nm = drop_blend(sub, f)
            # 블렌드 층 (로그 대조용) — 각 팔 oracle shift 후 쌍체
            pA = shift_to_prior(_sigmoid(z_ctl), y.mean())
            pB = shift_to_prior(_sigmoid(z_drop), y.mean())
            db, sb = paired_p(pA, pB, y)
            # 배포 최종층 — 같은 adj를 두 팔에 얹고 shift
            qA = shift_to_prior(_sigmoid(z_ctl + adj), y.mean())
            qB = shift_to_prior(_sigmoid(z_drop + adj), y.mean())
            dd, sd = paired_p(qA, qB, y)
            print(f"  [{name}] (mlp {nm}개)")
            print(f"    블렌드층  ctl {score(z_ctl, y):>7.1f}  drop {score(z_drop, y):>7.1f}"
                  f"   Δ {db:+6.2f} ± {sb:.2f}  (t {db/sb:+.1f})")
            print(f"    배포층    ctl {score(np.log(qA/(1-qA)), y):>7.1f}"
                  f"  drop {score(np.log(qB/(1-qB)), y):>7.1f}"
                  f"   Δ {dd:+6.2f} ± {sd:.2f}  (t {dd/sd:+.1f})")


if __name__ == "__main__":
    main()
