"""Trackman **타자** 피처 — 배포 모델의 잔차에 직접 묻는다 (§7.51 ②단계).

`ablate42.py`가 남긴 질문
    대조군 대비로는 정보가 분명히 있다 (tmb_mix가 shuf 대비 +31.9 / +22.0,
    같은 부호·크기비 1.45). 그런데 base 대비로는 fold2024에서 −3.7이다.
    → **정보는 있는데 컬럼으로 넣는 방식이 그걸 못 살린다**는 뜻일 수 있고,
      아니면 **배포 모델이 이미 그 정보를 갖고 있다**는 뜻일 수도 있다.

    이 둘은 ablate로 구분되지 않는다. 잔차 회귀가 구분한다:
        t ≈ 0   → 모델이 이미 안다. 채널이 닫힌다
        |t| 큼  → 모델이 모른다. 전달 방식(사후 로짓 가산)을 바꿔볼 근거가 생긴다

⚠️ |t|가 커도 **채택 근거가 아니다.** §7.39가 투수 쪽에서 확인했듯 t는
   "모델이 모른다"만 말하고 "2025에서 이득이다"는 말하지 않는다.
   결정적인 것은 **부호가 두 fold에서 같은가**다 — 투수 Trackman은 정확히
   여기서 갈렸다(t 부호가 fold마다 반전 → LB −13.6).

실행:  .venv/Scripts/python.exe -u src/tmbresid.py 2>&1 | tee experiments/tmbresid.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

import ctxfit
import foldcache
from config import CTX_AXES, CTX_K, MLP_ALPHA, MODEL_DIR, TARGET_COL, TRAIN_CSV
from ctxsplit import resid_t
from data import load_csv
from features import logit_adjust
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    tmb = pd.read_csv(MODEL_DIR / "trackman_batter.csv")
    cols = [c for c in tmb.columns if c not in ("batter_id", "season")]
    print(f"타자 테이블 {tmb.shape}   컬럼 {len(cols)}개\n")

    cached = foldcache.load(FOLDS, verbose=False)
    tstore = {}

    for fold in FOLDS:
        tr, va = s < fold, s == fold
        d = df[va]
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]

        # 배포 상태 — 블렌드 + δ(hand) + δ(count4)
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        adj, _ = logit_adjust(d, {"ctx": ctx})
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-(z + adj))), yv.mean())
        base = bss(yv, p0)
        r = yv - p0

        sub = tmb[tmb.season == fold].set_index("batter_id")
        bid = d["batter_id"].to_numpy(dtype=np.int64)
        hit = np.isin(bid, sub.index.to_numpy())

        print("=" * 92)
        print(f"fold {fold}   기준 BSS(배포상태) {base:.1f}   타자 테이블 조회 적중 {hit.mean():.1%}")
        print("=" * 92)
        print(f"  {'컬럼':<26}{'결측':>8}{'잔차 t':>10}{'|t|>2':>7}")
        for c in cols:
            v = sub[c].reindex(bid).to_numpy(dtype=np.float64)
            miss = np.isnan(v)
            if miss.all():
                continue
            v = np.where(miss, np.nanmean(v), v)
            _, t = resid_t(r, v)
            tstore.setdefault(c, {})[fold] = t
            print(f"  {c:<26}{miss.mean():>7.1%}{t:>10.2f}{'  ✓' if abs(t) > 2 else '':>7}")

    print("\n" + "=" * 92)
    print("부호 일치 판정 — §7.39가 투수 Trackman을 죽인 바로 그 검사")
    print("=" * 92)
    print(f"  {'컬럼':<26}{'f2023 t':>10}{'f2024 t':>10}   판정")
    flip, agree = [], []
    for c, byf in tstore.items():
        a, b = byf.get(2023), byf.get(2024)
        if a is None or b is None:
            continue
        same = (a > 0) == (b > 0)
        strong = min(abs(a), abs(b)) > 2
        tag = ("★ 반전" if not same else ("일치 (둘 다 유의)" if strong else "일치 (약함)"))
        (agree if same else flip).append(c)
        print(f"  {c:<26}{a:>+10.2f}{b:>+10.2f}   {tag}")

    print(f"\n  부호 일치 {len(agree)}개 / 반전 {len(flip)}개")
    if flip:
        print(f"  반전: {flip}")
    strong_agree = [c for c in agree
                    if min(abs(tstore[c][2023]), abs(tstore[c][2024])) > 2]
    print(f"  **양 fold |t|>2 이면서 부호 일치: {len(strong_agree)}개**")
    if strong_agree:
        print(f"    {strong_agree}")
        print("\n  → 이 컬럼들만 사후 로짓 가산(β 스윕)으로 재볼 근거가 있다.")
        print("     컬럼 추가가 아니라 가산이므로 ablate42가 본 '컬럼 비용'을 안 낸다.")
    else:
        print("\n  → 부호가 안정하면서 유의한 컬럼이 없다. 이 채널은 닫힌다.")


if __name__ == "__main__":
    main()
