"""Trackman 피처 — **현재 모델의 잔차**에 직접 묻는다

왜 다시 묻나
    §7.9는 Trackman을 홀드아웃 +13.3로 채택했다가 LB에서 **−13.6**을 두 번 맞았다.
    그때 적어둔 "가장 유력한 원인"은 **오매핑**이었다 (신인 투수는 투표 경기가 적어
    남의 릴리스포인트를 갖게 된다). `src/tmconf.py`가 그 가설을 죽였다 —
    경기 단위 분할로 624명 전원을 검증하니 일치율 99.92%, ~200투구 구간도 100%다.

    두 번째 가설(시즌 경계 누출)도 죽었다 — `trackman_features.py`는
    **시즌 t 피처 = 시즌 ≤ t−1 누적**으로 올바르게 자르고 있다.

    가설이 둘 다 죽었으니 남은 질문은 하나다. **모델이 이 정보를 이미 갖고 있나?**

무엇이 달라졌나
    §7.9는 LGBM 단독 시절의 측정이다. 그 뒤 MLP 블렌드(§7.16)와 δ 축 둘(§7.36·§7.37)이
    들어와 fold2024 BSS가 742 → **799**로 올랐다. Trackman이 채우던 잔차를 그 사이에
    다른 것이 채웠을 수 있다.

    잔차 회귀는 **학습이 필요 없다.** 캐시된 z에 배포 δ를 얹은 잔차에 회귀할 뿐이다.
    t가 0에 가까우면 "모델이 이미 안다"이고, 그러면 Trackman은 근거를 갖고 닫힌다
    (지금은 '원인 미상'으로 닫혀 있다 — 그건 언제든 다시 열리는 상태다).

⚠️ 잔차 t가 크게 나와도 **채택 근거가 아니다.** LB가 두 번 −13.6이라고 답했다.
   t는 "모델이 모른다"만 말하고, "2025에서 이득이다"는 말하지 않는다.

실행:  .venv/Scripts/python.exe -u src/tmresid.py 2>&1 | tee experiments/tmresid.log
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
# §7.9가 고른 것 — "릴리스포인트 산포 10개"가 가장 좋았다 (전체 28개는 오히려 내려감)
KEY_COLS = ["tm_relh_std", "tm_rels_std", "tm_relpoint_disp",
            "tmfb_relh_std", "tmfb_rels_std", "tmfb_relpoint_disp",
            "tm_ext_std", "tm_relh_mean", "tm_rels_mean", "tm_ext_mean"]


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    tm = pd.read_csv(MODEL_DIR / "trackman_pitcher.csv")
    cols = [c for c in KEY_COLS if c in tm.columns]
    print(f"Trackman 테이블 {tm.shape}   쓰는 컬럼 {len(cols)}개\n")

    cached = foldcache.load(FOLDS, verbose=False)
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

        # 이 fold의 검증 시즌에 해당하는 Trackman 행 (시즌 t 피처 = ≤t-1 누적)
        sub = tm[tm.season == fold].set_index("pitcher_id")
        pid = d["pitcher_id"].to_numpy(dtype=np.int64)
        hit = np.isin(pid, sub.index.to_numpy())

        print("=" * 88)
        print(f"fold {fold}   기준 BSS(배포상태) {base:.1f}   Trackman 조회 적중 {hit.mean():.1%}")
        print("=" * 88)
        print(f"  {'컬럼':<24}{'결측':>8}{'잔차 t':>10}{'|t|>2':>7}")
        ts = []
        for c in cols:
            v = sub[c].reindex(pid).to_numpy(dtype=np.float64)
            miss = np.isnan(v)
            if miss.all():
                continue
            # 결측은 관측 평균으로 채운다 (회귀에서 결측 행이 빠지면 t가 부풀 수 있다)
            v = np.where(miss, np.nanmean(v), v)
            _, t = resid_t(r, v)
            ts.append((c, t))
            print(f"  {c:<24}{miss.mean():>7.1%}{t:>10.2f}{'  ✓' if abs(t) > 2 else '':>7}")
        big = [c for c, t in ts if abs(t) > 2]
        print(f"\n  |t| > 2 인 컬럼 {len(big)}/{len(ts)}개"
              f"   최대 |t| {max(abs(t) for _, t in ts):.2f}")

    print("\n" + "=" * 88)
    print("읽는 법")
    print("=" * 88)
    print("  · |t|가 전반적으로 작다 → 모델이 이미 안다. Trackman은 **근거를 갖고 닫힌다**")
    print("  · |t|가 크다           → 모델이 모른다. 그래도 **채택 근거는 아니다** —")
    print("                           LB가 두 번 −13.6이라고 답했다. t는 전이를 말하지 않는다.")
    print("  · §7.9 당시(LGBM 단독, fold2024 742)와 지금(블렌드+δ, 799)은 다른 모델이다.")


if __name__ == "__main__":
    main()


# ── 2단계: 부호가 일관된 컬럼만 로짓 가산으로 재본다 ──────────────────
# §7.9는 릴리스포인트 10개를 **묶음으로** 넣었고 그중 대부분이 부호 반전 항목이었다.
# `tm_ext_std`는 두 fold 다 양수(+6.63/+3.96)인데 **단독으로 시험된 적이 없다.**
# δ와 같은 기계(사후 로짓 가산 + β 스윕)로 재면 학습이 필요 없다.
#
# ⚠️ δ와 결정적으로 다른 점: 이건 **투수 수준 레벨**이지 자기 평균 대비 편차가 아니다.
#    EDA §G가 "투수 개인차를 과거로 추정하면 상수 모델보다 나빠진다(−93.5)"고 쟀다.
#    그 함정에 정면으로 걸리는 형태이므로, 통과하더라도 근거를 따로 봐야 한다.
BETAS2 = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def stage2():
    from decide import decide
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    tm = pd.read_csv(MODEL_DIR / "trackman_pitcher.csv")
    cached = foldcache.load(FOLDS, verbose=False)
    cand = ["tm_ext_std", "tm_relh_std", "tmfb_relh_std"]   # 부호 일관 3개
    out = {}

    for fold in FOLDS:
        tr, va = s < fold, s == fold
        d = df[va]
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        adj, _ = logit_adjust(d, {"ctx": ctx})
        z0 = z + adj
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean())
        base = bss(yv, p0)
        sub = tm[tm.season == fold].set_index("pitcher_id")
        pid = d["pitcher_id"].to_numpy(dtype=np.int64)

        print(f"\nfold {fold}  기준 {base:.1f}   " + " ".join(f"β{b:<5}" for b in BETAS2))
        for c in cand:
            v = sub[c].reindex(pid).to_numpy(dtype=np.float64)
            m = np.isnan(v)
            # **학습 시즌 통계로만 표준화한다.** 검증 행의 평균/표준편차를 쓰면 리크다.
            trv = tm[tm.season <= fold][c].to_numpy(dtype=np.float64)
            mu, sd = np.nanmean(trv), np.nanstd(trv)
            v = np.where(m, 0.0, (v - mu) / (sd if sd > 0 else 1.0))   # 결측 → 0(=평균)
            row = []
            for b in BETAS2:
                g = bss(yv, shift_to_prior(1.0 / (1.0 + np.exp(-(z0 + b * v))),
                                           yv.mean())) - base
                row.append(g)
            out.setdefault(c, {})[fold] = row
            print(f"  {c:<20}" + " ".join(f"{g:>+6.2f}" for g in row))

    print("\n" + "=" * 88)
    print("판정 (src/decide.py — 조건 ①②③ 순서대로)")
    print("=" * 88)
    for c, byf in out.items():
        a, b = np.array(byf[2023]), np.array(byf[2024])
        i = int(np.argmax(np.minimum(a, b)))
        v, why, extra = decide(c, float(a[i]), float(b[i]))
        print(f"  {c:<20} β={BETAS2[i]:<5} f2023 {a[i]:+6.2f}  f2024 {b[i]:+6.2f}"
              f"   → {v}  {why.splitlines()[0]}")
        for e in extra:
            print(f"       · {e}")


if __name__ == "__main__":
    stage2()
