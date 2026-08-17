"""시즌 진행분 — **사후 로짓 가산 β 스윕.** 학습이 없다.

§7.54가 남긴 것: `gap_success` 잔차 t = +16.05 / +8.02 (대조군 −0.68 / +0.12).
t는 "모델이 모른다"만 말한다. 여기서 묻는 것은 **"그래서 점수가 오르나"** 다.

전달 방식
    z' = z + β · v      (v = 표준화된 시즌 진행분 피처)
    컬럼 추가와 달리 **학습 비용도, §7.51이 본 '컬럼 추가 비용'도 안 낸다.**
    δ 축(§7.36·§7.37)이 채택된 것과 같은 경로다.

⚠️ 표준화는 **학습 시즌 통계로만** 한다 (tmbfit의 교훈). 검증 행의 평균·표준편차를
   쓰면 그 자체가 리크다. 배포에서는 artifacts에 얼린다.

판정 기준 — §7.39 ③ 이 세워둔 것
    δ(count4)   β 0.25~2.0 (8배 구간) 전체에서 양 fold 양수  → 채택
    tm_ext_std  유효 β 폭 2배뿐                              → 기각
    tmb_*       β=0.05에서 이미 fold2024 음수, 크기비 6~10   → 기각

  → **고원(plateau)이 있어야 한다.** 한 점에서만 양수면 그건 노이즈에 맞춘 것이다.

⚠️ fold2023은 기준 BSS 130.2인 단절 fold다. §7.51에서 대조군조차 +25.9를 받았고
   §7.54 잔차 검정에서도 대조군 t가 +8.06까지 나왔다. **판정은 fold2024 중심으로**
   하고 fold2023은 부호 확인용으로만 읽는다.

실행:  .venv/Scripts/python.exe -u src/asofbeta.py 2>&1 | tee experiments/asofbeta.log
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
import foldcache
from config import (ASOF_AXES, ASOF_STRENGTH, CTX_AXES, CTX_K, MLP_ALPHA, SEED,
                    TARGET_COL, TRAIN_CSV)
from data import load_csv
from features import logit_adjust
from mixresid import build
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
# ⚠️ **부호가 있는 그리드**여야 한다. 초안은 양의 β만 쓸었는데, `gap_middle`은
#    잔차 기울기가 음수(t −4.52/−2.92)라 음의 β로 얹어야 맞는다. 양수만 쓸면
#    "고원 없음"이 나오는데 그건 피처의 성질이 아니라 **그리드의 성질**이다.
BETAS = [-0.08, -0.05, -0.03, -0.02, -0.01, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08]
# §7.54 잔차 검정에서 부호 일치 + 대조군 초과를 통과한 것만.
CAND = ["gap_success", "r_szn_success", "gap_middle"]


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(SEED)
    out, combo = {}, {}

    for fold in FOLDS:
        tr, va = s < fold, s == fold
        d = df[va].copy()
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        adj, _ = logit_adjust(d, {"ctx": ctx})
        z0 = z + adj
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean())
        base = bss(yv, p0)

        art = asofdelta.fit_offsets(df[tr])
        league = float(df.loc[tr, TARGET_COL].mean())
        fv = build(d, art, league)
        # 표준화 통계는 **학습 시즌 행**에서만 뽑는다 (배포에서는 artifacts에 동결)
        ftr = build(df[tr], art, league)

        dsh = d.copy()
        dsh["pitcher_id"] = rng.permutation(d["pitcher_id"].to_numpy())
        fsh = build(dsh, art, league)

        def std(v, ref):
            mu, sd = float(np.nanmean(ref)), float(np.nanstd(ref))
            return np.nan_to_num((v - mu) / (sd if sd > 0 else 1.0))

        def gain(v, b):
            return bss(yv, shift_to_prior(1.0 / (1.0 + np.exp(-(z0 + b * v))),
                                          yv.mean())) - base

        print(f"\nfold {fold}  기준 {base:.1f}   행 {len(d):,}")
        print(f"  {'피처':<22}" + "".join(f"{'β'+str(b):>9}" for b in BETAS))
        vecs = {}
        for c in CAND + ["gap_success[대조군]"]:
            src = fsh if "대조군" in c else fv
            key = c.replace("[대조군]", "")
            v = std(src[key], ftr[key])
            vecs[c] = v
            row = [gain(v, b) for b in BETAS]
            out.setdefault(c, {})[fold] = row
            print(f"  {c:<22}" + "".join(f"{g:>+9.2f}" for g in row))

        # 조합 — 개별 최적 β를 그대로 얹었을 때 더해지나 (겹치면 안 더해진다)
        for pair in (("gap_success", "gap_middle"),
                     ("gap_success", "r_szn_success")):
            best = {}
            for c in pair:
                r = out[c][fold]
                best[c] = BETAS[int(np.argmax(r))]
            v = sum(best[c] * vecs[c] for c in pair)
            g = gain(v, 1.0)
            combo.setdefault(pair, {})[fold] = (g, dict(best))
            print(f"  {'+'.join(pair):<22}{g:>+9.2f}   (β {best})")

    print("\n" + "=" * 92)
    print("판정 — 부호 · 크기비 · β 고원")
    print("=" * 92)
    for c, byf in out.items():
        a, b = np.array(byf[2023]), np.array(byf[2024])
        both = (a > 0) & (b > 0)
        i = int(np.argmax(np.minimum(a, b))) if both.any() else int(np.argmax(b))
        ratio = max(abs(a[i]), abs(b[i])) / max(min(abs(a[i]), abs(b[i])), 1e-9)
        pl = [BETAS[j] for j in range(len(BETAS)) if both[j]]
        span = (max(pl) / min(pl)) if pl else 0.0
        print(f"  {c:<22} 최적 β={BETAS[i]:<5} f2023 {a[i]:+8.2f}  f2024 {b[i]:+8.2f}"
              f"  크기비 {ratio:>5.1f}")
        print(f"  {'':22} 양 fold 양수 β: {pl if pl else '없음'}   (폭 {span:.0f}배)")
    print("\n  조합:")
    for pair, byf in combo.items():
        print(f"  {'+'.join(pair):<22} f2023 {byf[2023][0]:+8.2f}  f2024 {byf[2024][0]:+8.2f}")
    print("\n  기준: δ(count4) 8배 구간 전체 양수 → 채택 / tm_ext_std 폭 2배 → 기각")
    print("  ⚠️ fold2023은 단절 fold(기준 130.2). 판정은 fold2024와 대조군 대비로 한다.")


def incr():
    """★ **배포 상태(+ asof gap β) 위에** 새 축을 얹는 증분 스윕 (§7.57).

    §7.37이 count4를 δ_hand 위의 증분으로 잰 것과 같은 설계다. 축을 쌓을 때
    기준선에 이전 축이 빠져 있으면 같은 신호를 두 번 세게 된다.

    후보 (`mixresid.incremental` 통과):
        gap_bsuccess  증분 t +12.43 / +6.53 — `gap_success`와 **거의 직교**
        gap_reverse   증분 t  −9.40 / −3.30 — 절반쯤 겹치지만 생존
    """
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(SEED)
    from mixresid import GAP_SPEC
    CANDS = ["gap_bsuccess", "gap_reverse"]
    out = {}

    print(f"기준선 = 블렌드 + δ{CTX_AXES} + **asof {ASOF_AXES}** (배포 전부)\n")
    for fold in FOLDS:
        tr, va = s < fold, s == fold
        d = df[va].copy()
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        a = asofdelta.fit_asof_axes(df[tr], ASOF_AXES, ASOF_STRENGTH)
        adj, _ = logit_adjust(d, {"ctx": ctx, "asof": a})
        z0 = z + adj
        base = bss(yv, shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean()))

        art = asofdelta.fit_offsets(df[tr])
        league = float(df.loc[tr, TARGET_COL].mean())
        priors = {c: float(np.nanmean(df.loc[tr, c].to_numpy(dtype=np.float64)))
                  for _, c, _ in GAP_SPEC}
        fv, ftr = build(d, art, league, priors), build(df[tr], art, league, priors)
        dsh = d.copy()
        dsh["pitcher_id"] = rng.permutation(d["pitcher_id"].to_numpy())
        dsh["batter_id"] = rng.permutation(d["batter_id"].to_numpy())
        fsh = build(dsh, art, league, priors)

        def std(v, ref):
            mu, sd = float(np.nanmean(ref)), float(np.nanstd(ref))
            return np.nan_to_num((v - mu) / (sd if sd > 0 else 1.0))

        def gain(v, b):
            return bss(yv, shift_to_prior(1.0 / (1.0 + np.exp(-(z0 + b * v))),
                                          yv.mean())) - base

        print(f"fold {fold}  기준 {base:.1f}")
        print(f"  {'피처':<22}" + "".join(f"{'β'+str(b):>9}" for b in BETAS))
        for c in CANDS + [x + "[대조군]" for x in CANDS]:
            key = c.replace("[대조군]", "")
            v = std((fsh if "대조군" in c else fv)[key], ftr[key])
            row = [gain(v, b) for b in BETAS]
            out.setdefault(c, {})[fold] = row
            print(f"  {c:<22}" + "".join(f"{g:>+9.2f}" for g in row))
        print()

    print("=" * 92)
    print("판정 — 부호 · β 고원 · 대조군")
    print("=" * 92)
    for c, byf in out.items():
        a2, b2 = np.array(byf[2023]), np.array(byf[2024])
        both = (a2 > 0) & (b2 > 0)
        i = int(np.argmax(np.minimum(a2, b2))) if both.any() else int(np.argmax(b2))
        pl = [BETAS[j] for j in range(len(BETAS)) if both[j]]
        # ⚠️ 폭은 **|β| 기준**이다. 음수 β 구간에서 max/min을 그대로 쓰면 0.5 같은
        #    값이 나와 "폭 0배"로 찍힌다 (gap_reverse에서 실제로 그랬다).
        span = (max(map(abs, pl)) / min(map(abs, pl))) if len(pl) > 1 else (1.0 if pl else 0.0)
        print(f"  {c:<22} 최적 β={BETAS[i]:<7} f2023 {a2[i]:+8.2f}  f2024 {b2[i]:+8.2f}"
              f"   양 fold 양수 β: {pl if pl else '없음'} (폭 {span:.0f}배)")
    print("\n  기준: δ(count4) 8배 → 채택 / asof gap 10배 → 채택 / tm_ext_std 2배 → 기각")


if __name__ == "__main__":
    incr() if "--incr" in sys.argv else main()
