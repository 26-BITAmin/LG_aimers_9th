"""§7.204 — `gap_strike` 재개봉. **선별 게이트가 틀린 기준선 위에서 계산됐다.**

발단: 사용자가 다른 세션에서 나온 `gap_strike` 를 다시 보라고 했다.

두 번의 폐쇄가 있었고 둘 다 근거가 불완전하다.

    §7.57 (우리)   1단계 잔차 t  f2023 −4.68 / f2024 +4.05  ->  "부호 반전" 기각
    혜진 §11-2     분산 기반      84.4 vs 위약 83.6 = 실질 ≈ +1  ->  `gap_ball` 열세로 종결
                   (혜진 자신의 종결 표: **"전이 0.380/미측정"** — 전이를 안 쟀다)

★ 재개봉 사유 — `mixresid.main()` 의 선별 순서

    incremental(df, s, cached, [k for k, v in keep.items()
                                if (v[0][0] > 0) == (v[1][0] > 0) ...])   # <- 1단계 게이트

    **증분 측정은 1단계를 통과한 축만 받는다.** 그런데 `incremental()` 자신의
    docstring 이 이렇게 적어놨다:

        "위 1단계 기준선에는 배포된 asof 항이 **빠져 있어서** 같은 신호를
         두 번 세고 있을 수 있다"

    즉 **누가 올바른(증분) 측정을 받을지 정하는 게이트가, 틀린(비증분) 기준선
    위에서 계산된다.** 배포 asof 항과 상관된 축은 구조적으로 오선별된다.
    ([[telescoping-chains-are-not-evidence]] 의 사촌 — 경로가 판정을 정한다.)

★ 그리고 `gap_strike` 가 정확히 그 축이다 (이번에 실측)

    corr(gap_ball, gap_strike) = **−0.658 (f2023) / −0.711 (f2024)**
    시즌분 middle+ball+strike 합 = 0.974~0.991, 1~99 백분위 [0.833, 1.172]
      -> 거의 조성이지만 **닫힌 단체가 아니다**. 30~50% 직교 성분이 남는다.
         ([[check-the-simplex-before-a-composition-transform]])

★ 상태 변화 — `gap_ball` 은 이제 **배포돼 있다**

    `sy02_c4.zip:model/artifacts.joblib` -> artifacts["asof"]
        gap_success   β +0.03
        gap_bsuccess  β +0.03
        gap_ball      β **−0.0038**      <- 혜진 라운드 11. §7.57 시점엔 없었다

    ⚠️ `model/meta.json` 은 앞의 둘만 적어놨다 (`gap_ball` 누락). 진실원은
       `artifacts.joblib` 이다 — 혜진 `tools/fix_meta.py` 가 경고한 그 거짓말이고,
       이 조사에서 실제로 한 번 나를 속였다. ([[audit-inherited-constants-not-just-registered-gates]])

    ⚠️ 우리 `config.ASOF_AXES` 에도 `gap_ball` 이 없다 — 로컬 설정이 배포보다 뒤처져 있다.

설계 — 2팔로 "이미 사고 있는가" 를 직접 가른다

    A팔   기준선 = 블렌드 + ctx δ + asof(gap_success, gap_bsuccess)          <- §7.57 이 쓴 것
    B팔   기준선 = A + **asof(gap_ball, −0.0038)**                          <- 실제 배포 상태

    같은 축을 두 기준선의 잔차에 회귀한다. A에서 살고 B에서 죽으면
    **`gap_ball` 이 이미 사고 있는 신호**다. 둘 다 살면 새 축이다.
    ([[placebo-pass-doesnt-mean-incremental]] — 위약은 "실재하는가", 2×2는 "이미 사고 있는가")

    대조군은 §7.57 과 같다 — 검증 시즌 안에서 pitcher_id·batter_id 를 재배정해
    컬럼 수·주변분포·결측 패턴을 보존한 채 **엔티티↔피처 연결만 끊는다.**

    ⚠️ 1단계 게이트를 **적용하지 않는다.** 전 축을 증분까지 끌고 간다 — 그것이
       이 실험의 요점이다.

실행:  .venv/Scripts/python.exe -u src/gapstrike.py > experiments/gapstrike.log 2>&1
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
from config import (ASOF_STRENGTH, CTX_AXES, CTX_K, MLP_ALPHA, SEED, TARGET_COL,
                    TRAIN_CSV)
from ctxsplit import resid_t
from data import load_csv
from features import logit_adjust
from mixresid import GAP_SPEC, build
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]

# 배포 진실원(`artifacts.joblib`)에서 읽은 값. `config.ASOF_AXES` 가 아니다.
DEPLOYED_BASE = [("gap_success", 0.03), ("gap_bsuccess", 0.03)]
GAP_BALL = ("gap_ball", -0.0038)

# `ASOF_SPEC` 에 없는 축을 δ 로 얹으려면 명세가 필요하다 (배포 등록이 아니라 측정용).
EXTRA_SPEC = {
    "gap_ball":   dict(id="pitcher_id", n_col="asof_pitcher_n",
                       r_col="asof_pitcher_ball_rate"),
    "gap_strike": dict(id="pitcher_id", n_col="asof_pitcher_n",
                       r_col="asof_pitcher_strike_rate"),
    "gap_middle": dict(id="pitcher_id", n_col="asof_pitcher_n",
                       r_col="asof_pitcher_middle_rate"),
    "gap_bmiddle": dict(id="batter_id", n_col="asof_batter_n",
                        r_col="asof_batter_middle_rate"),
}
PROBED = ["gap_strike", "gap_middle", "gap_bmiddle", "gap_ball", "gap_reverse"]


def baseline(df, tr, d, yv, cached_fold, axes):
    """블렌드 + ctx δ + asof δ(axes) 상태의 잔차."""
    z = (1 - MLP_ALPHA) * cached_fold[0] + MLP_ALPHA * cached_fold[1]
    ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
    a = asofdelta.fit_asof_axes(df[tr], axes, ASOF_STRENGTH)
    adj, _ = logit_adjust(d, {"ctx": ctx, "asof": a})
    p0 = shift_to_prior(1.0 / (1.0 + np.exp(-(z + adj))), yv.mean())
    return yv - p0, p0


def main():
    for k, v in EXTRA_SPEC.items():
        asofdelta.ASOF_SPEC.setdefault(k, v)

    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(SEED)

    print("§7.204  gap_strike 재개봉 — 1단계 게이트 우회, 전 축을 증분까지")
    print(f"  A팔 = 블렌드(α={MLP_ALPHA}) + ctx{[n for n, _ in CTX_AXES]} + asof{[n for n, _ in DEPLOYED_BASE]}")
    print(f"  B팔 = A + asof({GAP_BALL[0]} β={GAP_BALL[1]})   <- 실제 배포 상태\n")

    acc = {}
    for fold in FOLDS:
        tr, va = s < fold, s == fold
        d = df[va].copy()
        yv = cached[fold][2]

        # 피처와 대조군 (§7.57 과 같은 대조군)
        art = asofdelta.fit_offsets(df[tr])
        league = float(df.loc[tr, TARGET_COL].mean())
        priors = {c: float(np.nanmean(df.loc[tr, c].to_numpy(dtype=np.float64)))
                  for _, c, _ in GAP_SPEC}
        feat = build(d, art, league, priors)
        dsh = d.copy()
        dsh["pitcher_id"] = rng.permutation(d["pitcher_id"].to_numpy())
        dsh["batter_id"] = rng.permutation(d["batter_id"].to_numpy())
        fsh = build(dsh, art, league, priors)

        rA, pA = baseline(df, tr, d, yv, cached[fold], DEPLOYED_BASE)
        rB, pB = baseline(df, tr, d, yv, cached[fold], DEPLOYED_BASE + [GAP_BALL])
        assert not np.array_equal(rA, rB), "두 팔의 잔차가 같다 — gap_ball 이 안 얹혔다"

        print("=" * 86)
        print(f"fold {fold}   행 {len(d):,}   BSS  A {bss(yv, pA):.2f}   B {bss(yv, pB):.2f}"
              f"   (B−A {bss(yv, pB) - bss(yv, pA):+.2f})")
        print("=" * 86)
        print(f"  {'축':<14}{'A: 증분 t':>12}{'A 대조군':>11}{'B: 증분 t':>12}{'B 대조군':>11}"
              f"{'B−A':>9}{'게이트':>8}")
        for c in PROBED:
            if c not in feat:
                continue
            _, tA = resid_t(rA, feat[c]); _, cA = resid_t(rA, fsh[c])
            _, tB = resid_t(rB, feat[c]); _, cB = resid_t(rB, fsh[c])
            acc.setdefault(c, []).append((tA, cA, tB, cB))
            g = "✓" if abs(tB) > 2 and abs(tB) > 2 * abs(cB) else "·"
            print(f"  {c:<14}{tA:>12.2f}{cA:>11.2f}{tB:>12.2f}{cB:>11.2f}{tB - tA:>9.2f}{g:>8}")

    print("\n" + "=" * 86)
    print("판정 — 실제 배포 상태(B팔) 기준")
    print("=" * 86)
    print(f"  {'축':<14}{'tB(2023)':>11}{'tB(2024)':>11}{'부호일치':>10}"
          f"{'대조군초과':>12}{'A->B 변화':>22}")
    for c, v in acc.items():
        (tA1, cA1, tB1, cB1), (tA2, cA2, tB2, cB2) = v
        same = (tB1 > 0) == (tB2 > 0)
        over = abs(tB1) > 2 * abs(cB1) and abs(tB2) > 2 * abs(cB2)
        ch = f"{tA1:+.2f}->{tB1:+.2f} · {tA2:+.2f}->{tB2:+.2f}"
        print(f"  {c:<14}{tB1:>11.2f}{tB2:>11.2f}{'✓' if same else '✗':>10}"
              f"{'✓' if over else '✗':>12}{ch:>22}")

    print("\n  읽는 법")
    print("    A에서 크고 B에서 죽으면  -> gap_ball 이 이미 사고 있다 (증분 없음)")
    print("    A와 B가 같으면           -> gap_ball 과 직교. 1단계 판정이 그대로 유효")
    print("    B에서 부호가 정리되면    -> **1단계 부호 반전이 미통제 교락의 산물이었다**")


if __name__ == "__main__":
    main()
