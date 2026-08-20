"""레퍼토리 균형도 축 (§7.58 `r3_ent`) — **fit + 조밀 β 스윕.**

§7.58이 남긴 것 (배포 전체 위에서 잰 값이다 — 관문 ① 증분은 이미 통과):

    잔차 t      +8.72 / +5.99      대조군 −0.65 / −1.47
    β 스윕      0.002  0.005  0.01   0.02   0.04
      f2023     +3.21  +7.67 +14.17 +23.61 +28.26
      f2024     +2.24  +5.24  +9.25 +13.60  +7.68   ← 0.02에서 꺾인다

**여기서 다시 재는 이유가 두 가지다.**

1. 고원 [0.002, 0.04]의 **양 끝이 격자 끝점**이다. §7.37 밴드 규칙은
   `[0.5β, 1.5β] ⊂ 고원`을 요구하는데, 고원 경계를 모르면 밴드를 못 그린다.
   f2024가 0.02→0.04에서 13.60→7.68로 꺾였으니 영점이 0.04~0.06 어딘가다.
2. §7.58의 측정은 cold 행을 **Trackman 라벨 분포**로 채웠다. 배포 축은
   trackman에 의존하면 안 되므로 `league`를 **train.csv 믹스 컬럼 자체**에서
   뽑는다. 정의가 바뀌었으니 이득이 재현되는지 확인해야 한다.

⚠️ 피처 코드는 `features.mixent_lookup` **하나뿐**이다. 이 스크립트는 그것을
   호출만 한다 — 측정과 배포가 다른 코드를 보면 §7.36이 반복된다.

실행:  .venv/Scripts/python.exe -u src/mixent.py 2>&1 | tee experiments/mixent.log
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
from config import (ASOF_AXES, ASOF_STRENGTH, CTX_AXES, CTX_K, MLP_ALPHA, SEED,
                    TARGET_COL, TRAIN_CSV)
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
MIX_COLS = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate"]

MIX_SPEC = {
    "r3_ent": dict(cols=MIX_COLS, n_col="asof_pitcher_pitchmix_n"),
}

# 0.04 위쪽을 채운다 — f2024의 영점을 찾아야 고원의 오른쪽 끝이 정해진다.
BETAS = [0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08]


def fit_mixent(train_df, axes):
    """축 리스트를 만든다. **얼려지는 것은 축당 스칼라 6개** (league 3 + mu + sd + β).

    `league`는 재정규화된 믹스율의 train 평균이다. cold 행(믹스 정보 없음)을
    여기로 떨어뜨린다 — 0으로 두면 엔트로피 0이 되어 "한 구종만 던지는 투수"와
    섞인다.
    """
    out = []
    for name, beta in axes:
        if not beta:
            continue
        spec = MIX_SPEC[name]
        raw = np.stack([train_df[c].to_numpy(dtype=np.float32).astype(np.float64)
                        for c in spec["cols"]], axis=1)
        raw = np.nan_to_num(raw, nan=0.0)
        tot = raw.sum(1)
        n_mix = np.nan_to_num(train_df[spec["n_col"]].to_numpy(dtype=np.float32)
                              .astype(np.float64), nan=0.0)
        ok = (tot > 0) & (n_mix > 0)
        r3 = raw[ok] / tot[ok][:, None]
        league = r3.mean(0)
        league = league / league.sum()

        a = dict(spec, name=name, league=league.tolist(),
                 mu=0.0, sd=1.0, beta=0.0)
        v = F.mixent_lookup(train_df, a)          # ← 배포와 **같은 코드**
        a["mu"] = float(np.mean(v))
        a["sd"] = float(np.std(v)) or 1.0
        a["beta"] = float(beta)
        out.append(a)
    return out


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(SEED)
    out = {}

    print(f"기준선 = 블렌드 + δ{CTX_AXES} + asof {ASOF_AXES}  (s17 배포 전부)")
    for fold in FOLDS:
        tr, va = s < fold, s == fold
        d = df[va].copy()
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        asof = asofdelta.fit_asof_axes(df[tr], ASOF_AXES, ASOF_STRENGTH)
        adj, _ = F.logit_adjust(d, {"ctx": ctx, "asof": asof})
        z0 = z + adj
        base = bss(yv, shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean()))

        # 표준화 상수는 **학습 시즌에서만** (검증 행 통계를 쓰면 그 자체가 리크)
        axes = fit_mixent(df[tr], [("r3_ent", 1.0)])
        a = axes[0]
        v, ok = F.mixent_lookup(d, a, with_hit=True)

        def gain(vec, b):
            p = shift_to_prior(1.0 / (1.0 + np.exp(-(z0 + b * vec))), yv.mean())
            return bss(yv, p) - base

        # 대조군 — 값의 분포는 그대로 두고 **행과의 연결만** 끊는다.
        vs = v[rng.permutation(len(v))]

        print(f"\nfold {fold}  기준 {base:.1f}   행 {len(d):,}   "
              f"믹스 적중 {ok.mean() * 100:.1f}%   league={np.round(a['league'], 3)}")
        print(f"  {'':<12}" + "".join(f"{b:>8}" for b in BETAS))
        for nm, vec in (("r3_ent", v), ("[대조군]", vs)):
            row = [gain(vec, b) for b in BETAS]
            out.setdefault(nm, {})[fold] = row
            print(f"  {nm:<12}" + "".join(f"{g:>+8.2f}" for g in row))

    print("\n" + "=" * 96)
    print("판정 — §7.37 밴드 규칙:  [0.5β, 1.5β] 전체가 양 fold 양수 구간 안에")
    print("=" * 96)
    a23, a24 = np.array(out["r3_ent"][2023]), np.array(out["r3_ent"][2024])
    both = (a23 > 0) & (a24 > 0)
    pl = [BETAS[j] for j in range(len(BETAS)) if both[j]]
    if pl:
        lo, hi = min(pl), max(pl)
        print(f"  양 fold 양수 구간 [{lo}, {hi}]   폭 {hi / lo:.0f}배")
        print(f"  밴드 규칙 허용 β  [{lo / 0.5:.4f}, {hi / 1.5:.4f}]")
        ok_b = [b for b in BETAS if 0.5 * b >= lo and 1.5 * b <= hi]
        print(f"  격자 위 후보      {ok_b}")
        for b in ok_b:
            i = BETAS.index(b)
            r = max(abs(a23[i]), abs(a24[i])) / max(min(abs(a23[i]), abs(a24[i])), 1e-9)
            print(f"    β={b:<7} f2023 {a23[i]:+8.2f}  f2024 {a24[i]:+8.2f}   크기비 {r:.2f}")
    else:
        print("  양 fold 공통 양수 구간 없음 → 기각")
    print("\n  ⚠️ 최댓값을 고르지 않는다. CV argmax는 검증 fold에 맞춘 값이다 (§7.37).")


if __name__ == "__main__":
    main()
