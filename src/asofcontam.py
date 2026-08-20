"""§7.60 — gap 축이 **오염된 기준선**을 쓰고 있었다.

발견
    `build_features`는 pre-2023 F 오염을 보정한다 (§7.49, LB +10.48).
    그런데 `asof_lookup`(gap 축 경로)은 **원본 `asof_*`를 읽는다.** 배선을 빠뜨렸다.

왜 문제인가
    gap = (올해 진행분) − (통산)

    k_szn = k_now − k_off 는 **깨끗하다.** 오염 투구는 전부 개막 이전이라 두 항에
        똑같이 들어가 상쇄된다 (2024·2025 투구는 새 기준으로 라벨됨).
    r_car 는 **오염된 채로 남는다.** 성공률이 과대평가돼 있고, gap은 그걸 빼는
        기준선으로 쓴다. → gap이 D/m 만큼 아래로 밀린다.

    2024 행 측정:  55.2%가 노출,  평균 0.87σ,  최대 10.26σ   (σ = gap의 sd)

    노출량이 투수마다 다르므로 이건 상수 편향이 아니라 **선수별 편향**이다.
    표준화가 평균은 흡수하지만 선수 간 변동은 못 흡수한다.
    게다가 2025 활동 투수는 2023~2025 투구가 쌓여 D/m이 더 작다 →
    홀드아웃과 배포에서 **편향의 분포가 다르다.**

무엇을 묻나
    ① 보정하면 홀드아웃 이득이 커지나 (같은 β에서)
    ② β 고원이 넓어지나 / 옮겨지나
    ③ **조건 ④(관계의 시즌 간 지속성)가 올라가나**  ← 이게 가장 중요하다.
       0.82/0.78이 오염 때문에 깎여 있었다면 보정 후 올라가야 한다.
    ④ `gap_reverse`는 오염이 가장 심한 컬럼이다 (.048 → .300).
       0.65로 애매하게 기각됐는데, 보정하면 판정이 바뀌나.

실행:  .venv/Scripts/python.exe -u src/asofcontam.py 2>&1 | tee experiments/asofcontam.log
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
from config import (ASOF_AXES, ASOF_STRENGTH, CONTAM_AXES, CTX_AXES, CTX_K,
                    MLP_ALPHA, SEED, TARGET_COL, TRAIN_CSV)
from data import load_csv
from mixdiag import shape
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
BETAS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06]
# 배포 2축 + 오염이 가장 심한 축. reverse는 부호가 음수라 β도 음수로 쓴다 (§7.57).
CAND = [("gap_success", +1), ("gap_bsuccess", +1), ("gap_reverse", -1)]


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(SEED)

    # ── ① · ② 점수 ────────────────────────────────────────────────
    print("=" * 100)
    print("① ② 홀드아웃 이득 — 원본 기준선 vs 오염 보정 기준선")
    print("=" * 100)
    for fold in FOLDS:
        tr, va = s < fold, s == fold
        d = df[va].copy()
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        ct = F.fit_artifacts(df[tr], contam_axes=CONTAM_AXES)["contam"]

        # 기준선에는 asof를 넣지 않는다 — 두 판본을 같은 자리에서 비교하려면
        # 축 자체가 빠진 상태에서 시작해야 한다.
        adj, _ = F.logit_adjust(d, {"ctx": ctx})
        z0 = z + adj
        base = bss(yv, shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean()))

        def gain(v, b):
            p = shift_to_prior(1.0 / (1.0 + np.exp(-(z0 + b * v))), yv.mean())
            return bss(yv, p) - base

        print(f"\nfold {fold}   기준 {base:.1f}  (δ만, asof 없음)")
        print(f"  {'축':<16}{'판본':<8}" + "".join(f"{b:>9}" for b in BETAS))
        for name, sign in CAND:
            for tag, cc in (("원본", None), ("보정", ct)):
                a = asofdelta.fit_asof_axes(df[tr], [(name, 1.0)],
                                            ASOF_STRENGTH, ct=cc)[0]
                v = F.asof_lookup(d, a, ct=cc)
                row = [gain(v, sign * b) for b in BETAS]
                print(f"  {name:<16}{tag:<8}" + "".join(f"{g:>+9.2f}" for g in row))

    # ── ③ · ④ 조건 ④ (관계의 시즌 간 지속성) ───────────────────────
    print("\n" + "=" * 100)
    print("③ ④ 조건 ④ — 십분위 모양의 시즌 간 상관 / 노이즈 천장")
    print("=" * 100)
    tr22 = s <= 2022
    ct22 = F.fit_artifacts(df[tr22], contam_axes=CONTAM_AXES)["contam"]
    off_all = asofdelta.fit_offsets(df)          # 각 시즌 gap이 제대로 계산되게 (mixdiag와 동일)

    print(f"  {'축':<16}{'판본':<8}"
          + "".join(f"{a}-{b}".rjust(11) for a, b in zip(SEASONS, SEASONS[1:]))
          + "   최근3최소     천장     비율")
    for name, _ in CAND:
        for tag, cc in (("원본", None), ("보정", ct22)):
            a = asofdelta.fit_asof_axes(df[tr22], [(name, 1.0)],
                                        ASOF_STRENGTH, ct=cc)[0]
            o = off_all[a["n_col"]]
            a["keys"], a["n_off"] = o["keys"], o["n_off"]
            a["k_off"] = o["k_off"][a["r_col"]]
            v = F.asof_lookup(df, a, ct=cc)

            sh, rel = [], []
            for yr in SEASONS:
                m = s == yr
                yy, vv = y[m], v[m]
                sh.append(shape(yy, vv))
                h = rng.random(len(yy)) < 0.5
                r = float(np.corrcoef(shape(yy[h], vv[h]),
                                      shape(yy[~h], vv[~h]))[0, 1])
                rel.append(2 * r / (1 + r) if r > -1 else np.nan)
            cs = [float(np.corrcoef(sh[i], sh[i + 1])[0, 1])
                  for i in range(len(sh) - 1)]
            lo, ceil = min(cs[-3:]), float(np.nanmean(rel[-3:]))
            print(f"  {name:<16}{tag:<8}" + "".join(f"{x:>+11.3f}" for x in cs)
                  + f"{lo:>+11.3f}{ceil:>9.3f}{lo / ceil if ceil else np.nan:>9.2f}")

    print("\n" + "=" * 100)
    print("읽는 법")
    print("=" * 100)
    print("  ① 이득이 커지면 오염이 신호를 가리고 있었다는 뜻.")
    print("  ③ 비율(조건 ④)이 올라가면 **더 중요하다** — 관계 자체가 더 안정해진 것이고,")
    print("     그건 홀드아웃 점수보다 전이를 잘 예측한다 (§7.59).")
    print("  둘이 엇갈리면 조건 ④를 따른다. 점수는 fold 국한 오차를 섞기 때문이다.")


if __name__ == "__main__":
    main()
