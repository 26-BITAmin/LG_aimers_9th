"""§7.150 진단 — 중심화값의 **제곱/절대편차**가 배포 잔차에 신호를 남기는가. 학습 0회.

사용자 제안: *"센터링만 하는데 여기에 100을 곱하고 제곱까지 해서 해보자.
어차피 standard scaling 적용되니까 크기 자체는 괜찮겠지."*

두 조작을 나눠 본다.

**① ×100 은 두 학습기 모두에서 정확히 무연산이다 — 실험할 것이 없다.**
   · LGBM: 분할이 `x ≤ t` 비교라 강단조 변환에 불변. 히스토그램 구간도 분위수 기반이라
     경계가 같은 자리에 온다. `f(x) ≤ f(t) ⟺ x ≤ t`.
   · MLP: standard scaling 이 `(100x − mean(100x)) / std(100x)` = `(x − mean)/std`.
     평균·표준편차가 **둘 다** 100배라 정확히 상쇄된다. (사용자 관찰이 맞고, 따라서
     결론은 "크기가 괜찮다"가 아니라 **"효과가 0이다"** 이다.)
   → 아래에서는 **제곱만** 검정한다.

**② 제곱은 부호를 접는다.** `(v−μ)²` 은 −0.03 과 +0.03 을 같은 값으로 만든다.
   그리고 **트리에게 `x²` 은 `|x|` 과 같은 피처다** (음이 아닌 구간에서 서로 단조).
   즉 "제곱해서 넣는다"는 트리 입장에서 **"절대편차로 바꾼다"** 와 동치다.
   → 교체하면 방향 정보가 사라진다. 물어야 할 것은 **추가**했을 때 값이 있느냐다.

검정 (MEMORY: 새 피처는 잔차에 먼저 물어라 — 학습 전에 t 부터):
    배포 층 잔차 r = y − p̂ 에 후보를 회귀한다. 주효과 대조군을 반드시 같이 둔다
    (c 자체도 같은 방식으로 재서, c² 의 t 가 c 의 t 와 비교되게).

    c   = v − μ_lag        중심화값 (부호 유지) — **주효과 대조군**
    c²  = (v − μ_lag)²     사용자 제안
    |c| = |v − μ_lag|      트리 관점에서 c² 과 동치인 형태 (확인용)

각 후보를 z-표준화(train 통계)한 뒤 β 스윕으로 배포 로짓에 더해 ΔBSS·클러스터 t 를
잰다. **in-sample 적합이라 상한이다** — 이득이 상한에서도 없으면 학습은 무의미하다.
전이(fold A 계수 → fold B)도 같이 낸다.

실행:  .venv/Scripts/python.exe -u src/sqdiag.py 2>&1 | tee experiments/sqdiag.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import deploy29 as D                                                # noqa: E402
import features as F                                                # noqa: E402
from config import (CONTAM_AXES, RECENCY_HALF_LIFE, SMOOTH_K_BATTER,  # noqa: E402
                    SMOOTH_K_PITCHER)
from data import recency_weight                                     # noqa: E402
from features import build_features, fit_artifacts                  # noqa: E402

FOLDS = [2023, 2024]
BETAS = np.arange(-1.0, 1.001, 0.02)


def centered(df, s, fold):
    """fold 검증 행의 **lag 중심화값** c 와 train 쪽 표준화 통계를 돌려준다."""
    tr, va = df[s < fold], df[s == fold]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                        center_asof="lag5")
    Xtr = build_features(tr.iloc[::7], art)      # 표준화 통계용 서브샘플
    Xva = build_features(va, art)
    out = {}
    for col in F.CENTER_SUBSET5:
        a = Xtr[col].to_numpy(dtype=np.float64)
        b = Xva[col].to_numpy(dtype=np.float64)
        fa = np.isfinite(a)
        out[col] = (b, float(a[fa].mean()), float(a[fa].std()))
    return out


def cands(c, mu, sd):
    """후보 셋. 전부 train 통계로 표준화한다 (스케일이 β 그리드와 맞아야 한다)."""
    fin = np.isfinite(c)
    cc = np.where(fin, c, mu)                     # 결측은 train 평균으로
    z = (cc - mu) / max(sd, 1e-9)
    return {"c (주효과)": z, "c^2 (제안)": (z ** 2 - 1.0), "|c|": np.abs(z) - np.abs(z).mean()}


def main():
    df, s, y = D.load_df()
    ST, FEAT = {}, {}
    for f in FOLDS:
        ST[f] = D.state(df, s, y, f)
        print(f"fold {f} 배포 기준 BSS {ST[f]['base']:.1f}  (하네스 검증 통과)")
    for f in FOLDS:
        FEAT[f] = centered(df, s, f)
        print(f"fold {f} 중심화값 준비 완료")

    print(f"\n{'='*92}")
    print("█ 잔차 진단 — 배포 로짓에 β·후보 를 더했을 때 (in-sample 최적 β = **상한**)")
    print(f"{'='*92}")
    print(f"  {'컬럼':<38}{'후보':<14}" + "".join(f"{f'f{f} Δ(β*)':>12}{'t':>8}" for f in FOLDS))

    BEST = {}
    for col in F.CENTER_SUBSET5:
        for name in ("c (주효과)", "c^2 (제안)", "|c|"):
            cells, row = [], {}
            for f in FOLDS:
                cv, mu, sd = FEAT[f][col]
                v = cands(cv, mu, sd)[name][ST[f]["mm"]]
                ds = [D.rescore(ST[f], b * v) for b in BETAS]
                j = int(np.argmax(ds))
                t = D.cluster_t(ST[f], BETAS[j] * v)
                row[f] = (BETAS[j], v)
                cells.append(f"{ds[j]:>12.1f}{t:>8.2f}")
            BEST[(col, name)] = row
            print(f"  {col:<38}{name:<14}" + "".join(cells))

    print(f"\n  █ 전이 — fold A 에서 고른 β 를 fold B 에 적용 (in-sample 아님)")
    print(f"  {'컬럼':<38}{'후보':<14}{'2023→2024':>14}{'2024→2023':>14}")
    for col in F.CENTER_SUBSET5:
        for name in ("c (주효과)", "c^2 (제안)", "|c|"):
            row = BEST[(col, name)]
            a = D.rescore(ST[2024], row[2023][0] * row[2024][1])
            b = D.rescore(ST[2023], row[2024][0] * row[2023][1])
            print(f"  {col:<38}{name:<14}{a:>14.1f}{b:>14.1f}")

    print(f"\n  판정 기준: 상한(in-sample)에서도 |t| < 2 이고 Δ 가 시드 노이즈(4~9) 아래면")
    print(f"            학습 팔을 돌릴 근거가 없다. 전이가 음수면 더욱.")


if __name__ == "__main__":
    main()
