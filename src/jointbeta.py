"""사후 보정축 4개의 β를 **결합 적합**한다 — 순차 증분 선택의 대안 (사용자 제안).

질문
    "이럴 거면 blend raw에 저것들을 독립변수로 넣은 회귀를 만들면 되지 않나?"

    맞다 — 배포 형태가 이미 `z_blend`를 offset으로 둔 로지스틱 회귀다.
        logit(p) = 1.0·z_blend + Σ β_j v_j + c
    다른 것은 β를 **어떻게 뽑았는가** 하나뿐이다. 현재 값은 전부 홀드아웃
    argmax **아래**이고, 축을 하나씩 증분으로 골랐다 (hand → count4 →
    gap_success → gap_bsuccess). 축 간 상관은 증분 잔차 t로 **확인**만 했지
    결합 **최적화**를 한 적이 없다. 그걸 여기서 잰다.

세 변형 (무엇을 자유롭게 두는가)
    A   β 4개만            c=1 고정, 절편 0 고정
    B   β 4개 + 절편        c=1 고정
    C   β 4개 + 절편 + c    ← §7.19(로짓 온도) 재개봉. 그때는 보정축이 하나도
                              없었다. 4축이 붙은 지금도 두 fold가 갈리는지가 질문.

두 목적함수
    brier    대회 지표 그대로. 저장소의 β 스윕이 전부 이걸로 됐다.
    logloss  "회귀를 만든다"의 표준 해석(MLE). 둘이 다른 β를 주는지도 정보다.

⛔ 평가 규약 — **적합 fold와 평가 fold를 분리한다.**
    같은 fold에서 적합하고 평가하면 λ=0이 항상 이긴다. 그건 MLE의 정의지
    발견이 아니다. in-fold 점수는 **과적합 폭을 보여주기 위해서만** 찍는다.
    2 fold뿐이라 교차는 2가지: 2023→2024, 2024→2023.

ridge
    지금의 "±50% 밴드 규칙"은 사실 엉성한 ridge다 (β를 0쪽으로 민다).
    정식화하면 λ‖β‖². λ→0이 MLE, λ→∞가 무보정, 중간이 지금 배포값 근방일 것.
    ⚠️ 축 스케일이 100배 차이나므로(δ~0.01, v~1) **표준화해서 적합하고**
       계수는 원단위로 되돌려 보고한다. 안 그러면 ridge가 특정 축만 벌준다.

⚠️ λ를 홀드아웃 점수로 고르면 안 된다 — 그럼 λ=0으로 간다. 이 스크립트는
   λ 곡선과 **fold 간 계수 일치도**를 같이 찍는다. 선택은 사람이 한다.

실행:  .venv/Scripts/python.exe -u src/jointbeta.py 2>&1 | tee experiments/jointbeta.log
"""
import os
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

# 배포 구성(s24)의 지문에 못 박는다. rookie.py와 같은 규약 — config가 바뀌어도
# 캐시가 무효화돼 fold당 25분짜리 재학습이 시작되는 사고를 막는다.
_PIN = os.environ.get("LGA_FOLD_KEY") or "5488ef1b18"
if _PIN != "auto":
    foldcache._key = lambda: _PIN

from config import (ASOF_AXES, ASOF_STRENGTH, CONTAM_AXES, CTX_AXES, CTX_HALF_LIFE,
                    CTX_K, MLP_ALPHA, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
LAMBDAS = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
# 자유 파라미터 마스크: [c, 절편, β1..β4]
VARIANTS = {
    "A  β4":            [False, False, True, True, True, True],
    "B  β4+절편":        [False, True,  True, True, True, True],
    "C  β4+절편+c":      [True,  True,  True, True, True, True],
}


# ── 축 값 (β를 곱하지 않은 원단위) ──────────────────────────────────────
def axis_matrix(df, s, fold):
    """fold의 검증 행에 대해 배포 4축의 **원단위 값**을 열로 쌓는다.

    `ctx_lookup`/`asof_lookup`은 β를 곱하지 않는다 (곱셈은 `logit_adjust`가
    한다). 그래서 배포 아티팩트를 그대로 만들고 조회만 하면 설계행렬이 된다 —
    **배포와 같은 코드로 만든 값**이라는 것이 이 측정의 전제다.
    """
    tr_m, va_m = s < fold, s == fold
    tr, va = df[tr_m], df[va_m].copy()
    ref = int(s[tr_m].max())
    w = recency_weight(s[tr_m], CTX_HALF_LIFE, ref_season=ref)

    ctx = ctxfit.fit_ctx(tr, TARGET_COL, CTX_AXES, CTX_K, weights=w)
    art = F.fit_artifacts(tr, ctx=ctx, contam_axes=CONTAM_AXES)
    art["asof"] = asofdelta.fit_asof_axes(tr, ASOF_AXES, ASOF_STRENGTH,
                                          ct=art.get("contam"))

    cols, names, betas = [], [], []
    for ax in ctx:
        cols.append(F.ctx_lookup(va, ax))
        names.append(ax["name"])
        betas.append(float(ax["beta"]))
    for a in art["asof"]:
        cols.append(F.asof_lookup(va, a, ct=art.get("contam")))
        names.append(a["name"])
        betas.append(float(a["beta"]))
    return np.column_stack(cols), names, np.array(betas), va


# ── 적합 (Newton, 6 파라미터라 헤시안이 6×6이다) ────────────────────────
def _obj(kind, z, y):
    """(손실, dL/dη, d²L/dη²) 를 돌려준다. η에 대한 미분이면 체인룰로 끝난다."""
    p = 1.0 / (1.0 + np.exp(-z))
    v = p * (1.0 - p)
    if kind == "logloss":
        return None, (p - y), v
    r = p - y
    # d/dη (r²) = 2 r v ;  d²/dη² = 2[v² + r·v(1−2p)]
    return None, 2.0 * r * v, 2.0 * (v * v + r * v * (1.0 - 2.0 * p))


def fit(X, z0, y, free, lam, kind="brier", iters=40):
    """θ = [c, 절편, β...] 를 Newton으로 푼다. ridge는 β 성분에만 건다.

    X는 **표준화된** 설계행렬이어야 한다 (호출부 책임). 반환도 표준화 단위다.
    """
    n, k = X.shape
    D = np.column_stack([z0, np.ones(n), X])          # [c, 절편, β...]
    th = np.zeros(2 + k)
    th[0] = 1.0                                        # c=1에서 출발
    pen = np.zeros(2 + k)
    pen[2:] = lam
    fr = np.asarray(free)

    for _ in range(iters):
        eta = D @ th
        _, g1, g2 = _obj(kind, eta, y)
        g = D.T @ g1 / n + 2.0 * pen * th
        H = (D.T * (g2 / n)) @ D + 2.0 * np.diag(pen)
        g, H = g[fr], H[np.ix_(fr, fr)]
        H += np.eye(fr.sum()) * 1e-10                  # 감쇠 (Brier 헤시안은 PD가 아닐 수 있다)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        th[fr] -= step
        if np.max(np.abs(step)) < 1e-12:
            break
    return th


def score(th, X, z0, y):
    """배포와 **같은 후처리**로 점수를 낸다 — prior shift는 오라클(저장소 관례)."""
    z = th[0] * z0 + th[1] + X @ th[2:]
    return bss(y, shift_to_prior(1.0 / (1.0 + np.exp(-z)), y.mean()))


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)

    D = {}
    for f in FOLDS:
        X, names, dep, va = axis_matrix(df, s, f)
        y = cached[f][2]
        z0 = (1 - MLP_ALPHA) * cached[f][0] + MLP_ALPHA * cached[f][1]
        assert np.array_equal(y, va[TARGET_COL].to_numpy(dtype=np.float64))
        sd = X.std(axis=0)
        D[f] = dict(X=X, Xs=X / sd, sd=sd, y=y, z0=z0)
        print(f"fold {f}: n={len(y):,}  z0 std={z0.std():.4f}  ȳ={y.mean():.4f}")
    print(f"\n축 {names}\n배포 β {dict(zip(names, dep))}")
    print("축 sd  " + "  ".join(f"{n}={D[FOLDS[0]]['sd'][i]:.4f}"
                                for i, n in enumerate(names)))

    # 기준선 — 배포 β (표준화 단위로 환산해 같은 함수에 태운다)
    base = {}
    for f in FOLDS:
        th = np.concatenate([[1.0, 0.0], dep * D[f]["sd"]])
        raw = bss(D[f]["y"], shift_to_prior(
            1.0 / (1.0 + np.exp(-D[f]["z0"])), D[f]["y"].mean()))
        base[f] = score(th, D[f]["Xs"], D[f]["z0"], D[f]["y"])
        print(f"\nfold {f}  블렌드만 {raw:.1f}   +배포 4축 {base[f]:.1f}"
              f"   (보정 기여 {base[f]-raw:+.1f})")

    for kind in ("brier", "logloss"):
        print(f"\n{'='*78}\n■ 목적함수 = {kind}\n{'='*78}")
        for vn, free in VARIANTS.items():
            print(f"\n── 변형 {vn} " + "─" * 50)
            print(f"  {'λ':>7} │ {'적합2023→평가2024':>18} {'적합2024→평가2023':>18} │"
                  f" {'in-fold 2023':>13} {'in-fold 2024':>13}")
            print(f"  {'':>7} │ {'배포 대비 Δ':>18} {'배포 대비 Δ':>18} │"
                  f" {'(과적합 상한)':>13} {'':>13}")
            rows = {}
            for lam in LAMBDAS:
                th = {f: fit(D[f]["Xs"], D[f]["z0"], D[f]["y"], free, lam, kind)
                      for f in FOLDS}
                rows[lam] = th
                x23 = score(th[2023], D[2024]["Xs"], D[2024]["z0"], D[2024]["y"])
                x24 = score(th[2024], D[2023]["Xs"], D[2023]["z0"], D[2023]["y"])
                i23 = score(th[2023], D[2023]["Xs"], D[2023]["z0"], D[2023]["y"])
                i24 = score(th[2024], D[2024]["Xs"], D[2024]["z0"], D[2024]["y"])
                print(f"  {lam:>7.5f} │ {x23:>9.1f} {x23-base[2024]:>+8.1f}"
                      f" {x24:>9.1f} {x24-base[2023]:>+8.1f} │"
                      f" {i23-base[2023]:>+13.1f} {i24-base[2024]:>+13.1f}")

            # 계수 표 — fold 간 일치 여부가 진짜 판정자다
            for lam in (0.0, 1e-3, 1e-2):
                t23, t24 = rows[lam][2023], rows[lam][2024]
                print(f"\n  계수 (λ={lam})   원단위 β / σ기여 = β·sd(v) [로짓]")
                print(f"    {'항':<16}{'배포':>10}{'f2023':>12}{'f2024':>12}"
                      f"{'비':>9}{'부호':>6}")
                if free[0]:
                    print(f"    {'c (z_blend)':<16}{1.0:>10.4f}"
                          f"{t23[0]:>12.4f}{t24[0]:>12.4f}"
                          f"{t23[0]/t24[0] if t24[0] else float('nan'):>9.2f}"
                          f"{'✓' if t23[0]*t24[0] > 0 else '✗':>6}")
                if free[1]:
                    print(f"    {'절편':<16}{0.0:>10.4f}{t23[1]:>12.4f}{t24[1]:>12.4f}"
                          f"{'':>9}{'✓' if t23[1]*t24[1] > 0 else '✗':>6}")
                for i, nm in enumerate(names):
                    b23 = t23[2 + i] / D[2023]["sd"][i]
                    b24 = t24[2 + i] / D[2024]["sd"][i]
                    r = max(abs(b23), abs(b24)) / max(min(abs(b23), abs(b24)), 1e-12)
                    print(f"    {nm:<16}{dep[i]:>10.4f}{b23:>12.4f}{b24:>12.4f}"
                          f"{r:>9.2f}{'✓' if b23*b24 > 0 else '✗':>6}")
                print(f"    {'σ기여':<16}{'':>10}"
                      + "  ".join(f"{nm}:{t23[2+i]:+.4f}/{t24[2+i]:+.4f}"
                                  for i, nm in enumerate(names)))

    # 축 간 상관 — "순차 선택이 왜 문제인가"의 크기
    print(f"\n{'='*78}\n■ 축 상관 (결합 적합이 순차와 갈리는 이유)\n{'='*78}")
    for f in FOLDS:
        C = np.corrcoef(D[f]["Xs"].T)
        print(f"\n  fold {f}")
        print("    " + " " * 16 + "".join(f"{n:>16}" for n in names))
        for i, n in enumerate(names):
            print(f"    {n:<16}" + "".join(f"{C[i, j]:>16.3f}" for j in range(len(names))))


if __name__ == "__main__":
    main()
