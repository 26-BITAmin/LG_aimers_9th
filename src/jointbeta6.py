"""외부 제안 검증 — Murphy 분해 + 로짓 스케일 불일치 가설 (사용자가 가져온 테스트).

받은 제안 두 가지
    테스트 1  ŵ^MLE 와 ŵ^Brier 를 그냥 출력해 대소를 본다.
    테스트 2  Brier를 Murphy 3항으로 쪼개 어디서 잃었는지 본다.
                BS = E_k(p̄_k − ȳ_k)²  −  E_k(ȳ_k − r)²  +  r(1−r)
                     ─── Reliability ──   ─── Resolution ──   Uncertainty
              Resolution 감소 → sharpness 손실 / Reliability 증가 → 캘리브레이션 손실.
              후자면 "OOF 로짓 vs holdout 로짓 스케일 불일치"를 의심하라는 제안.

테스트 1은 `jointbeta.py`가 이미 두 목적함수로 돌린다. 결론: **부호는 맞지만
크기가 1% 이하**다 (전이 점수 −25.4 vs −25.2). 여기서는 기록만 하고 넘어간다.
기전: p가 전부 .4~.6 근방이라 p(1−p)≈.25로 거의 상수 → 두 손실의 기울기가
상수배로 수렴한다. 확률이 좁은 문제에서 Brier와 logloss는 갈리지 않는다.

여기서 재는 것
    ① Murphy 분해 — 배포 β / fold2023 결합적합 β 를 fold2024에 얹고 REL·RES 비교.
       UNC는 평가셋이 같으면 동일하므로 **ΔBS = ΔREL − ΔRES 로 정확히 귀속된다.**
       ⚠️ 분위 구간 분해는 구간 내 p 분산만큼 항등식이 안 맞는다. 잔차를 같이 찍는다.

    ② 스케일 불일치 가설의 **직접 검정.** 이 저장소에는 조건이 실재한다:
           std(z0)  fold2023 0.1537   fold2024 0.1677   비 1.091
       가설이 맞다면 β를 옮길 때 **1.091배 키워야** 한다 (기여/기저 비를 맞춘다).
       → β_2023 에 전역 배율 g를 걸어 fold2024 전이 최적 g*를 찾는다.
         g* ≈ 1.09 면 가설 지지, g* < 1 이면 **정반대**이고 가설 기각.
       ⛔ λ 스윕이 이미 수축을 원했으므로(λ=0.03에서 −25.4 → −10.0) g*<1이 예상된다.
          그래도 직접 재서 숫자로 닫는다.

실행:  .venv/Scripts/python.exe -u src/jointbeta6.py 2>&1 | tee experiments/jointbeta6.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import jointbeta as J
from config import ASOF_AXES, CTX_AXES, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import brier, bss, shift_to_prior

FOLDS = [2023, 2024]
FREE_A = [False, False, True, True, True, True]
NBIN = 20


def murphy(y, p, nbin=NBIN):
    """BS = REL − RES + UNC. 분위 구간으로 쪼갠다.

    ⚠️ 항등식은 **구간 안에서 p가 상수일 때만** 정확하다. 분위 구간이면
       구간 내 분산만큼 어긋나므로 잔차를 같이 돌려준다 (숨기면 오독한다).
    """
    y = np.asarray(y, float)
    q = np.quantile(p, np.linspace(0, 1, nbin + 1))
    q[0], q[-1] = -np.inf, np.inf
    k = np.clip(np.searchsorted(q, p, side="right") - 1, 0, nbin - 1)
    r = y.mean()
    rel = res = 0.0
    for b in range(nbin):
        m = k == b
        if not m.any():
            continue
        w = m.mean()
        rel += w * (p[m].mean() - y[m].mean()) ** 2
        res += w * (y[m].mean() - r) ** 2
    bs = brier(y, p)
    return dict(BS=bs, REL=rel, RES=res, UNC=r * (1 - r),
                resid=bs - (rel - res + r * (1 - r)))


def P(z, y):
    return shift_to_prior(1 / (1 + np.exp(-z)), y.mean())


def cox(y, p, iters=50):
    """Cox 캘리브레이션 — logit(P(y=1)) = a + b·logit(p) 를 MLE로 푼다.

    **구간화가 없다.** b<1 = 과신(예측이 너무 벌어짐), b>1 = 과소신.
    Murphy 분해와 달리 표본 수에만 의존하고 임의 격자에 의존하지 않는다.
    """
    x = np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12)))
    D = np.column_stack([np.ones(len(x)), x])
    th = np.array([0.0, 1.0])
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(D @ th)))
        g = D.T @ (q - y) / len(y)
        H = (D.T * (q * (1 - q) / len(y))) @ D + np.eye(2) * 1e-12
        st = np.linalg.solve(H, g)
        th -= st
        if np.max(np.abs(st)) < 1e-12:
            break
    # b의 표준오차 (역헤시안 대각), n으로 나눈 스케일을 되돌린다
    q = 1 / (1 + np.exp(-(D @ th)))
    cov = np.linalg.inv((D.T * (q * (1 - q))) @ D)
    return float(th[0]), float(th[1]), float(np.sqrt(cov[1, 1]))


def profile(y, p):
    """구간화 없는 3지표. sharpness · discrimination · calibration."""
    a, b, se = cox(y, p)
    return dict(sd=float(p.std()), cov=float(np.cov(p, y)[0, 1]),
                slope=b, slope_se=se, icpt=a)


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = J.foldcache.load(FOLDS, verbose=False)
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])

    D = {}
    for f in FOLDS:
        X, names, _, va = J.axis_matrix(df, s, f)
        y = cached[f][2]
        z0 = (1 - MLP_ALPHA) * cached[f][0] + MLP_ALPHA * cached[f][1]
        assert np.array_equal(y, va[TARGET_COL].to_numpy(dtype=np.float64))
        sd = X.std(axis=0)
        D[f] = dict(X=X, Xs=X / sd, sd=sd, y=y, z0=z0,
                    R=(va["game_type"].to_numpy() != "F"))
        print(f"fold {f}  n={len(y):,}  std(z0)={z0.std():.4f}  ȳ={y.mean():.4f}")
    rr = D[2024]["z0"].std() / D[2023]["z0"].std()
    print(f"\n로짓 스케일 비  std(z0_2024)/std(z0_2023) = {rr:.4f}")
    print(f"축 {names}   배포 β {dict(zip(names, dep))}")

    # ── 테스트 1 — 계수 대소 (기록용. jointbeta.log와 같은 값이어야 한다) ──
    print(f"\n{'='*78}\n■ 테스트 1 — ŵ^MLE vs ŵ^Brier\n{'='*78}")
    B = {f: J.fit(D[f]["Xs"], D[f]["z0"], D[f]["y"], FREE_A, 0.0, "brier")
         for f in FOLDS}
    M = {f: J.fit(D[f]["Xs"], D[f]["z0"], D[f]["y"], FREE_A, 0.0, "logloss")
         for f in FOLDS}
    print(f"  {'축':<15}{'fold':>6}{'Brier':>10}{'MLE':>10}{'MLE/Brier':>11}{'판정':>8}")
    for i, nm in enumerate(names):
        for f in FOLDS:
            b, m = B[f][2 + i] / D[f]["sd"][i], M[f][2 + i] / D[f]["sd"][i]
            print(f"  {nm:<15}{f:>6}{b:>10.4f}{m:>10.4f}{m/b:>11.4f}"
                  f"{('MLE<Brier' if m < b else 'MLE>Brier' if m > b else '동률'):>12}")

    # ── 테스트 2 — Murphy 분해 ──────────────────────────────────────────
    print(f"\n{'='*78}\n■ 테스트 2 — Murphy 분해 (fold2024 평가, {NBIN}분위)\n{'='*78}")
    print("  ΔBS = ΔREL − ΔRES  (UNC는 평가셋이 같아 상쇄된다)")
    print("  BSS는 1e5·(1−BS/UNC)이므로 ΔBSS = −1e5·ΔBS/UNC 로 환산된다.\n")
    for pop, mask in (("ALL", np.ones(len(D[2024]["y"]), bool)), ("R", D[2024]["R"])):
        y = D[2024]["y"][mask]
        cands = {
            "배포 β": dep,
            "2023 결합적합 (Brier)": B[2023][2:] / D[2023]["sd"],
            "2023 결합적합 (MLE)": M[2023][2:] / D[2023]["sd"],
            "2024 자기적합 (Brier, in-fold)": B[2024][2:] / D[2024]["sd"],
        }
        print(f"── 모집단 {pop}  n={mask.sum():,}  UNC={y.mean()*(1-y.mean()):.6f} " + "─" * 20)
        print(f"  {'β 출처':<30}{'BSS':>9}{'REL×1e5':>10}{'RES×1e5':>10}"
              f"{'ΔREL':>9}{'ΔRES':>9}{'잔차':>10}")
        ref = None
        for nm2, b in cands.items():
            p = P((D[2024]["z0"] + D[2024]["X"] @ b)[mask], y)
            d = murphy(y, p)
            if ref is None:
                ref = d
            print(f"  {nm2:<30}{bss(y, p):>9.1f}{d['REL']*1e5:>10.2f}"
                  f"{d['RES']*1e5:>10.2f}{(d['REL']-ref['REL'])*1e5:>+9.2f}"
                  f"{(d['RES']-ref['RES'])*1e5:>+9.2f}{d['resid']*1e5:>10.3f}")
        # ⚠️ 구간 수 민감도 — 잔차가 읽으려는 효과와 같은 자릿수면 분해가 성립 안 한다.
        #    구간을 늘리면 잔차(구간 내 p 분산)는 줄고 REL 추정 분산은 는다.
        print(f"\n  구간 수 민감도 (ΔREL / ΔRES, 배포 β 대비):")
        print(f"    {'nbin':>6}" + "".join(
            f"{k:>26}" for k in list(cands)[1:3]) + f"{'잔차(배포)':>12}")
        for nb in (20, 50, 100, 200, 500, 1000):
            r0 = murphy(y, P((D[2024]["z0"] + D[2024]["X"] @ dep)[mask], y), nb)
            cells = []
            for nm2 in list(cands)[1:3]:
                d = murphy(y, P((D[2024]["z0"] + D[2024]["X"] @ cands[nm2])[mask], y), nb)
                cells.append(f"{(d['REL']-r0['REL'])*1e5:>+11.2f} /"
                             f"{(d['RES']-r0['RES'])*1e5:>+11.2f}")
            print(f"    {nb:>6}" + "".join(f"{c:>26}" for c in cells)
                  + f"{r0['resid']*1e5:>12.3f}")

        # ── 구간화 없는 대체 지표 (Murphy가 식별 안 될 때의 답) ──────────
        print(f"\n  구간화 없는 3지표 — sharpness / discrimination / calibration")
        print(f"    {'β 출처':<30}{'std(p)':>9}{'cov(p,y)':>11}"
              f"{'Cox 기울기 b':>14}{'SE':>8}{'(b−1)/SE':>10}")
        pf = {}
        for nm2, b_ in cands.items():
            p = P((D[2024]["z0"] + D[2024]["X"] @ b_)[mask], y)
            pf[nm2] = profile(y, p)
            print(f"    {nm2:<30}{pf[nm2]['sd']:>9.5f}{pf[nm2]['cov']:>11.6f}"
                  f"{pf[nm2]['slope']:>14.4f}{pf[nm2]['slope_se']:>8.4f}"
                  f"{(pf[nm2]['slope']-1)/pf[nm2]['slope_se']:>+10.1f}")

        # ── ★ 정확 분해 — 구간이 없다. 항등식이 그냥 성립한다 ──────────
        #   BS = var(y) + var(p) − 2cov(p,y) + (p̄ − ȳ)²   ← 평균은 prior shift가 맞춘다
        #   ⇒ ΔBS = Δvar(p) − 2Δcov(p,y)
        #   즉 "퍼뜨린 만큼(Δvar) 진짜를 맞혔는가(2Δcov)"의 손익계산이고,
        #   **2Δcov/Δvar > 1 이어야 그 퍼뜨림이 값을 한다.**
        print(f"\n  ★ 정확 분해 (구간화 없음)   ΔBS = Δvar(p) − 2·Δcov(p,y)")
        print(f"    {'β 출처':<30}{'Δvar(p)':>12}{'2Δcov':>12}{'ΔBS':>12}"
              f"{'ΔBSS(환산)':>12}{'2Δcov/Δvar':>12}")
        r0 = pf["배포 β"]
        for nm2 in cands:
            dv = pf[nm2]['sd'] ** 2 - r0['sd'] ** 2
            dc = 2 * (pf[nm2]['cov'] - r0['cov'])
            db = dv - dc
            print(f"    {nm2:<30}{dv*1e5:>+12.3f}{dc*1e5:>+12.3f}{db*1e5:>+12.3f}"
                  f"{-1e5*db/(y.mean()*(1-y.mean())):>+12.1f}"
                  + (f"{dc/dv:>12.3f}" if abs(dv) > 1e-12 else f"{'—':>12}"))
        print()

    # ── 스케일 불일치 가설의 직접 검정 ──────────────────────────────────
    print(f"{'='*78}\n■ 로짓 스케일 불일치 가설 — β에 전역 배율 g\n{'='*78}")
    print(f"  가설: 기저 로짓이 fold2024에서 {rr:.3f}배 넓으므로 β도 {rr:.3f}배 키워야 한다.")
    print(f"  기각 조건: 전이 최적 g* < 1 (수축을 원하면 방향이 정반대다).\n")
    for src, tgt in ((2023, 2024), (2024, 2023)):
        b0 = B[src][2:] / D[src]["sd"]
        base = bss(D[tgt]["y"], P(D[tgt]["z0"] + D[tgt]["X"] @ dep, D[tgt]["y"]))
        print(f"  ── 적합 {src} → 평가 {tgt}   (배포 기준선 {base:.1f})"
              + ("   ← 전방향" if src < tgt else "   ← 역방향(참고)"))
        row = []
        for g in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.091, 1.2, 1.4]:
            v = bss(D[tgt]["y"], P(D[tgt]["z0"] + D[tgt]["X"] @ (b0 * g), D[tgt]["y"]))
            row.append((g, v))
        gs = max(row, key=lambda t: t[1])
        print("     g   " + "".join(f"{g:>9.3f}" for g, _ in row))
        print("   BSS   " + "".join(f"{v:>9.1f}" for _, v in row))
        print("  배포Δ   " + "".join(f"{v-base:>+9.1f}" for _, v in row))
        print(f"     → g* = {gs[0]:.3f}  (BSS {gs[1]:.1f})"
              f"   스케일 가설 예측 {rr if src < tgt else 1/rr:.3f}"
              f"   {'✅ 지지' if abs(gs[0]-(rr if src < tgt else 1/rr)) < 0.1 else '❌ 기각'}\n")


if __name__ == "__main__":
    main()
