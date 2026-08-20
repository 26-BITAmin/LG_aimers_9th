"""`gap_success`의 fold2023 아웃라이어 — 구조인가 노이즈인가 (사용자 질문).

사실
    solo 적합 β (R-only):   2021 .0392   2022 .0369   **2023 .0730**   2024 .0333
    2023만 약 2배다. §7.87·§7.88에서 `Atrend`와 recency 가중이 **둘 다 이 튐 때문에**
    실패했다. 이 채널을 다시 열려면 이 값의 정체를 먼저 알아야 한다.

네 가설을 분리한다

  H0 표집 노이즈    β̂의 SE가 커서 .073과 .037이 구별 안 되는 것.
                    → 검정: 샌드위치 SE + fold 내 split-half (무작위 / 시간).
                    ⚠️ Brier는 우도가 아니므로 H⁻¹만 쓰면 안 된다. H⁻¹JH⁻¹를 쓴다.

  H1 기저 모델 구조  fold2023은 기저가 망가진 fold다 (블렌드 단독 BSS 641.5 vs
                    2024의 837.6). **기저가 약하면 보정축이 먹을 잔차가 커서 최적 β가
                    자동으로 커진다.** 그러면 구조이긴 한데 `gap_success`에 관한
                    구조가 아니라 **기저 모델에 관한 구조**다.
                    → 검정: z0에 배율 c를 걸어 기저를 인위로 약화시키고 β̂ 재적합.
                      c가 내려갈 때 β̂가 오르면 기전 확인.

  H2 오염 잔여      `gap`은 (올해) − (커리어)인데 2023 행의 커리어는 **거의 전부
                    오염 시즌**(2019~2022)이다. §7.60의 보정이 불완전하면 그 잔여가
                    2023에 가장 크게 남고, gap이 "이 투수가 오염으로 얼마나 부풀었나"를
                    추가로 실어 β를 키운다. 2024는 커리어에 깨끗한 2023이 섞여 덜하다.
                    → 검정: 오염 보정 ON/OFF로 β̂이 얼마나 움직이는지 fold별 비교.
                      2023의 이동폭만 크면 H2.

  H3 진짜 축 구조    2023에 `gap_success`가 실제로 더 유용했다 (선수 이동·규칙 변화 등).
                    → H0~H2를 전부 배제해야만 남는다. **잔여 가설로만 다룬다.**

⛔ 어느 가설이든 **2025에 쓸 수 있는가**가 최종 질문이다. 세 가설의 함의:
     H0 → 무시하고 pooled.   H1 → 2025 기저는 2024를 닮으므로 2024쪽 β.
     H2 → 2023은 전환기, 2025는 멀다 → 무시.   H3 → 그때만 열린다.

실행:  .venv/Scripts/python.exe -u src/gapstruct.py 2>&1 | tee experiments/gapstruct.log
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
import jointbeta as J
from config import (ASOF_AXES, ASOF_STRENGTH, CONTAM_AXES, CTX_AXES, CTX_HALF_LIFE,
                    CTX_K, MLP_ALPHA, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from validation import _logit, bss, shift_to_prior

FOLDS = [2021, 2022, 2023, 2024]
GI = 2                      # gap_success 의 열 인덱스
SEED = 42


def axes_at(df, s, fold, contam=True):
    """§7.83의 `axis_matrix`와 같은데 오염 보정을 끌 수 있다 (H2 검정용)."""
    tr_m, va_m = s < fold, s == fold
    tr, va = df[tr_m], df[va_m].copy()
    ref = int(s[tr_m].max())
    w = recency_weight(s[tr_m], CTX_HALF_LIFE, ref_season=ref)
    ctx = ctxfit.fit_ctx(tr, TARGET_COL, CTX_AXES, CTX_K, weights=w)
    art = F.fit_artifacts(tr, ctx=ctx, contam_axes=(CONTAM_AXES if contam else None))
    ct = art.get("contam") if contam else None
    asof = asofdelta.fit_asof_axes(tr, ASOF_AXES, ASOF_STRENGTH, ct=ct)
    cols = [F.ctx_lookup(va, ax) for ax in ctx]
    cols += [F.asof_lookup(va, a, ct=ct) for a in asof]
    names = [ax["name"] for ax in ctx] + [a["name"] for a in asof]
    return np.column_stack(cols), names, va


def fit_se(X, z0, y, iters=60):
    """가중 없는 Brier M-추정 + **샌드위치 공분산** H⁻¹JH⁻¹.

    Brier는 우도가 아니다. H⁻¹만 쓰면 SE가 틀린다 (정보등식이 성립 안 한다).
    """
    n, k = X.shape
    th = np.zeros(k)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(z0 + X @ th)))
        v, r = p * (1 - p), None
        r = p - y
        gi = 2.0 * r * v                       # 행별 dL/dη
        hi = 2.0 * (v * v + r * v * (1 - 2 * p))
        g = X.T @ gi / n
        H = (X.T * (hi / n)) @ X + np.eye(k) * 1e-12
        st = np.linalg.solve(H, g)
        th -= st
        if np.max(np.abs(st)) < 1e-13:
            break
    p = 1 / (1 + np.exp(-(z0 + X @ th)))
    v, r = p * (1 - p), p - y
    gi = 2.0 * r * v
    hi = 2.0 * (v * v + r * v * (1 - 2 * p))
    # ⚠️ 목적함수가 **평균**(1/n)이므로 A·B 둘 다 1/n을 달고, Var는 다시 1/n이다.
    #    Var(β̂) = (1/n)·A⁻¹ B A⁻¹,  A = (1/n)XᵀdiagₕX,  B = (1/n)Xᵀdiag_{g²}X
    #    1/n을 빠뜨리면 SE가 n배(≈22만배)로 나온다.
    A = (X.T * (hi / n)) @ X
    Bm = (X.T * (gi ** 2 / n)) @ X
    Ai = np.linalg.inv(A + np.eye(k) * 1e-12)
    cov = Ai @ Bm @ Ai / n
    return th, np.sqrt(np.diag(cov))


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = J.foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(SEED)

    D = {}
    for f in FOLDS:
        X, names, va = axes_at(df, s, f, contam=True)
        y = cached[f][2]
        z0 = (1 - MLP_ALPHA) * cached[f][0] + MLP_ALPHA * cached[f][1]
        assert np.array_equal(y, va[TARGET_COL].to_numpy(dtype=np.float64))
        m = va["game_type"].to_numpy() != "F"
        Xr, yr, zr = X[m], y[m], z0[m]
        zr = _logit(shift_to_prior(1 / (1 + np.exp(-zr)), yr.mean()))
        D[f] = dict(X=Xr, y=yr, z0=zr, sd=Xr.std(axis=0),
                    mon=va["game_month"].to_numpy()[m], n=int(m.sum()))
        raw = bss(yr, shift_to_prior(1 / (1 + np.exp(-zr)), yr.mean()))
        D[f]["raw"] = raw
        print(f"fold {f}  R n={D[f]['n']:,}  블렌드 단독 BSS {raw:.1f}"
              f"  std(z0) {zr.std():.4f}")
    print(f"\n축 {names}   (검정 대상 = {names[GI]})")

    def solo(f, X=None, z0=None, y=None):
        X = D[f]["X"] if X is None else X
        z0 = D[f]["z0"] if z0 is None else z0
        y = D[f]["y"] if y is None else y
        sd = X.std(axis=0)
        th, se = fit_se(X / sd, z0, y)
        return th / sd, se / sd

    # ── H0-a 샌드위치 SE ────────────────────────────────────────────────
    print(f"\n{'='*80}\n■ H0-a  β̂의 샌드위치 SE — .073과 .037이 구별되는가\n{'='*80}")
    B, S = {}, {}
    for f in FOLDS:
        B[f], S[f] = solo(f)
    print(f"  {'축':<15}" + "".join(f"{f:>18}" for f in FOLDS))
    for i, nm in enumerate(names):
        print(f"  {nm:<15}" + "".join(f"{B[f][i]:>11.4f}±{S[f][i]:<6.4f}"
                                     for f in FOLDS))
    others = [f for f in FOLDS if f != 2023]
    om = float(np.mean([B[f][GI] for f in others]))
    ose = float(np.sqrt(sum(S[f][GI] ** 2 for f in others)) / len(others))
    z = (B[2023][GI] - om) / np.sqrt(S[2023][GI] ** 2 + ose ** 2)
    print(f"\n  2023 {B[2023][GI]:.4f}±{S[2023][GI]:.4f}   "
          f"나머지 셋 평균 {om:.4f}±{ose:.4f}   →  **{z:+.1f}σ**")
    print(f"  → {'H0(표집 노이즈) 기각 — 진짜 아웃라이어' if abs(z) > 3 else 'H0 못 기각'}")

    # ── H0-b split-half ─────────────────────────────────────────────────
    print(f"\n{'='*80}\n■ H0-b  fold 내 split-half — 추정치가 fold 안에서 안정적인가\n{'='*80}")
    print("  무작위 분할 = 표집 안정성.  시간 분할(전반기/후반기) = 시즌 내 균질성.\n")
    # 무작위 분할은 **10회** 반복한다 — 1회 |차|는 그 자체가 노이즈다.
    # 반쪽 차의 표준편차 ≈ 2·SE(전체) 이므로 샌드위치 SE의 독립 검증이 된다.
    K = 10
    print(f"  {'fold':>6}{'전체':>9}{'무작위 반쪽 sd':>15}{'→ 추정 SE':>11}"
          f"{'샌드위치 SE':>13}{'전반기':>9}{'후반기':>9}{'차':>9}")
    for f in FOLDS:
        n = D[f]["n"]
        dif = []
        for _ in range(K):
            idx = rng.permutation(n)
            ha, hb = idx[: n // 2], idx[n // 2:]
            dif.append(solo(f, D[f]["X"][ha], D[f]["z0"][ha], D[f]["y"][ha])[0][GI]
                       - solo(f, D[f]["X"][hb], D[f]["z0"][hb], D[f]["y"][hb])[0][GI])
        sd_d = float(np.std(dif, ddof=1))
        m1 = D[f]["mon"] < 7
        t1 = solo(f, D[f]["X"][m1], D[f]["z0"][m1], D[f]["y"][m1])[0][GI]
        t2 = solo(f, D[f]["X"][~m1], D[f]["z0"][~m1], D[f]["y"][~m1])[0][GI]
        print(f"  {f:>6}{B[f][GI]:>9.4f}{sd_d:>15.4f}{sd_d/2:>11.4f}"
              f"{S[f][GI]:>13.4f}{t1:>9.4f}{t2:>9.4f}{t2-t1:>+9.4f}")

    # ── H1 기저 모델 강도 ───────────────────────────────────────────────
    print(f"\n{'='*80}\n■ H1  기저 약화 → β̂ 상승?  (z0에 배율 c)\n{'='*80}")
    print("  기저가 약하면 보정축이 먹을 잔차가 커진다. c를 내려 인위로 약화시킨다.\n")
    print(f"  {'c':>6} │ " + " │ ".join(f"{f:>16}" for f in FOLDS))
    for c in (1.4, 1.2, 1.0, 0.8, 0.6, 0.4):
        row = []
        for f in FOLDS:
            b = solo(f, z0=D[f]["z0"] * c)[0][GI]
            row.append(f"{b:>16.4f}")
        print(f"  {c:>6.1f} │ " + " │ ".join(row))
    print(f"\n  참고 — 블렌드 단독 BSS  " + "  ".join(f"{f}:{D[f]['raw']:.0f}"
                                                  for f in FOLDS))
    print(f"        β̂(gap_success)    " + "  ".join(f"{f}:{B[f][GI]:.4f}"
                                                   for f in FOLDS))

    # ── H2 오염 보정 ON/OFF ─────────────────────────────────────────────
    print(f"\n{'='*80}\n■ H2  오염 보정 ON/OFF — 2023의 이동폭만 큰가\n{'='*80}")
    print("  gap = (올해) − (커리어). 2023 행의 커리어는 거의 전부 오염 시즌이다.\n")
    print(f"  {'fold':>6}{'보정 ON':>12}{'보정 OFF':>12}{'Δ':>10}{'Δ/ON':>9}"
          f"{'커리어 오염시즌 비중':>22}")
    for f in FOLDS:
        Xo, _, va = axes_at(df, s, f, contam=False)
        m = va["game_type"].to_numpy() != "F"
        bo = solo(f, X=Xo[m])[0][GI]
        # 커리어 중 오염 시즌(≤2022) 비중 — 그 fold 이전 시즌 구성
        past = [p for p in range(2019, f)]
        share = sum(1 for p in past if p <= 2022) / max(len(past), 1)
        print(f"  {f:>6}{B[f][GI]:>12.4f}{bo:>12.4f}{bo-B[f][GI]:>+10.4f}"
              f"{(bo-B[f][GI])/B[f][GI]:>+9.1%}{share:>21.0%}")

    # ── H2b 전환기 서명 — 커리어 길이별 β ───────────────────────────────
    # 2023은 반대성 판정 기준이 **켜진 첫 시즌**이다 (§7.48). 그러면 커리어가
    # 전부 옛 기준으로 쌓인 베테랑일수록 gap이 "기준 전환분"을 더 많이 싣는다.
    #   전환기 가설  → 2023에서만 커리어 길이에 따라 β가 오른다
    #   기계적 효과  → 모든 fold에서 같은 기울기가 나온다 (표본 적으면 수축돼서)
    # **대조군이 판정자다** (memory: 노출 진단은 노출 채널만 닫는다).
    print(f"\n{'='*80}\n■ H2b  전환기 서명 — 커리어 길이 4분위별 β̂(gap_success)\n{'='*80}")
    print("  2023은 반대성 기준이 켜진 첫 시즌. 커리어가 전부 옛 기준인 베테랑일수록")
    print("  gap이 '기준 전환분'을 싣는다면 **2023에서만** 기울기가 나와야 한다.\n")
    print(f"  {'fold':>6}" + "".join(f"{f'Q{q+1}':>11}" for q in range(4))
          + f"{'Q4/Q1':>9}{'Q4−Q1':>10}")
    for f in FOLDS:
        _, _, va = axes_at(df, s, f, contam=True)
        m = va["game_type"].to_numpy() != "F"
        hist = df.loc[s < f, "pitcher_id"].value_counts()
        cn = hist.reindex(va["pitcher_id"].to_numpy()).fillna(0).to_numpy()[m]
        qs = np.quantile(cn, [0.25, 0.5, 0.75])
        lab = np.searchsorted(qs, cn, side="right")
        row = []
        for q in range(4):
            mm = lab == q
            row.append(solo(f, D[f]["X"][mm], D[f]["z0"][mm], D[f]["y"][mm])[0][GI]
                       if mm.sum() > 5000 else float("nan"))
        print(f"  {f:>6}" + "".join(f"{x:>11.4f}" for x in row)
              + f"{row[3]/row[0]:>9.2f}{row[3]-row[0]:>+10.4f}")


if __name__ == "__main__":
    main()
