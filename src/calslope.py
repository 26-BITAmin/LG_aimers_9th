"""calibration slope 진단 — δ 보정(β) 전/후의 로짓 기울기를 fold 별로 잰다.

    logit(y=1) = a + b·z,   z = logit(p)

`b > 1` 이면 예측 로짓이 **너무 눌려 있다**(underconfident) → 벌려야 하고,
`b < 1` 이면 과신이라 줄여야 한다. Brier 는 proper scoring rule 이므로 이 기울기
오차는 곧바로 점수다.

### 이 저장소에서 반드시 붙는 단서 세 가지

1. **`a_large` 는 항등적으로 0 이다.** 하네스가 `p = shift_to_prior(σ(z), ȳ)` 로
   평균을 못박는데, 그건 `mean(σ(z+δ)) = ȳ` 를 푸는 것이고 offset GLM 의 절편
   추정식과 **같은 식**이다. 그래서 절편은 정보를 안 준다 — 검증용으로만 찍는다.
   대신 `a_nsh` = 평균 고정을 **풀었을 때**의 절편을 같이 낸다. 이것이 배포의
   `PROB_SHIFT` 가 채워야 하는 레벨이고, 그쪽은 실제로 정보가 있다.

2. **배포 스택에는 이미 기울기 노브가 있다.** `predict.py` 의
   `p' = P0 + 1.05·(p − P0)` 는 p≈0.48 근방에서 로짓 기울기 ≈1.05 다.
   하네스 층에는 이게 **안 들어가 있다.** 그래서 층을 셋으로 나눠 잰다:
       p_raw  = δ 없음, 스케일 없음
       p_beta = δ 있음, 스케일 없음      ← `deploy29.state()["p0"]` 와 동일
       p_dep  = δ 있음, 스케일 1.05      ← 실제 배포와 등가
   **`p_dep` 의 b 가 이 축에 남은 여지다.** `p_beta` 의 b 만 보고 움직이면
   이미 배포한 상수를 두 번 넣는다.

3. **β 는 이미 LB 로 닫혔다** (§7.132, `β* = 0.9907`, 전액 +0.007).
   그러니 이 진단이 "δ 를 더/덜 걸어라" 로 읽히면 그건 LB 확정과 충돌한다.
   **δ 의 세기(β)와 전체 로짓의 기울기(b)는 다른 조작이다** — β 는 보정항만,
   b 는 스택 전체를 곱한다. 아래는 후자를 잰다. 전자에 대해서는 `p_raw` 와
   `p_beta` 의 b 를 **비교**하는 것으로만 발언한다(δ 가 기울기를 고쳤는가).

### 채택 판단은 in-sample b 로 하지 않는다

같은 fold 에서 적합한 b 를 그 fold 에 적용하면 이득이 구조적으로 양수다.
저장소 규약대로 **다른 fold 에서 적합한 b 를 가져와 적용한 ΔBSS**(전이)를
투수 클러스터 부트스트랩과 함께 같이 낸다. 그것이 결정 수치다.

실행:  .venv/Scripts/python.exe -u src/calslope.py 2>&1 | tee experiments/calslope.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np
from scipy import stats

import deploy29 as D
from catens import SEEDS, VARIANTS
from config import FSPEC_ALPHA, MLP_ALPHA
from predict import PROB_CENTER, PROB_SCALE
from validation import bss, shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
CAT_W = 0.10                 # §7.142 사전 고정
B_BOOT = 8000               # §7.145 — B=400 은 0 배제 판정을 뒤집을 수 있다
EPS = 1e-6


# ── binomial GLM (logit) — statsmodels 없이 ──────────────────────────
# statsmodels 를 넣으면 pinned scipy/pandas 가 딸려 올라가 제출 pickle 이
# 깨질 수 있다 (requirements 정확 일치 규칙). 2열짜리 IRLS 는 직접 쓴다.
def _glm(X, y, offset=None, tol=1e-11, max_iter=100):
    """Newton-Raphson(= IRLS). (beta, mu, XtWX) 를 돌려준다."""
    n, k = X.shape
    beta = np.zeros(k)
    off = np.zeros(n) if offset is None else offset
    XtWX = None
    for _ in range(max_iter):
        mu = 1.0 / (1.0 + np.exp(-(X @ beta + off)))
        w = np.clip(mu * (1.0 - mu), 1e-12, None)
        XtWX = X.T @ (X * w[:, None])
        step = np.linalg.solve(XtWX, X.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    mu = 1.0 / (1.0 + np.exp(-(X @ beta + off)))
    return beta, mu, XtWX


def _llf(y, mu):
    mu = np.clip(mu, EPS, 1 - EPS)
    return float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))


def _cluster_cov(X, y, mu, XtWX, groups, n_g):
    """CR1 샌드위치 — statsmodels cov_type='cluster' 기본 보정과 동형.

        V = (X'WX)^-1 · (Σ_g u_g u_g') · (X'WX)^-1 · c,
        u_g = Σ_{i∈g} x_i (y_i − μ_i),  c = G/(G−1) · (n−1)/(n−k)
    """
    n, k = X.shape
    resid = y - mu
    U = np.column_stack([np.bincount(groups, weights=X[:, j] * resid,
                                     minlength=n_g) for j in range(k)])
    seen = np.flatnonzero(np.bincount(groups, minlength=n_g) > 0)
    U = U[seen]
    G = len(seen)
    bread = np.linalg.inv(XtWX)
    c = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    return bread @ (U.T @ U) @ bread * c, G


def calib_slope(y, p, groups=None, n_g=None, bss_current=None):
    """(1) calibration-in-the-large, (2) calibration slope, (3) joint 검정."""
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    z = np.log(p / (1 - p))
    n = len(y)

    # (1) slope 를 1 로 고정하고 절편만 — offset GLM
    a_large, _, _ = _glm(np.ones((n, 1)), y, offset=z)
    a_large = float(a_large[0])

    # (2) 절편 + 기울기
    X = np.column_stack([np.ones(n), z])
    (a, b), mu, XtWX = _glm(X, y)
    V_model = np.linalg.inv(XtWX)
    if groups is not None:
        V, G = _cluster_cov(X, y, mu, XtWX, groups, n_g)
    else:
        V, G = V_model, None
    se_a, se_b = np.sqrt(np.diag(V))
    lo, hi = b - 1.96 * se_b, b + 1.96 * se_b

    # (3) (a,b)=(0,1) 에 대한 joint 검정, df=2
    ll_null = _llf(y, p)                       # a=0, b=1 → μ = p
    LR = 2.0 * (_llf(y, mu) - ll_null)
    p_lr = float(stats.chi2.sf(LR, df=2))
    d = np.array([a - 0.0, b - 1.0])           # 클러스터 강건 Wald (같은 귀무가설)
    W = float(d @ np.linalg.solve(V, d))
    p_wald = float(stats.chi2.sf(W, df=2))

    return dict(a_large=a_large, a=float(a), b=float(b), se_b=float(se_b),
                b_ci=(float(lo), float(hi)), se_ratio=float(se_b / np.sqrt(V_model[1, 1])),
                LR=float(LR), p=p_lr, W=W, p_wald=p_wald, G=G,
                dBSS_expected=(float(b) - 1.0) ** 2 * (bss_current or 0.0))


# ── 기울기 재보정을 실제로 적용했을 때의 ΔBSS ────────────────────────
def apply_slope(z, ybar, b):
    """σ(b·z) 를 다시 prior 로 옮긴다.

    절편 a 는 재-shift 가 흡수하므로 여기 안 들어간다 — 배포에 이미 레벨 노브가
    있으니 **기울기만의 효과**를 재는 것이 맞다.
    """
    return shift_to_prior(1.0 / (1.0 + np.exp(-(b * z))), ybar)


def dbss_boot(y, p0, p1, cl, n_id, rng, B=B_BOOT):
    """투수 클러스터 부트스트랩 ΔBSS 구간 (catfr2.sub_bss_delta 와 같은 기계)."""
    ybar = y.mean()
    ref = ybar * (1 - ybar)
    e0 = (p0 - y) ** 2
    e1 = (p1 - y) ** 2
    n = np.bincount(cl, minlength=n_id)
    s0 = np.bincount(cl, weights=e0, minlength=n_id)
    s1 = np.bincount(cl, weights=e1, minlength=n_id)
    seen = np.flatnonzero(n > 0)
    d = 1e5 * (s0.sum() - s1.sum()) / (n.sum() * ref)
    draws = np.empty(B)
    for i in range(B):
        pick = rng.choice(seen, size=len(seen), replace=True)
        draws[i] = 1e5 * (s0[pick].sum() - s1[pick].sum()) / (n[pick].sum() * ref)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(d), float(lo), float(hi), float(np.mean(draws > 0))


# ── 층 구성 ──────────────────────────────────────────────────────────
def build_layers(df, s, fold, mm):
    """fold 의 clean 행 기준 z_blend 층들. (δ 는 여기 안 들어간다)"""
    with np.load(CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
        z_mlp = d0["z_mlp"]
    z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
    z29 = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
    out = [("s29", z29)]

    fs = CACHE / f"s35_fspec_fold{fold}.npz"
    if fs.exists():                        # fspec 학습셋이 2023 F 라 fold2024 만
        z35 = z29.copy()
        vf = df[s == fold]["game_type"].to_numpy().astype(str) == "F"
        z35[vf] = (1 - FSPEC_ALPHA) * z29[vf] + FSPEC_ALPHA * np.load(fs)["z"][vf]
        out.append(("s35", z35))

    ens = np.mean([np.load(CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz")["z"]
                   for v in VARIANTS for sd in SEEDS], axis=0)
    # cat10 은 **두 층 모두**에 얹는다. s35 는 fold2024 에만 있어서, 공통 층이
    # s29 계열밖에 없으면 전이(다른 fold 의 b 적용) 비교가 불가능해진다.
    for nm, z in list(out):
        out.append((nm + "+cat10", (1 - CAT_W) * z + CAT_W * ens))
    return [(nm, z[mm]) for nm, z in out]


def main():
    df, s, y = D.load_df()
    rng = np.random.default_rng(42)
    print(f"PROB_CENTER {PROB_CENTER}  PROB_SCALE {PROB_SCALE}  "
          f"CAT_W {CAT_W}  B {B_BOOT}\n")

    store = {}      # (fold, layer, variant) -> dict(z=, p=, b=, ybar=, ...)
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        yv, cl, n_id, mm = st["yv"], st["cl"], st["n_id"], st["mm"]
        ax = st["Xax"] @ st["dep"]              # δ 보정 로짓 (clean 행)
        ybar = yv.mean()
        print(f"\n{'='*96}\nfold {fold}   clean {len(yv):,}행 · 투수 "
              f"{len(np.unique(cl)):,}명 · ȳ {ybar:.6f} · 기준 BSS {st['base']:.1f}")
        print(f"{'='*96}")

        for nm, zb in build_layers(df, s, fold, mm):
            p_raw = shift_to_prior(1 / (1 + np.exp(-zb)), ybar)
            p_beta = shift_to_prior(1 / (1 + np.exp(-(zb + ax))), ybar)
            p_dep = shift_to_prior(
                PROB_CENTER + PROB_SCALE * (p_beta - PROB_CENTER), ybar)
            # 평균 고정을 푼 판 — PROB_SHIFT 가 채워야 하는 레벨을 보려고
            p_nsh = 1 / (1 + np.exp(-(zb + ax)))

            print(f"\n  ── 층 {nm} ──  σ(z)={np.std(zb + ax):.4f}  "
                  f"p∈[{p_beta.min():.3f}, {p_beta.max():.3f}]")
            print(f"    {'변형':<22s}{'BSS':>9s}{'a_large':>10s}{'b':>8s}"
                  f"{'95% CI (클러스터)':>22s}{'SE배':>7s}{'LR p':>9s}"
                  f"{'Wald p':>9s}{'ΔBSS기대':>10s}")
            for vn, pp in (("p_raw  (δ 없음)", p_raw),
                           ("p_beta (δ, 배포하네스)", p_beta),
                           ("p_dep  (δ+스케일1.05)", p_dep)):
                cur = bss(yv, pp)
                r = calib_slope(yv, pp, groups=cl, n_g=n_id, bss_current=cur)
                print(f"    {vn:<22s}{cur:>9.1f}{r['a_large']:>10.5f}"
                      f"{r['b']:>8.4f}   [{r['b_ci'][0]:>7.4f}, {r['b_ci'][1]:>7.4f}]"
                      f"{r['se_ratio']:>7.2f}{r['p']:>9.2e}{r['p_wald']:>9.2e}"
                      f"{r['dBSS_expected']:>10.2f}")
                store[(fold, nm, vn)] = dict(
                    z=np.log(np.clip(pp, EPS, 1 - EPS)
                             / (1 - np.clip(pp, EPS, 1 - EPS))),
                    p=pp, b=r["b"], ybar=ybar, cur=cur, yv=yv, cl=cl, n_id=n_id)

            rn = calib_slope(yv, np.clip(p_nsh, EPS, 1 - EPS), groups=cl, n_g=n_id)
            print(f"    {'(평균 고정 해제)':<22s}{'':>9s}{rn['a_large']:>10.5f}"
                  f"{rn['b']:>8.4f}   ← a_large 가 PROB_SHIFT 가 채우는 레벨")

            # in-sample 재보정 이득 (구조적 양수 — 상한으로만 읽는다)
            for vn in ("p_beta (δ, 배포하네스)", "p_dep  (δ+스케일1.05)"):
                e = store[(fold, nm, vn)]
                d = bss(yv, apply_slope(e["z"], ybar, e["b"])) - e["cur"]
                print(f"    {'in-sample 재보정':<22s}{vn:<24s}b={e['b']:.4f}"
                      f"  ΔBSS {d:+.2f}  (상한)")

    # ── 전이: 다른 fold 에서 적합한 b 를 가져와 적용 ────────────────
    print(f"\n\n{'='*96}\n전이 — b 를 **다른 fold** 에서 적합해 적용한 ΔBSS "
          f"(클러스터 부트스트랩 B={B_BOOT})\n{'='*96}")
    print(f"  {'fold':<6s}{'층':<12s}{'변형':<24s}{'b_타fold':>10s}"
          f"{'ΔBSS':>9s}{'2.5%':>9s}{'97.5%':>9s}{'P(Δ>0)':>9s}")
    for (fold, nm, vn), e in store.items():
        other = [f for f in D.FOLDS if f != fold]
        for of in other:
            src = store.get((of, nm, vn))
            if src is None:
                continue
            p1 = apply_slope(e["z"], e["ybar"], src["b"])
            d, lo, hi, pg = dbss_boot(e["yv"], e["p"], p1, e["cl"], e["n_id"], rng)
            print(f"  {fold:<6d}{nm:<12s}{vn:<24s}{src['b']:>10.4f}"
                  f"{d:>9.2f}{lo:>9.2f}{hi:>9.2f}{pg:>9.3f}")


def pair_boot(yv, plist, cl, n_id, rng, B=400):
    """투수 클러스터를 재표집하고 **같은 재표집 위에서** 여러 p 의 b 를 다시 적합.

    b_beta 와 b_raw 는 **같은 행**에서 나온다. 각자의 주변 CI 는 그 상관을 버려서
    차이의 증거를 과소평가한다 (저장소 교훈: "쌍체 비교는 클러스터 상관을 흡수한다").
    여기서는 draw 마다 두 GLM 을 다시 풀어 **차이의 분포**를 직접 만든다.
    """
    order = np.argsort(cl, kind="stable")
    cs = cl[order]
    bounds = np.searchsorted(cs, np.arange(n_id + 1))
    seen = np.flatnonzero(np.diff(bounds) > 0)
    zs = [np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))[order]
          for p in plist]
    ys = yv[order]
    out = np.empty((B, len(plist)))
    for i in range(B):
        pick = rng.choice(seen, size=len(seen), replace=True)
        idx = np.concatenate([order[bounds[g]:bounds[g + 1]] - order[bounds[g]:bounds[g + 1]]
                              + np.arange(bounds[g], bounds[g + 1]) for g in pick])
        yb = ys[idx]
        for j, z in enumerate(zs):
            zb = z[idx]
            out[i, j] = _glm(np.column_stack([np.ones(len(zb)), zb]), yb)[0][1]
    return out


# 부분군별 slope 는 `src/calsub.py` 로 분리했다 (§7.147 후속).


def main_pair(B=400):
    """b_beta − b_raw 와 b_dep − 1 을 쌍체 클러스터 부트스트랩으로."""
    df, s, y = D.load_df()
    rng = np.random.default_rng(42)
    print(f"쌍체 클러스터 부트스트랩  B={B}\n")
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        yv, cl, n_id, mm = st["yv"], st["cl"], st["n_id"], st["mm"]
        ax = st["Xax"] @ st["dep"]
        ybar = yv.mean()
        layers = build_layers(df, s, fold, mm)
        nm, zb = layers[-1]                      # 그 fold 의 최상위 배포 등가 층
        p_raw = shift_to_prior(1 / (1 + np.exp(-zb)), ybar)
        p_beta = shift_to_prior(1 / (1 + np.exp(-(zb + ax))), ybar)
        p_dep = shift_to_prior(PROB_CENTER + PROB_SCALE * (p_beta - PROB_CENTER), ybar)
        d = pair_boot(yv, [p_raw, p_beta, p_dep], cl, n_id, rng, B)
        br, bb, bd = d[:, 0], d[:, 1], d[:, 2]
        diff = bb - br
        print(f"fold {fold}  층 {nm}  (투수 {len(np.unique(cl))}명)")
        for lab, v, null in (("b_raw", br, 1.0), ("b_beta", bb, 1.0),
                             ("b_dep", bd, 1.0)):
            lo, hi = np.percentile(v, [2.5, 97.5])
            print(f"    {lab:<18s}{v.mean():>8.4f}  [{lo:>7.4f}, {hi:>7.4f}]"
                  f"   P(>{null:.0f}) {np.mean(v > null):.3f}")
        lo, hi = np.percentile(diff, [2.5, 97.5])
        print(f"    {'b_beta − b_raw':<18s}{diff.mean():>8.4f}  [{lo:>7.4f}, {hi:>7.4f}]"
              f"   P(<0) {np.mean(diff < 0):.3f}   ← δ 가 기울기를 1 쪽으로 당겼나")
        print(f"    (쌍체 SD {diff.std():.4f} vs 주변 SD 합 "
              f"{np.sqrt(br.var() + bb.var()):.4f} — 상관 흡수분)\n")


if __name__ == "__main__":
    if "--pair" in sys.argv:
        main_pair()
    else:
        main()
