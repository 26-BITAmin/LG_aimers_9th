""""튀는 투수에 대해 모델이 더 확신하게" — 투수 레벨 캘리브레이션 기울기 (사용자 요청).

    "나는 이렇게 확실하게 좀 튀는 선수들에 대해 모델이 더 확신을 가지고
     (0이나 1에 더 가깝게) 예측했으면 좋겠음"

§7.104 파트 E 와 **다른 축이다.** 거기서는 투수마다 자유 파라미터 1개씩(≈700개)을
과거 **잔차**로 추정해 더했고 위약과 같은 자리에서 죽었다. 이번 요청은 모델이 이미
만들어낸 투수별 편차를 **γ배로 키우자**는 것 — 파라미터가 **스칼라 1개**다.
[전이 배율은 얼리는 파라미터 수로 갈린다]: 스칼라 1개 0.93 / 표 3.3 / 트리 분할 68.6.
700개가 죽었다는 사실은 1개에 대한 증거가 아니다.

핵심 통계량 — **투수 레벨 캘리브레이션 기울기 β**

    m_j = mean_j(p̂) − mean(p̂)        모델이 말하는 투수 j 의 편차
    d_j = mean_j(y) − mean(y)         실제 편차
    d_j = β · m_j

    β > 1  →  모델이 **과소확신**. 요청이 맞고 키우면 이득이다.
    β = 1  →  이미 정확. 키우면 손해다.
    β < 1  →  이미 **과대확신**. 오히려 더 수축해야 한다.

행 수준으로 옮기면 `resid ~ m_j` 의 기울기 s 가 정확히 **β−1** 이다
(d_j = m_j + resid_j 이므로). §7.90·§7.104에서 검증된 `resid_t` 하네스를 그대로 쓴다.
⚠️ m_j 는 **예측의 평균**이라 라벨을 안 쓴다 — 자기 행이 m_j 에 들어가도 편향이 없다.

파트
  1  전역 온도 c 스윕 (참고 기준선 — 모델이 통째로 과소확신인가)
  2  ★ 투수 레벨 β — fold별 · |m| 3분위별 · 위약
  3  행동: z' = z + γ·h_j  (h_j 는 **과거 시즌 예측만**으로 만든 투수 오프셋)
       γ 스윕 · 꼬리 게이팅 · 위약. 배포 가능한 형태로만 잰다.

실행:  .venv/Scripts/python.exe -u src/psharp.py 2>&1 | tee experiments/psharp.log
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

import jointbeta as J
from ctxsplit import resid_t
from config import ASOF_AXES, CTX_AXES, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = [2021, 2022, 2023, 2024]
ACT_FOLDS = [2022, 2023, 2024]          # 과거 예측이 있어야 h_j 를 만든다
GAMMAS = [-0.50, -0.25, 0.0, 0.25, 0.50, 1.00, 2.00]
SEED = 42
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def resid_t_cluster(r, x, cl, n_id):
    """`resid_t` 의 **클러스터(투수) 강건** 판. 기울기는 같고 t 만 바뀐다.

    ⚠️ OLS SE 는 행을 독립으로 센다. 잔차가 투수 안에서 상관되어 있으면(여기서는
       σ_b=0.0147 만큼 실제로 그렇다) t 가 √(행/투수) 배쯤 부풀려진다.
       23만 행 / 400명이면 24배다. CR0 로 다시 센다.
    """
    xc = x - x.mean()
    den = float((xc ** 2).sum())
    if den <= 0:
        return 0.0, 0.0
    sl = float((xc * r).sum() / den)
    e = r - r.mean() - sl * xc
    g = np.bincount(cl, weights=xc * e, minlength=n_id)     # 투수별 스코어 합
    v = float((g ** 2).sum()) / den ** 2
    return sl, (sl / np.sqrt(v) if v > 0 else 0.0)


def main():
    rng = np.random.default_rng(SEED)
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    pid = df["pitcher_id"].to_numpy()
    codes, uniq = pd.factorize(pid)
    n_id = len(uniq)
    mclean = ~((gt == "F") & (s < 2023))

    # ── 배포 홀드아웃 예측 (§7.104 와 동일 절차) ────────────────────────
    cached = J.foldcache.load(FOLDS, verbose=False)
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    zraw = np.full(len(df), np.nan)      # prior shift **전** 로짓 (배포 순서: 샤프닝→시프트)
    pmod = np.full(len(df), np.nan)
    for f in FOLDS:
        X, _, _, va = J.axis_matrix(df, s, f)
        yv = cached[f][2]
        vi = np.where(s == f)[0]
        assert len(vi) == len(yv) and np.allclose(y[vi], yv), f"fold {f} 정렬 실패"
        mm = mclean[vi]
        zb = (1 - MLP_ALPHA) * cached[f][0] + MLP_ALPHA * cached[f][1]
        zd = zb + X @ dep
        z_ = np.full(len(vi), np.nan); z_[mm] = zd[mm]
        p_ = np.full(len(vi), np.nan)
        p_[mm] = shift_to_prior(1 / (1 + np.exp(-zd[mm])), yv[mm].mean())
        zraw[vi], pmod[vi] = z_, p_
    mmod = mclean & np.isfinite(pmod)
    print(f"train {df.shape}   clean 2021~2024 홀드아웃 {int(mmod.sum()):,}행")

    # ── 파트 1 — 전역 온도 (참고) ───────────────────────────────────────
    print(f"\n{'='*80}\n█ 파트 1 — 전역 온도 c (참고 기준선)\n{'='*80}")
    print("  z' = c·(z − z̄).  c>1 = 통째로 더 확신. 배포는 c=1 이다.\n")
    cs = [0.90, 0.95, 1.00, 1.05, 1.10, 1.20]
    print(f"  {'fold':<8}{'std(p̂)':>9}{'std(y)':>8}" + "".join(f"{f'c={c:.2f}':>10}" for c in cs))
    for f in FOLDS:
        m = mmod & (s == f)
        zz, yy = zraw[m], y[m]
        row = []
        for c in cs:
            p = shift_to_prior(1 / (1 + np.exp(-(zz.mean() + c * (zz - zz.mean())))), yy.mean())
            row.append(bss(yy, p))
        print(f"  {f:<8}{pmod[m].std():>9.4f}{yy.std():>8.4f}"
              + "".join(f"{v:>10.1f}" for v in row))

    # ── 파트 2 — ★ 투수 레벨 캘리브레이션 기울기 ─────────────────────────
    print(f"\n{'='*80}\n█ 파트 2 — ★ 투수 레벨 기울기 β   (β = 1 + slope[resid ~ m_j])\n{'='*80}")
    print("  β>1 과소확신(키우면 이득) · β=1 정확 · β<1 과대확신(더 수축해야)\n")
    print("  ⚠️ t 는 **투수 클러스터 강건**(CR0). OLS t 는 행을 독립으로 세서 부풀려진다.\n")
    print(f"  {'fold':<8}{'투수':>6}{'std(m_j)':>10}{'β':>8}{'t(CR)':>8}{'t(OLS)':>8}"
          f"{'위약 β':>9}{'위약 t':>8}   │ |m| 3분위별 β (하위/중간/상위)")
    for f in FOLDS:
        m = mmod & (s == f)
        cf, yy, pp = codes[m], y[m], pmod[m]
        n = np.bincount(cf, minlength=n_id).astype(float)
        P = np.bincount(cf, weights=pp, minlength=n_id)
        mj = np.where(n > 0, P / np.maximum(n, 1) - pp.mean(), 0.0)
        r = yy - pp
        sl, t = resid_t_cluster(r, mj[cf], cf, n_id)
        t_ols = resid_t(r, mj[cf])[1]
        # 위약 — 투수 정체만 뒤섞는다 (m_j 의 분포·크기는 그대로)
        seen = np.where(n > 0)[0]
        ms = mj.copy(); ms[seen] = mj[rng.permutation(seen)]
        sl_s, t_s = resid_t_cluster(r, ms[cf], cf, n_id)
        # |m| 3분위 — 꼬리에서 기울기가 다른가 (사용자 주장의 핵심)
        a = np.abs(mj[cf])
        q = np.quantile(a, [1 / 3, 2 / 3])
        ter = []
        for lo, hi in ((-1, q[0]), (q[0], q[1]), (q[1], 9)):
            g = (a > lo) & (a <= hi)
            ter.append(1 + resid_t_cluster(r[g], mj[cf][g], cf[g], n_id)[0])
        print(f"  {f:<8}{int((n>0).sum()):>6}{mj[cf].std():>10.4f}{1+sl:>8.3f}{t:>8.2f}{t_ols:>8.2f}"
              f"{1+sl_s:>9.3f}{t_s:>8.2f}   │ "
              + "  ".join(f"{b:.3f}" for b in ter))

    # ── 파트 2b — ★ m_j 를 '미리 알 수 있는 성분'과 '그 시즌 성분'으로 쪼갠다 ──
    print(f"\n{'─'*80}")
    print("  ★ 파트 2b — β>1 은 **어느 성분**에 붙어 있나")
    print(f"{'─'*80}")
    print("  m_j (그 시즌 예측 평균) = 투수 정체 + 그 시즌 등판 상황 구성. 앞쪽만 미리 안다.")
    print("  h_j = 과거 시즌 예측만으로 만든 오프셋 → **이것이 배포에 쓸 수 있는 전부**다.\n")
    print(f"  {'fold':<7}{'corr(h,m)':>11}{'m~h 기울기':>12}"
          f"{'β(resid~m)':>12}{'t(CR)':>8}   │{'β(resid~h)':>12}{'t(CR)':>8}{'위약 t':>8}")
    for f in ACT_FOLDS:
        mv, mp = mmod & (s == f), mmod & (s < f)
        cf, yy, pp, zz = codes[mv], y[mv], pmod[mv], zraw[mv]
        n = np.bincount(cf, minlength=n_id).astype(float)
        P = np.bincount(cf, weights=pp, minlength=n_id)
        mj = np.where(n > 0, P / np.maximum(n, 1) - pp.mean(), 0.0)
        zp = logit(pmod[mp])
        n_p = np.bincount(codes[mp], minlength=n_id).astype(float)
        Z_p = np.bincount(codes[mp], weights=zp, minlength=n_id)
        h = np.where(n_p > 0, Z_p / np.maximum(n_p, 1) - zp.mean(), 0.0)
        # h 를 확률 스케일로 (비교 가능하게) — p(1-p)≈0.25 이므로 ×0.25
        hp = h * 0.25
        both = (n > 0) & (n_p > 0)
        r = yy - pp
        c_hm = float(np.corrcoef(hp[both], mj[both])[0, 1])
        sl_mh = float(np.polyfit(hp[both], mj[both], 1)[0])
        b_m, t_m = resid_t_cluster(r, mj[cf], cf, n_id)
        b_h, t_h = resid_t_cluster(r, hp[cf], cf, n_id)
        seen = np.where(n_p > 0)[0]
        hs = hp.copy(); hs[seen] = hp[rng.permutation(seen)]
        t_hs = resid_t_cluster(r, hs[cf], cf, n_id)[1]
        print(f"  {f:<7}{c_hm:>11.3f}{sl_mh:>12.3f}{1+b_m:>12.3f}{t_m:>8.2f}"
              f"   │{1+b_h:>12.3f}{t_h:>8.2f}{t_hs:>8.2f}")

    # ── 파트 2c — 행 자신의 as-of 컬럼은? (마지막 합법 채널) ─────────────
    print(f"\n{'─'*80}")
    print("  ★ 파트 2c — **행 자신의 `asof_pitcher_success_rate`** 방향 (합법·행 독립)")
    print(f"{'─'*80}")
    print("  h_j 는 과거 **시즌**까지만 본다. as-of 컬럼은 그 투구 직전까지 — 올해분이 들어 있다.")
    print("  §7.90의 b_j 는 '시즌 경계에서 자른' 거친 양이라 같은 벡터가 아니다. 다시 묻는다.")
    print("  u = n/(n+200)·(asof_success − 리그평균)   ← 배포 `_sm` 과 같은 수축\n")
    ar = df["asof_pitcher_success_rate"].to_numpy(dtype=np.float64)
    an = df["asof_pitcher_n"].to_numpy(dtype=np.float64)
    print(f"  {'fold':<7}{'결측':>8}{'std(u)':>9}{'β(resid~u)':>12}{'t(CR)':>8}{'위약 t':>8}"
          f"   │{'β(resid~m)':>12}{'u 통제 후 β(m)':>15}")
    for f in FOLDS:
        m = mmod & (s == f)
        cf, yy, pp = codes[m], y[m], pmod[m]
        av, nv = ar[m], np.nan_to_num(an[m])
        miss = ~np.isfinite(av)
        base_r = np.nanmean(av)
        u = np.where(miss, 0.0, (nv / (nv + 200.0)) * (np.nan_to_num(av) - base_r))
        n = np.bincount(cf, minlength=n_id).astype(float)
        P = np.bincount(cf, weights=pp, minlength=n_id)
        mj = np.where(n > 0, P / np.maximum(n, 1) - pp.mean(), 0.0)[cf]
        r = yy - pp
        b_u, t_u = resid_t_cluster(r, u, cf, n_id)
        # 위약 — 투수 정체를 뒤섞되 u 의 분포는 유지 (투수 단위 평균을 셔플)
        us = np.bincount(cf, weights=u, minlength=n_id) / np.maximum(
            np.bincount(cf, minlength=n_id), 1)
        seen = np.where(n > 0)[0]
        usp = us.copy(); usp[seen] = us[rng.permutation(seen)]
        t_us = resid_t_cluster(r, usp[cf], cf, n_id)[1]
        # u 를 통제하고도 m 의 기울기가 남는가 (2변수 OLS)
        A = np.column_stack([u - u.mean(), mj - mj.mean()])
        coef = np.linalg.lstsq(A, r - r.mean(), rcond=None)[0]
        print(f"  {f:<7}{miss.mean():>8.1%}{u.std():>9.4f}{1+b_u:>12.3f}{t_u:>8.2f}{t_us:>8.2f}"
              f"   │{1+resid_t_cluster(r, mj, cf, n_id)[0]:>12.3f}{1+coef[1]:>15.3f}")

    # ── 파트 3 — 행동 (과거 예측만으로 만든 오프셋을 γ배) ────────────────
    print(f"\n{'='*80}\n█ 파트 3 — 행동: z' = z + γ·h_j   (h_j = **과거 시즌 예측만**)\n{'='*80}")
    print("  h_j 는 라벨을 안 쓴다 — 과거 행의 배포 예측 로짓 평균에서 리그 평균을 뺀 값.")
    print("  test 에도 그대로 만들 수 있다(train 에서 fit, test 에서 조회). γ=0 이 배포다.\n")
    print(f"  {'fold':<7}{'커버율':>8}{'std(h)':>8}" + "".join(f"{f'γ={g:+.2f}':>10}" for g in GAMMAS)
          + f"{'위약 γ=+1':>11}")
    for f in ACT_FOLDS:
        mv, mp = mmod & (s == f), mmod & (s < f)
        cf, yy, zz = codes[mv], y[mv], zraw[mv]
        zp = logit(pmod[mp])
        n_p = np.bincount(codes[mp], minlength=n_id).astype(float)
        Z_p = np.bincount(codes[mp], weights=zp, minlength=n_id)
        h = np.where(n_p > 0, Z_p / np.maximum(n_p, 1) - zp.mean(), 0.0)
        seen = np.where(n_p > 0)[0]
        hs = h.copy(); hs[seen] = h[rng.permutation(seen)]
        cov = float((n_p[cf] > 0).mean())

        def run(hv, g):
            zn = zz + g * hv[cf]
            return bss(yy, shift_to_prior(1 / (1 + np.exp(-zn)), yy.mean()))

        base = run(h, 0.0)
        print(f"  {f:<7}{cov:>8.1%}{h[cf].std():>8.4f}"
              + "".join(f"{run(h,g)-base:>10.1f}" for g in GAMMAS)
              + f"{run(hs,1.0)-base:>11.1f}")
    print(f"  (숫자는 배포 대비 ΔBSS. 기준 BSS 는 fold별로 "
          + ", ".join(f"{f}={bss(y[mmod&(s==f)], pmod[mmod&(s==f)]):.1f}" for f in ACT_FOLDS) + ")")

    # 꼬리 게이팅 — "확실하게 좀 튀는 선수들에게만"
    print(f"\n  ── 꼬리 게이팅: |h_j| 상위 q 에게만 γ 적용 ──")
    print(f"  {'fold':<7}{'q':>6}{'대상투수':>9}{'행 비중':>9}"
          + "".join(f"{f'γ={g:+.2f}':>10}" for g in GAMMAS[2:]))
    for f in ACT_FOLDS:
        mv, mp = mmod & (s == f), mmod & (s < f)
        cf, yy, zz = codes[mv], y[mv], zraw[mv]
        zp = logit(pmod[mp])
        n_p = np.bincount(codes[mp], minlength=n_id).astype(float)
        Z_p = np.bincount(codes[mp], weights=zp, minlength=n_id)
        h = np.where(n_p > 0, Z_p / np.maximum(n_p, 1) - zp.mean(), 0.0)
        base = bss(yy, shift_to_prior(1 / (1 + np.exp(-zz)), yy.mean()))
        for q in (0.80, 0.90):
            seen = n_p > 0
            thr = np.quantile(np.abs(h[seen]), q)
            hg = np.where(np.abs(h) > thr, h, 0.0)
            row = []
            for g in GAMMAS[2:]:
                zn = zz + g * hg[cf]
                row.append(bss(yy, shift_to_prior(1 / (1 + np.exp(-zn)), yy.mean())) - base)
            print(f"  {f:<7}{q:>6.2f}{int((np.abs(h)>thr).sum()):>9}"
                  f"{float((hg[cf]!=0).mean()):>9.1%}" + "".join(f"{v:>10.1f}" for v in row))

    # ── 파트 3b — ★ 2c 방향을 실제 BSS 로 (배포 가능한 형태) ─────────────
    print(f"\n{'='*80}\n█ 파트 3b — ★ as-of 방향 증폭: z' = z + g·u   (합법·행 독립·배포 가능)\n{'='*80}")
    print("  u 는 행 자신의 컬럼만 쓴다 — test 다른 행을 안 본다. 리그평균은 train 에서 얼린다.")
    print("  확률 스케일 계수 (β−1) ≈ g·0.25 이므로 2c 의 0.06~0.09 는 g ≈ 0.25~0.35 다.\n")
    GS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.60, 1.00]
    ubank = {}
    print("  ⚠️ **온도 대조군이 필요하다.** z+g·u 는 정체 정보만 더하는 게 아니라 예측")
    print("     분산도 키운다. 파트 1에서 fold2023 은 c=1.10 을 원했다(+3.8) — 그 fold 는")
    print("     아무 방향으로나 분산만 키워도 오른다. 그래서 **같은 std 를 만드는 전역 c**")
    print("     팔을 짝지어 빼고, 그 차이만 '정체의 값어치'로 읽는다.\n")
    print(f"  {'fold':<7}{'기준':>8}" + "".join(f"{f'g={g:.2f}':>17}" for g in GS[1:5])
          + f"{'위약 g=.3':>10}{'최적 g':>7}")
    print(f"  {'':7}{'':8}" + "".join(f"{'u팔  (−온도)':>17}" for _ in GS[1:5]))
    for f in FOLDS:
        m = mmod & (s == f)
        cf, yy, zz = codes[m], y[m], zraw[m]
        av, nv = ar[m], np.nan_to_num(an[m])
        # ⚠️ 리그평균은 **train(<f) 에서 얼린다.** 검증 fold 평균을 쓰면 test-side
        #    통계다. (prior shift 가 상수를 먹어 결과는 거의 같지만 규정은 규정이다.)
        ref = float(np.nanmean(ar[mclean & (s < f)]))
        u = np.where(~np.isfinite(av), 0.0,
                     (nv / (nv + 200.0)) * (np.nan_to_num(av) - ref))
        ubank[f] = (yy, zz, u, cf)
        n = np.bincount(cf, minlength=n_id).astype(float)
        us = np.bincount(cf, weights=u, minlength=n_id) / np.maximum(n, 1)
        seen = np.where(n > 0)[0]
        usp = us.copy(); usp[seen] = us[rng.permutation(seen)]
        zc, sd0 = zz.mean(), zz.std()

        def sc(uu, g):
            return bss(yy, shift_to_prior(1 / (1 + np.exp(-(zz + g * uu))), yy.mean()))

        def temp(target_sd):
            """같은 std 를 만드는 전역 온도 팔 — '분산만 키운' 영가설."""
            c = target_sd / sd0
            return bss(yy, shift_to_prior(1 / (1 + np.exp(-(zc + c * (zz - zc)))), yy.mean()))

        b0 = sc(u, 0.0)
        cells = []
        for g in GS[1:5]:
            raw = sc(u, g) - b0
            net = raw - (temp((zz + g * u).std()) - b0)
            cells.append(f"{raw:>7.1f} ({net:+5.1f})")
        gs = np.linspace(0, 1.2, 61)
        net_curve = [sc(u, g) - temp((zz + g * u).std()) for g in gs]
        best = gs[int(np.argmax(net_curve))]
        print(f"  {f:<7}{b0:>8.1f}" + "".join(f"{c:>17}" for c in cells)
              + f"{sc(usp[cf],0.30)-b0:>10.1f}{best:>7.2f}")
    print("  괄호 = 같은 std 를 만드는 전역 온도 팔을 뺀 **정체 순이득**. 최적 g 도 순이득 기준.")

    print("\n  ── 정직한 선택: g 를 **과거 fold에서만** 고르고 미래 fold에서 평가 ──")
    for pick, ev in ((("2021", 2021), 2022), (("2021·2022", (2021, 2022)), 2023),
                     (("2021·2022·2023", (2021, 2022, 2023)), 2024)):
        lab, tr = pick
        trf = (tr,) if isinstance(tr, int) else tr
        def arm(f, g):
            yy, zz, u, _ = ubank[f]
            zc, sd0 = zz.mean(), zz.std()
            a = bss(yy, shift_to_prior(1 / (1 + np.exp(-(zz + g * u))), yy.mean()))
            c = (zz + g * u).std() / sd0
            t = bss(yy, shift_to_prior(1 / (1 + np.exp(-(zc + c * (zz - zc)))), yy.mean()))
            return a, t

        gs = np.linspace(0, 1.2, 61)
        # g 는 **순이득**(u팔 − 온도팔) 합으로 고른다. 온도는 g 가 아니라 c 로 살 것이다
        net_sum = [sum(a - t for a, t in (arm(f, g) for f in trf)) for g in gs]
        gsel = float(gs[int(np.argmax(net_sum))])
        b0 = arm(ev, 0.0)[0]
        b1, bt = arm(ev, gsel)
        print(f"  {lab:<16} → g={gsel:.2f}   평가 fold {ev}:  {b0:.1f} → {b1:.1f}"
              f"   Δ {b1-b0:+.1f}   (온도팔 {bt-b0:+.1f} → **순 {b1-bt:+.1f}**)")

    print("\n  ── 꼬리 게이팅: |u| 상위 q 행에만 적용 (사용자 요청 그대로) ──")
    print("     같은 온도 대조군을 뺀 **순이득**을 괄호로 (괄호 밖 raw 는 온도가 섞인 값).")
    print(f"  {'fold':<7}{'q':>6}{'행 비중':>9}" + "".join(f"{f'g={g:.2f}':>16}" for g in GS[1:5]))
    for f in FOLDS:
        yy, zz, u, cf = ubank[f]
        zc, sd0 = zz.mean(), zz.std()
        b0 = bss(yy, shift_to_prior(1 / (1 + np.exp(-zz)), yy.mean()))
        for q in (0.70, 0.90):
            thr = np.quantile(np.abs(u), q)
            ug = np.where(np.abs(u) > thr, u, 0.0)
            cells = []
            for g in GS[1:5]:
                a = bss(yy, shift_to_prior(1 / (1 + np.exp(-(zz + g * ug))), yy.mean())) - b0
                c = (zz + g * ug).std() / sd0
                t = bss(yy, shift_to_prior(1 / (1 + np.exp(-(zc + c * (zz - zc)))), yy.mean())) - b0
                cells.append(f"{a:>6.1f} ({a-t:+5.1f})")
            print(f"  {f:<7}{q:>6.2f}{float((ug!=0).mean()):>9.1%}"
                  + "".join(f"{c:>16}" for c in cells))

    # ── 파트 3c — ★ 정직한 판정: (q, g) 를 과거 fold에서만 고른다 ────────
    print(f"\n{'='*80}\n█ 파트 3c — ★ 정직한 판정: (q, g) 전방 선택 + 클러스터 부트스트랩\n{'='*80}")
    print("  §7.101 ②의 교훈 — 평가 fold 를 보고 노브를 고르면 이 문제에서 +2.0 이 부풀려진다.")
    print("  q(게이트 분위)와 g(강도)를 **과거 fold 합산 순이득**으로 고르고 미래 fold 에서만 평가한다.")
    print("  오차막대는 **투수 클러스터 부트스트랩** — 효과가 투수 단위라 행 부트스트랩은 과소추정한다.\n")
    QS = [0.0, 0.50, 0.70, 0.80, 0.90, 0.95]
    GG = np.linspace(0.05, 1.0, 20)

    def net(f, q, g, idx=None):
        yy, zz, u, cf = ubank[f]
        if idx is not None:
            yy, zz, u = yy[idx], zz[idx], u[idx]
        if q > 0:
            thr = np.quantile(np.abs(u), q)
            u = np.where(np.abs(u) > thr, u, 0.0)
        zc, sd0 = zz.mean(), zz.std()
        zn = zz + g * u
        a = bss(yy, shift_to_prior(1 / (1 + np.exp(-zn)), yy.mean()))
        c = zn.std() / sd0
        t = bss(yy, shift_to_prior(1 / (1 + np.exp(-(zc + c * (zz - zc)))), yy.mean()))
        return a - t

    print(f"  {'선택 fold':<16}{'→ (q, g)':>16}{'평가':>7}{'순이득':>9}"
          f"{'95% 클러스터 부트스트랩':>26}{'게이트 위약':>12}")
    for trf, ev in (((2021,), 2022), ((2021, 2022), 2023), ((2021, 2022, 2023), 2024)):
        best, bq, bg = -1e9, 0.0, 0.0
        for q in QS:
            for g in GG:
                v = sum(net(f, q, g) for f in trf)
                if v > best:
                    best, bq, bg = v, q, g
        val = net(ev, bq, bg)
        # 클러스터 부트스트랩 — 투수를 복원추출한다
        yy, zz, u, cf = ubank[ev]
        order = np.argsort(cf, kind="stable")
        bounds = np.searchsorted(cf[order], np.arange(n_id + 1))
        seen = np.where(np.diff(bounds) > 0)[0]
        bs = []
        for _ in range(200):
            pick = rng.choice(seen, size=len(seen), replace=True)
            idx = np.concatenate([order[bounds[j]:bounds[j + 1]] for j in pick])
            bs.append(net(ev, bq, bg, idx))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        # 게이트 위약 — 같은 q·g 인데 |u| 대신 투수 정체를 뒤섞은 u
        n_ = np.bincount(cf, minlength=n_id).astype(float)
        us = np.bincount(cf, weights=u, minlength=n_id) / np.maximum(n_, 1)
        sn = np.where(n_ > 0)[0]
        usp = us.copy(); usp[sn] = us[rng.permutation(sn)]
        keep = ubank[ev]
        ubank[ev] = (yy, zz, usp[cf], cf)
        pl = net(ev, bq, bg)
        ubank[ev] = keep
        print(f"  {'·'.join(map(str,trf)):<16}{f'q={bq:.2f}, g={bg:.2f}':>16}{ev:>7}"
              f"{val:>9.1f}{f'[{lo:+.1f}, {hi:+.1f}]':>26}{pl:>12.1f}")

    # ── 파트 4 — ★ 기전: β>1 은 버그인가 정보 부족인가 ────────────────────
    print(f"\n{'='*80}\n█ 파트 4 — ★ 기전 검정: β 는 시즌이 갈수록 1 로 수렴하는가\n{'='*80}")
    print("  m_j 는 정보량이 **서로 다른** 행들의 예측 평균이다 — 4월 행은 올해를 모르고")
    print("  9월 행은 안다. 정보가 다른 사후평균을 평균내면 합집합 기준 사후평균보다")
    print("  **과수축**된다 (타워 성질이 안 성립하는 지점). 그러면 β>1 은 버그가 아니라")
    print("  정보 부족의 흔적이고, 예측 시점에 고칠 수 없다.")
    print("  → 검정 가능한 예측: **β 는 시즌 초에 크고 시즌 말로 갈수록 1 에 가까워야 한다.**\n")
    gm = df["game_month"].to_numpy()
    BUCKETS = [(3, 4, "3~4월"), (5, 6, "5~6월"), (7, 8, "7~8월"), (9, 12, "9월~")]
    print(f"  {'fold':<7}" + "".join(f"{lab:>18}" for _, _, lab in BUCKETS))
    print(f"  {'':7}" + "".join(f"{'β   (t_CR) [n]':>18}" for _ in BUCKETS))
    for f in FOLDS:
        cells = []
        for lo, hi, _ in BUCKETS:
            m = mmod & (s == f) & (gm >= lo) & (gm <= hi)
            if m.sum() < 5000:
                cells.append(f"{'—':>18}")
                continue
            cf, yy, pp = codes[m], y[m], pmod[m]
            n = np.bincount(cf, minlength=n_id).astype(float)
            P = np.bincount(cf, weights=pp, minlength=n_id)
            mj = np.where(n > 0, P / np.maximum(n, 1) - pp.mean(), 0.0)
            sl, t = resid_t_cluster(yy - pp, mj[cf], cf, n_id)
            cells.append(f"{1+sl:>6.2f} ({t:>4.1f}) {int(m.sum()/1000):>3}k")
        print(f"  {f:<7}" + "".join(f"{c:>18}" for c in cells))
    print("\n  ⚠️ 버킷별 m_j 는 그 버킷 안에서 다시 계산한다 — 각 버킷의 정보 상태를 반영해야")
    print("     하고, 시즌 전체 m_j 를 쓰면 미래 행이 섞여 검정이 무의미해진다.")

    print(f"\n{'='*80}\n■ 읽는 법\n{'='*80}")
    print("  파트 2 β 가 fold 전부에서 >1 이고 위약 β≈1 이면 → 사용자 요청이 맞다.")
    print("     3분위 열에서 상위(꼬리)의 β 가 더 크면 → '튀는 선수에게만' 이 정확한 처방.")
    print("  파트 3 이 실제 배포 가능한 형태의 측정이다. γ>0 이 fold 전부에서 양수여야")
    print("     채택 후보다 (§2 조건 ① 부호 일치 ② 크기비 ③ 위약 대비).")


if __name__ == "__main__":
    main()
