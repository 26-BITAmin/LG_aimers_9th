"""투수/타자 개인차의 **극단**이 존재하는가 (사용자 질문).

    "전체 선수들(F에서 2023 이전은 제외) 중에서 유독 못하는 선수와 잘하는 선수가
     존재하나"

§7.90(`ebshrink.py`)은 이미 투수 절편 `b_j`를 배포 잔차에 물었고 |t|<1.8로 죽었다.
**그건 선형 검정이다.** `resid ~ b_j` OLS 기울기는 "전체 투수가 비례해서 어긋나는가"를
묻는다. 전체의 2%만 어긋나 있으면 나머지 98%가 기울기를 0으로 끌어내린다.
"유독"은 기울기가 아니라 **꼬리**의 질문이라 같은 자리가 아니다.

영가설을 셋으로 나눈다 — 셋의 답이 전부 다를 수 있다.

  H0-노이즈   개인차가 아예 없다 (모두 리그 평균).  → 위약: 시즌 내 ID 셔플
  H0-매끄러움 개인차는 있지만 **정규 랜덤효과**다 (연속 스펙트럼, 특이점 없음).
              → 영가설: σ̂²_b 적합 후 **시뮬레이션**. 이게 "유독"의 진짜 영가설이다.
  H0-소진     극단은 실재하지만 **배포 모델이 이미 안다**.
              → 기준선을 시즌 상수가 아니라 배포 예측 p̂ 로 바꿔 같은 통계를 다시.

파트
  A  존재·형태  : 시즌 상수 대비 z_j, σ̂²_b 적률 추정, 매끄러움 시뮬레이션 대비 꼬리
  B  지속성      : 시즌 t의 편차가 t+1에 남는가 (안 남으면 "유독한 놈"이 아니라 그 해의 운)
  C  모델 대비   : foldcache 배포 예측 p̂ 기준으로 A·B를 반복 → 남은 헤드룸
  D  명단        : 상·하위 선수의 n / 실측률 / 수축 편차 / 모델 대비 z

데이터: train.csv 에서 **(game_type=='F') & (season<2023) 행 제외** (§7.101 오염 구간).
        파트 C 는 fold 캐시가 있는 2021~2024 검증 시즌만 (전부 홀드아웃 예측).

⚠️ z_j 는 표본수에 비례해 커진다. n=5,000 투수는 참편차 .01 로도 |z|=1.6 이 나온다.
   그래서 z(유의성)와 수축 편차 b_j(크기)를 **항상 같이** 찍는다. 하나만 보면 틀린다.

실행:  .venv/Scripts/python.exe -u src/pextreme.py 2>&1 | tee experiments/pextreme.log
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
from config import ASOF_AXES, CTX_AXES, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = [2021, 2022, 2023, 2024]
MIN_N = 300          # 꼬리 분석 대상 (n<300 은 z 의 정규근사가 흔들린다)
MIN_N_SEQ = 200      # 지속성: 두 시즌 모두 이 이상
N_SIM = 400          # 매끄러움 영가설 시뮬레이션 반복
SEED = 42


# ── 집계 ────────────────────────────────────────────────────────────────
def agg(idx, n_id, y, p):
    """선수별 (n, k, E, V). E=Σp, V=Σp(1-p) — 포아송이항의 평균·분산."""
    n = np.bincount(idx, minlength=n_id).astype(np.float64)
    k = np.bincount(idx, weights=y, minlength=n_id)
    E = np.bincount(idx, weights=p, minlength=n_id)
    V = np.bincount(idx, weights=p * (1.0 - p), minlength=n_id)
    return n, k, E, V


def zstat(n, k, E, V):
    return (k - E) / np.sqrt(np.maximum(V, 1e-12))


def sigma2_b(n, k, E, V, m):
    """적률법. Var_j(d_j) = σ²_b + E[V_j/n_j²]  →  뺄셈 없이 쓰면 과대다 (§7.90 ①)."""
    d = k[m] / n[m] - E[m] / n[m]
    naive = float(d.var())
    noise = float(np.mean(V[m] / n[m] ** 2))
    return max(naive - noise, 0.0), naive, noise


def tails(z, cuts=(2.0, 3.0, 4.0, 5.0)):
    return [int((np.abs(z) > c).sum()) for c in cuts]


def sigma_b_robust(n, E, V, obs_iqr, rng, n_sim=40):
    """**강건** σ_b — 관측 d_j 의 IQR 을 재현하는 σ 를 이분법으로 찾는다.

    ⚠️ 적률 σ̂_b 는 분산 기반이라 **이상치가 자기 영가설을 부풀린다.** 소수의 특이
       개체가 σ_b 를 키우고, 커진 σ_b 가 만든 넓은 꼬리가 자기 자신을 정상으로
       보이게 한다. IQR 은 중앙 50%만 보므로 그 되먹임이 끊긴다.
    """
    def iqr_of(sigma):
        v = []
        for _ in range(n_sim):
            b = rng.normal(0.0, sigma, size=len(n))
            var = np.maximum(V + b * (n - 2 * E) - n * b * b, 1e-9)
            d = (rng.normal(E + n * b, np.sqrt(var)) - E) / n
            v.append(np.subtract(*np.percentile(d, [75, 25])))
        return float(np.mean(v))

    lo, hi = 0.0, 0.20
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if iqr_of(mid) < obs_iqr:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sim_smooth(n, E, V, sb, rng, n_sim=N_SIM):
    """H0-매끄러움: b_j ~ N(0, sb²) 인 **연속** 개인차만으로 같은 꼬리가 나오는가.

    k_j | b_j 의 정확한 분산은 Σ(p_i+b)(1-p_i-b) = V + b(n-2E) - n b² 다.
    (Σp_i²=E-V 를 쓰면 바로 나온다. 이항으로 근사하면 within 문맥 분산만큼 과대다.)
    """
    out_t, out_mx, out_sd = [], [], []
    for _ in range(n_sim):
        b = rng.normal(0.0, sb, size=len(n))
        mu = E + n * b
        var = np.maximum(V + b * (n - 2 * E) - n * b * b, 1e-9)
        ks = rng.normal(mu, np.sqrt(var))
        z = zstat(n, ks, E, V)
        out_t.append(tails(z))
        out_mx.append(float(np.abs(z).max()))
        out_sd.append(float(z.std()))
    return np.array(out_t, float), np.array(out_mx), np.array(out_sd)


def block(title, name, idx, n_id, y, p, base_lab, rng, shuf_idx=None):
    """A/C 공용 — 기준선 p 를 갈아 끼우며 같은 통계를 낸다."""
    n, k, E, V = agg(idx, n_id, y, p)
    m = n >= MIN_N
    z = zstat(n, k, E, V)
    sb, naive, noise = sigma2_b(n, k, E, V, m)
    sb = np.sqrt(sb)

    zm = z[m]
    obs_t = tails(zm)
    st, smx, ssd = sim_smooth(n[m], E[m], V[m], sb, rng)
    # 강건 팔 — 이상치가 자기 영가설을 부풀리는 되먹임을 끊는다
    d_obs = k[m] / n[m] - E[m] / n[m]
    sb_r = sigma_b_robust(n[m], E[m], V[m],
                          float(np.subtract(*np.percentile(d_obs, [75, 25]))), rng)
    rt, rmx, rsd = sim_smooth(n[m], E[m], V[m], sb_r, rng, n_sim=200)
    # 위약 — 시즌 내 ID 셔플 (H0-노이즈)
    zp = None
    if shuf_idx is not None:
        np_, kp, Ep, Vp = agg(shuf_idx, n_id, y, p)
        mp = np_ >= MIN_N
        zp = zstat(np_, kp, Ep, Vp)[mp]

    print(f"\n{'─'*78}")
    print(f"■ {title}   기준선 = {base_lab}")
    print(f"{'─'*78}")
    print(f"  {name} {int(m.sum()):,}명 (n≥{MIN_N})  총 {n_id:,}명  행 {len(y):,}")
    print(f"  σ_b(적률) = {sb:.4f}   [순진 {np.sqrt(naive):.4f}  노이즈분 {np.sqrt(noise):.4f}]"
          f"   →  ±1σ 구간 폭 {2*sb*100:.1f}%p")
    print(f"  σ_b(강건·IQR) = {sb_r:.4f}   비 {sb_r/max(sb,1e-9):.3f}"
          f"   [1보다 많이 작으면 꼬리가 적률 추정을 부풀린 것]")
    print(f"  z 의 sd :  관측 {zm.std():>6.2f}   매끄러움 영가설 {ssd.mean():>5.2f}"
          f"   위약(ID셔플) {zp.std() if zp is not None else float('nan'):>5.2f}   [노이즈만이면 1.00]")
    print(f"  max|z|  :  관측 {np.abs(zm).max():>6.2f}   매끄러움 영가설 {smx.mean():>5.2f}"
          f" (95%p {np.percentile(smx,95):.2f})")
    print(f"\n  {'꼬리':<10}{'관측':>8}{'매끄러움(적률σ)':>18}{'p':>7}"
          f"{'매끄러움(강건σ)':>18}{'p':>7}{'위약':>7}")
    for i, c in enumerate((2.0, 3.0, 4.0, 5.0)):
        pv = float((st[:, i] >= obs_t[i]).mean())
        pr = float((rt[:, i] >= obs_t[i]).mean())
        pl = int((np.abs(zp) > c).sum()) if zp is not None else -1
        print(f"  |z|>{c:<6.0f}{obs_t[i]:>8d}{st[:,i].mean():>12.1f} ±{st[:,i].std():>4.1f}"
              f"{pv:>7.3f}{rt[:,i].mean():>12.1f} ±{rt[:,i].std():>4.1f}{pr:>7.3f}{pl:>7d}")
    return n, k, E, V, m, z, sb


# ── 파트 B: 지속성 ───────────────────────────────────────────────────────
def persistence(df, mclean, idcol, y, p_base, lab, rng, pcol=None):
    """시즌 t 편차 → t+1 편차. 상관 + '참편차가 완전히 안정일 때의 상한'."""
    s = df["season"].to_numpy()
    codes, uniq = pd.factorize(df[idcol].to_numpy())
    n_id = len(uniq)
    seasons = sorted(set(s[mclean].tolist()))
    per = {}
    for t in seasons:
        mt = mclean & (s == t)
        if mt.sum() == 0:
            continue
        pv = p_base[mt] if pcol is None else pcol[mt]
        n, k, E, V = agg(codes[mt], n_id, y[mt], pv)
        per[t] = (n, k, E, V)

    print(f"\n{'─'*78}")
    print(f"■ 지속성 — {lab}")
    print(f"{'─'*78}")
    print(f"  {'t → t+1':<12}{'쌍':>6}{'r(d_t, d_t+1)':>16}{'안정 상한 r*':>14}"
          f"{'r/r*':>8}{'상위10%→':>11}{'하위10%→':>11}")
    rows = []
    for a, b in zip(seasons, seasons[1:]):
        if a not in per or b not in per:
            continue
        na, ka, Ea, Va = per[a]
        nb, kb, Eb, Vb = per[b]
        m = (na >= MIN_N_SEQ) & (nb >= MIN_N_SEQ)
        if m.sum() < 30:
            continue
        da = ka[m] / na[m] - Ea[m] / na[m]
        db = kb[m] / nb[m] - Eb[m] / nb[m]
        r = float(np.corrcoef(da, db)[0, 1])
        # r* = 참편차가 시즌 간 완전히 같을 때 관측될 상관 (감쇠 보정의 역)
        sb2 = max(float(np.var(da)) - float(np.mean(Va[m] / na[m] ** 2)), 0.0)
        rel_a = sb2 / max(float(np.var(da)), 1e-12)
        rel_b = sb2 / max(float(np.var(db)), 1e-12)
        rstar = float(np.sqrt(max(rel_a * rel_b, 0.0)))
        q = np.quantile(da, [0.10, 0.90])
        top = float(db[da >= q[1]].mean()) * 100
        bot = float(db[da <= q[0]].mean()) * 100
        print(f"  {a}→{b:<7}{int(m.sum()):>6}{r:>16.3f}{rstar:>14.3f}"
              f"{r/rstar if rstar>0 else float('nan'):>8.2f}"
              f"{top:>10.2f}%p{bot:>10.2f}%p")
        rows.append((r, rstar))
    if rows:
        R = np.array(rows)
        print(f"  {'평균':<12}{'':>6}{R[:,0].mean():>16.3f}{R[:,1].mean():>14.3f}"
              f"{(R[:,0]/np.maximum(R[:,1],1e-9)).mean():>8.2f}")
    return per


def main():
    rng = np.random.default_rng(SEED)
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    mclean = ~((gt == "F") & (s < 2023))

    print(f"train {df.shape}   clean(F<2023 제외) {int(mclean.sum()):,}행"
          f"  ({mclean.mean():.1%})   양성률 {y[mclean].mean():.4f}")
    print(f"  제외된 F 행 {int((~mclean).sum()):,}   시즌별 clean 양성률:")
    for t in sorted(set(s.tolist())):
        mt = mclean & (s == t)
        print(f"    {t}  n={int(mt.sum()):>9,}  ȳ={y[mt].mean():.4f}")

    # 시즌 상수 기준선 (시즌 × game_type 평균) — 문맥이 아니라 '레벨'만 뺀 것
    base = np.zeros(len(df))
    key = pd.Series(s).astype(str) + "|" + pd.Series(gt).astype(str)
    for kk, g in pd.DataFrame({"k": key, "y": y, "m": mclean}).groupby("k"):
        mm = g["m"].to_numpy()
        if mm.sum() == 0:
            continue
        base[g.index.to_numpy()] = g["y"].to_numpy()[mm].mean()

    print(f"\n{'='*78}\n█ 파트 A — 존재하는가 · 스펙트럼인가 극단인가 (2019~2024 clean)\n{'='*78}")
    out = {}
    for idcol, lab in (("pitcher_id", "투수"), ("batter_id", "타자")):
        codes, uniq = pd.factorize(df[idcol].to_numpy())
        ci, ni = codes[mclean], len(uniq)
        # 위약: 시즌 안에서 ID 셔플 (시즌 구성은 유지, 정체만 지운다)
        sh = ci.copy()
        ss = s[mclean]
        for t in np.unique(ss):
            w = np.where(ss == t)[0]
            sh[w] = ci[rng.permutation(w)]
        out[idcol] = block(f"{lab} 개인차", lab, ci, ni, y[mclean], base[mclean],
                           "시즌×game_type 평균 (레벨만 제거)", rng, shuf_idx=sh)

    print(f"\n{'='*78}\n█ 파트 B — 그 편차가 다음 시즌에도 남는가\n{'='*78}")
    for idcol, lab in (("pitcher_id", "투수"), ("batter_id", "타자")):
        persistence(df, mclean, idcol, y, base, f"{lab} (시즌 상수 기준)", rng)

    # ── 파트 C — 배포 모델 예측을 기준선으로 ────────────────────────────
    print(f"\n{'='*78}\n█ 파트 C — 배포 모델이 이미 아는가 (홀드아웃 z_lgb/z_mlp + δ축)\n{'='*78}")
    cached = J.foldcache.load(FOLDS, verbose=False)
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    pmod = np.full(len(df), np.nan)
    pblend = np.full(len(df), np.nan)
    for f in FOLDS:
        X, names, _, va = J.axis_matrix(df, s, f)
        yv = cached[f][2]
        vi = np.where(s == f)[0]
        assert len(vi) == len(yv) and np.allclose(y[vi], yv), f"fold {f} 정렬 실패"
        mm = mclean[vi]
        zl, zm_ = cached[f][0], cached[f][1]
        zb = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm_
        zd = zb + X @ dep
        # prior shift 는 **분석 부분집합**의 평균으로 — 레벨 편향을 지우고 개인차만 남긴다
        pb = np.full(len(vi), np.nan)
        pd_ = np.full(len(vi), np.nan)
        pb[mm] = shift_to_prior(1 / (1 + np.exp(-zb[mm])), yv[mm].mean())
        pd_[mm] = shift_to_prior(1 / (1 + np.exp(-zd[mm])), yv[mm].mean())
        pblend[vi], pmod[vi] = pb, pd_
        print(f"  fold {f}: clean n={int(mm.sum()):,}  BSS blend {bss(yv[mm], pb[mm]):.1f}"
              f"  deploy {bss(yv[mm], pd_[mm]):.1f}")

    mmod = mclean & np.isfinite(pmod)
    print(f"\n  파트 C 대상 = 2021~2024 clean {int(mmod.sum()):,}행")
    for idcol, lab in (("pitcher_id", "투수"), ("batter_id", "타자")):
        codes, uniq = pd.factorize(df[idcol].to_numpy())
        ci, ni = codes[mmod], len(uniq)
        sh = ci.copy()
        ss = s[mmod]
        for t in np.unique(ss):
            w = np.where(ss == t)[0]
            sh[w] = ci[rng.permutation(w)]
        block(f"{lab} — 시즌 상수 대비 (동일 부분집합 대조군)", lab, ci, ni,
              y[mmod], base[mmod], "시즌×game_type 평균", rng, shuf_idx=sh)
        block(f"{lab} — **배포 모델** 대비", lab, ci, ni, y[mmod], pmod[mmod],
              "배포 홀드아웃 예측 p̂ (blend+δ, prior shift)", rng, shuf_idx=sh)

    print(f"\n{'='*78}\n█ 파트 B' — 모델 잔차의 지속성 (남은 헤드룸의 실체)\n{'='*78}")
    for idcol, lab in (("pitcher_id", "투수"), ("batter_id", "타자")):
        persistence(df, mmod, idcol, y, base, f"{lab} (배포 모델 기준)", rng, pcol=pmod)

    print(f"\n{'='*78}\n█ 파트 F — **꼬리 집단만** 따로 본 지속성\n{'='*78}")
    print("  파트 B 의 상관은 전체 평균이라 다수에 희석된다. '유독한 놈'이 형질이라면")
    print("  그 집단에서 직접 재야 한다. t 시즌 |편차| 최상위 10명 → t+1 시즌 실적.\n")
    for idcol, lab, pc, bl in (("pitcher_id", "투수", None, "시즌상수"),
                               ("pitcher_id", "투수", pmod, "배포모델"),
                               ("batter_id", "타자", None, "시즌상수"),
                               ("batter_id", "타자", pmod, "배포모델")):
        codes, uniq = pd.factorize(df[idcol].to_numpy())
        mset = mmod if pc is not None else mclean
        pv_all = pmod if pc is not None else base
        seasons = sorted(set(s[mset].tolist()))
        line = []
        for a, b in zip(seasons, seasons[1:]):
            ma, mb = mset & (s == a), mset & (s == b)
            na, ka, Ea, Va = agg(codes[ma], len(uniq), y[ma], pv_all[ma])
            nb, kb, Eb, Vb = agg(codes[mb], len(uniq), y[mb], pv_all[mb])
            ok = (na >= MIN_N_SEQ) & (nb >= MIN_N_SEQ)
            if ok.sum() < 30:
                continue
            da = np.where(na > 0, (ka - Ea) / np.maximum(na, 1), 0.0)
            db = np.where(nb > 0, (kb - Eb) / np.maximum(nb, 1), 0.0)
            idx = np.where(ok)[0]
            top = idx[np.argsort(-np.abs(da[idx]))[:10]]
            # 유지율 = 다음 시즌 편차 / 이번 시즌 편차 (부호 살려서)
            # ⚠️ 부호 살린 평균은 상하위가 서로 상쇄한다. **방향 정렬**해서 유지량을 잰다
            sg = np.sign(da[top])
            line.append((a, b, np.abs(da[top]).mean() * 100,
                         float((sg * db[top]).mean()) * 100,
                         float(np.mean(sg == np.sign(db[top])))))
        print(f"  {lab}·{bl:<8}" + "".join(
            f"{a}→{str(b)[2:]}: {x:+.2f}→{yy:+.2f}%p(부호{sg:.0%})  " for a, b, x, yy, sg in line))

    # ── 파트 E — 얼마짜리인가 (오라클 천장 vs 정직한 전이) ────────────────
    print(f"\n{'='*78}\n█ 파트 E — 남은 개인차의 값어치 (BSS, clean 행)\n{'='*78}")
    print("  오라클 = **그 시즌 라벨로** 투수 절편을 추정해 되먹임 (전이 불가, 천장)")
    print("  전이   = 과거 시즌(<f)에서만 추정해 적용 (실제로 배포 가능한 형태)")
    print("  위약   = 전이와 같은 값을 **투수끼리 뒤섞어** 적용 (정체만 파괴)\n")
    codes, uniq = pd.factorize(df["pitcher_id"].to_numpy())
    n_id = len(uniq)
    print("  세 팔 모두 **검증 행 기준 중심화** — 레벨 성분을 빼고 정체 채널만 잰다\n")
    print(f"  {'fold':<8}{'기준 BSS':>11}{'오라클 Δ':>12}{'λ_o':>7}"
          f"{'전이 Δ':>11}{'λ_t':>7}{'위약 Δ':>11}{'전이(중심화X)':>13}{'평균b %p':>11}")

    def shrunk(idx, n_all, yy, pp):
        n, k, E, V = agg(idx, n_id, yy, pp)
        mm = n >= MIN_N
        s2 = sigma2_b(n, k, E, V, mm)[0]
        lam = float(np.mean(V[mm] / n[mm])) / max(s2, 1e-12)
        d = np.where(n > 0, (k - E) / np.maximum(n, 1), 0.0)
        return (n / (n + lam)) * d, lam, n

    for f in (2022, 2023, 2024):
        mv, mp = mmod & (s == f), mmod & (s < f)
        iv, yv, pv = codes[mv], y[mv], pmod[mv]
        base = bss(yv, pv)
        b_o, lam_o, _ = shrunk(iv, n_id, yv, pv)
        b_t, lam_t, n_t = shrunk(codes[mp], n_id, y[mp], pmod[mp])
        seen = np.where(n_t > 0)[0]
        b_s = b_t.copy()
        b_s[seen] = b_t[rng.permutation(seen)]

        # ⚠️ **검증 행 기준으로 중심화한다.** 안 하면 b 의 평균이 그대로 레벨 이동이
        #    되고, 이 지표에서 레벨은 BSS ≈ −400,700·(Δ평균)² 으로 갚아야 한다.
        #    0.8%p 만 밀려도 −25 다 — 전이 팔의 손실이 '정체가 무정보'인지 '레벨이
        #    밀렸는지' 구분이 안 된다. 중심화하면 남는 것은 **정체 채널뿐**이다.
        def apply(b):
            v = b[iv]
            return bss(yv, np.clip(pv + (v - v.mean()), 1e-6, 1 - 1e-6)) - base

        # 중심화 전/후를 같이 찍어 레벨 성분의 크기를 드러낸다
        raw = bss(yv, np.clip(pv + b_t[iv], 1e-6, 1 - 1e-6)) - base
        g = [apply(b) for b in (b_o, b_t, b_s)]
        print(f"  {f:<8}{base:>11.1f}{g[0]:>12.1f}{lam_o:>7.0f}"
              f"{g[1]:>11.1f}{lam_t:>7.0f}{g[2]:>11.1f}{raw:>13.1f}"
              f"{b_t[iv].mean()*100:>11.2f}")

    # ── 파트 D — 명단 ───────────────────────────────────────────────────
    print(f"\n{'='*78}\n█ 파트 D — 실제로 누구인가 (투수, 2019~2024 clean)\n{'='*78}")
    codes, uniq = pd.factorize(df["pitcher_id"].to_numpy())
    ci = codes[mclean]
    n, k, E, V, m, z, sb = out["pitcher_id"]
    lam = (float(np.mean(V[m] / n[m])) / max(sb ** 2, 1e-12))
    d = np.where(n > 0, k / np.maximum(n, 1) - E / np.maximum(n, 1), 0.0)
    b_shrunk = (n / (n + lam)) * d
    # 모델 대비 z (2021~2024 부분집합)
    cm = codes[mmod]
    nm, km, Em, Vm = agg(cm, len(uniq), y[mmod], pmod[mmod])
    zmod = zstat(nm, km, Em, Vm)
    dmod = np.where(nm > 0, km / np.maximum(nm, 1) - Em / np.maximum(nm, 1), 0.0)

    order = np.argsort(b_shrunk)
    order = [i for i in order if m[i]]
    print(f"  λ_EB(수축) = {lam:.1f}      (배포 SMOOTH_K_PITCHER = 200)")
    print(f"\n  {'pitcher_id':>11}{'n':>8}{'실측률':>9}{'기대률':>9}{'편차%p':>9}"
          f"{'수축편차':>10}{'z(상수)':>9}   │{'n(21-24)':>10}{'모델편차%p':>12}{'z(모델)':>9}")
    for tag, rows in (("── 유독 못하는 쪽 (하위 12) ──", order[:12]),
                      ("── 유독 잘하는 쪽 (상위 12) ──", order[-12:][::-1])):
        print(f"  {tag}")
        for i in rows:
            print(f"  {uniq[i]:>11}{int(n[i]):>8,}{k[i]/n[i]:>9.4f}{E[i]/n[i]:>9.4f}"
                  f"{(k[i]/n[i]-E[i]/n[i])*100:>9.2f}{b_shrunk[i]*100:>10.2f}{z[i]:>9.2f}"
                  f"   │{int(nm[i]):>10,}{dmod[i]*100:>12.2f}{zmod[i]:>9.2f}")

    print(f"\n{'='*78}\n■ 읽는 법\n{'='*78}")
    print("  파트 A '매끄러움 영가설' 열 = 개인차가 **정규 연속 분포**일 때 나올 꼬리 개수.")
    print("     관측 ≈ 영가설  →  '유독한 선수'는 없다. 연속 스펙트럼의 양끝일 뿐이다.")
    print("     관측 ≫ 영가설  →  분포로 설명 안 되는 특이 개체가 실재한다.")
    print("  파트 B r/r* = 관측 상관 / 참편차가 완전 안정일 때의 상한. 1에 가까우면")
    print("     '그 시즌의 운'이 아니라 **형질**이다. 0에 가까우면 다음 시즌엔 사라진다.")
    print("  파트 C 배포 기준선에서 꼬리가 사라지면 → 모델이 이미 안다 (헤드룸 없음).")
    print("     남으면 → §7.90이 선형 검정이라 놓친 것이 있다는 뜻이다.")


if __name__ == "__main__":
    main()
