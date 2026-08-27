"""§7.189 — 참고문헌 직역 모델 XPC (xCTRL × Location+ × Marcel) — 독립 계층 확률 모델

사용자: 참고 링크 4개(Wharton xCTRL · Fangraphs Stuff/Location+ 프라이머 ·
THT 프로젝션 평가 · arXiv 2508.19184 "Separating Intent from Execution")를 주고
"현 파이프라인과 완전 달라도 되니 모델을 만들어봐".

문헌 → 이 데이터로의 번역
    xCTRL/arXiv  위치 공간의 투수별 GMM (의도 타깃 추정 + 실행 분산)
                 → 위치 컬럼이 없다. 대신 타깃 자체가 실패 3모드의 합집합이고
                   §7.40이 투구별 모드 라벨을 100% 복원했다. **실패 채널 공간**
                   (성공/몰림만/반대만/몰림+반대/벗어남 5클래스)에서 투수별
                   프로파일을 만들고, GMM 으로 아키타입을 찾아 **군집 평균으로
                   부분 풀링**한다 (리그 0 으로의 수축이 아니라 — §7.37 count12
                   기각 사유였던 "셀당 표본 부족 → 과전이"의 xCTRL식 처방).
    Location+    카운트×손잡이별 기대 위치 → count12×플래툰 24셀의 리그 채널
                 분포표 = "의도 지도". §7.40 발견(카운트는 성공률이 아니라
                 실패의 구성을 바꾼다: 벗어남 4.6배)이 정확히 이 표다.
                 투수 효과는 리그 전체가 아니라 **자기 셀 분포 기준선 대비**로
                 잰다 (xCTRL 의 "자기 기준(individual tendencies)" 철학).
    Marcel/THT   recency 가중(hl) + EB 평균회귀(S) + 시스템 평균(마지막에
                 배포 블렌드와의 α 혼합 측정).

모델 (전 파라미터가 train 에서 얼려지는 표 — 행 독립, 규정 안전)
    score_k(row) = log π_k(cell) + u_{p,k} + v_{b,k},   p = softmax → P(성공)
      cell   = count12 × (투수손=타자손)  (24셀, recency 가중 Laplace 평활)
      u_{p,k} = log( EB[(n_p·f_p + S·prior_p) / (n_p+S)] / E_p )
                E_p = 투수 자신의 셀 분포로 기대되는 리그 채널 분포 (의도 보정)
                prior_p = E_p (기본) 또는 GMM 군집 사후가중 프로파일 (A4)
      v_{b,k} = 타자 동형 (S_B 더 강하게)
    복원 라벨은 **표를 적합하는 학습측 타깃**으로만 쓴다 (배포 aux 헤드와 같은
    지위 — 추론엔 필요 없고, test 행 독립성 유지).
    pre-2023 F 행은 표 적합에서 제외한다 (§7.48 라벨 오염 — soften 대신 단순 제외.
    이 모델의 표는 셀 단위라 §7.61 의 "오염이 양쪽 항에 공통이면 상쇄" 조건이
    안 성립하는 항(E_p)이 있어서, 넣는 쪽이 더 위험하다).

사다리 (무엇이 점수를 내는가 — 문헌 주장별 분리)
    A0  셀 표만 (Location+ 리그 의도 지도)
    A1  + 투수·타자 효과, 성공/실패 2클래스 (Marcel 이진 — 분해 없음)
    A2  A1 을 5클래스로 (의도-실행 분해의 순수 증분)      ⚠️ 투수만
    A3  A2 + 타자 효과 (주 모델)
    A4  A3 + GMM 아키타입 풀링 (xCTRL 증분)
    PLB A3 에서 투수·타자 표를 엔티티 간 순열 (셀 항 보존 — 정체만 지운 위약,
        [[placebo-must-keep-shared-terms]])

측정
    ① 단독 BSS oracle (shift_to_prior(ȳ_valid)) — fold2023 / fold2024
    ② hl × S_P 소형 스윕 (A3, 정보용 — 기본값 hl=1.0 · S_P=300 은 CTX_HALF_LIFE ·
       수축 전례에서 미리 고정)
    ③ 배포 등가층(resprobe.deploy_base) 위 corr · α 블렌드 곡선 · 쌍체 부트스트랩
    ④ §7.185 예측식 — LB 기대 = 상관이득 × c − 분산비용 × 1, c ∈ {0.16, 0.93}.
       이 모델의 정체성은 커리어 레벨형(c≈0.16 가족)이므로 **기대는 낮다.
       목표는 채택이 아니라 문헌 번역의 정직한 측정이다.**

실행:  .venv/Scripts/python.exe -u src/xpc.py 2>&1 | tee experiments/xpc.log
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache as _fc
import modelabel
from config import TARGET_COL, TRAIN_CSV
from data import load_csv, recency_weight
from validation import bss, shift_to_prior

EPS = 1e-6
LAPL = 25.0                  # 셀 표 Laplace 평활 (셀당 유효표본 수만 행 — 사실상 무감)
HL = 1.0                     # 기본 recency 반감기 — CTX_HALF_LIFE 전례
S_P = 300.0                  # 투수 EB 강도 (유효 투구 수 단위)
S_B = 1000.0                 # 타자 EB 강도
GMM_NMIN = 800.0             # GMM 적합에 쓰는 투수의 최소 유효 표본
GMM_KMAX = 5
ALPHAS = (0.02, 0.05, 0.10, 0.15)
ALPHA_GATE = 0.05
B_BOOT = 2000
SWEEP_HL = (0.5, 1.0, 2.0)
SWEEP_SP = (100.0, 300.0, 800.0)
CLS5 = ("성공", "몰림만", "반대만", "몰림+반대", "벗어남")


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def make_classes(tr):
    """fold 학습행의 5클래스 라벨. (cls int64, ok bool) — ok 아닌 행은 표에서 제외."""
    lab, _, _ = modelabel.recover(tr)
    y = tr[TARGET_COL].to_numpy(dtype=np.float64)
    suc, mid, rev = (np.round(lab[k]) for k in ("success", "middle", "reverse"))
    ok = ~(np.isnan(suc) | np.isnan(mid) | np.isnan(rev))
    agree = float((suc[ok] == y[ok]).mean())
    assert agree > 0.999, f"복원 success 와 y 불일치 {agree:.4%}"
    cls = np.full(len(tr), 4, dtype=np.int64)          # 벗어남 (실패, 몰림·반대 아님)
    cls[y == 1] = 0
    m1, r1 = mid == 1, rev == 1
    cls[(y == 0) & m1 & ~r1] = 1
    cls[(y == 0) & ~m1 & r1] = 2
    cls[(y == 0) & m1 & r1] = 3
    return cls, ok


def cell_of(d):
    """count12 × 플래툰 → 0..23."""
    b = np.clip(d["balls_before"].to_numpy(dtype=np.int64), 0, 3)
    s = np.clip(d["strikes_before"].to_numpy(dtype=np.int64), 0, 2)
    same = (d["pitcher_hand"].to_numpy() == d["batter_hand"].to_numpy()).astype(np.int64)
    return (b * 3 + s) * 2 + same


def fit_xpc(tr, cls, ok, w, n_cls, s_p, s_b, use_batter=True, use_pitcher=True,
            use_gmm=False, rng=None, permute=False):
    """모든 표를 적합해 dict 로 돌려준다. 전 상수가 fold 학습 시즌에서만 나온다."""
    m = ok.copy()
    gt = tr["game_type"].to_numpy()
    sn = tr["season"].to_numpy()
    m &= ~((gt == "F") & (sn < 2023))                  # §7.48 오염 구간 제외
    d, c, ww = tr[m], cls[m], w[m]
    if n_cls == 2:
        c = (c > 0).astype(np.int64)                   # 성공 0 / 실패 1
    cell = cell_of(d)

    cnt = np.zeros((24, n_cls))
    np.add.at(cnt, (cell, c), ww)
    pi_g = cnt.sum(0) / cnt.sum()
    P_cell = (cnt + LAPL * pi_g) / (cnt.sum(1, keepdims=True) + LAPL)
    logP = np.log(np.clip(P_cell, EPS, None))

    def entity(ids, S):
        uids, inv = np.unique(ids, return_inverse=True)
        sw = np.bincount(inv, weights=ww)
        sw2 = np.bincount(inv, weights=ww * ww)
        neff = sw ** 2 / np.maximum(sw2, 1e-12)
        C = np.zeros((len(uids), n_cls))
        np.add.at(C, (inv, c), ww)
        f = C / np.maximum(sw, 1e-12)[:, None]
        E = np.zeros((len(uids), n_cls))
        for k in range(n_cls):
            E[:, k] = np.bincount(inv, weights=ww * P_cell[cell, k])
        E /= np.maximum(sw, 1e-12)[:, None]
        E = np.clip(E, EPS, None)
        E /= E.sum(1, keepdims=True)
        post = (neff[:, None] * f + S * E) / (neff[:, None] + S)
        u = np.log(np.clip(post, EPS, None) / E)
        return uids, u, neff, f, E

    out = dict(logP=logP, n_cls=n_cls)
    if use_pitcher:
        uids, u, neff, f, E = entity(d["pitcher_id"].to_numpy(dtype=np.int64), s_p)
        gmm_info = ""
        if use_gmm and n_cls == 5:
            from sklearn.mixture import GaussianMixture
            fitm = neff >= GMM_NMIN
            best, bic0 = None, np.inf
            for K in range(1, GMM_KMAX + 1):
                gm = GaussianMixture(K, covariance_type="full", random_state=42,
                                     n_init=3).fit(u[fitm])
                b = gm.bic(u[fitm])
                if b < bic0:
                    best, bic0, bestK = gm, b, K
            gam = best.predict_proba(u)                # 전 투수 소프트 배정
            adj = gam @ best.means_                    # (n_p, 5) 군집 프로파일
            pri = E * np.exp(adj)
            pri /= pri.sum(1, keepdims=True)
            post = (neff[:, None] * f + s_p * pri) / (neff[:, None] + s_p)
            u = np.log(np.clip(post, EPS, None) / E)
            gmm_info = (f"GMM K={bestK} (BIC, 적합 투수 {int(fitm.sum())}명"
                        f"/{len(uids)}명, 점유 {np.round(gam.mean(0), 2)})")
        if permute:
            perm = rng.permutation(len(uids))
            u = u[perm]
        out.update(p_ids=uids, p_u=u, gmm_info=gmm_info)
    if use_batter:
        uids, u, neff, f, E = entity(d["batter_id"].to_numpy(dtype=np.int64), s_b)
        if permute:
            perm = rng.permutation(len(uids))
            u = u[perm]
        out.update(b_ids=uids, b_u=u)
    return out


def predict_xpc(mdl, va):
    z = mdl["logP"][cell_of(va)].copy()
    for key, col in (("p", "pitcher_id"), ("b", "batter_id")):
        if f"{key}_ids" not in mdl:
            continue
        ids = va[col].to_numpy(dtype=np.int64)
        pos = np.searchsorted(mdl[f"{key}_ids"], ids)
        pos = np.clip(pos, 0, len(mdl[f"{key}_ids"]) - 1)
        hit = mdl[f"{key}_ids"][pos] == ids
        z[hit] += mdl[f"{key}_u"][pos[hit]]
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    p = e[:, 0] / e.sum(1)
    return np.clip(p, EPS, 1 - EPS)


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)

    # 배포 등가층 (캐시 필요 — 없으면 단독 평가만)
    deploy = {}
    try:
        import deploy29 as D
        import resprobe as R
        for fold in (2023, 2024):
            deploy[fold] = R.deploy_base(df, s, y, fold)
    except Exception as e:                             # noqa: BLE001
        print(f"⚠️ 배포 등가층 로드 실패 ({type(e).__name__}: {e}) — 단독 평가만 한다")
        deploy = None

    res = {}
    for fold in (2023, 2024):
        print(f"\n{'='*96}\n=== fold {fold} ===")
        mtr, mva = _fc.SPLITS[fold](s, df["game_month"].to_numpy())
        tr = df[mtr]
        if deploy:
            va = deploy[fold]["st"]["d"]
            yv = deploy[fold]["yv"]
            assert len(va) == int(mva.sum()), "배포층 검증행 수 불일치"
        else:
            va = df[mva]
            yv = va[TARGET_COL].to_numpy(dtype=np.float64)
        ybar = yv.mean()

        cls, ok = make_classes(tr)
        w = recency_weight(tr["season"].to_numpy(), HL,
                           ref_season=int(tr["season"].max()))
        print(f"  학습 {len(tr):,}행 (복원 {ok.mean():.1%}) → 검증 {len(va):,}행"
              f"  ({time.time()-t0:.0f}s)")
        dist = np.bincount(cls[ok], minlength=5) / ok.sum()
        print("  5클래스 분포  " + "  ".join(f"{n} {v:.4f}" for n, v in zip(CLS5, dist)))

        def sc(p):
            return bss(yv, shift_to_prior(p, ybar))

        rng = np.random.default_rng(189 + fold)
        arms = {}
        arms["A0_셀만"] = fit_xpc(tr, cls, ok, w, 5, S_P, S_B,
                                  use_pitcher=False, use_batter=False)
        arms["A1_이진"] = fit_xpc(tr, cls, ok, w, 2, S_P, S_B)
        arms["A2_5cls_투수만"] = fit_xpc(tr, cls, ok, w, 5, S_P, S_B, use_batter=False)
        arms["A3_5cls_주모델"] = fit_xpc(tr, cls, ok, w, 5, S_P, S_B)
        arms["A4_GMM풀링"] = fit_xpc(tr, cls, ok, w, 5, S_P, S_B, use_gmm=True)
        arms["PLB_정체순열"] = fit_xpc(tr, cls, ok, w, 5, S_P, S_B,
                                       rng=rng, permute=True)
        if arms["A4_GMM풀링"].get("gmm_info"):
            print("  " + arms["A4_GMM풀링"]["gmm_info"])

        P = {nm: predict_xpc(m, va) for nm, m in arms.items()}
        print(f"\n  ── 단독 BSS oracle (shift→ȳ={ybar:.4f}) ──")
        for nm, p in P.items():
            extra = ""
            if deploy:
                extra = (f"   corr(z, z_dep) "
                         f"{np.corrcoef(_logit(p), deploy[fold]['z'])[0, 1]:.3f}")
            print(f"  {nm:<14s} {sc(p):>+9.1f}{extra}")

        # hl × S_P 스윕 (A3, 정보용)
        print(f"\n  ── A3 스윕 (행=hl, 열=S_P; S_B=S_P×3.3) ──")
        print("        " + "".join(f"{sp:>10.0f}" for sp in SWEEP_SP))
        for hl in SWEEP_HL:
            wh = recency_weight(tr["season"].to_numpy(), hl,
                                ref_season=int(tr["season"].max()))
            row = []
            for sp in SWEEP_SP:
                mm = fit_xpc(tr, cls, ok, wh, 5, sp, sp * 10 / 3)
                row.append(sc(predict_xpc(mm, va)))
            print(f"  hl={hl:<4}" + "".join(f"{v:>+10.1f}" for v in row))

        if deploy:
            b = deploy[fold]
            zd = b["z"]
            print(f"\n  ── 배포 등가층 블렌드 (기준 {b['base']:.1f}) ──")
            print("  %-14s" % "α" + "".join(f"{a:>9.2f}" for a in ALPHAS))
            curves = {}
            for nm in ("A2_5cls_투수만", "A3_5cls_주모델", "A4_GMM풀링", "PLB_정체순열"):
                zr = _logit(P[nm])
                ds = [sc_blend(zd, zr, a, ybar, yv, b["base"]) for a in ALPHAS]
                curves[nm] = ds
                print(f"  {nm:<14s}" + "".join(f"{d:>+9.2f}" for d in ds))
            # 쌍체 부트스트랩 + §7.185 예측식 (α_GATE)
            import resprobe as R
            pmix = {nm: shift_to_prior(
                        1 / (1 + np.exp(-((1 - ALPHA_GATE) * zd
                                          + ALPHA_GATE * _logit(P[nm])))), ybar)
                    for nm in ("A2_5cls_투수만", "A3_5cls_주모델", "A4_GMM풀링",
                               "PLB_정체순열")}
            point, draws = R.boot_delta(yv, b["p"], list(pmix.values()),
                                        b["cl"], b["n_id"],
                                        np.random.default_rng(42), B=B_BOOT)
            ref = ybar * (1 - ybar)
            for j, nm in enumerate(pmix):
                lo, hi = np.percentile(draws[:, j], [2.5, 97.5])
                cost = 1e5 * np.mean((pmix[nm] - b["p"]) ** 2) / ref
                gain = point[j] + cost
                fc16 = gain * 0.16 - cost
                fc93 = gain * 0.93 - cost
                print(f"  {nm:<14s} Δ(α={ALPHA_GATE}) {point[j]:>+7.2f} "
                      f"[{lo:>+6.2f}, {hi:>+6.2f}]  분산비용 {cost:5.2f}"
                      f"  → LB예측 c=.16 {fc16:+.2f} / c=.93 {fc93:+.2f}")
            res[fold] = dict(curves=curves, point=dict(zip(pmix, point)))

    if deploy and len(res) == 2:
        print(f"\n{'='*96}\n판정 (사전 등록: knnmember 와 동일 — ⓐ 공통 양수 α 폭≥2"
              " ⓑ 위약은 그 구간 없음)")
        for nm in ("A2_5cls_투수만", "A3_5cls_주모델", "A4_GMM풀링"):
            both = [i for i in range(len(ALPHAS))
                    if res[2023]["curves"][nm][i] > 0
                    and res[2024]["curves"][nm][i] > 0]
            print(f"  {nm}: 공통 양수 {len(both)}칸 → "
                  f"{'생존(추가 검증 필요)' if len(both) >= 2 else '기각'}")
        plb = [i for i in range(len(ALPHAS))
               if res[2023]["curves"]["PLB_정체순열"][i] > 0
               and res[2024]["curves"]["PLB_정체순열"][i] > 0]
        print(f"  위약 공통 양수 {len(plb)}칸 → {'⚠️ 위약 생존' if len(plb) >= 2 else 'OK'}")
    print(f"\n  총 {time.time()-t0:.0f}s")


def sc_blend(zd, zr, a, ybar, yv, base):
    p = shift_to_prior(1 / (1 + np.exp(-((1 - a) * zd + a * zr))), ybar)
    return bss(yv, p) - base


if __name__ == "__main__":
    main()
