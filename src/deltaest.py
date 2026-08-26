"""δ **추정량** 3건 재측정 (RESUME.md "3. 만약 모델을 더 판다면") + 왜 δ는 RES를 올리는가.

축이 아니라 추정량을 건드린다. 현재 배포:

    δ(p,c) = n_eff/(n_eff+k) · (rate_pc − rate_p)      k=200, recency hl=1.0

  V1  recency half-life    §7.61에서 가중 자체는 들어갔다. **값이 1.0이고 배포 학습은 0.5**다.
                           정합이 안 맞는 상태 → hl 스윕.
  V2  로짓차 δ             확률차를 **로짓에** 더하는 것은 스케일 불일치다. dp/dz는 p=.5에서
                           .25, p=.30에서 .21 → 극단 투수에서 같은 Δp가 다른 Δz를 뜻한다.
                           δ = logit(rate_pc) − logit(rate_p) 가 자연 스케일.
  V3  그룹 평균으로 수축    지금 저표본 셀은 δ→0 (= "상황 편차 없음")으로 간다.
                           (던지는손 × 상황코드) 그룹 평균 편차로 수축하면?
                           ⚠️ 그 그룹 평균은 **주효과**라 모델이 이미 알 수 있다 → 이중계상 위험.
                              사전 예측: V3는 진다. 그래도 재는 이유는 크기를 알기 위해서다.

정직성 장치 (§7.101 규약)
  · 기준선은 **배포 스택** (blend α=0.75 + 4축 + prior shift, clean 행). 학습 없음 — z 캐시.
  · **스케일 정합**: 각 변형 δ를 `train 행 조회값의 std`가 배포 δ와 같아지도록 재스케일한다.
    안 그러면 "모양 바꿈"과 "세기 바꿈"이 섞인다. 스케일은 **train에서만** 잰다.
  · β는 배포값 고정. 추가로 전역 배율 γ를 **fold2023에서 골라 fold2024로만** 평가한다.
  · 오차막대는 투수 클러스터 부트스트랩.

실행:  .venv/Scripts/python.exe -u src/deltaest.py 2>&1 | tee experiments/deltaest.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache as _fc

KEY = _fc._key()

import features as F                                              # noqa: E402
import asofdelta                                                  # noqa: E402
from config import (ASOF_AXES, ASOF_STRENGTH, CONTAM_AXES, CTX_AXES,
                    CTX_HALF_LIFE, CTX_K, MLP_ALPHA, TARGET_COL,
                    TRAIN_CSV)                                    # noqa: E402
from data import load_csv, recency_weight                         # noqa: E402
from pclip import murphy_rank_debias                              # noqa: E402
from validation import bss, shift_to_prior                        # noqa: E402

FOLDS = [2023, 2024]
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED = 42
N_BOOT = 200
RCLIP = 0.01              # 로짓차에서 rate를 자르는 폭 (저표본 셀 0/1 방지)


def fit_ctx_var(tr, axes, k, w, *, mode="prob", group_prior=False):
    """`ctxfit.fit_ctx` 의 변형판. mode / group_prior 두 축만 다르다.

    mode="prob"   raw = rate_pc − rate_p                (배포)
    mode="logit"  raw = logit(rate_pc) − logit(rate_p)  (V2)

    group_prior=False  δ = s·raw                        (수축 목표 0, 배포)
    group_prior=True   δ = s·raw + (1−s)·ḡ              (V3, ḡ = 그룹 평균 편차)
                       그룹 = (pitcher_hand, 상황코드). n_eff 가중 평균.
    """
    y = tr[TARGET_COL].to_numpy(dtype=np.float64)
    pid = tr["pitcher_id"].to_numpy(dtype=np.int64)
    phand = tr["pitcher_hand"].to_numpy(dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)

    upid, invp = np.unique(pid, return_inverse=True)
    sw_p = np.bincount(invp, weights=w)
    rate_p = np.bincount(invp, weights=w * y) / np.maximum(sw_p, 1e-12)
    # 투수 → 던지는손 (한 투수는 한 손이다)
    hand_of = np.zeros(len(upid), dtype=np.int64)
    hand_of[invp] = phand

    def _l(r):
        r = np.clip(r, RCLIP, 1 - RCLIP)
        return np.log(r / (1 - r))

    out = []
    for name, beta in axes:
        if not beta:
            continue
        code, m = F.CTX_CODERS[name](tr)
        key = pid * m + code
        u, inv = np.unique(key, return_inverse=True)
        sw = np.bincount(inv, weights=w)
        sw2 = np.bincount(inv, weights=w * w)
        n_eff = np.where(sw2 > 0, sw * sw / np.maximum(sw2, 1e-12), 0.0)
        t = np.bincount(inv, weights=w * y)
        pi = np.searchsorted(upid, u // m)          # 셀 → 투수 인덱스
        rc = t / np.maximum(sw, 1e-12)              # 셀 성공률
        rp = rate_p[pi]

        raw = (_l(rc) - _l(rp)) if mode == "logit" else (rc - rp)
        sh = n_eff / (n_eff + k)

        if group_prior:
            g = hand_of[pi] * m + (u % m)
            ug, ginv = np.unique(g, return_inverse=True)
            gs = np.bincount(ginv, weights=n_eff)
            gm = np.bincount(ginv, weights=n_eff * raw) / np.maximum(gs, 1e-12)
            delta = sh * raw + (1.0 - sh) * gm[ginv]
        else:
            delta = sh * raw

        out.append({"name": name, "beta": float(beta), "keys": u,
                    "delta": delta.astype(np.float64)})
    return out


def build(df, s, fold, *, hl=CTX_HALF_LIFE, mode="prob", group_prior=False,
          scale_ref=None):
    """fold 검증행의 설계행렬(4축 원단위)과 train 조회값 std 를 돌려준다."""
    tr, va = df[s < fold], df[s == fold]
    ref = int(s[s < fold].max())
    w = recency_weight(s[s < fold], hl, ref_season=ref)

    ctx = fit_ctx_var(tr, CTX_AXES, CTX_K, w, mode=mode, group_prior=group_prior)
    art = F.fit_artifacts(tr, ctx=ctx, contam_axes=CONTAM_AXES)
    art["asof"] = asofdelta.fit_asof_axes(tr, ASOF_AXES, ASOF_STRENGTH,
                                          ct=art.get("contam"))

    cols_va, sd_tr, betas = [], [], []
    for ax in ctx:
        cols_va.append(F.ctx_lookup(va, ax))
        sd_tr.append(float(F.ctx_lookup(tr, ax).std()))   # 스케일은 train 에서만
        betas.append(float(ax["beta"]))
    for a in art["asof"]:                                 # asof 축은 변형 대상 아님
        cols_va.append(F.asof_lookup(va, a, ct=art.get("contam")))
        betas.append(float(a["beta"]))

    Xva = np.column_stack(cols_va)
    sd_tr = np.array(sd_tr)
    if scale_ref is not None:                             # 배포 δ 와 세기를 맞춘다
        fscale = np.where(sd_tr > 0, scale_ref / np.maximum(sd_tr, 1e-12), 1.0)
        Xva[:, :len(ctx)] *= fscale
        sd_tr = sd_tr * fscale
    return Xva, np.array(betas), sd_tr


def main():
    rng = np.random.default_rng(SEED)
    import pandas as pd

    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    codes, uniq = pd.factorize(df["pitcher_id"].to_numpy())
    n_id = len(uniq)
    mclean = ~((gt == "F") & (s < 2023))
    NC = len(CTX_AXES)

    print(f"train {df.shape}   alpha={MLP_ALPHA}   지문 {KEY}   기준선 = s29 배포 스택")
    print(f"배포 δ: 확률차 · hl={CTX_HALF_LIFE} · k={CTX_K} · 수축목표 0   "
          f"β={dict(CTX_AXES)}")
    print()

    Z0, Y, CF, MM = {}, {}, {}, {}
    for f in FOLDS:
        vi = np.where(s == f)[0]
        mm = mclean[vi]
        with np.load(CACHE / f"fold{f}_{KEY}.npz") as d:
            z_mlp, yc = d["z_mlp"], d["y"]
        assert np.allclose(yc, y[vi]), "캐시 y 불일치"
        z_lgb = np.load(CACHE / f"initdep_fold{f}_ctr0.npz")["z"]
        Z0[f] = ((1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp)[mm]
        Y[f] = y[vi][mm]
        CF[f] = codes[vi][mm]
        MM[f] = mm

    def score(f, Xva, betas, gamma=1.0):
        b = betas.copy()
        b[:NC] *= gamma
        z = Z0[f] + Xva[MM[f]] @ b
        return shift_to_prior(1 / (1 + np.exp(-z)), Y[f].mean())

    def rep(f, p):
        return bss(Y[f], p)

    # ── 0. 개념 확인 ────────────────────────────────────────────────────
    print("=" * 104)
    print("█ 0. 왜 δ(사후 보정)는 RES 를 올리고 클리핑은 못 올리나 — 같은 자로")
    print("=" * 104)
    print("  RES 는 **분할**의 함수다. δ 는 같은 p 를 가진 행을 x 로 **쪼갠다**(셀 분리 → RES↑).")
    print("  클리핑은 다른 p 를 가진 행을 하나로 **뭉친다**(셀 병합 → RES↓). 값이 아니라 분할이다.")
    print()
    print(f"  {'fold':<8}{'예보':<18}{'셀수':>8}{'REL_true':>11}{'RES_true':>11}"
          f"{'dREL(BSS)':>12}{'dRES(BSS)':>12}{'BSS':>10}")
    BASE = {}
    for f in FOLDS:
        Xva, betas, sd0 = build(df, s, f)
        BASE[f] = (Xva, betas, sd0)
        p_off = shift_to_prior(1 / (1 + np.exp(-Z0[f])), Y[f].mean())
        p_on = score(f, Xva, betas)
        r0 = None
        for lab, pp in (("δ 4축 OFF", p_off), ("δ 4축 ON(배포)", p_on)):
            R, S, nz, U, nb = murphy_rank_debias(Y[f], pp, 1000)
            c = 100000.0 / U
            Rt, St = max(R - nz, 0.0), S - nz
            if r0 is None:
                r0 = (Rt, St)
                print(f"  {f:<8}{lab:<18}{nb:>8,}{Rt:>11.6f}{St:>11.6f}"
                      f"{'':>12}{'':>12}{rep(f, pp):>10.1f}")
            else:
                print(f"  {f:<8}{lab:<18}{nb:>8,}{Rt:>11.6f}{St:>11.6f}"
                      f"{-c * (Rt - r0[0]):>+12.1f}{c * (St - r0[1]):>+12.1f}"
                      f"{rep(f, pp):>10.1f}")
        print()

    B = {f: rep(f, score(f, BASE[f][0], BASE[f][1])) for f in FOLDS}
    print("=" * 104)
    print("█ 1. 기준선 (배포 δ):  " + "   ".join(f"fold{f} {B[f]:.1f}" for f in FOLDS))
    print("=" * 104)
    print()

    # ── V1 ──────────────────────────────────────────────────────────────
    print("=" * 104)
    print("█ V1 — δ 표의 recency half-life (배포 1.0 · 모델 학습 가중은 0.5)")
    print("=" * 104)
    print(f"  {'hl':<10}" + "".join(f"{f'fold{f} BSS':>14}{'Δ':>9}" for f in FOLDS)
          + f"{'δ std (hand/count4)':>26}")
    for hl in (0.25, 0.5, 1.0, 2.0, 99.0):
        cells, sds = [], None
        for f in FOLDS:
            Xva, betas, sd = build(df, s, f, hl=hl)
            v = rep(f, score(f, Xva, betas))
            cells.append(f"{v:>14.1f}{v - B[f]:>+9.1f}")
            if f == 2024:
                sds = sd
        tag = "균등" if hl > 50 else f"{hl}"
        print(f"  {tag:<10}" + "".join(cells)
              + f"{f'{sds[0]:.5f} / {sds[1]:.5f}':>26}")
    print()

    # ── V2 / V3 ─────────────────────────────────────────────────────────
    print("=" * 104)
    print("█ V2 로짓차 δ · V3 그룹평균 수축 — 배포 δ 와 **train std 를 맞춘 뒤** 비교")
    print("=" * 104)
    print(f"  {'변형':<26}" + "".join(f"{f'fold{f} BSS':>14}{'Δ':>9}" for f in FOLDS))
    VAR = {}
    for lab, kw in (("V2 로짓차", dict(mode="logit")),
                    ("V3 그룹평균 수축", dict(group_prior=True)),
                    ("V2+V3", dict(mode="logit", group_prior=True))):
        cells, keep = [], {}
        for f in FOLDS:
            Xva, betas, _ = build(df, s, f, scale_ref=BASE[f][2], **kw)
            keep[f] = (Xva, betas)
            v = rep(f, score(f, Xva, betas))
            cells.append(f"{v:>14.1f}{v - B[f]:>+9.1f}")
        VAR[lab] = keep
        print(f"  {lab:<26}" + "".join(cells))
    print()

    # ── 전역 배율 γ ─────────────────────────────────────────────────────
    print("=" * 104)
    print("█ 전역 배율 γ 를 **fold2023 에서 고르고 fold2024 로만 평가** (§7.101 규약)")
    print("=" * 104)
    GAM = np.arange(0.6, 1.65, 0.05)
    print(f"  {'변형':<26}{'g*(2023)':>10}{'fold2023 BSS':>15}"
          f"{'fold2024 BSS':>15}{'Δ vs 배포':>12}")
    allv = [("배포 δ", {f: (BASE[f][0], BASE[f][1]) for f in FOLDS})] + list(VAR.items())
    for lab, keep in allv:
        sc23 = [rep(2023, score(2023, keep[2023][0], keep[2023][1], g)) for g in GAM]
        g = float(GAM[int(np.argmax(sc23))])
        v24 = rep(2024, score(2024, keep[2024][0], keep[2024][1], g))
        print(f"  {lab:<26}{g:>10.2f}{max(sc23):>15.1f}{v24:>15.1f}"
              f"{v24 - B[2024]:>+12.1f}")
    print()

    # ── 오차막대 ────────────────────────────────────────────────────────
    print("=" * 104)
    print(f"█ 클러스터 부트스트랩 (투수 복원추출 {N_BOOT}회) — Δ vs 배포 δ, γ=1 고정")
    print("=" * 104)
    print(f"  {'변형':<26}" + "".join(f"{f'fold{f}  Δ [95%]':>30}" for f in FOLDS))
    for lab, keep in VAR.items():
        cells = []
        for f in FOLDS:
            p1 = score(f, keep[f][0], keep[f][1])
            p0 = score(f, BASE[f][0], BASE[f][1])
            cf, yv = CF[f], Y[f]
            order = np.argsort(cf, kind="stable")
            bnd = np.searchsorted(cf[order], np.arange(n_id + 1))
            seen = np.where(np.diff(bnd) > 0)[0]
            bs = []
            for _ in range(N_BOOT):
                pick = rng.choice(seen, size=len(seen), replace=True)
                idx = np.concatenate([order[bnd[j]:bnd[j + 1]] for j in pick])
                yy = yv[idx]
                bs.append(bss(yy, p1[idx]) - bss(yy, p0[idx]))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            cells.append(f"{rep(f, p1) - B[f]:+7.1f}  [{lo:+6.1f}, {hi:+6.1f}]")
        print(f"  {lab:<26}" + "".join(f"{c:>30}" for c in cells))


if __name__ == "__main__":
    main()
