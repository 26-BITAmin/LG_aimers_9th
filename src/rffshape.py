"""§7.196a — 사용자 반문: "RFF 823차원이면 width/depth 도 같이 키워야 공정한 것 아닌가"
   (작성 시점 번호 §7.191 은 동시 세션의 라우팅 실험이 선점 — §7.196a으로 이동)
        = §7.25(폭)·§7.26(RFF) 의 **현 스택 재감사** + 상호작용 2×2

왜 다시 재나 (④류 재감사 — §7.181 분류)
    §7.25 wide256 −88/−75 · §7.26 rff8 −285/−148 은 **soft label(§7.70)·aux head(§7.41)
    이전 하네스** 측정이다. LGBM 쪽 §7.18 용량 기각이 같은 이유로 T1 계보에서
    역전됐다 (CAP2 +1.72, LB +1.02). MLP 쪽 근접 증거는 음성(§7.162 TabM ·
    §7.176a dropout/wd)이지만 RFF×width **상호작용 칸**은 미측정이다.
    사용자 논지: 입력 73→823 이면 hid 128 병목이 부당하다. 대항 논지: 첫 층
    파라미터는 이미 11배 늘었고 RFF 는 같은 스칼라의 결정적 변환이라 정보량이
    같다. → 실측으로 가른다.

팔 (전부 **현 배포 등가 경로**: fc.fold_data → aux_labels → soften →
    mlp.encode → train_mlp(emb4 cfg, aux=λ1.0) — foldcache.compute 와 동일 코드열)
    CTL128  emb4 그대로 (hid 128)                 3시드 — 앵커 겸 기준
    RFF128  + numemb=rff8 (d=8, σ=.2, §7.26 스펙)  3시드 — §7.26 재감사
    CTL256  hid 256                                3시드 — §7.25 wide256 재감사
    RFF256  hid 256 + rff8                         3시드 — 상호작용 칸 (사용자 제안)
    RFF512  hid 512 + rff8                         2시드 — 프로브 (첫층 병목 최소화)
    depth 는 §7.25 "깊이는 공짜" 인용으로 대체 (재실행 없음 — 한계로 명기).

평가 (사전 등록)
    MLP 단독 Δ(arm − CTL128) + 블렌드 Δ — 블렌드 = 0.25·z_lgb(캐시) +
    0.75·(0.875·z_arm + 0.125·z_noid(캐시, 팔 공통)). 투수 클러스터 쌍체
    부트스트랩 B=2000, 양 fold.
    후보 조건: 어떤 RFF 칸이라도 블렌드 Δ 양 fold 양수 & P≥0.9.
    2×2 분해(128/256): RFF 주효과 · 폭 주효과 · 상호작용 병기.
    앵커: CTL128 재학습 z vs foldcache 캐시 emb4 멤버 — 대조 보고.
    미달 → §7.25·§7.26 기각이 현 스택에서 유지 + 상호작용 칸 폐쇄로 기록.

실행:  .venv/Scripts/python.exe -u src/rffshape.py 2>&1 | tee experiments/rffshape.log
"""
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

import foldcache as fc
import mlp
import modelabel
import numemb
import soften
from config import (MLP_AUX_LAMBDA, MLP_VARIANTS, SOFT_LABEL_F, TARGET_COL,
                    TRAIN_CSV)
from data import load_csv
from validation import bss, shift_to_prior

CACHE = fc.CACHE
KEY = fc._key()
B_BOOT = 2000
FOLDS = (2023, 2024)
BASE_CFG = MLP_VARIANTS["emb4"][0]           # epochs 9 · lr 2e-3 · do .3 · wd 1e-2
ARMS = {
    "CTL128": dict(hid=128, rff=False, seeds=(42, 7, 2024)),
    "RFF128": dict(hid=128, rff=True,  seeds=(42, 7, 2024)),
    "CTL256": dict(hid=256, rff=False, seeds=(42, 7, 2024)),
    "RFF256": dict(hid=256, rff=True,  seeds=(42, 7, 2024)),
    "RFF512": dict(hid=512, rff=True,  seeds=(42, 7)),
}


def run_fold(fold, df, base, t0):
    Xtr, ytr, w, Xva, yva, art, tr = fc.fold_data(df, fold)
    aux, cov = modelabel.aux_labels(tr)
    assert SOFT_LABEL_F
    Xtr, ytr, w, aux, _h = soften.soften(tr, Xtr, ytr, w, aux)
    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    del Xtr, Xva
    gc.collect()
    spec = numemb.fit_rff(np.asarray(A), d=8, sigma=0.2, seed=42)
    print(f"  fold {fold}: 학습 {len(A):,}행 · 수치 {A.shape[1]} → RFF "
          f"{numemb.out_dim(spec, A.shape[1])}차원 · 보조 라벨 {cov:.1%} "
          f"({time.time()-t0:.0f}s)")

    Z = {}
    for arm, cfg in ARMS.items():
        zs = []
        for sd in cfg["seeds"]:
            pm = CACHE / f"rffshape_fold{fold}_{arm}_{sd}_{KEY}.npz"
            if pm.exists():
                zs.append(np.load(pm)["z"])
                continue
            over = dict(BASE_CFG, hid=cfg["hid"], aux=MLP_AUX_LAMBDA)
            if cfg["rff"]:
                over["numemb"] = spec
            _, curve = mlp.train_mlp(A, C, ytr, w, B, Cv, sizes, seed=sd,
                                     verbose=False, y_aux=aux, **over)
            zs.append(curve[-1])
            np.savez_compressed(pm, z=zs[-1])
            print(f"    {arm}/{sd} 단독 {mlp.score(zs[-1], yva):.1f} "
                  f"({time.time()-t0:.0f}s)")
        Z[arm] = np.mean(zs, axis=0)

    # 앵커 — CTL128/42 재학습 z vs foldcache 캐시 emb4_42 멤버
    mem = base[fold][3]
    z42 = np.load(CACHE / f"rffshape_fold{fold}_CTL128_42_{KEY}.npz")["z"]
    print(f"  앵커: |CTL128/42 − 캐시 emb4_42| max {np.abs(z42 - mem[0]).max():.2e} · "
          f"corr {np.corrcoef(z42, mem[0])[0, 1]:.6f}")
    return Z, yva


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    base = fc.load(list(FOLDS), verbose=True)

    ZZ, Y, CL = {}, {}, {}
    for fold in FOLDS:
        ZZ[fold], Y[fold] = run_fold(fold, df, base, t0)
        _, mva = fc.SPLITS[fold](s, df.game_month.to_numpy())
        CL[fold] = pd.factorize(df[mva]["pitcher_id"].to_numpy())[0]
        assert np.allclose(Y[fold], base[fold][2])
    del df
    gc.collect()

    print("\n" + "=" * 100)
    print("§7.196a RFF × 폭 2×2 — 현 배포 등가 하네스 (soften·aux) · B=%d" % B_BOOT)
    print("=" * 100)
    verdict = {}
    for fold in FOLDS:
        zl, y = base[fold][0], Y[fold]
        mem = base[fold][3]
        z_noid = mem[5:8].mean(axis=0)
        cl = CL[fold]
        n_id = cl.max() + 1
        ybar = y.mean()
        ref = ybar * (1 - ybar)

        def pool_p(z_arm):
            zp = 0.25 * zl + 0.75 * (0.875 * z_arm + 0.125 * z_noid)
            return shift_to_prior(1 / (1 + np.exp(-zp)), ybar)

        n = np.bincount(cl, minlength=n_id)
        seen = np.flatnonzero(n > 0)
        rng = np.random.default_rng(42)
        picks = [rng.choice(seen, size=len(seen), replace=True)
                 for _ in range(B_BOOT)]

        def boot(pA, pB):
            SA = np.bincount(cl, weights=(pA - y) ** 2, minlength=n_id)
            SB = np.bincount(cl, weights=(pB - y) ** 2, minlength=n_id)
            return np.array([1e5 * (SB[pk].sum() - SA[pk].sum())
                             / (n[pk].sum() * ref) for pk in picks])

        P = {a: pool_p(z) for a, z in ZZ[fold].items()}
        p_ctl = P["CTL128"]
        b_ctl = bss(y, p_ctl)
        print(f"\n  ── fold {fold} ──   CTL128 블렌드 {b_ctl:.1f} · "
              f"MLP 단독 {mlp.score(ZZ[fold]['CTL128'], y):.1f}")
        print(f"  {'팔':<8s}{'MLP 단독':>10s}{'Δ단독':>9s}{'블렌드':>9s}"
              f"{'Δ블렌드':>9s}{'95% 구간':>20s}{'P(Δ>0)':>9s}")
        D = {}
        for a in ARMS:
            if a == "CTL128":
                continue
            d = boot(P[a], p_ctl)
            D[a] = d
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"  {a:<8s}{mlp.score(ZZ[fold][a], y):>10.1f}"
                  f"{mlp.score(ZZ[fold][a], y) - mlp.score(ZZ[fold]['CTL128'], y):>+9.1f}"
                  f"{bss(y, P[a]):>9.1f}{bss(y, P[a]) - b_ctl:>+9.2f}"
                  f"   [{lo:>+7.2f}, {hi:>+7.2f}]{np.mean(d > 0):>9.3f}")
        # 2×2 분해 (블렌드 draws 재사용 — 같은 picks 이므로 선형 결합 가능)
        inter = D["RFF256"] - D["RFF128"] - D["CTL256"]
        lo, hi = np.percentile(inter, [2.5, 97.5])
        print(f"  {'상호작용(RFF×폭)':<16s}{np.mean(inter):>+9.2f}"
              f"   [{lo:>+7.2f}, {hi:>+7.2f}]{np.mean(inter > 0):>9.3f}")
        verdict[fold] = {a: (float(np.mean(D[a])), float(np.mean(D[a] > 0)))
                         for a in D}

    print("\n" + "=" * 100)
    print("판정 (사전 등록: 어떤 RFF 칸이라도 블렌드 Δ 양 fold 양수 & P≥0.9 → 후보)")
    ok_any = False
    for a in ("RFF128", "RFF256", "RFF512"):
        oks = [verdict[f][a][0] > 0 and verdict[f][a][1] >= 0.9 for f in FOLDS]
        ok_any |= all(oks)
        print(f"  {a}: " + "  ".join(
            f"f{f} Δ{verdict[f][a][0]:+.2f} P{verdict[f][a][1]:.3f}" for f in FOLDS)
            + ("  ✅" if all(oks) else ""))
    print("→ " + ("**후보 생존** — LB 슬롯 별도 논의"
                  if ok_any else
                  "미달 — §7.25(폭)·§7.26(RFF) 기각이 현 스택에서 유지, 상호작용 칸 폐쇄"))
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
