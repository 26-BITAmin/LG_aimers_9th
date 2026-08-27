"""§7.180 — 잔차 프로브 + generic residual stacker: "마지막 로짓 위에 또 한 번 예측"

사용자: *"파이프라인에서 마지막 로짓이 나오는 패턴을 보고 또 예측을 할 수는 없나
        → 잔차 프로브도 해보고, 직접 해보는 것도 수행해봐"*

무엇을 재나
    §7.115a 가 증명한 것: p 만 보는 2차 예측(값 변환)은 RES 불변이라 죽은 채널.
    열려 있는 형태는 f(p, x) 뿐이다. 지금까지 그 채널은 수작업 축(δ·fspec·
    init_score)으로만 캤다. 여기서는 **범용 학습기**(2차 LGBM)를 배포층 잔차
    전체에 얹어, 수작업 축이 못 판 잔차 신호가 남아 있는지 잰다.

    A. 잔차 프로브 — 배포 78피처 각각 vs 잔차 r = y − p 의 투수 클러스터 강건 t.
       두 fold 부호 일치 여부. 78개 스크린이므로 |t|>2 는 ~3.5개가 우연 기대치다.
    B. 스태커 — LGBM(binary, init_score=z_dep, 피처=X+z_dep) 을 fold A 검증행에
       학습하고 fold B 에 전이 적용. **2023→2024 가 배포 방향**(스태커를 과거
       홀드아웃에 학습해 다음 시즌에 적용)이고 2024→2023 은 지속성 대조.

베이스층 (두 fold 공통, 정직한 홀드아웃)
    §7.150b calslope 와 동일 — s29 블렌드 + cat10(w=.10) + δ축 + 1.05 스케일 +
    prior shift. s41/T1 예측은 fold2024 를 in-sample 로 봐서(아티팩트가 2024 포함)
    잔차 측정에 쓸 수 없다. fspec 은 fold2023 에 없어 공통층에서 제외.

팔 (전이 방향마다)
    XZ    2차 LGBM, 피처 = 배포 78피처 + z_dep, init_score = z_dep   ← 후보
    ZO    같은 구성, 피처 = z_dep 하나                               ← 죽은 채널 영가설
          (§7.115a: p 만 보는 변환은 RES 불변 → 이 팔의 분포가 노이즈 프로파일)
    PLB   XZ 의 dz 를 평가 fold 안에서 행 셔플                       ← 분산 팽창 위약
          ([[variance-expanding-changes-need-a-temperature-control]])
    수축  c ∈ {0.1, 0.25, 0.5, 1.0} × dz  (레벨형 교정의 전이 감쇠 전례 c≈0.16)

사전 등록 판정 (후보 = XZ, c=1)
    ① 2023→2024 Δ > 0, P(Δ>0) ≥ 0.9 (투수 클러스터 부트스트랩 B=4000)
    ② 같은 draw 쌍체로 Δ(XZ) − Δ(PLB) > 0, P ≥ 0.9
    ③ 2024→2023 부호 일치 ([[check-sign-agreement-on-the-relationship-not-the-metric]])
    ④ ZO 는 게이트가 아니라 자(尺) — XZ 가 ZO 와 구분되지 않으면 x 는 일 안 한 것
    in-fold(학습 fold 자신) 이득은 in-sample성 지표로만 찍는다 — 증거 아님.

실행:  .venv/Scripts/python.exe -u src/resprobe.py 2>&1 | tee experiments/resprobe.log
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import lightgbm as lgb
import numpy as np
import pandas as pd

import deploy29 as D
import foldcache as _fc
from catens import SEEDS as CAT_SEEDS, VARIANTS as CAT_VARIANTS
from config import (CENTER_ASOF, CONTAM_AXES, DROP_FEATURES, RECENCY_HALF_LIFE,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, USE_PLAYER_IDS)
from data import recency_weight
from features import build_features, fit_artifacts
from predict import PROB_CENTER, PROB_SCALE
from validation import bss, shift_to_prior

CACHE = D.CACHE
CAT_W = 0.10                       # §7.142 사전 고정
B_BOOT = 4000                      # [[bootstrap-B-is-a-hyperparameter]]
EPS = 1e-6
SHRINK = (0.10, 0.25, 0.50, 1.00)

# 2차 LGBM — 사전 등록 (스윕으로 고르지 않는다. 작게: 250k 행 잔차용)
STACK_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31,
                    min_data_in_leaf=1000, feature_fraction=0.8,
                    bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                    seed=42, bagging_seed=42, feature_fraction_seed=42,
                    data_random_seed=42)
STACK_ROUNDS = 300


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def deploy_base(df, s, y, fold):
    """fold 의 배포 등가 베이스: z_dep(로짓), p_dep, 잔차, 클러스터."""
    st = D.state(df, s, y, fold)
    ax = st["Xax"] @ st["dep"]
    zb = st["z0"] - ax                                   # 블렌드 (δ 이전)
    ens = np.mean([np.load(CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz")["z"]
                   for v in CAT_VARIANTS for sd in CAT_SEEDS], axis=0)[st["mm"]]
    zc = (1 - CAT_W) * zb + CAT_W * ens + ax
    ybar = st["yv"].mean()
    p_beta = shift_to_prior(1 / (1 + np.exp(-zc)), ybar)
    p_dep = shift_to_prior(PROB_CENTER + PROB_SCALE * (p_beta - PROB_CENTER), ybar)
    z_dep = _logit(p_dep)
    print(f"  fold {fold}: n={len(st['yv']):,}  s29층 {st['base']:.1f} → "
          f"+cat10+스케일 {bss(st['yv'], p_dep):.1f}")
    return dict(st=st, z=z_dep, p=p_dep, yv=st["yv"], ybar=ybar,
                cl=st["cl"], n_id=st["n_id"], base=bss(st["yv"], p_dep))


def fold_X(df, fold):
    """fold 검증행의 배포 피처 — 아티팩트는 그 fold 의 학습 시즌으로만 적합."""
    f = CACHE / f"resprobe_X_fold{fold}_{D.KEY}.npz"
    if f.exists():
        with np.load(f, allow_pickle=True) as d:
            return pd.DataFrame(d["X"], columns=list(d["cols"]))
    mtr, mva = _fc.SPLITS[fold](df.season.to_numpy(), df.game_month.to_numpy())
    tr, va = df[mtr], df[mva]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                        center_asof=CENTER_ASOF)
    X = build_features(va, art)
    if USE_PLAYER_IDS:
        for c in ("pitcher_id", "batter_id"):
            X[c] = va[c].to_numpy(dtype=np.float32)
    X = X.drop(columns=[c for c in DROP_FEATURES if c in X.columns])
    np.savez_compressed(f, X=X.to_numpy(dtype=np.float32),
                        cols=np.array(X.columns, dtype=object))
    return X


# ── A. 잔차 프로브 ────────────────────────────────────────────────────
def probe(X, r, cl, n_id):
    """열별 score 검정: T = Σ f̃ᵢrᵢ / √(Σ_g s_g²),  s_g = 클러스터 합. CR0 동형."""
    out = {}
    for c in X.columns:
        v = X[c].to_numpy(dtype=np.float64)
        m = np.isfinite(v)
        u = np.zeros(len(v))
        u[m] = (v[m] - v[m].mean()) * r[m]              # NaN 행은 기여 0
        sg = np.bincount(cl, weights=u, minlength=n_id)
        den = np.sqrt(np.sum(sg ** 2))
        out[c] = float(u.sum() / den) if den > 0 else 0.0
    return out


def run_probe(F, XX):
    print("\n" + "=" * 96)
    print("A. 잔차 프로브 — 피처 vs r = y − p_dep, 투수 클러스터 강건 t  "
          "(78개 스크린: |t|>2 우연 기대 ~3.5개)")
    print("=" * 96)
    T = {}
    for fold in D.FOLDS:
        b = F[fold]
        Xp = XX[fold].copy()
        Xp["(z_dep)"] = b["z"]
        Xp["(|z_dep-med|)"] = np.abs(b["z"] - np.median(b["z"]))
        T[fold] = probe(Xp, b["yv"] - b["p"], b["cl"], b["n_id"])
    cols = list(T[2023].keys())
    t23 = np.array([T[2023][c] for c in cols])
    t24 = np.array([T[2024][c] for c in cols])
    both = np.minimum(np.abs(t23), np.abs(t24)) * (np.sign(t23) == np.sign(t24))
    order = np.argsort(-both)
    print(f"  {'피처':<34s}{'t_f2023':>9s}{'t_f2024':>9s}   부호일치·min|t| 상위 15")
    for i in order[:15]:
        print(f"  {cols[i]:<34s}{t23[i]:>9.2f}{t24[i]:>9.2f}")
    n2 = int(np.sum((np.abs(t23) > 2) & (np.abs(t24) > 2)
                    & (np.sign(t23) == np.sign(t24))))
    n3 = int(np.sum((np.abs(t23) > 3) & (np.abs(t24) > 3)
                    & (np.sign(t23) == np.sign(t24))))
    print(f"\n  부호 일치 & 양fold |t|>2: {n2}개   |t|>3: {n3}개   "
          f"(단독 |t|>2: f23 {int(np.sum(np.abs(t23) > 2))} / f24 {int(np.sum(np.abs(t24) > 2))})")
    return T


# ── B. 스태커 ────────────────────────────────────────────────────────
def fit_stacker(Xtr, ztr, ytr, cols):
    Xs = Xtr.copy()
    Xs["z_dep"] = ztr
    b = lgb.train(STACK_PARAMS, lgb.Dataset(Xs[cols], ytr, init_score=ztr),
                  num_boost_round=STACK_ROUNDS)
    return b


def boot_delta(yv, p0, plist, cl, n_id, rng, B=B_BOOT):
    """같은 클러스터 draw 위에서 여러 팔의 ΔBSS 분포 (쌍체 가능)."""
    ref = yv.mean() * (1 - yv.mean())
    n = np.bincount(cl, minlength=n_id)
    s0 = np.bincount(cl, weights=(p0 - yv) ** 2, minlength=n_id)
    S = [np.bincount(cl, weights=(p - yv) ** 2, minlength=n_id) for p in plist]
    seen = np.flatnonzero(n > 0)
    point = [1e5 * (s0.sum() - s1.sum()) / (n.sum() * ref) for s1 in S]
    draws = np.empty((B, len(plist)))
    for i in range(B):
        pick = rng.choice(seen, size=len(seen), replace=True)
        npick = n[pick].sum() * ref
        s0p = s0[pick].sum()
        for j, s1 in enumerate(S):
            draws[i, j] = 1e5 * (s0p - s1[pick].sum()) / npick
    return point, draws


def apply_dz(b, dz, c=1.0):
    return shift_to_prior(1 / (1 + np.exp(-(b["z"] + c * dz))), b["ybar"])


def run_stacker(F, XX):
    print("\n" + "=" * 96)
    print(f"B. generic residual stacker — LGBM(init_score=z_dep, "
          f"lr {STACK_PARAMS['learning_rate']}, leaves {STACK_PARAMS['num_leaves']}, "
          f"min_leaf {STACK_PARAMS['min_data_in_leaf']}, {STACK_ROUNDS}r)")
    print("=" * 96)
    gate = {}
    for src, dst in ((2023, 2024), (2024, 2023)):
        a, b = F[src], F[dst]
        cols = list(XX[src].columns) + ["z_dep"]
        assert cols == list(XX[dst].columns) + ["z_dep"], "fold 간 피처 불일치"

        bst_xz = fit_stacker(XX[src], a["z"], a["yv"], cols)
        bst_zo = fit_stacker(XX[src].iloc[:, :0], a["z"], a["yv"], ["z_dep"])

        Xe = XX[dst].copy()
        Xe["z_dep"] = b["z"]
        dz_xz = np.asarray(bst_xz.predict(Xe[cols], raw_score=True))
        dz_zo = np.asarray(bst_zo.predict(Xe[["z_dep"]], raw_score=True))
        rng_p = np.random.default_rng(12345)
        dz_plb = rng_p.permutation(dz_xz)                # 위약: x-정렬만 파괴

        # in-fold (학습 fold 자신) — in-sample성 지표. 증거 아님
        Xi = XX[src].copy()
        Xi["z_dep"] = a["z"]
        dzi = np.asarray(bst_xz.predict(Xi[cols], raw_score=True))
        d_in = bss(a["yv"], apply_dz(a, dzi)) - a["base"]
        print(f"\n  ── {src} → {dst} ──   std(dz)={dz_xz.std():.4f}  "
              f"in-fold Δ {d_in:+.1f} (in-sample, 참고)")

        arms = ([(f"XZ  c={c:.2f}", apply_dz(b, dz_xz, c)) for c in SHRINK]
                + [("ZO  (z만, 죽은채널 영가설)", apply_dz(b, dz_zo)),
                   ("PLB (dz 셔플 위약)", apply_dz(b, dz_plb))])
        rng = np.random.default_rng(42)
        point, draws = boot_delta(b["yv"], b["p"], [p for _, p in arms],
                                  b["cl"], b["n_id"], rng)
        print(f"  {'팔':<28s}{'ΔBSS':>9s}{'95% 구간':>22s}{'P(Δ>0)':>9s}")
        for j, (nm, _) in enumerate(arms):
            lo, hi = np.percentile(draws[:, j], [2.5, 97.5])
            print(f"  {nm:<28s}{point[j]:>+9.2f}   [{lo:>+7.2f}, {hi:>+7.2f}]"
                  f"{np.mean(draws[:, j] > 0):>9.3f}")
        j_xz = SHRINK.index(1.00)
        j_plb = len(arms) - 1
        dpair = draws[:, j_xz] - draws[:, j_plb]
        lo, hi = np.percentile(dpair, [2.5, 97.5])
        print(f"  {'Δ(XZ c=1 − PLB) 쌍체':<28s}{point[j_xz]-point[j_plb]:>+9.2f}"
              f"   [{lo:>+7.2f}, {hi:>+7.2f}]{np.mean(dpair > 0):>9.3f}")
        gate[(src, dst)] = dict(d=point[j_xz], p=np.mean(draws[:, j_xz] > 0),
                                pp=np.mean(dpair > 0))

    g = gate[(2023, 2024)]
    ok1 = g["d"] > 0 and g["p"] >= 0.9
    ok2 = g["pp"] >= 0.9
    ok3 = np.sign(gate[(2024, 2023)]["d"]) == np.sign(g["d"])
    print("\n" + "=" * 96)
    print(f"판정 (사전 등록)  ① 2023→2024 Δ>0·P≥.9: {'통과' if ok1 else '미달'} "
          f"(Δ{g['d']:+.2f}, P {g['p']:.3f})   ② 위약 쌍체 P≥.9: "
          f"{'통과' if ok2 else '미달'} ({g['pp']:.3f})   ③ 역방향 부호 일치: "
          f"{'통과' if ok3 else '미달'} ({gate[(2024, 2023)]['d']:+.2f})")
    print(("→ 후보 생존 — LB 슬롯 검토" if ok1 and ok2 and ok3
           else "→ 기각 — generic stacker 축 폐쇄") )


def main():
    t0 = time.time()
    df, s, y = D.load_df()
    F, XX = {}, {}
    for fold in D.FOLDS:
        F[fold] = deploy_base(df, s, y, fold)
        XX[fold] = fold_X(df, fold)
        print(f"  fold {fold} 피처 {XX[fold].shape}  ({time.time()-t0:.0f}s)")
    del df
    run_probe(F, XX)
    run_stacker(F, XX)
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
