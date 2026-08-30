"""§7.194 — Caruana greedy forward ensemble selection (hill climbing) 실측

⚠️ 로그(experiments/hillclimb.log)에는 §7.191 로 찍혀 있다 — 병렬 세션이 §7.191~193
   을 선점한 것을 실행 후에 발견해 §7.194 로 재번호했다 (중복 번호 지도 참조).

사용자 지적: NVIDIA playbook 의 "hill climbing" 은 Caruana+ (ICML 2004,
Ensemble Selection from Libraries of Models) 의 **멤버 라이브러리 greedy 전진
선택(중복 허용)**을 가리킨다. §7.190 표의 "✅ 변형(α·β 밴드 규칙)"은 과대 표기 —
스칼라 2~3개 튜닝이지 멤버 단위 조합 탐색이 아니다. 실제로 미측정.

무엇이 새 정보인가
    배포 블렌드는 **가족 내 균등 평균 + 가족 간 고정 스칼라**다
    (LGBM 12 균등 · MLP 8 변형가중 · Cat 12 균등, α = .25/.75/.10).
    Caruana selection 은 멤버 32개의 단체(simplex) 위 임의 가중을 탐색한다.
    §7.180(스태커, f(p,x) — 폐쇄)과 다른 물건: 새 피처 없음, 자유도 ~31.

층 — §7.141·§7.180 과 동일한 "두 fold 공통의 정직한 홀드아웃" (deploy29):
    z(w) = Σ_i w_i z_i + Xax@β(δ축 4)  →  shift_to_prior  →  BSS   (clean 행)
    멤버 32 = initdep LGBM 12 (4변형×3시드) + MLP 8 (emb4 5 + noid 3)
              + CatBoost CPU 12 (4변형×3시드)

⚠️ 사전 등록 (결과를 보고 완화하지 않는다)
  · **판정은 시즌 간 전이뿐이다.** sel=fold2023 에서 고른 w 를 fold2024 에서 채점.
    in-fold Δ 는 in-sample성 크기 보고용 (§7.180: in-fold +2,245 → 전이 −290).
  · 기준선 = **B1** (배포 챔피언 등가: MLP 변형가중 .875/.125, cat w=.10).
    B0(측정 하네스 관례 = MLP 평평 8평균)도 참고 보고.
  · 게이트: Δ(f2024, vs B1) > 0 이고 P(Δ>0) ≥ 0.9 (투수 클러스터 부트스트랩
    B=1000). 역방향(sel 2024 → eval 2023)의 부호 불일치는 경고등.
    통과 팔은 독립 시드 B=4000 재확인 후에만 LB 후보 (§7.181 규약).
  · 팔 3개 — RAW32(전 멤버) · GRP10(시드 평균 10그룹 — 시드 노이즈 추적 불가)
    · LOCK(가족 합을 B1 값에 잠그고 가족 안에서만 선택 = 순수 새 축).
  · 선택 절차는 Caruana 의 bagged selection: 16백, 백마다 멤버 절반 +
    클러스터 부트스트랩 fit 행, OOB 클러스터가 스텝 수를 고른다.
    **평가 fold 는 선택 절차 어디에도 안 들어간다.**

전례 경고 (이 축의 사전 확률을 정하는 것들)
  · 블렌드 가중의 홀드아웃 꼭짓점은 LB 꼭짓점과 3번 어긋났다
    (MLP_ALPHA §7.126 · fspec §7.132 · noid 비중 §7.27 LB −4.97).
  · [[blend-weight-curves-measure-in-sample-ness]] — 가중 곡선은 in-sample성을 잰다.
  · 그래서 여기서는 곡선의 꼭짓점을 고르지 않고 **전이 실측**만 읽는다.

실행:  .venv/Scripts/python.exe -u src/hillclimb.py 2>&1 | tee experiments/hillclimb.log
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

import deploy29 as D
from config import (CONTAM_AXES, DROP_FEATURES, ENSEMBLE_VARIANTS,
                    FINAL_LGB_PARAMS, MLP_ALPHA, N_ROUNDS, RECENCY_HALF_LIFE,
                    SEED_ENSEMBLE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    TARGET_COL, USE_PLAYER_IDS)
from validation import bss, shift_to_prior

CACHE = D.CACHE
FOLDS = (2023, 2024)
LGB_SEEDS = SEED_ENSEMBLE[:3]                    # fold 하네스 관례 (배포는 5시드)
LGB_VARIANTS = ("base", "wide", "extra", "lowff")
CAT_VARIANTS = ("base", "wide", "sym", "depthwise")
CAT_SEEDS = (42, 7, 2024)
MLP_TAGS = [("emb4", s) for s in (42, 7, 2024, 1234, 31337)] + \
           [("noid", s) for s in (42, 7, 2024)]

N_BAGS = 16
STEPS = 120
NEWTON_IT = 5
B_BOOT = 1000
RNG_SEED = 42
W_CAT = 0.10                                     # §7.141 사전 고정값


# ── 멤버 준비 ──────────────────────────────────────────────────────────

def train_lgb_members(df, fold, t0):
    """initdeploy.train_lgb 와 같은 경로, 단 멤버별로 저장한다.
    12멤버 평균이 initdep_fold{f}_ctr0 캐시와 일치해야 한다 — 배관 검증."""
    paths = {(v, sd): CACHE / f"initdepm_fold{fold}_{v}_{sd}.npz"
             for v in LGB_VARIANTS for sd in LGB_SEEDS}
    if all(p.exists() for p in paths.values()):
        return {k: np.load(p)["z"] for k, p in paths.items()}

    import lightgbm as lgb
    import soften
    from data import recency_weight
    from features import build_features, fit_artifacts
    from initfix3 import level_init

    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                        center_asof=False)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xtr, Xva = X(tr), X(va)
    ss = tr["season"].to_numpy(dtype=np.int64)
    gg = tr["game_type"].astype("object").to_numpy().astype(str)
    Xtr, y2, w2, _, _ = soften.soften(tr, Xtr, y_tr, w)
    h = soften.fit_h(tr)
    h_row = np.zeros(len(tr))
    for yr, v in h.items():
        h_row[(ss == yr) & (gg == "F")] = v
    dup = (h_row > 0) & (y_tr == 1)
    ss2, gg2 = np.concatenate([ss, ss[dup]]), np.concatenate([gg, gg[dup]])
    init_tr = level_init(ss2, gg2, y2, w2)
    assert len(init_tr) == len(Xtr)

    out = {}
    for v in LGB_VARIANTS:
        for sd in LGB_SEEDS:
            if paths[(v, sd)].exists():
                out[(v, sd)] = np.load(paths[(v, sd)])["z"]
                continue
            p = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v], seed=sd,
                     bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
            bst = lgb.train(p, lgb.Dataset(Xtr, y2, weight=w2, init_score=init_tr),
                            num_boost_round=N_ROUNDS)
            z = bst.predict(Xva, raw_score=True)
            del bst
            np.savez_compressed(paths[(v, sd)], z=z)
            out[(v, sd)] = z
            print(f"    lgb fold{fold} [{v}/{sd}] ({time.time()-t0:.0f}s)")
    return out


def load_library(df, fold, t0):
    """멤버 32 × 검증행 z 행렬. 배관 검증 2건 포함."""
    lgbm = train_lgb_members(df, fold, t0)
    z_lgb_mean = np.mean([lgbm[k] for k in lgbm], axis=0)
    cached = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
    mad = float(np.abs(z_lgb_mean - cached).max())
    assert mad < 1e-3, f"LGBM 멤버 평균 ≠ initdep 캐시 (max|Δ| {mad:.2e})"
    print(f"  fold{fold} LGBM 12멤버 재구성 OK  max|Δ|={mad:.1e}")

    with np.load(CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
        mem = d0["members"]
        z_mlp_flat = d0["z_mlp"]
    assert mem is not None and mem.shape[0] == 8, "MLP 멤버 캐시 없음"
    assert np.abs(mem.mean(axis=0) - z_mlp_flat).max() < 1e-6

    cats = {}
    for v in CAT_VARIANTS:
        for sd in CAT_SEEDS:
            p = CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz"
            assert p.exists(), f"cat 캐시 없음: {p.name}"
            cats[(v, sd)] = np.load(p)["z"]

    names, cols = [], []
    for v in LGB_VARIANTS:
        for sd in LGB_SEEDS:
            names.append(f"lgb_{v}_{sd}")
            cols.append(lgbm[(v, sd)])
    for i, (v, sd) in enumerate(MLP_TAGS):
        names.append(f"mlp_{v}_{sd}")
        cols.append(mem[i])
    for v in CAT_VARIANTS:
        for sd in CAT_SEEDS:
            names.append(f"cat_{v}_{sd}")
            cols.append(cats[(v, sd)])
    Z = np.stack(cols, axis=1).astype(np.float64)
    fam = np.array(["lgb"] * 12 + ["emb4"] * 5 + ["noid"] * 3 + ["cat"] * 12)
    return Z, names, fam


def baseline_w(fam, flavor):
    """배포 등가 가중. B0 = MLP 평평 8평균(측정 하네스 관례), B1 = 변형가중 .875/.125."""
    w = np.zeros(len(fam))
    w[fam == "lgb"] = (1 - W_CAT) * (1 - MLP_ALPHA) / 12
    if flavor == "B0":
        w[(fam == "emb4") | (fam == "noid")] = (1 - W_CAT) * MLP_ALPHA / 8
    else:
        w[fam == "emb4"] = (1 - W_CAT) * MLP_ALPHA * 0.875 / 5
        w[fam == "noid"] = (1 - W_CAT) * MLP_ALPHA * 0.125 / 3
    w[fam == "cat"] = W_CAT / 12
    assert abs(w.sum() - 1) < 1e-12
    return w


# ── 평가 기계 ──────────────────────────────────────────────────────────

def eval_w(st, Z, w):
    z = Z @ w
    p = shift_to_prior(1 / (1 + np.exp(-(z[st["mask"]] + st["ax"]))), st["ybar"])
    return p, bss(st["yv"], p)


def boot_delta(st, p_new, p_ref, rng, B=B_BOOT):
    """투수 클러스터 부트스트랩 — catcpuens 와 동형."""
    yv, cl, n_id = st["yv"], st["cl"], st["n_id"]
    ref = st["ybar"] * (1 - st["ybar"])
    n = np.bincount(cl, minlength=n_id)
    s0 = np.bincount(cl, weights=(p_ref - yv) ** 2, minlength=n_id)
    s1 = np.bincount(cl, weights=(p_new - yv) ** 2, minlength=n_id)
    seen = np.where(n > 0)[0]
    draws = np.empty(B)
    for b in range(B):
        pick = rng.choice(seen, size=len(seen), replace=True)
        N = n[pick].sum()
        draws[b] = 1e5 * (s0[pick].sum() - s1[pick].sum()) / (N * ref)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return lo, hi, float(np.mean(draws > 0))


def newton_shift(Zc, target, it=NEWTON_IT):
    """열별 로짓 평행이동 d: mean σ(z+d) = target. 마지막 P 를 돌려준다."""
    d = np.zeros(Zc.shape[1])
    P = None
    for _ in range(it):
        P = 1.0 / (1.0 + np.exp(-(Zc + d)))
        m = P.mean(axis=0)
        g = (P * (1 - P)).mean(axis=0)
        d += (target - m) / np.maximum(g, 1e-9)
    return 1.0 / (1.0 + np.exp(-(Zc + d)))


def greedy(Zf, axf, yf, pool, steps, scale=1.0, z_fixed=0.0,
           Zs=None, axs=None, ys=None, zs_fixed=0.0):
    """Caruana 전진 선택(중복 허용). 로짓 평균 공간.

    z(k) = z_fixed + scale · (선택된 k개의 평균) + ax  → 열별 prior shift → Brier
    Zs(스텝 정지용 홀드아웃)가 있으면 스텝별 Brier 곡선도 돌려준다."""
    n, M = Zf.shape
    target = yf.mean()
    zsum = np.zeros(n)
    cnt = np.zeros(M)
    hist = np.zeros((steps, M))
    stop_curve = np.full(steps, np.nan)
    if Zs is not None:
        zsum_s = np.zeros(len(Zs))
        target_s = ys.mean()
    Cf = Zf[:, pool]
    for k in range(1, steps + 1):
        Zc = z_fixed[:, None] if isinstance(z_fixed, np.ndarray) else z_fixed
        Zc = Zc + scale * (zsum[:, None] + Cf) / k + axf[:, None]
        P = newton_shift(Zc, target)
        br = ((P - yf[:, None]) ** 2).mean(axis=0)
        jj = int(np.argmin(br))
        j = pool[jj]
        zsum += Zf[:, j]
        cnt[j] += 1
        hist[k - 1] = cnt
        if Zs is not None:
            zsum_s += Zs[:, j]
            zc = zs_fixed + scale * zsum_s / k + axs
            Ps = newton_shift(zc[:, None], target_s)
            stop_curve[k - 1] = float(((Ps[:, 0] - ys) ** 2).mean())
    return hist, stop_curve


def bagged_select(st, Z, fam, rng, group=False, lock=None, w_base=None):
    """16백 bagged selection. 평가 fold 는 안 본다.

    group=True 면 시드 평균 10그룹 라이브러리.
    lock='B1' 이면 가족 합을 w_base 의 가족 합에 잠그고 가족 안에서만 선택."""
    mask, ax, yv, cl = st["mask"], st["ax"], st["yv"], st["cl"]
    Zc = Z[mask]
    if group:
        groups, gname = [], []
        for g in ("lgb_base", "lgb_wide", "lgb_extra", "lgb_lowff"):
            v = g.split("_")[1]
            idx = [i for i, nm in enumerate(st["names"]) if nm.startswith(f"lgb_{v}_")]
            groups.append(Zc[:, idx].mean(axis=1)); gname.append(g)
        for g, pre in (("emb4", "mlp_emb4"), ("noid", "mlp_noid")):
            idx = [i for i, nm in enumerate(st["names"]) if nm.startswith(pre)]
            groups.append(Zc[:, idx].mean(axis=1)); gname.append(g)
        for v in CAT_VARIANTS:
            idx = [i for i, nm in enumerate(st["names"]) if nm.startswith(f"cat_{v}_")]
            groups.append(Zc[:, idx].mean(axis=1)); gname.append(f"cat_{v}")
        Zc = np.stack(groups, axis=1)
    M = Zc.shape[1]

    n_id = st["n_id"]
    nrow = np.bincount(cl, minlength=n_id)
    seen = np.where(nrow > 0)[0]
    order = np.argsort(cl, kind="stable")
    bnd = np.searchsorted(cl[order], np.arange(n_id + 1))

    fams = None
    if lock is not None:
        fams = [("lgb", np.array([i for i in range(M)
                                  if st["names"][i].startswith("lgb")])),
                ("mlp", np.array([i for i in range(M)
                                  if st["names"][i].startswith("mlp")])),
                ("cat", np.array([i for i in range(M)
                                  if st["names"][i].startswith("cat")]))]

    W = np.zeros((N_BAGS, M))
    ks = []
    for b in range(N_BAGS):
        boot = rng.choice(seen, size=len(seen), replace=True)
        oob = np.setdiff1d(seen, np.unique(boot))
        fit_ix = np.concatenate([order[bnd[c]:bnd[c + 1]] for c in boot])
        oob_ix = np.concatenate([order[bnd[c]:bnd[c + 1]] for c in oob])
        Zf, axf, yf = Zc[fit_ix], ax[fit_ix], yv[fit_ix]
        Zs, axs, ys = Zc[oob_ix], ax[oob_ix], yv[oob_ix]

        if lock is None:
            pool = rng.choice(M, max(2, M // 2), replace=False)
            hist, curve = greedy(Zf, axf, yf, np.sort(pool), STEPS,
                                 Zs=Zs, axs=axs, ys=ys)
            kb = int(np.nanargmin(curve)) + 1
            W[b] = hist[kb - 1] / kb
            ks.append(kb)
        else:
            wb = np.zeros(M)
            for fname, fidx in fams:
                alpha = w_base[fidx].sum()
                others = np.setdiff1d(np.arange(M), fidx)
                zfix_f = Zf[:, others] @ w_base[others]
                zfix_s = Zs[:, others] @ w_base[others]
                pool = rng.choice(fidx, max(2, len(fidx) // 2), replace=False)
                hist, curve = greedy(Zf, axf, yf, np.sort(pool),
                                     STEPS // 2, scale=alpha, z_fixed=zfix_f,
                                     Zs=Zs, axs=axs, ys=ys, zs_fixed=zfix_s)
                kb = int(np.nanargmin(curve)) + 1
                wb[fidx] = alpha * hist[kb - 1, fidx] / kb
                ks.append(kb)
            W[b] = wb
    w = W.mean(axis=0)
    w /= w.sum()
    return w, (np.array(ks), gname if group else st["names"])


# ── 상태 준비 ──────────────────────────────────────────────────────────

def make_state(df, s, y, fold, t0):
    st0 = D.state(df, s, y, fold)              # 앵커 assert 포함 (807.2 / 924.5)
    Z, names, fam = load_library(df, fold, t0)
    mm, vi = st0["mm"], st0["vi"]
    # 배관 검증 — 멤버로 재구성한 블렌드가 dep29 z_blend 과 일치해야 한다
    z_rec = ((1 - MLP_ALPHA) * Z[:, fam == "lgb"].mean(axis=1)
             + MLP_ALPHA * Z[:, (fam == "emb4") | (fam == "noid")].mean(axis=1))
    z_ref = st0["z0"] - st0["Xax"] @ st0["dep"]
    mad = float(np.abs(z_rec[mm] - z_ref).max())
    assert mad < 1e-6, f"블렌드 재구성 실패 max|Δ| {mad:.2e}"
    print(f"  fold{fold} 블렌드 재구성 OK  max|Δ|={mad:.1e}   base(BSS)={st0['base']:.1f}")
    return dict(Z=Z, names=names, fam=fam, mask=mm,
                ax=st0["Xax"] @ st0["dep"], yv=st0["yv"], ybar=st0["yv"].mean(),
                cl=st0["cl"], n_id=st0["n_id"], base=st0["base"])


def fam_sums(w, fam):
    return {f: float(w[fam == f].sum()) for f in ("lgb", "emb4", "noid", "cat")}


def main():
    t0 = time.time()
    print(f"§7.194 Caruana hill climbing   지문 {D.KEY}   B_BAGS={N_BAGS} STEPS={STEPS}")
    df, s, y = D.load_df()
    ST = {f: make_state(df, s, y, f, t0) for f in FOLDS}
    del df
    rng = np.random.default_rng(RNG_SEED)

    # ── 기준선 ────────────────────────────────────────────────────
    P_ref, S_ref = {}, {}
    for f in FOLDS:
        st = ST[f]
        for lab in ("B0", "B1"):
            p, sc = eval_w(st, st["Z"], baseline_w(st["fam"], lab))
            P_ref[(f, lab)] = p
            S_ref[(f, lab)] = sc
        print(f"  fold{f}  B0(평평 MLP)+cat {S_ref[(f,'B0')]:.2f}   "
              f"B1(변형가중 MLP)+cat {S_ref[(f,'B1')]:.2f}   "
              f"(dep29 무cat 앵커 {st['base']:.1f})")

    # ── 진단: 전 fold greedy 곡선 (bagging 없음, in-sample성 크기 시각화) ──
    print(f"\n[진단] full-fold greedy 곡선 — 선택 fold vs 전이 fold  ({time.time()-t0:.0f}s)")
    for sf, ef in ((2023, 2024), (2024, 2023)):
        a, b = ST[sf], ST[ef]
        Za, Zb = a["Z"][a["mask"]], b["Z"][b["mask"]]
        hist, _ = greedy(Za, a["ax"], a["yv"], np.arange(Za.shape[1]), STEPS)
        marks = [5, 10, 20, 40, 80, 120]
        row_in, row_tr = [], []
        for k in marks:
            w = hist[k - 1] / k
            row_in.append(eval_w(a, a["Z"], w)[1] - S_ref[(sf, "B1")])
            row_tr.append(eval_w(b, b["Z"], w)[1] - S_ref[(ef, "B1")])
        print(f"  sel={sf}  k:      " + "".join(f"{k:>9d}" for k in marks))
        print(f"    in-fold Δ vs B1 " + "".join(f"{d:>+9.2f}" for d in row_in))
        print(f"    transfer Δ      " + "".join(f"{d:>+9.2f}" for d in row_tr))

    # ── 본 실험: bagged selection 3팔 × 2방향 ─────────────────────
    results = {}
    for arm, kw in (("RAW32", dict()), ("GRP10", dict(group=True)),
                    ("LOCK", dict(lock="B1"))):
        for sf, ef in ((2023, 2024), (2024, 2023)):
            a, b = ST[sf], ST[ef]
            wb = baseline_w(a["fam"], "B1")
            w, (ks, names) = bagged_select(a, a["Z"], a["fam"], rng,
                                           w_base=wb, **kw)
            if kw.get("group"):
                # 그룹 가중을 멤버 가중으로 환산 (그룹 안 균등)
                wm = np.zeros(len(a["fam"]))
                for gi, g in enumerate(names):
                    if g.startswith("lgb_"):
                        v = g.split("_")[1]
                        idx = [i for i, nm in enumerate(a["names"])
                               if nm.startswith(f"lgb_{v}_")]
                    elif g in ("emb4", "noid"):
                        idx = [i for i, nm in enumerate(a["names"])
                               if nm.startswith(f"mlp_{g}")]
                    else:
                        v = g.split("_", 1)[1]
                        idx = [i for i, nm in enumerate(a["names"])
                               if nm.startswith(f"cat_{v}_")]
                    wm[idx] = w[gi] / len(idx)
                w = wm
            p_in, s_in = eval_w(a, a["Z"], w)
            p_tr, s_tr = eval_w(b, b["Z"], w)
            d_in = s_in - S_ref[(sf, "B1")]
            d_tr = s_tr - S_ref[(ef, "B1")]
            d_tr0 = s_tr - S_ref[(ef, "B0")]
            lo, hi, pgt = boot_delta(b, p_tr, P_ref[(ef, "B1")], rng)
            results[(arm, sf)] = (d_in, d_tr, lo, hi, pgt, d_tr0, w)
            print(f"\n[{arm}] sel={sf} → eval={ef}   ({time.time()-t0:.0f}s)")
            print(f"  스텝 수 중앙값 {int(np.median(ks))}   "
                  f"가족 합 {fam_sums(w, a['fam'])}")
            print(f"  in-fold Δ vs B1  {d_in:+8.2f}")
            print(f"  transfer Δ vs B1 {d_tr:+8.2f}  95% [{lo:+.2f}, {hi:+.2f}]"
                  f"  P(Δ>0)={pgt:.3f}   (vs B0 {d_tr0:+.2f})")
            top = np.argsort(w)[::-1][:8]
            print("  상위 가중:" + "".join(
                f"  {a['names'][i]} {w[i]:.3f}" for i in top))

    # ── 판정 ──────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("판정 — 사전 등록: Δ(sel2023→f2024, vs B1) > 0 & P≥0.9. 역방향 부호는 경고등")
    print("=" * 92)
    print(f"  {'팔':<7s}{'in-fold23':>11s}{'→f2024 Δ':>11s}{'[95%]':>20s}"
          f"{'P':>7s}{'in-fold24':>11s}{'→f2023 Δ':>11s}{'게이트':>8s}")
    for arm in ("RAW32", "GRP10", "LOCK"):
        di23, dt24, lo, hi, p24, _, _ = results[(arm, 2023)]
        di24, dt23, *_ = results[(arm, 2024)]
        ok = "통과" if (dt24 > 0 and p24 >= 0.9) else "기각"
        warn = "" if dt23 > 0 else " (역방향 음수)"
        print(f"  {arm:<7s}{di23:>+11.2f}{dt24:>+11.2f}"
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>20s}{p24:>7.2f}"
              f"{di24:>+11.2f}{dt23:>+11.2f}{ok:>8s}{warn}")

    np.savez_compressed(CACHE / "hillclimb_weights.npz",
                        **{f"{arm}_{sf}": results[(arm, sf)][6]
                           for arm in ("RAW32", "GRP10", "LOCK")
                           for sf in FOLDS},
                        names=np.array(ST[2023]["names"]))
    print(f"\n  가중 저장: hillclimb_weights.npz   총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
