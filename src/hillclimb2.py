"""§7.194b — Caruana selection **파트 2: 라이브러리 확장** (기각 모델 전부 투입)

⚠️ 로그(experiments/hillclimb2.log)에는 §7.191b 로 찍혀 있다 — §7.194 재번호 사유는
   hillclimb.py 머리말 참조.

사용자: "나는 이제까지 사용했던 모델들을 섞어보는 그런 것인 줄 알았는데."
— 맞다. Caruana (ICML 2004) 의 원형은 "Libraries of **Models**": 약한 모델도
라이브러리에 넣어두고 selection 이 넣을지·얼마나 넣을지 정하게 한다.
파트 1(hillclimb.py)은 배포 32멤버의 가중 재배정만 물었다. 여기서는
**fold z 캐시가 살아 있는 기각 모델 전부**를 라이브러리에 더한다.

추가 멤버 (양 fold 캐시 실존分만, 재학습 없음)
    tabm_*      TabM 변형 (§7.162~171 기각 — 위약 통과·2×2 증분 없음)
    p5/p5aux    TabM ones **위약** — 일부러 넣는다. selection 이 위약에 질량을
                주면 그 자체가 선택 과적합의 경고등 (죽은 축 영가설 재사용)
    ctrl*       TabM 하네스의 독립 MLP 대조군 (실모델)
    catens_*    CatBoost GPU 12멤버 (§7.140 — CPU 로 대체돼 기각)
    catgpu_*    CatBoost 피처셋 변형 allid/none/lowcard (§7.139)
    mono_*      단조 제약 LGBM 24멤버 (§7.144? 기각 · 옛 지문 c3d43f1a57)
    gnncol_GNN  GNN 컬럼 추가 LGBM 앙상블 (§7.183b 기각)

기각 모델을 넣는 것이 왜 정당한가: 기각 사유는 전부 "**고정 α 로 블렌드하면**
손해"였다. selection 은 채택/기각이 아니라 질량 배정을 묻는다 — 0 이 답이면
0 을 주면 된다. 반대로 여기서 질량을 받는 멤버는 α 실험이 놓친 조합 이득이
있다는 뜻이다.

사전 등록 (파트 1 과 동일 — 결과를 보고 완화하지 않는다)
  · 판정: sel=fold2023 (bagged, 평가 fold 무접촉) → fold2024 전이 Δ vs **B1**
    (배포 챔피언 등가) > 0, 투수 클러스터 부트스트랩 P(Δ>0) ≥ 0.9.
  · 역방향(sel 2024 → eval 2023) 부호 불일치는 경고등.
  · **위약 멤버(p5/p5aux) 가중 합 > 라이브러리 균등 기대(2/M)의 2배**면
    선택 노이즈 오염으로 판정을 보류하고 기각 쪽으로 읽는다.
  · 통과 시 독립 시드 B=4000 재확인 후에만 LB 후보.

실행 (⚠️ hillclimb.py 완료 후 — initdepm 캐시 공유):
    .venv/Scripts/python.exe -u src/hillclimb2.py 2>&1 | tee experiments/hillclimb2.log
"""
import re
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
import hillclimb as H
from validation import bss

CACHE = D.CACHE
FOLDS = (2023, 2024)
N_BAGS = 12
STEPS = 60          # 파트 1 실측: bag 스텝 중앙값 1~6 — 60이면 충분
B_BOOT = 1000
RNG_SEED = 43            # 파트 1 과 다른 시드 (독립 draw)
CLUSTER_FRAC = 0.6       # 백 fit 행 = 클러스터 60% 부트스트랩 (속도·다양성)


# ── 추가 멤버 발견 ────────────────────────────────────────────────────

def find_extras():
    """양 fold 모두 캐시가 있는 추가 멤버 목록: [(name, fam, {fold: path})]"""
    out = []

    def both(name, fam, p23, p24):
        if p23.exists() and p24.exists():
            out.append((name, fam, {2023: p23, 2024: p24}))

    # TabM 계열 — 태그를 f2023 쪽에서 발견하고 f2024 짝을 요구한다
    tags = set()
    for p in CACHE.glob("tabm_f2023_*.npz"):
        tags.add(p.name[len("tabm_f2023_"):-len(".npz")])
    for t in sorted(tags):
        if t.startswith(("p5",)):
            fam = "tabm_plb"          # ones 위약 — 노이즈 지표
        elif t.startswith("ctrl"):
            fam = "tabm_ctrl"
        else:
            fam = "tabm"
        both(f"tabm_{t}", fam,
             CACHE / f"tabm_f2023_{t}.npz", CACHE / f"tabm_f2024_{t}.npz")

    for v in ("base", "wide", "sym", "depthwise"):
        for sd in (42, 7, 2024):
            both(f"catgpuens_{v}_{sd}", "catgpu12",
                 CACHE / f"catens_fold2023_{v}_{sd}.npz",
                 CACHE / f"catens_fold2024_{v}_{sd}.npz")
    for t in ("allid", "none", "lowcard"):
        both(f"catfeat_{t}", "catfeat",
             CACHE / f"catgpu_fold2023_{t}.npz", CACHE / f"catgpu_fold2024_{t}.npz")
    for m in ("core", "wide"):
        for v in ("base", "wide", "extra", "lowff"):
            for sd in (42, 7, 2024):
                both(f"mono_{m}_{v}_{sd}", "mono",
                     CACHE / f"mono_mono_{m}_f2023_c3d43f1a57_{v}_{sd}.npz",
                     CACHE / f"mono_mono_{m}_f2024_c3d43f1a57_{v}_{sd}.npz")
    both("gnncol_GNN", "gnn",
         CACHE / "gnncol_fold2023_GNN_5488ef1b18.npz",
         CACHE / "gnncol_fold2024_GNN_5488ef1b18.npz")
    return out


def load_extra(path, n_expect):
    z = np.load(path, allow_pickle=True)["z"]
    z = np.asarray(z, dtype=np.float64)
    assert z.shape == (n_expect,), f"{path.name}: {z.shape} != {n_expect}"
    assert np.isfinite(z).all(), f"{path.name}: NaN/inf"
    # 확률로 저장된 파일 방어 — 로짓이어야 한다
    assert z.min() < 0 or z.max() > 1, f"{path.name}: 확률처럼 보인다"
    return z


# ── bagged selection (범용 풀 버전) ───────────────────────────────────

def bagged_pool(st, Zc, rng):
    """멤버 절반 + 클러스터 부트스트랩 fit / OOB 정지. 평가 fold 무접촉."""
    ax, yv, cl, n_id = st["ax"], st["yv"], st["cl"], st["n_id"]
    M = Zc.shape[1]
    nrow = np.bincount(cl, minlength=n_id)
    seen = np.where(nrow > 0)[0]
    order = np.argsort(cl, kind="stable")
    bnd = np.searchsorted(cl[order], np.arange(n_id + 1))

    W = np.zeros((N_BAGS, M))
    ks = []
    for b in range(N_BAGS):
        boot = rng.choice(seen, size=int(len(seen) * CLUSTER_FRAC), replace=True)
        oob = np.setdiff1d(seen, np.unique(boot))
        fit_ix = np.concatenate([order[bnd[c]:bnd[c + 1]] for c in boot])
        oob_ix = np.concatenate([order[bnd[c]:bnd[c + 1]] for c in oob])
        pool = np.sort(rng.choice(M, M // 2, replace=False))
        hist, curve = H.greedy(Zc[fit_ix], ax[fit_ix], yv[fit_ix], pool, STEPS,
                               Zs=Zc[oob_ix], axs=ax[oob_ix], ys=yv[oob_ix])
        kb = int(np.nanargmin(curve)) + 1
        W[b] = hist[kb - 1] / kb
        ks.append(kb)
    w = W.mean(axis=0)
    return w / w.sum(), np.array(ks)


def main():
    t0 = time.time()
    print(f"§7.194b 라이브러리 확장 Caruana   지문 {D.KEY}   "
          f"BAGS={N_BAGS} STEPS={STEPS} frac={CLUSTER_FRAC}")
    df, s, y = D.load_df()
    ST = {f: H.make_state(df, s, y, f, t0) for f in FOLDS}
    del df

    extras = find_extras()
    n_fold = {f: len(ST[f]["mask"]) for f in FOLDS}
    print(f"\n추가 멤버 {len(extras)}개 발견:")
    fam_ct = {}
    for name, fam, _ in extras:
        fam_ct[fam] = fam_ct.get(fam, 0) + 1
    print("  " + "  ".join(f"{k} {v}" for k, v in sorted(fam_ct.items())))

    # 라이브러리 조립: 배포 32 + 추가 — clean 행으로 자른다
    LIB = {}
    for f in FOLDS:
        st = ST[f]
        cols = [st["Z"][:, i][st["mask"]] for i in range(st["Z"].shape[1])]
        names = list(st["names"])
        fams = list(st["fam"])
        for name, fam, paths in extras:
            z = load_extra(paths[f], n_fold[f])
            cols.append(z[st["mask"]])
            names.append(name)
            fams.append(fam)
        LIB[f] = (np.stack(cols, axis=1), names, np.array(fams))
    names, fams = LIB[2023][1], LIB[2023][2]
    assert LIB[2024][1] == names
    M = len(names)
    print(f"  라이브러리 총 {M}멤버   ({time.time()-t0:.0f}s)")

    # 기준선 B1/B0 (추가 멤버 가중 0) — 파트 1 과 같은 수여야 한다
    P_ref, S_ref = {}, {}
    for f in FOLDS:
        st = ST[f]
        for lab in ("B0", "B1"):
            p, sc = H.eval_w(st, st["Z"], H.baseline_w(st["fam"], lab))
            P_ref[(f, lab)], S_ref[(f, lab)] = p, sc
        print(f"  fold{f}  B0 {S_ref[(f,'B0')]:.2f}   B1 {S_ref[(f,'B1')]:.2f}")

    # 참고: 추가 가족별 단독 BSS (가족 평균 멤버, prior shift 후)
    print(f"\n[참고] 추가 가족 단독 BSS (가족 내 평균 로짓)")
    for f in FOLDS:
        st = ST[f]
        Zc = LIB[f][0]
        b1z = Zc @ np.concatenate([H.baseline_w(st["fam"], "B1"),
                                   np.zeros(M - len(st["fam"]))])
        row = []
        for fam in sorted(set(fams) - {"lgb", "emb4", "noid", "cat"}):
            idx = np.where(fams == fam)[0]
            zm = Zc[:, idx].mean(axis=1)
            p = H.shift_to_prior(1 / (1 + np.exp(-(zm + st["ax"]))), st["ybar"])
            c = np.corrcoef(zm, b1z)[0, 1]
            row.append(f"{fam} {bss(st['yv'], p):7.1f} (corr {c:.3f})")
        print(f"  fold{f}: " + "   ".join(row))

    # ── bagged selection 2방향 ────────────────────────────────────
    rng = np.random.default_rng(RNG_SEED)
    results = {}
    for sf, ef in ((2023, 2024), (2024, 2023)):
        a, b = ST[sf], ST[ef]
        Za, _, _ = LIB[sf]
        Zb, _, _ = LIB[ef]
        w, ks = bagged_pool(a, Za, rng)

        # 평가 (Zc 는 이미 clean 행으로 잘려 있다 — eval_w 대신 직접)
        def sc(st, Zc, w):
            p = H.shift_to_prior(1 / (1 + np.exp(-(Zc @ w + st["ax"]))), st["ybar"])
            return p, bss(st["yv"], p)
        p_in, s_in = sc(a, Za, w)
        p_tr, s_tr = sc(b, Zb, w)
        d_in = s_in - S_ref[(sf, "B1")]
        d_tr = s_tr - S_ref[(ef, "B1")]
        lo, hi, pgt = H.boot_delta(b, p_tr, P_ref[(ef, "B1")],
                                   np.random.default_rng(RNG_SEED + ef))
        fam_w = {}
        for fam in sorted(set(fams)):
            fam_w[fam] = float(w[fams == fam].sum())
        plb = fam_w.get("tabm_plb", 0.0)
        n_plb = int((fams == "tabm_plb").sum())
        results[sf] = (d_in, d_tr, lo, hi, pgt, w, plb)
        print(f"\n[EXT{M}] sel={sf} → eval={ef}   스텝 중앙값 {int(np.median(ks))}"
              f"   ({time.time()-t0:.0f}s)")
        print("  가족 가중: " + "  ".join(f"{k} {v:.3f}" for k, v in fam_w.items()
                                       if v > 0.0005))
        print(f"  위약(p5*) 가중 {plb:.4f}  (균등 기대 {n_plb/M:.4f} — "
              f"2배 초과 시 선택 노이즈 경고)")
        print(f"  in-fold Δ vs B1  {d_in:+8.2f}")
        print(f"  transfer Δ vs B1 {d_tr:+8.2f}  95% [{lo:+.2f}, {hi:+.2f}]"
              f"  P(Δ>0)={pgt:.3f}")
        top = np.argsort(w)[::-1][:12]
        print("  상위 가중:" + "".join(f"  {names[i]} {w[i]:.3f}" for i in top))

    # ── 판정 ──────────────────────────────────────────────────────
    d_in23, d_tr24, lo, hi, p24, w23, plb23 = results[2023]
    _, d_tr23, *_ = results[2024]
    n_plb = int((fams == "tabm_plb").sum())
    noise_flag = plb23 > 2 * n_plb / M
    ok = d_tr24 > 0 and p24 >= 0.9 and not noise_flag
    print("\n" + "=" * 92)
    print("판정 — 사전 등록: Δ(sel2023→f2024, vs B1) > 0 & P≥0.9 & 위약 가중 정상")
    print("=" * 92)
    print(f"  in-fold(2023) {d_in23:+.2f}   → f2024 {d_tr24:+.2f} [{lo:+.2f}, {hi:+.2f}]"
          f" P={p24:.3f}   역방향 → f2023 {d_tr23:+.2f}")
    print(f"  위약 가중 {plb23:.4f} ({'⚠️ 초과' if noise_flag else '정상'})"
          f"   게이트: {'통과' if ok else '기각'}"
          + ("" if d_tr23 > 0 else "   (역방향 음수 경고)"))

    np.savez_compressed(CACHE / "hillclimb2_weights.npz",
                        w2023=results[2023][5], w2024=results[2024][5],
                        names=np.array(names), fams=fams)
    print(f"\n  저장: hillclimb2_weights.npz   총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
