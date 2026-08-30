"""§7.195 — 힐클라임 전이 실패의 F 오염 채널 진단 (재학습 없음)

사용자 가설: "§7.194 hill climbing 이 못 쓰게 된 것은 F 리그 오염 데이터 때문이다.
오염 데이터를 수정하면 살릴 수 있다."

오염이 선택층에 들어올 통로는 멤버 학습 데이터뿐이다 — deploy29 의 선택·평가 행은
pre-2023 F 를 이미 제외한다 (mclean_all). fold2023 멤버는 F 학습 신호가 100% 오염
구간(2019~22)이고 fold2024 멤버는 깨끗한 2023 F 를 한 시즌 본다. 이 비대칭이
선택 가중의 전이를 깨는 것이라면, 수리(멤버 재학습·오염 행 제외)가 의미 있다.

비싼 수리 전에 싼 위치 진단 두 개:

  [A] 분해 — §7.194 저장 가중(RAW32·GRP10·LOCK·EXT195)의 전이 Δ(vs B1)를
      평가행 그룹별 기여로 쪼갠다:
        · game_type (R / F)                       ← F 분포 자체
        · 투수 오염 노출 share_preF               ← §7.48 의 잔차 축 (career 내
          pre-2023 F 투구 비율, s<eval_fold 기준)  asof 오염은 R 행에도 번진다
      영가설 = 균등 피해 (Δ_total × 질량몫). 초과분 CI 는 투수 클러스터 부트스트랩.
      ⚠️ 가중은 반대 fold 에서 선택된 것이므로 이 분해는 out-of-sample — 세그먼트
      분해의 in-sample성 문제([[segment-decomposition-also-measures-in-sampleness]])
      는 여기 해당 없음.

  [B] R-행 전용 선택 — full-fold greedy 의 fit 행을 R 로 제한하고 전이를 잰다.
      오염 유래 F 성분이 선택이 증폭하던 것이라면 R 전용 선택은 전이가 개선돼야
      한다. 그대로 음수면 실패는 F 와 무관 → 데이터 수리로 힐클라임 못 살린다.

판정 기준 (사전 등록)
  · [A] F 그룹 초과 기여 CI 가 0 을 음수(손실 집중) 쪽에서 배제하고 그 크기가
    전체 손실의 절반 이상 → 오염 채널 유력, 수리 실험 진행.
  · [B] R 전용 선택의 R-행 전이 Δ 가 원선택 대비 부호 개선 (음수 → 0 포함 이상)
    → 같은 결론. 둘 다 아니면 축 폐쇄 유지.

실행:  .venv/Scripts/python.exe -u src/hcfdiag.py 2>&1 | tee experiments/hcfdiag.log
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
import pandas as pd

import deploy29 as D
import hillclimb as H
import hillclimb2 as H2
from validation import bss, shift_to_prior

CACHE = D.CACHE
FOLDS = (2023, 2024)
B_BOOT = 1000
RNG_SEED = 42
MARKS = (5, 10, 20, 40, 80, 120)


# ── 그룹 정의 ──────────────────────────────────────────────────────────

def build_groups(df, s, codes, fold, vi, mm):
    """clean 평가행의 (game_type 그룹, 오염노출 그룹, 라벨들)."""
    gt = df["game_type"].to_numpy().astype(str)[vi][mm]
    g_gt = (gt == "F").astype(np.int64)                  # 0=R 1=F
    gt_names = ["R", "F"]

    # 투수 오염 노출: s<fold 이력 중 pre-2023 F 비율
    hist = s < fold
    n_pre = np.bincount(codes[hist], minlength=codes.max() + 1)
    contam = hist & (s < 2023) & (df["game_type"].to_numpy().astype(str) == "F")
    n_cf = np.bincount(codes[contam], minlength=codes.max() + 1)
    share = n_cf / np.maximum(n_pre, 1)

    c_eval = codes[vi][mm]
    sh = share[c_eval]
    rookie = n_pre[c_eval] == 0
    g_ex = np.full(len(sh), 0, dtype=np.int64)
    g_ex[rookie] = 4
    g_ex[~rookie & (sh > 0.00)] = 1
    g_ex[~rookie & (sh > 0.05)] = 2
    g_ex[~rookie & (sh > 0.15)] = 3
    ex_names = ["share=0", "0<sh≤.05", ".05<sh≤.15", "sh>.15", "rookie"]
    return (g_gt, gt_names), (g_ex, ex_names)


# ── 분해 기계 ──────────────────────────────────────────────────────────

def decompose(p_new, p_ref, yv, g, G, cl, n_id, rng=None, B=0):
    """Δ_total 을 그룹 기여 C_g 로 쪼갠다 (Σ C_g = Δ_total).

    C_g = 1e5·Σ_{i∈g}(e_ref−e_new)/(N·refvar).  초과 = C_g − Δ_total·m_g.
    B>0 이면 투수 클러스터 부트스트랩으로 초과분 CI.
    """
    refvar = yv.mean() * (1 - yv.mean())
    N = len(yv)
    diff = (p_ref - yv) ** 2 - (p_new - yv) ** 2
    key = cl.astype(np.int64) * G + g
    S = np.bincount(key, weights=diff, minlength=n_id * G).reshape(n_id, G)
    Ncg = np.bincount(key, minlength=n_id * G).reshape(n_id, G)
    C = 1e5 * S.sum(axis=0) / (N * refvar)
    m = Ncg.sum(axis=0) / N
    total = C.sum()
    excess = C - total * m

    ci = None
    if B and rng is not None:
        seen = np.where(Ncg.sum(axis=1) > 0)[0]
        draws = np.empty((B, G))
        for b in range(B):
            pick = rng.choice(seen, size=len(seen), replace=True)
            Sd, Nd = S[pick].sum(axis=0), Ncg[pick].sum(axis=0)
            Ntot = Nd.sum()
            Cd = 1e5 * Sd / (Ntot * refvar)
            draws[b] = Cd - Cd.sum() * (Nd / Ntot)
        ci = np.percentile(draws, [2.5, 97.5], axis=0)
    return C, m, excess, ci


def show(tag, names, C, m, excess, ci=None):
    tot = C.sum()
    print(f"  {tag}  Δ_total {tot:+8.2f}")
    for i, nm in enumerate(names):
        line = (f"    {nm:<10s} 기여 {C[i]:+8.2f}  질량 {m[i]:.3f}"
                f"  균등기대 {tot*m[i]:+7.2f}  초과 {excess[i]:+7.2f}")
        if ci is not None:
            line += f"  95%[{ci[0,i]:+.2f}, {ci[1,i]:+.2f}]"
        print(line)


def subset_delta(p_new, p_ref, yv, mask):
    """서브셋 자체 기준의 ΔBSS (서브셋 refvar 정규화)."""
    ys = yv[mask]
    rv = ys.mean() * (1 - ys.mean())
    d = ((p_ref[mask] - ys) ** 2 - (p_new[mask] - ys) ** 2).mean()
    return 1e5 * d / rv


# ── 메인 ───────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"§7.195 힐클라임 F 오염 채널 진단   지문 {D.KEY}")
    df, s, y = D.load_df()
    codes, _ = pd.factorize(df["pitcher_id"].to_numpy())
    ST = {f: H.make_state(df, s, y, f, t0) for f in FOLDS}

    GRP = {}
    for f in FOLDS:
        vi = np.where(s == f)[0]
        GRP[f] = build_groups(df, s, codes, f, vi, ST[f]["mask"])
        (g_gt, _), (g_ex, ex_names) = GRP[f]
        n = len(g_gt)
        print(f"  fold{f} clean {n}행  F {g_gt.sum()} ({g_gt.mean():.3f})  노출: "
              + "  ".join(f"{nm} {int((g_ex==i).sum())}"
                          for i, nm in enumerate(ex_names)))
    del df

    rng = np.random.default_rng(RNG_SEED)

    # 기준선 B1
    P_ref, S_ref = {}, {}
    for f in FOLDS:
        st = ST[f]
        p, sc = H.eval_w(st, st["Z"], H.baseline_w(st["fam"], "B1"))
        P_ref[f], S_ref[f] = p, sc
        print(f"  fold{f} B1 {sc:.2f}")

    # ── [A1] 파트 1 가중 (RAW32·GRP10·LOCK) ──────────────────────
    W1 = np.load(CACHE / "hillclimb_weights.npz", allow_pickle=True)
    print(f"\n[A] 전이 Δ(vs B1) 분해 — 가중은 반대 fold 에서 선택된 것 (out-of-sample)")
    for arm in ("RAW32", "GRP10", "LOCK"):
        for sf, ef in ((2023, 2024), (2024, 2023)):
            w = W1[f"{arm}_{sf}"]
            st = ST[ef]
            p_new, s_new = H.eval_w(st, st["Z"], w)
            (g_gt, gt_names), (g_ex, ex_names) = GRP[ef]
            head = arm == "LOCK" and sf == 2023
            print(f"\n [{arm}] sel={sf} → eval={ef}   Δ {s_new-S_ref[ef]:+.2f}")
            C, m, ex_, ci = decompose(p_new, P_ref[ef], st["yv"], g_gt, 2,
                                      st["cl"], st["n_id"],
                                      rng=rng, B=B_BOOT if head else 0)
            show("game_type", gt_names, C, m, ex_, ci)
            C, m, ex_, ci = decompose(p_new, P_ref[ef], st["yv"], g_ex, 5,
                                      st["cl"], st["n_id"],
                                      rng=rng, B=B_BOOT if head else 0)
            show("노출", ex_names, C, m, ex_, ci)

    # ── [A2] 파트 2 가중 (EXT195) ─────────────────────────────────
    W2 = np.load(CACHE / "hillclimb2_weights.npz", allow_pickle=True)
    extras = H2.find_extras()
    n_fold = {f: len(ST[f]["mask"]) for f in FOLDS}
    LIB = {}
    for f in FOLDS:
        st = ST[f]
        cols = [st["Z"][:, i][st["mask"]] for i in range(st["Z"].shape[1])]
        for name, fam, paths in extras:
            cols.append(H2.load_extra(paths[f], n_fold[f])[st["mask"]])
        LIB[f] = np.stack(cols, axis=1)
    assert LIB[2023].shape[1] == len(W2["w2023"]), \
        f"EXT 라이브러리 {LIB[2023].shape[1]} != 가중 {len(W2['w2023'])}"

    for sf, ef in ((2023, 2024), (2024, 2023)):
        w = W2[f"w{sf}"]
        st = ST[ef]
        p_new = shift_to_prior(
            1 / (1 + np.exp(-(LIB[ef] @ w + st["ax"]))), st["ybar"])
        s_new = bss(st["yv"], p_new)
        (g_gt, gt_names), (g_ex, ex_names) = GRP[ef]
        head = sf == 2023
        print(f"\n [EXT{LIB[ef].shape[1]}] sel={sf} → eval={ef}   Δ {s_new-S_ref[ef]:+.2f}")
        C, m, ex_, ci = decompose(p_new, P_ref[ef], st["yv"], g_gt, 2,
                                  st["cl"], st["n_id"],
                                  rng=rng, B=B_BOOT if head else 0)
        show("game_type", gt_names, C, m, ex_, ci)
        C, m, ex_, ci = decompose(p_new, P_ref[ef], st["yv"], g_ex, 5,
                                  st["cl"], st["n_id"],
                                  rng=rng, B=B_BOOT if head else 0)
        show("노출", ex_names, C, m, ex_, ci)

    # ── [B] R-행 전용 full-fold greedy (RAW32) ────────────────────
    print(f"\n[B] fit 행 제한 greedy — 선택 목적함수에서 F 를 뺀다  ({time.time()-t0:.0f}s)")
    for sf, ef in ((2023, 2024), (2024, 2023)):
        a, b = ST[sf], ST[ef]
        (g_gt_a, _), _ = GRP[sf]
        (g_gt_b, _), _ = GRP[ef]
        Za, Zb = a["Z"][a["mask"]], b["Z"][b["mask"]]
        for fit_tag, fmask in (("전체행", np.ones(len(g_gt_a), bool)),
                               ("R-전용", g_gt_a == 0)):
            hist, _ = H.greedy(Za[fmask], a["ax"][fmask], a["yv"][fmask],
                               np.arange(Za.shape[1]), max(MARKS))
            rows = {"in-fold Δ": [], "전이 Δ(전체)": [],
                    "전이 Δ(R행)": [], "전이 Δ(F행)": []}
            for k in MARKS:
                w = hist[k - 1] / k
                _, s_in = H.eval_w(a, a["Z"], w)
                p_tr, s_tr = H.eval_w(b, b["Z"], w)
                rows["in-fold Δ"].append(s_in - S_ref[sf])
                rows["전이 Δ(전체)"].append(s_tr - S_ref[ef])
                rows["전이 Δ(R행)"].append(
                    subset_delta(p_tr, P_ref[ef], b["yv"], g_gt_b == 0))
                rows["전이 Δ(F행)"].append(
                    subset_delta(p_tr, P_ref[ef], b["yv"], g_gt_b == 1))
            print(f"\n  sel={sf}→{ef} [{fit_tag}]  k:  "
                  + "".join(f"{k:>9d}" for k in MARKS))
            for nm, v in rows.items():
                print(f"    {nm:<11s}" + "".join(f"{d:>+9.2f}" for d in v))

    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
