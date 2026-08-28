"""§7.190b — 사용자 반문: "블렌드(MLP 75%) 때문에 전멸한 것일 수 있다 — LGBM 단독으로 확인"

기전 (사전 등록 해석 틀 — 결과 보기 전)
    z_blend = 0.25·z_L + 0.75·z_M. LGBM 만 놓치고 MLP 가 잡는 신호 βc 는
    블렌드 잔차에 0.25β 로만 남는다 → §7.190 프로브는 그 부류를 t 기준 ~4배
    희석해서 봤다 (S≥4 게이트 = LGBM 층 t≈16 요구). 반대로 **두 팔이 다 놓치는**
    신호는 희석 없이 보였고 없었다 (max S 2.95). 따라서:
      - L 층에서 새로 나오는 히트 ≈ "LGBM 은 놓치고 MLP 는 잡는" 신호.
      - 그 컬럼을 LGBM 에 넣었을 때 **블렌드** 이득의 1차 상한은 (0.25)² 배.
      - 단 트리의 비선형 활용(기존 피처와 3차 상호작용)은 프로브 밖 —
        승격 후보의 최종 판정은 학습 경로(§7.187 gnncol 하네스, 블렌드 Δ)다.

무엇을 재나
    §7.190 과 같은 4,830 쌍 후보 + 단독 70 + 위약 뱅크 1,000 을 세 층 잔차에 프로브:
      L  y − shift_to_prior(σ(z_lgb))   z_lgb = initdep 팔 (배포 init_score, 20부스터)
      M  y − shift_to_prior(σ(z_mlp))   대조 — 어느 팔이 무엇을 놓치는지 지도
      B  §7.190 블렌드층 (동일 코드 재계산 — 수치 대조용 앵커)

승격 게이트 (L 층, 사전 등록): ① S_L ≥ 4  ② S_L > 위약(L) max  ③ S_L > 両부모 S_L
    통과 ≥1 → massfeat3 (학습 경로: MASS vs 순열 PLB, 판정은 블렌드 Δ 양 fold P≥0.9)
    전멸   → §7.190 폐쇄가 층 분해로도 확정

실행:  .venv/Scripts/python.exe -u src/massfeat2.py 2>&1 | tee experiments/massfeat2.log
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
import resprobe as RP
from massfeat import BLOCK, EXCLUDE, N_PLACEBO, S_GATE, probe_block, probe_pairs
from validation import bss, shift_to_prior

FOLDS = D.FOLDS
LAYERS = ("L", "M", "B")


def fold_states(df, s, y, fold):
    """세 층의 (Z 공유, r 층별) 상태. §7.190 fold_state 와 같은 정렬·표준화."""
    b = RP.deploy_base(df, s, y, fold)
    X = RP.fold_X(df, fold)
    cols = [c for c in X.columns if c not in EXCLUDE]
    order = np.argsort(b["cl"], kind="stable")
    bnd = np.searchsorted(b["cl"][order], np.unique(b["cl"][order]))
    V = X[cols].to_numpy(dtype=np.float64)[order]
    mu = np.nanmean(V, axis=0)
    sd = np.nanstd(V, axis=0)
    sd[sd == 0] = 1.0
    Z = (V - mu) / sd
    yv = b["yv"]

    vi = np.where(s == fold)[0]
    z_lgb = np.load(D.CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
    with np.load(D.CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
        z_mlp = d0["z_mlp"]
    assert len(z_lgb) == len(vi) == len(z_mlp) == len(yv)

    P = {"L": shift_to_prior(1 / (1 + np.exp(-z_lgb)), yv.mean()),
         "M": shift_to_prior(1 / (1 + np.exp(-z_mlp)), yv.mean()),
         "B": b["p"]}
    st = {}
    for lay, p in P.items():
        st[lay] = dict(Z=Z, r=(yv - p)[order], bnd=bnd, cols=cols,
                       base=bss(yv, p))
    print(f"  fold {fold} 층별 BSS  L {st['L']['base']:.1f}  "
          f"M {st['M']['base']:.1f}  B {st['B']['base']:.1f} (§7.190 앵커 "
          f"{'812.6' if fold == 2023 else '929.9'})")
    return st


def placebo_bank(ST, lay, p_feat, rng_seed=20260828):
    """§7.190 과 같은 구성의 위약 뱅크 (층별 r)."""
    rng = np.random.default_rng(rng_seed)
    pi = rng.integers(0, p_feat, N_PLACEBO)
    pj = rng.integers(0, p_feat, N_PLACEBO)
    pop = rng.integers(0, 2, N_PLACEBO)
    TP = {}
    for f in FOLDS:
        s0 = ST[f][lay]
        Z, r, bnd = s0["Z"], s0["r"], s0["bnd"]
        perm = rng.permutation(len(r))
        out = np.empty(N_PLACEBO)
        for k in range(0, N_PLACEBO, BLOCK):
            i, j, o = pi[k:k+BLOCK], pj[k:k+BLOCK], pop[k:k+BLOCK]
            Zb = Z[perm][:, j]
            C = np.where(o[None, :] == 0, Z[:, i] * Zb, Z[:, i] - Zb)
            out[k:k+BLOCK] = probe_block(C, r, bnd)
        TP[f] = out
    return np.where(np.sign(TP[2023]) == np.sign(TP[2024]),
                    np.minimum(np.abs(TP[2023]), np.abs(TP[2024])), 0.0)


def main():
    t0 = time.time()
    df, s, y = D.load_df()
    ST = {f: fold_states(df, s, y, f) for f in FOLDS}
    del df
    cols = ST[FOLDS[0]]["L"]["cols"]
    p_feat = len(cols)
    I, J = np.triu_indices(p_feat, k=1)
    print(f"산술 {p_feat}피처 · 쌍 {len(I):,} × 2연산 · 층 {LAYERS}  ({time.time()-t0:.0f}s)")

    S_all, T_all, s_par, s_plb = {}, {}, {}, {}
    for lay in LAYERS:
        T1 = {f: probe_block(ST[f][lay]["Z"], ST[f][lay]["r"], ST[f][lay]["bnd"])
              for f in FOLDS}
        s_par[lay] = np.where(np.sign(T1[2023]) == np.sign(T1[2024]),
                              np.minimum(np.abs(T1[2023]), np.abs(T1[2024])), 0.0)
        for op in ("PROD", "DIFF"):
            T = {f: probe_pairs(ST[f][lay], I, J, op) for f in FOLDS}
            T_all[(lay, op)] = T
            S_all[(lay, op)] = np.where(
                np.sign(T[2023]) == np.sign(T[2024]),
                np.minimum(np.abs(T[2023]), np.abs(T[2024])), 0.0)
        s_plb[lay] = placebo_bank(ST, lay, p_feat)
        print(f"[{lay}] 프로브 완료 — 단독 S≥2 {int((s_par[lay] >= 2).sum())}개 "
              f"(max {s_par[lay].max():.2f}) · 위약 max {s_plb[lay].max():.2f} · "
              f"99% {np.percentile(s_plb[lay], 99):.2f}  ({time.time()-t0:.0f}s)")

    # ── 층별 집계 + L 층 게이트 ──────────────────────────────────
    print("\n" + "=" * 112)
    print("층별 집계 (S = 부호일치·min|t|)  — B 층은 §7.190 재계산 앵커")
    print("=" * 112)
    print(f"  {'층':<3s}{'op':<6s}{'S≥2':>6s}{'S≥3':>6s}{'S≥4':>6s}"
          f"{'maxS':>8s}{'위약max':>9s}{'위약S≥2%':>10s}")
    for lay in LAYERS:
        for op in ("PROD", "DIFF"):
            S = S_all[(lay, op)]
            print(f"  {lay:<3s}{op:<6s}{int((S>=2).sum()):>6d}{int((S>=3).sum()):>6d}"
                  f"{int((S>=4).sum()):>6d}{S.max():>8.2f}{s_plb[lay].max():>9.2f}"
                  f"{(s_plb[lay]>=2).mean()*100:>9.1f}%")

    n_pass = 0
    rows = []
    for op in ("PROD", "DIFF"):
        S = S_all[("L", op)]
        ok = ((S >= S_GATE) & (S > s_plb["L"].max())
              & (S > s_par["L"][I]) & (S > s_par["L"][J]))
        n_pass += int(ok.sum())
        for k in np.argsort(-S)[:25]:
            rows.append(dict(op=op, a=cols[I[k]], b=cols[J[k]],
                             tL23=T_all[("L", op)][2023][k],
                             tL24=T_all[("L", op)][2024][k],
                             SL=S[k], SB=S_all[("B", op)][k],
                             SM=S_all[("M", op)][k],
                             Sa=s_par["L"][I[k]], Sb=s_par["L"][J[k]],
                             ok=bool(ok[k])))
    rows.sort(key=lambda r: -r["SL"])
    print("\nL(LGBM 단독) 층 상위 25 — 게이트: ① S_L≥4 ② >위약max(%.2f) ③ >両부모"
          % s_plb["L"].max())
    print(f"  {'op':<5s}{'a':<36s}{'b':<36s}{'tL23':>7s}{'tL24':>7s}"
          f"{'S_L':>6s}{'S_M':>6s}{'S_B':>6s}{'Sa':>5s}{'Sb':>5s}  게이트")
    for r in rows[:25]:
        print(f"  {r['op']:<5s}{r['a']:<36s}{r['b']:<36s}{r['tL23']:>7.2f}"
              f"{r['tL24']:>7.2f}{r['SL']:>6.2f}{r['SM']:>6.2f}{r['SB']:>6.2f}"
              f"{r['Sa']:>5.2f}{r['Sb']:>5.2f}  {'✅' if r['ok'] else ''}")

    # 단독(부모) 층별 대조 상위 — 어느 팔이 무엇을 놓치는지 지도
    print("\n단독 70 층별 대조 (S_L 상위 10)")
    top = np.argsort(-s_par["L"])[:10]
    print(f"  {'피처':<38s}{'S_L':>7s}{'S_M':>7s}{'S_B':>7s}")
    for k in top:
        print(f"  {cols[k]:<38s}{s_par['L'][k]:>7.2f}{s_par['M'][k]:>7.2f}"
              f"{s_par['B'][k]:>7.2f}")

    print("\n판정 (사전 등록): L 층 게이트 ①②③ 통과 %d개 → %s" % (
        n_pass,
        "massfeat3 (학습 경로 MASS vs PLB, 판정=블렌드 Δ) 진행"
        if n_pass else "층 분해로도 전멸 — §7.190 폐쇄 확정"))
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
