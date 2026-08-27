"""§7.183 — GNN 프로브: 대결 그래프 전파 피처를 배포층 잔차에 묻는다.

GNN 제안의 분해 (MODEL.md §7.13 ①ID밀집표현 · §7.20 ②쌍상호작용은 기측정-기각):
    남은 유일한 신규 채널 = ③ **그래프를 타고 흐르는 이웃 정보**.
    GNN 1홉 메시지 패싱 = 이웃 값의 가중 평균이므로, 학습 없이 피처로 등가 계산이
    가능하다. 이게 죽으면 GNN은 상위비용일 뿐이다 (a fortiori).

후보 (fold의 **학습 시즌 행만**으로 구축, recency 가중 hl=0.5, k=50 스무딩):
    p1_opp   투수 1홉 — 상대해온 타자들의 피제구율(allow rate), 대결수 가중 LOO 평균.
             LOO(자기 대결 제외)가 주효과 분리다 — 안 빼면 투수 본인 실력이 섞인다.
    b1_opp   타자 1홉 — 상대해온 투수들의 제구율, 대결수 가중 LOO 평균.
             (배포의 gap_bsuccess 는 이것의 시즌내 차분판 — 잔차에는 그 몫이 이미 없다)
    p2_coh   투수 2홉 — 내 상대 타자들을 상대한 **다른** 투수들의 실력, 이중 가중 평균.
             GNN 고유의 "공유 이웃을 통한 유사도" 채널.

위약 (기하 보존·정체 파괴 — [[placebo-must-keep-shared-terms]]):
    *_plb    같은 대결 가중치 구조에서 엔티티 값만 fold별 시드 순열.

죽은-축 영가설 ([[use-a-known-dead-axis-as-the-null]]):
    ref_own  투수 본인 가중 성공률 — 실재하는 신호지만 모델이 이미 샀다.
             후보가 이 프로파일과 구분되지 않으면 "실재하되 기구매"다.

판정 (사전 등록):
    ① 양 fold 부호 일치 & min|t| ≥ 3 (§7.180 스크린에서 배포 78피처는 이 기준 0개)
    ② 후보 |t| 가 양 fold 모두 위약 max|t| 를 초과
    ③ ①② 통과 시에만 stage B — β 를 fold2023 에서 적합해 fold2024 에 전이,
       투수 클러스터 부트스트랩 B=4000 으로 ΔBSS 구간. 미통과면 채널 폐쇄.

실행:  .venv/Scripts/python.exe -u src/gnnprobe.py 2>&1 | tee experiments/gnnprobe.log
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

import foldcache as _fc
import resprobe as R
import deploy29 as D
from config import RECENCY_HALF_LIFE, TARGET_COL
from data import recency_weight
from validation import bss, shift_to_prior

K_SMOOTH = 50.0          # 가중 단위. SMOOTH_K(200)는 레벨용 — 차분이 아니므로 절반 이하
B_BOOT = 4000
BETAS = np.linspace(-0.2, 0.2, 81)      # 표준화 컬럼의 로짓 계수 — t~2.5 규모면 |β*|≈0.01


def graph_tables(df, s, fold, rng):
    """fold 학습 행만으로 엔티티별 그래프 전파 값을 만든다. 반환: dict[name] -> (index, values)"""
    mtr, _ = _fc.SPLITS[fold](s, df.game_month.to_numpy())
    tr = df[mtr]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    y = tr[TARGET_COL].to_numpy(dtype=np.float64)
    g = (pd.DataFrame({"p": tr.pitcher_id.to_numpy(), "b": tr.batter_id.to_numpy(),
                       "sw": w * y, "nw": w})
         .groupby(["p", "b"], as_index=False).sum())
    ybar = g.sw.sum() / g.nw.sum()
    P = g.groupby("p")[["sw", "nw"]].sum()
    B = g.groupby("b")[["sw", "nw"]].sum()
    print(f"    학습 {len(tr):,}행 → 대결쌍 {len(g):,}  투수 {len(P):,}  타자 {len(B):,}"
          f"  ȳ_w={ybar:.4f}")

    sw, nw = g.sw.to_numpy(), g.nw.to_numpy()
    SWb, NWb = B.sw.reindex(g.b).to_numpy(), B.nw.reindex(g.b).to_numpy()
    SWp, NWp = P.sw.reindex(g.p).to_numpy(), P.nw.reindex(g.p).to_numpy()

    # ── 1홉 LOO (자기 대결 제외 + k 스무딩) ──
    al_loo = (SWb - sw + K_SMOOTH * ybar) / (NWb - nw + K_SMOOTH)   # 타자 피제구율^{-p}
    sk_loo = (SWp - sw + K_SMOOTH * ybar) / (NWp - nw + K_SMOOTH)   # 투수 제구율^{-b}

    def agg(keys, vals, wts):
        num = pd.Series(vals * wts).groupby(keys).sum()
        den = pd.Series(wts).groupby(keys).sum()
        return num / den

    p1 = agg(g.p.to_numpy(), al_loo, nw)
    b1 = agg(g.b.to_numpy(), sk_loo, nw)

    # ── 2홉: 타자별 "상대한 투수 실력" 집계에서 자기(p) 항 제거 ──
    sk_p = ((P.sw + K_SMOOTH * ybar) / (P.nw + K_SMOOTH))           # 비LOO 투수 실력
    sk_pair = sk_p.reindex(g.p).to_numpy()
    Tb = pd.Series(nw * sk_pair).groupby(g.b.to_numpy()).sum()      # Σ_p nw·sk_p
    Tb_pair, Wb_pair = Tb.reindex(g.b).to_numpy(), NWb
    den2 = Wb_pair - nw
    b1x = np.where(den2 > 1e-9, (Tb_pair - nw * sk_pair) / np.maximum(den2, 1e-9),
                   np.nan)                                          # b1^{-p}(b)
    m2 = np.isfinite(b1x)
    p2 = agg(g.p.to_numpy()[m2], b1x[m2], nw[m2])

    # ── 위약: 값 순열 (가중 기하 보존) ──
    al_b = ((B.sw + K_SMOOTH * ybar) / (B.nw + K_SMOOTH))
    al_perm = pd.Series(rng.permutation(al_b.to_numpy()), index=al_b.index)
    sk_perm = pd.Series(rng.permutation(sk_p.to_numpy()), index=sk_p.index)
    b1_all = Tb / B.nw.reindex(Tb.index)
    b1_perm = pd.Series(rng.permutation(b1_all.to_numpy()), index=b1_all.index)
    p1_plb = agg(g.p.to_numpy(), al_perm.reindex(g.b).to_numpy(), nw)
    b1_plb = agg(g.b.to_numpy(), sk_perm.reindex(g.p).to_numpy(), nw)
    p2_plb = agg(g.p.to_numpy(), b1_perm.reindex(g.b).to_numpy(), nw)

    return {"p1_opp": ("p", p1), "b1_opp": ("b", b1), "p2_coh": ("p", p2),
            "p1_plb": ("p", p1_plb), "b1_plb": ("b", b1_plb), "p2_plb": ("p", p2_plb),
            "ref_own": ("p", sk_p)}


def to_rows(tabs, dva):
    """엔티티 표를 검증 행으로 사상. 미등장 엔티티 → NaN (probe 가 기여 0 처리)."""
    pid, bid = dva.pitcher_id.to_numpy(), dva.batter_id.to_numpy()
    X = pd.DataFrame(index=range(len(dva)))
    for name, (axis, tab) in tabs.items():
        keys = pid if axis == "p" else bid
        X[name] = pd.Series(keys).map(tab).to_numpy(dtype=np.float64)
    return X


def _std_col(Xf, c):
    v = Xf[c].to_numpy(dtype=np.float64)
    m = np.isfinite(v)
    return np.where(m, (v - v[m].mean()) / v[m].std(), 0.0)


def _fit_beta(a, z):
    best, bb = None, 0.0
    za = np.log(a["p"] / (1 - a["p"]))
    for be in BETAS:
        p = shift_to_prior(1 / (1 + np.exp(-(za + be * z))), a["ybar"])
        sc = bss(a["yv"], p)
        if best is None or sc > best:
            best, bb = sc, be
    return bb, best


def stage_b(F, Xc):
    """β 를 src fold 잔차에서 적합 → dst fold 에 전이. 위약 쌍둥이에 같은 절차.

    2023→2024 가 배포 방향, 2024→2023 은 지속성 대조 (§7.180 과 동일 규약).
    쌍체 Δ(후보 − 위약)는 같은 클러스터 draw 위에서 계산한다."""
    print("\n" + "=" * 96)
    print("B. 전이 검정 — β fit@src → apply@dst (표준화 컬럼, NaN→0, 로짓 가산)")
    print("=" * 96)
    for f in D.FOLDS:
        cc = np.corrcoef(_std_col(Xc[f], "p1_opp"), _std_col(Xc[f], "b1_opp"))[0, 1]
        print(f"  corr(p1_opp, b1_opp) fold{f} = {cc:.3f}")
    pairs = [("p1_opp", "p1_plb"), ("b1_opp", "b1_plb"), ("p2_coh", "p2_plb")]
    for src, dst in ((2023, 2024), (2024, 2023)):
        a, b = F[src], F[dst]
        zb_dst = np.log(b["p"] / (1 - b["p"]))
        arms, names = [], []
        info = {}
        for cand, plb in pairs:
            for c in (cand, plb):
                bb, best = _fit_beta(a, _std_col(Xc[src], c))
                p1 = shift_to_prior(1 / (1 + np.exp(-(zb_dst + bb * _std_col(Xc[dst], c)))),
                                    b["ybar"])
                arms.append(p1)
                names.append(c)
                info[c] = (bb, best - a["base"])
        point, draws = R.boot_delta(b["yv"], b["p"], arms, b["cl"], b["n_id"],
                                    np.random.default_rng(42), B=B_BOOT)
        print(f"\n  ── {src} → {dst} ──")
        print(f"  {'팔':<10s}{'β*':>7s}{'in-foldΔ':>10s}{'Δdst':>9s}{'95% 구간':>20s}"
              f"{'P(Δ>0)':>9s}")
        for j, c in enumerate(names):
            lo, hi = np.percentile(draws[:, j], [2.5, 97.5])
            print(f"  {c:<10s}{info[c][0]:>+7.3f}{info[c][1]:>+10.2f}{point[j]:>+9.2f}"
                  f"   [{lo:>+7.2f}, {hi:>+7.2f}]{np.mean(draws[:, j] > 0):>9.3f}")
        for cand, plb in pairs:
            jc, jp = names.index(cand), names.index(plb)
            dp = draws[:, jc] - draws[:, jp]
            lo, hi = np.percentile(dp, [2.5, 97.5])
            print(f"  Δ({cand} − 위약) 쌍체  {point[jc]-point[jp]:>+7.2f}"
                  f"   [{lo:>+7.2f}, {hi:>+7.2f}]  P {np.mean(dp > 0):.3f}")


def main():
    t0 = time.time()
    df, s, y = D.load_df()
    F, Xc = {}, {}
    for fold in D.FOLDS:
        print(f"\n=== fold {fold} ===")
        F[fold] = R.deploy_base(df, s, y, fold)
        rng = np.random.default_rng(1000 + fold)
        tabs = graph_tables(df, s, fold, rng)
        Xc[fold] = to_rows(tabs, F[fold]["st"]["d"])
        cov = Xc[fold].notna().mean()
        print("    coverage: " + "  ".join(f"{c} {cov[c]:.1%}" for c in Xc[fold].columns))
        print(f"    ({time.time()-t0:.0f}s)")
    del df

    print("\n" + "=" * 96)
    print("A. 잔차 프로브 — 그래프 전파 컬럼 vs r = y − p_dep, 투수 클러스터 강건 t")
    print("   (§7.180 기준선: 배포 78피처 중 양fold 부호일치 |t|>2 는 0개)")
    print("=" * 96)
    T = {f: R.probe(Xc[f], F[f]["yv"] - F[f]["p"], F[f]["cl"], F[f]["n_id"])
         for f in D.FOLDS}
    cols = list(Xc[2023].columns)
    print(f"  {'컬럼':<10s}{'t_f2023':>9s}{'t_f2024':>9s}   비고")
    note = {"p1_plb": "위약(p1)", "b1_plb": "위약(b1)", "p2_plb": "위약(p2)",
            "ref_own": "죽은 축(기구매)"}
    for c in cols:
        print(f"  {c:<10s}{T[2023][c]:>9.2f}{T[2024][c]:>9.2f}   {note.get(c, '후보')}")

    plb_max = {f: max(abs(T[f][c]) for c in ("p1_plb", "b1_plb", "p2_plb"))
               for f in D.FOLDS}
    print(f"\n  위약 max|t|:  f2023 {plb_max[2023]:.2f}   f2024 {plb_max[2024]:.2f}")
    alive = []
    for c in ("p1_opp", "b1_opp", "p2_coh"):
        t23, t24 = T[2023][c], T[2024][c]
        ok1 = np.sign(t23) == np.sign(t24) and min(abs(t23), abs(t24)) >= 3
        ok2 = abs(t23) > plb_max[2023] and abs(t24) > plb_max[2024]
        v = "생존" if ok1 and ok2 else "기각"
        print(f"  {c}: ①부호일치·min|t|≥3 {'통과' if ok1 else '미달'}"
              f"  ②위약 초과 {'통과' if ok2 else '미달'}  → {v}")
        if ok1 and ok2:
            alive.append(c)

    if alive:
        stage_b(F, Xc)
    elif "--exploratory" in sys.argv:
        print("\n→ 사전 등록 관문 미달. 아래 stage B 는 **참고용(증거 아님)** — "
              "효과 크기 스케일 확인 목적.")
        stage_b(F, Xc)
    else:
        print("\n→ stage B 생략 — 전 후보 기각. 그래프 전파 채널 폐쇄 "
              "(GNN 은 a fortiori 기각)")
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
