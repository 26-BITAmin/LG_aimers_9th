"""§7.190 — NVIDIA Grandmasters Playbook 대조: **대량 피처 생성** (전 쌍 상호작용) 잔차 스크린

배경 (사용자: playbook 7대 기법 중 "대량 feature 생성 + pseudo-labeling 을 진행해줘")
    playbook 항목별 저장소 대조는 MODEL.md §7.190 서두에 있다. 피처 축에서
    아직 안 밟은 공간은 **쌍 조합**뿐이다:
      - 단독 78개 잔차 스크린: §7.91 · §7.180 — 부호 일치 |t|>2 = 0개
      - 제곱/멱: §7.150a·§7.151a 기각 (트리에 제곱은 거의 무연산)
      - 곱 상호작용: §7.120a 단일 쌍 1개만 측정 (전수는 처음)
      - 차분: form_*/gap_* 수작업 축만 존재. 전 쌍 차분은 처음

무엇을 재나 (Stage 1 — 학습 0회, 배포층 잔차 프로브)
    산술 가능 70피처의 전 쌍 2,415 × {PROD(z-곱, 중심화 곱과 t 동치), DIFF(z-차)}
    = 4,830 후보 + 단독 70(부모 기준선) + **위약 뱅크 1,000**(b 행 순열 후보).
    프로브 = resprobe.probe 와 같은 투수 클러스터 강건 score t,
    잔차 r = y − p_dep (deploy29 배포 등가층 + cat10 + 1.05 스케일 + prior shift).

    스크린 통계  S = min(|t_f2023|, |t_f2024|) · 1[부호 일치]
    두 fold 는 독립 시즌이므로 S 는 관계의 시즌 간 지속을 재는 통계다
    ([[check-sign-agreement-on-the-relationship-not-the-metric]],
     [[holdout-gain-conflates-signal-with-fold-specific-error]]).

사전 등록 게이트 (결과 보기 전 고정)
    ① S ≥ 4                      (가우시안 널에서 4,830개 중 기대 오검출 ~1e-5)
    ② S > 위약 뱅크 1,000 개의 max  (경험적 널 — [[use-a-known-dead-axis-as-the-null]])
    ③ S > 両부모의 S              (부모 단독으로 설명되면 새 채널이 아니다)
    통과 0개 → "대량 피처 축 폐쇄" 로 §7.190 에 기록하고 닫는다.
    통과 ≥1개 → Stage 2 (massfeat2.py — gnncol 하네스, 12부스터 학습 경로,
    MASS vs 순열 PLB, 블렌드 Δ 양 fold P≥0.9)로만 채택 논의. 컬럼 경로의
    LB 전례(§7.56 4전 4패, §7.187)는 통과해도 장부에 명기한다.

한계 (명기)
    프로브는 기존 피처와의 상호작용으로만 발현되는 신호를 못 본다
    ([[test-new-feature-against-model-residual]]) — 그러나 후보 자신이 이미
    상호작용(쌍)이므로 이 스크린이 정확히 그 채널을 겨냥한다. 3차 이상은 안 본다.

실행:  .venv/Scripts/python.exe -u src/massfeat.py 2>&1 | tee experiments/massfeat.log
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
import resprobe as RP

FOLDS = D.FOLDS
S_GATE = 4.0
N_PLACEBO = 1000
BLOCK = 64
EXCLUDE = {"pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
           "base_state", "count_state", "hand_matchup", "top_bottom"}


def fold_state(df, s, y, fold):
    b = RP.deploy_base(df, s, y, fold)
    X = RP.fold_X(df, fold)
    cols = [c for c in X.columns if c not in EXCLUDE]
    # 클러스터 정렬 (reduceat 용)
    order = np.argsort(b["cl"], kind="stable")
    bnd = np.searchsorted(b["cl"][order],
                          np.unique(b["cl"][order]))
    V = X[cols].to_numpy(dtype=np.float64)[order]
    mu = np.nanmean(V, axis=0)
    sd = np.nanstd(V, axis=0)
    sd[sd == 0] = 1.0
    Z = (V - mu) / sd                       # NaN 보존
    r = (b["yv"] - b["p"])[order]
    return dict(Z=Z, r=r, bnd=bnd, cols=cols, base=b["base"], n=len(r))


def probe_block(C, r, bnd):
    """C (n × m, NaN 허용, 클러스터 정렬됨) 의 클러스터 강건 score t (m,)."""
    M = np.isfinite(C)
    cnt = M.sum(axis=0).astype(np.float64)
    cnt[cnt == 0] = 1.0
    Cz = np.where(M, C, 0.0)
    mu = Cz.sum(axis=0) / cnt
    U = np.where(M, C - mu, 0.0) * r[:, None]
    num = U.sum(axis=0)
    Sg = np.add.reduceat(U, bnd, axis=0)
    den = np.sqrt((Sg * Sg).sum(axis=0))
    den[den == 0] = np.inf
    return num / den


def probe_pairs(st, I, J, op):
    """쌍 목록 (I, J) 의 op 후보 t 를 블록으로 계산."""
    Z, r, bnd = st["Z"], st["r"], st["bnd"]
    out = np.empty(len(I))
    for k in range(0, len(I), BLOCK):
        i, j = I[k:k + BLOCK], J[k:k + BLOCK]
        C = Z[:, i] * Z[:, j] if op == "PROD" else Z[:, i] - Z[:, j]
        out[k:k + BLOCK] = probe_block(C, r, bnd)
    return out


def main():
    t0 = time.time()
    df, s, y = D.load_df()
    ST = {f: fold_state(df, s, y, f) for f in FOLDS}
    del df
    cols = ST[FOLDS[0]]["cols"]
    assert cols == ST[FOLDS[1]]["cols"]
    p = len(cols)
    I, J = np.triu_indices(p, k=1)
    print(f"산술 피처 {p}개 → 쌍 {len(I):,} × 2연산 = {2*len(I):,} 후보  "
          f"(n: {ST[2023]['n']:,} / {ST[2024]['n']:,})  ({time.time()-t0:.0f}s)")

    # ── 부모(단독) 기준선 ────────────────────────────────────────
    T1 = {f: probe_block(ST[f]["Z"], ST[f]["r"], ST[f]["bnd"]) for f in FOLDS}
    s_par = np.where(np.sign(T1[2023]) == np.sign(T1[2024]),
                     np.minimum(np.abs(T1[2023]), np.abs(T1[2024])), 0.0)
    print(f"단독 70 재확인 — S≥2: {int((s_par >= 2).sum())}개 "
          f"(§7.180 의 0개와 정합해야 한다. max S = {s_par.max():.2f})")

    # ── 전 쌍 후보 ───────────────────────────────────────────────
    T = {}
    for op in ("PROD", "DIFF"):
        for f in FOLDS:
            T[(op, f)] = probe_pairs(ST[f], I, J, op)
        print(f"{op} 프로브 완료 ({time.time()-t0:.0f}s)")

    # ── 위약 뱅크 — b 열을 행 순열한 뒤 같은 연산 ───────────────
    rng = np.random.default_rng(20260828)
    pi = rng.integers(0, p, N_PLACEBO)
    pj = rng.integers(0, p, N_PLACEBO)
    pop = rng.integers(0, 2, N_PLACEBO)          # 0=PROD 1=DIFF
    TP = {}
    for f in FOLDS:
        Z, r, bnd = ST[f]["Z"], ST[f]["r"], ST[f]["bnd"]
        perm = rng.permutation(len(r))
        out = np.empty(N_PLACEBO)
        for k in range(0, N_PLACEBO, BLOCK):
            i, j, o = pi[k:k+BLOCK], pj[k:k+BLOCK], pop[k:k+BLOCK]
            Zb = Z[perm][:, j]
            C = np.where(o[None, :] == 0, Z[:, i] * Zb, Z[:, i] - Zb)
            out[k:k+BLOCK] = probe_block(C, r, bnd)
        TP[f] = out
    s_plb = np.where(np.sign(TP[2023]) == np.sign(TP[2024]),
                     np.minimum(np.abs(TP[2023]), np.abs(TP[2024])), 0.0)
    print(f"위약 뱅크 {N_PLACEBO} — S max {s_plb.max():.2f} · "
          f"99% {np.percentile(s_plb, 99):.2f} · S≥2 {int((s_plb >= 2).sum())}개"
          f"  ({time.time()-t0:.0f}s)")

    # ── 스크린 통계 + 게이트 (전 후보 집계) ──────────────────────
    rows, n_pass = [], 0
    for op in ("PROD", "DIFF"):
        t23, t24 = T[(op, 2023)], T[(op, 2024)]
        agree = np.sign(t23) == np.sign(t24)
        S = np.where(agree, np.minimum(np.abs(t23), np.abs(t24)), 0.0)
        ok = ((S >= S_GATE) & (S > s_plb.max())
              & (S > s_par[I]) & (S > s_par[J]))
        n_pass += int(ok.sum())
        for k in np.argsort(-S)[:40]:
            rows.append(dict(op=op, a=cols[I[k]], b=cols[J[k]],
                             t23=t23[k], t24=t24[k], S=S[k],
                             Sa=s_par[I[k]], Sb=s_par[J[k]], ok=bool(ok[k])))
    rows.sort(key=lambda r: -r["S"])

    print("\n" + "=" * 110)
    print("상위 30 (S = 부호일치 · min|t|)   게이트: ① S≥4  ② S>위약max(%.2f)  ③ S>両부모S"
          % s_plb.max())
    print("=" * 110)
    print(f"  {'op':<5s}{'a':<38s}{'b':<38s}{'t23':>7s}{'t24':>7s}{'S':>6s}"
          f"{'Sa':>6s}{'Sb':>6s}  게이트")
    for r in rows[:30]:
        print(f"  {r['op']:<5s}{r['a']:<38s}{r['b']:<38s}{r['t23']:>7.2f}"
              f"{r['t24']:>7.2f}{r['S']:>6.2f}{r['Sa']:>6.2f}{r['Sb']:>6.2f}"
              f"  {'✅ 통과' if r['ok'] else ''}")

    for op in ("PROD", "DIFF"):
        t23, t24 = T[(op, 2023)], T[(op, 2024)]
        agree = np.sign(t23) == np.sign(t24)
        S = np.where(agree, np.minimum(np.abs(t23), np.abs(t24)), 0.0)
        print(f"  {op}: S≥2 {int((S >= 2).sum())}개 · S≥3 {int((S >= 3).sum())}개 · "
              f"S≥4 {int((S >= 4).sum())}개   (4,830 중 위약 뱅크 S≥2 비율 "
              f"{(s_plb >= 2).mean()*100:.1f}% → 기대 {int((s_plb >= 2).mean()*len(I)):d}개/연산)")

    print("\n판정 (사전 등록): 게이트 ①②③ 동시 통과 %d개 → %s" % (
        n_pass,
        "Stage 2 (massfeat2.py 학습 경로) 진행" if n_pass
        else "대량 피처(쌍 조합) 축 폐쇄 — §7.190 기록"))
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
