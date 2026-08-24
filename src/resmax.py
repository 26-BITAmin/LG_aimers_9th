"""RES_max 장부 — 블록별 한계 신호 지도 (§7.160). fit 0회, LB 비용 0.

오늘 닫힌 다섯(전역 기울기 · 버킷 기울기 · 버킷 레벨 · 계층적 μ · distillation)이
**전부 출력단·라벨단 조작**이었다. 남은 가설은 하나 — "현 피처셋의 정보 천장에
이미 근접했다". 이 장부가 그것을 숫자로 판정한다.

⚠️ **이 측정은 시즌 외삽을 하나도 하지 않는다.** 지도이지 베팅이 아니라서 실패
모드가 슬롯 손실이 아니라 우선순위 오판이고, §7.158 류의 추정량 리스크가 구조적으로
낮다. 남는 가정은 노이즈 모형 하나뿐이고 그것을 부트스트랩으로 대체한다.

### 추정량 — 3열 적률법

블록 `g` 마다 holdout 에서 세 분산을 잰다. `w_g` = 행 점유율,
`ȳ_g` / `f̄_g` / `ē_g` = 그룹별 라벨 / 배포 예측(p_dep) / 잔차(y−f) 평균.

    A 천장   Var_w(ȳ_g) − 노이즈    `g` 만 아는 오라클의 RES (한계 신호 총량)
    B 표현   Var_w(f̄_g) − 노이즈    모델이 `g` 축으로 이미 내보내는 RES
    C 헤드룸 Var_w(ē_g) − 노이즈    ★ **판정 열** — `g` 축에 남은 미회수 신호

**C 가 1차 판정 열인 이유**: `A−B` 는 정렬 가정 없이는 헤드룸이 아니다 (모델이 `g`
변동을 다른 축으로 표현할 수 있다). `C` 는 "그룹 상수 보정이 지금 건질 수 있는 양"을
**직접** 잰다. §7.155 가 특정 그룹들에서 한 일의 전 블록 일반화다.
보고 단위는 `C/UNC × 1e5` (BSS 환산), `UNC = ȳ(1−ȳ)`.

### ★ 노이즈 항 — 이항 공식이 아니라 클러스터 부트스트랩

`ē_g = μ_g + ε_g`, `Var(ε_g) = v_g` 로 놓으면 정확히

    E[Σ_g w_g(ē_g − ē)²] = V_true + Σ_g w_g(1−w_g)·v_g

이다. 이항 공식 `v_g = r_g(1−r_g)/(n_g−1)` 은 **그룹 내 독립 가정** 위에 서 있고
등판 내 상관이 그걸 깨서 천장을 과대추정한다. 여기서는 `v_g` 를 **그룹 내 경기
단위 재표집**(B=2000)으로 추정해 그 가정을 제거한다. 계수 `w_g(1−w_g)` 는 유지.
세 열에 **같은 규칙·같은 추첨**을 걸어 열 간 비교가 오염되지 않게 한다.
(§7.155 의 mugnull 이 증명한 원칙 — 행 영가설이 아니라 경기 영가설 — 의 연장.)

몬테카를로 오차가 판정을 뒤집은 전례(§7.145)가 있으므로 **해석적 CR0**
`v_g = Σ_c (S_c − n_c·ē_g)² / (Σ_c n_c)²` 를 같이 찍어 대조한다.

### 위약 바닥

블록 카디널리티 급별로 무작위 **경기** 분할 25회의 C 분포를 만들어 **95분위**를
그 급의 바닥으로 쓴다. 판독 규칙: C 가 **양 fold 모두** 자기 급 바닥 위일 때만 플래그.

⚠️ **한계 하나를 명시한다.** 엔티티 블록(투수·타자)은 한 그룹이 여러 경기에 걸쳐
있어 무작위 경기 분할과 **클러스터 기하가 다르다.** 그쪽 바닥은 근사이고,
1차 방어는 위약이 아니라 그룹 내 부트스트랩 노이즈 차감이다.

실행:  .venv/Scripts/python.exe -u src/resmax.py 2>&1 | tee experiments/resmax.log
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np
import pandas as pd

import deploy29 as D
from calslope import build_layers
from calsub import EDGE_SEASON
from predict import PROB_CENTER, PROB_SCALE
from validation import shift_to_prior

B_BOOT, N_PLACEBO = 2000, 25
# ⚠️ **`수백~수천 급` 바닥은 무효다 — 쓰지 말 것** (§7.160 정정, `resmax2.py`).
#    G=400 무작위 경기 분할은 그룹당 경기가 2개뿐이고, m=2 에서 부트스트랩 분산이
#    (m−1)/m·s²/m = s²/4 로 참값의 절반이라 노이즈가 절반만 차감된다. 그래서 위약
#    C 가 +302/+248 로 나온다 — 신호 바닥이 아니라 **노이즈 추정의 붕괴**다.
#    진짜 엔티티 블록은 m_g 중앙이 14~31 이라 기하가 다르다.
#    엔티티 블록의 바닥은 `resmax2.py` 의 **등판 순열 위약**을 쓴다.
TIERS = [("2급", 2), ("수 급", 8), ("수십 급", 30), ("수백~수천 급(무효)", 400)]


def tier_of(G):
    if G <= 2:
        return "2급"
    if G <= 15:
        return "수 급"
    if G <= 99:
        return "수십 급"
    return "수백~수천 급"


def three_cols(codes, ng, cl, cols, rng, B=B_BOOT):
    """(A,B,C) 원단위 분산 + 해석적 CR0 대조. cols = (y, f, e) 행 벡터 3개."""
    n = np.bincount(codes, minlength=ng).astype(np.float64)
    seen = np.flatnonzero(n > 0)
    w = n[seen] / n.sum()
    out, out_cr = [], []
    # (그룹, 경기) 단위 합계를 한 번만 만든다 — 세 열이 **같은 추첨**을 공유한다
    key = codes.astype(np.int64) * (cl.max() + 1) + cl
    uk, inv = np.unique(key, return_inverse=True)
    gk = (uk // (cl.max() + 1)).astype(np.int64)          # 각 (그룹,경기) 의 그룹
    nc = np.bincount(inv).astype(np.float64)              # 클러스터 크기
    Sc = {i: np.bincount(inv, weights=v).astype(np.float64) for i, v in enumerate(cols)}
    order = np.argsort(gk, kind="stable")
    gsort = gk[order]
    bounds = np.searchsorted(gsort, np.arange(ng + 1))

    for i, v in enumerate(cols):
        gm = np.bincount(codes, weights=v, minlength=ng)[seen] / n[seen]
        gbar = float((w * gm).sum())
        raw = float((w * (gm - gbar) ** 2).sum())
        vg = np.zeros(len(seen))
        vg_cr = np.zeros(len(seen))
        S, N_ = Sc[i], nc
        for j, g in enumerate(seen):
            sl = order[bounds[g]:bounds[g + 1]]
            s_, n_ = S[sl], N_[sl]
            m = len(sl)
            if m < 2:
                continue
            idx = rng.integers(0, m, size=(B, m))
            means = s_[idx].sum(1) / np.maximum(n_[idx].sum(1), 1e-9)
            vg[j] = float(np.var(means))
            tot = n_.sum()
            vg_cr[j] = float(np.sum((s_ - n_ * gm[j]) ** 2) / tot ** 2)
        noise = float((w * (1 - w) * vg).sum())
        noise_cr = float((w * (1 - w) * vg_cr).sum())
        out.append(raw - noise)
        out_cr.append(raw - noise_cr)
    return out, out_cr, len(seen)


def blocks(d, apn_edges):
    """서빙 시점에 **행 독립**으로 알 수 있는 범주형 분할. (이름, 코드, 착취가능)"""
    def q10(col):
        v = d[col].to_numpy(dtype=np.float64)
        e = np.nanquantile(v, np.linspace(.1, .9, 9))
        return np.digitize(np.nan_to_num(v, nan=np.nanmedian(v)), e)

    B = []
    B.append(("투수 정체", pd.factorize(d.pitcher_id)[0], True))
    B.append(("타자 정체", pd.factorize(d.batter_id)[0], True))
    B.append(("카운트(b,s)", (d.balls_before * 3 + d.strikes_before).to_numpy(), True))
    B.append(("주자·아웃", (pd.factorize(d.base_state)[0] * 3
                            + d.outs_before.to_numpy()), True))
    B.append(("F/R", (d.game_type.to_numpy().astype(str) == "F").astype(int), True))
    B.append(("홈/원정", (d.top_bottom.to_numpy().astype(str) == "T").astype(int), True))
    B.append(("투수팀", pd.factorize(d.pitcher_team_id)[0], True))
    B.append(("타자팀", pd.factorize(d.batter_team_id)[0], True))
    B.append(("이닝", np.clip(d.inning.to_numpy(), 1, 10) - 1, True))
    B.append(("apn 5분위", np.digitize(d.asof_pitcher_n.to_numpy(float), apn_edges), True))
    B.append(("월 (드리프트)", pd.factorize(d.game_month)[0], False))
    for c in ("asof_pitcher_success_rate", "asof_batter_success_rate",
              "asof_pitcher_reverse_rate", "li"):
        B.append((f"{c} 10분위", q10(c), True))
    B.append(("★위약 요일", pd.factorize(d.game_dayofweek)[0], None))
    return B


def main():
    t0 = time.time()
    df, s, y = D.load_df()
    apn_edges = np.quantile(df.loc[s < EDGE_SEASON, "asof_pitcher_n"].to_numpy(float),
                            [.2, .4, .6, .8])
    pairk = (np.minimum(df.pitcher_team_id, df.batter_team_id).astype(np.int64) * 100
             + np.maximum(df.pitcher_team_id, df.batter_team_id))
    key = df.season.astype(np.int64) * 10000 + df.game_month * 100 + df.game_dayofweek
    half = df.inning.astype(np.int64) * 2 + (df.top_bottom == "B")
    game = (((pairk != pairk.shift()) | (key != key.shift())
             | (half < half.shift())).cumsum()).to_numpy()

    res = {}
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        yv, mm = st["yv"], st["mm"]
        ax = st["Xax"] @ st["dep"]
        nm_l, zb = build_layers(df, s, fold, mm)[-1]
        pb = shift_to_prior(1 / (1 + np.exp(-(zb + ax))), yv.mean())
        f = shift_to_prior(PROB_CENTER + PROB_SCALE * (pb - PROB_CENTER), yv.mean())
        e = yv - f
        d = df.iloc[st["vi"][mm]]
        cl = pd.factorize(game[st["vi"][mm]])[0]
        UNC = yv.mean() * (1 - yv.mean())
        K = 1e5 / UNC
        rng = np.random.default_rng(20260824)

        print(f"\n{'='*104}\nfold{fold}  층 {nm_l}  {len(yv):,}행 · 경기 "
              f"{cl.max()+1:,}  UNC {UNC:.5f}  배포 BSS 참고\n{'='*104}")
        print(f"  {'블록':<26s}{'G':>6s}{'급':>12s}{'A 천장':>9s}{'B 표현':>9s}"
              f"{'C 헤드룸':>10s}{'C(CR0)':>9s}{'착취':>6s}")
        tab = {}
        for name, codes, expl in blocks(d, apn_edges):
            codes = np.asarray(codes, dtype=np.int64)
            codes -= codes.min()
            ng = int(codes.max()) + 1
            (A, Bv, C), (Ac, Bc, Cc), G = three_cols(codes, ng, cl, (yv, f, e), rng)
            tab[name] = (G, C * K, expl)
            print(f"  {name:<26s}{G:>6d}{tier_of(G):>12s}{A*K:>9.1f}{Bv*K:>9.1f}"
                  f"{C*K:>10.1f}{Cc*K:>9.1f}"
                  f"{('O' if expl else ('—' if expl is None else 'X')):>6s}"
                  f"   ({time.time()-t0:.0f}s)")

        # ── 위약 바닥: 급별 무작위 경기 분할 ────────────────────────
        print(f"\n  ── 위약 바닥 (무작위 경기 분할 × {N_PLACEBO}, C 의 95분위) ──")
        floors = {}
        ug = np.unique(cl)
        for tname, G_ in TIERS:
            vals = []
            for _ in range(N_PLACEBO):
                lab = rng.integers(0, G_, len(ug))
                cd = pd.Series(lab, index=ug).reindex(cl).to_numpy()
                (_, _, C), _, _ = three_cols(cd, G_, cl, (yv, f, e), rng, B=400)
                vals.append(C * K)
            floors[tname] = float(np.percentile(vals, 95))
            print(f"    {tname:<14s} G={G_:<5d} 중앙 {np.median(vals):>8.1f}"
                  f"  **95분위 {floors[tname]:>8.1f}**  최대 {max(vals):>8.1f}")
        res[fold] = (tab, floors)

    # ── 판독 ────────────────────────────────────────────────────────
    print(f"\n{'='*104}\n판독 — C 가 **양 fold 모두** 자기 급 바닥 위인가\n{'='*104}")
    names = list(res[D.FOLDS[0]][0])
    print(f"  {'블록':<26s}{'C f2023':>10s}{'바닥':>9s}{'C f2024':>10s}"
          f"{'바닥':>9s}{'판정':>10s}")
    flagged = []
    for nmb in names:
        row = []
        ok = True
        for fold in D.FOLDS:
            tab, fl = res[fold]
            G, C, expl = tab[nmb]
            b = fl[tier_of(G)]
            row += [C, b]
            ok &= C > b
        mark = "🚩플래그" if ok else "—"
        if ok and res[D.FOLDS[0]][0][nmb][2] is False:
            mark = "🚩드리프트"
        if ok and res[D.FOLDS[0]][0][nmb][2] is None:
            mark = "❌위약누수"
        if ok:
            flagged.append(nmb)
        print(f"  {nmb:<26s}{row[0]:>10.1f}{row[1]:>9.1f}"
              f"{row[2]:>10.1f}{row[3]:>9.1f}{mark:>10s}")
    print(f"\n  플래그된 블록: {flagged if flagged else '없음'}")
    if not flagged:
        print("  ⇒ **전 블록 무플래그 — '현 피처셋 정보 천장 근접' 가설 채택.**")
        print("     남은 게임 크기 ≈ 위약 바닥 수준. 전략을 신규 신호 탐색에서")
        print("     **챔피언 방어**(시드·앙상블 분산 축소, 잔여 슬롯 보존)로 전환.")
    print("\n  ⚠️ 한계: 이 장부는 **한계(marginal) 지도**다. 1차 C 에 안 보이는")
    print("     고차 교호·비선형 신호를 놓칠 수 있다. 무플래그 결론은 '증명'이 아니라")
    print("     **'탐색 기대값이 위약 바닥 아래'** 라는 뜻으로 기록한다.")


if __name__ == "__main__":
    main()
