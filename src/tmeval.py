"""Trackman arm 평가 — 학습기 배정 격자 · δ/γ 재적합 · 레벨 3종 · 라우팅 게이트

`src/tmarm.py` 가 캐시한 멤버 z 를 배포 등가 32멤버 층(B1)에서 평가한다.
학습 0회.

────────────────────────────────────────────────────────────────────
① 학습기 배정 격자 — "LGBM 만 돌리지 말고 전체를"
    가족 3개(lgb 12 · mlp 8 · cat 12) 각각에 base / tm 을 독립으로 배정 → **8칸**.
    멤버 z 를 가족별로 따로 캐시했기 때문에 재학습 없이 전부 열린다.
    [[augmentation-targets-learners-not-just-rows]] — 같은 조작이
    LGBM +8.5 / MLP −12.1 로 갈린 전례가 있다. §7.9 는 이 축을 물은 적이 없다
    (LGBM 단독 시절이라 물을 수가 없었다).

② δ · γ — 하네스가 최적점을 옮기는 문제
    배포 δ 는 β=(hand 1.875, count4 1.25, gap_success .03, gap_bsuccess .03) 로
    **base 층에서 고정된** 값이다. 트랙맨을 넣으면 그 최적점이 움직인다.
    고정한 채로 재면 "트랙맨 + 잘못 맞춰진 나머지" 를 재는 것이다.
    [[ablation-harness-omitting-a-substitute-shifts-the-optimum]] — 위약·부호·기전
    다 통과한 후보가 대체재를 켜니 네 칸 전부 부호 반전한 전례가 있다.

    γ = δ 항 전체에 걸리는 전역 배율 (β 를 축별로 다시 풀지 않고 한 스칼라로).
    β = 축별 재적합 (좌표 상승).
    ⚠️ 둘 다 **평가 fold 에서 고르면 in-sample** 이다. §7.101 규약을 따른다 —
       **f2023 에서 고르고 f2024 로만 채점**, 역방향도 같이 찍어 부호를 본다.
       [[blend-weight-curves-measure-in-sample-ness]]

③ 레벨(shift) — 세 가지를 다 찍는다
    oracle   shift_to_prior(p, ȳ_valid)   저장소 표준. 레벨 차이를 지우고 refinement 만.
                                          arm 비교에는 옳지만 **오라클**이다.
    frozen   시프트 없음                   자연 레벨. 트랙맨이 레벨을 움직였는지 그대로 보인다.
    shared   두 arm 을 **같은 목표**로      base 의 자연 평균으로 통일. 오라클 없이 refinement.
    ★ 왜 중요한가: BSS 손실 ≈ −400,700 × (예측평균 − 실제평균)² 이다 (MODEL.md §0).
      oracle 만 보면 레벨로 인한 손해가 **통째로 지워진다** — §7.9 의 LB −13.6 이
      난 자리가 정확히 거기다 (§6 버그 2: 캘리브레이션 조건 ≠ 추론 조건).

④ 라우팅 게이트 (형태 C)
    p = w(상황)·p_tm + (1−w(상황))·p_base, 상황 버킷별 w 스윕.
    §7.191 의 R/F 라우팅과 같은 형태. 여기서도 w 는 f2023 에서 고르고 f2024 로 채점한다.

⚠️ 판정 전제 — §7.199 재학습 뽑기 잡음
    SD(Δ) LGBM 0.22/0.40 · Cat 0.23/0.22 · **MLP 4.8/7.1**.
    가족 배정이 MLP 를 건드리는 칸은 잡음 바닥이 ~5~7 이고, LGBM/Cat 만 건드리는
    칸은 ~0.3 이다. **칸마다 잡음 눈금이 다르다** — 같은 Δ 를 같게 읽으면 안 된다.

실행:  .venv/Scripts/python.exe -u src/tmeval.py 2>&1 | tee experiments/tmeval.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import deploy29 as D
from config import MLP_ALPHA, MLP_SEEDS, MLP_VARIANTS, ROOT, SEED_ENSEMBLE
from tmarm import CAT_SEEDS, CAT_VARIANTS, LGB_SEEDS, LGB_VARIANTS
from validation import bss, shift_to_prior

CACHE = ROOT / "experiments" / "cache"
FOLDS = [2023, 2024]
W_CAT = 0.10
B_BOOT = 4000
RNG_SEED = 42

MLP_TAGS = [(v, s) for v, (_c, n) in MLP_VARIANTS.items() for s in MLP_SEEDS[:n]]
GAMMAS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]


# ── 멤버 로딩 ────────────────────────────────────────────────────
def load_family(featset, fam, fold):
    """가족별 멤버 z 행렬 (n_valid, n_member). featset='base' 는 기존 배포 캐시."""
    out = []
    if fam == "lgb":
        for v in LGB_VARIANTS:
            for sd in LGB_SEEDS:
                p = (CACHE / f"initdepm_fold{fold}_{v}_{sd}.npz" if featset == "base"
                     else CACHE / f"tmarm_{featset}_lgb_f{fold}_{v}_{sd}.npz")
                out.append(np.load(p)["z"])
    elif fam == "cat":
        for v in CAT_VARIANTS:
            for sd in CAT_SEEDS:
                p = (CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz" if featset == "base"
                     else CACHE / f"tmarm_{featset}_cat_f{fold}_{v}_{sd}.npz")
                out.append(np.load(p)["z"])
    elif fam == "mlp":
        if featset == "base":
            with np.load(CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
                mem = d0["members"]
            assert mem is not None and mem.shape[0] == 8, "MLP 멤버 캐시 없음"
            out = [mem[i] for i in range(8)]
        else:
            for v, sd in MLP_TAGS:
                out.append(np.load(CACHE / f"tmarm_{featset}_mlp_f{fold}_{v}_{sd}.npz")["z"])
    return np.stack(out, axis=1).astype(np.float64)

def _tm_hit(st):
    """그 행의 (pitcher_id, season) 이 trackman 표에 있는가. build_features 의 tm_matched 와 같다."""
    import pandas as pd
    from tmarm import TM_CSV
    tm = pd.read_csv(TM_CSV)
    d = st["d"]
    key = set(zip(tm.pitcher_id.to_numpy(), tm.season.to_numpy()))
    pid = d["pitcher_id"].to_numpy()
    ssn = d["season"].to_numpy()
    return np.fromiter(((int(a), int(b)) in key for a, b in zip(pid, ssn)),
                       dtype=np.int64, count=len(pid))


def blend_z(members):
    """B1 가중 — 배포 챔피언 등가. lgb 12 균등 / emb4 5 · noid 3 (.875/.125) / cat 12 균등."""
    zl = members["lgb"].mean(axis=1)
    M = members["mlp"]
    zm = 0.875 * M[:, :5].mean(axis=1) + 0.125 * M[:, 5:].mean(axis=1)
    zc = members["cat"].mean(axis=1)
    return (1 - W_CAT) * ((1 - MLP_ALPHA) * zl + MLP_ALPHA * zm) + W_CAT * zc


# ── 평가 기계 ────────────────────────────────────────────────────
def score(st, z, mode="oracle", target=None, gamma=1.0, beta=None):
    """(BSS, 확률평균). mode: oracle / frozen / shared."""
    ax = st["Xax"] @ (st["dep"] if beta is None else beta)
    zz = z[st["mm"]] + gamma * ax
    p = 1.0 / (1.0 + np.exp(-zz))
    if mode == "oracle":
        p = shift_to_prior(p, st["ybar"])
    elif mode == "shared":
        p = shift_to_prior(p, target)
    return bss(st["yv"], p), float(p.mean())


def probs(st, z, gamma=1.0, beta=None, mode="oracle", target=None):
    ax = st["Xax"] @ (st["dep"] if beta is None else beta)
    p = 1.0 / (1.0 + np.exp(-(z[st["mm"]] + gamma * ax)))
    if mode == "oracle":
        return shift_to_prior(p, st["ybar"])
    if mode == "shared":
        return shift_to_prior(p, target)
    return p


def boot(st, p_new, p_ref, rng, B=B_BOOT):
    """투수 클러스터 부트스트랩 — hillclimb.boot_delta 와 동형."""
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


def refit_beta(st, z, iters=4, m=13):
    """좌표 상승으로 δ 축 β 재적합 (BSS 최대). oracle 레벨에서.

    ⚠️ 격자 폭을 **반복마다 절반으로 줄인다.** 폭이 고정이면 좌표를 몇 번 돌아도
       해상도가 안 올라가서 '수렴한 것처럼 보이는 격자 스냅' 이 나온다.
    """
    b = st["dep"].astype(np.float64).copy()
    span = np.maximum(np.abs(b), 0.05) * 2.0
    for _ in range(iters):
        for j in range(len(b)):
            grid = b[j] + np.linspace(-1.0, 1.0, m) * span[j]
            best, bj = -1e18, b[j]
            for g in grid:
                bb = b.copy()
                bb[j] = g
                s, _ = score(st, z, "oracle", beta=bb)
                if s > best:
                    best, bj = s, g
            b[j] = bj
        span *= 0.5
    return b


def main():
    rng = np.random.default_rng(RNG_SEED)
    print(f"§7.205 후속 — Trackman arm 평가   지문 {D.KEY}   B_BOOT={B_BOOT}")
    df, s, y = D.load_df()
    ST, MEM = {}, {}
    for f in FOLDS:
        ST[f] = D.state(df, s, y, f)
        ST[f]["ybar"] = float(ST[f]["yv"].mean())     # deploy29.state 에 없는 키
        MEM[f] = {}
        for fs in ("base", "tm"):
            MEM[f][fs] = {fam: load_family(fs, fam, f) for fam in ("lgb", "mlp", "cat")}
    del df

    # ── 하네스 앵커 — base 가 B1 등가(807.68 / 925.75)를 재현해야 한다 ──
    print("\n" + "=" * 92)
    print("하네스 앵커 — base 조합이 §7.194 B1 을 재현하는가 (807.68 / 925.75)")
    print("=" * 92)
    Zb = {}
    for f in FOLDS:
        Zb[f] = blend_z(MEM[f]["base"])
        sc, pm = score(ST[f], Zb[f], "oracle")
        print(f"  fold{f}  B1(base) = {sc:.2f}   p̄ {pm:.5f}   "
              f"deploy29 base {ST[f]['base']:.1f}")

    # ── ① 학습기 배정 격자 (8칸) ────────────────────────────────
    print("\n" + "=" * 92)
    print("① 학습기 배정 격자 — 가족별 base/tm 독립 배정 (oracle 레벨, 배포 δ·γ=1)")
    print("=" * 92)
    print("  §7.199 잡음 SD:  lgb 0.22/0.40 · cat 0.23/0.22 · mlp 4.8/7.1")
    print("  → MLP 를 바꾸는 칸(★)은 잡음 바닥이 ~5~7, 나머지는 ~0.3 이다\n")
    print(f"  {'lgb':>5}{'mlp':>5}{'cat':>5}"
          f"{'f2023':>10}{'Δ':>8}{'f2024':>10}{'Δ':>8}   {'p̄ f24':>9}  잡음")
    grid = {}
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                key = ("base", "tm")
                mm = {f: {"lgb": MEM[f][key[a]]["lgb"], "mlp": MEM[f][key[b]]["mlp"],
                          "cat": MEM[f][key[c]]["cat"]} for f in FOLDS}
                zz = {f: blend_z(mm[f]) for f in FOLDS}
                sc = {f: score(ST[f], zz[f], "oracle") for f in FOLDS}
                base_sc = {f: score(ST[f], Zb[f], "oracle")[0] for f in FOLDS}
                d23 = sc[2023][0] - base_sc[2023]
                d24 = sc[2024][0] - base_sc[2024]
                noise = "★~5-7" if b else ("~0.3" if (a or c) else "—")
                print(f"  {key[a]:>5}{key[b]:>5}{key[c]:>5}"
                      f"{sc[2023][0]:>10.2f}{d23:>+8.2f}{sc[2024][0]:>10.2f}{d24:>+8.2f}"
                      f"{sc[2024][1]:>10.5f}  {noise}")
                grid[(a, b, c)] = (zz, d23, d24)

    Zt = {f: blend_z(MEM[f]["tm"]) for f in FOLDS}

    # ── ③ 레벨 3종 — 전 학습기 tm vs base ────────────────────────
    print("\n" + "=" * 92)
    print("③ 레벨(shift) 3종 — 전 학습기 tm arm. ★ oracle 만 보면 레벨 손해가 지워진다")
    print("=" * 92)
    print(f"  {'fold':>6}{'모드':>9}{'base':>10}{'tm':>10}{'Δ':>9}"
          f"{'p̄ base':>10}{'p̄ tm':>10}{'ȳ':>10}")
    for f in FOLDS:
        st = ST[f]
        _, pm_b = score(st, Zb[f], "frozen")
        for mode in ("oracle", "frozen", "shared"):
            tgt = pm_b if mode == "shared" else None
            sb, mb = score(st, Zb[f], mode, target=tgt)
            sti, mt = score(st, Zt[f], mode, target=tgt)
            print(f"  {f:>6}{mode:>9}{sb:>10.2f}{sti:>10.2f}{sti-sb:>+9.2f}"
                  f"{mb:>10.5f}{mt:>10.5f}{st['ybar']:>10.5f}")
    print("  · frozen 의 Δ 가 oracle 의 Δ 와 크게 다르면 → 트랙맨이 **레벨을 통해** 움직인 것.")
    print("    그 몫은 배포 시 PROB_SHIFT 재보정으로 사라지므로 refinement 로 세면 안 된다.")

    # ── ② γ 스윕 + β 재적합 ─────────────────────────────────────
    print("\n" + "=" * 92)
    print("② δ·γ 재적합 — 배포 β 는 base 층에서 고정된 값이다 (oracle 레벨)")
    print("=" * 92)
    print(f"  {'arm':>5}{'fold':>6}  " + "".join(f"γ{g:<6.2f}" for g in GAMMAS))
    best_g = {}
    for nm, Zx in (("base", Zb), ("tm", Zt)):
        for f in FOLDS:
            row = [score(ST[f], Zx[f], "oracle", gamma=g)[0] for g in GAMMAS]
            best_g[(nm, f)] = GAMMAS[int(np.argmax(row))]
            print(f"  {nm:>5}{f:>6}  " + "".join(f"{v:>7.1f}" for v in row))
    print(f"\n  최적 γ:  " + "  ".join(f"{k[0]}/f{k[1]}={v}" for k, v in best_g.items()))
    print("  ⚠️ 이 꼭짓점은 in-sample 이다. 아래 §7.101 규약(f2023 에서 고르고 f2024 채점)이 판정.")

    print(f"\n  β 재적합 (좌표 상승, 축 4개) — f2023 에서 적합 → f2024 채점")
    for nm, Zx in (("base", Zb), ("tm", Zt)):
        b_fit = refit_beta(ST[2023], Zx[2023])
        s_dep, _ = score(ST[2024], Zx[2024], "oracle")
        s_ref, _ = score(ST[2024], Zx[2024], "oracle", beta=b_fit)
        gsel = best_g[(nm, 2023)]
        s_gam, _ = score(ST[2024], Zx[2024], "oracle", gamma=gsel)
        print(f"    {nm:<5} 배포β {s_dep:8.2f}   재적합β {s_ref:8.2f} ({s_ref-s_dep:+.2f})"
              f"   γ={gsel} {s_gam:8.2f} ({s_gam-s_dep:+.2f})"
              f"   β̂={np.round(b_fit, 4).tolist()}")
    print("    (배포 β = " + str(np.round(ST[2023]['dep'], 4).tolist()) + ")")

    # ── ④ 상황 게이트 (형태 C) — w·γ 를 f2023 에서 **함께** 고르고 f2024 채점 ──
    print()
    print("=" * 92)
    print("④ 상황 게이트 — p = w(상황)·p_tm + (1−w)·p_base,  w 와 γ 를 함께 고른다")
    print("=" * 92)
    print("  왜 함께 고르나: ① 라우팅이 예측 분포를 바꾸면 δ 최적 배율 γ 가 움직인다")
    print("                  ② 라우팅이 평균을 바꾸므로 레벨을 다시 풀어야 한다")
    print("                  ③ 레벨을 다시 풀면 **버킷들이 커플링된다** — 버킷별 독립")
    print("                     최적화는 그 커플링을 놓친다 → 좌표 상승 2회전")
    print("  선택은 f2023 에서만. 채점은 f2024 에서만 (§7.101). in-fold 는 크기 보고용.")
    print()

    WGRID = np.linspace(0.0, 1.0, 11)
    GGRID = [0.75, 0.9, 1.0, 1.1, 1.25]
    AXES = {
        "전역(1칸)":     lambda st: np.zeros(len(st["yv"]), dtype=np.int64),
        "game_type R/F": lambda st: (st["d"]["game_type"].to_numpy() == "F").astype(np.int64),
        "2스트라이크":   lambda st: (st["d"]["strikes_before"].to_numpy() >= 2).astype(np.int64),
        "주자유무":      lambda st: (st["d"]["num_runners_on"].to_numpy() > 0).astype(np.int64),
        "count4":        lambda st: ((st["d"]["balls_before"].to_numpy() >= 3).astype(np.int64)
                                     + 2 * (st["d"]["strikes_before"].to_numpy() >= 2).astype(np.int64)),
        # 트랙맨이 실제로 조회된 행에만 게이트를 연다 — 형태 A(조건부)와 C(라우팅)의 교집합
        "tm 적중":       lambda st: _tm_hit(st),
    }

    def route_p(st, codes, w, gamma, mode="oracle", target=None):
        """버킷별 w 로 로짓 공간에서 섞고, 레벨을 다시 푼다."""
        ax = st["Xax"] @ st["dep"]
        zb = Zb_[st["f"]][st["mm"]] + gamma * ax
        zt = Zt_[st["f"]][st["mm"]] + gamma * ax
        z = zb + (np.asarray(w)[codes]) * (zt - zb)
        p = 1.0 / (1.0 + np.exp(-z))
        if mode == "oracle":
            return shift_to_prior(p, st["ybar"])
        if mode == "shared":
            return shift_to_prior(p, target)
        return p

    Zb_, Zt_ = {}, {}
    for f in FOLDS:
        ST[f]["f"] = f
        Zb_[f], Zt_[f] = Zb[f], Zt[f]

    print(f"  {'축':<15}{'선택 w (f2023)':<26}{'γ':>5}"
          f"{'f2024 Δ':>10}{'95% CI':>20}{'P':>6}   {'frozen Δ':>10}")
    for name, fn in AXES.items():
        if fn is None:
            continue
        codes = {f: fn(ST[f]) for f in FOLDS}
        m = int(max(c.max() for c in codes.values())) + 1
        # 작은 버킷은 아예 안 고른다 (w=0 고정)
        big = [k for k in range(m) if (codes[2023] == k).sum() >= 2000]

        w = np.zeros(m)
        g = 1.0
        for _round in range(2):                      # 좌표 상승 — 커플링 때문에 2회전
            for k in big:
                best, bw = -1e18, w[k]
                for cand in WGRID:
                    w2 = w.copy(); w2[k] = cand
                    v = bss(ST[2023]["yv"], route_p(ST[2023], codes[2023], w2, g))
                    if v > best:
                        best, bw = v, cand
                w[k] = bw
            bestg, bg = -1e18, g
            for cand in GGRID:
                v = bss(ST[2023]["yv"], route_p(ST[2023], codes[2023], w, cand))
                if v > bestg:
                    bestg, bg = v, cand
            g = bg

        p24 = route_p(ST[2024], codes[2024], w, g)
        p24b = route_p(ST[2024], codes[2024], np.zeros(m), g)
        d24 = bss(ST[2024]["yv"], p24) - bss(ST[2024]["yv"], p24b)
        lo, hi, pg = boot(ST[2024], p24, p24b, rng)
        # frozen 레벨 — 게이트가 레벨을 통해 움직인 몫을 분리한다
        fz = (bss(ST[2024]["yv"], route_p(ST[2024], codes[2024], w, g, "frozen"))
              - bss(ST[2024]["yv"], route_p(ST[2024], codes[2024], np.zeros(m), g, "frozen")))
        print(f"  {name:<15}{str(np.round(w, 2).tolist()):<26}{g:>5.2f}"
              f"{d24:>+10.2f}   [{lo:+7.2f},{hi:+7.2f}]{pg:>6.2f}   {fz:>+10.2f}")

    print()
    print("  · f2024 Δ 는 **같은 γ 에서** 게이트 켬/끔의 차이다 (γ 효과를 섞지 않는다).")
    print("  · frozen Δ 가 oracle Δ 와 크게 다르면 게이트가 **레벨을 통해** 움직인 것 —")
    print("    그 몫은 배포에서 PROB_SHIFT 재보정으로 사라지므로 refinement 로 세면 안 된다.")
    print("  · w 가 전 버킷 0 이면 f2023 조차 트랙맨을 안 쓰기로 한 것이다 (게이트 사망).")

    # ── 종합 ────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("종합 — 전 학습기 tm arm, 클러스터 부트스트랩 (oracle 레벨, 배포 δ)")
    print("=" * 92)
    for f in FOLDS:
        lo, hi, pg = boot(ST[f], probs(ST[f], Zt[f]), probs(ST[f], Zb[f]), rng)
        sb, _ = score(ST[f], Zb[f], "oracle")
        sti, _ = score(ST[f], Zt[f], "oracle")
        print(f"  fold{f}  base {sb:8.2f}  tm {sti:8.2f}  Δ {sti-sb:+8.2f}"
              f"   95% [{lo:+.2f}, {hi:+.2f}]  P(Δ>0) {pg:.3f}")
    print("\n  ⚠️ Δ 는 채택 근거가 아니다 — §7.9 가 홀드아웃 +13.3 → LB −13.6 을 두 번 냈다.")
    print("     [[holdout-gain-conflates-signal-with-fold-specific-error]]")
    print("     이 표가 답하는 것은 '현 스택에서 어느 학습기가, 어떤 레벨 처리에서,")
    print("     어떤 δ 설정으로 무엇을 얻는가' 이지 '2025 에서 이득인가' 가 아니다.")


if __name__ == "__main__":
    main()
