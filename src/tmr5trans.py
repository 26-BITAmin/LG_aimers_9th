"""Trackman 재구축 5단계 — **4시즌 홀드아웃 BSS + 시즌 전이** (학습 0회)

왜 4시즌인가
    사용자 지시는 "무조건 holdout 2023 2024 결과를 보여줘" 였다. 3단계가 그걸 준다
    (재학습 경로, 팔당 52분). 이 파일은 **같은 질문을 2시즌이 아니라 4시즌**에 대해
    학습 없이 답한다 — `foldcache` 에 2021·2022·2023·2024 가 전부 캐시돼 있다.

    2점으로는 표류와 진동을 못 가른다. §7.204c 가 `gap_strike` 를 4시즌으로 다시 열어
    "z 3.05 부호 분열" 이 실은 평균 0 짜리 진동이었음을 밝혔다. 같은 실수를 안 하려면
    trackman 도 4점으로 봐야 한다.

무엇을 재나
    각 시즌 t (2021~2024) 에 대해:
        기준선 z   = 캐시된 블렌드(α) + δ(hand) + δ(count4)   ← 배포 상태
        후보       = z + β · standardize(trackman 컬럼)
        BSS(β) − BSS(0)  를 β 스윕으로

    표준화 통계는 **시즌 ≤ t−1 로만** 낸다 (검증 시즌 통계를 쓰면 리크).

    이건 3단계와 **다른 질문**이다:
        3단계  재학습 — "모델이 이 정보를 쓸 수 있나" (트리가 비선형·상호작용까지 쓴다)
        5단계  사후 β — "**얼린 계수 하나가 네 시즌 전부에 통하나**" (전이 질문)
    §7.9 가 죽은 자리가 정확히 후자다 — 홀드아웃 +13.3 인데 LB −13.6.

판정 (사전 등록 — 결과 보기 전에 박는다)
    ① 네 시즌 **전부** 양수인 β 구간이 있는가 (하나라도 음수면 전이 근거 약함)
    ② 최적 β 에서 시즌 간 이득 크기비 ≤ 3 (§2 채택 조건 ③)
    ③ β 기울기 시계열의 **안정 성분** — 평균 ± SE 가 0 을 배제하고 Q 가 과분산 아님
    ⚠️ ①②③ 를 다 통과해도 **채택이 아니다.** 채택은 LB 가 정한다.
    ⚠️ 하나라도 실패해도 **자동 기각 아니다** — 사용자 지시("기각 함부로 하지 말 것").
       실패 항목과 그 크기를 적고, 3단계 재학습 결과와 **같이** 읽는다.

실행:  .venv/Scripts/python.exe -u src/tmr5trans.py 2>&1 | tee experiments/tmr5trans.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

import ctxfit
import foldcache
from config import CTX_AXES, CTX_K, MLP_ALPHA, ROOT, TARGET_COL, TRAIN_CSV
from data import load_csv
from features import logit_adjust
from validation import bss, shift_to_prior

SEASONS = [2021, 2022, 2023, 2024]
TMR_DIR = ROOT / "experiments" / "tmr"
BETAS = [-0.20, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.20]

# 2단계 예측상관이 정한 두 군 — §7.9 의 선택을 물려받지 않는다
#   §7.9 가 고른 것 (ρ .15~.30, 운반 최하위)  vs  한 번도 단독 시험 안 된 것 (ρ .73~.86)
COLS_S79 = ["tm_relh_std", "tm_rels_std", "tm_relpoint_disp", "tm_ext_std"]
COLS_NEW = ["tm_ivb_std", "tm_hb_std", "tm_spin_std", "tm_velo_std"]
COLS_MEAN = ["tm_ivb_mean", "tm_hb_mean", "tm_velo_mean", "tm_ext_mean"]


def cluster_se(r, d, g):
    """기울기와 투수 클러스터 CR0 SE. trackman 컬럼은 투수 안에서 상수라 필수."""
    dc = d - d.mean()
    den = float((dc ** 2).sum())
    if den <= 0:
        return 0.0, np.inf
    sl = float((dc * r).sum() / den)
    e = (r - r.mean()) - sl * dc
    h = dc * e
    _, inv = np.unique(g, return_inverse=True)
    G = inv.max() + 1
    sg = np.bincount(inv, weights=h, minlength=G)
    meat = float((sg ** 2).sum()) * G / max(G - 1, 1)
    return sl, float(np.sqrt(meat) / den)


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(SEASONS, verbose=False)

    variants = {}
    for v in ("deploy", "within", "drift", "regime", "recent"):
        p = TMR_DIR / f"tm_{v}.csv"
        if p.exists():
            variants[v] = pd.read_csv(p)
    print(f"변형 {list(variants)}   시즌 {SEASONS}")

    # ── 시즌별 기준선·잔차 ────────────────────────────────────
    base = {}
    for t in SEASONS:
        tr, va = s < t, s == t
        d = df[va]
        yv = cached[t][2]
        z = (1 - MLP_ALPHA) * cached[t][0] + MLP_ALPHA * cached[t][1]
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        adj, _ = logit_adjust(d, {"ctx": ctx})
        z0 = z + adj
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean())
        base[t] = dict(z0=z0, y=yv, bss=bss(yv, p0), r=yv - p0,
                       pid=d["pitcher_id"].to_numpy(dtype=np.int64))
        print(f"  fold {t}  기준 BSS(배포) {base[t]['bss']:>8.1f}   n {len(yv):,}")

    for vname, tab in variants.items():
        cols = [c for c in (COLS_S79 + COLS_NEW + COLS_MEAN) if c in tab.columns]
        if not cols:            # drift 는 컬럼 이름이 다르다
            cols = [c for c in tab.columns if c not in ("pitcher_id", "season", "tm_n")]
        print("\n" + "=" * 100)
        print(f"변형 **{vname}** — 4시즌 홀드아웃 BSS 이득 (사후 β 가산, 학습 0회)")
        print("=" * 100)
        print(f"  {'컬럼':<20}{'β':>7}" + "".join(f"{t:>9}" for t in SEASONS)
              + f"{'최소':>8}{'크기비':>8}{'CR0 t 부호':>12}")

        for c in cols:
            grid = {}
            slopes = []
            for t in SEASONS:
                B = base[t]
                sub = tab[tab.season == t].set_index("pitcher_id")
                if c not in sub.columns:
                    grid[t] = None
                    continue
                v = sub[c].reindex(B["pid"]).to_numpy(dtype=np.float64)
                # 표준화는 **시즌 ≤ t−1 통계로만** (검증 시즌 통계는 리크)
                past = tab[tab.season <= t][c].to_numpy(dtype=np.float64)
                mu, sd = np.nanmean(past), np.nanstd(past)
                vz = np.where(np.isnan(v), 0.0, (v - mu) / (sd if sd > 0 else 1.0))
                grid[t] = [bss(B["y"], shift_to_prior(
                    1.0 / (1.0 + np.exp(-(B["z0"] + b * vz))), B["y"].mean())) - B["bss"]
                    for b in BETAS]
                sl, se = cluster_se(B["r"], vz, B["pid"])
                slopes.append((t, sl, se))
            if any(g is None for g in grid.values()):
                continue

            M = np.array([grid[t] for t in SEASONS])          # (시즌, β)
            mins = M.min(axis=0)
            j = int(np.argmax(mins))                          # 최악 시즌을 최대화하는 β
            best = M[:, j]
            pos = best > 0
            ratio = (np.abs(best).max() / max(np.abs(best).min(), 1e-9))
            sgn = "".join("+" if sl > 0 else "−" for _, sl, _ in slopes)
            mark = " ✅ 4시즌 전부 양수" if pos.all() else ""
            print(f"  {c:<20}{BETAS[j]:>7.2f}" + "".join(f"{x:>+9.1f}" for x in best)
                  + f"{mins[j]:>+8.1f}{ratio:>8.1f}{sgn:>12}{mark}")

            # ③ 안정 성분 — 기울기 시계열 (§7.204c 와 같은 기계)
            if pos.all():
                a = np.array([sl for _, sl, _ in slopes])
                e = np.array([se for _, _, se in slopes])
                w = 1.0 / np.maximum(e ** 2, 1e-30)
                mbar = float((w * a).sum() / w.sum())
                sem = float(np.sqrt(1.0 / w.sum()))
                Q = float((w * (a - mbar) ** 2).sum())
                print(f"       ③ 기울기 안정성: 평균 {mbar:+.5f} ± {sem:.5f}"
                      f" (t {mbar/sem:+.2f})   Q {Q:.2f} (df 3, 7.81=5% 과분산선)")

    print("\n" + "=" * 100)
    print("읽는 법 — 사전 등록한 대로")
    print("=" * 100)
    print("  ① 네 시즌 전부 양수인 β 가 있는가        (✅ 표시)")
    print("  ② 최적 β 에서 시즌 간 크기비 ≤ 3         (크기비 열)")
    print("  ③ 기울기 시계열에 안정 성분이 있는가      (t · Q)")
    print("  ⚠️ 다 통과해도 채택 아니다 — LB 가 정한다. §7.9 는 홀드아웃 +13.3 → LB −13.6.")
    print("  ⚠️ 실패해도 자동 기각 아니다 — 3단계 재학습 BSS 와 **같이** 읽는다.")
    print("     사후 β 는 선형 가산만 본다. 트리는 비선형·상호작용도 쓴다.")


if __name__ == "__main__":
    main()
