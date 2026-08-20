"""투수 절편 empirical Bayes 수축 (사용자가 가져온 제안) — 공식 검정 + 적용 검정.

받은 제안
    b_j = n_j/(n_j + λ) · (ȳ_j − ȳ),      λ = σ̂²_within / σ̂²_between

공식은 맞다 — Bühlmann credibility / James–Stein의 표준형이다. 두 가지를 확인한다.

① 추정 함정 — `σ̂²_between`을 어떻게 내는지가 빠져 있다
    `Var_j(ȳ_j)`를 그냥 쓰면 **표집 노이즈가 섞여 과대**가 된다:
        Var_j(ȳ_j) = σ²_b + E[σ²_w / n_j]
    → 순진한 추정은 σ²_b를 부풀리고, λ를 과소하게 만들고, **수축을 덜 하게** 만든다.
    적률법으로 E[σ²_w/n_j]를 빼야 한다. 두 값을 나란히 찍어 함정 크기를 본다.

② 새 방법이 아니다 — 이미 같은 공식이 배포돼 있다
        config.SMOOTH_K_PITCHER = 200      `asof_pitcher_success_rate_sm` 의 K
        config.CTX_K            = 200      δ 표의 수축
        config.ASOF_STRENGTH    =  35      gap 의 EB 수축
    제안은 "새 수축"이 아니라 **그 200을 데이터에서 다시 유도하자**는 것이다.
      λ_EB ≈ 200  →  배포값이 EB 최적임이 확인된다
      크게 다름   →  그게 뉴스다 (§7.31은 K를 스윕해 200을 확인했지만 EB로는 안 물었다)

③ 결정적 검정 — **b_j가 배포 모델의 잔차에 뭘 남기나**
    배포 모델은 이미 `asof_pitcher_success_rate`(커리어, as-of)와 그 `_sm`(K=200 수축)을
    피처로 본다. b_j는 그보다 **거친** 양이다(시즌 경계까지만 잘린 평균).
    → 잔차 t ≈ 0 이면 새 정보가 없다. (memory: 새 피처는 잔차에 먼저 물어라)
    대조군: 시즌 안에서 `pitcher_id` 셔플. **대조군이 판정자다.**

④ t가 살면 그때 β 스윕 + 전방향 전이 (§7.86 절차). 안 살면 거기서 끝난다.

실행:  .venv/Scripts/python.exe -u src/ebshrink.py 2>&1 | tee experiments/ebshrink.log
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

import jointbeta as J
from ctxsplit import resid_t
from config import (ASOF_AXES, ASOF_STRENGTH, CTX_K, MLP_ALPHA, SMOOTH_K_PITCHER,
                    CTX_AXES, TARGET_COL, TRAIN_CSV)
from data import load_csv
from validation import _logit, bss, shift_to_prior

FOLDS = [2021, 2022, 2023, 2024]
MINN = [1, 10, 30, 100, 300]          # σ²_b 추정에 쓸 최소 표본 (민감도)
SEED = 42


def lam_eb(pid, y, min_n=30):
    """(λ_적률, λ_순진, σ²_w, σ²_b_적률, σ²_b_순진, 투수 수)."""
    d = pd.DataFrame({"p": pid, "y": y})
    g = d.groupby("p")["y"].agg(["mean", "count"])
    g = g[g["count"] >= min_n]
    n = g["count"].to_numpy(dtype=np.float64)
    m = g["mean"].to_numpy(dtype=np.float64)
    # 이항이므로 그룹 내 분산 = p_j(1−p_j). n으로 가중해 pooled.
    s2w = float((n * m * (1 - m)).sum() / n.sum())
    v_obs = float(np.var(m, ddof=1))                    # 순진한 σ²_b (노이즈 포함)
    bias = float(np.mean(s2w / n))                      # E[σ²_w / n_j]
    s2b = max(v_obs - bias, 1e-12)                      # 적률법
    return (s2w / s2b, s2w / max(v_obs, 1e-12), s2w, s2b, v_obs, len(g))


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y_all = df[TARGET_COL].to_numpy(dtype=np.float64)
    pid_all = df["pitcher_id"].to_numpy(dtype=np.int64)

    # ── ① 함정 크기 + λ_EB ──────────────────────────────────────────────
    print(f"{'='*84}\n■ ① λ_EB 추정 — 적률법 vs 순진한 방법\n{'='*84}")
    print("  Var_j(ȳ_j) = σ²_b + E[σ²_w/n_j].  뺄셈을 빠뜨리면 σ²_b 과대 → λ 과소 → 수축 부족.\n")
    print(f"  {'최소 n':>7}{'투수 수':>9}{'σ²_w':>10}{'σ²_b(적률)':>12}{'σ²_b(순진)':>12}"
          f"{'λ_적률':>10}{'λ_순진':>10}{'비':>7}")
    for mn in MINN:
        le, ln, s2w, s2b, vo, k = lam_eb(pid_all, y_all, mn)
        print(f"  {mn:>7}{k:>9,}{s2w:>10.5f}{s2b:>12.6f}{vo:>12.6f}"
              f"{le:>10.1f}{ln:>10.1f}{le/max(ln,1e-9):>7.2f}")
    le30 = lam_eb(pid_all, y_all, 30)[0]
    print(f"\n  배포 상수:  SMOOTH_K_PITCHER = {SMOOTH_K_PITCHER:g}"
          f"   CTX_K = {CTX_K:g}   ASOF_STRENGTH = {ASOF_STRENGTH:g}")
    print(f"  λ_EB(min_n=30, 적률) = {le30:.1f}"
          f"   →  SMOOTH_K 대비 {le30/SMOOTH_K_PITCHER:.2f}배")

    # ── ② 배포 잔차에 b_j를 물어본다 ────────────────────────────────────
    print(f"\n{'='*84}\n■ ② 결정적 검정 — b_j 가 배포 모델의 잔차에 남는가\n{'='*84}")
    print("  배포 모델은 이미 `asof_pitcher_success_rate`(as-of 커리어)와 `_sm`(K=200)을 본다.")
    print("  b_j는 그보다 거친 양이다. 새 정보가 없으면 t ≈ 0. **대조군이 판정자.**\n")
    cached = J.foldcache.load(FOLDS, verbose=False)
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    rng = np.random.default_rng(SEED)

    print(f"  {'fold':>6}{'λ_EB(train)':>13}   " + "".join(
        f"{h:>13}" for h in ("t(λ_EB)", "t(λ=200)", "t(λ=0)", "대조군 t", "|t|/|t_대조|")))
    keep = {}
    for f in FOLDS:

        X, names, _, va = J.axis_matrix(df, s, f)
        yv = cached[f][2]
        z0 = (1 - MLP_ALPHA) * cached[f][0] + MLP_ALPHA * cached[f][1]
        m = va["game_type"].to_numpy() != "F"
        X, yv, z0, va = X[m], yv[m], z0[m], va[m]
        z0 = _logit(shift_to_prior(1 / (1 + np.exp(-z0)), yv.mean()))
        p = shift_to_prior(1 / (1 + np.exp(-(z0 + X @ dep))), yv.mean())
        r = yv - p                                   # 배포 예측의 잔차

        # b_j 는 **train(<f)에서만** 만든다
        tr_m = s < f
        lam_f = lam_eb(pid_all[tr_m], y_all[tr_m], 30)[0]
        d = pd.DataFrame({"p": pid_all[tr_m], "y": y_all[tr_m]})
        g = d.groupby("p")["y"].agg(["mean", "count"])
        ybar = float(y_all[tr_m].mean())
        vp = va["pitcher_id"].to_numpy(dtype=np.int64)
        gm = g["mean"].reindex(vp).fillna(ybar).to_numpy()
        gn = g["count"].reindex(vp).fillna(0.0).to_numpy()

        def bvec(lmb):
            return (gn / (gn + lmb)) * (gm - ybar) if lmb > 0 else (gm - ybar)

        ts = [resid_t(r, bvec(l))[1] for l in (lam_f, 200.0, 0.0)]
        # 대조군 — 시즌 안에서 투수 재배정 (b_j의 구조는 유지, 연결만 끊는다)
        perm = rng.permutation(len(vp))
        gm2, gn2 = gm[perm], gn[perm]
        bs = (gn2 / (gn2 + lam_f)) * (gm2 - ybar)
        tc = resid_t(r, bs)[1]
        keep[f] = dict(r=r, z0=z0, X=X, y=yv, b=bvec(lam_f), lam=lam_f)
        print(f"  {f:>6}{lam_f:>13.1f}   " + "".join(f"{t:>13.2f}" for t in ts)
              + f"{tc:>13.2f}{abs(ts[0])/max(abs(tc),1e-9):>13.1f}")

    # ── ③ t가 살면 β 스윕 ───────────────────────────────────────────────
    print(f"\n{'='*84}\n■ ③ β 스윕 — 잔차 t가 살았을 때만 의미가 있다\n{'='*84}")
    base = {f: bss(keep[f]["y"], shift_to_prior(
        1 / (1 + np.exp(-(keep[f]["z0"] + keep[f]["X"] @ dep))), keep[f]["y"].mean()))
        for f in FOLDS}
    print(f"  배포 기준선(R)  " + "  ".join(f"{f}:{base[f]:.1f}" for f in FOLDS))
    print(f"\n  {'β':>8} │ " + " │ ".join(f"{f:>9}" for f in FOLDS) + " │  부호")
    for b in (-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0, 4.0):
        ds = []
        for f in FOLDS:
            z = keep[f]["z0"] + keep[f]["X"] @ dep + b * keep[f]["b"]
            ds.append(bss(keep[f]["y"], shift_to_prior(
                1 / (1 + np.exp(-z)), keep[f]["y"].mean())) - base[f])
        sign = "✓" if all(x > 0 for x in ds) else ("✗" if all(x < 0 for x in ds) else "혼")
        print(f"  {b:>8.2f} │ " + " │ ".join(f"{d:>+9.1f}" for d in ds)
              + f" │ {sign:>5}")


if __name__ == "__main__":
    main()
