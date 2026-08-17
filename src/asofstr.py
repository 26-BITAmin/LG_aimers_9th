"""§7.63 — 한 번도 스윕 안 한 노브 하나, 그리고 상호작용항.

**A. `ASOF_STRENGTH` (EB 수축 강도).**
    gap = (k_szn + S·prior)/(n_szn + S) − prior
    S는 "시즌 초 저표본을 커리어로 얼마나 끌어당기나"다. 200으로 박혀 있는데
    **스윕한 기록이 없다.** LB +40을 준 축의 1-파라미터 노브다.

    S가 작으면  시즌 초 30구짜리 추정치를 그대로 믿는다 → 노이즈
    S가 크면    시즌분이 커리어로 뭉개진다 → 신호 소실
    최적이 200일 이유가 없다. `SMOOTH_K`(=200)에서 그냥 가져온 값이다.

**B. 두 gap 축의 상호작용.**
    투수도 올해 무너졌고 타자도 상대 제구가 나쁘면 — 곱해서 더 나쁜가?
    현재는 z + β_p·v_p + β_b·v_b 로 **가법**이다. 곱항은 파라미터 1개짜리
    사후 가산으로 얹을 수 있다 (§7.56의 컬럼 경로와 달리 학습이 없다).

    ⚠️ §7.56이 컬럼 경로에서 본 상호작용 이득(+39.2)은 LB로 1/69만 왔다.
       그건 **트리 분할 수만 개**를 얼린 결과였다. 여기는 스칼라 1개다.

설계
    기준선 = 블렌드 + δ(배포 hl=1.0, β 1.875/1.25)
    각 S마다 §7.60과 **같은 절차**로 β를 다시 고른다 (S가 바뀌면 고원이 이동한다).
      단계 A  gap_success   → 밴드 규칙 → β_A
      단계 B  gap_bsuccess  → 밴드 규칙 → β_B   (A 위의 증분)
    그 다음 상호작용항을 A+B 위의 증분으로 스윕한다.

⚠️ 판정은 fold2024 중심. fold2023은 단절 fold다 (§7.51 — 대조군조차 +25.9).
⛔ LB를 보고 고르지 않는다.

실행:  .venv/Scripts/python.exe -u src/asofstr.py 2>&1 | tee experiments/asofstr.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import asofdelta
import ctxfit
import features as F
import foldcache
from config import (ASOF_AXES, ASOF_STRENGTH, CONTAM_AXES, CTX_AXES, CTX_HALF_LIFE,
                    CTX_K, MLP_ALPHA, RECENCY_HALF_LIFE, SEED, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
STRENGTHS = [18.0, 25.0, 35.0, 50.0, 70.0, 200.0]
BETAS = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.14]
IBETAS = [-0.04, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.04]
NAMES = ["gap_success", "gap_bsuccess"]


def band_pick(a23, a24, grid):
    """양 fold 양수 구간을 찾고 [0.5β, 1.5β] ⊂ 구간 인 β 중 **최대의 한 칸 아래**.

    앵커가 고원 기하이므로 검증 fold 점수를 보지 않는다 (§7.60에서 정한 규칙).
    """
    both = [(a23[i] > 0) and (a24[i] > 0) for i in range(len(grid))]
    pl = [grid[i] for i in range(len(grid)) if both[i]]
    if not pl:
        return None, None, []
    lo, hi = min(pl), max(pl)
    ok = [b for b in grid if 0.5 * b >= lo and 1.5 * b <= hi]
    if not ok:
        return None, (lo, hi), []
    return ok[max(len(ok) - 2, 0)], (lo, hi), ok


def build_axes(train_df, off_all, ct, league, strength):
    """오프셋을 재사용하고 **strength만 바꿔** 축을 만든다 (mu/sd는 다시 뽑는다)."""
    out = []
    for name in NAMES:
        spec = asofdelta.ASOF_SPEC[name]
        o = off_all[spec["n_col"]]
        a = dict(spec, name=name, keys=o["keys"], n_off=o["n_off"],
                 k_off=o["k_off"][spec["r_col"]], mu=0.0, sd=1.0, beta=0.0,
                 league=league, strength=float(strength),
                 cadj=asofdelta.CONTAM_OF[name])
        raw = F.asof_lookup(train_df, a, ct=ct)      # ⚠️ 학습 행에서만 표준화
        a["mu"] = float(np.mean(raw))
        a["sd"] = float(np.std(raw)) or 1.0
        out.append(a)
    return out


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(SEED)

    st = {}
    for fold in FOLDS:
        tr_m, va = s < fold, s == fold
        tr, d = df[tr_m], df[va].copy()
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ref = int(s[tr_m].max())
        ctx = ctxfit.fit_ctx(tr, TARGET_COL, CTX_AXES, CTX_K,
                             weights=recency_weight(s[tr_m], CTX_HALF_LIFE,
                                                    ref_season=ref))
        art = F.fit_artifacts(tr, contam_axes=CONTAM_AXES)
        adj, _ = F.logit_adjust(d, {"ctx": ctx})
        z0 = z + adj
        st[fold] = dict(
            tr=tr, d=d, yv=yv, z0=z0,
            base=bss(yv, shift_to_prior(1 / (1 + np.exp(-z0)), yv.mean())),
            ct=art["contam"],
            off=asofdelta.fit_offsets(tr),
            league=float(tr[TARGET_COL].mean()))
        print(f"fold {fold}  기준 {st[fold]['base']:.1f}  (블렌드 + δ 배포판)")

    # ── A · B  strength별로 β를 다시 고른다 ──────────────────────────────
    print("\n" + "=" * 104)
    print("A · B — strength별 β 재선정 (밴드 규칙) 후 두 축 합산 이득")
    print("=" * 104)
    print(f"  {'S':<8}{'β_p':>8}{'β_b':>8}{'f2023':>10}{'f2024':>10}{'크기비':>8}"
          f"{'고원(p)':>16}{'고원(b)':>16}")
    table = {}
    for S in STRENGTHS:
        V, chosen, gains = {}, [], {}
        for fold in FOLDS:
            x = st[fold]
            ax = build_axes(x["tr"], x["off"], x["ct"], x["league"], S)
            V[fold] = {a["name"]: F.asof_lookup(x["d"], a, ct=x["ct"]) for a in ax}
        plateau = {}
        for stage, name in enumerate(NAMES):
            rows = {}
            for fold in FOLDS:
                x = st[fold]
                fixed = sum(b * V[fold][n] for n, b in chosen) if chosen else 0.0
                zb = x["z0"] + fixed
                bb = bss(x["yv"], shift_to_prior(1 / (1 + np.exp(-zb)), x["yv"].mean()))
                rows[fold] = [
                    bss(x["yv"], shift_to_prior(1 / (1 + np.exp(-(zb + b * V[fold][name]))),
                                                x["yv"].mean())) - bb
                    for b in BETAS]
                gains[(fold, name)] = (bb, rows[fold])
            b, pl, _ = band_pick(np.array(rows[2023]), np.array(rows[2024]), BETAS)
            plateau[name] = pl
            if b is None:
                chosen = None
                break
            chosen.append((name, b))
        if chosen is None:
            print(f"  {S:<8}  밴드 규칙 만족 β 없음 → 제외")
            continue
        tot = {}
        for fold in FOLDS:
            x = st[fold]
            zb = x["z0"] + sum(b * V[fold][n] for n, b in chosen)
            tot[fold] = bss(x["yv"], shift_to_prior(1 / (1 + np.exp(-zb)),
                                                    x["yv"].mean())) - x["base"]
        r = max(abs(tot[2023]), abs(tot[2024])) / max(min(abs(tot[2023]), abs(tot[2024])), 1e-9)
        table[S] = (dict(chosen), tot, r, V)
        pp, pb = plateau[NAMES[0]], plateau[NAMES[1]]
        print(f"  {S:<8}{chosen[0][1]:>8}{chosen[1][1]:>8}{tot[2023]:>+10.2f}"
              f"{tot[2024]:>+10.2f}{r:>8.2f}{str(pp):>16}{str(pb):>16}")

    if not table:
        print("\n  전부 제외됐다. 종료.")
        return
    print(f"\n  배포값 S={ASOF_STRENGTH} 대비:")
    b0 = table.get(ASOF_STRENGTH)
    if b0:
        for S, (ch, tot, r, _) in sorted(table.items()):
            print(f"    S={S:<8} Δf2023 {tot[2023]-b0[1][2023]:>+8.2f}   "
                  f"Δf2024 {tot[2024]-b0[1][2024]:>+8.2f}")

    # ── C  상호작용 — 배포 S에서, A+B 위의 증분 ──────────────────────────
    print("\n" + "=" * 104)
    print(f"C — 상호작용 v_p · v_b  (S={ASOF_STRENGTH}, A+B 위의 증분, 파라미터 1개)")
    print("=" * 104)
    ch, tot, _, V = table[ASOF_STRENGTH]
    print(f"  {'':<14}" + "".join(f"{b:>9}" for b in IBETAS))
    irows = {}
    for fold in FOLDS:
        x = st[fold]
        zb = x["z0"] + sum(b * V[fold][n] for n, b in ch.items())
        bb = bss(x["yv"], shift_to_prior(1 / (1 + np.exp(-zb)), x["yv"].mean()))
        vp, vb = V[fold]["gap_success"], V[fold]["gap_bsuccess"]
        # ⚠️ 곱항도 **학습 행 통계로** 표준화해야 한다. 검증 행 평균을 쓰면 리크다.
        ax = build_axes(x["tr"], x["off"], x["ct"], x["league"], ASOF_STRENGTH)
        tp = F.asof_lookup(x["tr"], ax[0], ct=x["ct"])
        tb = F.asof_lookup(x["tr"], ax[1], ct=x["ct"])
        pr = tp * tb
        mu, sd = float(pr.mean()), float(pr.std()) or 1.0
        vi = (vp * vb - mu) / sd
        vs = vi[rng.permutation(len(vi))]                # 대조군
        for tag, v in (("v_p·v_b", vi), ("[대조군]", vs)):
            row = [bss(x["yv"], shift_to_prior(1 / (1 + np.exp(-(zb + b * v))),
                                               x["yv"].mean())) - bb for b in IBETAS]
            irows.setdefault(tag, {})[fold] = row
        print(f"  fold {fold} (기준 {bb:.1f})")
        for tag in ("v_p·v_b", "[대조군]"):
            print(f"    {tag:<12}" + "".join(f"{g:>+9.2f}" for g in irows[tag][fold]))

    a23 = np.array(irows["v_p·v_b"][2023]); a24 = np.array(irows["v_p·v_b"][2024])
    b, pl, ok = band_pick(a23, a24, IBETAS)
    print(f"\n  양 fold 양수 구간 {pl}   밴드 허용 {ok}   채택 {b}")
    if b is not None:
        i = IBETAS.index(b)
        r = max(abs(a23[i]), abs(a24[i])) / max(min(abs(a23[i]), abs(a24[i])), 1e-9)
        print(f"  β_int={b}   f2023 {a23[i]:+.2f}  f2024 {a24[i]:+.2f}  크기비 {r:.2f}")
        print(f"  {'✅ 조건 ③ 통과' if r <= 3 else '❌ 조건 ③ 위반 (>3)'}")
    else:
        print("  ❌ 상호작용 기각 — 양 fold 공통 구간에서 밴드를 못 만든다")


if __name__ == "__main__":
    main()
