"""§7.67 — 기각된 gap 축들을 **축별 S**로 다시 연다.

왜 다시 여는가
    §7.63에서 `ASOF_STRENGTH`가 6배 틀렸다는 것이 드러났다 (200 → 35, f2024 +9.08).
    **지금까지 기각된 gap 축은 전부 S=200에서 판정됐다.**
    수축이 과했다면 신호가 뭉개진 상태로 채점된 것이고, 기각이 S 탓일 수 있다.

    그리고 S는 지금 **전역**이다. 축마다 셀 크기와 신호대잡음이 다른데
    같은 수축을 강요할 이유가 없다 (§7.64에서 셀 크기 설명은 기각됐지만,
    "축마다 다르다"는 사실 자체는 그대로다 — S=35가 맞는 축과 K=200이 맞는 축이 있다).

    memory: "닫은 진단도 새 도구로 다시 물어라" — 결론이 맞아도 기전이 비어 있으면
    이득이 잠겨 있다. 여기서 새 도구는 **축별 S**다.

기각 이력 (전부 S=200에서)
    gap_reverse   크기비 4.03~5.97 (조건 ③)      조건 ④ 0.83
    gap_middle    조건 ④ 0.01                    잔차 기울기 음수
    gap_ball      조건 ④ 0.14
    gap_bmiddle   미측정
    gap_strike    미측정

설계
    기준선 = 배포 전체 (블렌드 + δ + gap_success@S35 + gap_bsuccess@S35)
    후보마다 (S, β) 2차원 스윕 → S별로 밴드 규칙으로 β를 고르고 조건 ③으로 판정.
    β는 **부호 있는 격자** — reverse·middle·ball은 잔차 기울기가 음수다 (§7.57).

⚠️ 판정은 fold2024 중심 (fold2023은 단절 fold).
⛔ LB를 보고 고르지 않는다.

실행:  .venv/Scripts/python.exe -u src/asofrevisit.py 2>&1 | tee experiments/asofrevisit.log
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
                    CTX_K, MLP_ALPHA, SEED, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
SS = [12.0, 25.0, 35.0, 50.0, 100.0, 200.0]
MAG = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08]

# (축, id컬럼, n컬럼, rate컬럼).  오염 보정은 asofdelta.CONTAM_OF 에 있는 축만.
CAND = [
    ("gap_reverse",  "pitcher_id", "asof_pitcher_n", "asof_pitcher_reverse_rate"),
    ("gap_middle",   "pitcher_id", "asof_pitcher_n", "asof_pitcher_middle_rate"),
    ("gap_ball",     "pitcher_id", "asof_pitcher_n", "asof_pitcher_ball_rate"),
    ("gap_strike",   "pitcher_id", "asof_pitcher_n", "asof_pitcher_strike_rate"),
    ("gap_bmiddle",  "batter_id",  "asof_batter_n",  "asof_batter_middle_rate"),
]


def band_pick(a23, a24, mags):
    """|β| 격자에서 양 fold 양수 구간을 찾고 밴드 규칙 통과 값 중 최대의 한 칸 아래."""
    both = [(a23[i] > 0) and (a24[i] > 0) for i in range(len(mags))]
    pl = [mags[i] for i in range(len(mags)) if both[i]]
    if not pl:
        return None, None
    lo, hi = min(pl), max(pl)
    ok = [b for b in mags if 0.5 * b >= lo and 1.5 * b <= hi]
    if not ok:
        return None, (lo, hi)
    return ok[max(len(ok) - 2, 0)], (lo, hi)


def make_axis(off_all, ct, league, idc, ncol, rcol, name, S, train_df):
    a = dict(id=idc, n_col=ncol, r_col=rcol, name=name,
             keys=off_all[ncol]["keys"], n_off=off_all[ncol]["n_off"],
             k_off=off_all[ncol]["k_off"][rcol], mu=0.0, sd=1.0, beta=0.0,
             league=league, strength=float(S))
    if name in asofdelta.CONTAM_OF:
        a["cadj"] = asofdelta.CONTAM_OF[name]
    raw = F.asof_lookup(train_df, a, ct=ct)        # ⚠️ 학습 행에서만 표준화
    a["mu"] = float(np.mean(raw))
    a["sd"] = float(np.std(raw)) or 1.0
    return a


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
        art = F.fit_artifacts(tr, ctx=ctx, contam_axes=CONTAM_AXES)
        art["asof"] = asofdelta.fit_asof_axes(tr, ASOF_AXES, ASOF_STRENGTH,
                                              ct=art.get("contam"))
        adj, _ = F.logit_adjust(d, art)                 # 배포 전체
        z0 = z + adj
        st[fold] = dict(tr=tr, d=d, yv=yv, z0=z0, ct=art["contam"],
                        off=asofdelta.fit_offsets(tr),
                        league=float(tr[TARGET_COL].mean()),
                        base=bss(yv, shift_to_prior(1 / (1 + np.exp(-z0)), yv.mean())))
        print(f"fold {fold}  기준 {st[fold]['base']:.1f}  (배포 전체 = s21)")

    print("\n" + "=" * 100)
    print("축별 (S, β) 2차원 — S마다 밴드 규칙으로 β 선택, 배포 위의 **증분**")
    print("=" * 100)
    for name, idc, ncol, rcol in CAND:
        print(f"\n  ── {name}"
              + ("  [오염보정 있음]" if name in asofdelta.CONTAM_OF else ""))
        print(f"     {'S':<8}{'sign':>6}{'|β|':>8}{'f2023':>10}{'f2024':>10}"
              f"{'크기비':>8}{'고원':>16}")
        best = None
        for S in SS:
            V = {}
            for fold in FOLDS:
                x = st[fold]
                a = make_axis(x["off"], x["ct"], x["league"], idc, ncol, rcol,
                              name, S, x["tr"])
                V[fold] = F.asof_lookup(x["d"], a, ct=x["ct"])
            for sign in (+1, -1):
                rows = {}
                for fold in FOLDS:
                    x = st[fold]
                    rows[fold] = [
                        bss(x["yv"], shift_to_prior(
                            1 / (1 + np.exp(-(x["z0"] + sign * b * V[fold]))),
                            x["yv"].mean())) - x["base"] for b in MAG]
                a23, a24 = np.array(rows[2023]), np.array(rows[2024])
                b, pl = band_pick(a23, a24, MAG)
                if b is None:
                    continue
                i = MAG.index(b)
                r = max(abs(a23[i]), abs(a24[i])) / max(min(abs(a23[i]), abs(a24[i])), 1e-9)
                print(f"     {S:<8}{sign:>6}{b:>8}{a23[i]:>+10.2f}{a24[i]:>+10.2f}"
                      f"{r:>8.2f}{str(pl):>16}")
                if r <= 3.0 and (best is None or a24[i] > best[3]):
                    best = (S, sign, b, a24[i], a23[i], r)
        if best is None:
            print("     → 조건 ②③을 함께 통과하는 (S, β) 없음  ❌ 기각 유지")
        else:
            S, sg, b, g24, g23, r = best
            print(f"     ★ 최선  S={S}  β={sg*b:+}  f2023 {g23:+.2f}  f2024 {g24:+.2f}"
                  f"  크기비 {r:.2f}")
            # 대조군 — 값 분포는 그대로, 행과의 연결만 끊는다
            for fold in FOLDS:
                x = st[fold]
                a = make_axis(x["off"], x["ct"], x["league"], idc, ncol, rcol,
                              name, S, x["tr"])
                v = F.asof_lookup(x["d"], a, ct=x["ct"])
                vs = v[rng.permutation(len(v))]
                g = bss(x["yv"], shift_to_prior(
                    1 / (1 + np.exp(-(x["z0"] + sg * b * vs))), x["yv"].mean())) - x["base"]
                print(f"       대조군 fold{fold}  {g:+.2f}")

    print("\n" + "=" * 100)
    print("읽는 법 — 기각된 축이 살아나면 그것은 **S 탓이었다**는 뜻이다.")
    print("  살아나지 않으면 기각은 S와 무관했고, §7.63의 이득은 그 두 축에 국한된다.")
    print("  어느 쪽이든 '축별 S'라는 자유도 하나를 닫는다.")


if __name__ == "__main__":
    main()
