"""§7.61 — δ 표가 **모델이 무시하는 시즌**으로 만들어지고 있다.

§7.60이 asof 축에서 찾은 것과 같은 종류의 누락이다.

    fit_ctx 는 bincount — 가중이 없다.
      δ 표의 질량      2019~2022 = 66%
      모델 학습 질량    2019~2022 =  6%   (recency half-life 0.5)

    게다가 pre-2023 F 105,308행(7.1%)은 라벨이 오염돼 있다:
      2022  R .5037 / F .7087     ← F가 R보다 +0.205
      2023  R .5031 / F .4729     ← 정의 변경 후 −0.030
    이 행들이 δ 표에서 가중 1.0이다 (같은 행의 모델 학습 가중은 0.023).

δ = n/(n+k)·(rate_pc − rate_p) 는 **자기 평균으로부터의 편차**라서 리그 레벨
드리프트에는 중립이다. 하지만 오염이 투수 내부에서 셀마다 고르지 않으면 남는다.
그리고 레벨과 무관하게, 옛 시즌이 표를 지배하면 δ는 **2019~2022의 투수별 편차**를
말하게 된다 — 2025에 쓸 값이 아니다.

변형 (β는 §7.36·§7.37의 비(1.5 : 1.0)를 유지하고 **전역 배율 g만** 쓴다.
      표의 스케일이 바뀌므로 g를 다시 잡아야 하고, 자유도는 1개면 충분하다.)

    now          균등 가중 (현재 배포)
    recency      모델과 **같은** recency 가중
    noF          균등이지만 pre-2023 F 제외
    recency_noF  둘 다

기준선은 **블렌드 + asof(보정판, 배포)** 다 — δ의 배포 위치에서의 증분을 본다.

실행:  .venv/Scripts/python.exe -u src/ctxw.py 2>&1 | tee experiments/ctxw.log
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
from config import (ASOF_AXES, ASOF_STRENGTH, CONTAM_AXES, CTX_AXES, CTX_K,
                    MLP_ALPHA, RECENCY_HALF_LIFE, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from mixdiag import shape
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
# 전역 배율. 1.0 = §7.36·§7.37이 고른 (1.5, 1.0) 그대로.
GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def variants(tr, s_tr):
    """(이름, 가중) — 가중 None 은 균등(=현재 배포)."""
    ref = int(s_tr.max())
    rw = recency_weight(s_tr, RECENCY_HALF_LIFE, ref_season=ref)
    gt = tr["game_type"].astype("object").to_numpy()
    keep = ~((s_tr <= 2022) & (gt == "F"))          # 오염 구간 제외 마스크
    return [("now", None),
            ("recency", rw),
            ("noF", keep.astype(np.float64)),
            ("recency_noF", rw * keep)]


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    out = {}

    for fold in FOLDS:
        tr_m, va = s < fold, s == fold
        tr, d = df[tr_m], df[va].copy()
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]

        # 기준선 = 블렌드 + asof(보정판 = 배포)
        art0 = F.fit_artifacts(tr, contam_axes=CONTAM_AXES)
        art0["asof"] = asofdelta.fit_asof_axes(tr, ASOF_AXES, ASOF_STRENGTH,
                                               ct=art0.get("contam"))
        a_asof, _ = F.logit_adjust(d, {"asof": art0["asof"]})
        z0 = z + a_asof
        base = bss(yv, shift_to_prior(1 / (1 + np.exp(-z0)), yv.mean()))

        print(f"\nfold {fold}  기준 {base:.1f}  (블렌드 + asof 보정판, δ 없음)")
        print(f"  {'변형':<14}{'셀':>9}" + "".join(f"{g:>8}" for g in GRID))
        for name, w in variants(tr, s[tr_m]):
            ctx = ctxfit.fit_ctx(tr, TARGET_COL, CTX_AXES, CTX_K, weights=w)
            v, _ = F.logit_adjust(d, {"ctx": ctx})       # 이미 β가 곱해진 Σβδ
            cells = sum(len(x["keys"]) for x in ctx)
            row = []
            for g in GRID:
                p = shift_to_prior(1 / (1 + np.exp(-(z0 + g * v))), yv.mean())
                row.append(bss(yv, p) - base)
            out.setdefault(name, {})[fold] = row
            print(f"  {name:<14}{cells:>9,}" + "".join(f"{x:>+8.2f}" for x in row))

    print("\n" + "=" * 100)
    print("판정 — 양 fold 양수 g 구간 · 밴드 규칙 · 크기비")
    print("=" * 100)
    print(f"  {'변형':<14}{'양수 g 구간':>18}{'폭':>6}{'허용 g':>22}"
          f"{'g':>6}{'f2023':>10}{'f2024':>10}{'크기비':>8}")
    best = {}
    for name, byf in out.items():
        a23, a24 = np.array(byf[2023]), np.array(byf[2024])
        both = (a23 > 0) & (a24 > 0)
        pl = [GRID[i] for i in range(len(GRID)) if both[i]]
        if not pl:
            print(f"  {name:<14}{'없음':>18}")
            continue
        lo, hi = min(pl), max(pl)
        ok = [g for g in GRID if 0.5 * g >= lo and 1.5 * g <= hi]
        if not ok:
            print(f"  {name:<14}[{lo}, {hi}]  밴드 만족 g 없음")
            continue
        g = ok[max(len(ok) - 2, 0)]              # 허용 최대의 한 칸 아래 (§7.60과 같은 규칙)
        i = GRID.index(g)
        r = max(abs(a23[i]), abs(a24[i])) / max(min(abs(a23[i]), abs(a24[i])), 1e-9)
        best[name] = (g, a23[i], a24[i], r)
        print(f"  {name:<14}{f'[{lo}, {hi}]':>18}{hi/lo:>5.0f}배"
              f"{str(ok):>22}{g:>6}{a23[i]:>+10.2f}{a24[i]:>+10.2f}{r:>8.2f}")

    # ── 조건 ④ — δ를 ≤2021에서 fit 해 2022·2023·2024 모양을 out-of-sample로 본다
    print("\n" + "=" * 100)
    print("조건 ④ — 관계의 시즌 간 지속성 (δ는 ≤2021에서 fit → 2022~2024는 전부 out-of-sample)")
    print("=" * 100)
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    tr21 = s <= 2021
    rng = np.random.default_rng(42)
    LATE = [2022, 2023, 2024]
    print(f"  {'변형':<14}{'2022-23':>11}{'2023-24':>11}{'최소':>9}{'천장':>9}{'비율':>8}")
    for name, w in variants(df[tr21], s[tr21]):
        ctx = ctxfit.fit_ctx(df[tr21], TARGET_COL, CTX_AXES, CTX_K, weights=w)
        v, _ = F.logit_adjust(df, {"ctx": ctx})
        sh, rel = [], []
        for yr in LATE:
            m = s == yr
            yy, vv = y[m], v[m]
            sh.append(shape(yy, vv))
            h = rng.random(len(yy)) < 0.5
            rr = float(np.corrcoef(shape(yy[h], vv[h]), shape(yy[~h], vv[~h]))[0, 1])
            rel.append(2 * rr / (1 + rr) if rr > -1 else np.nan)
        cs = [float(np.corrcoef(sh[i], sh[i + 1])[0, 1]) for i in range(len(sh) - 1)]
        ceil = float(np.nanmean(rel))
        print(f"  {name:<14}" + "".join(f"{c:>+11.3f}" for c in cs)
              + f"{min(cs):>+9.3f}{ceil:>9.3f}{min(cs)/ceil if ceil else np.nan:>8.2f}")

    print("\n  ⚠️ δ는 자기 평균으로부터의 편차라 레벨에는 중립이다. 여기서 묻는 것은")
    print("     **어느 시즌의 투수별 편차를 담고 있나**다. 옛 시즌이 표를 지배하면")
    print("     2025에 쓸 값이 아니게 된다 (오염 여부와 별개의 문제).")


def hl():
    """★ half-life 스윕 — **정렬 이득 vs 분산 손실**을 분리한다.

    `recency`(hl=0.5)가 f2024 +1.96이지만 f2023은 +0.09뿐이었다. 두 힘이 싸운다:
        정렬  가중을 최근으로 옮기면 δ가 **배포 시즌의** 투수별 편차를 담는다  (+)
        분산  유효 표본이 줄어 셀별 δ가 노이즈해진다                        (−)

    ⚠️ 조건 ④를 이 비교에 그대로 쓸 수 없다. §7.60에서는 원본/보정이 **같은 표본 수**를
       가졌지만, 여기서는 변형마다 유효 표본이 다르다. δ 표 자체의 노이즈가 시즌 간
       모양 상관을 감쇠시키므로, 낮은 조건 ④가 "관계 불안정"인지 "표가 노이즈"인지
       구별되지 않는다. (실제로 hl=0.5의 조건 ④가 더 낮게 나왔다 — 표가 2021 한 시즌에
       가깝게 만들어지기 때문이다.)

    → 그래서 기전을 직접 시험한다. 정렬이 원인이면 **중간 half-life에 최적**이 있어야
      한다. 단조 개선도 단조 악화도 아니고 아무 구조가 없으면 노이즈로 보고 접는다.
    """
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    HLS = [None, 6.0, 4.0, 2.5, 1.5, 1.0, 0.5]      # None = 균등(현재 배포)
    G = [0.75, 1.0, 1.25]
    res = {}

    for fold in FOLDS:
        tr_m, va = s < fold, s == fold
        tr, d = df[tr_m], df[va].copy()
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        art0 = F.fit_artifacts(tr, contam_axes=CONTAM_AXES)
        art0["asof"] = asofdelta.fit_asof_axes(tr, ASOF_AXES, ASOF_STRENGTH,
                                               ct=art0.get("contam"))
        a_asof, _ = F.logit_adjust(d, {"asof": art0["asof"]})
        z0 = z + a_asof
        base = bss(yv, shift_to_prior(1 / (1 + np.exp(-z0)), yv.mean()))
        ref = int(s[tr_m].max())

        print()
        print(f"fold {fold}  기준 {base:.1f}")
        print(f"  {'half-life':<12}{'유효표본':>12}" + "".join(f"{'g='+str(g):>10}" for g in G))
        for h in HLS:
            w = None if h is None else recency_weight(s[tr_m], h, ref_season=ref)
            neff = len(tr) if w is None else w.sum() ** 2 / (w ** 2).sum()
            ctx = ctxfit.fit_ctx(tr, TARGET_COL, CTX_AXES, CTX_K, weights=w)
            v, _ = F.logit_adjust(d, {"ctx": ctx})
            row = []
            for g in G:
                p = shift_to_prior(1 / (1 + np.exp(-(z0 + g * v))), yv.mean())
                row.append(bss(yv, p) - base)
            res.setdefault(h, {})[fold] = row
            print(f"  {str(h):<12}{neff:>12,.0f}" + "".join(f"{x:>+10.2f}" for x in row))

    print()
    print("=" * 88)
    print("판정 — g=1.0(배포값) 기준, 균등 대비 증감")
    print("=" * 88)
    b23, b24 = res[None][2023][1], res[None][2024][1]
    print(f"  {'half-life':<12}{'f2023':>10}{'Δ':>9}{'f2024':>10}{'Δ':>9}{'양쪽Δ최소':>11}")
    for h in HLS:
        a, b = res[h][2023][1], res[h][2024][1]
        print(f"  {str(h):<12}{a:>+10.2f}{a-b23:>+9.2f}{b:>+10.2f}{b-b24:>+9.2f}"
              f"{min(a-b23, b-b24):>+11.2f}")
    print()
    print("  중간 half-life에 봉우리가 있으면 정렬 기전이 실재한다.")
    print("  단조도 봉우리도 없으면 노이즈 → 접는다 (배포 유지).")


if __name__ == "__main__":
    hl() if "--hl" in sys.argv else main()
