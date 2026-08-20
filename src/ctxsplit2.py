"""투수 × 카운트 δ — 스크린에서 살아남은 축을 단독으로 다시 잰다

`ctxsplit.py`는 9축을 재고 최댓값을 골랐다. **그 최댓값은 편향돼 있다.**
여기서는 count4 / count12 둘만 놓고 채택 판정에 필요한 것을 전부 만든다.

  ① **쌍체 SE** — 같은 행에서 잰 차이라 표본변동이 상쇄된다.
     BSS 이득 = −(100000/BS_ref)·mean(d),  d_i = (p_new−y)² − (p_old−y)²
     SE       =  (100000/BS_ref)·std(d)/√n
     조건 ①(|Δ| > 2·SE)을 이걸로 판정한다. 시드 노이즈와는 다른 축의 불확실성이므로
     둘 다 봐야 한다 (시드 노이즈 fold2024 ±3.3).

  ② **k 강건성** — k=200은 플래툰에서 물려받은 상수다. 카운트 셀은 크기가 다르다.
     여러 k를 찍되 **fold를 보고 argmax하지 않는다.** 고원인지만 확인한다.

  ③ **β 고원** — 양 fold 동시 양수 구간을 보고, 하방 비대칭으로 고른다.
     (플래툰 β=1.5를 고른 것과 같은 논리. 최댓값이라서 고르는 게 아니다.)

  ④ **δ 평균 · 적중률** — §6 버그 2(참조 분포 불일치)의 크기를 가늠한다.
     δ가 자기평균 대비 편차라 0에 가까워야 정상이고, 레벨 영향이 없어야 한다.

  ⑤ **미적중 행 분리** — 이득이 어디서 나오는지. 적중 행에서만 나와야 정상이다.

기준선은 **배포 상태**(블렌드 + PLATOON_BETA·δ_hand)다. 카운트 δ의 이득은
손잡이 δ 위에 얹은 **증분**이라야 의미가 있다.

실행:  .venv/Scripts/python.exe -u src/ctxsplit2.py 2>&1 | tee experiments/ctxsplit2.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache
from config import MLP_ALPHA, PLATOON_BETA, PLATOON_K, TARGET_COL, TRAIN_CSV
from ctxsplit import build_delta, lookup, resid_t
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
KS = [50.0, 100.0, 200.0, 500.0, 1000.0]
BETAS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
CLIP_EPS = 1e-9


def count4(df):
    b = df["balls_before"].to_numpy().astype(np.int64)
    s = df["strikes_before"].to_numpy().astype(np.int64)
    return (b >= 3).astype(np.int64) + 2 * (s >= 2).astype(np.int64), 4


def count12(df):
    b = np.clip(df["balls_before"].to_numpy().astype(np.int64), 0, 3)
    s = np.clip(df["strikes_before"].to_numpy().astype(np.int64), 0, 2)
    return b * 3 + s, 12


AXES = [("count4", count4), ("count12", count12)]


def paired(y, p_old, p_new):
    """(BSS 이득, 쌍체 SE). 같은 행이므로 표본변동이 차이에서 상쇄된다."""
    y = np.asarray(y, dtype=np.float64)
    ref = y.mean() * (1 - y.mean())
    d = (np.clip(p_new, CLIP_EPS, 1 - CLIP_EPS) - y) ** 2 \
        - (np.clip(p_old, CLIP_EPS, 1 - CLIP_EPS) - y) ** 2
    c = 100000.0 / ref
    return -c * d.mean(), c * d.std(ddof=1) / np.sqrt(len(d))


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    pid = df["pitcher_id"].to_numpy(dtype=np.int64)
    bh = df["batter_hand"].to_numpy(dtype=np.int64)
    cached = foldcache.load(FOLDS, verbose=False)

    print(f"기준선 = 블렌드(α={MLP_ALPHA}) + {PLATOON_BETA}·δ_hand   [배포 상태]")
    print("이득은 전부 **손잡이 δ 위에 얹은 증분**이다.\n")

    store = {}
    for fold in FOLDS:
        tr, va = s < fold, s == fold
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        uh, dh, _ = build_delta(pid[tr], bh[tr], y[tr], 8, PLATOON_K)
        dh_row, _ = lookup(uh, dh, pid[va], bh[va], 8)
        z0 = z + PLATOON_BETA * dh_row
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean())
        base = bss(yv, p0)
        r = yv - p0

        print("=" * 96)
        print(f"fold {fold}   valid {va.sum():,}행   기준 BSS {base:.1f}")
        print("=" * 96)

        for name, fn in AXES:
            code, m = fn(df)

            # ── ② k 강건성 (β는 1.0 고정. k와 β를 동시에 고르지 않는다) ──
            print(f"\n  [{name}]  k 강건성 (β=1.0 고정)")
            print(f"    {'k':>7}{'이득':>9}{'±SE':>8}{'t':>7}{'잔차t':>8}{'δ산포':>9}")
            for k in KS:
                u, d, _ = build_delta(pid[tr], code[tr], y[tr], m, k)
                d_row, hit = lookup(u, d, pid[va], code[va], m)
                pn = shift_to_prior(1.0 / (1.0 + np.exp(-(z0 + 1.0 * d_row))),
                                    yv.mean())
                g, se = paired(yv, p0, pn)
                _, t2 = resid_t(r, d_row)
                mark = "  ←배포k" if k == PLATOON_K else ""
                print(f"    {k:>7.0f}{g:>+9.2f}{se:>8.2f}{g / se:>7.1f}"
                      f"{t2:>8.1f}{d_row.std():>9.4f}{mark}")

            # ── ③④⑤ 배포 k에서 β 스윕 ──────────────────────────
            u, d, _ = build_delta(pid[tr], code[tr], y[tr], m, PLATOON_K)
            d_row, hit = lookup(u, d, pid[va], code[va], m)
            print(f"\n  [{name}]  β 스윕 (k={PLATOON_K:.0f})"
                  f"   적중 {hit.mean():.1%}   δ평균 {d_row.mean():+.5f}"
                  f"   로짓기여평균 {d_row.mean():+.5f}×β")
            print(f"    {'β':>7}{'이득':>9}{'±SE':>8}{'t':>7}"
                  f"{'적중행만':>10}{'미적중행':>10}")
            for be in BETAS:
                pn = shift_to_prior(1.0 / (1.0 + np.exp(-(z0 + be * d_row))),
                                    yv.mean())
                g, se = paired(yv, p0, pn)
                gh, _ = paired(yv[hit], p0[hit], pn[hit])
                gm, _ = paired(yv[~hit], p0[~hit], pn[~hit])
                store.setdefault((name, be), {})[fold] = (g, se)
                print(f"    {be:>7.2f}{g:>+9.2f}{se:>8.2f}{g / se:>7.1f}"
                      f"{gh:>+10.2f}{gm:>+10.2f}")

    # ── 판정 ───────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("채택 조건 판정 — ① |Δ|>2·SE (쌍체)  ② 양 fold 부호일치  ③ 크기비 ≤3")
    print("=" * 96)
    for name, _ in AXES:
        print(f"\n  [{name}]")
        print(f"    {'β':>7}{'f2023':>10}{'f2024':>10}{'t23':>7}{'t24':>7}"
              f"{'크기비':>8}   판정")
        for be in BETAS:
            (a, sa), (b, sb) = store[(name, be)][2023], store[(name, be)][2024]
            ratio = max(abs(a), abs(b)) / max(min(abs(a), abs(b)), 1e-9)
            ok = (a > 2 * sa) and (b > 2 * sb) and ratio <= 3
            print(f"    {be:>7.2f}{a:>+10.2f}{b:>+10.2f}"
                  f"{a / sa:>7.1f}{b / sb:>7.1f}{ratio:>8.2f}   "
                  f"{'✓' if ok else '✗'}")

    print("\n  ⚠️ 쌍체 SE는 **행 표본변동**만 담는다. 시드 노이즈(fold2024 ±3.3)는 별도다.")
    print("     둘 다 넘어야 채택이다.")
    print("  ⚠️ β는 최댓값이 아니라 **하방 비대칭**으로 고른다 (플래툰 β=1.5와 같은 논리).")
    print("     2025에서 전이가 1.5배면 실효 β가 1.5배가 된다 — 그 지점이 음수면 안 된다.")


if __name__ == "__main__":
    main()
