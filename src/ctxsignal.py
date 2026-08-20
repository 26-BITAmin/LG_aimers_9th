"""δ 이득이 **검증 시즌의 신호량**에 따라 줄어드는가 — 배율 3.3의 기전 검정

왜 이걸 재는가
    `lbcalib.py`가 정직하게 말해준 것: 정보 추가 계열의 배율 3.3은 **표본 2개**다.
    표본 2개짜리 상수를 계속 쓰는 것은 LB 2점에 파라미터를 맞추는 것과 다르지 않다.
    나가는 길은 **"왜 3.3인가"를 train 안에서 재는 것**뿐이다.

가설 (MODEL.md §7.36의 후보 ①)
    2025는 고신호 시즌이다 — LB refinement 985+ vs fold2024 799.
    **기본 모델이 투수 축을 더 많이 뽑아내고 있으면 δ가 채울 잔차가 그만큼 작다.**
    참이면 δ 이득은 검증 시즌의 신호량과 **음의 관계**여야 한다.

    이 가설이 배율이 축마다 같았던 것(hand 3.38 / count4 3.16)도 설명한다 —
    원인이 축의 성질이 아니라 **시즌의 성질**이면 축이 바뀌어도 배율이 같다.

설계 — §7.22가 남겨둔 캐시를 그대로 쓴다 (학습 없음)
    `h_w2021` / `h_w2022`가 **학습기를 고정하고 검증 연도만 밀어내는** 설계로
    이미 만들어져 있다. 창을 고정하면 **δ 테이블이 완전히 동일**하므로
    "δ가 약해서 이득이 작다"는 교란이 사라진다.

        W2021 (δ 고정) → 2022 · 2023 · 2024      ← 주 증거
        W2022 (δ 고정) → 2023 · 2024
        W2023          → 2024                    ← foldcache 재사용

    시즌 신호량 (§7.21 foldsurvey):  2022 **1849.2** / 2023 **−10.2** / 2024 **741.9**

교란과 그 방향 (미리 적어둔다)
    창 고정 설계에서는 드리프트 거리 h가 같이 움직인다 (2022는 h=1, 2024는 h=3).
    **h는 가설과 반대 방향으로 민다** — h가 작을수록 δ가 잘 맞아야 하므로 2022의
    이득이 커야 한다. 그런데도 2022의 이득이 작으면 신호량 쪽이 이긴 것이다.
    → **h가 알리바이를 제공하지 않는 방향의 결과만 증거로 쓴다.**

사전 예측 (실행 전에 박는다. lbcalib의 예측 기록은 1승 5패다)
    P1  W2021에서 이득(2022) < 이득(2024)        예측: **그렇다** (가설 지지)
    P2  이득 ÷ 시즌 신호량 이 시즌 간에 더 안정적  예측: **아니다** (2023 신호량이 −10이라
                                                 비율이 발산한다 — 이 지표는 애초에 못 쓴다)
    P3  잔차 회귀 t 가 신호량과 음의 관계          예측: **그렇다**

실행:  .venv/Scripts/python.exe -u src/ctxsignal.py 2>&1 | tee experiments/ctxsignal.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import ctxfit
from ablate23 import member_list, try_load
from config import CTX_AXES, CTX_K, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from ctxsplit import resid_t
from data import load_csv
from features import logit_adjust
from foldcache import CACHE, _key
from validation import bss, shift_to_prior

# (학습창, [검증 연도들]) — 창을 고정하면 δ 테이블이 같다
WINDOWS = [(2021, [2022, 2023, 2024]), (2022, [2023, 2024])]


def have(w):
    return all((CACHE / f"h_w{w}_{_key()}_{k}_{t}.npz").exists()
               for k in ("lgb", "mlp") for t in member_list(k))


def main():
    df = load_csv(TRAIN_CSV)
    season = df["season"].to_numpy()
    print(f"캐시 지문 {_key()}   배포 보정축 {CTX_AXES}  k={CTX_K}\n")

    rows = []
    for w, valids in WINDOWS:
        if not have(w):
            print(f"⏭  W{w} 체크포인트 없음 — 건너뛴다 (학습하지 않는다)")
            continue
        z = try_load(w, valids)

        # δ는 학습창으로만 fit한다. **valids 전부에 같은 테이블을 쓴다.**
        ctx = ctxfit.fit_ctx(df[season <= w], TARGET_COL, CTX_AXES, CTX_K)
        art = {"ctx": ctx}
        print(f"=== 학습창 W{w} — δ 고정 ===")
        for line in ctxfit.summary(ctx):
            print(f"    {line}")

        for v in valids:
            d = df[season == v]
            yv = d[TARGET_COL].to_numpy(dtype=np.float64)
            zb = (1 - MLP_ALPHA) * z["lgb"][v] + MLP_ALPHA * z["mlp"][v]
            p0 = shift_to_prior(1.0 / (1.0 + np.exp(-zb)), yv.mean())
            base = bss(yv, p0)

            adj, hits = logit_adjust(d, art)
            pn = shift_to_prior(1.0 / (1.0 + np.exp(-(zb + adj))), yv.mean())
            gain = bss(yv, pn) - base
            _, t = resid_t(yv - p0, adj)
            rows.append({"w": w, "v": v, "h": v - w, "base": base, "gain": gain,
                         "t": t, "hit": np.mean(adj != 0.0), "n": len(d)})
        print()

    if not rows:
        raise SystemExit("쓸 수 있는 체크포인트가 없다.")

    print("=" * 96)
    print("결과 — 창 안에서 δ는 **완전히 같다**. 다른 것은 검증 시즌뿐이다")
    print("=" * 96)
    print(f"  {'창':>6}{'검증':>6}{'h':>4}{'행':>10}{'기준BSS(신호량)':>16}"
          f"{'δ 이득':>10}{'잔차t':>8}{'적중':>7}")
    for r in rows:
        print(f"  W{r['w']:<5}{r['v']:>6}{r['h']:>4}{r['n']:>10,}{r['base']:>16.1f}"
              f"{r['gain']:>+10.2f}{r['t']:>8.1f}{r['hit']:>7.1%}")

    # ── P1: 창 고정, 신호량 대비 이득 ─────────────────────────────
    print("\n" + "=" * 96)
    print("P1 — 창을 고정했을 때 고신호 시즌에서 이득이 작은가")
    print("=" * 96)
    # 2023은 '저신호'가 아니라 '관계가 무너진' 시즌이라 극단값 대표로 쓰지 않는다.
    for w, _ in WINDOWS:
        sub = sorted([r for r in rows if r["w"] == w and r["v"] != 2023],
                     key=lambda r: -r["base"])
        if len(sub) < 2:
            print(f"\n  W{w} — 2023을 빼면 비교점이 1개뿐이다. 판정 불가")
            continue
        print(f"\n  W{w} (δ 동일, 2023 제외) — 신호량 내림차순:")
        for r in sub:
            print(f"    신호 {r['base']:>8.1f}  (h={r['h']})   이득 {r['gain']:>+7.2f}"
                  f"   잔차t {r['t']:>5.1f}")
        hi, lo = sub[0], sub[-1]
        # 가설: 신호량↑ → 이득↓.  교란 h: h↑ → 이득↓.
        # hi(고신호)의 h가 더 크면 **둘이 같은 예측**을 하므로 관측이 둘을 못 가른다.
        supported = hi["gain"] < lo["gain"]
        confounded = hi["h"] > lo["h"]
        if supported and not confounded:
            v = "✅ 알리바이 없는 지지 — h는 반대로 밀었는데도 이득이 작다"
        elif supported and confounded:
            v = "△ 지지 방향이지만 h가 같은 방향으로 민다 — **증거가 아니다**"
        elif not supported and not confounded:
            v = "△ 미지지 — 다만 h가 반대로 밀었으므로 **반증도 아니다**"
        else:
            v = "❌ 반증 — h가 도왔는데도 고신호 쪽 이득이 크다"
        print(f"    → 고신호 {hi['gain']:+.2f} (h={hi['h']}) vs "
              f"저신호 {lo['gain']:+.2f} (h={lo['h']})\n      {v}")

    # ── 창 간 차이가 시즌 간 차이보다 큰가 ─────────────────────────
    print("\n" + "=" * 96)
    print("부수 관측 — 무엇이 δ 이득을 가장 크게 움직이는가")
    print("=" * 96)
    by_w = {}
    for r in rows:
        by_w.setdefault(r["w"], []).append(r["gain"])
    for w, gs in sorted(by_w.items()):
        print(f"  W{w} (δ 학습 {w - 2018}시즌)   이득 {min(gs):+.2f} ~ {max(gs):+.2f}"
              f"   폭 {max(gs) - min(gs):.2f}")
    allg = [g for gs in by_w.values() for g in gs]
    within = max(max(gs) - min(gs) for gs in by_w.values())
    print(f"\n  창 **안**에서의 폭 (검증 시즌 효과)  최대 {within:.2f}")
    print(f"  창 **간**의 폭 (δ 학습량 효과)       {max(allg) - min(allg):.2f}")
    print("  → 배포 δ는 ≤2024(6시즌)로 여기 어느 창보다 많다. δ 학습량은 2025에"
          "\n    **유리한** 방향이므로 3.3배 축소를 설명하지 못한다.")

    # ── P3: 전체 점에서 신호량 vs 잔차t ───────────────────────────
    if len(rows) >= 3:
        b = np.array([r["base"] for r in rows])
        t = np.array([r["t"] for r in rows])
        g = np.array([r["gain"] for r in rows])
        print("\n" + "=" * 96)
        print(f"P3 — 전 {len(rows)}점에서의 상관 (표본이 적다. 방향만 읽는다)")
        print("=" * 96)
        print(f"  corr(신호량, 잔차t)  {np.corrcoef(b, t)[0,1]:+.3f}"
              f"     ← 음수면 가설 지지")
        print(f"  corr(신호량, δ이득)  {np.corrcoef(b, g)[0,1]:+.3f}")
        print(f"  corr(h, δ이득)       "
              f"{np.corrcoef([r['h'] for r in rows], g)[0,1]:+.3f}"
              f"     ← 교란 축의 크기")

    print("\n  ⚠️ 2023은 신호량 −10.2인 **망가진 시즌**이다 (§7.21).")
    print("     '저신호'가 아니라 '관계가 무너진' 것이라 저신호 대표로 읽으면 안 된다.")
    print("     2022(고신호) vs 2024(중신호) 비교가 이 표의 본체다.")


if __name__ == "__main__":
    main()
