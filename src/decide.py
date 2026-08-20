"""채택 판정 — 조건 ①②③을 **올바른 순서로** 적용하는 단일 함수

왜 필요한가
    조건 ①②③을 34번의 실험 동안 **나란히 세 개**로 써왔다. 과거 관측 9개에
    실제로 적용해보니 7개가 해석 불가였고, 실패가 양방향이었다.

        runner_on_1b  +0.20 / +0.20   크기비 1.03 → ②③ 통과   이득이 노이즈의 1/5
        α .30→.40     +5.88 / +1.16   크기비 5.1  → ③ 실패     LB에서 +6.70

    둘 다 **한쪽 fold가 노이즈 대역 안**이었다. 비율의 분모가 노이즈면 비율은
    아무 값이나 나온다 — 작으면 우연히 1.0이 나오고(가짜 통과), 0에 가까우면
    5.1이 나온다(가짜 탈락).

    → **조건 ①은 ②③과 나란한 세 번째가 아니라 그 아래 전제다.**
      ①을 먼저 걸고, 통과한 것에만 ②③을 적용한다.

이 파일이 LB를 예측하지 않는 이유
    할 수 없다. (홀드아웃 → LB) 쌍이 9개뿐이고 그중 해석 가능한 것이 2개다
    (`src/lbcalib.py` 참조). 2점에 맞춘 배율은 LB로 파라미터를 역산하는 것과 같다.
    **여기서 하는 것은 홀드아웃이 아는 것과 모르는 것을 정직하게 가르는 일뿐이다.**
    fold2024의 부호 적중은 6전 5승이다 — 부호는 이미 쓸 만하고, 고장난 것은 절차다.

판정 불가일 때 **값을 매긴다**
    "모르겠다"로 끝내지 않고 **얼마를 더 쓰면 알 수 있는지**를 계산한다.
    시드 노이즈는 시드 수의 √에 반비례하므로 필요한 시드 수가 나온다.
    대개 그 답은 "수백 개" 이고, 그러면 **포기가 정답이라는 것을 숫자로** 알 수 있다.

실행:  .venv/Scripts/python.exe -u src/decide.py            (과거 관측 재판정)
사용:  from decide import decide, paired_se
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

SEED_NOISE = {2023: 3.3, 2024: 3.3}   # fold별 시드 편차 (§7.3 · §7.27). 3시드 기준
N_SEEDS = 3
RATIO_GATE = 3.0
SCALE_HINT = 100.0                    # 조건 ④ — "이 정도는 돼야 LB에서 보인다"


def paired_se(y, p_old, p_new):
    """쌍체 SE — 같은 행의 두 예측이므로 표본변동이 차분에서 소거된다.

    시드 노이즈와는 **다른 축**이다. 이건 '이 행들에서 우연인가'이고,
    시드 노이즈는 '다른 시드로 학습했으면 달랐을까'다. 둘 다 넘어야 한다.
    """
    y = np.asarray(y, dtype=np.float64)
    ref = y.mean() * (1 - y.mean())
    d = (np.asarray(p_new) - y) ** 2 - (np.asarray(p_old) - y) ** 2
    c = 100000.0 / ref
    return -c * d.mean(), c * d.std(ddof=1) / np.sqrt(len(d))


def seeds_needed(delta, fold, target_sigma=2.0):
    """이 크기의 효과를 target_sigma로 분해하려면 시드가 몇 개 필요한가.

    시드 노이즈 σ는 √n에 반비례한다 → n = N_SEEDS · (σ_now · target / |Δ|)²
    """
    if delta == 0:
        return float("inf")
    need = N_SEEDS * (SEED_NOISE[fold] * target_sigma / abs(delta)) ** 2
    return int(np.ceil(need))


def decide(name, d23, d24, se23=None, se24=None, verbose=True):
    """(판정, 사유, 상세) — 조건을 **순서대로** 적용한다.

    d23/d24  : fold별 BSS 이득
    se23/se24: 쌍체 SE (있으면 조건 ①에 같이 쓴다. 없으면 시드 노이즈만으로 판정)
    """
    lines, detail = [], {}

    # ── ① 크기 게이트 — 이게 먼저다 ────────────────────────────────
    blocked = []
    for f, d, se in ((2023, d23, se23), (2024, d24, se24)):
        if d is None:
            blocked.append(f"fold{f} 값 없음")
            continue
        if abs(d) < SEED_NOISE[f]:
            n = seeds_needed(d, f)
            blocked.append(f"fold{f} |{d:+.2f}| < 시드 노이즈 ±{SEED_NOISE[f]}"
                           f"  (2σ로 가르려면 시드 {n:,}개 — 현재 {N_SEEDS}개)")
            detail[f"seeds_{f}"] = n
        elif se is not None and abs(d) < 2 * se:
            blocked.append(f"fold{f} |{d:+.2f}| < 2·쌍체SE({se:.2f})")
    if blocked:
        return "판정 불가", "조건 ① 미통과 — **②③을 읽으면 안 된다**", lines + blocked

    # ── ② 부호 ────────────────────────────────────────────────────
    if (d23 > 0) != (d24 > 0):
        return "기각", f"조건 ② — fold 부호 불일치 ({d23:+.2f} / {d24:+.2f})", lines

    # ── ③ 크기비 (①을 통과했으므로 이제 해석 가능하다) ──────────────
    ratio = max(abs(d23), abs(d24)) / min(abs(d23), abs(d24))
    detail["ratio"] = ratio
    if ratio > RATIO_GATE:
        return "기각", (f"조건 ③ — 크기비 {ratio:.1f} > {RATIO_GATE}. "
                      f"한 fold가 끌고 가는 효과다"), lines
    if d24 < 0:
        return "기각", f"양 fold 음수 ({d23:+.2f} / {d24:+.2f})", lines

    # ── ④ 규모 (참고. 기각 사유가 아니다) ──────────────────────────
    small = min(d23, d24) < SCALE_HINT
    note = (f"⚠️ 조건 ④ 미달 (min {min(d23, d24):+.1f} < {SCALE_HINT:.0f}) — "
            f"기각 사유는 아니다. 이 규모가 전이된 전례가 넷 있다"
            if small else "")
    return "채택 가능", (f"①②③ 통과. 크기비 {ratio:.2f}" +
                     (f"\n     {note}" if note else "")), lines


# 과거 관측 — (이름, f2023, f2024, LB실측, 당시 판정)
PAST = [
    ("Trackman 피처 (s03)",   +2.30,  +13.30, -13.60, "채택했다"),
    ("피처 51개 (s06)",       +73.80, +15.20,  -2.13, "채택했다"),
    ("MLP 3→8 (s08)",         -3.93,  +6.61,  +0.24, "채택했다"),
    ("α .30→.40 (s09)",       +5.88,  +1.16,  +6.70, "채택했다"),
    ("noid .625 (s10)",       +5.13,  -0.70,  -4.97, "채택했다(측정 목적)"),
    ("플래툰 δ (s11)",        +22.04, +20.03,  +5.93, "채택했다"),
    ("카운트 δ (s12)",        +7.49,  +9.74,  +3.08, "채택했다"),
    ("runner_on_1b (§7.37)",  +0.20,  +0.20,   None, "기각했다"),
    ("주자유무 (§7.37)",      +0.57,  +0.62,   None, "기각했다"),
    ("count12 (§7.37)",       +8.09,  +7.68,   None, "기각했다(과전이)"),
]


def main():
    print("=" * 100)
    print("과거 관측 재판정 — 조건을 **순서대로** 걸었다면 무엇이 달랐나")
    print("=" * 100)
    print(f"  {'변경':<24}{'f2023':>8}{'f2024':>8}{'LB':>8}  {'판정':<10} 사유")
    agree = disagree = 0
    detail_rows = []
    for name, a, b, lb, was in PAST:
        v, why, extra = decide(name, a, b)
        lbs = "—" if lb is None else f"{lb:+.2f}"
        print(f"  {name:<24}{a:>8.2f}{b:>8.2f}{lbs:>8}  {v:<10} {why.splitlines()[0]}")
        for e in extra:
            print(f"  {'':<48}   · {e}")
        # ⚠️ 판정 불가는 세지 않는다. 아무 말도 안 한 것을 '맞혔다'로 세면
        #    입을 다물수록 성적이 오른다 — 이 표를 무의미하게 만드는 지름길이다.
        if lb is not None and v in ("채택 가능", "기각"):
            good = (v == "채택 가능") == (lb > 0)
            agree += good
            disagree += not good
            detail_rows.append((name, v, lb, good))
    print(f"\n  **판정을 낸 {agree + disagree}건** 중 결과와 일치 {agree} / 불일치 {disagree}")
    for nm, v, lb, good in detail_rows:
        print(f"      {'✓' if good else '✗'} {nm:<24}{v:<8}LB {lb:+.2f}")
    n_unknown = sum(1 for _, a, b, lb, _ in PAST
                    if lb is not None and decide("", a, b)[0] == "판정 불가")
    print(f"  나머지 {n_unknown}건은 **'모른다'**고 말했을 것이다 (Trackman −13.6 / "
          f"α +6.70 / noid −4.97)")
    print("      → 둘은 손실을 막았을 것이고, 하나(α +6.70)는 기회를 놓치게 했을 것이다.")

    print("\n" + "=" * 100)
    print("이 순서가 실제로 바꾸는 것")
    print("=" * 100)
    print("  · runner_on_1b / 주자유무 — 예전 기준으로는 ②③ 통과. 지금은 ①에서 멈춘다")
    print("  · α .30→.40 / noid       — 예전엔 크기비로 기각됐다. 지금은 '판정 불가'다")
    print("    → **기각과 판정 불가는 다르다.** 전자는 '아니다', 후자는 '이 도구로는 모른다'.")
    print("       α는 LB에서 +6.70이었다. 기각으로 적어뒀으면 그 축을 닫았을 것이다.")
    print("\n  · 판정 불가에는 값이 붙는다 — 필요한 시드 수. 대개 수백 개고,")
    print("    그러면 **포기가 정답이라는 것을 숫자로** 알 수 있다.")
    print("\n  ⚠️ 이 함수가 못 보는 것: **β 같은 강도 파라미터에 대한 강건성.**")
    print("     count12는 여기서 '채택 가능'이 나오지만 §7.37은 기각했다 —")
    print("     β=2.5에서 양 fold 음수로 부러지기 때문이다. 그건 별도 스윕으로 본다.")


if __name__ == "__main__":
    main()
