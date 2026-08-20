"""홀드아웃 → LB 배율 보정 — **검증 설계를 바꾸지 않고 눈금만 고친다**

무엇이 아닌가
    fold 조합이나 가중을 LB에 맞추는 물건이 **아니다.** §7.21이 그걸 이미 채점해서
    떨어뜨렸다 — 세 설계가 전부 2승 1패였고, "개선된" 설계는 이 프로젝트에서 실제로
    점수를 깎은 유일한 실패 유형(피처 삭감)을 **무경고로 통과**시켰다.
    그리고 LB 몇 점에 검증 **설계**를 맞추는 것은 규칙 2(파라미터 역산)의 확장판이다.

무엇인가
    검증은 {2023, 2024} + 조건 ①②③ 그대로 둔다. 여기서 하는 것은
    **"이 계열의 변경은 홀드아웃이 몇 배로 틀리더라"** 를 표로 만들어 나누는 것뿐이다.
    s12에서 실제로 작동했다 — 예측 +2~+4, 실측 +3.08. 0승 5패 뒤 첫 적중이고
    직관 대신 **s11에서 잰 배율을 나눈 것**이 유일한 차이였다.

정직성 장치 (이게 이 파일의 본체다)
    · 표본이 2개 미만이면 **점 추정을 거부**하고 부호만 말한다
    · 같은 계열 안에서 배율이 2배 넘게 어긋나면 **거부**한다 (분산 축소가 여기 걸린다)
    · 조건 ③(크기비 ≤3)을 못 넘으면 **부호 보장 없음**을 경고한다 (Trackman이 그랬다)
    · `--loo` 로 자기 자신을 leave-one-out 채점한다. 과거 6번을 얼마나 맞혔을지

⛔ 배율을 **파라미터 상향 근거로 쓰지 말 것.** "30%만 실현됐으니 β를 올리자"는
   규칙 2 위반이고 논리도 틀렸다 — 이득이 β·cov − (β²/2)·var 꼴이면 실현 감소는
   cov 감소를 뜻하고, 그러면 최적 β는 **낮아진다.** 줄어든 것은 용량이 아니라 신호다.

실행:  .venv/Scripts/python.exe -u src/lbcalib.py --loo
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

# ── 관측 기록. 전부 SUBMISSIONS.md / MODEL.md에 근거가 있다 ───────────────
# (이름, 성격, fold2023, fold2024, LB실측)   fold 값이 없으면 None
HISTORY = [
    ("Trackman 피처 (s03)",  "정보추가", +2.30,  +13.30, -13.60),
    ("LGBM 다양성 (s05)",    "분산축소", None,    +5.90,  +6.16),
    ("피처 51개 (s06)",      "정보제거", +73.80, +15.20,  -2.13),
    ("MLP 도입 (s07)",       "새함수족", None,   +11.40, +29.50),
    ("MLP 3→8 (s08)",        "분산축소",  -3.93,  +6.61,  +0.24),
    ("α .30→.40 (s09)",      "재가중",    +5.88,  +1.16,  +6.70),
    ("noid .625 (s10)",      "재가중",    +5.13,  -0.70,  -4.97),
    ("플래툰 δ (s11)",       "정보추가", +22.04, +20.03,  +5.93),
    ("카운트 δ (s12)",       "정보추가",  +7.49,  +9.74,  +3.08),
    # 보조 헤드는 피처가 아니라 **학습 신호**를 늘린다. 그래도 "모델이 못 보던 정보를
    # 넣는다"는 점은 같아 정보추가로 분류했다 — 배율(3.73)도 같은 대역이다.
    ("보조 헤드 (s13)",      "정보추가", +20.55,  +9.91,  +2.66),
]

MIN_N = 2            # 계열당 최소 표본
MAX_SPREAD = 2.0     # 계열 안 배율이 이보다 어긋나면 거부
RATIO_GATE = 3.0     # 조건 ③ — fold 간 크기비
NOISE = 3.3          # fold2024 시드 노이즈 대역 (§7.27). fold2023도 같은 규모로 본다

# ⚠️ **비율 조건은 분자·분모가 둘 다 노이즈보다 클 때만 해석 가능하다.**
#    이 파일을 만들면서 드러난 사실이다. 게이트를 걸었더니 재가중 계열 표본이 0이 됐고,
#    이유를 보니 s09(+5.88 / +1.16)의 크기비 5.1이 **노이즈끼리의 비율**이었다.
#    +1.16은 ±3.3 안이다. 그런데 그 변경은 LB에서 +6.70을 줬다.
#    반대 방향의 같은 병도 있다 — `runner_on_1b`는 +0.20/+0.20으로 크기비 1.03이라
#    조건 ②③을 깨끗이 통과했지만 이득이 노이즈의 1/5이었다 (§7.37).
#    → 크기비를 읽기 전에 **양 fold가 노이즈보다 큰지부터** 본다.


def _ratio(h, lb):
    """홀드아웃 / LB. >1이면 홀드아웃이 과대평가, <1이면 과소평가."""
    return h / lb if lb != 0 else float("nan")


def _size_ratio(a, b):
    if a is None or b is None:
        return None
    return max(abs(a), abs(b)) / max(min(abs(a), abs(b)), 1e-9)


def verdict(f23, f24):
    """이 관측의 fold 값이 **해석 가능한가**. (해석가능, 사유) 를 돌려준다.

    순서가 중요하다 — 노이즈 판정이 크기비보다 **먼저**다.
    한쪽 fold가 노이즈면 크기비는 노이즈끼리의 비율이라 읽을 수 없다.
    """
    if f23 is None or f24 is None:
        return False, "fold 값 결측"
    small = [n for n, v in (("f2023", f23), ("f2024", f24)) if abs(v) < NOISE]
    if small:
        return False, f"{'·'.join(small)}가 노이즈 대역(±{NOISE}) 안 → 크기비 해석 불가"
    if (f23 > 0) != (f24 > 0):
        return False, "fold 부호 불일치 (조건 ②)"
    sr = _size_ratio(f23, f24)
    if sr > RATIO_GATE:
        return False, f"크기비 {sr:.1f} > {RATIO_GATE} (조건 ③)"
    return True, ""


def calibration(history=None):
    """계열 → {"ratios": [...], "usable": bool, "reason": str}

    **해석 가능한 관측만** 배율 추정에 쓴다 (`verdict` 참조).
    나머지는 버리지 않고 `excluded`에 사유와 함께 남긴다 — 어떤 관측이 왜
    빠졌는지가 이 표를 읽는 사람에게 가장 중요한 정보다.
    """
    out = {}
    for name, kind, f23, f24, lb in (history or HISTORY):
        d = out.setdefault(kind, {"ratios": [], "names": [], "excluded": []})
        ok, why = verdict(f23, f24)
        if not ok:
            d["excluded"].append((name, why, f24, lb))
            continue
        d["ratios"].append(_ratio(f24, lb))
        d["names"].append(name)

    for kind, d in out.items():
        r = [x for x in d["ratios"] if np.isfinite(x)]
        d["n"] = len(r)
        if len(r) < MIN_N:
            d["usable"], d["reason"] = False, f"표본 {len(r)}개 (최소 {MIN_N})"
        elif min(x > 0 for x in r) == 0 and max(x > 0 for x in r) == 1:
            d["usable"], d["reason"] = False, "계열 안에서 부호가 갈린다"
        else:
            spread = max(abs(x) for x in r) / max(min(abs(x) for x in r), 1e-9)
            d["spread"] = spread
            if spread > MAX_SPREAD:
                d["usable"] = False
                d["reason"] = f"배율이 {spread:.1f}배 어긋난다 ({min(r):.2f}~{max(r):.2f})"
            else:
                d["usable"], d["reason"] = True, ""
                d["ratio"] = float(np.mean(r))
                d["lo"], d["hi"] = float(min(r)), float(max(r))
    return out


def predict_lb(f2023, f2024, kind, cal=None):
    """(예측 문자열, 점추정 또는 None). 근거가 없으면 점 추정을 **거부**한다."""
    cal = cal or calibration()
    d = cal.get(kind)

    # 질의 자체가 해석 불가면 **경고가 아니라 거부**다.
    # (예전 판은 경고만 하고 점 추정을 냈다. LOO에서 Trackman에 +4.07을 내놓고
    #  실측 −13.60과 부호가 어긋났다 — 경고로는 못 막는다.)
    ok, why = verdict(f2023, f2024)
    if not ok:
        return (f"❌ 점 추정 거부 — {why}\n"
                f"   이 상태에서 홀드아웃 크기를 LB로 환산할 근거가 없다."), None

    if d is None:
        return f"❌ '{kind}' 계열의 관측이 없다. 예측 불가", None
    warn = []
    if not d["usable"]:
        sign = "양수" if f2024 and f2024 > 0 else "음수"
        return "\n".join(warn + [
            f"❌ 점 추정 거부 — {d['reason']}",
            f"   말할 수 있는 것: 부호만. 홀드아웃이 {sign}이므로 LB도 같은 부호로 본다"
            f" (조건 ②③를 통과했다면)"]), None

    pt = f2024 / d["ratio"]
    lo, hi = sorted([f2024 / d["hi"], f2024 / d["lo"]])
    body = [f"예측 {pt:+.2f}   (관측 배율 {d['lo']:.2f}~{d['hi']:.2f} → {lo:+.2f} ~ {hi:+.2f})",
            f"   근거: '{kind}' 계열 표본 {d['n']}개, 평균 배율 {d['ratio']:.2f}"]
    if d["n"] < 3:
        body.append(f"   ⚠️ 표본 {d['n']}개다. 밴드는 신뢰구간이 아니라 **관측 범위**일 뿐이다")
    return "\n".join(warn + body), pt


def loo():
    """자기 자신을 leave-one-out 채점 — 이 함수가 과거를 얼마나 맞혔을 것인가."""
    print("=" * 92)
    print("LOO 채점 — 각 제출을 **자기를 뺀 나머지**로 예측했다면")
    print("=" * 92)
    print(f"  {'변경':<22}{'성격':<10}{'f2024':>8}{'LB실측':>9}{'예측':>10}"
          f"{'오차':>8}   비고")
    err, hit = [], 0
    for i, (name, kind, f23, f24, lb) in enumerate(HISTORY):
        rest = [h for j, h in enumerate(HISTORY) if j != i]
        txt, pt = predict_lb(f23, f24, kind, calibration(rest))
        if pt is None:
            note = txt.split("\n")[-1].strip()[:44] if "❌" in txt else "거부"
            note = "거부 — " + (txt.split("—")[-1].split("\n")[0].strip()[:36]
                               if "—" in txt else "근거 없음")
            print(f"  {name:<22}{kind:<10}{f24 if f24 is not None else float('nan'):>8.2f}"
                  f"{lb:>9.2f}{'—':>10}{'—':>8}   {note}")
            continue
        e = pt - lb
        err.append(abs(e))
        if (pt > 0) == (lb > 0):
            hit += 1
        print(f"  {name:<22}{kind:<10}{f24:>8.2f}{lb:>9.2f}{pt:>+10.2f}{e:>+8.2f}"
              f"   {'부호 ✓' if (pt > 0) == (lb > 0) else '부호 ✗'}")
    if err:
        print(f"\n  점 추정을 낸 {len(err)}건 — 평균 절대오차 {np.mean(err):.2f} BSS,"
              f" 부호 적중 {hit}/{len(err)}")
    print("\n  ⚠️ LOO도 표본 9개 위의 이야기다. 계열당 2~3개뿐이라")
    print("     '이 함수가 검증됐다'가 아니라 **'지금까지는 어긋나지 않았다'** 로 읽어야 한다.")


def main():
    cal = calibration()
    print("=" * 92)
    print("계열별 배율 (홀드아웃 fold2024 ÷ LB 실측)")
    print("=" * 92)
    print(f"  {'성격':<10}{'표본':>5}{'배율':>18}   판정")
    for kind, d in sorted(cal.items(), key=lambda kv: -kv[1].get("n", 0)):
        rng = (f"{d['lo']:.2f} ~ {d['hi']:.2f}" if d.get("usable")
               else ", ".join(f"{x:.2f}" for x in d["ratios"]) or "—")
        print(f"  {kind:<10}{d.get('n', 0):>5}{rng:>18}   "
              f"{'✅ 사용 가능' if d.get('usable') else '❌ ' + d['reason']}")
        for nm, why, f24, lb in d["excluded"]:
            f24s = "—" if f24 is None else f"{f24:+.1f}"
            print(f"       · 제외 {nm}: {why}   (홀드아웃 {f24s} → LB {lb:+.1f})")

    print("\n" + "=" * 92)
    print("관측 전수 — 해석 가능한 것이 몇 개인가")
    print("=" * 92)
    print(f"  {'변경':<22}{'f2023':>8}{'f2024':>8}{'크기비':>8}{'LB':>8}   판정")
    n_ok = 0
    for name, kind, f23, f24, lb in HISTORY:
        ok, why = verdict(f23, f24)
        n_ok += ok
        sr = _size_ratio(f23, f24)
        print(f"  {name:<22}"
              f"{'—' if f23 is None else f'{f23:+.2f}':>8}"
              f"{'—' if f24 is None else f'{f24:+.2f}':>8}"
              f"{'—' if sr is None else f'{sr:.1f}':>8}{lb:>+8.2f}   "
              f"{'✅ 해석 가능' if ok else '· ' + why}")
    print(f"\n  → 9개 중 **{n_ok}개만** 크기비를 읽을 수 있다.")
    print("     비율 조건(②③)은 분자·분모가 둘 다 노이즈보다 클 때만 성립한다.")
    print("     한쪽이 노이즈면 크기비 5.1(s09)도 1.03(runner_on_1b)도 똑같이 무의미하다.")

    if "--loo" in sys.argv:
        print()
        loo()


if __name__ == "__main__":
    main()
