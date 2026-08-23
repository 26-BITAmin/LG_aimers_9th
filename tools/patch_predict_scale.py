#!/usr/bin/env python3
"""확률공간 아핀 패치 — `predict.py` 에 사후 스케일·레벨 이동을 넣는다.

수식적 근거 (`reports/11` §2-2·§4-5 · wonwoo MODEL.md §7.85)
-----------------------------------------------------------
y ∈ {0,1} 이므로 y² = y 이고, Brier 는 항등식으로 분해된다:

    BS = Var(p) − 2·Cov(p,y) + (p̄ − r)² + r(1−r)
         └─────────┬────────┘   └───┬───┘   └──┬──┘
              refinement            레벨     상수

예측을 `p' = P0 + c·(p − P0) + Δ` 로 아핀 변환하면 두 노브가 **서로 다른 항**만 건드린다.

`c` — refinement 항.  Var → c²Var, Cov → c·Cov 이므로

    목적함수 = 2c·Cov − c²·Var          →  최적  c* = Cov(p,y) / Var(p)
    이득     = Var·(c* − 1)²

**c 에 대해 정확히 2차다** (α 축은 σ'(z) 가 좁은 구간에서 일정하다는 근사에 기댔지만
여기는 근사가 없다). 따라서 현 챔피언(c=1) 을 포함해 3점이면 꼭짓점이 확정된다.

`Δ` — 레벨 항.  Var 도 Cov 도 안 바꾸고 p̄ 만 옮기므로

    BS' − BS = 2Δ·b + Δ²                (b ≡ p̄ − r)

**Δ² 의 계수가 정확히 1이다** — 적합할 게 없다. 그래서 LB **한 점**이면 b 가 확정된다:

    b = − (ΔScore / K + Δ²) / (2Δ)      K = 1e5 / (r(1−r)) ≈ 400,700

Δ 를 목표 레벨에 맞춰 잡으면 최종 평균이 P0·c 와 무관해지므로 **두 축은 정확히 직교한다**
(reports/11 §3-4 가 실측한 refinement 축끼리의 상호작용이 여기엔 없다).

왜 로짓 공간이 아니라 확률 공간인가
-----------------------------------
wonwoo §7.19 는 `z' = c·z` (로짓) 로 쟀고 두 fold 가 반대를 가리켜 기각됐다.
확률 공간이 세 가지로 낫다:

  1. 곡선이 **정확히** 2차다
  2. 평균 이동이 P0 기준으로만 일어나므로 prior shift 를 다시 풀 필요가 없다
  3. `train.py` 를 안 건드린다 — `predict.py` 네 줄이 전부다

행 독립성
---------
`P0` · `c` · `Δ` 는 **학습 시점에 동결된 스칼라 리터럴**로 파일에 박힌다.
배치 평균을 쓰면 각 행이 다른 행에 의존하게 되어 대회 규정 위반이다 — 쓰지 않는다.
기존 `np.clip(p, CLIP_EPS, 1-CLIP_EPS)` 가 범위를 지킨다.

사용법
------
    python tools/patch_predict_scale.py --scale 1.05 --shift -0.0051 --center 0.4801 \
        --src ~/Aimers_9th_champion/src/predict.py --out /tmp/predict_lv.py

실제 제출 빌드는 `notebooks/11_constant_sweep_colab.ipynb` 의 빌더 셀이 같은 로직을
인라인으로 수행한다. 이 파일은 **로컬 검증용**이다 — 둘이 어긋나면 안 된다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# 원우 predict.py 의 앵커. 바뀌면 즉시 실패해야 한다 (조용히 패치 누락 방지).
ANCHOR = """        # prior shift — 학습 때 고정된 스칼라(base+delta)의 행 단위 적용
        p = 1.0 / (1.0 + np.exp(-(z + self.delta)))
        return np.clip(p, CLIP_EPS, 1 - CLIP_EPS)"""

TEMPLATE = """        # prior shift — 학습 때 고정된 스칼라(base+delta)의 행 단위 적용
        p = 1.0 / (1.0 + np.exp(-(z + self.delta)))

        # ── 확률공간 아핀 (hyejin 패치) ───────────────────────────────
        # BS = Var(p) − 2·Cov(p,y) + (p̄−r)² + r(1−r) 에서
        # p' = P0 + c(p−P0) 은 Var → c²Var, Cov → c·Cov 이므로
        # 목적함수 2c·Cov − c²·Var 의 최적이 c* = Cov/Var 이다.
        # 그리고 p' = p + Δ 는 Var·Cov 를 안 건드리고 (p̄−r)² 만 옮기므로
        # BS' − BS = 2Δ·b + Δ² 이고 Δ² 의 계수가 정확히 1이다 (LB 한 점이면 b 확정).
        # 세 값 모두 **동결 스칼라**다 — 배치 평균을 쓰면
        # 각 행이 다른 행에 의존해 대회 규정(행 독립 예측) 위반이 된다.
        PROB_CENTER = {center!r}
        PROB_SCALE = {scale!r}
        PROB_SHIFT = {shift!r}
        if PROB_SCALE != 1.0 or PROB_SHIFT != 0.0:
            p = PROB_CENTER + PROB_SCALE * (p - PROB_CENTER) + PROB_SHIFT

        return np.clip(p, CLIP_EPS, 1 - CLIP_EPS)"""


def make_patched(src_text: str, scale: float, center: float, shift: float = 0.0) -> str:
    if ANCHOR not in src_text:
        raise SystemExit(
            "predict.py 앵커를 찾지 못했다 — 원우가 이 부분을 고쳤을 수 있다.\n"
            "  patch_predict_scale.py 의 ANCHOR 를 현재 파일에 맞춰 갱신할 것."
        )
    if src_text.count(ANCHOR) != 1:
        raise SystemExit(f"앵커가 {src_text.count(ANCHOR)}번 나온다 — 1번이어야 한다")
    if not (0.5 <= scale <= 2.0):
        raise SystemExit(f"scale {scale} 가 상식 범위(0.5~2.0) 밖이다")
    if not (0.3 <= center <= 0.7):
        raise SystemExit(f"center {center} 가 상식 범위(0.3~0.7) 밖이다")
    if not (-0.05 <= shift <= 0.05):
        raise SystemExit(f"shift {shift} 가 상식 범위(±0.05) 밖이다")
    return src_text.replace(ANCHOR, TEMPLATE.format(scale=float(scale),
                                                    center=float(center),
                                                    shift=float(shift)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True, help="원우 src/predict.py")
    ap.add_argument("--out", type=Path, required=True, help="패치본 저장 경로")
    ap.add_argument("--scale", type=float, required=True, help="c. 1.0 이면 무변경")
    ap.add_argument("--center", type=float, required=True, help="P0. 동결 스칼라")
    ap.add_argument("--shift", type=float, default=0.0, help="Δ. 0.0 이면 무변경")
    a = ap.parse_args()

    out = make_patched(a.src.read_text(encoding="utf-8"), a.scale, a.center, a.shift)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(out, encoding="utf-8")

    # 자기 검증 — 패치가 실제로 들어갔고 문법이 살아 있는가
    import ast
    ast.parse(out)
    assert f"PROB_SCALE = {float(a.scale)!r}" in out
    assert f"PROB_CENTER = {float(a.center)!r}" in out
    assert f"PROB_SHIFT = {float(a.shift)!r}" in out
    print(f"{a.out}  (c={a.scale}, Δ={a.shift:+g}, P0={a.center}, "
          f"{len(out.splitlines())}줄)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
