"""라우팅 ray 위 실측점들로 **포물선을 적합해 꼭짓점을 찾는다.**

혜진 §15-9 는 실측 2점(t=0, t=1)뿐이라 곡률 `C_eff` 를 가정해야 했고, 가정에 따라
꼭짓점이 R 0.031~0.084 로 갈렸다. 3점 이상이면 가정 없이 적합된다.

ray:  R(t) = 0.10 + 0.075·t      F(t) = 0.40 + 0.10·t
모형: Score(t) = s0 + L·t − C·t²      (위로 볼록이면 C > 0)

실행:  .venv/Scripts/python.exe src/rtvertex.py  t1=score1  t2=score2 ...
예:    ... src/rtvertex.py 0=1116.788371096 1=1115.2181199053 -0.25=1117.0
"""
import sys

import numpy as np

R0, F0, DR, DF = 0.10, 0.40, 0.075, 0.10


def fit(ts, ys):
    ts, ys = np.asarray(ts, float), np.asarray(ys, float)
    if len(ts) < 3:
        raise SystemExit(f"점이 {len(ts)}개다. 포물선에는 3개 이상이 필요하다 "
                         "— 이것이 §15-9 가 곡률을 가정해야 했던 이유다.")
    A = np.vstack([np.ones_like(ts), ts, ts**2]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    s0, L, negC = coef
    C = -negC
    resid = ys - A @ coef
    return s0, L, C, resid


def main(argv):
    pts = {}
    for a in argv:
        t, y = a.split("=")
        pts[float(t)] = float(y)
    ts = sorted(pts)
    ys = [pts[t] for t in ts]
    print(f"{'t':>8}{'R':>9}{'F':>9}{'LB':>18}")
    for t in ts:
        print(f"{t:>8.3f}{R0+DR*t:>9.4f}{F0+DF*t:>9.4f}{pts[t]:>18.6f}")

    s0, L, C, resid = fit(ts, ys)
    print(f"\n적합  Score(t) = {s0:.6f} + {L:+.6f}·t − {C:+.6f}·t²")
    print(f"잔차  max|r| = {np.abs(resid).max():.6f}   (점 {len(ts)}개, 자유도 {len(ts)-3})")
    if C <= 0:
        print("\n⚠️ C ≤ 0 — 위로 볼록이 아니다. 포물선 모형이 이 구간에서 안 맞는다.")
        return
    t_star = L / (2 * C)
    peak = s0 + L * t_star - C * t_star**2
    print(f"\n꼭짓점  t* = {t_star:+.4f}   R* = {R0+DR*t_star:.4f}   F* = {F0+DF*t_star:.4f}")
    print(f"        예상 LB = {peak:.6f}   (t=0 대비 {peak-pts.get(0.0, s0):+.4f})")
    best_t = ts[int(np.argmax(ys))]
    print(f"        실측 최고 t={best_t:+.3f} 대비 {peak-pts[best_t]:+.4f}")
    if not (min(ts) <= t_star <= max(ts)):
        print("        ⚠️ 꼭짓점이 실측 구간 **밖**이다 — 외삽이다. 한 점 더 재는 편이 안전하다.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
