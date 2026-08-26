"""구종 예측기 자체의 성능 — "구종을 잘 맞히는가"에 대한 직접 답 (§7.169 부록)

trackman 2024 홀드아웃에서 3그룹(fastball/breaking/offspeed) 분류 성능.
비교선: ① 무조건 다수 클래스 ② 투수 커리어 믹스 argmax ③ full 모델(§7.58 설정).

실행:  .venv/Scripts/python.exe -u src/pitchacc.py 2>&1 | tee experiments/pitchacc.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import lightgbm as lgb
import numpy as np

import pitchtype as PT

FOLD = 2024


def report(name, y, p):
    pred = p.argmax(1)
    acc = float((pred == y).mean())
    ll = PT.mlogloss(y, p)
    print(f"\n  {name:<22} 정확도 {acc:.4f}   logloss {ll:.4f}")
    for i, g in enumerate(PT.GROUPS):
        m = y == i
        rec = float((pred[m] == i).mean())
        prec_d = pred == i
        prec = float((y[prec_d] == i).mean()) if prec_d.any() else 0.0
        print(f"    {g:<10} 실제비중 {m.mean():.3f}  재현율 {rec:.3f}  정밀도 {prec:.3f}")
    return acc


def main():
    tm = PT.load_trackman()
    ok = tm["y"].to_numpy() >= 0
    fit = ok & (tm["season"].to_numpy() < FOLD)
    hld = ok & (tm["season"].to_numpy() == FOLD)
    yf = tm.loc[fit, "y"].to_numpy()
    yh = tm.loc[hld, "y"].to_numpy()
    league = np.array([float((yf == i).mean()) for i in range(3)])
    print(f"학습 {fit.sum():,}행 (<{FOLD})  홀드아웃 {hld.sum():,}행 ({FOLD} 시즌)")

    xh = PT.tm_matrix(tm[hld], league)
    r3h = xh[:, 6:9]

    report("① 다수 클래스(직구)", yh, np.tile(league, (len(yh), 1)))
    report("② 커리어 믹스 argmax", yh, r3h / np.maximum(r3h.sum(1, keepdims=True), 1e-9))

    xf = PT.tm_matrix(tm[fit], league)
    m_full = lgb.train(PT.PT_PARAMS, lgb.Dataset(xf, label=yf), PT.PT_ROUNDS)
    p = m_full.predict(xh)
    report("③ full 모델(§7.58)", yh, p)

    # 확신도별 — 확률이 높다고 말할 때는 실제로 맞는가 (캘리브레이션 요약)
    conf = p.max(1)
    pred = p.argmax(1)
    print("\n  확신도별 적중률 (full 모델)")
    for lo, hi in [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        print(f"    p∈[{lo:.1f},{hi:.1f})  행 {m.mean():>6.1%}  적중 {float((pred[m]==yh[m]).mean()):.3f}")

    # 카운트별 — 유리/불리 카운트에서 얼마나 달라지나
    b = tm.loc[hld, "balls_before"].to_numpy()
    s2 = tm.loc[hld, "strikes_before"].to_numpy()
    print("\n  카운트별 정확도 (full / 커리어믹스)")
    mix_pred = (r3h / np.maximum(r3h.sum(1, keepdims=True), 1e-9)).argmax(1)
    for tag, m in [("0-0", (b == 0) & (s2 == 0)), ("타자유리(3-x)", b == 3),
                   ("투수유리(x-2)", s2 == 2)]:
        print(f"    {tag:<12} {float((pred[m]==yh[m]).mean()):.3f} / "
              f"{float((mix_pred[m]==yh[m]).mean()):.3f}   (행 {m.sum():,})")


if __name__ == "__main__":
    main()
