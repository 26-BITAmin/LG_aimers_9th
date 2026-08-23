"""`FSPEC_ALPHA` 축 — **곡률을 라벨 없이 계산해 1슬롯으로 꼭짓점을 잡는다**
(`MODEL.md §7.131`).

§7.130 의 규칙: *앙상블을 고정한 채 상수만 바꾸는 변경은 draw 노이즈가 0이라
LB 차이가 그대로 효과다.* `fspec_alpha` 는 `artifacts.joblib` 의 스칼라 하나이고
**재학습이 없다** — 이 성질을 갖는다.

`p(a)` 를 `a` 로 전개하면

    BS(a) = BS₀ + (a−a₀)·(A₁ − 2B₁) + (a−a₀)²·A₂ + O(3)
    A₁ = mean(2·p₀·d),  A₂ = mean(d²),  B₁ = mean(y·d),   d = dp/da |_{a₀}

`A₁·A₂` 는 **라벨이 필요 없다** (더미에서 계산). `B₁` 만 LB 한 점이 준다:

    A₁ − 2B₁ = [−ΔScore/K − (a₁−a₀)²·A₂] / (a₁−a₀)
    a* = a₀ − (A₁ − 2B₁) / (2·A₂)

§7.120 의 레벨 축과 같은 구조다. 다만 레벨 축은 `Δ²` 계수가 **정확히 1**이었고
여기는 `A₂` 를 재야 한다 — 그리고 `p(a)` 가 `a` 에 정확히 선형이 아니므로
**3차 잔차를 먼저 확인한다.**
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "submit" / "model"


def curves(n=245789):
    import joblib
    import lightgbm as lgb
    import predict as P
    from build_submit import make_dummy_test
    from features import build_features, logit_adjust

    b = joblib.load(MODEL / "artifacts.joblib")
    art, cols = b["artifacts"], b["feature_cols"]
    a0 = float(b["fspec_alpha"])
    d = make_dummy_test(art=art).iloc[:n].reset_index(drop=True)

    pr = P.Predictor(str(MODEL))
    X = build_features(d, art)
    if art.get("use_player_ids"):
        for c in ("pitcher_id", "batter_id"):
            X[c] = d[c].to_numpy(dtype=np.float32)
    X = X[cols]

    z = np.zeros(len(X))
    for bst in pr.boosters:
        z += bst.predict(X, raw_score=True)
    z /= len(pr.boosters)
    if pr.alpha > 0:
        zm = np.zeros(len(X))
        for m, w in zip(pr.mlps, pr.mlp_w):
            zm += w * m.logit(X)
        z = (1 - pr.alpha) * z + pr.alpha * zm

    is_f = d["game_type"].to_numpy().astype(str) == "F"
    zf = np.zeros(int(is_f.sum()))
    for bst in pr.fspec:
        zf += bst.predict(X.iloc[np.flatnonzero(is_f)], raw_score=True)
    zf /= len(pr.fspec)

    zadj, _ = logit_adjust(d, art)

    def prob(a):
        zz = z.copy()
        zz[is_f] = (1 - a) * zz[is_f] + a * zf
        p = 1 / (1 + np.exp(-(zz + zadj + pr.delta)))
        p = P.PROB_CENTER + P.PROB_SCALE * (p - P.PROB_CENTER) + P.PROB_SHIFT
        return np.clip(p, 1e-6, 1 - 1e-6)

    return a0, prob, is_f


def main():
    import predict as P
    a0, prob, is_f = curves()
    r = P.PROB_TARGET
    K = 1e5 / (r * (1 - r))
    p0 = prob(a0)
    h = 1e-4
    dd = (prob(a0 + h) - prob(a0 - h)) / (2 * h)
    A1 = float(np.mean(2 * p0 * dd))
    A2 = float(np.mean(dd * dd))
    print(f"배포 FSPEC_ALPHA = {a0}   F 비중 {is_f.mean():.4f}   K = {K:,.0f}")
    print(f"  A1 = mean(2·p0·d) = {A1:.6e}")
    print(f"  A2 = mean(d²)     = {A2:.6e}    ->  곡률 K·A2 = {K*A2:,.1f} BSS/단위²")

    print("\n선형성 검사 — p(a) − p0 이 (a−a0)·d 와 얼마나 다른가")
    print(f"  {'a':>6s}{'최대 |비선형 잔차|':>20s}{'상대':>10s}")
    for a in (0.05, 0.10, 0.20, 0.30, 0.50):
        res = prob(a) - p0 - (a - a0) * dd
        lin = np.abs((a - a0) * dd)
        print(f"  {a:>6.2f}{np.abs(res).max():>20.3e}{np.abs(res).max()/max(lin.max(),1e-12):>10.2%}")

    print("\n제출점 후보 — 손실은 B1 에 따라 다르지만 **정보량**은 (a1−a0)² 에 비례한다")
    print(f"  {'a1':>6s}{'(a1−a0)':>10s}{'2차항만의 ΔScore':>18s}   비고")
    for a1 in (0.00, 0.05, 0.25, 0.30, 0.40):
        da = a1 - a0
        print(f"  {a1:>6.2f}{da:>+10.2f}{-K*da*da*A2:>18.2f}   "
              f"{'(선형항은 부호 미지)' if a1 != 0 else '(F전문가 끄기)'}")


if __name__ == "__main__":
    main()
