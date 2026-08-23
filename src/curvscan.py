"""후처리 상수들의 **곡률을 라벨 없이** 전수 계산한다 (`MODEL.md §7.131`).

§7.130 의 규칙: 앙상블을 고정한 채 상수만 바꾸면 draw 노이즈가 0이라 LB 가 결정적이다.
그래서 남은 상수 큐(ⓑ)는 측정 가능하다 — 문제는 **어느 것에 슬롯을 쓰느냐**다.

    BS(x) = BS₀ + (x−x₀)(A₁ − 2B₁) + (x−x₀)²·A₂ ,   d = dp/dx
    이득(x*) = K·A₂·(x* − x₀)²

`A₂ = mean(d²)` 는 **라벨이 필요 없다.** 따라서 각 축에 대해
*"꼭짓점이 Δ만큼 떨어져 있다면 얼마를 딸 수 있나"* 를 **제출 전에** 알 수 있다.
곡률이 작은 축은 꼭짓점이 어디에 있든 딸 것이 없으므로 슬롯을 쓰지 않는다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "submit" / "model"


def build(n=245789):
    import joblib
    import predict as P
    from build_submit import make_dummy_test
    from features import build_features, logit_adjust

    b = joblib.load(MODEL / "artifacts.joblib")
    art, cols = b["artifacts"], b["feature_cols"]
    d = make_dummy_test(art=art).iloc[:n].reset_index(drop=True)
    pr = P.Predictor(str(MODEL))
    X = build_features(d, art)
    if art.get("use_player_ids"):
        for c in ("pitcher_id", "batter_id"):
            X[c] = d[c].to_numpy(dtype=np.float32)
    X = X[cols]

    zl_each = np.array([bst.predict(X, raw_score=True) for bst in pr.boosters])
    zm_each = np.array([m.logit(X) for m in pr.mlps])
    is_f = d["game_type"].to_numpy().astype(str) == "F"
    zf = np.mean([bst.predict(X.iloc[np.flatnonzero(is_f)], raw_score=True)
                  for bst in pr.fspec], axis=0)
    zadj, _ = logit_adjust(d, art)
    names = [p.stem for p in sorted(MODEL.glob("lgbm_*.txt")) if "fspec" not in p.stem]
    mem = b["mlp_members"]
    return dict(pr=pr, b=b, d=d, zl=zl_each, zm=zm_each, zf=zf, zadj=zadj,
                is_f=is_f, lgb_names=names, mem=mem, P=P)


def make_prob(S):
    P, pr, b = S["P"], S["pr"], S["b"]
    mw = np.array(pr.mlp_w, dtype=float)
    lw = np.full(len(S["zl"]), 1.0 / len(S["zl"]))

    def prob(fspec=None, betamult=1.0, noid=None, wide=None):
        w = lw.copy()
        if wide is not None:
            isw = np.array(["wide" in n for n in S["lgb_names"]])
            w = np.where(isw, wide / isw.sum(), (1 - wide) / (~isw).sum())
        z = w @ S["zl"]
        m = mw.copy()
        if noid is not None:
            isn = np.array([x.startswith("noid") for x in S["mem"]])
            m = np.where(isn, noid / isn.sum(), (1 - noid) / (~isn).sum())
        z = (1 - pr.alpha) * z + pr.alpha * (m @ S["zm"])
        fa = pr.fspec_alpha if fspec is None else fspec
        z[S["is_f"]] = (1 - fa) * z[S["is_f"]] + fa * S["zf"]
        p = 1 / (1 + np.exp(-(z + betamult * S["zadj"] + pr.delta)))
        p = P.PROB_CENTER + P.PROB_SCALE * (p - P.PROB_CENTER) + P.PROB_SHIFT
        return np.clip(p, 1e-6, 1 - 1e-6)
    return prob


def main():
    S = build()
    P = S["P"]
    prob = make_prob(S)
    r = P.PROB_TARGET
    K = 1e5 / (r * (1 - r))
    mem = S["mem"]
    noid0 = sum(w for m, w in zip(mem, S["pr"].mlp_w) if m.startswith("noid"))
    isw = np.array(["wide" in n for n in S["lgb_names"]])
    wide0 = isw.sum() / len(isw)

    axes = [
        ("fspec_alpha", S["pr"].fspec_alpha, 1e-4,
         lambda v: prob(fspec=v), (0.0, 1.0)),
        ("δ β 공통배율", 1.0, 1e-3, lambda v: prob(betamult=v), (0.0, 2.0)),
        ("MLP noid 몫", noid0, 1e-4, lambda v: prob(noid=v), (0.0, 1.0)),
        ("LGBM wide 몫", wide0, 1e-4, lambda v: prob(wide=v), (0.0, 1.0)),
    ]
    print(f"K = {K:,.0f}   더미 245,789행\n")
    print(f"  {'축':<16s}{'배포값':>9s}{'K·A2':>10s}"
          f"{'|Δ|=0.05':>10s}{'0.10':>9s}{'0.25':>9s}{'경계까지':>10s}")
    for name, x0, h, f, (lo, hi) in axes:
        p0 = f(x0)
        dd = (f(x0 + h) - f(x0 - h)) / (2 * h)
        A2 = float(np.mean(dd * dd))
        c = K * A2
        far = max(abs(x0 - lo), abs(hi - x0))
        print(f"  {name:<16s}{x0:>9.4f}{c:>10.1f}"
              f"{c*0.05**2:>10.2f}{c*0.10**2:>9.2f}{c*0.25**2:>9.2f}{c*far*far:>10.1f}")
    print("\n  숫자는 '꼭짓점이 그만큼 떨어져 있을 때 딸 수 있는 BSS'다.")
    print("  곡률이 작으면 꼭짓점이 어디에 있든 딸 것이 없다 → 슬롯을 쓰지 않는다.")


if __name__ == "__main__":
    main()
