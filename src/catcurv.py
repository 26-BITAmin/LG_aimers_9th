"""§7.148 — `CAT_ALPHA` 꼭짓점을 **라벨 없는 곡률 + LB 두 점**으로 푼다 (§7.131 방식).

fold 에서 최대점을 고르면 §7.101 실측대로 **+2.0 이 부풀려진다.** 게다가 이 축은
fold 곡선이 0.30 까지 단조 증가인데(§7.141: +1.38/+2.47/+3.27/+3.78/+3.94)
LB 전이 배율이 2.27 이므로, fold 를 그대로 믿으면 α 를 과하게 올리게 된다.

**대신 §7.131 의 도구를 쓴다.**

    BS(α) = BS(α₀) + (α−α₀)·L₁ + (α−α₀)²·A₂ ,   d = dp/dα
    A₂ = Var(d)  ← **라벨이 필요 없다.** 곡률은 제출 없이 계산된다.
    Score(α) = Score(α₀) − K·(α−α₀)·L₁ − K·A₂·(α−α₀)²  ,  K = 1e5/(r(1−r))

이 축에는 **LB 점이 이미 둘** 있다:
    α = 0.00  →  s35 = 1109.2607832584
    α = 0.10  →  s37 = 1110.3461746799   (ΔScore +1.0853914215)
두 점 + 곡률이면 L₁ 이 풀리고 꼭짓점이 나온다 — **추가 제출 없이**:

    C  = K·A₂
    K·L₁ = (Score(0) − Score(α₀) + C·α₀²) / α₀          [α₀ = 0.10]
           ⚠️ 부호를 처음에 틀렸다. 재구성 assert 가 잡았다 — Score(0) 이
              LB0 과 0.59 어긋났다. **2차식은 반드시 관측점으로 되돌려 검증한다.**
    α* = α₀ − (K·L₁) / (2C)
    꼭짓점에서의 추가 이득 = C·(α* − α₀)²

⚠️ 가정 셋 — 결과와 함께 적는다
   ① Score 가 [0, α*] 구간에서 2차식이다 (§7.132 는 `fspec_alpha` 에서 4% 안에서 맞혔다).
   ② `logit_delta` 는 α 마다 다시 풀리지만 여기서는 배포값으로 고정한다.
      `PROB_SHIFT` 아핀이 레벨을 목표에 못박으므로 **형태 변화만** 남는다는 것이
      §7.131 의 전제이고, §7.132 가 실측으로 통과시켰다.
   ③ 두 LB 점 사이에 다른 변경이 없다 — s35→s37 은 CatBoost 합류 하나뿐이다
      (s36 미제출, §7.146 의 기준선 논증).

실행:  .venv/Scripts/python.exe -u src/catcurv.py 2>&1 | tee experiments/catcurv.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "submit" / "model"

LB0 = 1109.2607832584      # s35  (α = 0)
LB1 = 1110.3461746799      # s37  (α = 0.10)
A0 = 0.10                  # 배포값


def main():
    import joblib
    import predict as P
    from build_submit import make_dummy_test
    from catboost import Pool
    from features import build_features, logit_adjust

    b = joblib.load(MODEL / "artifacts.joblib")
    art, cols = b["artifacts"], b["feature_cols"]
    d = make_dummy_test(art=art).reset_index(drop=True)
    pr = P.Predictor(str(MODEL))
    X = build_features(d, art)
    if art.get("use_player_ids"):
        for c in ("pitcher_id", "batter_id"):
            X[c] = d[c].to_numpy(dtype=np.float32)
    X = X[cols]
    print(f"더미 {len(X):,}행 · 피처 {X.shape[1]}개")

    # ── 배포와 **같은 순서**로 z 를 조립한다 (predict.py 참조) ──
    zl = np.mean([bst.predict(X, raw_score=True) for bst in pr.boosters], axis=0)
    zm = np.zeros(len(X))
    for m, wt in zip(pr.mlps, pr.mlp_w):
        zm += wt * m.logit(X)
    z_base = (1 - pr.alpha) * zl + pr.alpha * zm
    is_f = d["game_type"].to_numpy().astype(str) == "F"
    zf = np.mean([bst.predict(X.iloc[np.flatnonzero(is_f)], raw_score=True)
                  for bst in pr.fspec], axis=0)
    z_base = z_base.copy()
    z_base[is_f] = (1 - pr.fspec_alpha) * z_base[is_f] + pr.fspec_alpha * zf
    pool = Pool(X)
    zc = np.mean([m.predict(pool, prediction_type="RawFormulaVal")
                  for m in pr.cats], axis=0)
    zadj, _ = logit_adjust(d, art)
    print(f"  corr(z_cat, z_base) = {np.corrcoef(zc, z_base)[0,1]:.4f}"
          f"   std(z_cat−z_base) = {np.std(zc - z_base):.4f}")

    def prob(a):
        z = (1 - a) * z_base + a * zc + zadj + pr.delta
        p = 1 / (1 + np.exp(-z))
        p = P.PROB_CENTER + P.PROB_SCALE * (p - P.PROB_CENTER) + P.PROB_SHIFT
        return np.clip(p, 1e-6, 1 - 1e-6)

    # 하네스 검증 — 배포 예측과 일치해야 한다
    p_dep = pr.predict(d)
    p_own = prob(pr.cat_alpha)
    print(f"  하네스 검증: max|배포 − 재조립| = {np.abs(p_dep - p_own).max():.2e}"
          f"   (1e-9 이내여야 정상)")

    r = P.PROB_TARGET
    K = 1e5 / (r * (1 - r))
    h = 1e-4
    dd = (prob(A0 + h) - prob(A0 - h)) / (2 * h)
    A2 = float(np.var(dd))
    C = K * A2
    print(f"\n  K = {K:,.0f}   A2 = Var(dp/dα) = {A2:.6e}   **C = K·A2 = {C:.2f}**")

    # ── 두 LB 점으로 선형항을 푼다 ─────────────────────────
    dS = LB0 - LB1                      # Score(0) − Score(α₀)
    # Score(α) = LB1 − KL1·u − C·u²  (u = α − A0).  u = −A0 에서 LB0 이어야 한다:
    #   LB0 − LB1 = +A0·KL1 − C·A0²   →   KL1 = (dS + C·A0²) / A0
    KL1 = (dS + C * A0 ** 2) / A0
    a_star = A0 - KL1 / (2 * C)
    gain = C * (a_star - A0) ** 2
    print(f"\n  LB 두 점:  α=0.00 → {LB0:.10f}   α={A0:.2f} → {LB1:.10f}"
          f"   (Δ {LB1-LB0:+.4f})")
    chk = LB1 - KL1 * (0.0 - A0) - C * (0.0 - A0) ** 2
    assert abs(chk - LB0) < 1e-6, f"2차식 재구성 실패: {chk:.6f} != {LB0:.6f}"
    print(f"  K·L₁ = {KL1:+.4f}   (재구성 검증: Score(0) = {chk:.10f} = LB0 ✓)")
    print(f"  **꼭짓점 α* = {a_star:.4f}**   현재 α={A0} 대비 추가 이득 **{gain:+.3f} BSS**")
    print(f"\n  {'α':>7s}{'예측 Score':>18s}{'현재 대비':>10s}")
    for a in (0.0, 0.05, 0.075, A0, 0.125, 0.15, 0.20, 0.25, round(a_star, 4)):
        sc = LB1 - K * (a - A0) * (KL1 / K) - C * (a - A0) ** 2
        tag = "  ← 배포" if abs(a - A0) < 1e-9 else ("  ← 꼭짓점" if abs(a - a_star) < 1e-9 else "")
        print(f"  {a:>7.4f}{sc:>18.4f}{sc-LB1:>+10.3f}{tag}")

    print(f"\n  ⚠️ 곡률이 ±20% 어긋나면 α* 는 "
          f"[{A0 - KL1/(2*C*1.2):.3f}, {A0 - KL1/(2*C*0.8):.3f}] 로 움직인다.")
    print(f"  fold 곡선은 0.30 까지 단조 증가였다 (§7.141) — **fold 를 믿으면 과도하게 올린다.**")
    print(f"  전이 배율 2.27 이 그 이유이고, 이 방법은 LB 두 점을 써서 그것을 우회한다.")


if __name__ == "__main__":
    main()
