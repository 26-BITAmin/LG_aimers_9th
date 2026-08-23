"""시드 노이즈 헤드룸 — **라벨 없이, 시즌과 무관하게** 잰다 (`MODEL.md §7.128`).

지금까지 판 모든 축은 *"이 관계가 2025에도 유지되나"* 에 거는 베팅이었다.
시드 평균은 다르다 — 줄이는 것이 **모델 자신의 몬테카를로 오차**라 시즌 무관이다.

    p = p_∞ + ε ,  E[ε]=0 , Var(ε)=v      =>      E[BS] = BS_∞ + v

`ε` 은 `y` 와 무관하므로 교차항이 0이고, **v 를 라벨 없이 재면 그대로 이득이 된다**:
`이득 = K·v`.  `v` 는 멤버 수에 `1/s` 로 줄어드는 알려진 함수라 외삽이 아니라 계산이다.

⚠️ **변형 간 차이는 안 줄어든다.** `base/extra/wide/lowff` 는 구조가 다른 예측기라
   시드를 늘려도 수렴하지 않는다. 줄어드는 것은 **변형 내부 시드 분산**뿐이다.
   그래서 변형별로 나눠 재고 블렌드 가중으로 전파한다.

§7.9 는 `3→8 멤버: 홀드아웃 +4.1 → LB +0.24 (배율 0.06)` 를 기록하고 8→16 을
"+0.3 수준" 으로 **외삽해서** 닫았다. 시드 분산은 외삽할 필요가 없는 양이다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import predict as P                                             # noqa: E402
from build_submit import make_dummy_test                        # noqa: E402
from features import build_features, logit_adjust               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "submit" / "model"
N = 60_000                     # 분산 추정에는 충분하다


def main():
    import joblib
    import lightgbm as lgb

    b = joblib.load(MODEL / "artifacts.joblib")
    art, cols = b["artifacts"], b["feature_cols"]
    d = make_dummy_test(art=art).iloc[:N].reset_index(drop=True)
    X = build_features(d, art)
    if art.get("use_player_ids"):
        for c in ("pitcher_id", "batter_id"):
            X[c] = d[c].to_numpy(dtype=np.float32)
    X = X[cols]

    # ── LGBM: 변형별 시드 로짓 ─────────────────────────────────────
    groups = {}
    for p in sorted(MODEL.glob("lgbm_*.txt")):
        var = p.stem.rsplit("_", 1)[0]
        if var == "lgbm_fspec":            # F 전용 — 블렌드 경로가 다르다
            continue
        with open(p, encoding="utf-8") as f:
            bst = lgb.Booster(model_str=f.read())
        groups.setdefault(var, []).append(bst.predict(X, raw_score=True))

    print(f"n={N:,}   변형별 시드 로짓 분산 (줄일 수 있는 부분)")
    print(f"  {'변형':<14s}{'시드':>5s}{'시드간 sd':>12s}{'평균의 sd':>12s}")
    var_of_mean = 0.0
    nv = len(groups)
    for v, zs in groups.items():
        Z = np.array(zs)
        # ⚠️ 시드별 **전역 레벨** 차이는 solve_delta 가 이미 흡수한다 —
        #    각 시드를 자기 평균으로 중심화한 뒤 재야 회수 가능한 몫만 남는다.
        Z = Z - Z.mean(axis=1, keepdims=True)
        sd = Z.std(axis=0, ddof=1).mean()                 # 행별 시드간 sd 의 평균
        vm = float((Z.std(axis=0, ddof=1) ** 2 / len(zs)).mean())
        var_of_mean += vm / nv ** 2                       # z_lgb = (1/nv)Σ_v mean_s
        print(f"  {v:<14s}{len(zs):>5d}{sd:>12.5f}{np.sqrt(vm):>12.5f}")
    z_lgb_var = var_of_mean

    # ── MLP: 변형별 시드 (가중은 artifacts 의 mlp_weights) ──────────
    from mlp_infer import MLPPredictor
    enc = b["mlp_enc"]
    mw = dict(zip(b["mlp_members"], b["mlp_weights"]))
    mg = {}
    for tag, w in mw.items():
        with np.load(MODEL / f"mlp_{tag}.npz") as z:
            m = MLPPredictor({k: z[k] for k in z.files}, enc)
        mg.setdefault(tag.rsplit("_", 1)[0], []).append((m.logit(X), w))
    z_mlp_var = 0.0
    for v, lst in mg.items():
        Z = np.array([a for a, _ in lst]); W = sum(w for _, w in lst)
        Z = Z - Z.mean(axis=1, keepdims=True)
        vm = float((Z.std(axis=0, ddof=1) ** 2 / len(lst)).mean())
        z_mlp_var += (W ** 2) * vm
        print(f"  {'mlp_'+v:<14s}{len(lst):>5d}"
              f"{Z.std(axis=0, ddof=1).mean():>12.5f}{np.sqrt(vm):>12.5f}")

    a = float(b.get("mlp_alpha") or 0.0)
    z_var = (1 - a) ** 2 * z_lgb_var + a ** 2 * z_mlp_var

    # ── 로짓 분산 → 확률 분산 → BSS ────────────────────────────────
    pr = P.Predictor(str(MODEL))
    p = pr.predict(d)
    slope = (p * (1 - p)).mean()                # dp/dz, 아핀 적용 후 p 기준
    pvar = (P.PROB_SCALE ** 2) * (slope ** 2) * z_var
    r = P.PROB_TARGET; K = 1e5 / (r * (1 - r))

    print(f"\n  블렌드 α = {a}")
    print(f"  Var(z_lgb 평균) = {z_lgb_var:.3e}   Var(z_mlp 평균) = {z_mlp_var:.3e}")
    print(f"  Var(z_blend)    = {z_var:.3e}   ->  Var(p) = {pvar:.3e}")
    print(f"\n  ★ 시드 노이즈가 먹고 있는 점수 = K·v = {K*pvar:+.3f} BSS")
    print(f"    (멤버를 s배로 늘리면 그 중 (1−1/s) 를 회수한다)")
    for s in (2, 3, 4):
        print(f"      멤버 x{s}  ->  {K*pvar*(1-1/s):+.3f} BSS")


if __name__ == "__main__":
    main()
