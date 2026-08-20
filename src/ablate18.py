"""18차 실험 — **손실함수를 바꾼 멤버**가 새 함수족이 되는가.

s07/s08이 알려준 것:
    s05 → s07  새 함수족(MLP) 도입   홀드아웃 +11.4 → LB **+29.5**
    s07 → s08  같은 함수족 확장       홀드아웃  +4.1 → LB  **+0.24**
→ 생산적인 축은 "모델 다양성"이 아니라 **"아직 없는 함수족"**이다.
  같은 계열 안에서의 확장은 항상 한 자릿수였다 (LightGBM 시드→변형도 +6.16).

아직 안 건드린 축: **손실함수.** 지금 앙상블의 모든 멤버가 logloss로 학습된다.
그런데 이 대회 지표는 **Brier**다.

    현재   objective="binary"     → logloss 최소화
    후보   objective="regression" → MSE 최소화 = **Brier 직접 최소화**

둘 다 proper scoring rule이지만 최적화 대상이 달라 적합되는 함수가 다르다.
logloss는 극단 확률에 큰 벌점을 줘 예측을 중앙으로 민다. MSE는 그러지 않는다.

⚠️ **기준선은 "지금 배포 중인 것"이다** — LightGBM 앙상블 + MLP 블렌드.
   단일 모델 대비로 재면 이득이 과대평가된다 (CatBoost가 정확히 그 경로로 죽었다).

⚠️ 판정 기준은 이미 확립돼 있다:
   · **현재 예측과의 로짓 상관.** 0.98 이상이면 실력이 비슷해도 얻을 게 없다
     (CatBoost 0.985 → 보너스 +3.5뿐. MLP 0.92 → +29.5).
   · 부호 일치 + **크기비 3배 이내** (조건 ③, 현재 4전 4중).

실행:  .venv/Scripts/python.exe -u src/ablate18.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import lightgbm as lgb
import numpy as np

import mlp
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA, N_ROUNDS,
                    TRAIN_CSV)
from data import load_csv

mlp.THREADS = 8

FOLDS = [2023, 2024]
LGB_SEEDS = (42, 7, 2024)
BETAS = (0.05, 0.10, 0.15, 0.20, 0.30)
EPS = 1e-6

EMB4 = {"pitcher_id": 4, "batter_id": 4, "pitcher_team_id": 4, "batter_team_id": 4,
        "base_state": 4, "hand_matchup": 4, "count_state": 4}
CFG_EMB4 = dict(epochs=9, hid=128, lr=2e-3, dropout=0.3, wd=1e-2,
                use_id_emb=True, emb_dim=EMB4)
CFG_NOID = dict(epochs=6, hid=128, lr=2e-3, dropout=0.3, wd=1e-2,
                use_id_emb=False, emb_dim=EMB4)

# MSE 학습 LightGBM. objective만 바꾸고 나머지는 배포 설정 그대로 둔다
# (한 번에 한 가지만 바꾼다). 라운드 수는 곡선을 찍어 fold별 정점을 확인한다.
MSE_ROUNDS = (160, 240, 320, 480, 640)


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def lgb_ens(Xtr, ytr, w, Xva, t0, objective="binary", rounds=N_ROUNDS,
            variants=None, tag=""):
    """로짓 평균. objective='regression'이면 예측이 확률이므로 로짓으로 되돌린다."""
    variants = ENSEMBLE_VARIANTS if variants is None else variants
    z, n = np.zeros(len(Xva)), 0
    for over in variants.values():
        for s in LGB_SEEDS:
            p = dict(FINAL_LGB_PARAMS, **over, objective=objective, seed=s,
                     bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
            bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=rounds)
            raw = bst.predict(Xva, raw_score=True)
            z += raw if objective == "binary" else _logit(raw)
            n += 1
    print(f"  {tag} {n}개  ({time.time()-t0:.0f}s)")
    return z / n


def lgb_mse_curve(Xtr, ytr, w, Xva, yva, t0):
    """MSE 학습 LightGBM을 라운드별로 평가. fold마다 정점이 크게 다르면 채택하지 않는다
    (CatBoost가 그 이유로 죽었다 — fold별 최적 iteration이 4배 차이)."""
    out = {}
    boosters = []
    for s in LGB_SEEDS:
        p = dict(FINAL_LGB_PARAMS, objective="regression", seed=s, bagging_seed=s,
                 feature_fraction_seed=s, data_random_seed=s)
        boosters.append(lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w),
                                  num_boost_round=max(MSE_ROUNDS)))
    for r in MSE_ROUNDS:
        z = np.mean([_logit(b.predict(Xva, num_iteration=r)) for b in boosters], axis=0)
        out[r] = z
        print(f"    lgbMSE r={r:<4d} 단독 {mlp.score(z, yva):>7.1f}   ({time.time()-t0:.0f}s)")
    return out


def mlp_ens(A, C, ytr, w, B, Cv, sizes, cfg, seeds, t0, tag=""):
    zs = []
    for s in seeds:
        _, curve = mlp.train_mlp(A, C, ytr, w, B, Cv, sizes, seed=s, verbose=False, **cfg)
        zs.append(curve[-1])
    print(f"  {tag} {len(zs)}시드  ({time.time()-t0:.0f}s)")
    return np.mean(zs, axis=0)


def verdict(d):
    same = bool(np.all(d > 0) or np.all(d < 0))
    sp = np.abs(d).max() / max(np.abs(d).min(), 1e-9) if same else np.inf
    return same, sp, (same and d.mean() > 0 and sp <= 3)


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    Z, Y = {}, {}

    for f in FOLDS:
        print(f"\n=== valid={f} ===")
        Xtr, ytr, w, Xva, yva, art = mlp.fold_data(df, f)
        A, C, B, Cv, num_cols, sizes = mlp.encode(Xtr, Xva, art)
        Y[f] = yva

        z_lgb = lgb_ens(Xtr, ytr, w, Xva, t0, tag="lgbm(logloss)")
        z_e = mlp_ens(A, C, ytr, w, B, Cv, sizes, CFG_EMB4, (42, 7, 2024), t0, "mlp emb4")
        z_n = mlp_ens(A, C, ytr, w, B, Cv, sizes, CFG_NOID, (42, 7), t0, "mlp noid")
        z_mlp = np.mean([z_e] * 3 + [z_n] * 2, axis=0)      # 배포 비율(5:3)에 가깝게
        z_cur = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
        print(f"  현재 배포 구성 BSS = {mlp.score(z_cur, yva):.1f}"
              f"   (lgbm 단독 {mlp.score(z_lgb, yva):.1f})")

        mse_curve = lgb_mse_curve(Xtr, ytr, w, Xva, yva, t0)
        z_mmlp = mlp_ens(A, C, ytr, w, B, Cv, sizes, dict(CFG_EMB4, loss="mse"),
                         (42, 7, 2024), t0, "mlp emb4-MSE")
        print(f"    mlpMSE      단독 {mlp.score(z_mmlp, yva):>7.1f}")

        Z[f] = {"cur": z_cur, "mlpMSE": z_mmlp,
                **{f"lgbMSE{r}": z for r, z in mse_curve.items()}}
        for k in Z[f]:
            if k != "cur":
                print(f"    corr(현재, {k:<10s}) = "
                      f"{np.corrcoef(z_cur, Z[f][k])[0,1]:.4f}")

    base = np.array([mlp.score(Z[f]["cur"], Y[f]) for f in FOLDS])
    print(f"\n기준(현재 배포 구성): " + "  ".join(f"{f}={b:.1f}" for f, b in zip(FOLDS, base)))

    names = [k for k in Z[FOLDS[0]] if k != "cur"]
    print(f"\n{'후보':<12s} {'beta':>5s} " + " ".join(f"{f:>8d}" for f in FOLDS)
          + f" {'기준대비':>9s} {'크기비':>7s} {'corr':>7s}")
    print("-" * 66)
    for n in names:
        corr = np.mean([np.corrcoef(Z[f]["cur"], Z[f][n])[0, 1] for f in FOLDS])
        for b in BETAS:
            v = np.array([mlp.score((1 - b) * Z[f]["cur"] + b * Z[f][n], Y[f])
                          for f in FOLDS])
            d = v - base
            same, sp, ok = verdict(d)
            print(f"{n:<12s} {b:>5.2f} " + " ".join(f"{x:>8.1f}" for x in v)
                  + f" {d.mean():>+9.1f} "
                  + (f"{sp:>7.1f}" if same else f"{'inf':>7s}")
                  + f" {corr:>7.4f}" + ("  <<<" if ok else ""))

    print("\n  판정: corr 0.98 이상이면 이득 없음 (CatBoost 0.985 → +3.5).")
    print("        MLP는 0.92에서 +29.5였다. 그 사이 어디인지를 본다.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
