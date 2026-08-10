"""19차 실험 — **MLP가 생긴 뒤의 LightGBM 재튜닝.**

LightGBM의 모든 하이퍼파라미터(lr 0.015, leaves 31, min_data 2000, hl 0.5, rounds 320)는
**MLP가 없던 시절에** 단독 성능 기준으로 골랐다. 지금은 블렌드에 매끄러운 성분이 있다.
역할 분담이 달라졌으니 최적 트리 설정도 달라질 수 있다 —
예컨대 MLP가 부드러운 부분을 맡으면 트리는 더 날카롭게 가는 쪽이 나을 수 있다.

측정 대상은 **트리 단독 성능이 아니라 블렌드 성능**이다. 이게 핵심이다:
`ablate18`에서 lgbMSE는 단독으로 fold2023에서 83.3(트리 단독 2.3 대비 훨씬 좋음)이었지만
블렌드에서는 fold2024를 깎아 기각됐다. **단독으로 좋은 것과 블렌드에 좋은 것은 다르다.**

MLP는 fold마다 한 번만 학습해 재사용한다 (트리 설정과 무관하므로 안전하고 빠르다).

채택 기준은 그대로 — 부호 일치 + 크기비 3배 이내 (조건 ③, 현재 4전 4중).

실행:  .venv/Scripts/python.exe -u src/ablate19.py
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
                    RECENCY_HALF_LIFE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS, DROP_FEATURES)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts

mlp.THREADS = 8

FOLDS = [2023, 2024]
LGB_SEEDS = (42, 7, 2024)

EMB4 = {"pitcher_id": 4, "batter_id": 4, "pitcher_team_id": 4, "batter_team_id": 4,
        "base_state": 4, "hand_matchup": 4, "count_state": 4}
CFG_EMB4 = dict(epochs=9, hid=128, lr=2e-3, dropout=0.3, wd=1e-2,
                use_id_emb=True, emb_dim=EMB4)
CFG_NOID = dict(epochs=6, hid=128, lr=2e-3, dropout=0.3, wd=1e-2,
                use_id_emb=False, emb_dim=EMB4)

# 트리 쪽 변형. ENSEMBLE_VARIANTS 위에 다시 덮어쓴다 (4변형 구조는 유지).
# 한 축씩만 흔든다 — 무엇이 효과를 냈는지 알 수 있어야 한다.
TREE_CFGS = {
    "현행 (기준)": {},
    "lr 0.010 x480": dict(_lr=0.010, _rounds=480),
    "lr 0.025 x200": dict(_lr=0.025, _rounds=200),
    "leaves 63":     dict(num_leaves=63),
    "leaves 15":     dict(num_leaves=15),
    "min_data 500":  dict(min_data_in_leaf=500),
    "min_data 6000": dict(min_data_in_leaf=6000),
    "l2 100":        dict(lambda_l2=100.0),
    "rounds 480":    dict(_rounds=480),
}


def fold_data(df, valid_season):
    tr = df[df.season < valid_season]
    va = df[df.season == valid_season]
    ref = int(tr.season.max())
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=ref)
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    return (X(tr), tr[TARGET_COL].to_numpy(dtype=np.float64), w,
            X(va), va[TARGET_COL].to_numpy(dtype=np.float64), art)


def lgb_ens(Xtr, ytr, w, Xva, over_all):
    """ENSEMBLE_VARIANTS 구조는 유지하고 그 위에 over_all을 덮는다."""
    lr = over_all.pop("_lr", None)
    rounds = over_all.pop("_rounds", N_ROUNDS)
    z, n = np.zeros(len(Xva)), 0
    for over in ENSEMBLE_VARIANTS.values():
        for s in LGB_SEEDS:
            p = dict(FINAL_LGB_PARAMS, **over, **over_all, seed=s, bagging_seed=s,
                     feature_fraction_seed=s, data_random_seed=s)
            if lr is not None:
                p["learning_rate"] = lr
            bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=rounds)
            z += bst.predict(Xva, raw_score=True)
            n += 1
    return z / n


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    solo, blend, Y = {}, {}, {}

    for f in FOLDS:
        print(f"\n=== valid={f} ===")
        Xtr, ytr, w, Xva, yva, art = fold_data(df, f)
        A, C, B, Cv, num_cols, sizes = mlp.encode(Xtr, Xva, art)
        Y[f] = yva

        # MLP는 트리 설정과 무관하므로 fold당 한 번만 학습해 재사용한다
        zs = []
        for cfg, seeds in ((CFG_EMB4, (42, 7, 2024)), (CFG_NOID, (42, 7))):
            for s in seeds:
                _, curve = mlp.train_mlp(A, C, ytr, w, B, Cv, sizes, seed=s,
                                         verbose=False, **cfg)
                zs.append(curve[-1])
        z_mlp = np.mean(zs, axis=0)
        print(f"  mlp {len(zs)}멤버  단독 {mlp.score(z_mlp, yva):.1f}"
              f"   ({time.time()-t0:.0f}s)")

        for name, over in TREE_CFGS.items():
            z_lgb = lgb_ens(Xtr, ytr, w, Xva, dict(over))
            solo.setdefault(name, {})[f] = mlp.score(z_lgb, yva)
            blend.setdefault(name, {})[f] = mlp.score(
                (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp, yva)
            print(f"  {name:<15s} 트리단독 {solo[name][f]:>7.1f}   "
                  f"블렌드 {blend[name][f]:>7.1f}   ({time.time()-t0:.0f}s)")

    base_s = np.array([solo["현행 (기준)"][f] for f in FOLDS])
    base_b = np.array([blend["현행 (기준)"][f] for f in FOLDS])
    print(f"\n기준  트리단독 {base_s.mean():.1f}   블렌드 {base_b.mean():.1f}"
          f"   (alpha={MLP_ALPHA})")

    print(f"\n{'설정':<15s} {'단독차':>8s} | " + " ".join(f"{f:>8d}" for f in FOLDS)
          + f" {'블렌드차':>9s} {'크기비':>7s}")
    print("-" * 62)
    rows = []
    for name in TREE_CFGS:
        v = np.array([blend[name][f] for f in FOLDS])
        ds = np.array([solo[name][f] for f in FOLDS]) - base_s
        d = v - base_b
        same = bool(np.all(d > 0) or np.all(d < 0))
        sp = np.abs(d).max() / max(np.abs(d).min(), 1e-9) if same else np.inf
        rows.append((name, ds.mean(), v, d.mean(), same, sp,
                     same and d.mean() > 0 and sp <= 3))
    for name, ds, v, db, same, sp, ok in sorted(rows, key=lambda r: -r[3]):
        print(f"{name:<15s} {ds:>+8.1f} | " + " ".join(f"{x:>8.1f}" for x in v)
              + f" {db:>+9.1f} " + (f"{sp:>7.1f}" if same else f"{'inf':>7s}")
              + ("  <<< 채택 조건 통과" if ok else ""))

    print("\n  ※ '단독차'와 '블렌드차'의 부호가 갈리는 설정을 주목할 것 —")
    print("     트리 단독으로 좋아지는 것이 블렌드에 좋다는 보장이 없다.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
