"""챔피언은 LGBM 이 4변형 x **5시드 = 20멤버**다 (`hjstruct.log` 실측).
우리 fold 하네스는 4 x 3 = 12 다. 빠진 시드 2개(1234, 31337)를 추가 학습한다.
경로는 `hillclimb.train_lgb_members` 와 **완전히 같다** (soften -> level_init -> init_score).

실행: .venv/Scripts/python.exe -u src/lgb5seed.py 2>&1 | tee experiments/lgb5seed.log
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass
import numpy as np
import lightgbm as lgb
import soften
from config import (CONTAM_AXES, DROP_FEATURES, ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS,
                    N_ROUNDS, RECENCY_HALF_LIFE, ROOT, SEED_ENSEMBLE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from initfix3 import level_init

CACHE = ROOT / "experiments" / "cache"
NEW_SEEDS = SEED_ENSEMBLE[3:5]          # 12멤버에 없던 2개
VARIANTS = ("base", "wide", "extra", "lowff")

def main():
    t0 = time.time()
    print(f"추가 시드 {NEW_SEEDS} x 변형 {VARIANTS} = 각 fold 8멤버")
    df = load_csv(TRAIN_CSV); s = df["season"].to_numpy()
    for fold in (2023, 2024):
        need = [(v, sd) for v in VARIANTS for sd in NEW_SEEDS
                if not (CACHE / f"initdepm_fold{fold}_{v}_{sd}.npz").exists()]
        if not need:
            print(f"fold{fold} 캐시 완비"); continue
        tr, va = df[s < fold], df[s == fold]
        y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
        w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                           ref_season=int(tr.season.max()))
        art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                            k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                            center_asof=False)
        def X(d):
            x = build_features(d, art)
            if USE_PLAYER_IDS:
                for c in ("pitcher_id", "batter_id"):
                    x[c] = d[c].to_numpy(dtype=np.float32)
            return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])
        Xtr, Xva = X(tr), X(va)
        ss = tr["season"].to_numpy(dtype=np.int64)
        gg = tr["game_type"].astype("object").to_numpy().astype(str)
        Xtr, y2, w2, _, _ = soften.soften(tr, Xtr, y_tr, w)
        h = soften.fit_h(tr)
        h_row = np.zeros(len(tr))
        for yr, v in h.items():
            h_row[(ss == yr) & (gg == "F")] = v
        dup = (h_row > 0) & (y_tr == 1)
        ss2, gg2 = np.concatenate([ss, ss[dup]]), np.concatenate([gg, gg[dup]])
        init_tr = level_init(ss2, gg2, y2, w2)
        for v, sd in need:
            p = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v], seed=sd,
                     bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
            bst = lgb.train(p, lgb.Dataset(Xtr, y2, weight=w2, init_score=init_tr),
                            num_boost_round=N_ROUNDS)
            z = bst.predict(Xva, raw_score=True); del bst
            np.savez_compressed(CACHE / f"initdepm_fold{fold}_{v}_{sd}.npz", z=z)
            print(f"  lgb fold{fold} [{v}/{sd}]  ({time.time()-t0:.0f}s)")
        del Xtr, Xva
    print(f"총 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
