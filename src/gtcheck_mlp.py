"""game_type 2×2의 MLP(emb4) 학습기 버전 (§7.71 후속)

§7.71의 표는 LGBM 단독 하네스다. 같은 4팔(gt 유지/드랍 × soft/원라벨)을
배포 MLP 설정(emb4, 보조 헤드 λ=1.0, 9에폭)으로 다시 잰다. 학습기의 귀납
편향이 다르므로(트리 분할 vs 매끄러운 함수 + 임베딩) gt 독성의 크기가
다르게 나올 수 있다 — 특히 MLP는 gt가 표준화된 수치 입력 하나로 들어가
전역 오프셋을 만들기 쉽다.

실행:  .venv/Scripts/python.exe -u src/gtcheck_mlp.py 2>&1 | tee experiments/gtcheck_mlp.log
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import mlp
import modelabel
import soften as soften_mod
from config import (CONTAM_AXES, MLP_AUX_LAMBDA, MLP_VARIANTS, RECENCY_HALF_LIFE,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV,
                    USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import bss, brier, shift_to_prior

FOLDS = [2024, 2023]
SEEDS = (42, 7, 2024)
ARMS = [("드랍+soft (배포)", False, True), ("유지+soft", True, True),
        ("드랍+원라벨", False, False), ("유지+원라벨", True, False)]
CFG = dict(MLP_VARIANTS["emb4"][0], aux=MLP_AUX_LAMBDA)


def sub(yva, p, m):
    ref = yva[m].mean() * (1 - yva[m].mean())
    return 1e5 * (1 - brier(yva[m], p[m]) / ref)


def main():
    df = load_csv(TRAIN_CSV)
    t0 = time.time()
    res = {}
    for fold in FOLDS:
        print(f"\n=== fold {fold} ===")
        tr = df[df.season < fold]
        va = df[df.season == fold]
        w0 = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                            ref_season=int(tr.season.max()))
        art = fit_artifacts(tr, weights=w0, k_pitcher=SMOOTH_K_PITCHER,
                            k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

        def X(d, keep):
            x = build_features(d, art)
            if USE_PLAYER_IDS:
                for c in ("pitcher_id", "batter_id"):
                    x[c] = d[c].to_numpy(dtype=np.float32)
            return x if keep else x.drop(columns=["game_type"])

        yva = va[TARGET_COL].to_numpy(dtype=np.float64)
        va_f = va["game_type"].to_numpy().astype(str) == "F"
        ytr0 = tr[TARGET_COL].to_numpy(dtype=np.float64)
        aux0, _ = modelabel.aux_labels(tr)

        for name, keep, soft in ARMS:
            Xtr, ytr, w, aux = X(tr, keep), ytr0, w0, aux0
            if soft:
                Xtr, ytr, w, aux, _ = soften_mod.soften(tr, Xtr, ytr, w, aux)
            A, Ctr, B, Cva, _, sizes = mlp.encode(Xtr, X(va, keep), art)
            vals = []
            for s in SEEDS:
                m, _ = mlp.train_mlp(A, Ctr, ytr, w, B, Cva, sizes, seed=s,
                                     verbose=False, y_aux=aux, **CFG)
                p = shift_to_prior(1 / (1 + np.exp(-mlp.infer(m, B, Cva))),
                                   yva.mean())
                vals.append((bss(yva, p), sub(yva, p, va_f), sub(yva, p, ~va_f)))
                del m
            r = np.array(vals)
            res[(name, fold)] = r
            print(f"  {name:<14} BSS {r[:,0].mean():>8.1f} ± {r[:,0].std(ddof=1):>5.1f}"
                  f"   F행 {r[:,1].mean():>8.1f}   R행 {r[:,2].mean():>8.1f}"
                  f"   ({time.time()-t0:.0f}s)")
            del A, Ctr, B, Cva, Xtr

    print("\n" + "=" * 76)
    print("game_type **유지 − 드랍** 쌍체 Δ (양수 = 유지가 이득) — MLP emb4")
    print("=" * 76)
    for soft, tag in [(True, "soft"), (False, "원라벨")]:
        keep_n = "유지+soft" if soft else "유지+원라벨"
        drop_n = "드랍+soft (배포)" if soft else "드랍+원라벨"
        for fold in FOLDS:
            d = res[(keep_n, fold)] - res[(drop_n, fold)]
            print(f"  {tag:<5} f{fold}  전체 {d[:,0].mean():>+8.1f}±{d[:,0].std(ddof=1)/np.sqrt(3):.1f}"
                  f"   F행 {d[:,1].mean():>+9.1f}   R행 {d[:,2].mean():>+8.1f}")


if __name__ == "__main__":
    main()
