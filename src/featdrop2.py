"""§7.92 후속 — `is_pitcher_home` **단독** 드랍 A/B (calprobe2 방식).

왜 이 변형인가
    §7.92는 miss 플래그 3 + top_bottom을 묶어 뺐다가 MLP 팔(−3.5/−7.8)에서
    기각됐다 — 손해의 기전은 miss 플래그(MLP의 결측 표현) 쪽이었다.
    이번엔 **중복 쌍 제거만 분리한다**: top_bottom ≡ is_pitcher_home 중
    파생 이진 쪽만 빼고, miss 플래그는 건드리지 않는다.

    is_pitcher_home은 EMB_COLS·nan_plan 어디에도 없다 — 수치 채널 1개가
    줄 뿐, 인코딩은 깨지지 않는다. top_bottom(범주 코드)이 같은 정보를 남긴다.

방법 (calprobe2와 동일)
    ctl  = foldcache 캐시 (s24, 현행 구성 그대로)
    drop = 같은 데이터·같은 시드에서 1컬럼만 빼고 LGBM 12 + MLP 8 재학습
    판정 = 배포 블렌드(0.6/0.4)끼리 쌍체 Δ(BSS) ± SE. 통과선 fold2024 t ≥ 3.

기대치
    §7.92의 LGBM 팔이 정확히 0이었으므로 여기도 0 근처가 영가설.
    사전 예측: 블렌드 Δ ∈ [−2, +4]. 중복 채널 하나로 유의한 이득이 나오면
    그게 더 놀라운 일이다.

실행:  .venv/Scripts/python.exe -u src/featdrop2.py 2>&1 | tee experiments/featdrop2.log
"""
import gc
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

import foldcache
import mlp
import modelabel
import soften
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA, MLP_AUX_LAMBDA,
                    MLP_SEEDS, MLP_VARIANTS, N_ROUNDS, SOFT_LABEL_F, TRAIN_CSV)
from data import load_csv
from validation import _sigmoid, shift_to_prior

FOLDS = [2023, 2024]
DROP = ["is_pitcher_home"]
PART = Path(__file__).parent.parent / "experiments" / "cache" / "featdrop2"


def paired(zA, zB, y):
    pA = shift_to_prior(_sigmoid(zA), y.mean())
    pB = shift_to_prior(_sigmoid(zB), y.mean())
    d = (pA - y) ** 2 - (pB - y) ** 2          # 양수 = B(drop)가 좋다
    ref = y.mean() * (1 - y.mean())
    k = 1e5 / ref
    return k * d.mean(), k * d.std(ddof=1) / np.sqrt(len(d))


def main():
    t0 = time.time()
    PART.mkdir(parents=True, exist_ok=True)
    cache = foldcache.load(FOLDS, verbose=False)
    df = load_csv(TRAIN_CSV)

    for f in FOLDS:
        z_lgb_c, z_mlp_c, y, _ = cache[f]
        Xtr, ytr, w, Xva, yva, art, tr = foldcache.fold_data(df, f)
        assert np.array_equal(y, yva)

        aux, _ = modelabel.aux_labels(tr)
        if SOFT_LABEL_F:
            Xtr, ytr, w, aux, _ = soften.soften(tr, Xtr, ytr, w, aux)

        Xtr = Xtr.drop(columns=DROP)
        Xva_d = Xva.drop(columns=DROP)
        del Xva
        print(f"\n=== fold {f}: 피처 {Xtr.shape[1]}개 ({time.time()-t0:.0f}s) ===")

        pl = PART / f"fold{f}_lgb.npz"
        if pl.exists():
            z_lgb_d = np.load(pl)["z"]
            print(f"  lgbm drop 재사용 ({time.time()-t0:.0f}s)")
        else:
            z_lgb_d, n = np.zeros(len(Xva_d)), 0
            for over in ENSEMBLE_VARIANTS.values():
                for s in foldcache.LGB_SEEDS:
                    p = dict(FINAL_LGB_PARAMS, **over, seed=s, bagging_seed=s,
                             feature_fraction_seed=s, data_random_seed=s)
                    bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w),
                                    num_boost_round=N_ROUNDS)
                    z_lgb_d += bst.predict(Xva_d, raw_score=True)
                    n += 1
                    del bst
            z_lgb_d /= n
            np.savez_compressed(pl, z=z_lgb_d)
            print(f"  lgbm drop {n}개 ({time.time()-t0:.0f}s)")

        A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva_d, art)
        del Xtr, Xva_d
        gc.collect()

        zs = []
        for vname, (cfg, nseed) in MLP_VARIANTS.items():
            for s in MLP_SEEDS[:nseed]:
                pm = PART / f"fold{f}_mlp_{vname}_{s}.npz"
                if pm.exists():
                    zs.append(np.load(pm)["z"])
                    print(f"  mlp {vname}_{s} 재사용 ({time.time()-t0:.0f}s)")
                    continue
                _, curve = mlp.train_mlp(A, C, ytr, w, B, Cv, sizes, seed=s,
                                         verbose=False, y_aux=aux,
                                         **dict(cfg, aux=MLP_AUX_LAMBDA))
                zs.append(curve[-1])
                np.savez_compressed(pm, z=zs[-1])
                print(f"  mlp {vname}_{s} 완료  단독 {mlp.score(zs[-1], y):.1f}"
                      f"  ({time.time()-t0:.0f}s)")
        z_mlp_d = np.mean(zs, axis=0)
        del A, C, B, Cv
        gc.collect()

        a = MLP_ALPHA
        bl_c = (1 - a) * z_lgb_c + a * z_mlp_c
        bl_d = (1 - a) * z_lgb_d + a * z_mlp_d
        for tag, zc, zd in (("lgb 단독", z_lgb_c, z_lgb_d),
                            ("mlp 단독", z_mlp_c, z_mlp_d),
                            ("블렌드  ", bl_c, bl_d)):
            dm, se = paired(zc, zd, y)
            print(f"  {tag}  ctl {mlp.score(zc, y):>7.1f}  drop {mlp.score(zd, y):>7.1f}"
                  f"  Δ {dm:+7.2f} ± {se:.2f}  (t {dm/se:+.1f})")


if __name__ == "__main__":
    main()
