"""§7.91 A버킷 드랍 A/B — 죽은 축 4개 제거의 블렌드 쌍체 판정 (calprobe2 방식).

무엇을 빼나 (§7.91 잔차 스크린의 A버킷에서 사용자가 고른 것)
    miss_batter_asof / miss_pitcher_asof / miss_prev_game
        결측 플래그 3개. yz̄이 음수(순수 노이즈), zR² ≤ 0.05%.
        §7.11 그룹 LOO에서도 제거가 +3.8이었던 그룹.
    top_bottom
        is_pitcher_home과 **완전 중복** (같은 정보의 범주 코드판).
        중복 쌍의 한쪽만 뺀다 — 정보는 남고 채널만 준다.

    LGBM은 결측을 네이티브로 처리하므로 miss 플래그가 없어도 원본 rate의
    NaN 분기로 같은 정보를 볼 수 있다. MLP는 nan_plan이 miss 플래그와 무관
    (NAN_COLS = SMOOTH_SPEC + PREV_COLS 원본)이고 top_bottom은 EMB_COLS에
    없어 수치 채널 하나가 줄 뿐이다 — 드랍이 인코딩을 깨지 않는다.

방법 (calprobe2와 동일)
    ctl  = foldcache 캐시 (s24, 현행 구성 그대로)
    drop = 같은 데이터·같은 시드에서 4컬럼만 빼고 LGBM 12 + MLP 8 재학습
    판정 = 배포 블렌드(0.6/0.4)끼리 쌍체 Δ(BSS) ± SE. 통과선 fold2024 t ≥ 3.

기대치 (§7.78 규약: 홀드아웃 이득 ÷ 20이 LB 기대)
    죽은 축 제거의 이득 원천은 분산 축소뿐이다. 4/78 컬럼이면 작을 것.
    사전 예측: 블렌드 Δ ∈ [−3, +8]. 음수로 유의하면(t ≤ −3) 되돌아보기 —
    miss 플래그가 MLP의 결측 표현이었다는 뜻이 된다.

실행:  .venv/Scripts/python.exe -u src/featdrop.py 2>&1 | tee experiments/featdrop.log
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
DROP = ["miss_batter_asof", "miss_pitcher_asof", "miss_prev_game", "top_bottom"]
PART = Path(__file__).parent.parent / "experiments" / "cache" / "featdrop"


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
