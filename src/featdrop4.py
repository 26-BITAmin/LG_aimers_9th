"""§7.93 후속 — **최저 중요도 피처의 순수 복제** 추가 A/B (사용자 제안).

§7.93의 발견: 완전 중복 쌍(top_bottom ≡ is_pitcher_home)에서 어느 쪽을 빼도
아프다 → "중복 채널 자체가 하중을 받친다". 이 해석의 대조 실험:

    가장 중요도 낮은 피처를 **그대로 복사**해서 한 컬럼 추가한다.
    복사본은 정보량 0 — 이득이 나면 순수 채널 효과(feature_fraction 노출·
    MLP 입력 용량)가 실재한다는 뜻이고, 0이면 §7.93의 효과는 "**쓸모 있는**
    정보의 노출 확대"였다는 쪽이 강해진다. 음수면 죽은 채널 추가의 희석 비용.

중요도: fold 2023 학습셋(soft label 포함, 배포 구성)에서 4변형 × seed 42
부스터의 total_gain 합. 최하위를 뽑고 하위 10개를 로그에 남긴다.
⚠️ 선정은 fold 2023에서 한 번만 하고 두 fold에 같은 컬럼을 쓴다 (선택이
fold마다 다르면 실험이 두 개가 된다).

방법 (calprobe2/featdrop과 동일)
    ctl = foldcache 캐시 (s24) / add = 같은 시드에서 복사 컬럼 1개 추가 후
    LGBM 12 + MLP 8 재학습. 판정 = 블렌드 쌍체 Δ ± SE, fold2024 t ≥ 3.

사전 예측: Δ ∈ [−4, +2] (0 근처 영가설, 희석이면 약한 음수).
+3 이상으로 유의하면 "순수 채널 효과 실재" — 핵심 피처 복제가 다음 실험이 된다.

MLP 주의: 복사본 이름은 NAN_COLS에 없으므로 원본에 NaN이 있으면 인코딩이
어긋난다(복사본만 0 채움). 그래서 NaN 있는 컬럼이 최하위면 NaN 없는
최하위로 내려간다 (로그에 명시).

실행:  .venv/Scripts/python.exe -u src/featdrop4.py 2>&1 | tee experiments/featdrop4.log
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
PART = Path(__file__).parent.parent / "experiments" / "cache" / "featdrop4"


def paired(zA, zB, y):
    pA = shift_to_prior(_sigmoid(zA), y.mean())
    pB = shift_to_prior(_sigmoid(zB), y.mean())
    d = (pA - y) ** 2 - (pB - y) ** 2          # 양수 = B(add)가 좋다
    ref = y.mean() * (1 - y.mean())
    k = 1e5 / ref
    return k * d.mean(), k * d.std(ddof=1) / np.sqrt(len(d))


def lowest_gain(Xtr, ytr, w, t0):
    """4변형 × seed 42의 total_gain 합에서 최하위 (NaN 없는 컬럼 중)."""
    gain = np.zeros(Xtr.shape[1])
    for over in ENSEMBLE_VARIANTS.values():
        p = dict(FINAL_LGB_PARAMS, **over, seed=42, bagging_seed=42,
                 feature_fraction_seed=42, data_random_seed=42)
        bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w),
                        num_boost_round=N_ROUNDS)
        gain += np.asarray(bst.feature_importance("gain"), dtype=np.float64)
        del bst
    order = np.argsort(gain)
    cols = list(Xtr.columns)
    print(f"  중요도(total_gain) 하위 10  ({time.time()-t0:.0f}s)")
    for j in order[:10]:
        nn = int(Xtr[cols[j]].isna().sum())
        print(f"    {cols[j]:<36} gain {gain[j]:>12.0f}   NaN {nn:,}")
    for j in order:
        if not Xtr[cols[j]].isna().any():
            if j != order[0]:
                print(f"  ⚠️ 최하위 {cols[order[0]]}에 NaN이 있어 "
                      f"{cols[j]}로 내려감 (MLL 인코딩 동일성)")
            return cols[j]
    raise SystemExit("NaN 없는 컬럼이 없다?")


def main():
    t0 = time.time()
    PART.mkdir(parents=True, exist_ok=True)
    cache = foldcache.load(FOLDS, verbose=False)
    df = load_csv(TRAIN_CSV)
    dup = None

    for f in FOLDS:
        z_lgb_c, z_mlp_c, y, _ = cache[f]
        Xtr, ytr, w, Xva, yva, art, tr = foldcache.fold_data(df, f)
        assert np.array_equal(y, yva)

        aux, _ = modelabel.aux_labels(tr)
        if SOFT_LABEL_F:
            Xtr, ytr, w, aux, _ = soften.soften(tr, Xtr, ytr, w, aux)

        if dup is None:
            dup = lowest_gain(Xtr, ytr, w, t0)
            print(f"  → 복제 대상: {dup}")

        Xtr[dup + "_dup"] = Xtr[dup]
        Xva_d = Xva.copy()
        Xva_d[dup + "_dup"] = Xva_d[dup]
        del Xva
        print(f"\n=== fold {f}: 피처 {Xtr.shape[1]}개 (+{dup}_dup) "
              f"({time.time()-t0:.0f}s) ===")

        pl = PART / f"fold{f}_lgb.npz"
        if pl.exists():
            z_lgb_d = np.load(pl)["z"]
            print(f"  lgbm add 재사용 ({time.time()-t0:.0f}s)")
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
            print(f"  lgbm add {n}개 ({time.time()-t0:.0f}s)")

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
            print(f"  {tag}  ctl {mlp.score(zc, y):>7.1f}  add {mlp.score(zd, y):>7.1f}"
                  f"  Δ {dm:+7.2f} ± {se:.2f}  (t {dm/se:+.1f})")


if __name__ == "__main__":
    main()
