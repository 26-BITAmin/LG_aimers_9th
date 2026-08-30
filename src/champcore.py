"""§7.212 — **진짜** 챔피언 코어를 fold 로 재학습한다. (§7.210 의 "챔피언 등가"는 거짓이었다)

사용자 지적: *"지금 챔피언 기준으로 한 거 맞는거야?"* — **아니었다.**

§7.210~§7.211 이 쓴 LGBM 층은 `initdep_fold*_ctr0.npz` 로, **s29 계보**다:
    4변형 × **3시드** = 12 · `num_leaves 31` · **증강 없음** · 8/22 캐시
챔피언 hj03 의 코어는 zip 에서 직접 읽은 지상 진실이:
    4변형 × **5시드** = 20 · `num_leaves` **62** (base/extra/lowff) · **~254**(wide)
    · 트리 320 · 피처 78 · **SMOTE 증강**(pre-2023 F·y=0 풀, r=0.40, rng 777) · soften · init_score

즉 임베딩 차원 하나 빼고 전부 다른 모델이었다. 여기서 코어를 챔피언 레시피로 다시 깐다.

레시피는 **재구현하지 않는다** — `t1core.build()` 를 그대로 부른다 (soften → SMOTE →
`season_gt_level` init_score). §7.182(s45) 가 쓴 것과 같은 함수·같은 rng(777)이고,
`num_leaves` 배수만 `t1cap.leaf2` 와 같은 규칙(×2)으로 적용한다.
([[simplified-reimplementations-need-a-known-number-check]])

⚠️ 여전히 재현 못 하는 것 (채연 계보, 학습 코드 없음): general LGBM 12 · compact 3 ·
   fspec 6 · 라우팅. 예측 가중 약 11%. 이건 재료의 한계이지 선택이 아니다.

⚠️ 피처는 챔피언 코어가 기대하는 **78개 목록 그대로** 쓴다 (zip 의 `lgbm_base_42.txt`
   에서 읽는다). 우리 `build_features` 산출과 집합이 같은지 assert 한다.

산출: experiments/cache/cc_lgb_f{fold}_{variant}_{seed}.npz  (20 × 2 fold)
실행: .venv/Scripts/python.exe -u src/champcore.py 2>&1 | tee experiments/champcore.log
"""
import gc
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import lightgbm as lgb
import numpy as np

import foldcache
import t1core
from config import ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, N_ROUNDS, ROOT, TRAIN_CSV
from data import load_csv
from mlp import score

CACHE = ROOT / "experiments" / "cache"
ZIP = ROOT / "hj03_dim8_cold.zip"
VARIANTS = ("base", "wide", "extra", "lowff")
SEEDS = (42, 7, 2024, 1234, 31337)          # 챔피언 meta 와 동일
LEAF_MULT = 2.0                             # §7.182 s45 — config 는 안 고쳐져 있다
FOLDS = (2023, 2024)


def champ_cols():
    """챔피언 코어가 기대하는 피처 이름 78개 — zip 이 진실원이다."""
    with zipfile.ZipFile(ZIP) as z:
        t = z.read("model/lgbm_base_42.txt").decode("utf-8", "replace")
    b = lgb.Booster(model_str=t)
    return b.feature_name()


def leaf2(v):
    nl = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v]).get("num_leaves", 31)
    return int(round(nl * LEAF_MULT))


def main():
    cols = champ_cols()
    print(f"챔피언 코어 피처 {len(cols)}개 · num_leaves "
          + " · ".join(f"{v} {leaf2(v)}" for v in VARIANTS))
    df = load_csv(TRAIN_CSV)
    t0 = time.time()

    for fold in FOLDS:
        need = [(v, s) for v in VARIANTS for s in SEEDS
                if not (CACHE / f"cc_lgb_f{fold}_{v}_{s}.npz").exists()]
        if not need:
            print(f"fold {fold} 캐시 적중 (20/20)")
            continue
        print(f"\n=== fold {fold} — 남은 멤버 {len(need)}/20 ===")
        Xtr, ytr, w, Xva, yva, art, tr = foldcache.fold_data(df, fold)
        miss = set(cols) - set(Xtr.columns)
        assert not miss, f"챔피언 피처 누락: {sorted(miss)[:5]}"
        extra = set(Xtr.columns) - set(cols)
        if extra:
            print(f"  ⚠️ 우리 쪽 여분 컬럼 {len(extra)}개 제외: {sorted(extra)[:5]}")
        Xtr, Xva = Xtr[cols], Xva[cols]

        # ★ 재구현 금지 — §7.182(s45) 가 쓴 그 함수, 그 rng
        X2, y2, w2, init, k = t1core.build(tr, Xtr, w, True, np.random.default_rng(777))
        print(f"  soften+SMOTE: {len(Xtr):,} → {len(X2):,}행 (합성 {k:,})"
              f"  ({time.time()-t0:.0f}s)")
        assert k > 0, "증강이 0행 — 챔피언 레시피가 아니다"
        del Xtr

        for v, s in need:
            p = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v], seed=s, bagging_seed=s,
                     feature_fraction_seed=s, data_random_seed=s)
            p["num_leaves"] = leaf2(v)          # wide 는 자체 값을 갖는다 — 덮어쓴다
            b = lgb.train(p, lgb.Dataset(X2, y2, weight=w2, init_score=init),
                          num_boost_round=N_ROUNDS)
            z = b.predict(Xva, raw_score=True)
            np.savez_compressed(CACHE / f"cc_lgb_f{fold}_{v}_{s}.npz", z=z)
            print(f"    {v}_{s} (leaves {p['num_leaves']}) 단독 {score(z, yva):.1f}"
                  f"  ({time.time()-t0:.0f}s)")
            del b
            gc.collect()
        del X2, y2, w2, Xva
        gc.collect()

    print("\n" + "=" * 76)
    print("코어 20멤버 평균 vs 기존 s29 계보 12멤버")
    print("=" * 76)
    for fold in FOLDS:
        y = np.load(CACHE / f"champ_y_f{fold}.npz")["y"]
        new = np.mean([np.load(CACHE / f"cc_lgb_f{fold}_{v}_{s}.npz")["z"]
                       for v in VARIANTS for s in SEEDS], axis=0)
        old = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
        print(f"  fold {fold}   챔피언레시피 20 {score(new, y):>8.1f}"
              f"   s29 계보 12 {score(old, y):>8.1f}"
              f"   Δ {score(new, y) - score(old, y):>+8.1f}"
              f"   corr {np.corrcoef(new, old)[0,1]:.5f}")


if __name__ == "__main__":
    main()
