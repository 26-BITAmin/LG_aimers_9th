"""Trackman arm 학습 — **배포 등가 32멤버 층** 위에서 (§7.205 후속, 1/2단계)

사용자 지시
    "A/C 게이트와 2단계 재집계를 해봐. LGBM 만 돌리지 말고 전체를 돌려봐.
     델타·감마 변하는 것하고 shift 도 생각을 좀 하고."

이 파일은 **학습·캐시**만 한다. 평가·판정은 `src/tmeval.py`.

무엇이 §7.9 와 다른가 — 세 가지가 동시에 다르다
    ① 층      §7.9 는 **LGBM 단독** 시절 (fold2024 742.3) 이다.
              여기는 배포 등가 32멤버 (LGBM 12 + MLP 8 + CatBoost 12, 앵커 925.75).
    ② 학습기  §7.9 는 "트랙맨을 켠다/끈다" 하나였다. 여기는 **어느 학습기에 주느냐**가 축이다.
              멤버 z 를 학습기별로 따로 캐시하므로 3(featset)^3(가족) 조합이 재학습 없이 열린다.
              [[augmentation-targets-learners-not-just-learners]] — 같은 조작이
              LGBM +8.5 / MLP −12.1 로 갈린 전례가 있다.
    ③ 게이트  형태 A(신뢰도 조건부 결측)를 arm 으로 만든다. §7.205 가 닫은 것은
              형태 B(시즌 집계값 위의 상호작용)뿐이다.

⚠️ **`foldcache` 는 트랙맨을 절대 안 쓴다** — 이걸 모르고 config 만 켜면 조용히 망한다
    `foldcache._key()` 에 `TRACKMAN_MODE` 가 **없고**, `fold_data()` 가
    `fit_artifacts()` 를 부를 때 `tm_table=` 을 **안 넘긴다**.
    → `TRACKMAN_MODE="release"` 로 바꿔도 같은 캐시를 다시 읽고 Δ = +0.0 이 나온다.
      그걸 "무효과" 로 읽으면 실험 전체가 무효다.
      [[assert-the-cache-key-actually-split]]
    그래서 이 파일은 자기 캐시(`tmarm_*`)를 쓰고, **X 가 실제로 갈렸는지 assert** 한다.

featset (base 는 학습하지 않는다 — 기존 배포 캐시를 그대로 쓴다)
    base   트랙맨 없음.  `initdepm_*` / `fold{f}_{KEY}.npz` members / `catcpu_*`
    tm     `load_table("release")` — §7.9 가 배포했던 그 10컬럼 (+ `tm_matched`)
    gA     같은 표에서 **`tm_n` < 500 행을 삭제**. 삭제이므로 그 투수는 미매핑과
           **구별 불가능**해진다 (`tm_matched=0`) — 이것이 형태 A 의 정확한 의미다.

           ★ 임계 500 은 사전 등록이다. fold 를 보고 고르지 않았다.
             근거: 시즌당 투구 수 중앙값이 550 (`tmbucket.log`) — 즉 "한 시즌치도
             안 되는 누적 표본". 이 임계에서 투수의 13~20% 가 게이트된다.
             (200 이면 4~6% 라 조작이 사실상 없고, 1000 이면 31~39% 로 채널을
              닫는 것에 가까워진다.)

경로는 배포와 **글자 그대로 같아야 한다**
    LGBM/CatBoost  `catgpu.prep` 계열 — soften → `level_init` init_score
    MLP            `foldcache.compute` 계열 — aux 라벨 → soften(aux 포함), init_score 없음
    (배포 등가 층에서 LGBM 은 init_score 를 쓰고 MLP 는 안 쓴다. 그대로 따른다.)

실행:  .venv/Scripts/python.exe -u src/tmarm.py 2>&1 | tee experiments/tmarm.log
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

import numpy as np
import pandas as pd

import deploy29 as D
import modelabel
import soften
from config import (CONTAM_AXES, DROP_FEATURES, ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS,
                    MLP_AUX_LAMBDA, MLP_AUX_MODES, MLP_SEEDS, MLP_VARIANTS, N_ROUNDS,
                    RECENCY_HALF_LIFE, ROOT, SEED_ENSEMBLE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, USE_PLAYER_IDS)
from data import recency_weight
from features import build_features, fit_artifacts
from initfix3 import level_init

CACHE = ROOT / "experiments" / "cache"
FOLDS = [2023, 2024]

LGB_SEEDS = SEED_ENSEMBLE[:3]
LGB_VARIANTS = ("base", "wide", "extra", "lowff")
CAT_VARIANTS = ("base", "wide", "sym", "depthwise")
CAT_SEEDS = (42, 7, 2024)
CAT_ITERS = 250
CPU_COMMON = dict(learning_rate=0.03, l2_leaf_reg=30.0, border_count=128,
                  loss_function="Logloss", verbose=0, allow_writing_files=False,
                  task_type="CPU", thread_count=6)
CAT_OVER = {
    "base":      dict(grow_policy="Lossguide", max_leaves=31, min_data_in_leaf=2000),
    "wide":      dict(grow_policy="Lossguide", max_leaves=127, min_data_in_leaf=8000),
    "sym":       dict(grow_policy="SymmetricTree", depth=6),
    "depthwise": dict(grow_policy="Depthwise", depth=8, min_data_in_leaf=2000),
}

TM_CSV = ROOT / "model_s28" / "trackman_pitcher.csv"
RELEASE_COLS = [
    "tm_relh_mean", "tm_relh_std", "tm_rels_mean", "tm_rels_std", "tm_relpoint_disp",
    "tmfb_relh_mean", "tmfb_relh_std", "tmfb_rels_mean", "tmfb_rels_std",
    "tmfb_relpoint_disp",
]
GATE_A_MIN_N = 500          # ★ 사전 등록. 근거는 파일 상단 참조


def tm_table(featset):
    """featset → trackman 표 (또는 None)."""
    if featset == "base":
        return None
    df = pd.read_csv(TM_CSV)
    if featset == "gA":
        n0 = len(df)
        df = df[df.tm_n >= GATE_A_MIN_N]
        print(f"  gateA: tm_n < {GATE_A_MIN_N} 삭제 — {n0:,} → {len(df):,}행 "
              f"({1-len(df)/n0:.1%} 게이트)")
    elif featset != "tm":
        raise ValueError(featset)
    return df[["pitcher_id", "season"] + RELEASE_COLS].copy()


def _art_and_X(df, fold, featset):
    """(art, Xtr_raw, Xva, tr, y_tr, w). soften 이전 상태."""
    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                        center_asof=False, tm_table=tm_table(featset))

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    return art, X(tr), X(va), tr, y_tr, w


def _assert_split(featset, Xva, fold):
    """★ 캐시/피처가 **실제로 갈렸는지** 검정한다. 이게 없으면 Δ=0 을 오독한다."""
    tmc = [c for c in Xva.columns if c.startswith(("tm_", "tmfb_"))]
    if featset == "base":
        assert not tmc, f"base 인데 트랙맨 컬럼이 있다: {tmc}"
        return {}
    assert "tm_matched" in Xva.columns, "tm_matched 플래그가 없다 — art['tm'] 미주입"
    assert len(tmc) >= len(RELEASE_COLS), f"트랙맨 컬럼 {len(tmc)}개뿐"
    hit = float(Xva["tm_matched"].mean())
    fp = {c: float(np.nansum(Xva[c].to_numpy(dtype=np.float64))) for c in RELEASE_COLS}
    print(f"  fold{fold}/{featset}  X {Xva.shape}  tm 컬럼 {len(tmc)}  적중 {hit:.1%}")
    assert hit > 0.4, f"적중률이 너무 낮다 {hit:.1%}"
    return dict(hit=hit, fp=fp)


def train_lgb(df, fold, featset, t0):
    paths = {(v, sd): CACHE / f"tmarm_{featset}_lgb_f{fold}_{v}_{sd}.npz"
             for v in LGB_VARIANTS for sd in LGB_SEEDS}
    if all(p.exists() for p in paths.values()):
        print(f"  fold{fold}/{featset} LGBM 캐시 적중")
        return
    import lightgbm as lgb
    art, Xtr, Xva, tr, y_tr, w = _art_and_X(df, fold, featset)
    _assert_split(featset, Xva, fold)

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
    assert len(init_tr) == len(Xtr)

    for v in LGB_VARIANTS:
        for sd in LGB_SEEDS:
            if paths[(v, sd)].exists():
                continue
            p = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v], seed=sd,
                     bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
            bst = lgb.train(p, lgb.Dataset(Xtr, y2, weight=w2, init_score=init_tr),
                            num_boost_round=N_ROUNDS)
            z = bst.predict(Xva, raw_score=True)
            del bst
            np.savez_compressed(paths[(v, sd)], z=z, cols=np.array(list(Xva.columns)))
            print(f"    lgb f{fold}/{featset} [{v}/{sd}]  ({time.time()-t0:.0f}s)")
    del Xtr, Xva
    gc.collect()


def train_cat(df, fold, featset, t0):
    paths = {(v, sd): CACHE / f"tmarm_{featset}_cat_f{fold}_{v}_{sd}.npz"
             for v in CAT_VARIANTS for sd in CAT_SEEDS}
    if all(p.exists() for p in paths.values()):
        print(f"  fold{fold}/{featset} CatBoost 캐시 적중")
        return
    from catboost import CatBoostClassifier, Pool
    art, Xtr, Xva, tr, y_tr, w = _art_and_X(df, fold, featset)
    _assert_split(featset, Xva, fold)

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

    pool = Pool(Xtr, y2, weight=w2, baseline=init_tr)
    ref = Pool(Xva)
    for v in CAT_VARIANTS:
        for sd in CAT_SEEDS:
            if paths[(v, sd)].exists():
                continue
            m = CatBoostClassifier(iterations=CAT_ITERS, random_seed=sd,
                                   **CPU_COMMON, **CAT_OVER[v])
            m.fit(pool)
            z = m.predict(ref, prediction_type="RawFormulaVal")
            del m
            np.savez_compressed(paths[(v, sd)], z=z, cols=np.array(list(Xva.columns)))
            print(f"    cat f{fold}/{featset} [{v}/{sd}]  ({time.time()-t0:.0f}s)")
    del pool, ref, Xtr, Xva
    gc.collect()


def train_mlp(df, fold, featset, t0):
    """MLP 경로는 `foldcache.compute` 와 같다 — aux 라벨 · soften(aux) · init_score 없음."""
    tags = [(v, s) for v, (_c, n) in MLP_VARIANTS.items() for s in MLP_SEEDS[:n]]
    paths = {k: CACHE / f"tmarm_{featset}_mlp_f{fold}_{k[0]}_{k[1]}.npz" for k in tags}
    if all(p.exists() for p in paths.values()):
        print(f"  fold{fold}/{featset} MLP 캐시 적중")
        return
    import mlp
    art, Xtr, Xva, tr, y_tr, w = _art_and_X(df, fold, featset)
    _assert_split(featset, Xva, fold)

    aux, cov = modelabel.aux_labels(tr)
    Xtr, ytr, w2, aux, soft_h = soften.soften(tr, Xtr, y_tr, w, aux)
    print(f"  soft label h={soft_h}  보조 라벨 {cov:.1%}  λ={MLP_AUX_LAMBDA}")
    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    cols = np.array(list(Xva.columns))
    del Xtr, Xva
    gc.collect()

    for vname, (cfg, nseed) in MLP_VARIANTS.items():
        for sd in MLP_SEEDS[:nseed]:
            if paths[(vname, sd)].exists():
                continue
            _, curve = mlp.train_mlp(A, C, ytr, w2, B, Cv, sizes, seed=sd,
                                     verbose=False, y_aux=aux,
                                     **dict(cfg, aux=MLP_AUX_LAMBDA))
            np.savez_compressed(paths[(vname, sd)], z=curve[-1], cols=cols)
            print(f"    mlp f{fold}/{featset} [{vname}/{sd}]  ({time.time()-t0:.0f}s)")
    del A, C, B, Cv
    gc.collect()


def main():
    t0 = time.time()
    print(f"§7.205 후속 — Trackman arm 학습   지문 {D.KEY}")
    print(f"게이트 A 임계 tm_n ≥ {GATE_A_MIN_N} (사전 등록)   "
          f"릴리스 컬럼 {len(RELEASE_COLS)}개\n")
    df, _s, _y = D.load_df()

    # ★ `gA` 는 **검출력 미달로 돌리지 않는다** (사전 계산, 결과를 보기 전).
    #   게이트가 바꾸는 행 질량과, 그때 가능한 효과의 상한 (§7.9 의 +13.3 이
    #   행 비례로 전이된다는 후한 가정):
    #       tm_n≥ 200   표 7.7%  게이트 → Δ행질량 0.7%  → ≤ 0.1 BSS
    #              500      23.4%              3.7%       ≤ 0.6
    #             1000      43.9%             11.1%       ≤ 1.8
    #             2000      69.1%             28.1%       ≤ 4.6
    #             5000      93.5%             63.4%       ≤ 10.4  ← 적중 23%, 게이트가 아니라 채널 제거
    #   §7.199 재학습 뽑기 잡음 SD 는 MLP 층이 지배해 4.8/7.1 → 2σ ≈ 10~14 BSS.
    #   **어떤 임계에서도 상한이 잡음선 아래다.** 신뢰도 게이트는 자기가 보호하려는
    #   대상(저표본 투수)이 정의상 투구를 적게 던져서 효과가 구조적으로 작다.
    #   [[preregister-power-not-just-thresholds]] — 규율만 지키고 동전을 던지지 않는다.
    #   재실행하려면 아래 튜플에 "gA" 를 넣으면 된다 (코드는 살려둔다).
    for featset in ("tm",):
        for fold in FOLDS:
            print(f"\n{'='*80}\n■ featset={featset}  fold={fold}"
                  f"   ({time.time()-t0:.0f}s)\n{'='*80}")
            train_lgb(df, fold, featset, t0)
            train_cat(df, fold, featset, t0)
            train_mlp(df, fold, featset, t0)

    print(f"\n총 {time.time()-t0:.0f}s. 평가는 src/tmeval.py")


if __name__ == "__main__":
    main()
