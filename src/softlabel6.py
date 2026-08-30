"""§7.205d — `soft2` 를 **챔피언 아키텍처**에서 잰다. 앞의 세 번은 전부 딴 모델이었다.

사용자: *"이 세션 테스트들 전부 hj03 챔피언 구조 기반으로 한 거 맞지?"* -> **아니었다.**

    §7.205   `softlabel2.train_eval`      LGBM 단독 (가중 0.25 채널)
    §7.205b  foldcache 스택               LGBM 12 · MLP 8 **dim 4** · init_score 없음 · Cat 없음
    §7.205c  거기에 init_score 추가        여전히 dim 4 · LGBM 12 · Cat 없음
    ---- 챔피언 `hj03_dim8_cold.zip` (`artifacts.joblib` 실측) ----
             LGBM **20** (4변형 × 5시드) · MLP 8 (emb4×5 **dim 8 + 콜드스타트**, noid×3)
             CatBoost **12** (α 0.10) · init_score ON · **fspec OFF** (`fspec_alpha 0.0`, 멤버 0)
             블렌드 α 0.75 · 확률공간 아핀 (center .4801 · scale 1.05)

    가장 큰 격차가 `pitcher_id` 임베딩 **4 vs 8** 이다 — 그게 LB +7.67 을 낸 축이고
    챔피언에는 콜드스타트까지 얹혀 있다. 라벨 쪽 변경은 학습기 구성에 직접 걸리므로
    ([[augmentation-targets-learners-not-just-learners]]: 같은 조작이 LGBM +8.5 / MLP −12.1)
    dim4 스택에서 잰 값으로 챔피언을 말할 수 없다.

⚠️ 원리적 한계 — 이것이 **얻을 수 있는 최선**이다
    `soft2` 는 **라벨 쪽 학습 변경**이라 hj03 산출물 위에서 직접 못 잰다. 라벨을 바꾸면
    재학습이고, 재학습하면 이미 hj03 이 아니다. 그래서 "챔피언 **아티팩트**" 가 아니라
    "챔피언 **아키텍처**" 로 간다. (콜드스타트·라우팅·catf0 는 추론측 결정적 패치라
    hj03 위에서 직접 쟀다 — §7.206·§7.207.)

배포 순서 복제 (`hj03:predict.py:186-232` 실측)
    fold_data -> aux_labels -> **soften(팔별)** -> init_score(soften 뒤 가중 레벨)
    -> LGBM 20 · MLP 8(dim8, 학습 후 emb0[0] <- mean) -> z = .25·z_lgb + .75·z_mlp
    -> fspec 생략(챔피언에서 OFF) -> **cat 블렌드 α .10** -> 확률공간 아핀 -> BSS

    두 팔은 `h_row` 하나만 다르고 **모든 시드를 공유한다**(쌍체).

실행:  .venv/Scripts/python.exe -u src/softlabel6.py > experiments/softlabel6.log 2>&1
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
from config import (CAT_ALPHA, CAT_ITERS, CAT_PARAMS, CAT_SEEDS, CAT_VARIANTS,
                    ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA,
                    MLP_AUX_LAMBDA, MLP_SEEDS, MLP_VARIANTS, N_ROUNDS,
                    SEED_ENSEMBLE, TARGET_COL, TRAIN_CSV)
from data import load_csv
from softlabel2 import rev_propensity
from softlabel4 import soften_with
from softlabel5 import h_rows
from train import _logit, _season_gt_level
from validation import bss

FOLDS = (2024, 2023)
BETA = 1.00                     # §7.161: 배포 설정에서 soft 의 최적. β 격자는 2단계.
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
PROB_CENTER, PROB_SCALE = 0.4801, 1.05      # hj03:predict.py:228-229
EMB_DIM8 = {"pitcher_id": 8, "batter_id": 4, "pitcher_team_id": 4,
            "batter_team_id": 4, "base_state": 4, "hand_matchup": 4,
            "count_state": 4}


def run_arm(arm, tr, X0, y0, w0, Xva, yva, art, aux0, r_sm, ss0, gt0, t0):
    hr, _ = h_rows(tr, arm, r_sm, BETA)
    m = hr > 0
    Xtr, ytr, w, aux, ndup = soften_with(tr, X0, y0, w0, aux0, hr)
    dup = (hr > 0) & (y0 == 1)
    init_tr = _season_gt_level(np.concatenate([ss0, ss0[dup]]),
                               np.concatenate([gt0, gt0[dup]]), ytr, w)
    assert len(Xtr) == len(ytr) == len(w) == len(init_tr) == len(aux)
    print(f"\n--- {arm} ---  h̄ {hr[m].mean():.4f}  최대 {hr[m].max():.3f}  복제 {ndup:,}행"
          f"  init 평균 {init_tr.mean():+.5f}")

    # ── LGBM 20 = 4변형 × 5시드 (챔피언과 동일) ───────────────────
    z_lgb, n = np.zeros(len(Xva)), 0
    for over in ENSEMBLE_VARIANTS.values():
        for s in SEED_ENSEMBLE:
            p = dict(FINAL_LGB_PARAMS, **over, seed=s, bagging_seed=s,
                     feature_fraction_seed=s, data_random_seed=s)
            bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w, init_score=init_tr),
                            num_boost_round=N_ROUNDS)
            z_lgb += bst.predict(Xva, raw_score=True)
            n += 1
            del bst
    base_va = float(np.average(y0[(ss0 == ss0.max()) & (gt0 == "R")],
                               weights=w0[(ss0 == ss0.max()) & (gt0 == "R")]))
    z_lgb = z_lgb / n + _logit(np.array([base_va]))[0]
    print(f"  lgbm {n}개  단독 {mlp.score(z_lgb, yva):.2f}  ({time.time()-t0:.0f}s)")

    # ── MLP 8 = emb4×5 (dim 8 + 콜드스타트) + noid×3 ──────────────
    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    zs = []
    for vname, (cfg, nseed) in MLP_VARIANTS.items():
        cfg = dict(cfg)
        if vname == "emb4":
            cfg["emb_dim"] = EMB_DIM8            # ★ 챔피언은 pitcher_id 8 이다
        for s in MLP_SEEDS[:nseed]:
            f = CACHE / f"sl6_{arm}_f{FOLD_NOW}_{vname}_{s}.npz"
            if f.exists():
                zs.append(np.load(f)["z"])
                continue
            m_, curve = mlp.train_mlp(A, C, ytr, w, B, Cv, sizes, seed=s,
                                      verbose=False, y_aux=aux,
                                      **dict(cfg, aux=MLP_AUX_LAMBDA))
            z_new = curve[-1]
            if vname == "emb4":
                # ★ 콜드스타트 — 챔피언 파이프라인의 일부다 (혜진 라운드 18).
                #    fold 검증셋에도 미지 투수가 13~20% 있으므로 실제로 걸린다.
                #    학습이 index 0 을 한 번도 안 건드리므로 그 자리는 초기값이다.
                import torch
                # 인덱스를 가정하지 않는다 — 이름으로 찾는다 (§7.36 사고 방지)
                j = list(m_.ecols).index("pitcher_id")
                with torch.no_grad():
                    e = m_.emb[j].weight
                    assert e.shape[1] == 8, f"emb{j} dim {e.shape[1]} (8 이 아니다)"
                    e[0] = e[1:].mean(0)
                z_new = mlp.infer(m_, B, Cv)
                if z_new.ndim == 2:
                    z_new = z_new.mean(1)
                d = float(np.abs(z_new - (curve[-1].mean(1) if curve[-1].ndim == 2
                                          else curve[-1])).max())
                assert d > 0, "콜드스타트가 예측에 반영되지 않았다"
            zs.append(z_new)
            np.savez_compressed(f, z=zs[-1])
            del m_
    z_mlp = np.mean(zs, axis=0)
    print(f"  mlp {len(zs)}개 (emb4 dim8)  단독 {mlp.score(z_mlp, yva):.2f}"
          f"  ({time.time()-t0:.0f}s)")
    del A, C, B, Cv
    gc.collect()

    z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp

    # ── CatBoost 12 (α 0.10) — 전 행 블렌드, baseline=init_tr ─────
    if CAT_ALPHA > 0:
        from catboost import CatBoostClassifier, Pool
        zc, nc = np.zeros(len(Xva)), 0
        pool_tr = Pool(Xtr, ytr, weight=w, baseline=init_tr)
        pool_va = Pool(Xva)
        for vname, over in CAT_VARIANTS.items():
            for s in CAT_SEEDS:
                mdl = CatBoostClassifier(**dict(CAT_PARAMS, **over,
                                                random_seed=s, iterations=CAT_ITERS))
                mdl.fit(pool_tr)
                zc += mdl.predict(pool_va, prediction_type="RawFormulaVal")
                nc += 1
                del mdl
        zc = zc / nc + _logit(np.array([base_va]))[0]
        print(f"  cat {nc}개  단독 {mlp.score(zc, yva):.2f}  ({time.time()-t0:.0f}s)")
        z = (1 - CAT_ALPHA) * z + CAT_ALPHA * zc

    # ── 확률공간 아핀 (챔피언 상수). PROB_SHIFT 는 팔별로 평균 정렬 ──
    p = 1.0 / (1.0 + np.exp(-z))
    p = PROB_CENTER + PROB_SCALE * (p - PROB_CENTER)
    p = np.clip(p + (yva.mean() - p.mean()), 1e-6, 1 - 1e-6)   # 레벨 고정 = 축 분리
    sc = bss(yva, p)
    print(f"  ★ BSS {sc:.2f}")
    del Xtr
    gc.collect()
    return sc, z_lgb, z_mlp


def main():
    global FOLD_NOW
    t0 = time.time()
    print("§7.205d  soft2 — 챔피언 아키텍처")
    print(f"  LGBM {len(ENSEMBLE_VARIANTS)}×{len(SEED_ENSEMBLE)}={len(ENSEMBLE_VARIANTS)*len(SEED_ENSEMBLE)}"
          f" · MLP 8 (emb4 dim8+콜드스타트) · Cat {len(CAT_VARIANTS)}×{len(CAT_SEEDS)} α{CAT_ALPHA}"
          f" · init_score ON · fspec OFF · β {BETA}\n")
    df = load_csv(TRAIN_CSV)
    out = {}
    for fold in FOLDS:
        FOLD_NOW = fold
        print("#" * 74 + f"\n#  fold {fold}\n" + "#" * 74)
        X0, y0, w0, Xva, yva, art, tr = foldcache.fold_data(df, fold)
        aux0, cov = modelabel.aux_labels(tr)
        r_sm = rev_propensity(tr, art)
        ss0 = tr["season"].to_numpy(dtype=np.int64)
        gt0 = tr["game_type"].astype("object").to_numpy().astype(str)
        res = {a: run_arm(a, tr, X0, y0, w0, Xva, yva, art, aux0, r_sm, ss0, gt0, t0)
               for a in ("soft", "soft2")}
        assert not np.array_equal(res["soft"][1], res["soft2"][1]), "두 팔 LGBM 동일"
        out[fold] = res["soft2"][0] - res["soft"][0]
        print(f"\n  fold {fold}  Δ(soft2−soft) = {out[fold]:+.2f}")
        del X0, Xva, tr
        gc.collect()

    print("\n" + "=" * 74)
    print("판정 — 챔피언 아키텍처")
    print("=" * 74)
    for f in FOLDS:
        print(f"  fold {f}   Δ = {out[f]:+.2f}")
    print(f"\n  부호 일치: {'✅' if out[FOLDS[0]] * out[FOLDS[1]] > 0 else '❌ 갈림'}")
    print(f"  참고 — §7.205 LGBM단독 −1.45±1.03 · §7.205b dim4블렌드(init 없음) −0.29")
    print(f"  ⚠️ 시드 1뽑기다. MLP 층 draw 잡음 SD 3~5 (§7.199) 를 붙여 읽을 것.")
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
