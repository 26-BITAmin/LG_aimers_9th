"""최종 학습 → submit/model/ 아티팩트 저장.

실행:  .venv/Scripts/python.exe -u src/train.py

구조:
    logit(p) = BASE + mean_s( raw_s(x) ) + DELTA
      raw_s   시드 s로 학습한 LightGBM의 raw score (로짓 공간)
      BASE    init_score를 쓸 때의 추론용 기준 로짓 (안 쓰면 0)
      DELTA   2025 목표 prior에 맞추는 보정. **학습 시점에 스칼라로 고정**
    BASE와 DELTA는 둘 다 상수이므로 하나로 합쳐 저장한다 (`logit_delta`).
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# 콘솔 인코딩이 UTF-8이 아니면(예: Windows CP949) print 한 줄에 프로세스가 죽는다.
# 평가 서버의 로케일을 알 수 없으므로 방어적으로 고정한다.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (AUG_LGB, AUG_RATIO, FSPEC_ALPHA, FSPEC_MAX_ROUNDS, FSPEC_MIN_SEASON,
                    FSPEC_PARAMS, FSPEC_SEEDS,
                    CAT_ALPHA, CAT_ITERS, CAT_PARAMS, CAT_SEEDS, CAT_VARIANTS,
                    CONTAM_AXES, DROP_FEATURES, ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS,
                    MLP_ALPHA, MLP_AUX_LAMBDA, MLP_AUX_MODES, MLP_SEEDS,
                    MLP_VARIANT_WEIGHT, MLP_VARIANTS,
                    MODEL_DIR, N_ROUNDS, PREDICT_CLIP_Q,
                    RECENCY_HALF_LIFE, SEED_ENSEMBLE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, CTX_AXES, CTX_K, CTX_HALF_LIFE, ASOF_AXES, ASOF_STRENGTH, GT_F_BETA, GT_F_DELTA, MIX_AXES, SOFT_LABEL_F, TARGET_COL, TARGET_PRIOR_2025, TRAIN_CSV,
                    TRACKMAN_MODE, USE_INIT_SCORE, USE_MLP, USE_PLAYER_IDS)
from data import load_csv, recency_weight
import asofdelta
import mixent
import ctxfit
import modelabel
import soften
from features import build_features, fit_artifacts, logit_adjust
from trackman_features import load_table
from validation import _logit, _sigmoid

REF_SEASON = 2024          # prior shift delta를 푸는 참조 행 (= test와 가장 가까운 시즌)
TEST_SEASON = 2025         # 평가 데이터의 시즌. Trackman 조회 키에 들어간다.

# ── `--artifacts-only` (§7.66) ────────────────────────────────────────────
# 사후 로짓 보정축(ctx / asof / mix)만 바꿨을 때는 **모델을 다시 만들 필요가 없다.**
# s17→s18→s19에서 해시로 세 번 확인했다 — 모델 파일 30개가 바이트 단위로 동일했고
# `artifacts.joblib`과 `meta.json`만 달랐다. 그런데 매번 26분을 냈다.
#
# 그래도 재학습이 필요했던 이유는 `logit_delta`(prior shift)가 보정축에 의존해서
# **참조행 예측을 다시 해야** 하기 때문이다. 모델은 디스크에 있으니 로드해서 쓰면 된다.
#
# ⚠️ **잘못 쓰면 조용히 틀린다.** `priors`·`clip`·`cat_maps`·`contam` 같은
#    `build_features` 입력이 바뀌면 저장된 모델은 **다른 피처로 학습된 것**이다.
#    그래서 참조 행렬의 바이트 해시를 아티팩트에 남기고, 여기서 대조해 다르면 죽는다.
#    (보정축은 `logit_adjust`만 타므로 피처를 안 바꾼다 → 해시가 같다.)
ARTIFACTS_ONLY = "--artifacts-only" in sys.argv


# 저장된 모델을 **결정하는** 설정 전부. 하나라도 바뀌면 모델이 달라지므로
# `--artifacts-only`가 낡은 모델을 재사용하면 안 된다.
#
# ⚠️ 배열(X_ref/w/y)만으로는 부족하다. `SOFT_LABEL_F`가 그 증거다 —
#    `soften()`은 **해시 계산보다 뒤에서** 타깃을 바꾸므로, 배열만 해시하면
#    soft label을 켜고 꺼도 해시가 같다. s23이 정확히 그 변경이었다.
#    "배열을 넣었으니 됐다"가 아니라 **모델을 정하는 것을 다 넣었나**로 물어야 한다.
_TRAIN_SHAPING = {
    "soft_label_f": SOFT_LABEL_F,
    # §7.162c — LGBM 만 다른 학습셋을 본다. 피처·참조행렬을 안 바꾸므로 여기
    # 없으면 `--artifacts-only` 가 낡은 모델을 통과시킨다 (soft_label_f 전례).
    "aug_lgb": AUG_LGB,
    "aug_ratio": AUG_RATIO if AUG_LGB else None,
    "recency_half_life": RECENCY_HALF_LIFE,
    "use_init_score": USE_INIT_SCORE,
    "n_rounds": N_ROUNDS,
    "seeds": list(SEED_ENSEMBLE),
    "variants": {k: dict(v) for k, v in ENSEMBLE_VARIANTS.items()},
    "lgb": dict(FINAL_LGB_PARAMS),
    "use_mlp": USE_MLP,
    # ⚠️ **로스터(시드 목록·멤버 수)는 넣지 않는다** (s39 §7.165). 멤버 하나를
    #    결정하는 것은 (X, y, w, aux, cfg, seed) 이고 cfg 는 여기 들어간다.
    #    어떤 멤버들을 섞을지는 번들 구성이라 ARTIFACTS_ONLY 의 로스터 재구성이
    #    npz 존재 검사와 함께 처리한다. 재추첨(수확 제출)마다 해시가 바뀌면
    #    관문이 "늘 마이그레이션되는 문"이 되어 보호 기능을 잃는다.
    "mlp_variants": {k: dict(c) for k, (c, _n) in MLP_VARIANTS.items()},
    "mlp_aux_lambda": MLP_AUX_LAMBDA,
    "mlp_aux_modes": list(MLP_AUX_MODES),
    "use_player_ids": USE_PLAYER_IDS,
    "drop_features": list(DROP_FEATURES),
}


def _feature_hash(X_ref, w=None, y=None):
    """저장된 모델을 재사용해도 되는지 판정하는 해시.

    ⚠️ **참조 행렬만 넣으면 구멍이 난다.** 학습 행 가중(`RECENCY_HALF_LIFE`,
       오염 구간 제외 등)이나 타깃을 바꾸는 변경은 **피처를 안 바꾼다** →
       해시가 같아지고 → 가드가 낡은 모델을 통과시킨다.
       모델을 결정하는 것은 (피처, 가중, 타깃) 셋이므로 셋 다 해시에 넣는다.
       (§7.66에서 처음엔 피처만 넣었고, §7.69 후보를 검토하다 발견했다.)
    """
    import hashlib
    h = hashlib.sha256(np.ascontiguousarray(
        X_ref.to_numpy(dtype=np.float64)).tobytes())
    h.update(("|".join(map(str, X_ref.columns))).encode())
    for arr in (w, y):
        if arr is not None:
            h.update(np.ascontiguousarray(
                np.asarray(arr, dtype=np.float64)).tobytes())
    h.update(json.dumps(_TRAIN_SHAPING, sort_keys=True,
                        default=str).encode("utf-8"))
    return h.hexdigest()


def load_members(model_dir, bundle, X_ref, art):
    """저장된 LGBM·MLP로 참조행 로짓을 다시 계산한다. **학습은 없다.**"""
    from mlp_infer import MLPPredictor

    raw = np.zeros(len(X_ref))
    for tag in bundle["members"]:
        txt = (model_dir / f"lgbm_{tag}.txt").read_text(encoding="utf-8")
        raw += lgb.Booster(model_str=txt).predict(X_ref, raw_score=True)
    raw /= len(bundle["members"])

    z_mlp = np.zeros(len(X_ref))
    for tag, wm in zip(bundle["mlp_members"], bundle["mlp_weights"]):
        W = dict(np.load(model_dir / f"mlp_{tag}.npz"))
        z_mlp += wm * MLPPredictor(W, bundle["mlp_enc"]).logit(X_ref)
    return raw, z_mlp


def reference_matrix(df, art, build_X, ref_season=None, test_season=None):
    """delta를 풀 참조 피처 행렬.

    ⚠️ 참조 '행'만 test와 비슷해선 부족하고, **피처 분포**가 추론 시점과 같아야 한다.
       Trackman 조회 키에 season이 들어가면서 둘이 어긋났다:

         delta를 풀 때   2024 행 → 2024용 테이블(≤2023 집계), 적중률 80.9%
         실제 추론       2025 행 → 2025용 테이블(≤2024 집계), 적중률 ~80% (신인은 미매핑)

       그대로 두면 모델이 목표 prior를 실제로 맞추지 못한다 (0.475 지시 → 0.473 출력).

    그래서 2024 행을 쓰되
      (1) season을 TEST_SEASON으로 바꿔 **추론 때와 같은 테이블**을 보게 하고,
      (2) REF_SEASON용 테이블에 없던 투수는 결측으로 되돌려 **커버리지도 맞춘다**
          (2025의 신인 투수가 매핑에 없는 상황과 같다).
    """
    ref_season = REF_SEASON if ref_season is None else ref_season
    test_season = TEST_SEASON if test_season is None else test_season

    ref = df[df.season == ref_season].copy()
    ref["season"] = test_season
    X = build_X(ref)

    tm = art.get("tm")
    if tm is not None:
        in_ref = (tm["keys"] % 10000) == ref_season
        known = set((tm["keys"] // 10000)[in_ref].tolist())
        rookie = ~ref["pitcher_id"].isin(known).to_numpy()
        cols = [c for c in tm["cols"] if c in X.columns]
        X.loc[rookie, cols] = np.nan
        if "tm_matched" in X.columns:
            X.loc[rookie, "tm_matched"] = 0.0
        print(f"  참조 분포 보정: 신인 취급 {rookie.mean():.1%} → "
              f"trackman 적중 {1-rookie.mean():.1%}")
    return X


def train_ensemble(X, y, w, init_score, X_ref, model_dir, t0=None):
    """앙상블(설정 변형 × 시드)을 학습해 저장하고, 참조 행렬의 평균 로짓을 반환한다.

    ⚠️ **학습 루프는 여기 한 곳에만 둔다.** 예전에 train.py와 verify_pipeline.py에
       루프가 중복돼 있었고, ENSEMBLE_VARIANTS를 train.py에만 반영해서
       검증이 배포 설정이 아닌 것을 테스트하는 사고가 있었다.
       (홀드아웃 점수가 이전 값과 소수점까지 같게 나와 발각됐다.)
    """
    t0 = time.time() if t0 is None else t0
    members, raw = [], np.zeros(len(X_ref))
    n = len(ENSEMBLE_VARIANTS) * len(SEED_ENSEMBLE)
    for vname, over in ENSEMBLE_VARIANTS.items():
        for seed in SEED_ENSEMBLE:
            params = dict(FINAL_LGB_PARAMS, **over, seed=seed, bagging_seed=seed,
                          feature_fraction_seed=seed, data_random_seed=seed)
            dtrain = lgb.Dataset(X, y, weight=w, init_score=init_score)
            bst = lgb.train(params, dtrain, num_boost_round=N_ROUNDS)
            tag = f"{vname}_{seed}"
            (model_dir / f"lgbm_{tag}.txt").write_text(bst.model_to_string(),
                                                       encoding="utf-8")
            members.append(tag)
            raw += bst.predict(X_ref, raw_score=True)
            print(f"  [{len(members)}/{n}] {tag} 완료  ({time.time()-t0:.0f}s)")
    return members, raw / len(members)


def train_fspec(df, build_X, X_ref, model_dir, t0=None):
    """F 전용 전문가 (§7.101). 반환: (태그들, X_ref에 대한 평균 로짓, 학습 행 수).

    ⚠️ 학습 데이터는 **2023년 이후 F만**이다. 2019~22 F는 reverse 검출 실패로 라벨이
       오염돼 있고(§7.48·§7.70), 표본 수를 똑같이 맞춰 넣어도 F_self가 537 → 92로
       무너진다 (N 통제 Δ +445.1 ± 53.8). "많을수록 좋다"가 성립하지 않는 구간이다.

    라운드 수는 학습셋 **내부** 시간분할(뒤 25%)로 정한다. 검증 fold를 보지 않는다.
    """
    t0 = time.time() if t0 is None else t0
    m = ((df["game_type"].to_numpy().astype(str) == "F")
         & (df["season"].to_numpy() >= FSPEC_MIN_SEASON))
    sub = df[m]
    Xf = build_X(sub)
    yf = sub[TARGET_COL].to_numpy(dtype=np.float64)
    # 시즌×월을 하나의 시간축으로 — 2023·2024가 섞여 있으므로 월만 쓰면 안 된다
    tt = sub["season"].to_numpy(dtype=np.int64) * 12 + sub["game_month"].to_numpy(dtype=np.int64)
    cut = np.quantile(tt, 0.75)
    itr, iva = tt <= cut, tt > cut
    tags, z = [], np.zeros(len(X_ref))
    for seed in FSPEC_SEEDS:
        prm = dict(FSPEC_PARAMS, seed=seed, bagging_seed=seed,
                   feature_fraction_seed=seed, data_random_seed=seed)
        b0 = lgb.train(prm, lgb.Dataset(Xf[itr], yf[itr]),
                       num_boost_round=FSPEC_MAX_ROUNDS,
                       valid_sets=[lgb.Dataset(Xf[iva], yf[iva])],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        nr = max(50, b0.best_iteration)
        bst = lgb.train(prm, lgb.Dataset(Xf, yf), num_boost_round=nr)
        tag = f"fspec_{seed}"
        (model_dir / f"lgbm_{tag}.txt").write_text(bst.model_to_string(),
                                                   encoding="utf-8")
        tags.append(tag)
        z += bst.predict(X_ref, raw_score=True)
        print(f"  F전문가 {tag}  rounds={nr}  ({time.time()-t0:.0f}s)")
    return tags, z / len(tags), int(m.sum())


def fspec_ref_logit(model_dir, tags, X_ref):
    """저장된 F전문가로 참조행 로짓만 다시 계산 (--artifacts-only 경로)."""
    z = np.zeros(len(X_ref))
    for tag in tags:
        txt = (model_dir / f"lgbm_{tag}.txt").read_text(encoding="utf-8")
        z += lgb.Booster(model_str=txt).predict(X_ref, raw_score=True)
    return z / len(tags)


def train_catboost(X, y, w, init_tr, X_ref, model_dir, t0=None):
    """CatBoost 앙상블 멤버 (§7.139~§7.142). 반환: (태그들, X_ref 평균 로짓).

    LGBM 멤버와 **완전히 같은 입력**을 본다 — soften 후 X/y/w 와 init_score.
    baseline=init_tr 로 시즌 레벨을 넘겨주는 것이 §7.111 을 CatBoost 에 적용한 것이다.

    ⚠️ CPU 고정. GPU 는 4.6배 빠르나 하드웨어 의존이라 코드 검증에서 다른 모델이
       되고, 실측에서 **CPU 가 더 좋았다** (f2023 핸디캡 GPU −9.1 vs CPU +10.3).
    """
    from catboost import CatBoostClassifier, Pool
    t0 = time.time() if t0 is None else t0
    pool = Pool(X, y, weight=w, baseline=init_tr)
    ref = Pool(X_ref)
    tags, z = [], np.zeros(len(X_ref))
    for var, over in CAT_VARIANTS.items():
        for seed in CAT_SEEDS:
            m = CatBoostClassifier(iterations=CAT_ITERS, random_seed=seed,
                                   **CAT_PARAMS, **over)
            m.fit(pool)
            tag = f"{var}_{seed}"
            m.save_model(str(model_dir / f"cat_{tag}.cbm"))
            tags.append(tag)
            z += m.predict(ref, prediction_type="RawFormulaVal")
            print(f"  CatBoost {tag}  ({time.time()-t0:.0f}s)")
            del m
    return tags, z / len(tags)


def cat_ref_logit(model_dir, tags, X_ref):
    """저장된 CatBoost 멤버로 참조행 로짓만 다시 계산 (--artifacts-only 경로)."""
    from catboost import CatBoostClassifier, Pool
    ref = Pool(X_ref)
    z = np.zeros(len(X_ref))
    for tag in tags:
        m = CatBoostClassifier()
        m.load_model(str(model_dir / f"cat_{tag}.cbm"))
        z += m.predict(ref, prediction_type="RawFormulaVal")
        del m
    return z / len(tags)


def blend_cat(raw_ref, z_cat):
    """전 행에 CatBoost 를 섞는다 — fspec **이후**, δ 보정 **이전**.
    이 지점이 §7.142 에서 α 를 고른 지점과 같아야 한다."""
    print(f"  CatBoost 블렌드 α={CAT_ALPHA}: "
          f"(로짓 상관 {np.corrcoef(raw_ref, z_cat)[0,1]:.3f})")
    return (1 - CAT_ALPHA) * raw_ref + CAT_ALPHA * z_cat


def blend_fspec(df, raw_ref, z_fs, n_train):
    """참조행(REF_SEASON)의 **F 행에만** 전문가를 섞는다. 추론과 같은 지점·같은 α."""
    ref_isF = (df.loc[df.season == REF_SEASON, "game_type"]
                 .to_numpy().astype(str) == "F")
    raw_ref = raw_ref.copy()
    raw_ref[ref_isF] = ((1 - FSPEC_ALPHA) * raw_ref[ref_isF]
                        + FSPEC_ALPHA * z_fs[ref_isF])
    print(f"  F전문가 블렌드 α={FSPEC_ALPHA}: 학습 {n_train:,}행 · "
          f"참조행 F {int(ref_isF.sum()):,}/{len(ref_isF):,} ({ref_isF.mean():.1%})")
    return raw_ref


def train_mlp_members(X, y, w, X_ref, art, model_dir, t0=None, *, y_aux):
    """MLP 멤버를 학습해 **numpy 가중치(npz)로** 내보내고 참조 로짓을 반환한다.

    ⚠️ torch는 학습에만 쓴다. 추론은 `mlp_infer.py`의 numpy 순전파다
       (제출 zip에 torch를 넣지 않는다 — 오프라인 설치 10분 / pickle 버전 /
        CUDA 유무 위험이 통째로 사라진다).

    ⚠️ 학습 루프는 `mlp.train_mlp` 한 곳에만 둔다. 여기서는 호출만 한다.

    ⚠️ `y_aux`는 **키워드 전용 필수 인자**다 (기본값 없음). 일부러 그렇게 뒀다 —
       §7.36에서 플래툰을 넣을 때 `verify_pipeline.py`를 빠뜨려 **검증이 그 기능 없는
       모델을 통과시켰다.** 기본값 None을 주면 같은 사고가 조용히 재발한다.
       여기에 인자를 추가할 때는 항상 이 형태로 두어 **호출부가 시끄럽게 죽게** 한다.
    """
    import numpy as _np

    import mlp
    from mlp_infer import MLPPredictor

    t0 = time.time() if t0 is None else t0
    enc = mlp.fit_encoder(X, art)
    A, C = mlp.encode_apply(X, enc)
    A_ref, C_ref = mlp.encode_apply(X_ref, enc)

    # 변형별 비중(config.MLP_VARIANT_WEIGHT)을 멤버 단위 가중으로 펼친다.
    # 변형 안에서는 균등(시드 평균 = 순수 분산 축소), 변형 사이만 비중을 준다.
    vw = {v: MLP_VARIANT_WEIGHT.get(v, 1.0) for v in MLP_VARIANTS}
    tot = sum(vw.values())
    assert tot > 0, "MLP_VARIANT_WEIGHT의 합이 0이다"

    tags, weights, z_ref = [], [], _np.zeros(len(X_ref))
    n = sum(k for _, k in MLP_VARIANTS.values())
    for vname, (cfg, nseed) in MLP_VARIANTS.items():
        wm = vw[vname] / tot / nseed          # 이 변형의 멤버 하나가 갖는 가중
        for seed in MLP_SEEDS[:nseed]:
            # 보조 헤드는 **두 변형 모두**에 단다 (MODEL.md §7.41 — `둘 다`가
            # `emb4만`보다 양 fold에서 낫다: +20.55/+9.91 vs +16.09/+7.27).
            # `export_weights`는 m.aux를 읽지 않으므로 npz에는 주 헤드만 들어간다
            # → mlp_infer.py와 추론 시간이 그대로다.
            m, _ = mlp.train_mlp(A, C, y, w, A_ref, C_ref, enc["sizes"], seed=seed,
                                 verbose=False, y_aux=y_aux,
                                 **dict(cfg, aux=MLP_AUX_LAMBDA))
            W = mlp.export_weights(m)
            tag = f"{vname}_{seed}"
            _np.savez(model_dir / f"mlp_{tag}.npz", **W)

            # 내보낸 직후 **바로 검증한다.** torch와 numpy가 어긋나면 여기서 잡힌다.
            z_np = MLPPredictor(W, enc).logit(X_ref)
            d = float(_np.abs(z_np - mlp.infer(m, A_ref, C_ref)).max())
            if d > 1e-4:
                raise RuntimeError(f"MLP export skew ({tag}): "
                                   f"max|torch-numpy| = {d:.3e}")

            tags.append(tag)
            weights.append(wm)
            z_ref += wm * z_np
            print(f"  [mlp {len(tags)}/{n}] {tag} 완료  가중 {wm:.4f}"
                  f"  (export skew {d:.1e})  ({time.time()-t0:.0f}s)")
    # z_ref는 이미 가중합이다. 여기서 또 나누면 delta가 어긋난다.
    assert abs(sum(weights) - 1.0) < 1e-9, f"가중 합이 1이 아니다: {sum(weights)}"
    return tags, weights, enc, z_ref


def _season_gt_level(ss, gg, y, w, min_n=500):
    """(시즌 × game_type) **가중** 양성률의 로짓. 행별 init_score 벡터.

    ⚠️ `w`를 soften 후 가중치로 넘기는 것이 요점이다 (§7.109 ④). 소프트 라벨이
       오염 F 행의 양성을 되돌린 **뒤**의 양성률이라야 탈오염 레벨이 되고, 그래야
       소프트 라벨과 같은 것을 두 번 빼지 않는다.
    ⚠️ 표본이 얇은 조합은 전체 평균으로 떨어뜨린다 — 조합 하나가 튀면 그 조합의
       모든 행이 통째로 밀린다.
    """
    ug = np.unique(gg)
    k = ss * len(ug) + np.searchsorted(ug, gg)
    out = np.empty(len(k), dtype=np.float64)
    gm = float(np.average(y, weights=w))
    for kk in np.unique(k):
        m = k == kk
        out[m] = float(np.average(y[m], weights=w[m])) if m.sum() >= min_n else gm
    return _logit(out)


def solve_delta(z_ref, target_mean, tol=1e-11, max_iter=300):
    """참조 분포의 로짓 z_ref에 대해 mean(sigmoid(z_ref + d)) == target_mean 인 d.

    ⚠️ 리키지 주의: d는 **학습 데이터로만** 푼다. 추론 시에는 이 스칼라를 그대로
    적용하므로 각 test 행은 다른 행과 무관하게 예측된다.
    """
    lo, hi, mid = -10.0, 10.0, 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        m = _sigmoid(z_ref + mid).mean()
        if abs(m - target_mean) < tol:
            break
        if m < target_mean:
            lo = mid
        else:
            hi = mid
    return float(mid)


def main():
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train.csv ...")
    df = load_csv(TRAIN_CSV)
    print(f"  {df.shape}  ({time.time()-t0:.1f}s)")

    w = recency_weight(df.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=REF_SEASON)
    print("시즌별 가중치:", {int(s): round(float(w[df.season == s][0]), 4)
                        for s in sorted(df.season.unique())})

    tm_table = load_table(TRACKMAN_MODE)
    print(f"Trackman 모드={TRACKMAN_MODE}  테이블="
          f"{None if tm_table is None else tm_table.shape}")
    # δ 표는 **모델과 별도의 half-life**를 쓴다 (§7.61). 모델은 0.5, δ는 1.0 —
    # δ는 셀당 표본이 필요해서 더 완만한 감쇠가 낫다 (스윕이 단봉을 보였다).
    ctx = ctxfit.fit_ctx(df, TARGET_COL, CTX_AXES, CTX_K,
                         weights=recency_weight(df["season"].to_numpy(),
                                                CTX_HALF_LIFE,
                                                ref_season=int(df.season.max())))
    for line in ctxfit.summary(ctx):
        print(f"보정축 {line}")
    # 시즌 진행분 (§7.55). β=0이면 fit 자체를 건너뛴다 — δ 축과 같은 규약.
    # ⚠️ **오염 보정을 먼저 만들고 asof 축에 넘긴다** (§7.60).
    #    순서가 뒤바뀌면 축이 원본 컬럼을 보게 되고, 그게 s15~s17의 결함이었다.
    art = fit_artifacts(df, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, tm_table=tm_table, ctx=ctx,
                        contam_axes=CONTAM_AXES)
    asof = asofdelta.fit_asof_axes(df, ASOF_AXES, ASOF_STRENGTH,
                                   ct=art.get("contam"))
    art["asof"] = asof
    if asof:
        for _a in asof:
            print(f"시즌진행분 {_a['name']}: 오프셋 {len(_a['keys']):,}개  "
                  f"mu={_a['mu']:+.5f} sd={_a['sd']:.5f}  β={_a['beta']}"
                  f"  오염보정={_a.get('cadj')}")
    # 레퍼토리 균형도 (§7.58). 얼려지는 것이 축당 스칼라 6개뿐이다.
    mix = mixent.fit_mixent(df, MIX_AXES)
    for _a in mix:
        print(f"레퍼토리 {_a['name']}: league={np.round(_a['league'], 3)}  "
              f"mu={_a['mu']:+.5f} sd={_a['sd']:.5f}  β={_a['beta']}")
    art["mix"] = mix
    art["use_player_ids"] = USE_PLAYER_IDS
    # 행 단위 F 레벨 δ_F (§7.69). fit이 없는 동결 상수 — β·δ 곱만 아티팩트에 싣는다.
    art["gt"] = {"delta": GT_F_BETA * GT_F_DELTA}
    if art["gt"]["delta"]:
        print(f"F레벨 δ_F: {GT_F_DELTA} x β {GT_F_BETA} = {art['gt']['delta']:+.5f}")

    def build_X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    X = build_X(df)
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    cols = X.columns.tolist()
    print(f"features={len(cols)}  (drop={DROP_FEATURES}, ids={USE_PLAYER_IDS})"
          f"  ({time.time()-t0:.1f}s)")

    # ── init_score: (시즌 x game_type) 레벨을 학습 전에 제거 ──
    # ⚠️ `init_tr` 자체는 **soften 뒤에서** 만든다 (아래 §7.109 블록). 두 이유다:
    #   ① 정렬 — soften이 행을 복제하므로 init_tr도 같이 늘어나야 한다.
    #      (예전 `assert init_tr is None`이 막고 있던 것이 이것이다.)
    #   ② **이중교정** — 소프트 라벨과 init_score는 보완재가 아니라 **대체재**다.
    #      둘 다 오염 F 행의 레벨을 낮춘다. 원자료 레벨로 빼면 두 번 빼서
    #      fold2023이 620.2 → −106.9가 된다 (§7.109 ③, 회복분 +1206.5).
    #      soften **후**의 가중 양성률이 곧 탈오염 레벨이고 그것만이 안전하다.
    # base(추론용)는 스칼라라 여기서 잡아도 된다 — `_finish`의 solve_delta가 흡수하므로
    # 값 자체는 결과를 바꾸지 않는다 (그래서 그룹별로 주면 안 된다 — §7.109 ①).
    init_tr = None
    base = 0.0
    if USE_INIT_SCORE:
        base = float(_logit(np.array([df.loc[df.season == REF_SEASON, TARGET_COL].mean()]))[0])
        print(f"init_score 사용(§7.109): base(추론용, {REF_SEASON} 레벨)={base:+.5f}")

    # ── 앙상블 학습 ──────────────────────────────────────────
    # 참조 행렬은 **추론 시점의 피처 분포**를 재현한 것이다 (reference_matrix 참조).
    X_ref = reference_matrix(df, art, build_X)[cols]
    fhash = _feature_hash(X_ref, w, y)

    if ARTIFACTS_ONLY:
        old = joblib.load(MODEL_DIR / "artifacts.joblib")
        prev = old.get("feature_hash")
        if prev is None:
            # `--assume-features-unchanged` — **일회성 마이그레이션용.**
            # 해시 도입 전에 학습된 아티팩트에 해시를 처음 심을 때만 쓴다.
            # 해시 **불일치**는 이 플래그로도 통과하지 않는다 (아래 분기).
            if "--assume-features-unchanged" not in sys.argv:
                raise SystemExit(
                    "저장된 아티팩트에 feature_hash가 없다 (--artifacts-only 도입 전 학습).\n"
                    "  → 전체 학습을 한 번 돌리거나, 피처가 안 바뀐 것이 확실하면\n"
                    "     --assume-features-unchanged 를 함께 줘서 해시를 심는다.")
            print(f"⚠️ feature_hash 없음 → 처음 심는다: {fhash[:16]}")
            prev = fhash
        if prev != fhash:
            raise SystemExit(
                "참조 행렬이 달라졌다 → 저장된 모델은 **다른 피처로 학습된 것**이다.\n"
                f"  저장 {prev[:16]}  현재 {fhash[:16]}\n"
                "  build_features 입력(priors/clip/cat_maps/contam/tm)이 바뀌면\n"
                "  --artifacts-only 를 쓸 수 없다. 전체 학습을 돌릴 것.")
        # ⚠️ soft label(§7.70)은 **라벨**만 바꾸므로 피처 해시로 잡히지 않는다.
        #    플래그가 다르면 저장된 모델은 다른 타깃으로 학습된 것이다 — 따로 막는다.
        if bool(old.get("soft_label_f", False)) != bool(SOFT_LABEL_F):
            raise SystemExit(
                "SOFT_LABEL_F가 저장된 모델과 다르다 — 라벨 변경은 피처 해시에\n"
                "  안 잡히므로 --artifacts-only 를 쓸 수 없다. 전체 학습을 돌릴 것.")
        print("--artifacts-only: 참조 행렬 해시 일치 → 저장된 모델 재사용")
        members, mlp_tags = old["members"], old["mlp_members"]
        mlp_wts, mlp_enc = old["mlp_weights"], old["mlp_enc"]
        # config 의 MLP 로스터가 번들과 다르면 (LGA_MLP_ONLY 로 멤버를 늘린 직후 —
        # s39 §7.161b) **config + 디스크** 기준으로 재구성한다. CatBoost 의 디스크
        # 복원(아래)과 같은 규약. 가중은 train_mlp_members 와 같은 식으로 편다.
        exp_tags = [f"{v}_{s}" for v, (_c, n) in MLP_VARIANTS.items()
                    for s in MLP_SEEDS[:n]]
        if USE_MLP and list(mlp_tags) != exp_tags:
            missing = [t for t in exp_tags
                       if not (MODEL_DIR / f"mlp_{t}.npz").exists()]
            if missing:
                raise SystemExit(
                    f"config 는 MLP {len(exp_tags)}멤버인데 디스크에 없다: {missing}\n"
                    "  → LGA_MLP_ONLY=1 로 먼저 학습할 것.")
            vw = {v: MLP_VARIANT_WEIGHT.get(v, 1.0) for v in MLP_VARIANTS}
            tot = sum(vw.values())
            nsd = {v: n for v, (_c, n) in MLP_VARIANTS.items()}
            mlp_wts = [vw[t.rsplit("_", 1)[0]] / tot / nsd[t.rsplit("_", 1)[0]]
                       for t in exp_tags]
            print(f"  MLP 로스터 재구성: 번들 {len(mlp_tags)}개 → config {len(exp_tags)}개"
                  f"  (가중 합 {sum(mlp_wts):.6f})")
            mlp_tags = exp_tags
            old["mlp_members"], old["mlp_weights"] = mlp_tags, mlp_wts
        raw_ref, z_mlp_ref = load_members(MODEL_DIR, old, X_ref, art)
        if USE_MLP:
            print(f"  블렌드: (1-{MLP_ALPHA})·lgbm + {MLP_ALPHA}·mlp   "
                  f"(로짓 상관 {np.corrcoef(raw_ref, z_mlp_ref)[0,1]:.3f})")
            raw_ref = (1 - MLP_ALPHA) * raw_ref + MLP_ALPHA * z_mlp_ref
        fspec_tags = list(old.get("fspec_members") or [])
        if FSPEC_ALPHA > 0 and fspec_tags:
            raw_ref = blend_fspec(df, raw_ref,
                                  fspec_ref_logit(MODEL_DIR, fspec_tags, X_ref), -1)
        elif FSPEC_ALPHA > 0:
            raise SystemExit(
                "FSPEC_ALPHA > 0 인데 저장된 fspec 멤버가 없다 — F전문가는 새 모델이라 "
                "--artifacts-only 로는 만들 수 없다. 전체 학습을 한 번 돌릴 것.")
        cat_tags = list(old.get("cat_members") or [])
        if not cat_tags and CAT_ALPHA > 0:
            # s36 전례와 같은 상황 — 기존 번들엔 없고 디스크에만 있다.
            have = {f.stem[4:] for f in MODEL_DIR.glob("cat_*.cbm")}
            cat_tags = [f"{v}_{sd}" for v in CAT_VARIANTS for sd in CAT_SEEDS
                        if f"{v}_{sd}" in have]
            if cat_tags:
                print(f"--artifacts-only: 디스크에서 CatBoost {len(cat_tags)}개 복원")
        if CAT_ALPHA > 0 and cat_tags:
            raw_ref = blend_cat(raw_ref, cat_ref_logit(MODEL_DIR, cat_tags, X_ref))
        elif CAT_ALPHA > 0:
            raise SystemExit(
                "CAT_ALPHA > 0 인데 저장된 CatBoost 멤버가 없다 — 새 모델이라 "
                "--artifacts-only 로는 만들 수 없다. 전체 학습을 한 번 돌릴 것.")
        return _finish(df, art, ctx, asof, mix, cols, base, raw_ref, members,
                       mlp_tags, mlp_wts, mlp_enc, fhash, t0,
                       soft_h=old.get("soft_h", {}), fspec_tags=fspec_tags,
                       cat_tags=cat_tags)

    # ── soft label (§7.70) — LGBM과 MLP가 **같은** (X, y, w, aux)를 봐야
    #    하므로 보조 라벨을 먼저 복원하고 soften을 한 번만 적용한다.
    aux = None
    if USE_MLP:
        # 보조 라벨 — asof 누적의 증분에서 복원한다. **학습 타깃 전용**이다.
        aux, aux_cov = modelabel.aux_labels(df)
        print(f"  보조 라벨 복원 {aux_cov:.1%}  λ={MLP_AUX_LAMBDA}  "
              f"축 {list(MLP_AUX_MODES)} ({aux.shape[1]}개)")
    soft_h = {}
    ss_lv = df["season"].to_numpy(dtype=np.int64)
    gt_lv = df["game_type"].astype("object").to_numpy().astype(str)
    if SOFT_LABEL_F:
        n0 = len(y)
        # 복제 마스크를 **soften과 같은 정의로** 미리 잡는다 (soften.py의 dup).
        # y가 soften에서 재할당되므로 반드시 호출 **전에** 계산해야 한다.
        h_pre = soften.fit_h(df)
        h_row = np.zeros(len(df))
        for _yr, _v in h_pre.items():
            h_row[(ss_lv == _yr) & (gt_lv == "F")] = _v
        dup = (h_row > 0) & (y == 1)
        X, y, w, aux, soft_h = soften.soften(df, X, y, w, aux)
        ss_lv = np.concatenate([ss_lv, ss_lv[dup]])
        gt_lv = np.concatenate([gt_lv, gt_lv[dup]])
        print(f"soft label(§7.70): +{len(y)-n0:,}행 복제  h = "
              + "  ".join(f"{k}:{v:.3f}" for k, v in sorted(soft_h.items())))

    # ── init_score 본체 (§7.109 ④) — **soften 후** 가중 레벨 = 탈오염 레벨 ──
    if USE_INIT_SCORE:
        init_tr = _season_gt_level(ss_lv, gt_lv, y, w)
        assert len(init_tr) == len(X) == len(y) == len(w), "init_score 정렬이 깨졌다"
        print(f"  init_score 레벨: {len(np.unique(np.round(init_tr, 6)))}개 구간, "
              f"평균 {init_tr.mean():+.5f}  범위 [{init_tr.min():+.5f}, {init_tr.max():+.5f}]")

    if os.environ.get("LGA_CAT_ONLY"):
        # CatBoost 멤버만 만들고 끝낸다. LGBM/MLP 는 손대지 않으므로 그 뒤
        # `--artifacts-only` 로 번들을 재생성하면 된다 (s36 이 쓴 경로).
        print("LGA_CAT_ONLY: CatBoost 멤버만 학습한다")
        tags, _ = train_catboost(X, y, w, init_tr, X_ref, MODEL_DIR, t0)
        print(f"완료 — {len(tags)}개 저장. 이제 --artifacts-only 를 돌릴 것.")
        return None

    if os.environ.get("LGA_MLP_ONLY"):
        # §7.161b s39 — MLP 시드 확장. **디스크에 없는 멤버만** 학습한다 (기존 npz 는
        # 불변). LGA_CAT_ONLY 와 같은 계약: 끝나면 --artifacts-only 로 번들 재생성.
        # 이 블록은 train_mlp_members 와 같은 (X, y, w, aux, X_ref) — soften 후 — 를 본다.
        print("LGA_MLP_ONLY: 디스크에 없는 MLP 멤버만 학습한다")
        import numpy as _np

        import mlp as _mlp
        from mlp_infer import MLPPredictor as _MP
        enc = _mlp.fit_encoder(X, art)
        A, C = _mlp.encode_apply(X, enc)
        A_ref, C_ref = _mlp.encode_apply(X_ref, enc)

        def _fit_one(vname, cfg, seed):
            m, _ = _mlp.train_mlp(A, C, y, w, A_ref, C_ref, enc["sizes"], seed=seed,
                                  verbose=False, y_aux=aux,
                                  **dict(cfg, aux=MLP_AUX_LAMBDA))
            W = _mlp.export_weights(m)
            z_np = _MP(W, enc).logit(X_ref)
            # ⚠️ NaN 은 `d > 1e-4` 비교를 **통과**한다 (NaN 비교는 False) — 따로 막는다.
            if not _np.isfinite(z_np).all():
                raise RuntimeError(f"MLP 로짓에 NaN/inf ({vname}_{seed})")
            d = float(_np.abs(z_np - _mlp.infer(m, A_ref, C_ref)).max())
            if d > 1e-4:
                raise RuntimeError(f"MLP export skew ({vname}_{seed}): {d:.3e}")
            return W

        # ── 재현 관문 — 기존 멤버 하나를 이 환경에서 다시 학습해 비트 대조한다.
        # m16 재사용의 해시 동일성은 "파일이 복사돼 왔다"까지만 증명한다. 새 멤버가
        # 같은 분포의 draw 이려면 **지금의 torch/numpy 가 그 함수를 재현해야** 한다
        # (memory: 간이 재구현은 알려진 수치 재현부터). LGA_MLP_REPRO=0 으로 끈다.
        if os.environ.get("LGA_MLP_REPRO", "1") == "1":
            gv, (gc, _n) = next(iter(MLP_VARIANTS.items()))
            # §7.167 — `LGA_MLP_REPRO_SEED` 로 대상 시드를 지정할 수 있다.
            # m16 재사용분(777 등)이 **같은 함수의 draw 인지**를 묻는 데 쓴다:
            # 해시 동일성은 "파일이 복사돼 왔다"까지만 증명하므로, s32 빌드가 앞
            # 5개를 재학습 없이 재사용했다면 나머지 5개의 동질성은 미증명이다.
            gs = int(os.environ.get("LGA_MLP_REPRO_SEED") or MLP_SEEDS[0])
            ref_p = MODEL_DIR / f"mlp_{gv}_{gs}.npz"
            if ref_p.exists():
                print(f"  재현 관문: {gv}_{gs} 재학습 → 배포 npz 와 대조")
                W_new = _fit_one(gv, gc, gs)
                W_old = dict(_np.load(ref_p))
                bad = [k for k in W_old
                       if not _np.array_equal(W_old[k], W_new.get(k))]
                if bad:
                    dmax = max((float(_np.abs(W_old[k] - W_new[k]).max())
                                for k in bad if k in W_new), default=float("nan"))
                    # ★ **판정은 비트가 아니라 예측 차이로 한다** (§7.167 사전 등록).
                    #   부동소수점 누적으로 비트가 갈리면서 로짓 차이가 1e-7 인
                    #   경우가 있다. 기준은 max|Δz| / sd(z) 이고 임계는 1%.
                    z_old = _MP(W_old, enc).logit(X_ref)
                    z_new = _MP(W_new, enc).logit(X_ref)
                    dz = float(_np.abs(z_new - z_old).max())
                    sdz = float(_np.std(z_old))
                    print(f"  ⚠️ 비트 불일치 (키 {len(bad)}개, max|Δw| {dmax:.3e})")
                    print(f"     max|Δz| = {dz:.3e} · sd(z) = {sdz:.4f} · "
                          f"비 = {dz/sdz:.2%}   [임계 1%]")
                    raise SystemExit(
                        f"재현 실패 — {gv}_{gs} 가 배포 npz 와 다르다 "
                        f"(키 {len(bad)}개, max|Δ| {dmax:.3e}).\n"
                        "  torch/numpy 버전이 학습 당시와 다를 가능성. 새 멤버가\n"
                        "  같은 분포의 draw 라는 전제가 깨지므로 중단한다.")
                print(f"  ✅ 재현 관문 통과 — 전 텐서 비트 동일  ({time.time()-t0:.0f}s)")

        todo = [(v, cfg, s) for v, (cfg, n) in MLP_VARIANTS.items()
                for s in MLP_SEEDS[:n]
                if not (MODEL_DIR / f"mlp_{v}_{s}.npz").exists()]
        print(f"  학습 대상 {len(todo)}개: {[f'{v}_{s}' for v, _c, s in todo]}")
        for i, (vname, cfg, seed) in enumerate(todo, 1):
            W = _fit_one(vname, cfg, seed)
            # tmp → rename. 저장 도중 프로세스가 죽으면 손상 파일이 "완료"로
            # 오인된다 (이 블록의 재개 로직이 존재 여부만 본다).
            # ⚠️ 경로가 아니라 **파일 핸들**로 넘긴다 — savez 는 .npz 로 안 끝나는
            #    경로에 확장자를 덧붙여 tmp.replace 가 못 찾게 된다.
            tmp = MODEL_DIR / f"mlp_{vname}_{seed}.npz.tmp"
            with open(tmp, "wb") as _f:
                _np.savez(_f, **W)
            tmp.replace(MODEL_DIR / f"mlp_{vname}_{seed}.npz")
            print(f"  [mlp-only {i}/{len(todo)}] {vname}_{seed} 저장"
                  f"  ({time.time()-t0:.0f}s)")
        print(f"완료 — {len(todo)}개 저장. 이제 --artifacts-only 를 돌릴 것.")
        return None

    # ── SMOTE 증강을 **LGBM 멤버에만** 준다 (§7.162c) ─────────────────
    # 같은 조작이 LGBM 917.5→926.0 (+8.5) 인데 MLP 는 917.3→905.2 (−12.1) 다.
    # 세 학습기 전부에 걸면 배포층 −5.7, LGBM 에만 걸면 **+1.36** [+0.31,+2.34].
    # 진짜 영가설(아무 행이나)은 −0.51 이라 대상 선택이 실제로 일한다.
    # ⚠️ MLP·CatBoost 는 아래에서 **원본 X/y/w** 를 그대로 받는다.
    Xl, yl, wl, il = X, y, w, init_tr
    if AUG_LGB:
        from smotenc import n_needed, smote
        n0 = len(df)
        s0 = df["season"].to_numpy(dtype=np.int64)
        g0 = df["game_type"].astype("object").to_numpy().astype(str)
        y0 = df[TARGET_COL].to_numpy()
        m0 = (s0 <= 2022) & (g0 == "F") & (y0 == 0)
        k = n_needed(int(((s0 <= 2022) & (g0 == "F") & (y0 == 1)).sum()),
                     int(m0.sum()), AUG_RATIO)
        rng = np.random.default_rng(777)
        new, sd_ = smote(X.iloc[:n0][m0].to_numpy(dtype=np.float64), k, rng)
        src = np.flatnonzero(m0)[sd_]
        Xl = pd.concat([X, pd.DataFrame(new.astype(np.float32), columns=X.columns)],
                       ignore_index=True)
        yl = np.concatenate([y, np.zeros(k)])
        wl = np.concatenate([w, np.asarray(w, dtype=np.float64)[src]])
        if USE_INIT_SCORE:
            il = _season_gt_level(np.concatenate([ss_lv, s0[src]]),
                                  np.concatenate([gt_lv, g0[src]]), yl, wl)
        print(f"LGBM 증강(§7.162c): +{k:,}행 합성 (pre-2023 F·y=0 풀 {int(m0.sum()):,}행)"
              f"  → LGBM 만 {len(Xl):,}행, MLP/Cat 은 {len(X):,}행")
    members, raw_ref = train_ensemble(Xl, yl, wl, il, X_ref, MODEL_DIR, t0)
    del Xl, yl, wl, il

    # ── MLP 멤버 (로짓 블렌드) ───────────────────────────────
    mlp_tags, mlp_wts, mlp_enc = [], [], None
    if USE_MLP:
        mlp_tags, mlp_wts, mlp_enc, z_mlp_ref = train_mlp_members(
            X, y, w, X_ref, art, MODEL_DIR, t0, y_aux=aux)
        print(f"  블렌드: (1-{MLP_ALPHA})·lgbm + {MLP_ALPHA}·mlp   "
              f"(로짓 상관 {np.corrcoef(raw_ref, z_mlp_ref)[0,1]:.3f})")
        raw_ref = (1 - MLP_ALPHA) * raw_ref + MLP_ALPHA * z_mlp_ref

    # ── F 전용 전문가 (§7.101) — 블렌드 **이후**, δ 보정 **이전** ──────
    # 이 순서가 fspec_verify.py에서 α를 고른 지점과 같아야 한다.
    # ⚠️ X는 soften으로 행이 복제된 뒤라 df와 정렬이 깨져 있다 → df에서 다시 만든다.
    fspec_tags = []
    if FSPEC_ALPHA > 0:
        fspec_tags, z_fs, n_fs = train_fspec(df, build_X, X_ref, MODEL_DIR, t0)
        raw_ref = blend_fspec(df, raw_ref, z_fs, n_fs)

    # ── CatBoost 앙상블 (§7.140~§7.142) — fspec **이후**, δ 보정 **이전** ──
    cat_tags = []
    if CAT_ALPHA > 0:
        cat_tags, z_cat = train_catboost(X, y, w, init_tr, X_ref, MODEL_DIR, t0)
        raw_ref = blend_cat(raw_ref, z_cat)

    # ── 사후 로짓 보정도 delta보다 **먼저** 얹는다 ──────────────
    # predict.py의 순서(블렌드 → 보정 → prior shift)와 같아야 한다.
    # `logit_adjust`는 추론·검증과 **같은 함수**다 — 축을 늘려도 여기는 안 고친다.
    # ⚠️ 참조행(REF_SEASON)의 δ 적중률은 추론행(2025)보다 높다 — §6 버그 2와 같은
    #    구조다. 다만 δ는 투수 자기 평균으로부터의 편차라 **평균이 0에 붙어 있고**
    #    (실측 hand −0.0001 / count4 +0.0007) 미적중은 0으로 떨어지므로 레벨 영향이
    #    사실상 없다. 조회 키에 season이 안 들어가는 것이 Trackman과의 결정적 차이다.
    return _finish(df, art, ctx, asof, mix, cols, base, raw_ref, members,
                   mlp_tags, mlp_wts, mlp_enc, fhash, t0, soft_h=soft_h,
                   fspec_tags=fspec_tags, cat_tags=cat_tags)


def _finish(df, art, ctx, asof, mix, cols, base, raw_ref, members,
            mlp_tags, mlp_wts, mlp_enc, fhash, t0, *, soft_h=None,
            fspec_tags=None, cat_tags=None):
    """블렌드 이후 — 보정 가산 → prior shift → 저장.

    전체 학습과 `--artifacts-only`가 **이 함수 하나를 공유한다.** 두 경로가
    각자 delta를 풀거나 각자 meta를 쓰면 조용히 어긋난다 (§5·§7.36의 반복 패턴).
    """
    z_adj, hits = logit_adjust(df[df.season == REF_SEASON], art)
    if len(art["ctx"]) or (art.get("gt") or {}).get("delta"):
        print(f"보정축 참조행 적중 " + "  ".join(f"{k} {v:.1%}" for k, v in hits.items())
              + f"   로짓 기여 평균 {z_adj.mean():+.5f}")
        raw_ref = raw_ref + z_adj

    # ── prior shift: BASE + DELTA를 하나의 스칼라로 ──────────
    # ⚠️ 블렌드 **이후**의 로짓으로 delta를 푼다. 순서가 바뀌면 목표 prior를 못 맞춘다.
    p_before = _sigmoid(base + raw_ref)
    delta = solve_delta(base + raw_ref, TARGET_PRIOR_2025)
    logit_delta = base + delta
    print(f"\nprior shift: ref({REF_SEASON}) pred_mean={p_before.mean():.4f} "
          f"→ target={TARGET_PRIOR_2025:.4f}")
    print(f"  base={base:+.5f}  delta={delta:+.5f}  → logit_delta={logit_delta:+.5f}")

    # ── 꼬리 하드 클리핑 경계 동결 (기본 꺼짐) ─────────────────────
    # 추론 시점에 test 분포를 보지 않으려면 경계가 **여기서** 확정돼야 한다.
    # 참조행(REF_SEASON)의 prior shift **이후** 확률 = 추론이 내놓을 것과 같은 축.
    clip_lo, clip_hi = None, None
    if PREDICT_CLIP_Q > 0:
        p_ref = _sigmoid(raw_ref + logit_delta)
        clip_lo = float(np.quantile(p_ref, PREDICT_CLIP_Q))
        clip_hi = float(np.quantile(p_ref, 1 - PREDICT_CLIP_Q))
        print(f"⚠️  꼬리 클리핑 ON q={PREDICT_CLIP_Q:.4f}  "
              f"lo={clip_lo:.4f} hi={clip_hi:.4f}  "
              f"(pclip.py 측정 −1100 BSS. 왜 켰는지 MODEL.md에 적을 것)")

    # ── 저장 ────────────────────────────────────────────────
    # β는 art["ctx"] 안에 축별로 들어 있다 — 번들에 따로 두지 않는다 (두 곳에 두면 어긋난다)
    joblib.dump({"artifacts": art, "feature_cols": cols, "logit_delta": logit_delta,
                 "members": members, "mlp_members": mlp_tags, "mlp_enc": mlp_enc,
                 "mlp_weights": mlp_wts,
                 "mlp_alpha": MLP_ALPHA if USE_MLP else 0.0,
                 # ⚠️ --artifacts-only 의 안전장치 (§7.66). 이게 있어야
                 #    "저장된 모델이 이 피처로 학습된 것인가"를 판정할 수 있다.
                 "feature_hash": fhash,
                 # ⚠️ soft label(§7.70)은 피처 해시에 안 잡힌다 — 따로 기록해 막는다.
                 "soft_label_f": SOFT_LABEL_F, "soft_h": dict(soft_h or {}),
                 # F 전용 전문가 (§7.101). α=0이거나 멤버가 없으면 추론에서 건너뛴다.
                 "fspec_members": list(fspec_tags or []),
                 "fspec_alpha": FSPEC_ALPHA if fspec_tags else 0.0,
                 "fspec_min_season": FSPEC_MIN_SEASON,
                 # CatBoost 앙상블 (§7.142). α=0이거나 멤버가 없으면 추론에서 건너뛴다.
                 "cat_members": list(cat_tags or []),
                 "cat_alpha": CAT_ALPHA if cat_tags else 0.0,
                 # 꼬리 하드 클리핑 경계. None이면 추론에서 건너뛴다.
                 "clip_q": PREDICT_CLIP_Q, "clip_lo": clip_lo, "clip_hi": clip_hi},
                MODEL_DIR / "artifacts.joblib", compress=3)
    (MODEL_DIR / "meta.json").write_text(json.dumps({
        "seeds": list(SEED_ENSEMBLE), "variants": list(ENSEMBLE_VARIANTS),
        "n_members": len(members), "n_rounds": N_ROUNDS,
        "fspec_members": list(fspec_tags or []),
        "fspec_alpha": FSPEC_ALPHA if fspec_tags else 0.0,
        "fspec_min_season": FSPEC_MIN_SEASON,
        # CatBoost 앙상블 (§7.140~§7.145). 태그·이력이 이 축을 봐야 한다 —
        # 없으면 α 고정 후 변형/시드만 바꾼 제출이 파일명으로 구분되지 않는다.
        "cat_members": list(cat_tags or []),
        "cat_alpha": CAT_ALPHA if cat_tags else 0.0,
        "cat_iters": CAT_ITERS, "cat_variants": list(CAT_VARIANTS),
        "use_mlp": USE_MLP, "mlp_alpha": MLP_ALPHA if USE_MLP else 0.0,
        "mlp_aux_lambda": MLP_AUX_LAMBDA if USE_MLP else 0.0,
        # ⚠️ 축 목록도 반드시 남긴다. λ만 남기면 헤드 2개짜리와 4개짜리가
        #    meta.json에서 구분이 안 되고, describe()의 태그도 같아진다.
        #    (이 종류의 누락이 s11 플래툰 · s12 카운트 · s13 λ 세 번 있었다.)
        "mlp_aux_modes": list(MLP_AUX_MODES) if USE_MLP else [],
        # 오염 보정 축 (§7.49). 위 경고와 같은 이유로 반드시 남긴다 —
        # 이게 없으면 s13과 s14의 파일명이 번호만 빼고 똑같아진다.
        "contam_axes": list(CONTAM_AXES or []),
        # soft label (§7.70). 같은 이유 — 이게 없으면 s22와 s23이 구분 안 된다.
        "soft_label_f": SOFT_LABEL_F, "soft_h": dict(soft_h or {}),
        "mlp_seeds": list(MLP_SEEDS) if USE_MLP else [],
        "mlp_members": mlp_tags, "mlp_variants": MLP_VARIANTS,
        "mlp_variant_weight": MLP_VARIANT_WEIGHT, "mlp_weights": mlp_wts,
        "recency_half_life": RECENCY_HALF_LIFE, "use_init_score": USE_INIT_SCORE,
        "use_player_ids": USE_PLAYER_IDS, "drop_features": list(DROP_FEATURES),
        "smooth_k": SMOOTH_K_PITCHER, "trackman_mode": TRACKMAN_MODE,
        "target_prior_2025": TARGET_PRIOR_2025, "ref_season": REF_SEASON,
        "base_logit": base, "delta": delta, "logit_delta": logit_delta,
        "n_features": len(cols), "train_rows": int(len(df)),
        "ctx_axes": {a["name"]: a["beta"] for a in ctx}, "ctx_k": CTX_K,
        "ctx_cells": {a["name"]: int(len(a["keys"])) for a in ctx},
        # δ 표의 recency half-life (§7.61). **태그에 없으므로 여기가 유일한 기록이다.**
        "ctx_half_life": CTX_HALF_LIFE,
        # 시즌 진행분 (§7.55). 위 경고와 같은 이유로 반드시 남긴다 —
        # 이게 없으면 s14와 s15의 파일명이 번호만 빼고 똑같아진다.
        "asof_axes": {a["name"]: a["beta"] for a in asof},
        "asof_strength": ASOF_STRENGTH,
        "asof_cells": {a["name"]: int(len(a["keys"])) for a in asof},
        # 레퍼토리 균형도 (§7.58). 같은 이유 — 태그가 두 제출을 구분하려면 여기 있어야 한다.
        "mix_axes": {a["name"]: a["beta"] for a in mix},
        # 행 단위 F 레벨 δ_F (§7.69). 같은 이유.
        "gt_f_delta": GT_F_DELTA, "gt_f_beta": GT_F_BETA,
        "lgb_params": FINAL_LGB_PARAMS, "lightgbm_version": lgb.__version__,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved to {MODEL_DIR}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
