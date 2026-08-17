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

from config import (CONTAM_AXES, DROP_FEATURES, ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS,
                    MLP_ALPHA, MLP_AUX_LAMBDA, MLP_AUX_MODES, MLP_SEEDS,
                    MLP_VARIANT_WEIGHT, MLP_VARIANTS,
                    MODEL_DIR, N_ROUNDS,
                    RECENCY_HALF_LIFE, SEED_ENSEMBLE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, CTX_AXES, CTX_K, ASOF_AXES, ASOF_STRENGTH, MIX_AXES, TARGET_COL, TARGET_PRIOR_2025, TRAIN_CSV,
                    TRACKMAN_MODE, USE_INIT_SCORE, USE_MLP, USE_PLAYER_IDS)
from data import load_csv, recency_weight
import asofdelta
import mixent
import ctxfit
import modelabel
from features import build_features, fit_artifacts, logit_adjust
from trackman_features import load_table
from validation import _logit, _sigmoid

REF_SEASON = 2024          # prior shift delta를 푸는 참조 행 (= test와 가장 가까운 시즌)
TEST_SEASON = 2025         # 평가 데이터의 시즌. Trackman 조회 키에 들어간다.


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
    ctx = ctxfit.fit_ctx(df, TARGET_COL, CTX_AXES, CTX_K)
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
    if USE_INIT_SCORE:
        rates = df.groupby(["season", "game_type"], observed=True)[TARGET_COL].mean()
        key = list(zip(df["season"].to_numpy(), df["game_type"].astype("object").to_numpy()))
        init_tr = _logit(np.array([rates[k] for k in key]))
        base = float(_logit(np.array([df.loc[df.season == REF_SEASON, TARGET_COL].mean()]))[0])
        print(f"init_score 사용: base(추론용, {REF_SEASON} 레벨)={base:+.5f}")
    else:
        init_tr, base = None, 0.0

    # ── 앙상블 학습 ──────────────────────────────────────────
    # 참조 행렬은 **추론 시점의 피처 분포**를 재현한 것이다 (reference_matrix 참조).
    X_ref = reference_matrix(df, art, build_X)[cols]
    members, raw_ref = train_ensemble(X, y, w, init_tr, X_ref, MODEL_DIR, t0)

    # ── MLP 멤버 (로짓 블렌드) ───────────────────────────────
    mlp_tags, mlp_wts, mlp_enc = [], [], None
    if USE_MLP:
        # 보조 라벨 — asof 누적의 증분에서 복원한다. **학습 타깃 전용**이다.
        aux, aux_cov = modelabel.aux_labels(df)
        print(f"  보조 라벨 복원 {aux_cov:.1%}  λ={MLP_AUX_LAMBDA}  "
              f"축 {list(MLP_AUX_MODES)} ({aux.shape[1]}개)")
        mlp_tags, mlp_wts, mlp_enc, z_mlp_ref = train_mlp_members(
            X, y, w, X_ref, art, MODEL_DIR, t0, y_aux=aux)
        print(f"  블렌드: (1-{MLP_ALPHA})·lgbm + {MLP_ALPHA}·mlp   "
              f"(로짓 상관 {np.corrcoef(raw_ref, z_mlp_ref)[0,1]:.3f})")
        raw_ref = (1 - MLP_ALPHA) * raw_ref + MLP_ALPHA * z_mlp_ref

    # ── 사후 로짓 보정도 delta보다 **먼저** 얹는다 ──────────────
    # predict.py의 순서(블렌드 → 보정 → prior shift)와 같아야 한다.
    # `logit_adjust`는 추론·검증과 **같은 함수**다 — 축을 늘려도 여기는 안 고친다.
    # ⚠️ 참조행(REF_SEASON)의 δ 적중률은 추론행(2025)보다 높다 — §6 버그 2와 같은
    #    구조다. 다만 δ는 투수 자기 평균으로부터의 편차라 **평균이 0에 붙어 있고**
    #    (실측 hand −0.0001 / count4 +0.0007) 미적중은 0으로 떨어지므로 레벨 영향이
    #    사실상 없다. 조회 키에 season이 안 들어가는 것이 Trackman과의 결정적 차이다.
    z_adj, hits = logit_adjust(df[df.season == REF_SEASON], art)
    if len(art["ctx"]):
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

    # ── 저장 ────────────────────────────────────────────────
    # β는 art["ctx"] 안에 축별로 들어 있다 — 번들에 따로 두지 않는다 (두 곳에 두면 어긋난다)
    joblib.dump({"artifacts": art, "feature_cols": cols, "logit_delta": logit_delta,
                 "members": members, "mlp_members": mlp_tags, "mlp_enc": mlp_enc,
                 "mlp_weights": mlp_wts,
                 "mlp_alpha": MLP_ALPHA if USE_MLP else 0.0},
                MODEL_DIR / "artifacts.joblib", compress=3)
    (MODEL_DIR / "meta.json").write_text(json.dumps({
        "seeds": list(SEED_ENSEMBLE), "variants": list(ENSEMBLE_VARIANTS),
        "n_members": len(members), "n_rounds": N_ROUNDS,
        "use_mlp": USE_MLP, "mlp_alpha": MLP_ALPHA if USE_MLP else 0.0,
        "mlp_aux_lambda": MLP_AUX_LAMBDA if USE_MLP else 0.0,
        # ⚠️ 축 목록도 반드시 남긴다. λ만 남기면 헤드 2개짜리와 4개짜리가
        #    meta.json에서 구분이 안 되고, describe()의 태그도 같아진다.
        #    (이 종류의 누락이 s11 플래툰 · s12 카운트 · s13 λ 세 번 있었다.)
        "mlp_aux_modes": list(MLP_AUX_MODES) if USE_MLP else [],
        # 오염 보정 축 (§7.49). 위 경고와 같은 이유로 반드시 남긴다 —
        # 이게 없으면 s13과 s14의 파일명이 번호만 빼고 똑같아진다.
        "contam_axes": list(CONTAM_AXES or []),
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
        # 시즌 진행분 (§7.55). 위 경고와 같은 이유로 반드시 남긴다 —
        # 이게 없으면 s14와 s15의 파일명이 번호만 빼고 똑같아진다.
        "asof_axes": {a["name"]: a["beta"] for a in asof},
        "asof_strength": ASOF_STRENGTH,
        "asof_cells": {a["name"]: int(len(a["keys"])) for a in asof},
        # 레퍼토리 균형도 (§7.58). 같은 이유 — 태그가 두 제출을 구분하려면 여기 있어야 한다.
        "mix_axes": {a["name"]: a["beta"] for a in mix},
        "lgb_params": FINAL_LGB_PARAMS, "lightgbm_version": lgb.__version__,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved to {MODEL_DIR}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
