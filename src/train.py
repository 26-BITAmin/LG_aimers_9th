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

from config import (DROP_FEATURES, ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS,
                    MLP_ALPHA, MLP_SEEDS, MLP_VARIANTS, MODEL_DIR, N_ROUNDS,
                    RECENCY_HALF_LIFE, SEED_ENSEMBLE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TARGET_PRIOR_2025, TRAIN_CSV,
                    TRACKMAN_MODE, USE_INIT_SCORE, USE_MLP, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
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


def train_mlp_members(X, y, w, X_ref, art, model_dir, t0=None):
    """MLP 멤버를 학습해 **numpy 가중치(npz)로** 내보내고 참조 로짓을 반환한다.

    ⚠️ torch는 학습에만 쓴다. 추론은 `mlp_infer.py`의 numpy 순전파다
       (제출 zip에 torch를 넣지 않는다 — 오프라인 설치 10분 / pickle 버전 /
        CUDA 유무 위험이 통째로 사라진다).

    ⚠️ 학습 루프는 `mlp.train_mlp` 한 곳에만 둔다. 여기서는 호출만 한다.
    """
    import numpy as _np

    import mlp
    from mlp_infer import MLPPredictor

    t0 = time.time() if t0 is None else t0
    enc = mlp.fit_encoder(X, art)
    A, C = mlp.encode_apply(X, enc)
    A_ref, C_ref = mlp.encode_apply(X_ref, enc)

    tags, z_ref = [], _np.zeros(len(X_ref))
    n = sum(k for _, k in MLP_VARIANTS.values())
    for vname, (cfg, nseed) in MLP_VARIANTS.items():
        for seed in MLP_SEEDS[:nseed]:
            m, _ = mlp.train_mlp(A, C, y, w, A_ref, C_ref, enc["sizes"], seed=seed,
                                 verbose=False, **cfg)
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
            z_ref += z_np
            print(f"  [mlp {len(tags)}/{n}] {tag} 완료 "
                  f"(export skew {d:.1e})  ({time.time()-t0:.0f}s)")
    return tags, enc, z_ref / len(tags)


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
    art = fit_artifacts(df, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, tm_table=tm_table)
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
    mlp_tags, mlp_enc = [], None
    if USE_MLP:
        mlp_tags, mlp_enc, z_mlp_ref = train_mlp_members(X, y, w, X_ref, art,
                                                         MODEL_DIR, t0)
        print(f"  블렌드: (1-{MLP_ALPHA})·lgbm + {MLP_ALPHA}·mlp   "
              f"(로짓 상관 {np.corrcoef(raw_ref, z_mlp_ref)[0,1]:.3f})")
        raw_ref = (1 - MLP_ALPHA) * raw_ref + MLP_ALPHA * z_mlp_ref

    # ── prior shift: BASE + DELTA를 하나의 스칼라로 ──────────
    # ⚠️ 블렌드 **이후**의 로짓으로 delta를 푼다. 순서가 바뀌면 목표 prior를 못 맞춘다.
    p_before = _sigmoid(base + raw_ref)
    delta = solve_delta(base + raw_ref, TARGET_PRIOR_2025)
    logit_delta = base + delta
    print(f"\nprior shift: ref({REF_SEASON}) pred_mean={p_before.mean():.4f} "
          f"→ target={TARGET_PRIOR_2025:.4f}")
    print(f"  base={base:+.5f}  delta={delta:+.5f}  → logit_delta={logit_delta:+.5f}")

    # ── 저장 ────────────────────────────────────────────────
    joblib.dump({"artifacts": art, "feature_cols": cols, "logit_delta": logit_delta,
                 "members": members, "mlp_members": mlp_tags, "mlp_enc": mlp_enc,
                 "mlp_alpha": MLP_ALPHA if USE_MLP else 0.0},
                MODEL_DIR / "artifacts.joblib", compress=3)
    (MODEL_DIR / "meta.json").write_text(json.dumps({
        "seeds": list(SEED_ENSEMBLE), "variants": list(ENSEMBLE_VARIANTS),
        "n_members": len(members), "n_rounds": N_ROUNDS,
        "use_mlp": USE_MLP, "mlp_alpha": MLP_ALPHA if USE_MLP else 0.0,
        "mlp_seeds": list(MLP_SEEDS) if USE_MLP else [],
        "mlp_members": mlp_tags, "mlp_variants": MLP_VARIANTS,
        "recency_half_life": RECENCY_HALF_LIFE, "use_init_score": USE_INIT_SCORE,
        "use_player_ids": USE_PLAYER_IDS, "drop_features": list(DROP_FEATURES),
        "smooth_k": SMOOTH_K_PITCHER, "trackman_mode": TRACKMAN_MODE,
        "target_prior_2025": TARGET_PRIOR_2025, "ref_season": REF_SEASON,
        "base_logit": base, "delta": delta, "logit_delta": logit_delta,
        "n_features": len(cols), "train_rows": int(len(df)),
        "lgb_params": FINAL_LGB_PARAMS, "lightgbm_version": lgb.__version__,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved to {MODEL_DIR}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
