"""엔드투엔드 검증 — 학습→직렬화→추론 경로가 실제로 맞는지 확인한다.

세 가지를 본다.

1. **train/serve skew**  학습 때 메모리 안에서 계산한 예측과,
   아티팩트를 저장했다 다시 읽어 `Predictor`로 계산한 예측이 일치하는가.
   (dtype 추론 차이로 범주형 매핑이 조용히 -1로 떨어지는 사고를 잡는다.)

2. **행 독립성**  test를 여러 조각으로 나눠 따로 예측한 결과가
   한 번에 예측한 결과와 **비트 단위로 같은가**.
   → 다르면 test 전체 통계를 쓰고 있다는 뜻 = 대회 규정 위반.
   AST 정적 검사는 "금지 호출이 없다"만 보여주지, 이걸 증명하진 못한다.

3. **실제 성능**  홀드아웃(2024)에서 Brier / BSS / 캘리브레이션 곡선.

실행:  .venv/Scripts/python.exe -u src/verify_pipeline.py
"""
import shutil
import sys
import tempfile
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

from config import (DROP_FEATURES, FINAL_LGB_PARAMS, MLP_ALPHA, N_ROUNDS,
                    RECENCY_HALF_LIFE, SEED_ENSEMBLE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TARGET_PRIOR_2025, TRACKMAN_MODE,
                    TRAIN_CSV, USE_INIT_SCORE, USE_MLP, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from predict import Predictor
from trackman_features import load_table
from train import (reference_matrix, solve_delta, train_ensemble,
                   train_mlp_members)
from validation import bss, decompose, logloss, reliability, _logit, _sigmoid

VALID_SEASON = 2024


def build_X(df, art):
    X = build_features(df, art)
    if USE_PLAYER_IDS:
        for c in ("pitcher_id", "batter_id"):
            X[c] = df[c].to_numpy(dtype=np.float32)
    if DROP_FEATURES:
        X = X.drop(columns=list(DROP_FEATURES))
    return X


def main():
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="verify_"))
    model_dir = tmp / "model"
    model_dir.mkdir()

    print("Loading train.csv ...")
    df = load_csv(TRAIN_CSV)
    tr = df[df.season < VALID_SEASON].reset_index(drop=True)
    va = df[df.season == VALID_SEASON].reset_index(drop=True)
    print(f"  train={len(tr)}  valid({VALID_SEASON})={len(va)}  ({time.time()-t0:.0f}s)")

    ref_season = int(tr.season.max())
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=ref_season)
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, tm_table=load_table(TRACKMAN_MODE))
    art["use_player_ids"] = USE_PLAYER_IDS

    Xtr = build_X(tr, art)
    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)

    if USE_INIT_SCORE:
        rates = tr.groupby(["season", "game_type"], observed=True)[TARGET_COL].mean()
        key = list(zip(tr["season"].to_numpy(), tr["game_type"].astype("object").to_numpy()))
        init_tr = _logit(np.array([rates[k] for k in key]))
        base = float(_logit(np.array([tr.loc[tr.season == ref_season, TARGET_COL].mean()]))[0])
    else:
        init_tr, base = None, 0.0

    print(f"학습: 앙상블 (train.train_ensemble — 배포와 동일한 루프) ...")
    X_ref = reference_matrix(tr, art, lambda d: build_X(d, art),
                             ref_season=ref_season, test_season=VALID_SEASON)[Xtr.columns]
    members, raw_ref = train_ensemble(Xtr, ytr, w, init_tr, X_ref, model_dir)

    mlp_tags, mlp_enc = [], None
    if USE_MLP:
        mlp_tags, mlp_enc, z_mlp_ref = train_mlp_members(Xtr, ytr, w, X_ref, art,
                                                         model_dir)
        raw_ref = (1 - MLP_ALPHA) * raw_ref + MLP_ALPHA * z_mlp_ref

    delta = solve_delta(base + raw_ref, TARGET_PRIOR_2025)
    joblib.dump({"artifacts": art, "feature_cols": Xtr.columns.tolist(),
                 "logit_delta": base + delta, "members": members,
                 "mlp_members": mlp_tags, "mlp_enc": mlp_enc,
                 "mlp_alpha": MLP_ALPHA if USE_MLP else 0.0},
                model_dir / "artifacts.joblib", compress=3)

    # ── 1. train/serve skew ────────────────────────────────
    print("\n=== 1. train/serve skew ===")
    csv_path = tmp / "test.csv"
    va.drop(columns=[TARGET_COL]).to_csv(csv_path, index=False, encoding="utf-8")
    va_served = pd.read_csv(csv_path, encoding="utf-8-sig")   # 평가 서버와 동일: dtype 추론

    predictor = Predictor(str(model_dir))
    p_served = predictor.predict(va_served)

    # 학습 경로에서 직접 계산한 값.
    # ⚠️ `va`는 train.csv를 float32 dtype으로 읽은 것이고 `va_served`는 CSV에서
    #    pandas 기본 추론(float64)으로 읽은 것이다. 이 두 경로가 갈리는 것이
    #    예전에 실제로 났던 사고다 (features._col64 주석 참조).
    #    MLP 인코더도 같은 위험이 있으므로 여기에 함께 태운다.
    #    (torch ↔ numpy 등가성은 export 시점에 train_mlp_members가 이미 검사했다.)
    Xva = build_X(va, art)[predictor.cols]
    z = np.zeros(len(Xva))
    for bst_ in predictor.boosters:
        z += bst_.predict(Xva, raw_score=True)
    z /= len(predictor.boosters)
    if predictor.alpha > 0:
        zm = np.zeros(len(Xva))
        for m_ in predictor.mlps:
            zm += m_.logit(Xva)
        z = (1 - predictor.alpha) * z + predictor.alpha * (zm / len(predictor.mlps))
    p_direct = _sigmoid(z + predictor.delta)

    d = np.abs(p_served - p_direct).max()
    print(f"  max|serve - direct| = {d:.3e}   {'✅ OK' if d < 1e-9 else '❌ SKEW!'}")
    failures = [] if d < 1e-9 else ["train/serve skew"]

    # ── 2. 행 독립성 ────────────────────────────────────────
    print("\n=== 2. 행 독립성 (조각별 예측 == 통짜 예측) ===")
    chunks = [va_served.iloc[i:i + 1000] for i in range(0, min(20000, len(va_served)), 1000)]
    p_chunked = np.concatenate([predictor.predict(c) for c in chunks])
    n = len(p_chunked)
    d2 = np.abs(p_chunked - p_served[:n]).max()
    print(f"  {len(chunks)}조각 x 1000행,  max diff = {d2:.3e}   "
          f"{'✅ 각 행이 독립적으로 예측된다' if d2 == 0.0 else '❌ 다른 행에 의존한다!'}")
    if d2 != 0.0:
        failures.append("행 독립성 (조각 분할)")

    # 행 순서를 섞어도 같은지 (순서 기반 피처가 없는지)
    perm = np.random.default_rng(0).permutation(len(va_served))
    p_perm = predictor.predict(va_served.iloc[perm].reset_index(drop=True))
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    d3 = np.abs(p_perm[inv] - p_served).max()
    print(f"  행 순서 셔플 후 max diff = {d3:.3e}   "
          f"{'✅ 순서 무관' if d3 == 0.0 else '❌ 순서에 의존한다!'}")
    if d3 != 0.0:
        failures.append("행 순서 독립성")

    # ── 3. 성능 ────────────────────────────────────────────
    print(f"\n=== 3. 홀드아웃 성능 (valid = {VALID_SEASON}) ===")
    dec = decompose(yva, p_served)
    print(f"  Brier      = {dec['brier']:.6f}   (BS_ref = {dec['ref']:.6f})")
    print(f"  BSS        = {bss(yva, p_served):.1f}")
    print(f"  LogLoss    = {logloss(yva, p_served):.6f}")
    print(f"  pred_mean  = {dec['pred_mean']:.4f}  (실제 {dec['true_mean']:.4f}, "
          f"레벨손실 {dec['mean_shift_loss']:.6f} → BSS {-dec['mean_shift_loss']/dec['ref']*1e5:.0f})")
    print(f"  pred_std   = {dec['pred_std']:.4f}")

    print("\n  캘리브레이션 (예측 십분위별):")
    print(f"    {'bin':>3s} {'n':>8s} {'예측평균':>9s} {'실제평균':>9s} {'차이':>9s}")
    for b, cnt, pm, ym in reliability(yva, p_served, n_bins=10):
        print(f"    {b:>3d} {cnt:>8d} {pm:>9.4f} {ym:>9.4f} {pm-ym:>+9.4f}")

    print(f"\n  ※ 위 pred_mean은 2025 목표 prior({TARGET_PRIOR_2025})로 이동시킨 값이므로")
    print(f"    2024 실제값(0.4861)과 어긋나는 것이 정상이다. 레벨을 뺀 순수 성능:")
    from validation import shift_to_prior
    print(f"    BSS(oracle shift) = {bss(yva, shift_to_prior(p_served, yva.mean())):.1f}")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\ntotal {time.time()-t0:.0f}s")

    if failures:
        raise SystemExit("❌ 검증 실패: " + ", ".join(failures))
    print("✅ 모든 검증 통과")


if __name__ == "__main__":
    main()
