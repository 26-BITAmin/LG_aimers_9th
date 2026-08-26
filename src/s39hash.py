"""s39 §7.161b — feature_hash 불일치의 원인 검증 + 증명된 경우에만 재스탬프.

--artifacts-only 가 `저장 d0c049db88dcb4ac / 현재 e8e3753aac7fce30` 으로 죽었다.
`_feature_hash` 는 (X_ref, w, y) 에 `_TRAIN_SHAPING` 을 섞는데, 거기에
`mlp_seeds`·`mlp_variants`(멤버 수) 가 들어 있다 — s39 의 로스터 확장(emb4 5→20)
자체가 해시를 바꾼다. 데이터 쪽이 정말 불변인지 **구 로스터 값을 패치해 재계산**으로
증명한다:

    h_old(구 로스터 payload) == 저장 해시   ⇒  X_ref·w·y 불변, 차이는 로스터뿐
                                              ⇒  기존 멤버 재사용은 안전
                                              ⇒  번들 해시를 h_new 로 재스탬프

일치하지 않으면 **아무것도 쓰지 않고** 죽는다 — 그때는 진짜 피처 드리프트다.
(백업: models_bak/s37_bundle/artifacts.joblib)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np

import train as T
from config import (ASOF_AXES, ASOF_STRENGTH, CONTAM_AXES, CTX_AXES,
                    CTX_HALF_LIFE, CTX_K, DROP_FEATURES, GT_F_BETA, GT_F_DELTA,
                    MIX_AXES, MODEL_DIR, RECENCY_HALF_LIFE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRACKMAN_MODE, TRAIN_CSV,
                    USE_PLAYER_IDS)

# 구 로스터 (s37 번들 시점의 config) — 검증 대상 가설
OLD_SEEDS = [42, 7, 2024, 1234, 31337]
OLD_EMB4_N = 5

t0 = time.time()

# ── main() 의 프리앰블을 그대로 재현한다 (train.py:406~487) ──────────────
df = T.load_csv(TRAIN_CSV)
w = T.recency_weight(df.season.to_numpy(), RECENCY_HALF_LIFE,
                     ref_season=T.REF_SEASON)
tm_table = T.load_table(TRACKMAN_MODE)
ctx = T.ctxfit.fit_ctx(df, TARGET_COL, CTX_AXES, CTX_K,
                       weights=T.recency_weight(df["season"].to_numpy(),
                                                CTX_HALF_LIFE,
                                                ref_season=int(df.season.max())))
art = T.fit_artifacts(df, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                      k_batter=SMOOTH_K_BATTER, tm_table=tm_table, ctx=ctx,
                      contam_axes=CONTAM_AXES)
asof = T.asofdelta.fit_asof_axes(df, ASOF_AXES, ASOF_STRENGTH,
                                 ct=art.get("contam"))
art["asof"] = asof
art["mix"] = T.mixent.fit_mixent(df, MIX_AXES)
art["use_player_ids"] = USE_PLAYER_IDS
art["gt"] = {"delta": GT_F_BETA * GT_F_DELTA}


def build_X(d):
    x = T.build_features(d, art)
    if USE_PLAYER_IDS:
        for c in ("pitcher_id", "batter_id"):
            x[c] = d[c].to_numpy(dtype=np.float32)
    return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])


X = build_X(df)
y = df[TARGET_COL].to_numpy(dtype=np.float64)
cols = X.columns.tolist()
X_ref = T.reference_matrix(df, art, build_X)[cols]
print(f"X_ref {X_ref.shape}  ({time.time()-t0:.0f}s)")

h_new = T._feature_hash(X_ref, w, y)

shaping_now = T._TRAIN_SHAPING
shaping_old = dict(shaping_now)
shaping_old["mlp_seeds"] = OLD_SEEDS
shaping_old["mlp_variants"] = {
    k: (c, OLD_EMB4_N if k == "emb4" else n)
    for k, (c, n) in shaping_now["mlp_variants"].items()}
T._TRAIN_SHAPING = shaping_old
h_old = T._feature_hash(X_ref, w, y)
T._TRAIN_SHAPING = shaping_now

bundle_path = MODEL_DIR / "artifacts.joblib"
bundle = joblib.load(bundle_path)
stored = bundle.get("feature_hash")

print(f"저장           {stored}")
print(f"h_old(구 로스터) {h_old}")
print(f"h_new(현 로스터) {h_new}")

if h_old != stored:
    raise SystemExit("❌ 구 로스터 payload 로도 저장 해시가 재현되지 않는다 — "
                     "로스터 외의 진짜 드리프트가 있다. 재스탬프하지 않는다.")

print("✅ 증명 — 저장 해시가 구 로스터 payload 로 정확히 재현된다.")
print("   X_ref·w·y 불변, 차이는 mlp 로스터뿐 → 기존 멤버 재사용 안전.")
bundle["feature_hash"] = h_new
joblib.dump(bundle, bundle_path)
print(f"재스탬프 완료: feature_hash = {h_new[:16]}…  → --artifacts-only 재실행 가능")
