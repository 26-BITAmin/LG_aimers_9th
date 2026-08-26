"""§7.165 — `_TRAIN_SHAPING` 로스터 분리에 따른 feature_hash 1회 마이그레이션.

train.py 가 payload 에서 `mlp_seeds` 와 멤버 수를 뺐다 (재추첨마다 관문이 막혀
"늘 마이그레이션되는 문"이 되는 것을 막기 위해 — 멤버별 cfg 는 그대로 해시된다).

s39hash.py 와 같은 계약: **구 payload(로스터 포함, 현 config 값)가 저장 해시를
바이트 재현할 때만** 새 payload 해시로 재스탬프한다. 재현 실패면 아무것도 안 쓴다.
⚠️ config 의 MLP_SEEDS 가 스탬프 시점과 같을 때 돌려야 한다 (s40 시드 교체 **전**).
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
                    MIX_AXES, MLP_SEEDS, MLP_VARIANTS, MODEL_DIR,
                    RECENCY_HALF_LIFE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    TARGET_COL, TRACKMAN_MODE, TRAIN_CSV, USE_PLAYER_IDS)

t0 = time.time()

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
art["asof"] = T.asofdelta.fit_asof_axes(df, ASOF_AXES, ASOF_STRENGTH,
                                        ct=art.get("contam"))
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

shaping_new = T._TRAIN_SHAPING          # 로스터 없는 현 payload
h_new = T._feature_hash(X_ref, w, y)

# 구 payload — 로스터 필드를 스탬프 시점(= 현 config) 값으로 되살린다
shaping_old = dict(shaping_new)
shaping_old["mlp_variants"] = {k: (dict(c), n) for k, (c, n) in MLP_VARIANTS.items()}
shaping_old["mlp_seeds"] = list(MLP_SEEDS)
T._TRAIN_SHAPING = shaping_old
h_old = T._feature_hash(X_ref, w, y)
T._TRAIN_SHAPING = shaping_new

bundle_path = MODEL_DIR / "artifacts.joblib"
bundle = joblib.load(bundle_path)
stored = bundle.get("feature_hash")

print(f"저장                {stored}")
print(f"h_old(로스터 포함)   {h_old}")
print(f"h_new(로스터 제외)   {h_new}")

if h_old != stored:
    raise SystemExit("❌ 구 payload 로 저장 해시가 재현되지 않는다 — 로스터 외의 "
                     "드리프트가 있다. 재스탬프하지 않는다.")

print("✅ 증명 — 차이는 payload 의 로스터 필드뿐. 재스탬프한다.")
bundle["feature_hash"] = h_new
joblib.dump(bundle, bundle_path)
print(f"완료: feature_hash = {h_new[:16]}…")
