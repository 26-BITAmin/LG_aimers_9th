# script.py
import os

import catboost as cb
import lightgbm as lgb
import pandas as pd

import features as feat

ID_COL = "row_id"
TARGET_COL = "control_success"

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

# 사후 로짓 보정 (exp46/exp48, model.md). beta=1.0 고정 -- 튜닝 안 함, exp46에서 쓴 값 그대로.
CTX_AXES = ("hand", "count4")
CTX_BETA = 1.0


# =======================
# 데이터 로드 유틸
# =======================

def load_test(path):
    """평가 데이터(csv) 로드. 한 행이 투구 하나."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    """sample_submission.csv 로드 — 제출 파일의 row_id 순서/컬럼 기준."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: "
            f"{list(df.columns)}")
    return df


# =======================
# 학습 때 사용한 전처리 (그대로, features.py 재사용)
# =======================

def build_features(df, lookup):
    """공식 47피처 + CUSTOM_STATIC_FEATURES(얼어붙은 투수별 lookup 적용).

    test.csv 각 행은 독립적으로 예측한다 — lookup은 train.csv 전체(2019-2024)로
    미리 빌드해 model/pitcher_lookup.parquet에 저장해둔 것이고, 여기선 행 자신의
    pitcher_id/pitcher_hand/batter_hand로만 join한다(경쟁 규칙 4-4의 "허용" 항목).
    """
    static_feats = feat.apply_static_features(df, lookup)
    out = pd.concat([df.reset_index(drop=True), static_feats.reset_index(drop=True)], axis=1)
    return out


# =======================
# 제출 파일 생성 유틸
# =======================

def merge_predictions(sub, ids, preds):
    """sample_submission의 row_id 순서에 맞춰 예측 확률 병합.

    예측에 없는 row_id는 sample_submission의 기존 값(placeholder)을 유지한다.
    """
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# =======================
# main
# =======================

def main():
    # ---- 경로 변수 ----
    TEST_DIR = "./data"
    MODEL_DIR = "./model"
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    LGB_MODEL_PATH = os.path.join(MODEL_DIR, "lgb_model.txt")
    CB_MODEL_PATH = os.path.join(MODEL_DIR, "cb_model.cbm")
    LOOKUP_PATH = os.path.join(MODEL_DIR, "pitcher_lookup.csv")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 로드 ----
    print("Load models...")
    lgb_booster = lgb.Booster(model_file=LGB_MODEL_PATH)
    cb_model = cb.CatBoostClassifier()
    cb_model.load_model(CB_MODEL_PATH)
    lookup = pd.read_csv(LOOKUP_PATH).set_index("pitcher_id")
    print(f" OK. lookup pitchers={len(lookup)}")

    ctx_art = {}
    for axis_name in CTX_AXES:
        ctx_path = os.path.join(MODEL_DIR, f"ctx_{axis_name}.csv")
        d = pd.read_csv(ctx_path)
        ctx_art[axis_name] = {"keys": d["key"].astype("int64").to_numpy(),
                               "delta": d["delta"].astype("float64").to_numpy()}
    print(f" ctx axes loaded: {list(ctx_art.keys())}")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    # ---- 전처리 (학습과 동일) ----
    print("Build features...")
    ids = test[ID_COL].tolist()
    full = build_features(test, lookup)
    feature_cols = feat.feature_list(use_custom=True)
    X = full[feature_cols].copy()
    print(f" features={X.shape[1]}")

    # ---- 예측 (제구 성공 확률) — LightGBM 50% + CatBoost 50% ----
    print("Inference LightGBM...")
    X_lgb = X.copy()
    for c in CAT_COLS:
        X_lgb[c] = X_lgb[c].astype("category")
    p_lgb = lgb_booster.predict(X_lgb) if len(X_lgb) else []

    print("Inference CatBoost...")
    X_cb = X.copy()
    for c in CAT_COLS:
        X_cb[c] = X_cb[c].astype(str)
    p_cb = cb_model.predict_proba(X_cb)[:, 1] if len(X_cb) else []

    p_ens = 0.5 * p_lgb + 0.5 * p_cb if len(X) else []
    print(f" ensemble preds={len(p_ens)}")

    # ---- 사후 로짓 보정 (exp46/exp48) -- full 행 자기 자신의 pitcher_id + 상황코드로
    # 조회만 한다 (재학습 없음, 리크 없음) ----
    if len(full):
        z = 0.0
        for axis_name in CTX_AXES:
            z = z + feat.ctx_lookup(full, axis_name, ctx_art)
        preds = feat.sigmoid(feat.logit(p_ens) + CTX_BETA * z)
    else:
        preds = p_ens
    print(f" preds (post ctx correction)={len(preds)}")

    # ---- sample_submission 기반 결과 생성 ----
    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
