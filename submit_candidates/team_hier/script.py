# script.py -- 챔피언(LightGBM+CatBoost+사후ctx보정) + 투수→팀→리그 3단 축소 피처 후보
import os

import catboost as cb
import lightgbm as lgb
import pandas as pd

import features as feat

ID_COL = "row_id"
TARGET_COL = "control_success"

CAT_COLS = ["top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]

CTX_AXES = ("hand", "count4")
CTX_BETA = 1.0


def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: {list(df.columns)}")
    return df


def build_features(df, lookup, team_lookup, league_rate):
    """공식 47피처 + CUSTOM_STATIC_FEATURES + 팀 계층 피처(row 자기 자신의
    pitcher_id/pitcher_team_id로만 join -- 경쟁 규칙 4-4 허용 항목)."""
    static_feats = feat.apply_static_features(
        df, lookup, with_team_hier=True, team_lookup=team_lookup, league_rate=league_rate)
    out = pd.concat([df.reset_index(drop=True), static_feats.reset_index(drop=True)], axis=1)
    return out


def merge_predictions(sub, ids, preds):
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


def main():
    TEST_DIR = "./data"
    MODEL_DIR = "./model"
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    LGB_MODEL_PATH = os.path.join(MODEL_DIR, "lgb_model.txt")
    CB_MODEL_PATH = os.path.join(MODEL_DIR, "cb_model.cbm")
    LOOKUP_PATH = os.path.join(MODEL_DIR, "pitcher_lookup.csv")
    TEAM_LOOKUP_PATH = os.path.join(MODEL_DIR, "team_lookup.csv")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    print("Load models...")
    lgb_booster = lgb.Booster(model_file=LGB_MODEL_PATH)
    cb_model = cb.CatBoostClassifier()
    cb_model.load_model(CB_MODEL_PATH)
    lookup = pd.read_csv(LOOKUP_PATH).set_index("pitcher_id")
    print(f" OK. lookup pitchers={len(lookup)}")

    team_raw = pd.read_csv(TEAM_LOOKUP_PATH)
    league_rate = float(team_raw["league_rate"].iloc[0])
    team_lookup = team_raw.set_index("pitcher_team_id")[["n_team", "rate_team"]]
    print(f" team_lookup teams={len(team_lookup)}  league_rate={league_rate:.4f}")

    ctx_art = {}
    for axis_name in CTX_AXES:
        d = pd.read_csv(os.path.join(MODEL_DIR, f"ctx_{axis_name}.csv"))
        ctx_art[axis_name] = {"keys": d["key"].astype("int64").to_numpy(),
                               "delta": d["delta"].astype("float64").to_numpy()}
    print(f" ctx axes loaded: {list(ctx_art.keys())}")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    print("Build features...")
    ids = test[ID_COL].tolist()
    full = build_features(test, lookup, team_lookup, league_rate)
    feature_cols = feat.feature_list(use_custom=True, use_team_hier=True)
    X = full[feature_cols].copy()
    print(f" features={X.shape[1]}")

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

    if len(full):
        z = 0.0
        for axis_name in CTX_AXES:
            z = z + feat.ctx_lookup(full, axis_name, ctx_art)
        preds = feat.sigmoid(feat.logit(p_ens) + CTX_BETA * z)
    else:
        preds = p_ens
    print(f" preds (post ctx correction)={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
