"""피처 정의 — 이 파일 한 곳에만 (CLAUDE.md). submit/script.py는 이 파일을 import 하거나
파일째 복사해야 하며, 재구현하면 평가 서버에서만 터진다.

★ 배포 제약 (competition.md §4-4): 평가 데이터 각 행은 독립적으로 예측해야 하고,
test.csv 내부 다른 행(자기 자신이 속한 시즌 안의 앞선 투구 포함)을 이용해 현재 행의 피처를
만들 수 없다. 허용되는 건 "공식 asof_* 컬럼"과 "train.csv로 만든 집계를 test 행의
자기 자신 피처로 lookup"뿐이다.

따라서 우리가 만드는 커스텀 누적 피처(R 전용 재구성, 좌우 매치업 분할)는 실제 배포 시
**test 시즌 안에서 계속 갱신될 수 없다.** 학습 구간(fit_df) 끝에서 얼어붙은(frozen)
투수별 정적 lookup이어야 한다 — `pitcher_train_totals.parquet`을 만든 이유가 이거다.

이 파일은 그래서 두 갈래로 나뉜다:
- `build_training_features(fit_df)`: 학습 데이터 자체를 만들 때는 fit_df 내부에서
  strictly-prior 행 단위 누적을 써도 안전하다 (fit_df 자신의 과거 행만 사용, 미래 유출 없음).
- `fit_pitcher_lookup` + `apply_static_features`: valid/test 행에 적용할 때는 fit_df
  전체를 학습 구간으로 압축한 "얼어붙은" 투수별 정적값만 join한다.

컬럼 이름은 두 경로 모두 `_static` 접미사로 통일해 모델이 같은 피처로 인식하게 한다.
"""
import numpy as np
import pandas as pd

TARGET = "control_success"

OFFICIAL_FEATURES = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before",
    "run_total_before", "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state",
    "home_win_expectancy", "away_win_expectancy", "li",
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
assert len(OFFICIAL_FEATURES) == 47

CATEGORICAL_FEATURES = [
    "top_bottom", "game_type", "base_state", "game_month", "game_dayofweek",
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
]

CUSTOM_STATIC_FEATURES = [
    "asof_success_R2_static",
    "asof_success_R_platoon_static",
    "asof_success_R_platoon_n_static",
    "asof_success_R_platoon_blend_static",
]

K_BLEND = 30.0


def _add_prior_rate(df, src, name, mask=None):
    """strictly-prior 누적: cumsum() - 자기값. df는 pitcher_id, asof_pitcher_n 순 정렬돼 있어야 함."""
    v = df[src].to_numpy(float)
    ok = np.isfinite(v)
    m = np.ones(len(df)) if mask is None else mask.astype(float)
    use = ok.astype(float) * m
    val = np.where(ok, v, 0.0) * m
    tmp = pd.DataFrame({"g": df["pitcher_id"].values, "u": use, "v": val})
    cu = tmp.groupby("g")["u"].cumsum() - tmp["u"]
    cv = tmp.groupby("g")["v"].cumsum() - tmp["v"]
    df[name] = np.where(cu > 0, cv / cu, np.nan)
    df[name + "_n"] = cu
    return df


def build_training_features(fit_df: pd.DataFrame) -> pd.DataFrame:
    """학습(fit) 구간 안에서 strictly-prior 행 단위 커스텀 피처를 만든다.

    fit_df 자신의 과거 행만 사용하므로 안전 (미래/타 구간 유출 없음).
    반환된 프레임은 pitcher_id, asof_pitcher_n 순으로 재정렬돼 있다 (원본 순서 보존 불필요 —
    학습에는 순서가 영향을 주지 않음).
    """
    P = fit_df.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index(drop=True)
    maskR = (P["game_type"] == "R").to_numpy()
    same_hand = (P["pitcher_hand"] == P["batter_hand"]).to_numpy()

    P = _add_prior_rate(P, TARGET, "asof_success_R2", mask=maskR)
    P = _add_prior_rate(P, TARGET, "asof_success_R_same", mask=maskR & same_hand)
    P = _add_prior_rate(P, TARGET, "asof_success_R_diff", mask=maskR & (~same_hand))

    platoon = np.where(same_hand, P["asof_success_R_same"], P["asof_success_R_diff"])
    platoon_n = np.where(same_hand, P["asof_success_R_same_n"], P["asof_success_R_diff_n"])
    base = P["asof_success_R2"].to_numpy(float)
    platoon_filled = np.where(np.isfinite(platoon), platoon, base)
    w = np.where(np.isfinite(platoon_n), platoon_n / (platoon_n + K_BLEND), 0.0)
    blend = np.where(np.isfinite(base), w * platoon_filled + (1 - w) * base, platoon_filled)

    P["asof_success_R2_static"] = base
    P["asof_success_R_platoon_static"] = platoon
    P["asof_success_R_platoon_n_static"] = platoon_n
    P["asof_success_R_platoon_blend_static"] = blend
    return P


def fit_pitcher_lookup(fit_df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """fit_df(학습 구간) 전체를 압축한 투수별 정적(frozen) lookup. 배포 시 이 형태로만 써야 한다."""
    R = fit_df.loc[fit_df["game_type"] == "R"]
    same = (R["pitcher_hand"] == R["batter_hand"])

    def agg(sub):
        g = sub.groupby("pitcher_id")[target].agg(["sum", "count"])
        g.columns = ["succ", "n"]
        g["rate"] = g["succ"] / g["n"]
        return g[["n", "rate"]]

    overall = agg(R).rename(columns={"n": "n_R2", "rate": "rate_R2"})
    same_g = agg(R[same]).rename(columns={"n": "n_same", "rate": "rate_same"})
    diff_g = agg(R[~same]).rename(columns={"n": "n_diff", "rate": "rate_diff"})

    lookup = overall.join(same_g, how="outer").join(diff_g, how="outer")
    return lookup


def apply_static_features(df: pd.DataFrame, lookup: pd.DataFrame, k_blend: float = K_BLEND) -> pd.DataFrame:
    """valid/test 행에 얼어붙은 lookup을 join. 현재 행 자신의 pitcher_hand/batter_hand로
    same/diff 버킷을 선택한다 (이건 test 행 '자기 자신'의 피처이므로 허용됨).
    """
    key = df[["pitcher_id", "pitcher_hand", "batter_hand"]].copy()
    same_hand = (key["pitcher_hand"] == key["batter_hand"]).to_numpy()
    j = key.join(lookup, on="pitcher_id")

    platoon_rate = np.where(same_hand, j["rate_same"], j["rate_diff"])
    platoon_n = np.where(same_hand, j["n_same"], j["n_diff"])
    base_rate = j["rate_R2"].to_numpy(float)

    platoon_filled = np.where(np.isfinite(platoon_rate), platoon_rate, base_rate)
    w = np.where(np.isfinite(platoon_n), platoon_n / (platoon_n + k_blend), 0.0)
    blend = np.where(np.isfinite(base_rate), w * platoon_filled + (1 - w) * base_rate, platoon_filled)

    return pd.DataFrame({
        "asof_success_R2_static": base_rate,
        "asof_success_R_platoon_static": platoon_rate,
        "asof_success_R_platoon_n_static": platoon_n,
        "asof_success_R_platoon_blend_static": blend,
    }, index=df.index)


def build_fold_frames(df: pd.DataFrame, train_mask: pd.Series, valid_mask: pd.Series, use_custom: bool):
    """train_mask/valid_mask(src.folds.split 결과)로부터 (fit_df, valid_df)를 만든다.

    use_custom=True면 CUSTOM_STATIC_FEATURES를 두 프레임에 동일한 이름으로 덧붙인다
    (학습 쪽은 strictly-prior 행 단위, 검증 쪽은 얼어붙은 정적 lookup — 배포와 동일 조건).
    """
    fit_raw = df.loc[train_mask].copy()
    valid_df = df.loc[valid_mask].copy()

    if not use_custom:
        return fit_raw, valid_df

    fit_df = build_training_features(fit_raw)
    lookup = fit_pitcher_lookup(fit_raw)
    static_feats = apply_static_features(valid_df, lookup)
    valid_df = pd.concat([valid_df.reset_index(drop=True), static_feats.reset_index(drop=True)], axis=1)
    return fit_df, valid_df


def feature_list(use_custom: bool):
    return OFFICIAL_FEATURES + (CUSTOM_STATIC_FEATURES if use_custom else [])


def categorical_list(use_custom: bool):
    return CATEGORICAL_FEATURES
