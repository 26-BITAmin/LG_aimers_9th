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
- `build_training_features(fit_df)`: 학습 구간 안에서도 **시즌 단위 frozen snapshot**만
  쓴다 (행 단위 strictly-prior cumsum이 아님 — 이유는 함수 docstring 참조, exp03에서
  실측으로 걸러낸 train/serve skew 버그다).
- `fit_pitcher_lookup` + `apply_static_features`: valid/test 행에 적용할 때는 fit_df
  전체를 학습 구간으로 압축한 "얼어붙은" 투수별 정적값만 join한다.

컬럼 이름은 두 경로 모두 `_static` 접미사로 통일해 모델이 같은 피처로 인식하게 한다.
"""
import os

import numpy as np
import pandas as pd

TARGET = "control_success"

_DECODED_PATH = os.path.join(os.path.dirname(__file__), "..", "EDA", "artifacts",
                              "decoded_pitch_outcomes.parquet")
_decoded_cache = None


def _load_decoded():
    """EDA 노트북 02가 만든 행 단위 실현 결과(reverse/middle/ball/strike). train.csv에서만
    파생됐고 test 행은 절대 건드리지 않는다 — 배포 규칙과 무관하게 항상 안전.
    투수당 마지막 1구는 디코딩 손실로 NaN (792행, 무시 가능)."""
    global _decoded_cache
    if _decoded_cache is None:
        d = pd.read_parquet(_DECODED_PATH, columns=["row_id", "reverse", "middle", "ball", "strike"])
        _decoded_cache = d.set_index("row_id")
    return _decoded_cache

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

TREND_FEATURES = ["trend_prev1", "trend_prev3", "trend_prev5"]

FAILMODE_FEATURES = ["asof_reverse_R_static", "asof_middle_R_static"]

BALLSTRIKE_FEATURES = ["asof_ball_R_static", "asof_strike_R_static"]

K_BLEND = 30.0


def _components(with_failmode, with_ballstrike):
    c = []
    if with_failmode:
        c += ["reverse", "middle"]
    if with_ballstrike:
        c += ["ball", "strike"]
    return c


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """공식 prev1/3/5 game rate를 career rate 대비 편차로 바꾼 "최근 컨디션" 피처.
    전부 이미 제공되는 공식 컬럼의 행 단위 뺄셈이라 train/valid/test 어디서든 안전 —
    frozen lookup 같은 특별 취급이 필요 없다.
    """
    df = df.copy()
    base = df["asof_pitcher_success_rate"]
    df["trend_prev1"] = df["asof_pitcher_prev1_game_success_rate"] - base
    df["trend_prev3"] = df["asof_pitcher_prev3_game_success_rate"] - base
    df["trend_prev5"] = df["asof_pitcher_prev5_game_success_rate"] - base
    return df


def build_training_features(fit_df: pd.DataFrame, with_failmode: bool = False,
                             with_ballstrike: bool = False) -> pd.DataFrame:
    """학습(fit) 구간 안에서 커스텀 피처를 만든다.

    ★ 시즌 단위 frozen snapshot — 행 단위 strictly-prior cumsum이 **아니다**.
    처음엔 행 단위(각 투구 직전까지 누적)로 만들었는데, exp03에서 실측 결과 오히려
    성능이 떨어졌다 (fold A/B/C 전부 악화). 원인을 진단해보니: 행 단위 누적은 학습 중
    같은 투수의 인접한 행들끼리 시계열 자기상관을 만들어 학습 분포가 매우 넓고 세밀한
    반면(고유값 27만 개, std 0.058, 0~1 범위), 실제 배포에서 검증/테스트 행에 적용하는
    값은 시즌 하나당 투수 1명에 상수 1개(고유값 225개, std 0.047, 0.36~0.76 범위)라
    분포가 전혀 다르다 — 전형적인 train/serve skew. 트리가 학습 때 본 넓은 분포에
    맞춰 분기를 잡으면 검증의 좁은 분포에서는 그 분기가 어긋난다.

    고쳐서: 학습 구간 안에서도 "그 시즌 시작 시점에 얼어붙은, 이전 시즌들만으로 만든
    투수별 정적값"을 쓴다 — 배포 시 결과물과 정확히 같은 결(texture)이 되도록.
    같은 이유로 fit_df의 **첫 시즌**은 그 이전 이력이 fit_df 안에 없으므로 전부 NaN
    (콜드스타트, 결측 유지 정책과 일치).
    """
    comps = _components(with_failmode, with_ballstrike)
    cols = CUSTOM_STATIC_FEATURES + (FAILMODE_FEATURES if with_failmode else []) \
        + (BALLSTRIKE_FEATURES if with_ballstrike else [])
    seasons = sorted(fit_df["season"].unique())
    frames = []
    for s in seasons:
        season_rows = fit_df.loc[fit_df["season"] == s]
        prior = fit_df.loc[fit_df["season"] < s]
        if len(prior) == 0:
            static_feats = pd.DataFrame({c: np.nan for c in cols}, index=season_rows.index)
        else:
            lookup = fit_pitcher_lookup(prior, components=comps)
            static_feats = apply_static_features(season_rows, lookup, components=comps)
        frames.append(pd.concat([season_rows, static_feats], axis=1))
    return pd.concat(frames, axis=0)


_COMPONENT_RATE_COL = {"reverse": "rate_reverse_R", "middle": "rate_middle_R",
                        "ball": "rate_ball_R", "strike": "rate_strike_R"}
_COMPONENT_OUT_COL = {"reverse": "asof_reverse_R_static", "middle": "asof_middle_R_static",
                       "ball": "asof_ball_R_static", "strike": "asof_strike_R_static"}


def fit_pitcher_lookup(fit_df: pd.DataFrame, target: str = TARGET,
                        components=()) -> pd.DataFrame:
    """fit_df(학습 구간) 전체를 압축한 투수별 정적(frozen) lookup. 배포 시 이 형태로만 써야 한다.

    components에 "reverse"/"middle"/"ball"/"strike"를 넣으면 그 성분의 R 전용 누적률도
    같이 만든다 (train.csv에서만 파생된 디코딩 결과, EDA/artifacts/decoded_pitch_outcomes.parquet).
    공식 asof_pitcher_*_rate는 전부 2022 이전 F 오염이 섞여 있는데(EDA_정리.md §5),
    success와 같은 방식으로 R만 걸러서 다시 만든다.
    """
    R = fit_df.loc[fit_df["game_type"] == "R"].copy()
    if components:
        dec = _load_decoded()
        R = R.join(dec, on="row_id")
    same = (R["pitcher_hand"] == R["batter_hand"])

    def agg(sub, col):
        g = sub.groupby("pitcher_id")[col].agg(["sum", "count"])
        g.columns = ["succ", "n"]
        g["rate"] = g["succ"] / g["n"]
        return g[["n", "rate"]]

    overall = agg(R, target).rename(columns={"n": "n_R2", "rate": "rate_R2"})
    same_g = agg(R[same], target).rename(columns={"n": "n_same", "rate": "rate_same"})
    diff_g = agg(R[~same], target).rename(columns={"n": "n_diff", "rate": "rate_diff"})
    lookup = overall.join(same_g, how="outer").join(diff_g, how="outer")

    for comp in components:
        col = _COMPONENT_RATE_COL[comp]
        g = agg(R, comp).rename(columns={"rate": col})[[col]]
        lookup = lookup.join(g, how="outer")
    return lookup


def apply_static_features(df: pd.DataFrame, lookup: pd.DataFrame, k_blend: float = K_BLEND,
                           components=()) -> pd.DataFrame:
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

    out = {
        "asof_success_R2_static": base_rate,
        "asof_success_R_platoon_static": platoon_rate,
        "asof_success_R_platoon_n_static": platoon_n,
        "asof_success_R_platoon_blend_static": blend,
    }
    for comp in components:
        out[_COMPONENT_OUT_COL[comp]] = j[_COMPONENT_RATE_COL[comp]].to_numpy(float)
    return pd.DataFrame(out, index=df.index)


def build_fold_frames(df: pd.DataFrame, train_mask: pd.Series, valid_mask: pd.Series,
                       use_custom: bool, use_trend: bool = False, use_failmode: bool = False,
                       use_ballstrike: bool = False):
    """train_mask/valid_mask(src.folds.split 결과)로부터 (fit_df, valid_df)를 만든다.

    use_custom=True면 CUSTOM_STATIC_FEATURES(+ use_failmode/use_ballstrike로 성분 추가)를
    두 프레임에 동일한 이름으로 덧붙인다 (시즌 단위 frozen snapshot, 배포와 동일 조건).
    use_trend=True면 TREND_FEATURES(행 단위 뺄셈, 항상 안전)를 덧붙인다.
    """
    fit_raw = df.loc[train_mask].copy()
    valid_df = df.loc[valid_mask].copy()
    comps = _components(use_failmode, use_ballstrike)

    if use_custom:
        fit_df = build_training_features(fit_raw, with_failmode=use_failmode,
                                          with_ballstrike=use_ballstrike)
        lookup = fit_pitcher_lookup(fit_raw, components=comps)
        static_feats = apply_static_features(valid_df, lookup, components=comps)
        valid_df = pd.concat([valid_df.reset_index(drop=True), static_feats.reset_index(drop=True)], axis=1)
    else:
        fit_df = fit_raw

    if use_trend:
        fit_df = add_trend_features(fit_df)
        valid_df = add_trend_features(valid_df)

    return fit_df, valid_df


def feature_list(use_custom: bool, use_trend: bool = False, use_failmode: bool = False,
                  use_ballstrike: bool = False):
    cols = list(OFFICIAL_FEATURES)
    if use_custom:
        cols += CUSTOM_STATIC_FEATURES
        if use_failmode:
            cols += FAILMODE_FEATURES
        if use_ballstrike:
            cols += BALLSTRIKE_FEATURES
    if use_trend:
        cols += TREND_FEATURES
    return cols


def categorical_list(use_custom: bool):
    return CATEGORICAL_FEATURES
