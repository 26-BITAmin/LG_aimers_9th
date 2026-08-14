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

FAILMODE_PLATOON_FEATURES = ["asof_reverse_R_platoon_blend_static", "asof_middle_R_platoon_blend_static"]

N_EXPOSED_FEATURES = ["asof_success_R2_n_static"]

SEASON_TREND_FEATURES = ["asof_success_R_last_season_blend_static", "asof_success_R_season_slope_static"]

TEAM_HIER_FEATURES = ["asof_success_R_team_blend_static"]

EWMA_FEATURES = ["asof_success_R_ewma_static"]

K_BLEND = 30.0
K_TEAM = 500.0
K_EWMA_HALFLIFE = 150.0  # 투구 단위 반감기 -- 대략 prev3~5게임 스케일, 튜닝 안 함


def _components(with_failmode, with_ballstrike):
    c = []
    if with_failmode:
        c += ["reverse", "middle"]
    if with_ballstrike:
        c += ["ball", "strike"]
    return c


def _platoon_components(with_failmode_platoon):
    return ["reverse", "middle"] if with_failmode_platoon else []


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
                             with_ballstrike: bool = False,
                             with_failmode_platoon: bool = False,
                             with_n_exposed: bool = False,
                             with_season_trend: bool = False,
                             with_team_hier: bool = False,
                             with_ewma: bool = False) -> pd.DataFrame:
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
    platoon_comps = _platoon_components(with_failmode_platoon)
    cols = CUSTOM_STATIC_FEATURES + (FAILMODE_FEATURES if with_failmode else []) \
        + (BALLSTRIKE_FEATURES if with_ballstrike else []) \
        + (FAILMODE_PLATOON_FEATURES if with_failmode_platoon else []) \
        + (N_EXPOSED_FEATURES if with_n_exposed else []) \
        + (SEASON_TREND_FEATURES if with_season_trend else []) \
        + (TEAM_HIER_FEATURES if with_team_hier else []) \
        + (EWMA_FEATURES if with_ewma else [])
    seasons = sorted(fit_df["season"].unique())
    frames = []
    for s in seasons:
        season_rows = fit_df.loc[fit_df["season"] == s]
        prior = fit_df.loc[fit_df["season"] < s]
        if len(prior) == 0:
            static_feats = pd.DataFrame({c: np.nan for c in cols}, index=season_rows.index)
        else:
            lookup = fit_pitcher_lookup(prior, components=comps, platoon_components=platoon_comps,
                                         with_season_trend=with_season_trend)
            team_lookup, league_rate = fit_team_lookup(prior) if with_team_hier else (None, None)
            ewma_lookup = fit_pitcher_ewma(prior) if with_ewma else None
            static_feats = apply_static_features(season_rows, lookup, components=comps,
                                                   platoon_components=platoon_comps,
                                                   with_n_exposed=with_n_exposed,
                                                   with_season_trend=with_season_trend,
                                                   with_team_hier=with_team_hier,
                                                   team_lookup=team_lookup, league_rate=league_rate,
                                                   with_ewma=with_ewma, ewma_lookup=ewma_lookup)
        frames.append(pd.concat([season_rows, static_feats], axis=1))
    return pd.concat(frames, axis=0)


_COMPONENT_RATE_COL = {"reverse": "rate_reverse_R", "middle": "rate_middle_R",
                        "ball": "rate_ball_R", "strike": "rate_strike_R"}
_COMPONENT_OUT_COL = {"reverse": "asof_reverse_R_static", "middle": "asof_middle_R_static",
                       "ball": "asof_ball_R_static", "strike": "asof_strike_R_static"}


def fit_pitcher_lookup(fit_df: pd.DataFrame, target: str = TARGET,
                        components=(), platoon_components=(),
                        with_season_trend: bool = False) -> pd.DataFrame:
    """fit_df(학습 구간) 전체를 압축한 투수별 정적(frozen) lookup. 배포 시 이 형태로만 써야 한다.

    components에 "reverse"/"middle"/"ball"/"strike"를 넣으면 그 성분의 R 전용 누적률(좌우
    구분 없음)도 같이 만든다. platoon_components에 넣으면 success와 동일한 same/diff-hand
    분할 + 카운트도 함께 만들어 apply_static_features에서 credibility-blend 할 수 있게 한다
    (success의 `asof_success_R_platoon_blend_static`이 +85~97점 낸 것과 같은 메커니즘을
    실패 성분에 재사용 — reverse는 정의상 좌우 매치업에 따라 갈릴 개연성이 있다).
    둘 다 train.csv에서만 파생된 디코딩 결과(EDA/artifacts/decoded_pitch_outcomes.parquet)를
    쓴다. 공식 asof_pitcher_*_rate는 전부 2022 이전 F 오염이 섞여 있는데(EDA_정리.md §5),
    success와 같은 방식으로 R만 걸러서 다시 만든다.
    """
    R = fit_df.loc[fit_df["game_type"] == "R"].copy()
    if components or platoon_components:
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

    for comp in platoon_components:
        base_col = _COMPONENT_RATE_COL[comp]
        n_same_col, rate_same_col = f"n_{comp}_same", f"rate_{comp}_same"
        n_diff_col, rate_diff_col = f"n_{comp}_diff", f"rate_{comp}_diff"
        base_g = agg(R, comp).rename(columns={"rate": base_col})[[base_col]]
        same_gc = agg(R[same], comp).rename(columns={"n": n_same_col, "rate": rate_same_col})
        diff_gc = agg(R[~same], comp).rename(columns={"n": n_diff_col, "rate": rate_diff_col})
        lookup = lookup.join(base_g, how="outer").join(same_gc, how="outer").join(diff_gc, how="outer")

    if with_season_trend:
        last_season = R["season"].max()
        is_last = (R["season"] == last_season)
        last_g = agg(R[is_last], target).rename(columns={"n": "n_last_season", "rate": "rate_last_season"})
        earlier_g = agg(R[~is_last], target).rename(columns={"n": "n_earlier_career", "rate": "rate_earlier_career"})
        lookup = lookup.join(last_g, how="outer").join(earlier_g, how="outer")
    return lookup


def fit_team_lookup(fit_df: pd.DataFrame, target: str = TARGET):
    """R전용 팀별 정적(frozen) lookup + 리그 전체 평균 — 투수→팀→리그 3단 축소의
    중간·최상위 계층. 신인/저표본 투수가 곧장 리그 평균으로 뭉개지지 않고 자기
    팀 수준(코칭스태프·포수진 등 체계 효과 가설)으로 먼저 당겨지게 한다.

    fit_pitcher_lookup과 마찬가지로 fit_df(학습 구간, strictly-prior)에서만 호출한다.
    """
    R = fit_df.loc[fit_df["game_type"] == "R"]
    league_rate = float(R[target].mean()) if len(R) else np.nan
    g = R.groupby("pitcher_team_id")[target].agg(["sum", "count"])
    g.columns = ["succ", "n_team"]
    g["rate_team"] = g["succ"] / g["n_team"]
    return g[["n_team", "rate_team"]], league_rate


def fit_pitcher_ewma(fit_df: pd.DataFrame, target: str = TARGET,
                      half_life: float = K_EWMA_HALFLIFE) -> pd.DataFrame:
    """R전용, 투구(행) 단위 지수감쇠 "최근 폼" -- `row_id` 연대순(CLAUDE.md 불변식)을 이용해
    최근 투구일수록 가중치가 크다. career 평균(`asof_success_R2_static`, 감쇠 없음)과는
    다른 축 -- "최근이 평생 평균에서 얼마나 벗어났는지"가 신호일 수 있다는 가설.

    공식 prev1/3/5_game_rate(TREND_FEATURES)와도 다르다: 그건 F오염된 공식 컬럼이고
    "게임" 단위인데, 여기는 게임 ID가 데이터에 없어(competition.md) 투구 단위로 감쇠시킨다.

    fit_pitcher_lookup과 마찬가지로 fit_df(학습 구간, strictly-prior)에서만 호출한다.
    """
    R = fit_df.loc[fit_df["game_type"] == "R"].sort_values("row_id")
    rank_from_end = R.groupby("pitcher_id").cumcount(ascending=False).to_numpy(dtype=np.float64)
    w = np.power(0.5, rank_from_end / half_life)
    y = R[target].to_numpy(dtype=np.float64)
    pid = R["pitcher_id"].to_numpy(dtype=np.int64)
    tmp = pd.DataFrame({"w": w, "wy": w * y}, index=pid)
    agg = tmp.groupby(tmp.index).sum()
    agg["rate_ewma"] = agg["wy"] / agg["w"]
    agg.index.name = "pitcher_id"
    return agg[["rate_ewma"]]


def _team_hier_blend(pitcher_rate, pitcher_n, team_rate, team_n, league_rate,
                      k_pitcher=K_BLEND, k_team=K_TEAM):
    """3단 축소: 투수 표본이 적으면 팀 수준으로, 팀 표본도 적거나 결측이면 리그로 당긴다."""
    team_filled = np.where(np.isfinite(team_rate), team_rate, league_rate)
    w_team = np.where(np.isfinite(team_n), team_n / (team_n + k_team), 0.0)
    team_blend = w_team * team_filled + (1 - w_team) * league_rate
    pitcher_filled = np.where(np.isfinite(pitcher_rate), pitcher_rate, team_blend)
    w_p = np.where(np.isfinite(pitcher_n), pitcher_n / (pitcher_n + k_pitcher), 0.0)
    return w_p * pitcher_filled + (1 - w_p) * team_blend


def _platoon_blend(base_rate, rate_same, n_same, rate_diff, n_diff, same_hand, k_blend):
    """success 피처(exp04)에서 검증된 same/diff-hand 선택 + credibility-blend 로직.
    platoon_n이 작을수록(=콜드스타트) base_rate 쪽으로 당겨진다 (w = n/(n+k_blend))."""
    platoon_rate = np.where(same_hand, rate_same, rate_diff)
    platoon_n = np.where(same_hand, n_same, n_diff)
    filled = np.where(np.isfinite(platoon_rate), platoon_rate, base_rate)
    w = np.where(np.isfinite(platoon_n), platoon_n / (platoon_n + k_blend), 0.0)
    blend = np.where(np.isfinite(base_rate), w * filled + (1 - w) * base_rate, filled)
    return platoon_rate, platoon_n, blend


def apply_static_features(df: pd.DataFrame, lookup: pd.DataFrame, k_blend: float = K_BLEND,
                           components=(), platoon_components=(),
                           with_n_exposed: bool = False,
                           with_season_trend: bool = False,
                           with_team_hier: bool = False,
                           team_lookup: pd.DataFrame = None,
                           league_rate: float = None,
                           with_ewma: bool = False,
                           ewma_lookup: pd.DataFrame = None) -> pd.DataFrame:
    """valid/test 행에 얼어붙은 lookup을 join. 현재 행 자신의 pitcher_hand/batter_hand로
    same/diff 버킷을 선택한다 (이건 test 행 '자기 자신'의 피처이므로 허용됨).

    with_n_exposed=True면 투수의 전체 R전용 frozen 표본 크기(`n_R2`) 자체를 피처로 노출한다.
    지금까지 `asof_success_R_platoon_blend_static`은 `k_blend=30` 고정값으로 크레딧빌리티를
    셰이딩했는데, 표본 크기를 트리에 직접 보여주면 그 신뢰도 가중을 트리가 스스로 배울 수
    있다 — `asof_success_R_platoon_n_static`은 이미 있지만 이건 same/diff 분기 카운트고,
    career 전체 표본 크기(n_R2)는 아직 노출된 적 없다.

    with_season_trend=True면 "가장 최근 얼어붙은 시즌 하나"의 R전용 성공률(credibility-blend,
    좌우매치업과 같은 메커니즘 재사용)과, 그걸 그 이전 전체 career 성공률과 뺀 slope를
    만든다. 지금까지의 TREND_FEATURES(prevN 게임, 공식 컬럼이라 2022 이전 F 오염 섞임)와
    질적으로 다른 축 — 게임 단위 컨디션이 아니라 시즌 단위 다년도 궤적(성장/노화)이고,
    R전용으로 클린하다. EDA_정리.md §3의 리그 전체 base rate 하락이 투수별로도 갈릴 수
    있다는 가설.
    """
    key = df[["pitcher_id", "pitcher_hand", "batter_hand"]].copy()
    same_hand = (key["pitcher_hand"] == key["batter_hand"]).to_numpy()
    j = key.join(lookup, on="pitcher_id")

    base_rate = j["rate_R2"].to_numpy(float)
    platoon_rate, platoon_n, blend = _platoon_blend(
        base_rate, j["rate_same"], j["n_same"], j["rate_diff"], j["n_diff"], same_hand, k_blend)

    out = {
        "asof_success_R2_static": base_rate,
        "asof_success_R_platoon_static": platoon_rate,
        "asof_success_R_platoon_n_static": platoon_n,
        "asof_success_R_platoon_blend_static": blend,
    }
    if with_n_exposed:
        out["asof_success_R2_n_static"] = j["n_R2"].to_numpy(float)
    for comp in components:
        out[_COMPONENT_OUT_COL[comp]] = j[_COMPONENT_RATE_COL[comp]].to_numpy(float)
    for comp in platoon_components:
        base_col = _COMPONENT_RATE_COL[comp]
        base_rate_c = j[base_col].to_numpy(float)
        _, _, blend_c = _platoon_blend(
            base_rate_c, j[f"rate_{comp}_same"], j[f"n_{comp}_same"],
            j[f"rate_{comp}_diff"], j[f"n_{comp}_diff"], same_hand, k_blend)
        out[f"asof_{comp}_R_platoon_blend_static"] = blend_c
    if with_season_trend:
        rate_earlier = j["rate_earlier_career"].to_numpy(float)
        rate_last = j["rate_last_season"].to_numpy(float)
        n_last = j["n_last_season"].to_numpy(float)
        filled_last = np.where(np.isfinite(rate_last), rate_last, rate_earlier)
        w = np.where(np.isfinite(n_last), n_last / (n_last + k_blend), 0.0)
        blend_last = np.where(np.isfinite(rate_earlier), w * filled_last + (1 - w) * rate_earlier, filled_last)
        slope = blend_last - rate_earlier
        out["asof_success_R_last_season_blend_static"] = blend_last
        out["asof_success_R_season_slope_static"] = slope
    if with_team_hier:
        # 팀 lookup은 df 자기 자신의 pitcher_team_id로 조회 -- same/diff-hand 매치업
        # 선택과 동일한 "test 행 자기 자신의 피처" 패턴이라 리크가 아니다.
        team_key = df[["pitcher_team_id"]].join(team_lookup, on="pitcher_team_id")
        team_rate_arr = team_key["rate_team"].to_numpy(float)
        team_n_arr = team_key["n_team"].to_numpy(float)
        pitcher_n_arr = j["n_R2"].to_numpy(float)
        out["asof_success_R_team_blend_static"] = _team_hier_blend(
            base_rate, pitcher_n_arr, team_rate_arr, team_n_arr, league_rate)
    if with_ewma:
        ewma_key = df[["pitcher_id"]].join(ewma_lookup, on="pitcher_id")
        out["asof_success_R_ewma_static"] = ewma_key["rate_ewma"].to_numpy(float)
    return pd.DataFrame(out, index=df.index)


def build_fold_frames(df: pd.DataFrame, train_mask: pd.Series, valid_mask: pd.Series,
                       use_custom: bool, use_trend: bool = False, use_failmode: bool = False,
                       use_ballstrike: bool = False, use_failmode_platoon: bool = False,
                       use_n_exposed: bool = False, use_season_trend: bool = False,
                       use_team_hier: bool = False, use_ewma: bool = False):
    """train_mask/valid_mask(src.folds.split 결과)로부터 (fit_df, valid_df)를 만든다.

    use_custom=True면 CUSTOM_STATIC_FEATURES(+ use_failmode/use_ballstrike/use_failmode_platoon/
    use_n_exposed/use_season_trend로 성분 추가)를 두 프레임에 동일한 이름으로 덧붙인다
    (시즌 단위 frozen snapshot, 배포와 동일 조건). use_trend=True면 TREND_FEATURES(행 단위
    뺄셈, 항상 안전)를 덧붙인다.
    """
    fit_raw = df.loc[train_mask].copy()
    valid_df = df.loc[valid_mask].copy()
    comps = _components(use_failmode, use_ballstrike)
    platoon_comps = _platoon_components(use_failmode_platoon)

    if use_custom:
        fit_df = build_training_features(fit_raw, with_failmode=use_failmode,
                                          with_ballstrike=use_ballstrike,
                                          with_failmode_platoon=use_failmode_platoon,
                                          with_n_exposed=use_n_exposed,
                                          with_season_trend=use_season_trend,
                                          with_team_hier=use_team_hier,
                                          with_ewma=use_ewma)
        lookup = fit_pitcher_lookup(fit_raw, components=comps, platoon_components=platoon_comps,
                                     with_season_trend=use_season_trend)
        team_lookup, league_rate = fit_team_lookup(fit_raw) if use_team_hier else (None, None)
        ewma_lookup = fit_pitcher_ewma(fit_raw) if use_ewma else None
        static_feats = apply_static_features(valid_df, lookup, components=comps,
                                              platoon_components=platoon_comps,
                                              with_n_exposed=use_n_exposed,
                                              with_season_trend=use_season_trend,
                                              with_team_hier=use_team_hier,
                                              team_lookup=team_lookup, league_rate=league_rate,
                                              with_ewma=use_ewma, ewma_lookup=ewma_lookup)
        valid_df = pd.concat([valid_df.reset_index(drop=True), static_feats.reset_index(drop=True)], axis=1)
    else:
        fit_df = fit_raw

    if use_trend:
        fit_df = add_trend_features(fit_df)
        valid_df = add_trend_features(valid_df)

    return fit_df, valid_df


def feature_list(use_custom: bool, use_trend: bool = False, use_failmode: bool = False,
                  use_ballstrike: bool = False, use_failmode_platoon: bool = False,
                  use_n_exposed: bool = False, use_season_trend: bool = False,
                  use_team_hier: bool = False, use_ewma: bool = False):
    cols = list(OFFICIAL_FEATURES)
    if use_custom:
        cols += CUSTOM_STATIC_FEATURES
        if use_failmode:
            cols += FAILMODE_FEATURES
        if use_ballstrike:
            cols += BALLSTRIKE_FEATURES
        if use_failmode_platoon:
            cols += FAILMODE_PLATOON_FEATURES
        if use_n_exposed:
            cols += N_EXPOSED_FEATURES
        if use_season_trend:
            cols += SEASON_TREND_FEATURES
        if use_team_hier:
            cols += TEAM_HIER_FEATURES
        if use_ewma:
            cols += EWMA_FEATURES
    if use_trend:
        cols += TREND_FEATURES
    return cols


def categorical_list(use_custom: bool):
    return CATEGORICAL_FEATURES


# ── 사후 로짓 보정 (post-hoc logit correction, exp46/model.md) ───────────────
#
# 모델 **입력이 아니라** 앙상블 예측이 끝난 로짓에 더하는 항 -- 재학습 불필요, train
# fit / valid·test lookup 구조라 리크 구조가 안 생긴다 (wonwoo 브랜치 아이디어 재구현,
# 우리 재구현 과정에서 batter_hand 1/2 인덱싱 버그를 발견해 수정했음 — 아래 _ctx_code_hand
# 참조). exp46: fold A/B/C 전부 부호 일치, hand+count4 조합이 exp47 seed 노이즈 게이트도
# 통과. exp48(부트스트랩 CI)로 최종 확인 중.
#
#     δ(pitcher, 상황코드) = n/(n+k) · (그 셀 성공률 − 그 투수 전체 성공률)   [train에서만 fit]
#     logit(p_corrected) = logit(p_ensemble) + Σ_axis β·δ                    [valid/test는 조회만]

CTX_K = 200.0


def _ctx_code_hand(df):
    """타자 손잡이. 실제 값은 {1,2}(CLAUDE.md) -- 0-index로 매핑해야 pid*m+code 키가
    인접 pitcher_id와 충돌하지 않는다."""
    return df["batter_hand"].to_numpy(dtype=np.int64) - 1, 2


def _ctx_code_count4(df):
    """0=평상시 1=3볼 2=2스트라이크 3=풀카운트급."""
    b = df["balls_before"].to_numpy(dtype=np.int64)
    s = df["strikes_before"].to_numpy(dtype=np.int64)
    return (b >= 3).astype(np.int64) + 2 * (s >= 2).astype(np.int64), 4


CTX_CODERS = {"hand": _ctx_code_hand, "count4": _ctx_code_count4}


def fit_ctx(fit_df: pd.DataFrame, axis_names, target: str = TARGET, k: float = CTX_K):
    """train(fit_df)에서만 호출. 사후 보정 축(들)의 δ를 fit해 딕셔너리로 반환."""
    y = fit_df[target].to_numpy(dtype=np.float64)
    pid = fit_df["pitcher_id"].to_numpy(dtype=np.int64)
    upid, invp = np.unique(pid, return_inverse=True)
    n_p = np.bincount(invp).astype(np.float64)
    rate_p = np.bincount(invp, weights=y) / n_p

    out = {}
    for name in axis_names:
        code, m = CTX_CODERS[name](fit_df)
        key = pid * m + code
        u, inv = np.unique(key, return_inverse=True)
        n = np.bincount(inv).astype(np.float64)
        t = np.bincount(inv, weights=y)
        rp = rate_p[np.searchsorted(upid, u // m)]
        delta = (n / (n + k)) * (t / n - rp)
        out[name] = {"m": m, "keys": u, "delta": delta}
    return out


def ctx_lookup(df: pd.DataFrame, axis_name: str, art: dict):
    """valid/test 행에서 조회만 -- df 자기 자신의 pitcher_id + 상황코드로 searchsorted."""
    a = art[axis_name]
    code, m = CTX_CODERS[axis_name](df)
    kv = df["pitcher_id"].to_numpy(dtype=np.int64) * m + code
    keys = a["keys"]
    j = np.clip(np.searchsorted(keys, kv), 0, len(keys) - 1)
    return np.where(keys[j] == kv, a["delta"][j], 0.0)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
