"""CSV 로딩 + dtype 최적화. 학습/추론 공용."""
import numpy as np
import pandas as pd

from config import ID_COL, TARGET_COL

_F32 = "float32"
_I16 = "int16"
_I8 = "int8"

# 명시적 dtype — 1.47M행을 float64로 읽으면 580MB, float32면 절반.
DTYPES = {
    "season": _I16, "game_month": _I8, "game_dayofweek": _I8, "inning": _I8,
    "top_bottom": "category", "game_type": "category", "base_state": "category",
    "balls_before": _I8, "strikes_before": _I8, "outs_before": _I8,
    "run_top_before": _I8, "run_bot_before": _I8, "run_total_before": _I16,
    "score_diff_home": _I8, "score_diff_pitcher_team": _I8,
    "runner_on_1b": _I8, "runner_on_2b": _I8, "runner_on_3b": _I8,
    "num_runners_on": _I8,
    "home_win_expectancy": _F32, "away_win_expectancy": _F32, "li": _F32,
    "pitcher_id": "int32", "batter_id": "int32",
    "pitcher_hand": _I8, "batter_hand": _I8,
    "pitcher_team_id": _I16, "batter_team_id": _I16,
    "asof_pitcher_n": "int32", "asof_batter_n": "int32", "asof_pitcher_pitchmix_n": "int32",
}
for _c in (
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
):
    DTYPES[_c] = _F32


def load_csv(path, seasons=None):
    """train/test 공통 로더. seasons를 주면 해당 시즌만 남긴다."""
    df = pd.read_csv(path, dtype=DTYPES, encoding="utf-8-sig")
    if seasons is not None:
        df = df[df["season"].isin(seasons)].reset_index(drop=True)
    return df


def recency_weight(season, half_life, ref_season=2024):
    """최근 시즌일수록 큰 가중치. 2019~2022 game_type='F' 정의가 달라진 것처럼
    오래된 시즌은 분포 자체가 다르다 — 버리지는 않되 발언권을 줄인다."""
    age = ref_season - np.asarray(season, dtype=np.float64)
    return np.power(0.5, age / half_life)


def split_xy(df):
    y = df[TARGET_COL].to_numpy(dtype=np.float32)
    return y


__all__ = ["load_csv", "recency_weight", "split_xy", "DTYPES", "ID_COL", "TARGET_COL"]
