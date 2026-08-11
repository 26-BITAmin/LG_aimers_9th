"""검증 폴드 정의 — EDA_정리.md §9, CLAUDE.md 검증 프로토콜과 동일.

절대 규칙 (CLAUDE.md):
- 학습에서 game_type=="F" & season<=2022 105,308행은 항상 제외 (튜닝 대상 아님)
- 평가는 R 전용
- fold D는 최종 확인 전용, 재튜닝 금지
"""
import numpy as np
import pandas as pd

FOLDS = {
    "A": dict(train=(2019, 2021), valid=2022),
    "B": dict(train=(2020, 2022), valid=2023),
    "C": dict(train=(2021, 2023), valid=2024),  # 드리프트 대응 주근거 — 튜닝은 여기서
    "D": dict(train=(2019, 2023), valid=2024),  # 최종 확인 전용 — 재튜닝 금지
}

TUNING_FOLDS = ("A", "B", "C")
FINAL_CHECK_FOLD = "D"


def _drop_bad_f(df: pd.DataFrame) -> pd.Series:
    """2022 이전 F는 라벨 생성 과정이 다르다 (EDA_정리.md §5). 학습에서 항상 제외."""
    return ~((df["game_type"] == "F") & (df["season"] <= 2022))


def split(df: pd.DataFrame, fold: str):
    """fold 이름 -> (train_mask, valid_mask). valid_mask는 game_type=='R'로 고정."""
    spec = FOLDS[fold]
    lo, hi = spec["train"]
    va_year = spec["valid"]

    train_mask = df["season"].between(lo, hi) & _drop_bad_f(df)
    valid_mask = (df["season"] == va_year) & (df["game_type"] == "R")
    return train_mask, valid_mask
