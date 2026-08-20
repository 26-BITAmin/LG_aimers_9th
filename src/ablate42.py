"""§7.51 — Trackman **타자** 피처 (§7.50 매핑 위에서).

가설
    타깃은 투수의 제구지만, **피칭 접근법**은 타자에 의존한다. 위험한 타자에겐
    존 가장자리로 던지고, 그러면 실패 3종의 구성이 달라진다. 제공 컬럼에는
    투수 구종 믹스(`asof_pitcher_*_rate`)만 있고 **타자 쪽 대응물이 없다** —
    `asof_batter_*` 3개는 모든 카운트·투수·구종을 평균내 그 축을 뭉갠다.

변형
    tmb_all    26개 전부
    tmb_dev    리그 대비 편차 + tmb_n  (레벨 제거, 구조 성분만)
    tmb_mix    구종 믹스 4개 + tmb_n   (기전에 가장 가까운 최소 집합)
    tmb_shuf   ★ **대조군** — 같은 테이블을 batter_id에 무작위 재배정

왜 대조군이 필요한가
    26개 컬럼을 더하는 것 자체가 (분할 후보 증가 · feature_fraction 상호작용으로)
    점수를 움직인다. `tmb_shuf`는 컬럼 수·주변분포·결측 패턴이 전부 같고
    **타자와의 연결만 끊긴** 상태다. 실제 이득이 이걸 넘지 못하면 정보가 아니라
    정규화를 재고 있는 것이다. (§7.41에서 `rand` 대조군이 같은 일을 했다.)

실행:  .venv/Scripts/python.exe -u src/ablate42.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

from config import MODEL_DIR, SEED, TRAIN_CSV
from data import load_csv
from evaluate import compare

SHUF_SEED = SEED


def load_tmb():
    return pd.read_csv(MODEL_DIR / "trackman_batter.csv")


def subset(df, cols):
    return df[["batter_id", "season"] + [c for c in cols if c in df.columns]]


def shuffled(df, seed=SHUF_SEED):
    """대조군 — 시즌 안에서 batter_id를 무작위 재배정한다.

    시즌 **안에서** 섞는 이유: 시즌별 행 수와 커버리지(어떤 타자가 그 시즌
    테이블에 있는가)를 그대로 유지해야 컬럼 수·결측 패턴이 진짜와 같아진다.
    피처 값 자체의 주변분포도 보존된다. 끊기는 것은 타자↔피처 연결 하나뿐이다.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    for s, idx in df.groupby("season").indices.items():
        ids = df.batter_id.to_numpy()[idx].copy()
        rng.shuffle(ids)
        out.iloc[idx, out.columns.get_loc("batter_id")] = ids
    return out


def main():
    tmb = load_tmb()
    dev_cols = ["tmb_n"] + [c for c in tmb.columns if c.endswith("_dev")]
    mix_cols = ["tmb_n"] + [c for c in tmb.columns if c.startswith("tmb_share_")
                            and not c.endswith("_dev")]
    print(f"타자 테이블 {tmb.shape}")
    print(f"  all {len(tmb.columns)-2}개 / dev {len(dev_cols)}개 / mix {len(mix_cols)}개")
    print(f"  dev: {dev_cols}")
    print(f"  mix: {mix_cols}\n")

    df = load_csv(TRAIN_CSV)
    compare(df, {
        "base":     dict(),
        "tmb_all":  dict(tmb_table=tmb),
        "tmb_dev":  dict(tmb_table=subset(tmb, dev_cols)),
        "tmb_mix":  dict(tmb_table=subset(tmb, mix_cols)),
        "tmb_shuf": dict(tmb_table=shuffled(tmb)),      # ★ 대조군
    })


if __name__ == "__main__":
    main()
