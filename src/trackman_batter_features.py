"""Trackman 로그 → **타자별** 요약 피처 테이블.

`src/batter_map.py`가 만든 `batter_id ↔ batter_trackman_id` 매핑(§7.50)을 써서
**그 타자가 실제로 상대한 공**을 집계한다. 투수 쪽(`trackman_features.py`)과
대칭이지만 집계 주체가 반대다.

왜 이 축인가 — 제공 컬럼에 구멍이 있다
    투수 쪽:  asof_pitcher_{fastball,breaking,offspeed}_rate  (구종 믹스 있음)
    타자 쪽:  asof_batter_{n,success_rate,middle_rate}         (구종 믹스 **없음**)

    타깃은 투수의 제구다. 타자가 거기 닿는 경로는 **피칭 접근법**이다 — 위험한
    타자에겐 존 가장자리로 던지고, 그러면 실패 3종 중 "크게 벗어남"이 늘고
    "가운데 몰림"이 준다. "이 타자에겐 변화구 비율이 높다"는 그 접근법의 직접
    측정치인데, 제공된 `asof_batter_*` 3개는 모든 카운트·투수·구종을 평균내
    그 축을 뭉갠다.

⚠️ 시즌 경계 (CLAUDE.md 리키지 규칙)
    시즌 t 행의 피처는 **시즌 ≤ t-1 로그로만** 만든다. 2019 행은 전부 결측.
    2025(test)는 2019~2024 전부를 쓴다 — 2025 Trackman은 제공되지 않는다.

⚠️ 레벨과 구조를 섞지 않는다
    구종 믹스는 리그 전체가 연도별로 이동한다(변화구 비율 상승 등). 원시 비율을
    주면 모델이 train과 2025에서 **다른 레벨**을 본다. 그래서 각 피처를
    **같은 과거 창(window)의 리그 평균 대비 편차**로도 함께 낸다 — 그쪽이
    시즌을 넘어 전이될 수 있는 구조 성분이다. 어느 쪽이 사는지는 fold가 답한다.

산출: submit/model/trackman_batter.csv  — (batter_id, season) → 피처
실행: .venv/Scripts/python.exe -u src/trackman_batter_features.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

from config import DATA_DIR, MODEL_DIR
from trackman_features import _finish, _moments   # 모멘트 누적은 검증된 코드를 재사용

# 상대한 공의 물리량. (컬럼, 접두사) — 평균과 표준편차를 모두 만든다.
NUM_COLS = {
    "rel_speed": "velo", "spin_rate": "spin",
    "induced_vert_break": "ivb", "horz_break": "hb",
    "zone_speed": "zvelo",
}
GROUPS = ("fastball", "breaking", "offspeed", "other")
TARGET_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

# 리그 대비 편차를 낼 컬럼 (레벨 성분 제거). 산포(_std)는 리그 이동이 작아 제외한다.
DEV_COLS = [f"tmb_share_{g}" for g in GROUPS] + [
    "tmb_velo_mean", "tmb_spin_mean", "tmb_ivb_mean", "tmb_hb_mean", "tmb_zone_drop",
]


def build():
    t0 = time.time()
    bm = pd.read_csv(MODEL_DIR / "batter_id_map.csv")
    t2m = dict(zip(bm.batter_trackman_id, bm.batter_id))
    print(f"매핑된 타자 {len(t2m):,}명  (게이트: §7.50 votes>=10)")

    use = ["season", "batter_trackman_id", "pitch_type_group", "tagged_pitch_type"] \
        + list(NUM_COLS)
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv", usecols=use)
    n_all = len(tm)
    tm["batter_id"] = tm.batter_trackman_id.map(t2m)
    tm = tm.dropna(subset=["batter_id"])
    tm["batter_id"] = tm.batter_id.astype(np.int32)
    print(f"매핑되는 trackman 행 {len(tm):,} / {n_all:,} ({len(tm)/n_all:.1%})"
          f"   ({time.time()-t0:.0f}s)")

    by = ["batter_id", "season"]
    mom = _moments(tm, NUM_COLS, by)
    mix = (tm.groupby(by + ["pitch_type_group"], observed=True).size()
             .unstack("pitch_type_group", fill_value=0))
    rep = tm.groupby(by, observed=True).tagged_pitch_type.nunique().rename("n_types")

    rows = []
    for target in TARGET_SEASONS:
        past = [s for s in range(2019, 2025) if s <= target - 1]
        if not past:
            continue

        def sel(d):
            return d[d.index.get_level_values("season").isin(past)]

        cum = sel(mom).groupby("batter_id").sum(min_count=1)
        feat = _finish(cum, NUM_COLS, "tmb_")

        mx = sel(mix).groupby("batter_id").sum()
        tot = mx.sum(axis=1).replace(0, np.nan)
        for g in GROUPS:
            feat[f"tmb_share_{g}"] = (mx[g] / tot) if g in mx else np.nan
        feat["tmb_n_types"] = sel(rep).groupby("batter_id").max()

        # 릴리스 구속 − 홈플레이트 구속. 변화구 성향의 연속 프록시고,
        # 구종 라벨(태깅 규칙이 바뀔 수 있다)에 의존하지 않는다.
        feat["tmb_zone_drop"] = feat["tmb_velo_mean"] - feat["tmb_zvelo_mean"]

        # ── 같은 창의 리그 평균 대비 편차 ─────────────────────
        # 리그 평균은 **투구 수 가중**으로 낸다 (타자별 단순 평균이 아니라
        # 실제 리그 분포). 이 창의 전체 로그로 직접 계산한다.
        win = tm[tm.season.isin(past)]
        lg = {}
        for c in NUM_COLS:
            lg[f"tmb_{NUM_COLS[c]}_mean"] = float(win[c].mean())
        wmx = win.pitch_type_group.value_counts(normalize=True)
        for g in GROUPS:
            lg[f"tmb_share_{g}"] = float(wmx.get(g, 0.0))
        lg["tmb_zone_drop"] = lg["tmb_velo_mean"] - lg["tmb_zvelo_mean"]
        for c in DEV_COLS:
            feat[f"{c}_dev"] = feat[c] - lg[c]

        feat["season"] = target
        rows.append(feat.reset_index())

    out = pd.concat(rows, ignore_index=True)
    out = out[["batter_id", "season"]
              + [c for c in out.columns if c not in ("batter_id", "season")]]
    path = MODEL_DIR / "trackman_batter.csv"
    out.to_csv(path, index=False)

    print(f"\n테이블 {out.shape}  → {path}")
    print(f"시즌별 행 수:\n{out.groupby('season').size().to_string()}")
    print(f"\n피처 {len(out.columns)-2}개:")
    print("  " + ", ".join(c for c in out.columns[2:]))

    x = out[out.season == 2025]
    print(f"\n결측률(season=2025, {len(x)}행):")
    miss = (x.isna().mean() * 100).round(1)
    print("  " + ", ".join(f"{k} {v}%" for k, v in miss.items() if v > 0) or "  없음")

    print(f"\n변별력 확인 — 2025 분포 (편차 컬럼은 0 근처에 모이면 신호가 없는 것):")
    show = [c for c in ("tmb_n", "tmb_share_fastball", "tmb_share_breaking",
                        "tmb_share_fastball_dev", "tmb_share_breaking_dev",
                        "tmb_velo_mean_dev", "tmb_zone_drop_dev") if c in x.columns]
    print(x[show].describe().round(4).to_string())
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    build()
