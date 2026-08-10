"""Trackman 로그 → 투수별 요약 피처 테이블.

`src/trackman_map.py`가 만든 `pitcher_id ↔ pitcher_trackman_id` 매핑을 써서
투수별 투구 특성을 집계한다.

⚠️ 시즌 경계 (CLAUDE.md 리키지 규칙):
    시즌 t 행의 피처는 **시즌 ≤ t-1 로그로만** 만든다.
    2019 행은 참고할 과거가 없으므로 전부 결측이다.
    2025(test)는 2019~2024 전부를 쓴다 — 2025 Trackman은 애초에 제공되지 않는다.

핵심 가설 (MODEL.md §4 Tier 3):
    **릴리스포인트가 일관된 투수가 제구가 좋다.** `rel_height` / `rel_side`의
    산포가 가장 인과적으로 그럴듯한 제구력 지표다. 구종 혼합 효과를 제거하기 위해
    패스트볼 한정 산포도 따로 만든다.

산출: submit/model/trackman_pitcher.csv  — (pitcher_id, season) → 피처
실행: .venv/Scripts/python.exe -u src/trackman_features.py
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

# (컬럼, 접두사) — 평균과 표준편차를 모두 만든다
NUM_COLS = {
    "rel_height": "relh", "rel_side": "rels", "extension": "ext",
    "rel_speed": "velo", "spin_rate": "spin",
    "induced_vert_break": "ivb", "horz_break": "hb",
}
TARGET_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]


def _moments(df, cols, by):
    """합·제곱합·개수 → 나중에 누적해서 평균/표준편차를 만든다."""
    out = {}
    g = df.groupby(by, observed=True)
    out["n"] = g.size()
    for c in cols:
        v = df[c]
        out[f"{c}__s"] = g[c].sum(min_count=1)
        out[f"{c}__ss"] = df.assign(_x=v * v).groupby(by, observed=True)["_x"].sum(min_count=1)
        out[f"{c}__c"] = g[c].count()
    return pd.DataFrame(out)


def _finish(cum, cols, prefix=""):
    """누적 모멘트 → 평균/표준편차."""
    out = pd.DataFrame(index=cum.index)
    out[f"{prefix}n"] = cum["n"]
    for c, tag in cols.items():
        n = cum[f"{c}__c"].replace(0, np.nan)
        mean = cum[f"{c}__s"] / n
        var = cum[f"{c}__ss"] / n - mean ** 2
        out[f"{prefix}{tag}_mean"] = mean
        out[f"{prefix}{tag}_std"] = np.sqrt(var.clip(lower=0))
    return out


# 릴리스포인트 산포 계열 — 채택된 부분집합 (fold2 5시드: 742.3 → 753.3, +10.9, 3.2σ).
# 익스텐션(tm_ext_*)을 더해봤지만 752.1로 오히려 미세하게 내려갔다(노이즈 내) → 제외.
RELEASE_COLS = [
    "tm_relh_mean", "tm_relh_std", "tm_rels_mean", "tm_rels_std", "tm_relpoint_disp",
    "tmfb_relh_mean", "tmfb_relh_std", "tmfb_rels_mean", "tmfb_rels_std",
    "tmfb_relpoint_disp",
]
EXT_COLS = ["tm_ext_mean", "tm_ext_std", "tmfb_n", "tm_n"]


def load_table(mode="release", path=None):
    """학습·추론이 공유하는 Trackman 테이블 로더. mode에 따라 컬럼을 고른다.

    none        Trackman 미사용
    release     릴리스포인트 산포 계열 10개  ← 채택
    release_ext release + 익스텐션 (비교용)
    all         28개 전부 (비교용)
    """
    if mode == "none":
        return None
    df = pd.read_csv(path or (MODEL_DIR / "trackman_pitcher.csv"))
    if mode == "all":
        return df
    cols = RELEASE_COLS + (EXT_COLS if mode == "release_ext" else [])
    if mode not in ("release", "release_ext"):
        raise ValueError(mode)
    return df[["pitcher_id", "season"] + [c for c in cols if c in df.columns]]


def build():
    t0 = time.time()
    mp = pd.read_csv(MODEL_DIR / "pitcher_id_map.csv")
    m2t = dict(zip(mp.pitcher_trackman_id, mp.pitcher_id))
    print(f"매핑된 투수 {len(m2t):,}명")

    use = ["season", "pitcher_trackman_id", "pitch_type_group", "tagged_pitch_type"] \
        + list(NUM_COLS)
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv", usecols=use)
    tm["pitcher_id"] = tm.pitcher_trackman_id.map(m2t)
    tm = tm.dropna(subset=["pitcher_id"])
    tm["pitcher_id"] = tm.pitcher_id.astype(np.int32)
    print(f"매핑되는 trackman 행 {len(tm):,} / 전체 ({time.time()-t0:.0f}s)")

    by = ["pitcher_id", "season"]
    mom_all = _moments(tm, NUM_COLS, by)
    fb = tm[tm.pitch_type_group == "fastball"]
    mom_fb = _moments(fb, ["rel_height", "rel_side"], by)

    # 구종 구성 · 레퍼토리 크기
    mix = (tm.groupby(by + ["pitch_type_group"], observed=True).size()
             .unstack("pitch_type_group", fill_value=0))
    rep = tm.groupby(by, observed=True).tagged_pitch_type.nunique().rename("n_pitchtypes")
    velo_fb = fb.groupby(by, observed=True).rel_speed.mean().rename("velo_fb")
    velo_off = (tm[tm.pitch_type_group == "offspeed"]
                .groupby(by, observed=True).rel_speed.mean().rename("velo_off"))

    # ── 시즌 경계: 시즌 t 피처 = 시즌 <= t-1 누적 ──────────
    rows = []
    for target in TARGET_SEASONS:
        past = [s for s in range(2019, 2025) if s <= target - 1]
        if not past:
            continue
        sel = lambda d: d[d.index.get_level_values("season").isin(past)]

        cum = sel(mom_all).groupby("pitcher_id").sum(min_count=1)
        feat = _finish(cum, NUM_COLS, "tm_")

        cumfb = sel(mom_fb).groupby("pitcher_id").sum(min_count=1)
        featfb = _finish(cumfb, {"rel_height": "relh", "rel_side": "rels"}, "tmfb_")
        feat = feat.join(featfb, how="left")

        mx = sel(mix).groupby("pitcher_id").sum()
        tot = mx.sum(axis=1).replace(0, np.nan)
        for c in ("fastball", "breaking", "offspeed", "other"):
            feat[f"tm_share_{c}"] = (mx[c] / tot) if c in mx else np.nan
        feat["tm_n_pitchtypes"] = sel(rep).groupby("pitcher_id").max()
        vf = sel(velo_fb.to_frame()).groupby("pitcher_id").velo_fb.mean()
        vo = sel(velo_off.to_frame()).groupby("pitcher_id").velo_off.mean()
        feat["tm_velo_gap"] = vf - vo

        # 핵심 가설: 릴리스포인트 2D 산포
        feat["tm_relpoint_disp"] = np.sqrt(feat["tm_relh_std"] ** 2 + feat["tm_rels_std"] ** 2)
        feat["tmfb_relpoint_disp"] = np.sqrt(feat["tmfb_relh_std"] ** 2
                                             + feat["tmfb_rels_std"] ** 2)
        feat["season"] = target
        rows.append(feat.reset_index())

    out = pd.concat(rows, ignore_index=True)
    out = out[["pitcher_id", "season"] + [c for c in out.columns
                                          if c not in ("pitcher_id", "season")]]
    path = MODEL_DIR / "trackman_pitcher.csv"
    out.to_csv(path, index=False)

    print(f"\n테이블 {out.shape}  → {path}")
    print(f"시즌별 행 수:\n{out.groupby('season').size().to_string()}")
    print(f"\n피처 {len(out.columns)-2}개: {[c for c in out.columns[2:]]}")
    print(f"\n결측률(2025):")
    x = out[out.season == 2025]
    print((x.isna().mean() * 100).round(1).to_string())
    print(f"\n릴리스포인트 산포 분포 (2025):")
    print(x[["tm_relpoint_disp", "tmfb_relpoint_disp", "tm_relh_std", "tm_rels_std"]]
          .describe().round(4).to_string())
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    build()
