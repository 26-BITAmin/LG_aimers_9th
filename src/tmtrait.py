"""E1 — Trackman 시즌 집계는 형질(trait)인가 (§7.116 사전 등록).

왜 재나
    Trackman 파생값은 2025 행에 **2024에 얼린 투수별 상수**로만 전달된다 (2025
    trackman 없음). §7.52 P2가 세운 관문: **얼린 상수는 형질일 때만 운반한다** —
    작년 값이 올해 값을 예측하지 못하면 상수는 아무것도 나르지 못한다.

    §7.52는 이 관문을 믹스 변동성(χ²/df, 차분량)에 물었고 corr −0.04로 죽었다.
    산포류(std)·수준류(mean)에는 **한 번도 묻지 않았다.** EDA(2026-08-23)가 mean류
    0.97~0.98을 실측했지만 std류는 미측정 — §7.116 E2(C)의 tm_relh_std 단독 스윕이
    의미가 있으려면 이 관문부터 통과해야 한다.

무엇을 재나 (매핑·모델·타깃 전부 불필요 — trackman 원본만)
    투수×시즌 집계(양쪽 시즌 n≥200)의 인접 시즌 투수 단위 상관.
    ⚠️ 원시 상관은 **추정 노이즈로 감쇠**된다 — std 추정치 자체가 노이즈라서
    진짜 형질성이 과소평가된다. split-half(짝/홀 투구) 신뢰도로 천장을 재고
    (Spearman–Brown), corrected = raw / 천장 으로 보정한다 (§7.59 mixdiag와 같은 기계).

판정 (사전 등록)
    corrected < 0.5  → 그 축의 동결 상수 운반체 즉사. E2(C) 스윕은 기록용으로만.
    사전 예측: mean류 ≥ 0.95, std류 0.4~0.7.

실행:  .venv/Scripts/python.exe -u src/tmtrait.py 2>&1 | tee experiments/tmtrait.log
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

from config import DATA_DIR

N_MIN = 200          # 시즌당 최소 투구 수 (양쪽 시즌 모두)
N_MIN_HALF = 100     # split-half 각 반쪽 최소 투구 수
PAIRS = [(2019, 2020), (2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)]

BASE = ["relh_mean", "relh_std", "rels_mean", "rels_std", "ext_mean", "ext_std"]
DERIVED = ["relpoint_disp", "abs_rels", "angle", "velo_gap"]
METRICS = BASE + DERIVED


def _agg(df):
    """(pid, season[, half]) → 집계 한 벌. 파생값 포함."""
    keys = [c for c in ("pid", "season", "half") if c in df.columns]
    g = df.groupby(keys, observed=True)
    out = g.agg(
        n=("rel_height", "count"),
        relh_mean=("rel_height", "mean"), relh_std=("rel_height", "std"),
        rels_mean=("rel_side", "mean"), rels_std=("rel_side", "std"),
        ext_mean=("extension", "mean"), ext_std=("extension", "std"),
    )
    out["relpoint_disp"] = np.sqrt(out.relh_std ** 2 + out.rels_std ** 2)
    out["abs_rels"] = out.rels_mean.abs()
    out["angle"] = np.degrees(np.arctan2(out.abs_rels, out.relh_mean))
    vf = (df[df.pitch_type_group == "fastball"]
          .groupby(keys, observed=True).rel_speed.mean())
    vo = (df[df.pitch_type_group == "offspeed"]
          .groupby(keys, observed=True).rel_speed.mean())
    out["velo_gap"] = vf - vo
    return out.reset_index()


def main():
    use = ["season", "pitcher_trackman_id", "pitch_type_group",
           "rel_height", "rel_side", "extension", "rel_speed"]
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv", usecols=use)
    tm = tm.rename(columns={"pitcher_trackman_id": "pid"})
    print(f"trackman {tm.shape}   투수 {tm.pid.nunique()}명   "
          f"게이트 n≥{N_MIN}/시즌, 반쪽 n≥{N_MIN_HALF}\n")

    full = _agg(tm)
    full = full[full.n >= N_MIN]

    # ── split-half 신뢰도 (시즌 안에서 짝/홀 투구) ─────────────────────
    # 시즌을 섞어 상관을 내면 시즌 레벨 분산(리그 드리프트)이 신뢰도를 부풀린다
    # → 시즌별로 재고 표본 가중 평균한다.
    tm["half"] = tm.groupby(["pid", "season"], observed=True).cumcount() % 2
    hv = _agg(tm)
    hv = hv[hv.n >= N_MIN_HALF]
    piv = hv.pivot_table(index=["pid", "season"], columns="half",
                         values=METRICS, observed=True)
    rel = {}
    for m in METRICS:
        rs, ws = [], []
        for szn, gg in piv.groupby(level="season"):
            a, b = gg[(m, 0)], gg[(m, 1)]
            ok = a.notna() & b.notna()
            if ok.sum() < 30:
                continue
            rs.append(np.corrcoef(a[ok], b[ok])[0, 1])
            ws.append(int(ok.sum()))
        r = float(np.average(rs, weights=ws)) if rs else np.nan
        rel[m] = 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan  # Spearman–Brown

    # ── 인접 시즌 형질 상관 ────────────────────────────────────────────
    print(f"{'축':<16}{'쌍별 Pearson (19-20 … 23-24)':<44}{'풀 P':>7}{'풀 S':>7}"
          f"{'신뢰도':>8}{'보정후':>8}{'판정':>7}")
    print("-" * 100)
    verdicts = {}
    for m in METRICS:
        cells, pooled_a, pooled_b = [], [], []
        for t0, t1 in PAIRS:
            a = full[full.season == t0].set_index("pid")[m]
            b = full[full.season == t1].set_index("pid")[m]
            j = a.index.intersection(b.index)
            aa, bb = a.loc[j], b.loc[j]
            ok = aa.notna() & bb.notna()
            if ok.sum() < 30:
                cells.append("    —")
                continue
            r = np.corrcoef(aa[ok], bb[ok])[0, 1]
            cells.append(f"{r:+.3f}")
            pooled_a.append(aa[ok].to_numpy())
            pooled_b.append(bb[ok].to_numpy())
        va = np.concatenate(pooled_a)
        vb = np.concatenate(pooled_b)
        rp = float(np.corrcoef(va, vb)[0, 1])
        rs = float(pd.Series(va).corr(pd.Series(vb), method="spearman"))
        corr_adj = rp / rel[m] if np.isfinite(rel.get(m, np.nan)) and rel[m] > 0 else np.nan
        gate = "PASS" if np.isfinite(corr_adj) and corr_adj >= 0.5 else "✗ 즉사"
        verdicts[m] = (rp, rel[m], corr_adj, gate)
        print(f"{m:<16}{' '.join(cells):<44}{rp:>+7.3f}{rs:>+7.3f}"
              f"{rel[m]:>8.3f}{corr_adj:>8.3f}{gate:>7}")

    print("\n" + "=" * 100)
    print("판정 (사전 등록 §7.116 E1: 보정후 < 0.5 → 동결 상수 운반체 즉사)")
    print("=" * 100)
    for m in ("relh_std", "rels_std", "relpoint_disp", "ext_std"):
        rp, rl, ca, g = verdicts[m]
        print(f"  {m:<16} raw {rp:+.3f} / 천장 {rl:.3f} / 보정 {ca:.3f}  → {g}")
    print("  (사전 예측: mean류 ≥0.95, std류 0.4~0.7 — §7.52 χ²/df 차분량과 달리 수준량)")


if __name__ == "__main__":
    main()
