"""2단계 — Trackman **버킷별 재집계**의 형질성 (§7.205 후속)

가설
    §7.116a E1 은 시즌 집계 산포가 **상태**라고 판정했다
    (`rels_std` 보정후 0.225 · `relpoint_disp` 0.326, 천장 0.995).
    그런데 시즌 집계 산포는 **상황 혼합값**이다 — 카운트마다 던지는 것이 다르고,
    투수의 카운트 분포는 해마다 흔들린다.

    → **낮은 상관이 성분의 불안정 때문인가, 혼합비의 변동 때문인가?**
      혼합비 때문이라면 버킷별로 집계했을 때 형질성이 올라가야 한다.

    이것이 "상황에 따라 게이트" 의 유일하게 남은 강한 판(§7.205 2단계)이다.
    §7.205 가 닫은 것은 시즌 집계값 위의 상호작용 게이트(형태 B)뿐이다.

⚠️ 재료의 한계 — 사전에 박아둔다
    `trackman_history.csv` 에는 `balls_before`·`strikes_before`·`outs_before`·
    `inning`·`pitch_of_pa` 가 있고 **주자 컬럼이 없다.**
    → 세트포지션 축(§7.205 에서 1순위로 꼽았던 기전)은 재집계로 **직접 못 만든다.**
      (§7.156 의 투구 단위 정렬로 메인 테이블 주자 상태를 붙이는 우회로는 있으나
       86% 정렬 신뢰도가 얹힌다 — 이 파일 범위 밖.)
    여기서 재는 것은 **카운트 계열**이다.

추정량은 E1 과 **글자 그대로 같아야 한다** (`tmtrait.py` 재사용)
    인접 시즌 투수 단위 Pearson + split-half(짝/홀) 천장 Spearman–Brown 보정.
    버킷을 쪼개면 셀당 n 이 줄어 **원시 상관이 추정 노이즈로 더 감쇠**한다 —
    보정이 정확히 그걸 되돌리는 장치이므로 비교는 **보정후 값**으로 한다.
    ⚠️ 대조군(pooled)도 같은 실행에서 같은 코드로 다시 낸다. E1 로그의 숫자를
       인용해 비교하면 게이트 하한(N_MIN)·표본이 달라 불공정해진다.

같이 감사하는 것 — Gate A(형태 A) 재료
    `tm_n` 이 테이블에 이미 있다. 조건부 결측 게이트의 게이트 변수가 될 수 있는지
    (분포·시즌 간 안정성·커버리지 손실)를 여기서 같이 찍는다.

실행:  .venv/Scripts/python.exe -u src/tmbucket.py 2>&1 | tee experiments/tmbucket.log
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

N_MIN = 200          # 셀당 최소 투구 수 (양쪽 시즌 모두) — E1 과 동일
N_MIN_HALF = 100     # split-half 각 반쪽 최소
PAIRS = [(2019, 2020), (2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)]

METRICS = ["relh_mean", "relh_std", "rels_mean", "rels_std",
           "ext_mean", "ext_std", "relpoint_disp"]


# ── 버킷 정의 ────────────────────────────────────────────────────
#    trackman_history 에 실재하는 컬럼만 쓴다.
def bk_pooled(df):
    return np.zeros(len(df), dtype=np.int64), 1, "pooled (E1 재현 · 대조군)"


def bk_two_strike(df):
    """0 = 여유 / 1 = 2스트라이크. '존에 넣는가 vs 코너를 노리는가'.
    2칸이라 셀당 n 손실이 가장 작다 — 재집계 가설의 최선 조건."""
    return (df.strikes_before.to_numpy() >= 2).astype(np.int64), 2, "2스트라이크 여부"


def bk_count4(df):
    """0 일반 / 1 3볼 / 2 2스트라이크 / 3 풀카운트급."""
    b = df.balls_before.to_numpy()
    s = df.strikes_before.to_numpy()
    return ((b >= 3).astype(np.int64) + 2 * (s >= 2).astype(np.int64)), 4, "count4"


def bk_ahead(df):
    """투수 우위 여부 (strikes > balls). 압박 방향."""
    return (df.strikes_before.to_numpy()
            > df.balls_before.to_numpy()).astype(np.int64), 2, "투수우위 여부"


def bk_firstpitch(df):
    """초구 여부. 타석 진입 시의 루틴 대 그 뒤의 조정."""
    return (df.pitch_of_pa.to_numpy() == 1).astype(np.int64), 2, "초구 여부"


BUCKETS = [bk_pooled, bk_two_strike, bk_ahead, bk_firstpitch, bk_count4]


def _agg(df, keys):
    """(keys) → 집계 한 벌. tmtrait._agg 와 같은 정의."""
    g = df.groupby(keys, observed=True)
    out = g.agg(
        n=("rel_height", "count"),
        relh_mean=("rel_height", "mean"), relh_std=("rel_height", "std"),
        rels_mean=("rel_side", "mean"), rels_std=("rel_side", "std"),
        ext_mean=("extension", "mean"), ext_std=("extension", "std"),
    )
    out["relpoint_disp"] = np.sqrt(out.relh_std ** 2 + out.rels_std ** 2)
    return out.reset_index()


def ceiling(tm, keys):
    """split-half(짝/홀) 신뢰도 → Spearman–Brown 천장. 시즌별로 재고 표본 가중 평균."""
    d = tm.copy()
    d["half"] = d.groupby(keys, observed=True).cumcount() % 2
    hv = _agg(d, keys + ["half"])
    hv = hv[hv.n >= N_MIN_HALF]
    piv = hv.pivot_table(index=keys, columns="half", values=METRICS, observed=True)
    rel = {}
    for m in METRICS:
        if (m, 0) not in piv.columns or (m, 1) not in piv.columns:
            rel[m] = np.nan
            continue
        rs, ws = [], []
        for _szn, gg in piv.groupby(level="season"):
            a, b = gg[(m, 0)], gg[(m, 1)]
            ok = a.notna() & b.notna()
            if ok.sum() < 30:
                continue
            rs.append(np.corrcoef(a[ok], b[ok])[0, 1])
            ws.append(int(ok.sum()))
        r = float(np.average(rs, weights=ws)) if rs else np.nan
        rel[m] = 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan
    return rel


def trait_table(full, rel, extra_keys):
    """인접 시즌 형질 상관. extra_keys 가 있으면 (pid, *extra) 를 개체로 본다."""
    idx = ["pid"] + extra_keys
    rows = {}
    for m in METRICS:
        pa, pb = [], []
        for t0, t1 in PAIRS:
            a = full[full.season == t0].set_index(idx)[m]
            b = full[full.season == t1].set_index(idx)[m]
            a = a[~a.index.duplicated()]
            b = b[~b.index.duplicated()]
            j = a.index.intersection(b.index)
            if len(j) < 30:
                continue
            aa, bb = a.loc[j], b.loc[j]
            ok = aa.notna() & bb.notna()
            if ok.sum() < 30:
                continue
            pa.append(aa[ok].to_numpy())
            pb.append(bb[ok].to_numpy())
        if not pa:
            rows[m] = (np.nan, np.nan, np.nan, 0)
            continue
        va, vb = np.concatenate(pa), np.concatenate(pb)
        rp = float(np.corrcoef(va, vb)[0, 1])
        c = rel.get(m, np.nan)
        adj = rp / c if np.isfinite(c) and c > 0 else np.nan
        rows[m] = (rp, c, adj, len(va))
    return rows


def main():
    use = ["season", "pitcher_trackman_id", "pitch_type_group",
           "balls_before", "strikes_before", "pitch_of_pa",
           "rel_height", "rel_side", "extension", "rel_speed"]
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv", usecols=use)
    tm = tm.rename(columns={"pitcher_trackman_id": "pid"})
    print(f"trackman {tm.shape}   투수 {tm.pid.nunique()}명")
    print(f"게이트 셀당 n≥{N_MIN}, 반쪽 n≥{N_MIN_HALF}   보정 = raw / split-half 천장\n")

    results = {}
    for fn in BUCKETS:
        code, m, label = fn(tm)
        tm["_bk"] = code
        extra = [] if m == 1 else ["_bk"]
        keys = ["pid", "season"] + extra

        full = _agg(tm, keys)
        full = full[full.n >= N_MIN]
        rel = ceiling(tm, keys)
        rows = trait_table(full, rel, extra)

        # 셀 규모 보고 — 버킷을 쪼개면 게이트를 넘는 셀이 줄어든다
        npc = full.n.median()
        ncell = len(full)
        print("=" * 96)
        print(f"■ {label}   ({m}칸)   게이트 통과 셀 {ncell:,}   셀당 중앙 n {npc:,.0f}")
        print("=" * 96)
        print(f"  {'축':<16}{'raw':>9}{'천장':>9}{'보정후':>9}{'쌍 n':>9}   판정")
        for k in METRICS:
            rp, c, adj, n = rows[k]
            gate = "PASS" if np.isfinite(adj) and adj >= 0.5 else "✗ 즉사"
            print(f"  {k:<16}{rp:>+9.3f}{c:>9.3f}{adj:>9.3f}{n:>9,}   {gate}")
        results[label] = rows
        print()

    # ── 핵심 대조: pooled 대비 버킷이 형질성을 올렸나 ────────────────
    base = results["pooled (E1 재현 · 대조군)"]
    print("=" * 96)
    print("★ 핵심 대조 — 보정후 상관, pooled 대비 (혼합 해제 가설의 직접 검정)")
    print("=" * 96)
    names = [lb for lb in results if lb != "pooled (E1 재현 · 대조군)"]
    print(f"  {'축':<16}{'pooled':>9}" + "".join(f"{n[:11]:>13}" for n in names))
    for k in METRICS:
        line = f"  {k:<16}{base[k][2]:>9.3f}"
        for n in names:
            a = results[n][k][2]
            d = a - base[k][2]
            line += f"{a:>8.3f}{d:>+5.2f}"
        print(line)

    print()
    print("  읽는 법")
    print("   · 산포류(`rels_std`·`relpoint_disp`)의 보정후가 pooled 대비 **크게 오르면**")
    print("     → 낮은 형질성의 원인이 혼합비였다. 버킷별 피처로 학습 arm 을 돌릴 값어치가 있다.")
    print("   · 안 오르거나 내려가면 → 성분 자체가 상태다. 2단계 폐쇄. 학습 arm 불필요.")
    print("   · mean 류는 원래 0.97~0.98 이라 천장 효과로 안 움직인다 — 판정 대상이 아니다.")
    print("   ⚠️ 보정후는 raw/천장 이라 천장이 낮으면 불안정하다. 천장 열을 같이 볼 것.")

    # ── Gate A 재료 — tm_n 감사 ────────────────────────────────────
    print("\n" + "=" * 96)
    print("Gate A(조건부 결측) 재료 — 시즌당 투구 수 `n` 의 분포와 안정성")
    print("=" * 96)
    tm["_bk"] = 0
    ps = _agg(tm, ["pid", "season"])
    q = ps.n.quantile([.1, .25, .5, .75, .9]).round(0)
    print(f"  투수×시즌 셀 {len(ps):,}   n 분위 "
          + "  ".join(f"p{int(k*100)}={int(v):,}" for k, v in q.items()))
    for thr in (100, 200, 500, 1000):
        keep = (ps.n >= thr)
        # 그 임계에서 살아남는 '행 몫' 근사 = 투구 수 기준 커버리지
        cov = ps.n[keep].sum() / ps.n.sum()
        print(f"    n≥{thr:<5} 셀 {keep.mean():>6.1%}   투구 커버리지 {cov:>6.1%}")
    a = ps[ps.season == 2023].set_index("pid").n
    b = ps[ps.season == 2024].set_index("pid").n
    j = a.index.intersection(b.index)
    print(f"  n 의 인접 시즌 상관 (2023→2024, 공통 {len(j)}명) "
          f"{np.corrcoef(a.loc[j], b.loc[j])[0,1]:+.3f}")
    print("   → 게이트 변수 자체가 시즌을 넘어 안정해야 2025 에 같은 게이트가 걸린다.")


if __name__ == "__main__":
    main()
