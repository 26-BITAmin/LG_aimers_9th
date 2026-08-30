"""Trackman 재구축 2단계 — **피처를 원본에서 다시 만든다. 변형 5종.**

1단계(`tmr1audit.py`)가 확정한 사실
    ✅ §7.116a E1 **독립 재현됨** — 단일시즌 인접 상관 rel_side_std .223 / disp .321 /
       rel_height_std .545 / extension_std .664 는 E1 의 .224/.326/.548/.678 과 일치.
       (내 "누적창 ρ .87~.98" 은 창을 공유해서 나온 자기상관이라 의미 없음.)
    ★★ **진짜 결함은 측정 레짐 단절이다** — `extension` 2021→2022 에 −0.145 (0.95 SD) 계단,
       `induced_vert_break` 27.5 → 22.8 단조 표류. 배포 피처는 이걸 **가로질러 pooled** 한다.
       target 2023 피처는 레짐 3:1 혼합, target 2025 는 3:3 — **같은 투수인데 값이 움직인다.**
    ★ 시즌간 평균 표류 / 시즌내 산포 = **0.632**. pooled std 는 두 가지를 섞고 있다.
    ★ 매핑은 살아 있다 (1단계 B 는 내 실수 — 퓨처스 팀 `MIN_*` 을 최빈팀으로 잡았다).
       여기서 접두사 정규화해 다시 잰다.

만드는 변형 (전부 시즌 경계 준수: 시즌 t 피처 = 시즌 ≤ t−1)
    V0  deploy    현행 재현 — 누적 pooled. 대조군이자 재현 검산용
    V1  within    **시즌내** 산포의 투구수 가중 평균. 시즌간 표류를 뺀다
    V2  drift     **시즌간 평균 표류** 자체를 피처로. 기존 작업이 한 번도 준 적 없는 신호
    V3  regime    시즌별 리그 중심화 후 누적 — 레짐 계단 제거
    V4  recent    직전 1시즌만 — 레짐 정렬 + 최신성

판정 지표 — **자기상관이 아니라 예측 상관**
    corr( feature(≤t−1) , 실제 season-t 산포 )
    이게 운반체가 실제로 하는 일이다. E1 의 단일시즌 인접 상관은 "형질인가" 를 묻고,
    이건 "**배포되는 형태가 다음 시즌을 맞히는가**" 를 묻는다. 둘은 다른 양이다
    (예측 쪽은 여러 시즌 풀링으로 추정 노이즈가 줄어든다).

⚠️ 여기서 아무것도 기각하지 않는다. 3단계(홀드아웃 BSS 2023/2024)가 판정자다.

실행:  .venv/Scripts/python.exe -u src/tmr2build.py 2>&1 | tee experiments/tmr2build.log
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

from config import DATA_DIR, ROOT, TRAIN_CSV

OUT_DIR = ROOT / "experiments" / "tmr"
MAP_CSV = ROOT / "model_s28" / "pitcher_id_map.csv"

# 집계 대상 측정 컬럼 (원본 7개 전부 — §7.9 의 "릴리스포인트 10개" 선택을 물려받지 않는다)
MEAS = {"rel_height": "relh", "rel_side": "rels", "extension": "ext",
        "rel_speed": "velo", "spin_rate": "spin",
        "induced_vert_break": "ivb", "horz_break": "hb"}
SEASONS = list(range(2019, 2025))
TARGETS = list(range(2020, 2026))
MIN_N = 100          # (투수, 시즌) 최소 투구 수 — 이보다 적으면 그 시즌은 집계에서 뺀다


def hr(t):
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


def norm_team(s):
    """퓨처스(2군) 접두사를 벗겨 모구단으로 정규화. `MIN_DOO` → `DOO`."""
    s = str(s)
    for p in ("MIN_", "KBO_"):
        if s.startswith(p):
            return s[len(p):]
    return s.split("_")[0]


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mp = pd.read_csv(MAP_CSV)
    m2t = dict(zip(mp.pitcher_trackman_id, mp.pitcher_id))
    use = ["season", "pitcher_trackman_id", "pitch_type_group", "tagged_pitch_type",
           "pitcher_team"] + list(MEAS)
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv", usecols=use)
    tm["pitcher_id"] = tm.pitcher_trackman_id.map(m2t)
    tm = tm.dropna(subset=["pitcher_id"])
    tm["pitcher_id"] = tm.pitcher_id.astype(np.int64)
    print(f"매핑된 trackman 행 {len(tm):,}   투수 {tm.pitcher_id.nunique()}명  "
          f"({time.time()-t0:.0f}s)")

    # ── B(재검). 매핑 독립 재검증 — 퓨처스 접두사 정규화 후 ────
    hr("B'. 매핑 독립 재검증 (수정) — 퓨처스/군팀 접두사 정규화 후 팀 이력 일치")
    main_df = pd.read_csv(TRAIN_CSV, usecols=["season", "pitcher_id", "pitcher_team_id"])
    mt = (main_df.groupby(["pitcher_id", "season"]).pitcher_team_id
          .agg(lambda s: s.value_counts().index[0]).rename("main_team"))
    tmn = tm.assign(t=tm.pitcher_team.map(norm_team))
    # 1군 경기만 보려면 원 문자열이 MIN_/KBO_ 로 시작하지 않는 행
    first = tmn[~tmn.pitcher_team.astype(str).str.startswith(("MIN_", "KBO_"))]
    tt = (first.groupby(["pitcher_id", "season"]).t
          .agg(lambda s: s.value_counts().index[0]).rename("tm_team"))
    j = pd.concat([mt, tt], axis=1).dropna()
    ct = pd.crosstab(j.main_team, j.tm_team)
    best = ct.idxmax(axis=1)
    hitn = sum(ct.loc[i, best[i]] for i in ct.index)
    print(f"  1군 경기 한정 (투수, 시즌) 쌍 {len(j):,}"
          f"   일치율 **{hitn/len(j):.2%}** ({hitn:,}/{len(j):,})")
    for i in sorted(ct.index):
        r = ct.loc[i]
        print(f"    team_id {int(i):>3} → {best[i]:<8} {r[best[i]]:>5,}/{r.sum():>5,}"
              f" = {r[best[i]]/r.sum():>6.1%}")
    print("  ★ 팀 문자열은 매핑 알고리즘에 주어진 적이 없다 — 독립 확증 신호.")

    # ── (투수, 시즌) 단위 1차 집계 ────────────────────────────
    hr("변형 빌드 — (투수, 시즌) 1차 집계 → 5개 운반 형태")
    by = ["pitcher_id", "season"]
    g = tm.groupby(by, observed=True)
    ps = pd.DataFrame({"n": g.size()})
    for c in MEAS:
        ps[f"{c}_m"] = g[c].mean()
        ps[f"{c}_s"] = g[c].std()
    ps = ps[ps.n >= MIN_N].reset_index()
    print(f"  (투수, 시즌) 셀 {len(ps):,}   n≥{MIN_N}")

    # 레짐 중심화용 — 시즌별 리그 평균/표준편차 (측정 레짐 보정)
    lg = tm.groupby("season")[list(MEAS)].agg(["mean", "std"])
    print("\n  시즌별 리그 평균 (레짐 계단 확인용)")
    for c in MEAS:
        print(f"    {c:<20}" + "  ".join(f"{s}:{lg[(c,'mean')][s]:>8.3f}" for s in SEASONS))

    # 레짐 중심화된 셀 통계 — 평균만 보정 (산포는 스케일 변화가 없어 보정 불필요)
    for c in MEAS:
        ps[f"{c}_mz"] = ps.apply(
            lambda r, c=c: (r[f"{c}_m"] - lg[(c, "mean")][r.season]), axis=1)

    def build_variant(kind):
        """target season → (pitcher_id 인덱스) 피처 DataFrame."""
        rows = []
        for target in TARGETS:
            past = [s for s in SEASONS if s <= target - 1]
            if not past:
                continue
            sub = ps[ps.season.isin(past)]
            if kind == "recent":
                sub = sub[sub.season == max(past)]
            if len(sub) == 0:
                continue
            w = sub.n.to_numpy(dtype=np.float64)
            out = {}
            gg = sub.groupby("pitcher_id")

            if kind in ("within", "recent"):
                # 시즌내 산포의 투구수 가중 평균 (표류 제외)
                for c, tag in MEAS.items():
                    num = sub.assign(_x=sub[f"{c}_s"] * sub.n).groupby("pitcher_id")._x.sum()
                    den = sub.assign(_x=np.where(sub[f"{c}_s"].notna(), sub.n, 0.0)
                                     ).groupby("pitcher_id")._x.sum().replace(0, np.nan)
                    out[f"tm_{tag}_std"] = num / den
                    numm = sub.assign(_x=sub[f"{c}_m"] * sub.n).groupby("pitcher_id")._x.sum()
                    out[f"tm_{tag}_mean"] = numm / den
            elif kind == "drift":
                # 시즌간 평균 표류 — 기존 작업이 한 번도 준 적 없는 신호
                for c, tag in MEAS.items():
                    out[f"tm_{tag}_drift"] = gg[f"{c}_mz"].std()
                    out[f"tm_{tag}_range"] = gg[f"{c}_mz"].max() - gg[f"{c}_mz"].min()
            elif kind == "regime":
                # 레짐 중심화된 평균 위에서 누적 pooled (계단 제거)
                for c, tag in MEAS.items():
                    numm = sub.assign(_x=sub[f"{c}_mz"] * sub.n).groupby("pitcher_id")._x.sum()
                    den = gg.n.sum().replace(0, np.nan)
                    out[f"tm_{tag}_mean"] = numm / den
                    # pooled 분산 = 시즌내 분산 가중평균 + 시즌간 평균의 가중분산
                    wi = sub.assign(_x=(sub[f"{c}_s"] ** 2) * sub.n).groupby("pitcher_id")._x.sum() / den
                    mu = numm / den
                    be = (sub.assign(_x=(sub[f"{c}_mz"] ** 2) * sub.n)
                          .groupby("pitcher_id")._x.sum() / den) - mu ** 2
                    out[f"tm_{tag}_std"] = np.sqrt((wi + be.clip(lower=0)).clip(lower=0))
            elif kind == "deploy":
                # 현행 재현 — 레짐 미보정 누적 pooled
                for c, tag in MEAS.items():
                    den = gg.n.sum().replace(0, np.nan)
                    numm = sub.assign(_x=sub[f"{c}_m"] * sub.n).groupby("pitcher_id")._x.sum()
                    mu = numm / den
                    wi = sub.assign(_x=(sub[f"{c}_s"] ** 2) * sub.n).groupby("pitcher_id")._x.sum() / den
                    be = (sub.assign(_x=(sub[f"{c}_m"] ** 2) * sub.n)
                          .groupby("pitcher_id")._x.sum() / den) - mu ** 2
                    out[f"tm_{tag}_mean"] = mu
                    out[f"tm_{tag}_std"] = np.sqrt((wi + be.clip(lower=0)).clip(lower=0))
            else:
                raise ValueError(kind)

            d = pd.DataFrame(out)
            if f"tm_relh_std" in d.columns and "tm_rels_std" in d.columns:
                d["tm_relpoint_disp"] = np.sqrt(d.tm_relh_std ** 2 + d.tm_rels_std ** 2)
            d["tm_n"] = gg.n.sum()
            d["season"] = target
            rows.append(d.reset_index())
        return pd.concat(rows, ignore_index=True)

    variants = {}
    for kind in ("deploy", "within", "drift", "regime", "recent"):
        v = build_variant(kind)
        variants[kind] = v
        p = OUT_DIR / f"tm_{kind}.csv"
        v.to_csv(p, index=False)
        nc = len([c for c in v.columns if c not in ("pitcher_id", "season")])
        print(f"  V:{kind:<8} {v.shape[0]:>6,}행 × {nc:>2}피처  → {p.name}")

    # ── 예측 상관 — 배포 형태가 다음 시즌 실제 산포를 맞히나 ──
    hr("판정 지표: corr( feature(≤t−1) , **실제 season-t 산포** )  — 자기상관이 아니다")
    print("  E1 의 단일시즌 인접 상관은 '형질인가'를 물었다. 이건 '배포되는 형태가")
    print("  다음 시즌을 맞히는가'를 묻는다 — 예측 쪽은 풀링으로 추정 노이즈가 줄어든다.\n")
    truth = ps.set_index(["pitcher_id", "season"])

    keys = ["tm_relh_std", "tm_rels_std", "tm_relpoint_disp", "tm_ext_std",
            "tm_velo_std", "tm_spin_std", "tm_ivb_std", "tm_hb_std"]
    tmap = {"tm_relh_std": "rel_height_s", "tm_rels_std": "rel_side_s",
            "tm_ext_std": "extension_s", "tm_velo_std": "rel_speed_s",
            "tm_spin_std": "spin_rate_s", "tm_ivb_std": "induced_vert_break_s",
            "tm_hb_std": "horz_break_s"}
    print(f"  {'컬럼':<20}" + "".join(f"{k:>10}" for k in
                                     ("deploy", "within", "regime", "recent")) + "   쌍수")
    for kc in keys:
        line, npair = [], 0
        for kind in ("deploy", "within", "regime", "recent"):
            v = variants[kind]
            if kc not in v.columns:
                line.append(np.nan)
                continue
            if kc == "tm_relpoint_disp":
                tv = np.sqrt(truth["rel_height_s"] ** 2 + truth["rel_side_s"] ** 2)
            else:
                tv = truth[tmap[kc]]
            m = v[["pitcher_id", "season", kc]].dropna()
            act = tv.rename("act").reset_index()
            j2 = m.merge(act, on=["pitcher_id", "season"]).dropna()
            npair = len(j2)
            line.append(j2[kc].corr(j2["act"]) if len(j2) >= 30 else np.nan)
        print(f"  {kc:<20}" + "".join(f"{x:>10.3f}" for x in line) + f"   {npair:>6,}")

    print("\n  drift 계열은 예측 대상(실제 산포)이 다르므로 위 표에 없다 — 3단계 BSS 로 직행.")
    print("\n⚠️ 여기서 아무것도 기각하지 않는다. 판정자는 3단계 홀드아웃 BSS(2023·2024)다.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
