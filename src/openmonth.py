"""개막 달 효과 — **월 단위** 전제 검정.

관찰(사용자): 새 시즌이 시작하는 달에 구종 비율이 들썩이는 투수들이 있다.

§7.52는 이 관찰을 **시즌 전체 vs 시즌 전체**로 잘못 옮겨 쟀다. 여기서 제대로 잰다.
단위가 달라지면 배포 조건도 달라진다 —

    시즌 간 비교  →  Trackman(2024 종료)에 갇힌다. 형질이 아니면 2025에 못 간다 (§7.52 P2)
    개막 달       →  `game_month`로 표현된다. **2025 test에 그대로 있다**

⚠️ 선행 결과와의 관계
    §7.47이 월 주기를 재고 기각했다. 그건 **타깃**(제구 성공)의 월 주기다
    (셔플 대조 −0.131 vs −0.159). 여기서 재는 것은 **믹스**의 개막 효과이고,
    M4는 "개막 달에 이력의 전이가 나쁜가"라는 또 다른 양이다. 겹치지 않는다.

⚠️ 해상도 제약 — 측정과 배포가 다르다
    월별 믹스는 Trackman으로만 만들 수 있다(main에는 구종이 없다). 2025에는 없다.
    그리고 test에서 월 경계를 잡으려면 다른 test 행을 봐야 하므로 **금지**다.
    → 2025에서 쓸 수 있는 해상도는 `game_month`(행 자체의 값)와
      **시즌 진행분**(asofdelta, 오프셋은 2024 종료 시점) 둘뿐이다.
      M1~M3은 기전 확인용, **M4가 배포 가능한 형태**다.

  M1  개막 달 믹스가 그 시즌 나머지와 다른가.  중간 달 대조군과 비교한다
  M2  방향이 있나 (리그 공통 = 달력 효과) 아니면 제각각인가 (투수별)
  M3  형질인가 — 작년 개막에 들썩인 투수가 올해 개막에도 들썩이나
  M4  ★ 개막 달에 **이력의 전이**가 더 나쁜가.  Trackman이 필요 없다
  M5  ★ M1의 대안 설명 — 개막이 특별한 게 아니라 **그냥 시간이 흐른 것** 아닌가

⚠️ M1의 함정 (M5가 존재하는 이유)
    '그 달 vs 그 시즌 나머지'는 **구조적으로 비대칭**이다. 개막 달의 '나머지'는
    전부 뒤에 있고 6월의 '나머지'는 앞뒤로 갈린다. 믹스가 시즌 내내 완만히
    흘러가기만 해도 **양 끝 달이 자동으로 크게 나온다.**
    M5는 (a) 연속한 두 달끼리, (b) **개월 간격을 맞춰서** 비교해 이걸 가른다.

실행:  .venv/Scripts/python.exe -u src/openmonth.py 2>&1 | tee experiments/openmonth.log
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

from config import DATA_DIR, MODEL_DIR, TARGET_COL, TRAIN_CSV
from mixvol import chi2_over_df

GROUPS = ["fastball", "breaking", "offspeed"]
MIN_M = 60      # 한 달 최소 투구 수
MIN_R = 200     # 비교 대상(나머지 기간) 최소 투구 수


def main():
    # ── Trackman 월별 믹스 ────────────────────────────────────────
    m = pd.read_csv(MODEL_DIR / "pitcher_id_map.csv")
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv",
                     usecols=["pitcher_trackman_id", "season", "game_month",
                              "pitch_type_group"])
    tm = tm.merge(m, on="pitcher_trackman_id", how="inner")
    tm["grp"] = tm["pitch_type_group"].astype(str).str.strip().str.lower()
    tm = tm[tm.grp.isin(GROUPS)]

    vol = tm.groupby(["season", "game_month"]).size()
    print("시즌×월 투구량 (개막 달 판정용)")
    print(vol.unstack(fill_value=0).to_string())
    # 개막 달 = 그 시즌에서 투구가 있는 **가장 이른 달**
    opens = tm.groupby("season")["game_month"].min().to_dict()
    print(f"\n개막 달: {opens}")

    cnt = (tm.groupby(["pitcher_id", "season", "game_month", "grp"]).size()
           .unstack(fill_value=0).reindex(columns=GROUPS, fill_value=0).reset_index())
    cnt["n"] = cnt[GROUPS].sum(axis=1)

    # ── M1. 그 달 vs 그 시즌 나머지 ───────────────────────────────
    tot = cnt.groupby(["pitcher_id", "season"])[GROUPS].sum()
    rows = []
    for _, r in cnt.iterrows():
        key = (r.pitcher_id, r.season)
        rest = tot.loc[key].to_numpy(dtype=np.float64) - r[GROUPS].to_numpy(dtype=np.float64)
        if r.n < MIN_M or rest.sum() < MIN_R:
            continue
        rows.append((r.pitcher_id, r.season, r.game_month, r.n, rest.sum(),
                     chi2_over_df(r[GROUPS].to_numpy(dtype=np.float64), rest),
                     r[GROUPS[0]] / r.n - rest[0] / rest.sum()))
    d = pd.DataFrame(rows, columns=["pitcher_id", "season", "month", "n", "rest",
                                    "chi2df", "d_ff"])
    d["is_open"] = [mo == opens[s] for s, mo in zip(d.season, d.month)]

    print(f"\n{'='*70}\n[M1] 개막 달이 그 시즌 나머지와 다른가   (투수,시즌,월) {len(d):,}\n{'='*70}")
    print("  귀무 = 1.0 (그 달과 나머지의 진짜 믹스가 같다). n으로 정규화됨\n")
    g = d.groupby("month").agg(표본=("chi2df", "size"), 투구중앙=("n", "median"),
                               chi2df중앙=("chi2df", "median"),
                               초과3비율=("chi2df", lambda v: (v > 3).mean()),
                               직구차평균=("d_ff", "mean"))
    g["개막"] = [mo in opens.values() for mo in g.index]
    print(g.to_string())

    op, mid = d[d.is_open], d[~d.is_open]
    print(f"\n  개막 달  χ²/df 중앙 {op.chi2df.median():.2f}   >3 비율 {(op.chi2df>3).mean():.3f}"
          f"   (표본 {len(op):,}, 투구중앙 {op.n.median():.0f})")
    print(f"  그 외    χ²/df 중앙 {mid.chi2df.median():.2f}   >3 비율 {(mid.chi2df>3).mean():.3f}"
          f"   (표본 {len(mid):,}, 투구중앙 {mid.n.median():.0f})")
    # 투구 수를 맞춘 대조 — 개막 달은 등판이 적어 표본이 작다
    lo, hi = op.n.quantile(.25), op.n.quantile(.75)
    mm = mid[(mid.n >= lo) & (mid.n <= hi)]
    print(f"  ▸ 투구 수 맞춘 대조({lo:.0f}~{hi:.0f}구): 그 외 χ²/df 중앙 "
          f"{mm.chi2df.median():.2f}   >3 비율 {(mm.chi2df>3).mean():.3f}  (표본 {len(mm):,})")

    # ── M2. 방향이 리그 공통인가 ──────────────────────────────────
    print(f"\n{'='*70}\n[M2] 방향 — 리그 공통(달력 효과)인가 제각각(투수별)인가\n{'='*70}")
    print(f"  개막 달 직구비중 차 : 평균 {op.d_ff.mean():+.4f}  std {op.d_ff.std():.4f}")
    print(f"  그 외   직구비중 차 : 평균 {mid.d_ff.mean():+.4f}  std {mid.d_ff.std():.4f}")
    se = op.d_ff.std() / np.sqrt(len(op))
    print(f"  개막 평균의 t = {op.d_ff.mean()/se:+.2f}   "
          "(|t|>2면 리그 공통 방향이 있다 = 달력 효과)")
    print("  시즌별 개막 달 직구비중 차 평균:")
    print("    " + "  ".join(f"{s}:{v:+.4f}" for s, v in
                             op.groupby("season").d_ff.mean().items()))

    # ── M3. 형질인가 ──────────────────────────────────────────────
    print(f"\n{'='*70}\n[M3] 형질인가 — 작년 개막에 들썩인 투수가 올해 개막에도\n{'='*70}")
    o = op[["pitcher_id", "season", "chi2df", "d_ff"]].copy()
    o["lc"] = np.log(o.chi2df.clip(lower=.05))
    j = o.merge(o.assign(season=o.season - 1), on=["pitcher_id", "season"],
                suffixes=("", "_next"))
    print(f"  연속 개막 쌍 {len(j):,} / 투수 {j.pitcher_id.nunique():,}")
    if len(j) >= 30:
        print(f"  corr(올해 log χ²/df, 내년) : Pearson {j[['lc','lc_next']].corr().iloc[0,1]:+.3f}"
              f"   Spearman {j[['lc','lc_next']].corr('spearman').iloc[0,1]:+.3f}")
        print(f"  corr(올해 직구차, 내년)    : Pearson "
              f"{j[['d_ff','d_ff_next']].corr().iloc[0,1]:+.3f}")

    # ── M4. ★ 개막 달에 이력의 전이가 더 나쁜가 (Trackman 불필요) ──
    print(f"\n{'='*70}\n[M4] ★ 개막 달에 **이력의 전이**가 더 나쁜가   (main 데이터만)\n{'='*70}")
    df = pd.read_csv(TRAIN_CSV, usecols=["season", "game_month", "pitcher_id", TARGET_COL])
    ss = (df.groupby(["pitcher_id", "season"])[TARGET_COL].agg(["sum", "size"])
          .rename(columns={"sum": "hits", "size": "n"}).reset_index()
          .sort_values(["pitcher_id", "season"]))
    # 시즌 시작 시점의 커리어(= 이전 시즌들 누적). 배포와 같은 구조
    ss["ck"] = ss.groupby("pitcher_id")["hits"].cumsum() - ss["hits"]
    ss["cn"] = ss.groupby("pitcher_id")["n"].cumsum() - ss["n"]
    ss["career"] = ss.ck / ss.cn.replace(0, np.nan)

    mo = (df.groupby(["pitcher_id", "season", "game_month"])[TARGET_COL]
          .agg(["mean", "size"]).rename(columns={"mean": "sr", "size": "n"}).reset_index())
    mo = mo.merge(ss[["pitcher_id", "season", "career", "cn"]], on=["pitcher_id", "season"])
    mo = mo[(mo.n >= MIN_M) & (mo.cn >= 300) & mo.career.notna()].copy()
    mo["z"] = (mo.sr - mo.career) / np.sqrt(mo.career * (1 - mo.career) / mo.n)
    mo["is_open"] = [m2 == opens.get(s, 3) for s, m2 in zip(mo.season, mo.game_month)]

    print("  z = (그 달 실제 성공률 − 시즌시작 커리어) / 우연오차")
    print("  E[z²] = 1 이면 커리어가 완벽. 클수록 이력이 안 넘어간다\n")
    t = mo.groupby("game_month").agg(표본=("z", "size"), 투구중앙=("n", "median"),
                                     전이오차Ez2=("z", lambda v: (v**2).mean()),
                                     편향Ez=("z", "mean"))
    print(t.to_string())
    a, b = mo[mo.is_open].z ** 2, mo[~mo.is_open].z ** 2
    dd = a.mean() - b.mean()
    sed = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    print(f"\n  개막 달 − 그 외 : ΔE[z²] = {dd:+.2f} ± {sed:.2f}   t = {dd/sed:+.2f}")
    print("  → 유의한 양수면 '개막 달에는 투수 이력을 덜 믿어야 한다' = game_month로 배포 가능")

    # 믹스 들썩임과 연결되나 (M1 ↔ M4)
    link = mo.merge(d.rename(columns={"month": "game_month"})[
        ["pitcher_id", "season", "game_month", "chi2df"]],
        on=["pitcher_id", "season", "game_month"], how="inner")
    print(f"\n  ▸ M1↔M4 연결: 믹스가 들썩인 (투수,달)에서 전이도 나쁜가  표본 {len(link):,}")
    if len(link) >= 100:
        lk = link.assign(l=np.log(link.chi2df.clip(lower=.05)), z2=link.z ** 2)
        print(f"      corr(log χ²/df, z²) Spearman "
              f"{lk[['l', 'z2']].corr('spearman').iloc[0, 1]:+.3f}")
        lo3 = lk[lk.chi2df <= lk.chi2df.quantile(1 / 3)]
        hi3 = lk[lk.chi2df >= lk.chi2df.quantile(2 / 3)]
        print(f"      믹스 안정 3분위 E[z²] {lo3.z2.mean():.2f}   "
              f"들썩 3분위 {hi3.z2.mean():.2f}")

    m5(cnt, opens)


def m5(cnt, opens):
    """★ 개막이 특별한가, 아니면 그냥 시간이 흐른 것인가.

    (a) 연속한 두 달 전이 — 트렌드면 시즌 내내 평평하다
    (b) **개월 간격을 맞춘** 비교 — 탐색기에서 시즌 경계가 커 보이는 것은
        그 두 점 사이가 5~6개월 비어 있기 때문일 수 있다. 그림은 등간격으로
        그리지만 시간은 등간격이 아니다.
    """
    # M1과 같은 표본 게이트를 건다 — 안 걸면 저표본 달이 섞여 중앙값이 1 쪽으로 눌린다
    cnt = cnt[cnt.n >= MIN_M]
    c = cnt.sort_values(["pitcher_id", "season", "game_month"])
    nx = c.groupby(["pitcher_id", "season"]).shift(-1)
    p = pd.DataFrame({"m0": c.game_month.to_numpy(), "m1": nx.game_month.to_numpy(),
                      "s": c.season.to_numpy(), "n0": c.n.to_numpy()})
    A, B = c[GROUPS].to_numpy(dtype=np.float64), nx[GROUPS].to_numpy(dtype=np.float64)
    ok = (p.m1 == p.m0 + 1).to_numpy() & np.isfinite(B).all(1)
    p, A, B = p[ok].reset_index(drop=True), A[ok], B[ok]
    p["chi2df"] = [chi2_over_df(A[i], B[i]) for i in range(len(p))]

    print(f"\n{'='*70}\n[M5a] 연속한 두 달 전이 — 개막이 특별한가   쌍 {len(p):,}\n{'='*70}")
    t = p.groupby("m0").agg(표본=("chi2df", "size"), 투구중앙=("n0", "median"),
                            chi2df중앙=("chi2df", "median"))
    t["개막→"] = [mm in opens.values() for mm in t.index]
    print(t.to_string())

    # (b) 개월 간격을 맞춘 비교 — 시즌 안 vs 비시즌을 건너뛴 쌍
    c2 = cnt.copy()
    c2["t"] = c2.season * 12 + c2.game_month
    c2 = c2.sort_values(["pitcher_id", "t"])
    rows = []
    for _, g in c2.groupby("pitcher_id", sort=False):
        M, tt, sv = g[GROUPS].to_numpy(dtype=np.float64), g.t.to_numpy(), g.season.to_numpy()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                gap = tt[j] - tt[i]
                if gap > 8:
                    break
                rows.append((gap, sv[i] != sv[j], chi2_over_df(M[i], M[j])))
    q = pd.DataFrame(rows, columns=["gap", "cross", "chi2df"])
    print(f"\n{'='*70}\n[M5b] ★ 개월 간격을 맞추면 — 시즌 경계가 정말 큰가   쌍 {len(q):,}"
          f"\n{'='*70}")
    print("  탐색기는 등간격으로 그리지만 시즌 경계는 실제로 5~6개월 비어 있다.\n")
    piv = q.groupby(["gap", "cross"]).chi2df.median().unstack()
    cnt_piv = q.groupby(["gap", "cross"]).size().unstack()
    piv.columns = ["시즌내", "경계넘음"][:len(piv.columns)]
    cnt_piv.columns = piv.columns
    print(piv.round(2).to_string())
    print("\n  (표본 수)")
    print(cnt_piv.fillna(0).astype(int).to_string())
    print("\n  → 같은 간격에서 '경계넘음'이 '시즌내'보다 크지 않으면,")
    print("     시즌 경계의 점프는 **비시즌이라서가 아니라 그냥 간격이 넓어서**다.")


if __name__ == "__main__":
    main()
