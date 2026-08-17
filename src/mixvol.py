"""레퍼토리 변동성 — **전제 검정.** 피처를 만들기 전에 네 가지를 먼저 묻는다.

관찰(사용자, notebooks/pitchmix_*): 시즌이 바뀔 때 구종 비율이 크게 들썩이는
투수가 있고 거의 안 움직이는 투수가 있다.

이 관찰이 배포까지 가려면 통과해야 하는 관문이 있다. 순서가 중요하다 —
앞이 깨지면 뒤는 잴 필요가 없다.

  P1  들썩임이 **표집 노이즈를 넘는가.**
      시즌 200구 투수의 믹스는 다항 표집만으로 크게 튄다. 원시 |Δp|는 사실상
      1/√n을 재는 것이다. → χ²/df 로 정규화한다. 귀무(진짜 믹스가 같음) 아래
      χ²/df ≈ 1 이고 **n에 무관**하다. 이게 노이즈 게이트다.
      같이 잰다: 보직 변화(경기당 투구 수)로 설명되는 몫 — 그건 다른 축이다.

  P2  그것이 **형질인가 상태인가.**  ★ 배포 가능성을 가르는 관문
      Trackman은 2024에서 끝난다. 2025 행에는 어떤 변동성 지표든 2024에 얼린
      상수로 들어간다. "이 사람은 원래 자주 바꾼다"(형질)여야 쓸 수 있고,
      "작년에 바꿨다"(상태)면 2025에 못 쓴다.
      → 커리어 전반부 변동성이 후반부 변동성을 예측하는가.

  P3  레퍼토리 변동성이 **제구 변동성**과 관계가 있는가.
      구종을 바꾸는 것과 커맨드가 흔들리는 것은 다른 얘기다. 도메인적으로
      그럴듯한 것과 데이터가 지지하는 것은 별개다 (§7.51).

  P4  ★ 지불 지점. **이력의 전이 실패를 예측하는가.**
      가설은 "변동성이 큰 투수는 제구가 나쁘다"가 아니라
      **"변동성이 큰 투수는 자기 과거가 다음 시즌으로 덜 넘어간다"** 이다.
      = 피처가 아니라 조절변수. 참이면 SMOOTH_K를 투수별로 둘 근거가 된다
      (§7.31의 전역 스윕은 이 축을 구조적으로 볼 수 없었다).

      노이즈 게이트: z = (실제 − 커리어예측) / √(p(1−p)/n).
      "커리어가 맞다" 아래 E[z²] = 1. 1을 넘는 몫이 진짜 개인 이탈이고,
      n 교란은 표준화가 자동으로 지운다.

⚠️ 여기서 t나 비율이 크게 나와도 **채택 근거가 아니다.** 전제가 살아 있다는
   뜻일 뿐이고, 2025 이득은 여전히 잔차 검정과 β 고원이 답한다.

실행:  .venv/Scripts/python.exe -u src/mixvol.py 2>&1 | tee experiments/mixvol.log
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

TM_CSV = DATA_DIR / "trackman_history.csv"
GROUPS = ["fastball", "breaking", "offspeed"]     # asof_* 3분류와 같은 축. other는 뺀다
MIN_N = 100                                       # 시즌 최소 투구 수 (양쪽 시즌 모두)


# ── χ²/df : 표본 수로 정규화된 믹스 변화량 ────────────────────────
def chi2_over_df(c1, c2):
    """두 시즌 카운트 벡터의 동질성 χ²를 자유도로 나눈 값.

    귀무(진짜 믹스가 같다) 아래 기대값 1, **n에 무관**하다.
    원시 L1 거리를 쓰면 저표본 투수가 전부 상위에 올라온다 — 그게 P1의 함정.
    """
    c1 = np.asarray(c1, dtype=np.float64)
    c2 = np.asarray(c2, dtype=np.float64)
    keep = (c1 + c2) > 0
    c1, c2 = c1[keep], c2[keep]
    if len(c1) < 2:
        return np.nan
    n1, n2, tot = c1.sum(), c2.sum(), c1.sum() + c2.sum()
    p = (c1 + c2) / tot
    e1, e2 = n1 * p, n2 * p
    chi2 = ((c1 - e1) ** 2 / e1).sum() + ((c2 - e2) ** 2 / e2).sum()
    return chi2 / (len(c1) - 1)


def main():
    # ── 데이터 ────────────────────────────────────────────────────
    m = pd.read_csv(MODEL_DIR / "pitcher_id_map.csv")
    tm = pd.read_csv(TM_CSV, usecols=["pitcher_trackman_id", "season",
                                      "pitch_type_group", "trackman_game_id"])
    tm = tm.merge(m, on="pitcher_trackman_id", how="inner")
    tm["grp"] = tm["pitch_type_group"].astype(str).str.strip().str.lower()
    print(f"Trackman 매핑행 {len(tm):,} / 투수 {tm.pitcher_id.nunique():,}")
    print(f"  구종군 분포: {tm['grp'].value_counts().to_dict()}")

    tmg = tm[tm.grp.isin(GROUPS)]
    cnt = (tmg.groupby(["pitcher_id", "season", "grp"]).size()
           .unstack(fill_value=0).reindex(columns=GROUPS, fill_value=0))
    cnt["n"] = cnt[GROUPS].sum(axis=1)
    # 보직 대리 지표: 경기당 투구 수 (Trackman에는 game_id가 있다)
    ppg = (tm.groupby(["pitcher_id", "season"])
           .agg(pit=("grp", "size"), g=("trackman_game_id", "nunique")))
    cnt["ppg"] = ppg["pit"] / ppg["g"]
    cnt = cnt.reset_index()

    # ── 연속 시즌 쌍 ──────────────────────────────────────────────
    a = cnt.sort_values(["pitcher_id", "season"])
    b = a.groupby("pitcher_id").shift(-1)
    pair = pd.DataFrame({
        "pitcher_id": a["pitcher_id"].to_numpy(),
        "s0": a["season"].to_numpy(), "s1": b["season"].to_numpy(),
        "n0": a["n"].to_numpy(), "n1": b["n"].to_numpy(),
        "ppg0": a["ppg"].to_numpy(), "ppg1": b["ppg"].to_numpy(),
    })
    for i, gname in enumerate(GROUPS):
        pair[f"c0_{i}"] = a[gname].to_numpy()
        pair[f"c1_{i}"] = b[gname].to_numpy()
    pair = pair[(pair.s1 == pair.s0 + 1) & (pair.n0 >= MIN_N) & (pair.n1 >= MIN_N)].copy()

    C0 = pair[[f"c0_{i}" for i in range(3)]].to_numpy()
    C1 = pair[[f"c1_{i}" for i in range(3)]].to_numpy()
    pair["chi2df"] = [chi2_over_df(C0[i], C1[i]) for i in range(len(pair))]
    pair["l1"] = np.abs(C0 / C0.sum(1, keepdims=True)
                        - C1 / C1.sum(1, keepdims=True)).sum(1)
    pair["nmin"] = np.minimum(pair.n0, pair.n1)
    pair["dppg"] = np.abs(pair.ppg1 - pair.ppg0)

    print(f"\n{'='*66}\n[P1] 들썩임이 표집 노이즈를 넘는가   쌍 {len(pair):,}"
          f" / 투수 {pair.pitcher_id.nunique():,}\n{'='*66}")
    q = [10, 25, 50, 75, 90, 99]
    print("  χ²/df 분위수 (귀무 = 1.0):  " +
          "  ".join(f"p{v}={np.nanquantile(pair.chi2df, v/100):.1f}" for v in q))
    print(f"  χ²/df > 3 비율 : {(pair.chi2df > 3).mean():.3f}"
          f"    > 10 : {(pair.chi2df > 10).mean():.3f}")
    print(f"\n  ▸ 노이즈 게이트 확인 — 지표가 표본 수와 얽혀 있나 (Spearman)")
    print(f"      corr(원시 L1,  n_min) = {pair[['l1','nmin']].corr('spearman').iloc[0,1]:+.3f}"
          "   ← 크면 '등판 적은 투수 찾기'가 된다")
    print(f"      corr(χ²/df,    n_min) = {pair[['chi2df','nmin']].corr('spearman').iloc[0,1]:+.3f}")
    print(f"\n  ▸ 보직 전환으로 설명되는 몫")
    print(f"      corr(χ²/df, |Δ경기당투구|) = {pair[['chi2df','dppg']].corr('spearman').iloc[0,1]:+.3f}")
    stable = pair[pair.dppg < pair.dppg.quantile(.5)]
    print(f"      보직 안정(Δppg 하위 50%)만: χ²/df 중앙 {stable.chi2df.median():.1f}"
          f"  >3 비율 {(stable.chi2df > 3).mean():.3f}")

    # ── P2. 형질인가 상태인가 ─────────────────────────────────────
    print(f"\n{'='*66}\n[P2] ★ 형질인가 상태인가 — 전반부가 후반부를 예측하는가"
          f"\n{'='*66}")
    per = pair.groupby("pitcher_id")
    ok = per.size()[per.size() >= 4].index          # 쌍 4개 이상 = 시즌 5개 이상
    sub = pair[pair.pitcher_id.isin(ok)].sort_values(["pitcher_id", "s0"])
    rows = []
    for pid, g in sub.groupby("pitcher_id"):
        h = len(g) // 2
        rows.append((pid, np.log(g.chi2df.iloc[:h]).mean(),
                     np.log(g.chi2df.iloc[h:]).mean(), g.n0.median()))
    tr = pd.DataFrame(rows, columns=["pitcher_id", "early", "late", "n"]).dropna()
    print(f"  대상 투수 {len(tr)}명 (연속쌍 4개 이상)")
    if len(tr) >= 30:
        print(f"  corr(전반 log χ²/df, 후반) : Pearson "
              f"{tr[['early','late']].corr().iloc[0,1]:+.3f}"
              f"   Spearman {tr[['early','late']].corr('spearman').iloc[0,1]:+.3f}")
        hi = tr.early > tr.early.median()
        print(f"  전반 상위50% 투수의 후반 평균 {tr.late[hi].mean():+.3f}"
              f"  vs 하위50% {tr.late[~hi].mean():+.3f}"
              f"   (log 스케일, 차이가 0이면 형질이 아니다)")

    # ── train 쪽: 시즌별 제구 성공률 ──────────────────────────────
    df = pd.read_csv(TRAIN_CSV, usecols=["season", "pitcher_id", TARGET_COL])
    ss = (df.groupby(["pitcher_id", "season"])[TARGET_COL]
          .agg(["mean", "size"]).rename(columns={"mean": "sr", "size": "np"}).reset_index())

    # ── P3. 제구 변동성과의 관계 ──────────────────────────────────
    print(f"\n{'='*66}\n[P3] 레퍼토리 변동성 ↔ 제구 변동성\n{'='*66}")
    e = pair.merge(ss.rename(columns={"season": "s0", "sr": "sr0", "np": "np0"}),
                   on=["pitcher_id", "s0"], how="inner")
    e = e.merge(ss.rename(columns={"season": "s1", "sr": "sr1", "np": "np1"}),
                on=["pitcher_id", "s1"], how="inner")
    e = e[(e.np0 >= MIN_N) & (e.np1 >= MIN_N)].copy()
    # 제구 변동성도 같은 게이트를 건다 — |Δsr|을 그 이항 SE로 나눈다
    se = np.sqrt(e.sr0 * (1 - e.sr0) / e.np0 + e.sr1 * (1 - e.sr1) / e.np1)
    e["zsr"] = np.abs(e.sr1 - e.sr0) / se
    print(f"  쌍 {len(e):,}")
    print(f"  corr(log χ²/df, |Δ성공률|/SE) : Spearman "
          f"{e[['chi2df','zsr']].corr('spearman').iloc[0,1]:+.3f}")
    ter = pd.qcut(e.chi2df, 3, labels=["안정", "중간", "들썩"])
    print(e.groupby(ter, observed=True).agg(
        쌍=("zsr", "size"), 믹스χ2df중앙=("chi2df", "median"),
        제구z평균=("zsr", "mean"), 제구z제곱평균=("zsr", lambda v: (v ** 2).mean())
    ).to_string())

    # ── P4. ★ 이력의 전이 실패를 예측하는가 ───────────────────────
    print(f"\n{'='*66}\n[P4] ★ 지불 지점 — 커리어 이력이 다음 시즌으로 전이되는가"
          f"\n{'='*66}")
    # 시즌 s1 시작 시점의 커리어 성공률 (s1 이전 전부)
    ss = ss.sort_values(["pitcher_id", "season"])
    ss["ck"] = ss.groupby("pitcher_id").apply(
        lambda g: (g.sr * g["np"]).cumsum() - g.sr * g["np"], include_groups=False).to_numpy()
    ss["cn"] = ss.groupby("pitcher_id")["np"].cumsum() - ss["np"]
    ss["career"] = ss.ck / ss.cn.replace(0, np.nan)

    f = pair.merge(ss.rename(columns={"season": "s1"})[
        ["pitcher_id", "s1", "sr", "np", "career", "cn"]], on=["pitcher_id", "s1"])
    f = f[(f["np"] >= MIN_N) & (f.cn >= 300) & f.career.notna()].copy()
    # z = (실제 − 커리어예측) / √(p(1−p)/n).  "커리어가 맞다" 아래 E[z²]=1
    f["z"] = (f.sr - f.career) / np.sqrt(f.career * (1 - f.career) / f["np"])
    print(f"  대상 (투수,시즌) {len(f):,}   전체 E[z²] = {(f.z**2).mean():.2f}"
          "   (1.0 = 커리어가 완벽, 클수록 개인 이탈이 크다)")
    ter = pd.qcut(f.chi2df, 3, labels=["안정", "중간", "들썩"])
    tab = f.groupby(ter, observed=True).agg(
        표본=("z", "size"), 믹스χ2df중앙=("chi2df", "median"),
        시즌투구중앙=("np", "median"), 커리어n중앙=("cn", "median"),
        전이오차Ez2=("z", lambda v: (v ** 2).mean()),
        평균편의Ez=("z", "mean"))
    print(tab.to_string())
    lo = f[ter == "안정"].z ** 2
    hi = f[ter == "들썩"].z ** 2
    d = hi.mean() - lo.mean()
    sed = np.sqrt(hi.var(ddof=1) / len(hi) + lo.var(ddof=1) / len(lo))
    print(f"\n  들썩 − 안정 : ΔE[z²] = {d:+.2f}  ± {sed:.2f} (SE)   t = {d/sed:+.2f}")
    print("  → 유의한 양수면 '변동성 큰 투수는 자기 과거가 덜 넘어간다' = 투수별 k의 근거")

    out = Path(__file__).parent.parent / "experiments" / "mixvol_pairs.csv"
    pair.to_csv(out, index=False)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
