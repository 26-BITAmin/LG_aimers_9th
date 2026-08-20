"""asof_* 누적이 시즌을 넘어가는가 — 그리고 그것이 무엇을 뜻하는가.

가설(사베르메트릭스의 era adjustment):
  `asof_pitcher_success_rate`가 **커리어 누적**이라면, 그 값은 투수가 던진 시기의
  **리그 평균 수준**을 흡수하고 있다. 리그 제구율은 .565(2019) → .486(2024)로
  8%p 내려갔다. 즉 같은 0.52라도 2019년에 쌓은 0.52와 2024년에 쌓은 0.52는
  실력이 다르다. OPS+/ERA+가 리그 상대값을 쓰는 것과 같은 이유다.

  전역 logit shift는 **평균** 레벨 오차만 지운다. 누적 구간이 투수마다 다르면
  편향도 투수마다 달라서 스칼라 하나로는 못 지운다.

여기서는 학습 없이 세 가지만 잰다.
  A. asof_pitcher_n이 시즌 경계에서 리셋되는가
  B. 같은 시즌 안에서 "누적 era 조성"이 투수마다 얼마나 다른가
  C. 그 조성 불일치가 **배포 모델 잔차**와 상관이 있는가  (있어야 후보다)
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C


def main():
    cols = ["season", "game_month", "pitcher_id", "asof_pitcher_n",
            "asof_pitcher_success_rate", "control_success"]
    df = pd.read_csv(C.TRAIN_CSV, usecols=cols)
    print(f"rows {len(df):,}")

    # ── A. 시즌 경계에서 리셋되는가 ────────────────────────────────
    # 각 (투수, 시즌)의 asof_n 최소·최대. 리셋되면 매 시즌 최소가 0 근처다.
    g = df.groupby(["pitcher_id", "season"])["asof_pitcher_n"].agg(["min", "max", "count"])
    g = g.reset_index().sort_values(["pitcher_id", "season"])
    g["prev_max"] = g.groupby("pitcher_id")["max"].shift(1)
    cont = g.dropna(subset=["prev_max"])
    # 이어지면 min ≈ prev_max, 리셋되면 min ≈ 0
    print("\n[A] 시즌 경계 연속성 (같은 투수의 다음 시즌 첫 행 asof_n)")
    print(f"    표본 (투수,시즌) 쌍       : {len(cont):,}")
    print(f"    min == 0 인 비율          : {(cont['min'] == 0).mean():.4f}")
    print(f"    min >= prev_max 인 비율   : {(cont['min'] >= cont['prev_max']).mean():.4f}")
    print(f"    중앙 |min - prev_max|     : {(cont['min'] - cont['prev_max']).abs().median():.1f}")

    career = (cont["min"] >= cont["prev_max"]).mean() > 0.9
    print(f"    → {'커리어 누적 (시즌을 넘어간다)' if career else '시즌 리셋'}")

    # ── B. era 조성의 산포 ────────────────────────────────────────
    # 시즌별 리그 성공률
    lg = df.groupby("season")["control_success"].mean()
    print("\n[B] 시즌별 리그 제구율")
    for s, v in lg.items():
        print(f"    {s}  {v:.4f}")

    # 각 (투수,시즌)의 누적 표본 중 과거 시즌이 차지하는 비중으로
    # "기대 리그 수준" = 누적 구간의 가중 평균 리그율을 만든다.
    # 투수별 시즌 투구 수
    per = df.groupby(["pitcher_id", "season"]).size().rename("n").reset_index()
    per = per.sort_values(["pitcher_id", "season"])
    per["cum_before"] = per.groupby("pitcher_id")["n"].cumsum() - per["n"]
    # 누적 리그율 합 (과거 시즌까지)
    per["lg"] = per["season"].map(lg)
    per["wsum_before"] = per.groupby("pitcher_id").apply(
        lambda d: (d["n"] * d["lg"]).cumsum() - d["n"] * d["lg"], include_groups=False
    ).reset_index(level=0, drop=True)

    print("\n[C] 같은 시즌 안에서 '누적 era 수준'의 산포")
    for s in [2023, 2024]:
        sub = per[(per["season"] == s) & (per["cum_before"] > 0)].copy()
        sub["era_level"] = sub["wsum_before"] / sub["cum_before"]
        w = sub["n"]
        m = np.average(sub["era_level"], weights=w)
        sd = np.sqrt(np.average((sub["era_level"] - m) ** 2, weights=w))
        print(f"    {s}: 투수 {len(sub):4d}  가중평균 {m:.4f}  가중std {sd:.4f}  "
              f"범위 {sub['era_level'].min():.4f}~{sub['era_level'].max():.4f}  "
              f"(당해 리그 {lg[s]:.4f})")
        # 5분위
        q = np.quantile(sub["era_level"], [0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"        분위 5/25/50/75/95: " + " ".join(f"{v:.4f}" for v in q))


if __name__ == "__main__":
    main()
