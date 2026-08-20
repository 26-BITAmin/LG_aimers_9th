"""`TARGET_PRIOR_2025` 백테스트 — 이 축은 판정 불가가 아니었다

왜 이게 큰가
    Brier에서 레벨 오차는 제곱으로 들어온다.

        BSS 손실 = 100000 · (Δmean)² / (p̄(1−p̄))  ≈  **400,700 · (Δmean)²**

        Δ 0.005 → −10      Δ 0.010 → −40
        Δ 0.015 → −90      Δ 0.020 → −160

    오늘 δ 채널 둘로 번 것이 LB +9.01이다. prior가 0.01 틀려 있으면 그 4배를 잃는다.
    **이 저장소에서 가장 큰 미검증 레버다.**

§11이 이 축을 접은 논리와 그 오류
    "fold에서 재는 것은 2023/2024 레벨을 맞췄는가이고, 2025 레벨은 fold 밖이다."
    → 2025 **레벨**은 못 잰다. 맞다.
    → 그런데 **"과거로 다음 시즌 레벨을 예측하는 규칙"** 은 잴 수 있다.
      2021·2022·2023·2024를 각각 그 이전 시즌들로 예측해 오차를 보면 된다.
      **완전히 train 안이고, 예측 규칙의 백테스트지 LB 역산이 아니다.**

    배포값 0.475는 근거가 기록돼 있지 않다. 이 파일이 그 근거를 만들거나 반증한다.

교란 하나는 미리 적어둔다
    `game_type` 구성(R/F 비율)이 시즌마다 다르면 전체 평균이 그것 때문에도 움직인다.
    그래서 **전체 / R만 / 구성 보정** 세 가지를 같이 낸다.

실행:  .venv/Scripts/python.exe -u src/priorfit.py 2>&1 | tee experiments/priorfit.log
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

from config import TARGET_COL, TARGET_PRIOR_2025, TRAIN_CSV
from data import load_csv

BSS_PER_SQ = None      # 아래에서 실제 p̄로 계산한다


def rules():
    """이름 → (과거 시즌 rate 시리즈) -> 다음 시즌 예측.

    전부 **과거만** 본다. 파라미터가 있는 규칙은 그 파라미터도 백테스트 대상이다.
    """
    def last(s):                      # 직전 시즌 그대로
        return s.iloc[-1]

    def mean_k(k):                    # 직전 k시즌 평균
        return lambda s: s.iloc[-k:].mean()

    def drift_k(k):                   # 직전 k개 차분의 평균만큼 더 민다
        def f(s):
            if len(s) < 2:
                return s.iloc[-1]
            d = np.diff(s.to_numpy())[-k:]
            return s.iloc[-1] + d.mean()
        return f

    def ols_k(k):                     # 직전 k시즌 선형 추세 외삽
        def f(s):
            if len(s) < 2:
                return s.iloc[-1]
            y = s.to_numpy()[-k:]
            x = np.arange(len(y))
            b, a = np.polyfit(x, y, 1)
            return a + b * len(y)
        return f

    return {
        "직전 시즌 그대로": last,
        "직전 2시즌 평균": mean_k(2),
        "직전 3시즌 평균": mean_k(3),
        "드리프트 1차분": drift_k(1),
        "드리프트 2차분평균": drift_k(2),
        "드리프트 3차분평균": drift_k(3),
        "선형추세 3시즌": ols_k(3),
        "선형추세 4시즌": ols_k(4),
        "선형추세 전체": ols_k(99),
    }


def backtest(rate, label):
    global BSS_PER_SQ
    seasons = list(rate.index)
    targets = seasons[2:]              # 예측하려면 과거가 최소 2시즌은 있어야 한다
    print("\n" + "=" * 92)
    print(f"[{label}]  시즌별 성공률: "
          + "  ".join(f"{s}:{rate[s]:.4f}" for s in seasons))
    print("=" * 92)
    print(f"  {'규칙':<20}" + "".join(f"{t:>10}" for t in targets)
          + f"{'MAE':>9}{'RMSE':>9}{'BSS손실':>10}{'2025예측':>10}")

    out = {}
    for name, fn in rules().items():
        errs, row = [], []
        for t in targets:
            past = rate[rate.index < t]
            pred = float(fn(past))
            e = pred - rate[t]
            errs.append(e)
            row.append(e)
        e = np.array(errs)
        mae, rmse = np.abs(e).mean(), np.sqrt((e ** 2).mean())
        loss = BSS_PER_SQ * (e ** 2).mean()          # 기대 BSS 손실
        p2025 = float(fn(rate))
        out[name] = (mae, rmse, loss, p2025)
        print(f"  {name:<20}" + "".join(f"{v:>+10.4f}" for v in row)
              + f"{mae:>9.4f}{rmse:>9.4f}{loss:>10.1f}{p2025:>10.4f}")
    return out


def main():
    global BSS_PER_SQ
    df = load_csv(TRAIN_CSV)[["season", "game_type", TARGET_COL]]
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    pbar = y.mean()
    BSS_PER_SQ = 100000.0 / (pbar * (1 - pbar))
    print(f"train {len(df):,}행   전체 성공률 {pbar:.4f}")
    print(f"레벨 오차 1단위당 BSS 손실 = {BSS_PER_SQ:,.0f} × (Δmean)²")
    print(f"  Δ0.005 → −{BSS_PER_SQ*0.005**2:.0f}    Δ0.010 → −{BSS_PER_SQ*0.010**2:.0f}"
          f"    Δ0.020 → −{BSS_PER_SQ*0.020**2:.0f}")

    gt = df.groupby("season", observed=True).game_type.value_counts(normalize=True).unstack()
    print(f"\n시즌별 game_type 구성:\n{gt.round(4).to_string()}")

    all_rate = df.groupby("season", observed=True)[TARGET_COL].mean()
    r_rate = (df[df.game_type == "R"].groupby("season", observed=True)[TARGET_COL].mean())

    res_all = backtest(all_rate, "전체")
    res_r = backtest(r_rate, "정규시즌(R)만")

    # ── 배포값과의 대조 ────────────────────────────────────────────
    print("\n" + "=" * 92)
    print(f"배포값 TARGET_PRIOR_2025 = {TARGET_PRIOR_2025}  vs  백테스트 규칙들의 2025 예측")
    print("=" * 92)
    best = min(res_all.items(), key=lambda kv: kv[1][1])
    print(f"  전체 기준 RMSE 최소 규칙: **{best[0]}**  RMSE {best[1][1]:.4f}"
          f"  → 2025 예측 **{best[1][3]:.4f}**")
    bestr = min(res_r.items(), key=lambda kv: kv[1][1])
    print(f"  R만  기준 RMSE 최소 규칙: **{bestr[0]}**  RMSE {bestr[1][1]:.4f}"
          f"  → 2025 예측 **{bestr[1][3]:.4f}**")

    print(f"\n  {'규칙':<20}{'2025예측':>10}{'배포와 차':>11}{'그 차이의 BSS':>14}")
    for name in rules():
        p = res_all[name][3]
        d = p - TARGET_PRIOR_2025
        print(f"  {name:<20}{p:>10.4f}{d:>+11.4f}{BSS_PER_SQ*d**2:>14.1f}")

    print("\n  ⚠️ 위 'BSS' 열은 **배포값과 그 규칙이 얼마나 다른가**의 크기지,")
    print("     이득이 아니다. 어느 쪽이 2025에 가까운지는 아무도 모른다.")
    print("     읽는 법: 규칙들이 배포값 근처에 모이면 0.475는 안전하고,")
    print("              한쪽으로 몰려 있으면 배포값이 편향돼 있다는 신호다.")
    print("\n  ⛔ LB로 이 값을 맞추지 않는다 (SUBMISSIONS.md 규칙 2). 여기서 나오는")
    print("     근거는 **train 안의 백테스트**뿐이고, 그것만으로 움직인다.")


if __name__ == "__main__":
    main()


# ── 복합 규칙: R/F를 따로 예측해 구성으로 합친다 ────────────────────────
# 전체 시리즈는 요철이 있고 R 시리즈는 매끄럽다. 요철의 정체가 game_type 구성이면
# **R·F를 따로 예측하고 구성 비율로 합치는 것**이 더 정확해야 한다.
#
# ⚠️ 2025의 R/F 구성은 **알 수 없다** (test.csv를 집계하면 규정 위반).
#    과거 평균으로 추정한다 — 그것도 백테스트 대상에 넣는다.
def composite():
    df = load_csv(TRAIN_CSV)[["season", "game_type", TARGET_COL]]
    rate = df.groupby(["season", "game_type"], observed=True)[TARGET_COL].mean().unstack()
    share = (df.groupby("season", observed=True).game_type
               .value_counts(normalize=True).unstack())
    total = df.groupby("season", observed=True)[TARGET_COL].mean()
    seasons = list(total.index)

    print("\n" + "=" * 92)
    print("복합 규칙 — R·F 따로 예측 후 구성으로 합산")
    print("=" * 92)
    print(f"  R rate: " + "  ".join(f"{s}:{rate.loc[s,'R']:.4f}" for s in seasons))
    print(f"  F rate: " + "  ".join(f"{s}:{rate.loc[s,'F']:.4f}" for s in seasons))
    print(f"  F 비중: " + "  ".join(f"{s}:{share.loc[s,'F']:.4f}" for s in seasons))

    targets = seasons[2:]
    print(f"\n  {'규칙(R·F 공통)':<20}" + "".join(f"{t:>10}" for t in targets)
          + f"{'RMSE':>9}{'BSS손실':>10}{'2025예측':>10}")
    out = {}
    for name, fn in rules().items():
        errs, row = [], []
        for t in targets:
            past = [s for s in seasons if s < t]
            pr = float(fn(rate.loc[past, "R"]))
            pf = float(fn(rate.loc[past, "F"]))
            sf = float(share.loc[past, "F"].mean())        # 구성은 과거 평균
            e = (1 - sf) * pr + sf * pf - total[t]
            errs.append(e); row.append(e)
        e = np.array(errs)
        rmse = np.sqrt((e ** 2).mean())
        pr = float(fn(rate["R"])); pf = float(fn(rate["F"]))
        sf = float(share["F"].mean())
        p2025 = (1 - sf) * pr + sf * pf
        out[name] = (rmse, p2025)
        print(f"  {name:<20}" + "".join(f"{v:>+10.4f}" for v in row)
              + f"{rmse:>9.4f}{BSS_PER_SQ*(e**2).mean():>10.1f}{p2025:>10.4f}")

    print("\n" + "=" * 92)
    print("최종 대조 — 세 경로의 2025 예측")
    print("=" * 92)
    b = min(out.items(), key=lambda kv: kv[1][0])
    print(f"  복합 최선: {b[0]}  RMSE {b[1][0]:.4f}  → **{b[1][1]:.4f}**")
    print(f"  배포값                                    → **{TARGET_PRIOR_2025:.4f}**")
    d = b[1][1] - TARGET_PRIOR_2025
    print(f"\n  차이 {d:+.4f}  →  BSS {BSS_PER_SQ*d**2:.1f}")
    print(f"  이 규칙의 백테스트 RMSE가 {b[1][0]:.4f}이므로 **줄일 수 없는 바닥이"
          f" {BSS_PER_SQ*b[1][0]**2:.0f} BSS**다.")
    print("  → 차이가 바닥보다 작으면 배포값을 바꿀 근거가 없다.")


composite()
