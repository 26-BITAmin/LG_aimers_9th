"""13차 실험 — 피처 그룹 leave-one-out.

지금까지 얻은 최대 이득이 **피처 제거**(`game_type`, +172)였는데,
그건 gain 순위를 보다 우연히 찾은 것이고 80개 피처에 대한 체계적 선택은 한 적이 없다.

게다가 피처 제거는 **분산 축소 계열**이다 — 정보를 더하는 게 아니라 노이즈를 빼는 것.
방금 LB에서 이 계열은 홀드아웃 예측대로 전이된다는 것이 확인됐다
(앙상블 +5.9 예측 → +6.16 실측).

방법: 그룹 단위로 하나씩 빼고 2 fold × 3시드로 평가한다.
      빼서 좋아지면 그 그룹은 노이즈다.

채택 조건은 §MODEL.md §2 — 부호 일치 + fold 간 크기비 3배 이내.

실행:  .venv/Scripts/python.exe -u src/ablate12.py
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

from config import DROP_FEATURES, TRAIN_CSV
from data import load_csv
from evaluate import run_fold

FOLDS = [2023, 2024]
SEEDS = (42, 7, 2024)

# 자연스러운 의미 단위로 묶는다. 개별 피처 단위는 노이즈가 커서 판정이 안 된다.
GROUPS = {
    "달력": ["game_month", "game_dayofweek"],
    "이닝·아웃": ["inning", "outs_before"],
    "점수": ["run_top_before", "run_bot_before", "run_total_before",
             "score_diff_home", "score_diff_pitcher_team", "abs_score_diff"],
    "주자": ["runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
             "base_state"],
    "승부처": ["home_win_expectancy", "li", "we_margin", "is_late_close"],
    "카운트 파생": ["count_state", "count_diff", "is_two_strikes", "is_three_balls",
                "is_full_count", "is_first_pitch"],
    "손잡이": ["pitcher_hand", "batter_hand", "hand_same", "hand_matchup"],
    "홈·초말": ["is_pitcher_home", "top_bottom"],
    "팀 ID": ["pitcher_team_id", "batter_team_id"],
    "선수 ID": ["pitcher_id", "batter_id"],
    "표본 수": ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"],
    "결측 플래그": ["miss_pitcher_asof", "miss_batter_asof", "miss_prev_game"],
    "asof 원본": ["asof_pitcher_success_rate", "asof_pitcher_middle_rate",
                "asof_pitcher_reverse_rate", "asof_pitcher_ball_rate",
                "asof_pitcher_strike_rate", "asof_batter_success_rate",
                "asof_batter_middle_rate", "asof_pitcher_fastball_rate",
                "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"],
    "asof 스무딩": ["asof_pitcher_success_rate_sm", "asof_pitcher_middle_rate_sm",
                 "asof_pitcher_reverse_rate_sm", "asof_pitcher_ball_rate_sm",
                 "asof_pitcher_strike_rate_sm", "asof_batter_success_rate_sm",
                 "asof_batter_middle_rate_sm", "asof_pitcher_fastball_rate_sm",
                 "asof_pitcher_breaking_rate_sm", "asof_pitcher_offspeed_rate_sm"],
    "직전경기": ["asof_pitcher_prev1_game_success_rate",
              "asof_pitcher_prev3_game_success_rate",
              "asof_pitcher_prev5_game_success_rate",
              "asof_pitcher_prev1_game_middle_rate",
              "asof_pitcher_prev3_game_middle_rate",
              "asof_pitcher_prev5_game_middle_rate"],
    "폼 차분": ["form_p1_minus_career", "form_p3_minus_career", "form_p1_minus_p5",
              "form_p3_minus_p5", "form_mid_p1_minus_career"],
    "매치업 엣지": ["matchup_success_edge", "matchup_middle_edge"],
    "실패 구성": ["fail_middle_share", "pitcher_ball_minus_strike"],
    "구종 믹스": ["pitchmix_entropy", "pitchmix_nonfastball"],
}


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)

    base = {f: run_fold(df, f, seeds=SEEDS) for f in FOLDS}
    b = np.array([base[f].mean() for f in FOLDS])
    print(f"기준선: " + "  ".join(f"{f}={base[f].mean():.1f}" for f in FOLDS)
          + f"   ({time.time()-t0:.0f}s)\n")

    rows = []
    for name, cols in GROUPS.items():
        drop = list(DROP_FEATURES) + cols
        v = np.array([run_fold(df, f, drop=drop, seeds=SEEDS).mean() for f in FOLDS])
        d = v - b
        same = bool(np.all(d > 0) or np.all(d < 0))
        sp = np.abs(d).max() / max(np.abs(d).min(), 1e-9) if same else np.inf
        rows.append((name, len(cols), d, same, sp))
        print(f"  −{name:<12s} ({len(cols):>2d}개)  fold별 {np.round(d,1)}  "
              f"평균 {d.mean():>+7.1f}  {'O' if same else 'X'}  {sp:>5.1f}배"
              f"   ({time.time()-t0:.0f}s)")

    rows.sort(key=lambda r: -r[2].mean())
    print(f"\n{'제거 그룹':<14s} {'개수':>4s} {'2023':>9s} {'2024':>9s} {'평균':>9s} "
          f"{'부호':>4s} {'크기비':>7s}")
    print("-" * 62)
    for name, n, d, same, sp in rows:
        mark = "  <<< 빼는 게 낫다" if (same and d.mean() > 0 and sp <= 3) else ""
        print(f"{name:<14s} {n:>4d} {d[0]:>9.1f} {d[1]:>9.1f} {d.mean():>+9.1f} "
              f"{'O' if same else 'X':>4s} {sp:>7.1f}{mark}")

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
