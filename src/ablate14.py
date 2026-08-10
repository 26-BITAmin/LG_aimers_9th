"""14차 실험 — 피처 제거 후보를 **묶어서 짝지은 시드로** 재검정.

왜 다시 하는가
    13차(ablate12)는 19개 그룹을 하나씩 뺐고 +2~+7짜리가 네 개 통과했다.
    그런데 시드 노이즈가 fold당 ±5 수준이다. **19번 뽑으면 그 정도는 노이즈가 만든다.**
    부호 일치 / 크기비 필터는 다중비교를 해결하지 못한다.

두 가지를 바꾼다
    1) **묶어서 잰다.** 개별 +3이 진짜라면 넷을 합쳤을 때 +17 근처가 나와야 한다.
       노이즈였다면 0 근처로 무너진다. 효과를 노이즈 위로 끌어올리는 방법이다.
    2) **시드를 짝짓는다.** 기준과 변경을 같은 시드로 돌리고 시드별 차이를 낸다.
       공통 노이즈가 상쇄돼 표준오차가 크게 줄어든다.
       (13차는 평균의 차를 냈다 — 짝짓기를 버린 셈이다.)

    덤으로 피처 행렬을 fold당 한 번만 만들고 컬럼만 골라 쓴다.
    `fit_artifacts`는 어떤 컬럼을 버릴지와 무관하므로 이 재사용은 안전하다.

실행:  .venv/Scripts/python.exe -u src/ablate14.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import lightgbm as lgb
import numpy as np

from config import (DROP_FEATURES, FINAL_LGB_PARAMS, N_ROUNDS, RECENCY_HALF_LIFE,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV,
                    USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
SEEDS = (42, 7, 2024, 1234, 31337)

G = {
    "달력": ["game_month", "game_dayofweek"],
    "점수": ["run_top_before", "run_bot_before", "run_total_before",
             "score_diff_home", "score_diff_pitcher_team", "abs_score_diff"],
    "결측": ["miss_pitcher_asof", "miss_batter_asof", "miss_prev_game"],
    "구종": ["pitchmix_entropy", "pitchmix_nonfastball"],
    "승부처": ["home_win_expectancy", "li", "we_margin", "is_late_close"],
    "폼차분": ["form_p1_minus_career", "form_p3_minus_career", "form_p1_minus_p5",
             "form_p3_minus_p5", "form_mid_p1_minus_career"],
    "실패구성": ["fail_middle_share", "pitcher_ball_minus_strike"],
    "표본수": ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"],
    "매치업": ["matchup_success_edge", "matchup_middle_edge"],
    "스무딩": ["asof_pitcher_success_rate_sm", "asof_pitcher_middle_rate_sm",
             "asof_pitcher_reverse_rate_sm", "asof_pitcher_ball_rate_sm",
             "asof_pitcher_strike_rate_sm", "asof_batter_success_rate_sm",
             "asof_batter_middle_rate_sm", "asof_pitcher_fastball_rate_sm",
             "asof_pitcher_breaking_rate_sm", "asof_pitcher_offspeed_rate_sm"],
}


def cols(*names):
    return [c for n in names for c in G[n]]


# 13차에서 "빼는 게 낫다"가 나온 넷 = 점수·결측·구종·승부처
CONFIGS = {
    "기준 (80개)": [],
    "−달력": cols("달력"),
    "−통과4": cols("점수", "결측", "구종", "승부처"),
    "−달력−통과4": cols("달력", "점수", "결측", "구종", "승부처"),
    "−양수전체": cols("달력", "점수", "결측", "구종", "승부처",
                  "폼차분", "실패구성", "표본수", "매치업"),
    "−달력−통과4−스무딩": cols("달력", "점수", "결측", "구종", "승부처", "스무딩"),
}


def fold_data(df, valid_season):
    tr = df[df.season < valid_season]
    va = df[df.season == valid_season]
    ref = int(tr.season.max())
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=ref)
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    return (X(tr), tr[TARGET_COL].to_numpy(dtype=np.float64), w,
            X(va), va[TARGET_COL].to_numpy(dtype=np.float64))


def fit_score(Xtr, ytr, w, Xva, yva, seed):
    p = dict(FINAL_LGB_PARAMS, seed=seed, bagging_seed=seed,
             feature_fraction_seed=seed, data_random_seed=seed)
    bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=N_ROUNDS)
    pr = 1 / (1 + np.exp(-bst.predict(Xva, raw_score=True)))
    return bss(yva, shift_to_prior(pr, yva.mean()))


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)

    # res[config][fold] = 시드별 BSS 배열 (시드 순서가 config 간에 정렬돼 있어야 짝짓기가 된다)
    res = {k: {} for k in CONFIGS}
    for f in FOLDS:
        Xtr, ytr, w, Xva, yva = fold_data(df, f)
        print(f"\n=== valid={f}  train={Xtr.shape}  ({time.time()-t0:.0f}s) ===")
        for name, drop in CONFIGS.items():
            keep = [c for c in Xtr.columns if c not in set(drop)]
            v = np.array([fit_score(Xtr[keep], ytr, w, Xva[keep], yva, s) for s in SEEDS])
            res[name][f] = v
            print(f"  {name:<20s} ({len(keep):>2d}개)  {v.mean():>8.1f} ± {v.std(ddof=1):>4.1f}"
                  f"   ({time.time()-t0:.0f}s)")

    base = "기준 (80개)"
    print(f"\n{'설정':<20s} " + " ".join(f"{f:>9d}" for f in FOLDS)
          + f" {'평균차':>9s} {'짝지은SE':>9s} {'t':>6s}")
    print("-" * 70)
    for name in CONFIGS:
        if name == base:
            print(f"{name:<20s} " + " ".join(f"{res[name][f].mean():>9.1f}" for f in FOLDS))
            continue
        # 시드별 짝지은 차이 — 공통 노이즈가 상쇄된다
        d = {f: res[name][f] - res[base][f] for f in FOLDS}
        all_d = np.concatenate([d[f] for f in FOLDS])
        se = all_d.std(ddof=1) / np.sqrt(len(all_d))
        t = all_d.mean() / max(se, 1e-9)
        print(f"{name:<20s} " + " ".join(f"{d[f].mean():>+9.1f}" for f in FOLDS)
              + f" {all_d.mean():>+9.1f} {se:>9.1f} {t:>6.1f}")

    print("\n=== 판정 ===")
    print("  짝지은 t가 |t| > 3 이고 두 fold 부호가 같으면 실제 효과로 본다.")
    print("  개별 +3짜리 넷을 묶어 +17이 안 나오면 그것들은 노이즈였다는 뜻이다.")
    for name in CONFIGS:
        if name == base:
            continue
        d = {f: (res[name][f] - res[base][f]).mean() for f in FOLDS}
        all_d = np.concatenate([res[name][f] - res[base][f] for f in FOLDS])
        se = all_d.std(ddof=1) / np.sqrt(len(all_d))
        t = all_d.mean() / max(se, 1e-9)
        same = (d[FOLDS[0]] > 0) == (d[FOLDS[1]] > 0)
        ok = abs(t) > 3 and same and all_d.mean() > 0
        print(f"  {name:<20s} 평균 {all_d.mean():+6.1f}  t={t:+5.1f}  "
              f"부호일치 {'O' if same else 'X'}  → {'채택' if ok else '보류'}")

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
