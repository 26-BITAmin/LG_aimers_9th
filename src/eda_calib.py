"""세그먼트별 캘리브레이션 — EDA(F)가 찾은 잔차 편향이 **재현되는가**

EDA §F가 배포 모델의 세그먼트별 레벨 오차를 보여줬다 (fold2024):
    asof_pitcher_n 최저구간  예측 .4752 vs 실제 .4657   (신인 구간)
    game_month 3            예측 .4907 vs 실제 .5026
    count_state 9 (3볼0스)   예측 .4675 vs 실제 .4820

이건 **refinement가 아니라 calibration** 손실이다 — 이 대회 지표가 정확히 보는 것이고,
행 독립적으로 고칠 수 있다 (각 행이 자기 세그먼트의 delta를 조회할 뿐).

⚠️ 그런데 크기를 재는 것과 **고칠 수 있는지**는 다른 질문이다. MODEL.md §11이
   경고한 대로 레벨 축은 fold 간 외삽이 식별되지 않는다. 그래서 여기서 묻는 것은
   "얼마나 잃고 있나"가 아니라 **"fold2023의 편향이 fold2024의 편향을 예측하는가"**다.

   예측하면 → 과거로 추정한 delta가 미래에 전이된다 = 고칠 수 있다
   예측 못 하면 → 매 시즌 다른 노이즈다 = 재는 순간 이미 끝난 이야기

실행:  .venv/Scripts/python.exe -u src/eda_calib.py 2>&1 | tee experiments/eda_calib.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache
from config import MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
MIN_N = 500


def segments(d):
    """세그먼트 정의. 전부 **투구 직전에 알 수 있는 값**이라 추론 시 조회 가능하다."""
    n = d["asof_pitcher_n"].to_numpy(dtype=np.float64)
    return {
        "count_state": d["balls_before"].to_numpy() * 3 + d["strikes_before"].to_numpy(),
        "game_month": d["game_month"].to_numpy(),
        "inning": np.clip(d["inning"].to_numpy(), 1, 10),
        "hand_matchup": d["pitcher_hand"].to_numpy() * 4 + d["batter_hand"].to_numpy(),
        "base_state": d["base_state"].astype("category").cat.codes.to_numpy(),
        "pitcher_team": d["pitcher_team_id"].to_numpy(),
        # 경계를 **고정 상수**로 준다. 검증셋에서 분위수를 뽑으면 fold 간 경계가
        # 달라져 편향을 비교할 수 없고, 배포 시엔 test 통계 사용이라 규정 위반이다.
        "asof_n_bucket": np.searchsorted(
            np.array([50, 200, 500, 1000, 2000, 5000, 10000], dtype=np.float64), n),
    }


def main():
    cached = foldcache.load(FOLDS, verbose=False)
    df = load_csv(TRAIN_CSV)
    season = df["season"].to_numpy()

    P, Y, S = {}, {}, {}
    for f in FOLDS:
        z = (1 - MLP_ALPHA) * cached[f][0] + MLP_ALPHA * cached[f][1]
        y = cached[f][2]
        d = df.loc[season == f].reset_index(drop=True)
        assert len(d) == len(y)
        P[f] = shift_to_prior(1.0 / (1.0 + np.exp(-z)), y.mean())
        Y[f], S[f] = y, segments(d)
        print(f"fold{f}  n={len(y):,}  BSS oracle = {bss(y, P[f]):.1f}")

    print("\n" + "=" * 96)
    print("세그먼트별 레벨 편향 (예측평균 − 실제평균). 전역 레벨은 이미 제거돼 있다")
    print("=" * 96)

    summary = []
    for name in segments(df.head(10)):
        levels, rows = [], []
        for f in FOLDS:
            key, y, p = S[f][name], Y[f], P[f]
            u = [k for k in np.unique(key) if (key == k).sum() >= MIN_N]
            levels.append(set(u.tolist() if hasattr(u, "tolist") else u))
        common = sorted(set.intersection(*levels))
        if len(common) < 3:
            continue

        bias, share = {}, {}
        for f in FOLDS:
            key, y, p = S[f][name], Y[f], P[f]
            bias[f] = np.array([p[key == k].mean() - y[key == k].mean() for k in common])
            share[f] = np.array([(key == k).mean() for k in common])

        # 이 세그먼트의 레벨 오차가 전체 BSS에서 차지하는 몫
        ref = {f: Y[f].mean() * (1 - Y[f].mean()) for f in FOLDS}
        cost = {f: float(np.sum(share[f] * bias[f] ** 2) * 100000.0 / ref[f])
                for f in FOLDS}

        r = float(np.corrcoef(bias[2023], bias[2024])[0, 1])
        agree = float(np.mean(np.sign(bias[2023]) == np.sign(bias[2024])))
        summary.append({"name": name, "k": len(common), "corr": r, "agree": agree,
                        "cost23": cost[2023], "cost24": cost[2024],
                        "levels": [float(x) for x in common],
                        "bias23": bias[2023].tolist(), "bias24": bias[2024].tolist(),
                        "share24": share[2024].tolist()})

        print(f"\n[{name}]  수준 {len(common)}개   "
              f"레벨 손실 fold2023 {cost[2023]:>6.1f} / fold2024 {cost[2024]:>6.1f} BSS")
        print(f"{'수준':>8} {'비중24':>8} {'편향23':>10} {'편향24':>10}  부호")
        order = np.argsort(-np.abs(bias[2024]))
        for i in order[:10]:
            sgn = "일치" if np.sign(bias[2023][i]) == np.sign(bias[2024][i]) else "  ✗"
            print(f"{common[i]:>8.0f} {share[2024][i]:>8.1%} {bias[2023][i]:>+10.4f}"
                  f" {bias[2024][i]:>+10.4f}  {sgn}")
        print(f"   corr(편향23, 편향24) = {r:+.3f}   부호 일치율 {agree:.0%}")

    print("\n" + "=" * 96)
    print("판정 — 과거 편향이 미래 편향을 예측하는가")
    print("=" * 96)
    print(f"{'세그먼트':>16} {'수준':>5} {'손실23':>8} {'손실24':>8} {'corr':>7} {'부호일치':>9}")
    for s in sorted(summary, key=lambda x: -x["cost24"]):
        print(f"{s['name']:>16} {s['k']:>5} {s['cost23']:>8.1f} {s['cost24']:>8.1f}"
              f" {s['corr']:>+7.3f} {s['agree']:>9.0%}")
    print("\n  corr이 높고 부호 일치율이 높아야 '과거로 추정한 delta가 전이된다'가 된다.")
    print("  corr ≈ 0 이면 매 시즌 다른 노이즈이므로 크기가 커도 고칠 수 없다.")

    import json
    from config import EXP_DIR
    (EXP_DIR / "eda_calib.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {EXP_DIR / 'eda_calib.json'}")


if __name__ == "__main__":
    main()
