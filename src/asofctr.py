"""`asof_*` 를 (시즌 × game_type) 으로 중심화하면 방향성이 살아나는가 (사용자 제안).

    "시즌별 누적 asof를 측정하고 시즌, game_type 별 asof 평균을 구하고 평균을 0으로
     설정해서 scale을 다시 설정하면 모델이 방향성을 더 잘 학습할 수 있을 것 같은데"

사실관계 (확인 완료):
    · `season`·`game_type` 은 **모델 피처가 아니다** (features.py:483, DROP_FEATURES).
    · `asof_*_sm` 의 스무딩 prior 는 **train 전체 가중평균 상수 1개**다 (시즌별 아님).
    → 모델은 asof=0.52 가 2019 것인지 2024 것인지 구분할 수단이 없다.
      그런데 리그 평균은 .565 → .486 으로 움직였다. 같은 0.52 가 2019 엔 평균 이하,
      2024 엔 평균 이상이다. **두 해의 상반된 라벨이 같은 피처값에 겹친다.**

가설 H: 그래서 기울기가 0 쪽으로 **감쇠**한다.
        → 검정: **풀링 기울기 < 시즌 내 기울기** 여야 한다. 아니면 가설이 틀린 것이다.
        → 이것이 §7.105 의 β=1.2(모델이 개인차를 20% 덜 표현)의 후보 원인이다.

파트
  A  드리프트 실측 — 컬럼별 (시즌×game_type) 평균이 실제로 움직이는가
  B  ★ 감쇠 검정 — 풀링 기울기 vs 시즌 내 기울기 (컬럼 전수)
  C  형태 검정 — 시즌 곡선이 평행이동인가 모양이 다른가 (십분위)
  D  ★ 전이 검정 — 값싼 로지스틱 프록시로 raw vs centered 를 2024 에서 겨룬다
       (GBDT 재학습 전에 축이 살아 있는지부터 본다)
  E  2025 상수 문제 — 중심화 상수를 test 에서 못 구한다. 외삽 오차의 비용을 잰다.

⚠️ 리키지: test 행으로 평균을 내는 것은 **규정 위반**이다 (CLAUDE.md §3.2 분포 통계).
   따라서 2025 상수는 train 에서 외삽해야 하고, E 가 그 오차의 대가를 잰다.

실행:  .venv/Scripts/python.exe -u src/asofctr.py 2>&1 | tee experiments/asofctr.log
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

from config import TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss

RATE_COLS = [
    "asof_pitcher_success_rate", "asof_pitcher_middle_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]
SEED = 42


def wslope(x, y, w=None):
    """단순 OLS 기울기 (결측 제외). 확률 스케일이라 p(1-p)≈.25 로 등분산에 가깝다."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return np.nan
    x, y = x[m], y[m]
    xc = x - x.mean()
    d = float((xc ** 2).sum())
    return float((xc * (y - y.mean())).sum() / d) if d > 0 else np.nan


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    mclean = ~((gt == "F") & (s < 2023))
    seasons = sorted(set(s[mclean].tolist()))
    print(f"train {df.shape}   clean(F<2023 제외) {int(mclean.sum()):,}행")

    # ── A — 드리프트 실측 ───────────────────────────────────────────────
    print(f"\n{'='*96}\n█ A — 컬럼별 (시즌) 평균 드리프트   [R 행만. F 는 2023~ 뿐이라 따로]\n{'='*96}")
    print(f"  {'컬럼':<42}" + "".join(f"{t:>8}" for t in seasons) + f"{'19→24':>9}{'std(y)대비':>11}")
    mR = mclean & (gt == "R")
    drift = {}
    for c in RATE_COLS:
        v = df[c].to_numpy(dtype=np.float64)
        ms = [np.nanmean(v[mR & (s == t)]) for t in seasons]
        d = ms[-1] - ms[0]
        drift[c] = d
        print(f"  {c:<42}" + "".join(f"{m:>8.4f}" for m in ms)
              + f"{d:>+9.4f}{abs(d)/np.nanstd(v[mR]):>11.2f}")
    ybar = [y[mR & (s == t)].mean() for t in seasons]
    print(f"  {'(참고) 타깃 control_success':<42}" + "".join(f"{m:>8.4f}" for m in ybar)
          + f"{ybar[-1]-ybar[0]:>+9.4f}")

    # ── B — ★ 감쇠 검정 ────────────────────────────────────────────────
    print(f"\n{'='*96}\n█ B — ★ 감쇠 검정: 풀링 기울기 vs 시즌 내 기울기\n{'='*96}")
    print("  H: 시즌 레벨이 섞여서 기울기가 0 쪽으로 감쇠한다  →  |풀링| < |시즌내평균| 이어야 한다.")
    print("  '중심화 이득' = 시즌내평균/풀링. 1 이면 감쇠 없음(제안 무효), >1 이면 감쇠 실재.\n")
    print(f"  {'컬럼':<42}{'풀링':>10}{'시즌내 평균':>13}{'시즌내 std':>12}"
          f"{'비(감쇠)':>10}{'시즌별 부호':>12}")
    att = {}
    for c in RATE_COLS:
        v = df[c].to_numpy(dtype=np.float64)
        pooled = wslope(v[mR], y[mR])
        within = [wslope(v[mR & (s == t)], y[mR & (s == t)]) for t in seasons]
        within = np.array(within, dtype=np.float64)
        wm = float(np.nanmean(within))
        ratio = wm / pooled if pooled not in (0, np.nan) and abs(pooled) > 1e-9 else np.nan
        att[c] = ratio
        sgn = "".join("+" if w > 0 else "-" for w in within)
        print(f"  {c:<42}{pooled:>10.4f}{wm:>13.4f}{np.nanstd(within):>12.4f}"
              f"{ratio:>10.2f}{sgn:>12}")

    # ── C — 형태 검정 (십분위 곡선이 평행인가) ──────────────────────────
    print(f"\n{'='*96}\n█ C — 형태: 시즌 곡선이 **평행이동**인가 모양이 다른가\n{'='*96}")
    print("  `asof_pitcher_success_rate` 십분위별 (시즌 내 분위) 성공률 − 그 시즌 평균")
    print("  평행이동이면 각 행이 시즌에 무관하게 같은 값이어야 한다 (= 중심화로 정렬됨).\n")
    c = "asof_pitcher_success_rate"
    v = df[c].to_numpy(dtype=np.float64)
    print(f"  {'시즌':<8}" + "".join(f"{f'D{i+1}':>8}" for i in range(10)) + f"{'D10−D1':>9}")
    for t in seasons:
        m = mR & (s == t) & np.isfinite(v)
        q = np.quantile(v[m], np.linspace(0, 1, 11))
        bi = np.clip(np.searchsorted(q[1:-1], v[m]), 0, 9)
        base = y[m].mean()
        cells = [y[m][bi == i].mean() - base for i in range(10)]
        print(f"  {t:<8}" + "".join(f"{x*100:>8.2f}" for x in cells)
              + f"{(cells[-1]-cells[0])*100:>9.2f}")

    # ── D — ★ 전이 검정 (값싼 로지스틱 프록시) ──────────────────────────
    print(f"\n{'='*96}\n█ D — ★ 전이: raw vs 중심화, 2024 홀드아웃 (로지스틱 프록시)\n{'='*96}")
    print("  GBDT 재학습 전에 축이 살아 있는지부터 본다. 학습 ≤2023 → 검증 2024.")
    print("  centered = 각 컬럼에서 **그 행의 (시즌×game_type) 평균**을 뺀다.")
    print("  2024 검증행의 중심화 상수는 **2023 값을 그대로 쓴다** (test 에서 할 수 있는 것과 동일).\n")

    def design(mask, center, ref=None):
        X = []
        for c in RATE_COLS:
            v = df[c].to_numpy(dtype=np.float64).copy()
            gmean = np.nanmean(v[mclean])
            v = np.where(np.isfinite(v), v, gmean)
            if center:
                key = pd.Series(s).astype(str) + "|" + pd.Series(gt).astype(str)
                if ref is None:
                    mu = pd.Series(np.where(mclean, v, np.nan)).groupby(key).transform("mean")
                    v = v - mu.to_numpy()
                else:
                    v = v - ref[c][key.to_numpy()]
            X.append(v[mask])
        return np.column_stack(X)

    key_all = (pd.Series(s).astype(str) + "|" + pd.Series(gt).astype(str)).to_numpy()
    # 2024 행에 쓸 중심화 상수 = 2023 값 (한 시즌 뒤처진 값 = test 에서 가능한 최선)
    ref23 = {}
    for c in RATE_COLS:
        v = df[c].to_numpy(dtype=np.float64)
        tab = {}
        for k in np.unique(key_all):
            mk = mclean & (key_all == k)
            tab[k] = np.nanmean(v[mk]) if mk.any() else np.nanmean(v[mclean])
        lag = {}
        for k in tab:
            yr, g = k.split("|")
            pk = f"{int(yr)-1}|{g}"
            lag[k] = tab.get(pk, tab[k])
        ref23[c] = pd.Series(lag)

    def fit_logistic(Xtr, ytr, iters=60, lam=1e-3):
        X = np.column_stack([np.ones(len(Xtr)), Xtr])
        b = np.zeros(X.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-X @ b))
            W = p * (1 - p) + 1e-9
            g = X.T @ (p - ytr) + lam * b
            H = (X * W[:, None]).T @ X + lam * np.eye(X.shape[1])
            b -= np.linalg.solve(H, g)
        return b

    mtr, mva = mclean & (s <= 2023), mclean & (s == 2024)
    for lab, center, ref in (("raw (현재)", False, None),
                             ("centered (오라클 상수)", True, None),
                             ("centered (직전시즌 상수)", True, ref23)):
        Xtr = design(mtr, center, None if ref is None else None)
        Xva = design(mva, center, ref)
        b = fit_logistic(Xtr, y[mtr])
        pv = 1 / (1 + np.exp(-(np.column_stack([np.ones(len(Xva)), Xva]) @ b)))
        # prior shift 로 레벨을 맞춘 뒤 비교 — 레벨이 아니라 **방향성**을 재는 것이다
        from validation import shift_to_prior
        ps = shift_to_prior(pv, y[mva].mean())
        print(f"  {lab:<28} BSS(레벨보정 전) {bss(y[mva], pv):>8.1f}"
              f"   BSS(레벨보정 후) {bss(y[mva], ps):>8.1f}")

    print(f"\n{'='*96}\n■ 읽는 법\n{'='*96}")
    print("  B 의 '비(감쇠)' 가 1 이면 제안은 무효다 — 섞여도 기울기가 안 죽는다는 뜻.")
    print("     1 보다 크면 감쇠가 실재하고, 그 크기가 중심화로 되찾을 수 있는 몫의 상한이다.")
    print("  C 가 시즌 간 거의 같은 행이면 '평행이동' — 중심화 하나로 정렬된다.")
    print("     모양이 다르면 중심화만으로는 부족하고 scale 까지 맞춰야 한다.")
    print("  D 의 '레벨보정 후' 열이 판정이다. 중심화가 raw 를 이겨야 GBDT 재학습을 간다.")


if __name__ == "__main__":
    main()
