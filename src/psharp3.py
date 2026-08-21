"""β의 월별 하강은 **달력 때문인가, 올해분 누적 때문인가** (사용자 질문).

§7.105 ⑥이 β(시즌 시기)가 1.53 → 1.05 로 떨어지는 것을 보였다. 사용자 지적:

    "이때는 시즌 정보 자체를 안 줬는데 왜 이렇게 나오는 거지. month도 없었을 텐데"

맞다 — `season`·`game_month` 둘 다 `DROP_FEATURES` 다. 모델은 달력을 못 본다.
그런데도 β가 움직인다면 원인은 **모델이 아니라 입력**이어야 한다:

    `asof_*` 는 커리어 누적이고 **이번 시즌 진행분이 계속 더해진다.**
    4월 행의 asof 는 사실상 작년까지의 커리어, 9월 행은 올해 5개월치가 녹아 있다.
    (§7.55 의 `gap_success` 가 "커리어 − 시즌시작상수 = 올해분" 으로 만들어진 것 자체가
     asof 가 시즌 내에서 갱신된다는 증거다.)

검정 가능한 예측 두 개
    P1  β 는 달력 월이 아니라 **그 투수의 올해 투구 수 `n_szn`** 을 따라 떨어져야 한다.
    P2  `n_szn` 을 고정하면 **월별 차이가 사라져야 한다.**
        → 남으면 달력에 다른 것이 있다 (9월 확장 엔트리·순위 경쟁 등).

`n_szn` 은 §7.55 와 같은 방식으로 복원한다: 그 (투수, 시즌)의 `asof_pitcher_n` 최솟값을
시즌 시작 상수로 보고 빼면 그 행 시점까지의 **올해 투구 수**가 된다.
⚠️ 이건 **진단 전용**이다 (train 전체 groupby). 배포 피처가 아니다.

실행:  .venv/Scripts/python.exe -u src/psharp3.py 2>&1 | tee experiments/psharp3.log
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

import jointbeta as J
from config import ASOF_AXES, CTX_AXES, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = [2021, 2022, 2023, 2024]
NB = [(0, 50), (50, 200), (200, 500), (500, 1200), (1200, 10 ** 9)]
NLAB = ["0~50", "50~200", "200~500", "500~1200", "1200+"]
MB = [(3, 4), (5, 6), (7, 8), (9, 12)]
MLAB = ["3~4월", "5~6월", "7~8월", "9월~"]


def resid_t_cluster(r, x, cl, n_id):
    """투수 클러스터 강건 t (CR0). 기울기는 OLS 와 같고 t 만 바뀐다."""
    xc = x - x.mean()
    den = float((xc ** 2).sum())
    if den <= 0 or len(r) < 50:
        return np.nan, 0.0
    sl = float((xc * r).sum() / den)
    e = r - r.mean() - sl * xc
    g = np.bincount(cl, weights=xc * e, minlength=n_id)
    v = float((g ** 2).sum()) / den ** 2
    return sl, (sl / np.sqrt(v) if v > 0 else 0.0)


def beta_of(mask, codes, y, pmod, n_id):
    """그 부분집합 안에서 m_j 를 다시 계산해 β = 1 + slope[resid ~ m_j] 를 낸다."""
    cf, yy, pp = codes[mask], y[mask], pmod[mask]
    n = np.bincount(cf, minlength=n_id).astype(float)
    P = np.bincount(cf, weights=pp, minlength=n_id)
    mj = np.where(n > 0, P / np.maximum(n, 1) - pp.mean(), 0.0)
    sl, t = resid_t_cluster(yy - pp, mj[cf], cf, n_id)
    return 1 + sl, t, int(mask.sum())


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    gm = df["game_month"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    codes, uniq = pd.factorize(df["pitcher_id"].to_numpy())
    n_id = len(uniq)
    mclean = ~((gt == "F") & (s < 2023))

    # ── n_szn 복원 (§7.55 와 같은 방식. 진단 전용) ───────────────────────
    an = df["asof_pitcher_n"].to_numpy(dtype=np.float64)
    key = pd.Series(codes.astype(np.int64) * 10000 + s.astype(np.int64))
    base = pd.Series(an).groupby(key).transform("min").to_numpy()
    n_szn = np.nan_to_num(an - base, nan=0.0)
    print(f"train {df.shape}   n_szn 복원: 중앙값 {np.median(n_szn[mclean]):.0f}"
          f"  90%p {np.percentile(n_szn[mclean], 90):.0f}")

    # ── 배포 홀드아웃 예측 ──────────────────────────────────────────────
    cached = J.foldcache.load(FOLDS, verbose=False)
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    pmod = np.full(len(df), np.nan)
    for f in FOLDS:
        X, _, _, _ = J.axis_matrix(df, s, f)
        yv = cached[f][2]
        vi = np.where(s == f)[0]
        assert len(vi) == len(yv) and np.allclose(y[vi], yv)
        mm = mclean[vi]
        zb = (1 - MLP_ALPHA) * cached[f][0] + MLP_ALPHA * cached[f][1]
        zd = (zb + X @ dep)[mm]
        p_ = np.full(len(vi), np.nan)
        p_[mm] = shift_to_prior(1 / (1 + np.exp(-zd)), yv[mm].mean())
        pmod[vi] = p_
    mmod = mclean & np.isfinite(pmod)
    print(f"홀드아웃 clean {int(mmod.sum()):,}행   (MLP_ALPHA={MLP_ALPHA})\n")

    # ── P1 — n_szn 을 따라 떨어지는가 ───────────────────────────────────
    print(f"{'='*94}\n█ P1 — β vs **그 투수의 올해 투구 수 n_szn** (달력이 아니라)\n{'='*94}")
    print(f"  {'fold':<7}" + "".join(f"{l:>18}" for l in NLAB))
    print(f"  {'':7}" + "".join(f"{'β  (t) [n]':>18}" for _ in NLAB))
    for f in FOLDS:
        cells = []
        for lo, hi in NB:
            m = mmod & (s == f) & (n_szn >= lo) & (n_szn < hi)
            b, t, n = beta_of(m, codes, y, pmod, n_id)
            cells.append(f"{b:>6.2f} ({t:>4.1f}) {n//1000:>3}k" if n >= 5000 else f"{'—':>18}")
        print(f"  {f:<7}" + "".join(f"{c:>18}" for c in cells))

    # ── 비교군: 같은 코드로 달력 월 ─────────────────────────────────────
    print(f"\n{'='*94}\n█ (비교) β vs 달력 월 — §7.105 ⑥ 재현\n{'='*94}")
    print(f"  {'fold':<7}" + "".join(f"{l:>18}" for l in MLAB))
    for f in FOLDS:
        cells = []
        for lo, hi in MB:
            m = mmod & (s == f) & (gm >= lo) & (gm <= hi)
            b, t, n = beta_of(m, codes, y, pmod, n_id)
            cells.append(f"{b:>6.2f} ({t:>4.1f}) {n//1000:>3}k" if n >= 5000 else f"{'—':>18}")
        print(f"  {f:<7}" + "".join(f"{c:>18}" for c in cells))

    # ── P2 — n_szn 을 고정하면 월별 차이가 사라지는가 ────────────────────
    print(f"\n{'='*94}\n█ P2 — ★ **n_szn 고정** 후 월별 β  (판정 fold 2023·2024 합산)\n{'='*94}")
    print("  같은 올해 투구 수 안에서 달력만 다르게. 차이가 사라지면 원인은 누적이다.")
    print("  남으면 달력에 다른 것이 있다 (9월 확장 엔트리·순위 경쟁 등).\n")
    print(f"  {'n_szn 밴드':<12}" + "".join(f"{l:>18}" for l in MLAB))
    for (lo, hi), lab in zip(NB, NLAB):
        cells = []
        for mlo, mhi in MB:
            m = mmod & np.isin(s, [2023, 2024]) & (n_szn >= lo) & (n_szn < hi) \
                & (gm >= mlo) & (gm <= mhi)
            b, t, n = beta_of(m, codes, y, pmod, n_id)
            cells.append(f"{b:>6.2f} ({t:>4.1f}) {n//1000:>3}k" if n >= 5000 else f"{'—':>18}")
        print(f"  {lab:<12}" + "".join(f"{c:>18}" for c in cells))

    # ── 교차 확인: 월 밴드 안에서 n_szn 분포 ────────────────────────────
    print(f"\n{'='*94}\n■ 교차 확인 — 월 밴드별 n_szn 중앙값 (두 축이 얼마나 겹치나)\n{'='*94}")
    print(f"  {'fold':<7}" + "".join(f"{l:>12}" for l in MLAB))
    for f in FOLDS:
        row = []
        for mlo, mhi in MB:
            m = mmod & (s == f) & (gm >= mlo) & (gm <= mhi)
            row.append(f"{np.median(n_szn[m]):>12.0f}" if m.sum() else f"{'—':>12}")
        print(f"  {f:<7}" + "".join(row))
    print("\n  두 축이 강하게 겹치면(단조 증가) P2 의 칸이 얇아진다 — n 을 같이 보고 읽어야 한다.")


if __name__ == "__main__":
    main()
