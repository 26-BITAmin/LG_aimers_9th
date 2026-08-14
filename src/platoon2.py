"""플래툰 δ — 낙관 편향을 걷어내고 다시 잰다

`platoon.py`의 T3은 **β와 k를 그 fold에서 argmax로 골랐다.** 그건 상한이지 실적이 아니다.
여기서는 셋을 고친다.

  ① k를 원칙으로 고정한다 — `SMOOTH_K_PITCHER = 200`. 배포 스무딩과 같은 상수이고
     이 fold를 보고 고른 값이 아니다.
  ② β 곡선을 **양 fold 다 찍어서** 공통 β를 눈으로 고른다. fold에서 argmax하지 않는다.
  ③ 대조군을 넣는다 — **리그 평균 플래툰**(투수별이 아닌 손잡이 주효과).
     모델은 이미 `batter_hand`·`hand_matchup`·`hand_same`을 갖고 있으므로 주효과는
     써먹고 있어야 정상이다. δ_pitcher의 이득이 사실은 주효과였다면 여기서 드러난다.

추가로 배포 전 확인해야 할 것 두 가지를 같이 잰다.
  · δ의 평균 — §6 버그 2(참조 분포 불일치)의 크기를 가늠한다. 2024 참조행은 δ 적중이
    ~100%인데 2025 추론행은 ~80%다. δ 평균이 0에 가까우면 레벨 영향이 무시할 만하다.
  · 적중률과 미적중 행의 특성 (신인 = δ 0)

실행:  .venv/Scripts/python.exe -u src/platoon2.py 2>&1 | tee experiments/platoon2.log
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
from config import MLP_ALPHA, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
K = SMOOTH_K_PITCHER                      # 200. 배포 상수 — fold를 보고 고르지 않았다
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]


def build_delta(pid_tr, bh_tr, y_tr, k):
    """train에서만 fit. (pitcher, hand) → δ.  δ = 수축((셀 성공률) − (투수 성공률))"""
    key = pid_tr * 8 + bh_tr
    u, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    t = np.bincount(inv, weights=y_tr)
    upid, invp = np.unique(pid_tr, return_inverse=True)
    n_p = np.bincount(invp).astype(np.float64)
    rate_p = np.bincount(invp, weights=y_tr) / n_p
    rp = rate_p[np.searchsorted(upid, u // 8)]
    return u, (n / (n + k)) * (t / n - rp)


def lookup(u, d, pid, bh):
    kv = pid * 8 + bh
    j = np.clip(np.searchsorted(u, kv), 0, len(u) - 1)
    hit = u[j] == kv
    return np.where(hit, d[j], 0.0), hit


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    pid = df["pitcher_id"].to_numpy(dtype=np.int64)
    bh = df["batter_hand"].to_numpy(dtype=np.int64)
    cached = foldcache.load(FOLDS, verbose=False)

    curves = {}
    print(f"k = {K} (배포 SMOOTH_K_PITCHER와 같은 값, fold 무관하게 고정)")
    for fold in FOLDS:
        tr, va = s < fold, s == fold
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        base = bss(yv, shift_to_prior(1.0 / (1.0 + np.exp(-z)), yv.mean()))

        u, d = build_delta(pid[tr], bh[tr], y[tr], K)
        d_row, hit = lookup(u, d, pid[va], bh[va])

        # 대조군: 리그 평균 플래툰 (투수별이 아님)
        lg = {}
        for h in np.unique(bh[tr]):
            lg[h] = y[tr][bh[tr] == h].mean() - y[tr].mean()
        d_lg = np.array([lg[h] for h in bh[va]])

        print("\n" + "=" * 88)
        print(f"fold {fold}   기준 BSS {base:.1f}   적중 {hit.mean():.1%}"
              f"   δ 평균 {d_row.mean():+.5f}   δ 산포 {d_row.std():.5f}")
        print(f"  미적중 행(신인 등) {int((~hit).sum()):,}개 — 이들은 δ=0")
        print(f"  리그 평균 플래툰: " + "  ".join(f"hand{h} {v:+.4f}" for h, v in lg.items()))
        print("=" * 88)
        print(f"  {'β':>6} " + " ".join(f"{b:>7.2f}" for b in BETAS))
        for nm, dd in (("투수별 δ", d_row), ("리그평균(대조)", d_lg)):
            row = []
            for b in BETAS:
                sc = bss(yv, shift_to_prior(
                    1.0 / (1.0 + np.exp(-(z + b * dd))), yv.mean()))
                row.append(sc - base)
            curves.setdefault(nm, {})[fold] = row
            print(f"  {nm:>14} " + " ".join(f"{v:>+7.2f}" for v in row))

    print("\n" + "=" * 88)
    print("공통 β 판정 — 양 fold에서 동시에 양수인 구간과 그 크기비")
    print("=" * 88)
    for nm, byf in curves.items():
        a, b = np.array(byf[FOLDS[0]]), np.array(byf[FOLDS[1]])
        print(f"\n  [{nm}]")
        print(f"  {'β':>6} {'f2023':>9} {'f2024':>9} {'최소':>9} {'크기비':>8}")
        for i, be in enumerate(BETAS):
            if be == 0:
                continue
            lo = min(a[i], b[i])
            ratio = (max(abs(a[i]), abs(b[i])) / max(min(abs(a[i]), abs(b[i])), 1e-9))
            mark = "  ✓" if (a[i] > 0 and b[i] > 0 and ratio <= 3) else ""
            print(f"  {be:>6.2f} {a[i]:>+9.2f} {b[i]:>+9.2f} {lo:>+9.2f} {ratio:>8.2f}{mark}")

    print("\n  ✓ = 양 fold 양수 + 크기비 ≤ 3 (조건 ②③ 동시 통과)")
    print("  대조군(리그평균)이 0 근처여야 '투수별'이 진짜다 —")
    print("  모델은 이미 batter_hand·hand_matchup을 갖고 있으므로 주효과는 써먹고 있어야 한다.")


if __name__ == "__main__":
    main()
