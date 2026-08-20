"""투수 신호의 **과전개(over-extension)** 검정 — Marcel의 2단계가 빠져 있는가.

`erafit.py`가 예상과 반대 방향의 것을 찾았다.

    잔차 회귀 T2 기울기가 **전부 음수**다 (f2023 t=−15.0, f2024 t=−5.55).
    즉 "과거가 좋다고 말하는 투수"에서 배포 모델이 **과대예측**한다.
    → 모델이 투수 신호를 **못 쓰고 있는 게 아니라 너무 믿고 있다.**

이건 정확히 Marcel의 2단계다. Marcel은 (1) 최근 3시즌 가중 평균 (2) **리그
평균으로 회귀(regression to the mean)** (3) 나이 보정으로 되어 있고, 투수는
134 아웃(≈44이닝)만큼 리그로 당긴다. 우리 파이프라인에는 (1)의 근사(`asof_*`)는
있지만 **(2)가 없다** — `SMOOTH_K`로 스무딩한 컬럼을 *원본과 나란히* 넣고
"모델이 고르게" 했더니, 모델이 **덜 수축하는 쪽**을 골랐다.

왜 모델이 틀리는가 (도메인)
---------------------------
학습 시즌 안에서는 투수 정체성이 안정적이라 `asof_*`의 지속성이 높게 보인다.
그런데 **한 시즌 앞으로 나가면** 그 지속성이 떨어진다 (부상·구속 저하·역할 변경·
나이). 학습 손실은 이 갭을 볼 수 없다 — 학습 분포 안에 미래 시즌이 없기 때문이다.
[[learned-feature-selection-blind-to-drift]]와 같은 실패 구조다.

무엇을 재는가
-------------
  ① 음수 β 스윕.  실제 BSS 이득 (선형 근사가 아니라 실측)
  ② **asof_n 버킷별** T2.  과전개가 cold-start 구간에 몰려 있는가.
     몰려 있다면 2025에서 **더 커진다** (신규 투수 비중 2023 13.8% → 2024 19.9% → 2025 ↑)
  ③ 배포 가능한 형태 — 행 단위 `asof` 편차에 직접 거는 수축.
     새 아티팩트 0개, 조회 0개, 순수 행 단위 변환이다.
  ④ 대조군 — 전역 온도 (§7.19에서 기각됨). 투수축 수축이 그것과 **다른 것**임을 보인다.

실행:  .venv/Scripts/python.exe -u src/erashrink.py 2>&1 | tee experiments/erashrink.log
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
from config import CTX_AXES, CTX_K, MLP_ALPHA, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV
from ctxsplit import build_delta, lookup, resid_t
from data import load_csv
from erafit import (MARCEL_W, league_rates, pitcher_estimate, season_table)
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
K = CTX_K
BETAS = [-1.5, -1.0, -0.75, -0.5, -0.35, -0.25, -0.15, 0.0, +0.25]
N_BUCKETS = [0, 200, 1000, 3000, 8000, 20000, 10 ** 9]


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    pid = df["pitcher_id"].to_numpy(dtype=np.int64)
    an = df["asof_pitcher_n"].to_numpy(dtype=np.float64)
    ar = df["asof_pitcher_success_rate"].to_numpy(dtype=np.float64)
    bh = df["batter_hand"].to_numpy(dtype=np.int64)
    b = df["balls_before"].to_numpy(dtype=np.int64)
    st = df["strikes_before"].to_numpy(dtype=np.int64)
    c4 = (b >= 3).astype(np.int64) + 2 * (st >= 2).astype(np.int64)

    cached = foldcache.load(FOLDS, verbose=False)
    print(f"train {df.shape}   배포 SMOOTH_K_PITCHER={SMOOTH_K_PITCHER}\n")

    summary = {}

    for fold in FOLDS:
        tr, va = s < fold, s == fold
        seasons = sorted(int(t) for t in np.unique(s[tr]))
        lg = league_rates(s[tr], y[tr], seasons)
        lg_tr = float(y[tr].mean())
        upit = np.unique(pid[tr])
        ntab, rtab = season_table(pid[tr], s[tr], y[tr], seasons)

        w_mar = [0.0] * len(seasons)
        for i, wv in enumerate(MARCEL_W):
            if len(seasons) - 1 - i >= 0:
                w_mar[len(seasons) - 1 - i] = wv

        d_del_car, _ = pitcher_estimate(upit, ntab, rtab, seasons, lg,
                                        [1.0] * len(seasons), True)
        d_del_mar, _ = pitcher_estimate(upit, ntab, rtab, seasons, lg, w_mar, True)

        jv = np.clip(np.searchsorted(upit, pid[va]), 0, len(upit) - 1)
        hit = upit[jv] == pid[va]

        # ── 배포 상태 기준선 ──────────────────────────────────────
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        z0 = z.copy()
        for name, beta in CTX_AXES:
            code, m = (bh, 8) if name == "hand" else (c4, 4)
            u, d, _ = build_delta(pid[tr], code[tr], y[tr], m, K)
            drow, _ = lookup(u, d, pid[va], code[va], m)
            z0 = z0 + beta * drow
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean())
        base = bss(yv, p0)
        r = yv - p0

        # ── 회귀자들 ─────────────────────────────────────────────
        nv = an[va]
        arv = np.where(np.isnan(ar[va]), lg_tr, ar[va])
        # 배포 파이프라인이 실제로 쓰는 스무딩과 같은 식 (features._smooth)
        sm = (nv * arv + SMOOTH_K_PITCHER * lg_tr) / (nv + SMOOTH_K_PITCHER)

        regs = {
            "del_career":  np.where(hit, d_del_car[jv], 0.0),
            "del_marcel":  np.where(hit, d_del_mar[jv], 0.0),
            # ↓ 배포 가능한 형태: 행 단위. 조회도 아티팩트도 없다.
            "asof_dev":    sm - lg_tr,
            # 대조군 — 전역 온도 (§7.19 기각). z0 자체를 수축한다
            "global_temp": z0 - z0.mean(),
        }

        print("=" * 104)
        print(f"fold {fold}   valid {va.sum():,}   기준 BSS(배포상태) {base:.1f}"
              f"   투수 적중 {hit.mean():.1%}")
        print("=" * 104)

        print("\n  [1] 잔차 회귀 — 음수 = 모델이 그 신호를 **과대 사용**")
        print(f"      {'회귀자':<14}{'산포':>9}{'기울기':>11}{'t':>9}")
        for tag, v in regs.items():
            sl, t2 = resid_t(r, v)
            print(f"      {tag:<14}{v.std():>9.5f}{sl:>+11.4f}{t2:>+9.2f}")

        print(f"\n  [2] asof_n 버킷별 과전개 — cold-start에 몰려 있는가"
              f"  (회귀자 = asof_dev)")
        print(f"      {'구간':<16}{'행수':>10}{'비중':>7}{'기울기':>11}{'t':>9}{'잔차평균':>10}")
        for lo, hi in zip(N_BUCKETS[:-1], N_BUCKETS[1:]):
            m = (nv >= lo) & (nv < hi)
            if m.sum() < 500:
                continue
            sl, t2 = resid_t(r[m], regs["asof_dev"][m])
            lab = f"{lo:,}~{hi:,}" if hi < 10 ** 9 else f"{lo:,}+"
            print(f"      {lab:<16}{m.sum():>10,}{m.mean():>7.1%}"
                  f"{sl:>+11.4f}{t2:>+9.2f}{r[m].mean():>+10.4f}")

        print("\n  [3] β 스윕 — z0 + β·회귀자 의 BSS 이득 (음수 β = 수축)")
        print(f"      {'회귀자':<14}" + "".join(f"{x:<8.2f}" for x in BETAS))
        for tag, v in regs.items():
            g = []
            for beta in BETAS:
                p = shift_to_prior(1.0 / (1.0 + np.exp(-(z0 + beta * v))), yv.mean())
                g.append(bss(yv, p) - base)
            summary.setdefault(tag, {})[fold] = g
            print(f"      {tag:<14}" + "".join(f"{val:>+8.2f}" for val in g))
        print()

    print("=" * 104)
    print("요약 — 조건 ②(부호일치) ③(크기비 ≤3)")
    print("=" * 104)
    print(f"  {'회귀자':<14}" + "".join(f"β{x:<11.2f}" for x in BETAS))
    for tag, per in summary.items():
        cells = ""
        for i in range(len(BETAS)):
            a, bb = per[FOLDS[0]][i], per[FOLDS[1]][i]
            ok = a > 0 and bb > 0
            ratio = max(abs(a), abs(bb)) / max(min(abs(a), abs(bb)), 1e-9)
            cells += f"{a:>+5.1f}/{bb:>+5.1f}{'✓' if ok and ratio <= 3 else ('~' if ok else ' ')}"
        print(f"  {tag:<14}{cells}")
    print("\n  ✓ = 양 fold 양수 이고 크기비 ≤3    ~ = 부호만 일치")


if __name__ == "__main__":
    main()
