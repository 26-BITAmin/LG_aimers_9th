"""투수별 플래툰 스플릿 — **만들기 전에 전이되는지부터 잰다**

동기 (MODEL.md §7.34 G)
    투수 × 타자손잡이 천장 1127.9 vs 투수 단독 838.8 → 상호작용에 in-season으로
    289점이 있다. 그리고 그 축의 **전이**는 투수 단독보다 175점 높다 (81.2 vs −93.5).
    `asof_*`에는 손잡이별 분할이 아예 없으므로 이 정보는 모델에 들어간 적이 없다.
    §7.15도 "손잡이 4개가 제거 시 −96.9로 1위, 명시적 상호작용을 만든다면 거기서
    시작한다"고 적어두고 실행하지 않았다.

이 파일은 피처를 만들지 않는다. 세 가지만 잰다.

    T1  스플릿이 시즌을 넘어 유지되나
        과거로 추정한 δ(p,h)와 검증 시즌의 실제 δ가 상관되는가 (셀 단위, 가중)

    T2  **배포 모델의 잔차와 상관되나** ← 이게 결정적이다
        모델이 이미 다 쓰고 있으면 잔차와 무상관이다. 남아 있으면 상관이 있다.
        캐시된 z를 쓰므로 학습이 필요 없다.

    T3  로짓에 β·δ를 더하면 실제로 BSS가 오르나 (β 스윕)

    T2·T3이 0이면 천장 1127.9는 **in-season 오라클로만 닿는 값**이고
    이 축은 규칙상 닫힌다. 그러면 피처를 만들지 않는다.

δ의 정의 (전부 train에서만 fit → 검증에 적용. 리크 없음)
    rate_ph = 투수 p가 손잡이 h 타자에게 낸 제구 성공률 (train)
    rate_p  = 투수 p의 전체 성공률 (train)
    δ(p,h)  = n_ph/(n_ph+k) · (rate_ph − rate_p)      ← 경험적 베이즈 수축
    미지 투수 → 0

실행:  .venv/Scripts/python.exe -u src/platoon.py 2>&1 | tee experiments/platoon.log
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
KS = [200.0, 1000.0, 5000.0]
BETAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
MIN_CELL = 100


def cell_stats(pid, hand, y, n_pitch):
    """(pitcher, hand) 셀과 pitcher 단위 합계를 한 번에."""
    key = pid.astype(np.int64) * 8 + hand.astype(np.int64)
    u, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    t = np.bincount(inv, weights=y)
    return u, n, t


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    pid = df["pitcher_id"].to_numpy(dtype=np.int64)
    bh = df["batter_hand"].to_numpy(dtype=np.int64)
    print(f"train {df.shape}   batter_hand 값 {sorted(set(bh.tolist()))}")

    cached = foldcache.load(FOLDS, verbose=False)

    for fold in FOLDS:
        tr, va = s < fold, s == fold
        print("\n" + "=" * 92)
        print(f"fold {fold}   train {tr.sum():,}행 → valid {va.sum():,}행")
        print("=" * 92)

        # ── train에서 δ를 만든다 ────────────────────────────────
        ut, nt, tt = cell_stats(pid[tr], bh[tr], y[tr], None)
        up, np_, tp = np.unique(pid[tr], return_counts=True), None, None
        upid, invp = np.unique(pid[tr], return_inverse=True)
        n_p = np.bincount(invp).astype(np.float64)
        t_p = np.bincount(invp, weights=y[tr])
        rate_p = t_p / n_p                              # 투수 전체 성공률
        # 셀 → 그 투수의 전체 성공률
        cell_pid = ut // 8
        pos = np.searchsorted(upid, cell_pid)
        rate_p_of_cell = rate_p[pos]
        rate_ph = tt / nt

        # ── valid 쪽 실제 δ (참값) ──────────────────────────────
        uv, nv, tv = cell_stats(pid[va], bh[va], y[va], None)
        v_pid = uv // 8
        upv, invpv = np.unique(pid[va], return_inverse=True)
        nvp = np.bincount(invpv).astype(np.float64)
        tvp = np.bincount(invpv, weights=y[va])
        rate_pv = tvp / nvp
        rate_pv_of_cell = rate_pv[np.searchsorted(upv, v_pid)]
        delta_actual = tv / nv - rate_pv_of_cell

        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        p = shift_to_prior(1.0 / (1.0 + np.exp(-z)), yv.mean())
        assert len(yv) == va.sum()
        r = yv - p                                       # 배포 모델 잔차

        for k in KS:
            raw = rate_ph - rate_p_of_cell
            delta = (nt / (nt + k)) * raw                # 수축된 δ

            # valid 행에 매핑 (미지 셀 → 0)
            kv = pid[va] * 8 + bh[va]
            j = np.searchsorted(ut, kv)
            jc = np.clip(j, 0, len(ut) - 1)
            hit = ut[jc] == kv
            d_row = np.where(hit, delta[jc], 0.0)

            # ── T1. 셀 단위 상관 (검증 셀 100투구 이상만) ──────
            j2 = np.searchsorted(ut, uv)
            j2c = np.clip(j2, 0, len(ut) - 1)
            h2 = (ut[j2c] == uv) & (nv >= MIN_CELL)
            if h2.sum() >= 20:
                w = nv[h2]
                a, b = delta[j2c][h2], delta_actual[h2]
                am, bm = np.average(a, weights=w), np.average(b, weights=w)
                cov = np.average((a - am) * (b - bm), weights=w)
                c1 = cov / np.sqrt(np.average((a - am) ** 2, weights=w)
                                   * np.average((b - bm) ** 2, weights=w))
                slope = cov / np.average((a - am) ** 2, weights=w)
            else:
                c1, slope = float("nan"), float("nan")

            # ── T2. 잔차와의 관계 ──────────────────────────────
            dm = d_row.mean()
            dc = d_row - dm
            den = float((dc ** 2).sum())
            sl = float((dc * r).sum() / den) if den > 0 else 0.0
            res = r - r.mean() - sl * dc
            se = float(np.sqrt((res ** 2).sum() / (len(r) - 2) / den)) if den > 0 else 0.0

            # ── T3. 로짓에 더해본다 ────────────────────────────
            best, bb = -1e18, None
            for beta in BETAS:
                sc = bss(yv, shift_to_prior(
                    1.0 / (1.0 + np.exp(-(z + beta * d_row))), yv.mean()))
                if beta == 0.0:
                    base = sc
                if sc > best:
                    best, bb = sc, beta

            print(f"\n  k={k:<7.0f} 적중 {hit.mean():>6.1%}   "
                  f"δ 산포 {d_row.std():.5f}  (|δ|>0.02 인 행 {np.mean(np.abs(d_row)>0.02):>5.1%})")
            print(f"    T1  corr(δ_hist, δ_actual) = {c1:>+.3f}   기울기 {slope:>+.3f}"
                  f"   (셀 {int(h2.sum())}개)")
            print(f"    T2  잔차 회귀 기울기 {sl:>+.4f} ± {se:.4f}"
                  f"   (t {sl/se if se else 0:>+5.1f})")
            print(f"    T3  β*={bb}  BSS {base:.1f} → {best:.1f}   ({best-base:+.2f})")

    print("\n" + "=" * 92)
    print("판정 기준")
    print("=" * 92)
    print("  T1 corr이 0 근처면 스플릿 자체가 시즌을 안 넘는다 → 끝.")
    print("  T1은 살아 있는데 T2 t가 0 근처면 **모델이 이미 다 쓰고 있다** → 끝.")
    print("  T2 t가 크고 T3 이득이 양 fold에서 양수여야 피처를 만들 값이 있다.")


if __name__ == "__main__":
    main()
