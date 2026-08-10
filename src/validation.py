"""Brier / BSS · 시간 기반 split · prior shift.

대회 지표:
    BS        = mean((p - y)^2)
    BS_ref    = p̄(1-p̄)          p̄ = **평가셋의 실제 평균 제구율**
    Score     = max(0, 100000 * (1 - BS/BS_ref))

Brier 분해가 이 대회의 전부다:
    BS = BS_ref + (mean(p) - p̄)^2 + [refinement/calibration 잔여항]
    → 예측 평균이 실제 평균에서 0.01 벗어나면 그것만으로 BSS 약 -40점.
"""
import numpy as np

from config import CLIP_EPS


def brier(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), CLIP_EPS, 1 - CLIP_EPS)
    return float(np.mean((p - np.asarray(y, dtype=np.float64)) ** 2))


def bss(y, p):
    """대회 규칙 그대로 — 기준선은 평가셋 자신의 평균."""
    y = np.asarray(y, dtype=np.float64)
    ref = y.mean() * (1 - y.mean())
    return 100000.0 * (1 - brier(y, p) / ref)


def logloss(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), CLIP_EPS, 1 - CLIP_EPS)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p):
    y = np.asarray(y)
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def decompose(y, p):
    """Brier를 해석 가능한 성분으로 쪼갠다. 어디서 지고 있는지 봐야 다음 수가 보인다."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    pbar = y.mean()
    return {
        "brier": brier(y, p),
        "ref": pbar * (1 - pbar),
        "mean_shift_loss": float((p.mean() - pbar) ** 2),   # 레벨(prior) 오차
        "pred_mean": float(p.mean()),
        "true_mean": float(pbar),
        "pred_std": float(p.std()),
    }


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), CLIP_EPS, 1 - CLIP_EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def shift_to_prior(p, target_mean, tol=1e-10, max_iter=200):
    """logit(p') = logit(p) + delta 를 mean(p') == target_mean 이 되도록 푼다.

    확률을 상수배 하거나 잘라내지 않고 **평행이동**만 하므로 순위가 보존되고
    0/1로 밀려나지 않는다 (Brier에서 극단값은 비싸다).
    """
    z = _logit(p)
    lo, hi = -10.0, 10.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        m = _sigmoid(z + mid).mean()
        if abs(m - target_mean) < tol:
            break
        if m < target_mean:
            lo = mid
        else:
            hi = mid
    return _sigmoid(z + mid)


def reliability(y, p, n_bins=10):
    """신뢰도 표 — 예측 확률 구간별 실제 양성률."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append((b, int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return rows


def report(name, y, p, extra=""):
    d = decompose(y, p)
    print(f"  {name:<34s} Brier={d['brier']:.6f}  BSS={bss(y, p):8.1f}  "
          f"AUC={auc(y, p):.4f}  pred_mean={d['pred_mean']:.4f} "
          f"(true {d['true_mean']:.4f}, level_loss={d['mean_shift_loss']:.6f}) {extra}")
    return d
