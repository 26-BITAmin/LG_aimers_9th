"""Competition metrics and probability diagnostics."""

from __future__ import annotations

import numpy as np


def _validate_probability_inputs(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true={y.shape}, y_prob={p.shape}")
    if y.ndim != 1:
        raise ValueError(f"expected one-dimensional inputs, got ndim={y.ndim}")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("y_true and y_prob must not contain NaN or infinity")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("y_prob must be between 0 and 1")
    return y, p


def brier_score(y_true, y_prob) -> float:
    """Mean squared probability error used by the competition."""
    y, p = _validate_probability_inputs(y_true, y_prob)
    return float(np.mean(np.square(p - y)))


def raw_brier_skill_score(y_true, y_prob) -> float:
    """Unclipped DACON score; negative values are useful for diagnostics."""
    y, p = _validate_probability_inputs(y_true, y_prob)
    rate = float(y.mean())
    reference = rate * (1.0 - rate)
    if reference <= 0.0:
        raise ValueError("Brier Skill Score is undefined when the target is constant")
    return float(100_000.0 * (1.0 - np.mean(np.square(p - y)) / reference))


def brier_skill_score(y_true, y_prob) -> float:
    """Official DACON score, clipped at zero."""
    return max(0.0, raw_brier_skill_score(y_true, y_prob))
