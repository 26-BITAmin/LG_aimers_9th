"""점수 변환 + 검증 지표 — EDA_정리.md §2, CLAUDE.md 불변식과 동일.

Score = max(0, 1e5 * (1 - Brier / (r*(1-r))))   (r = 검증 집합 실제 base rate)
pts_res = 오라클 절편 보정 후 점수 (해상도만, 절편 잡음 제거)
ΔBS 1e-4 ≈ 40점

절대 규칙: 점수는 리더보드 환산으로만 비교. 원시 Brier는 폴드마다 r이 달라 비교 불가.
"""
import numpy as np
import pandas as pd


def brier(p, y):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def score(p, y, clip=True):
    """리더보드 환산 점수. clip=True면 대회 산식과 동일하게 max(0, ...) 적용."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    r = y.mean()
    den = r * (1 - r)
    bs = brier(p, y)
    s = 1e5 * (1 - bs / den)
    return float(max(0.0, s)) if clip else float(s)


def score_pair(p, y):
    """pts_res(오라클 절편 보정=해상도) / pts_raw(클리핑 없는 실제 점수) / bias."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    r = y.mean()
    den = r * (1 - r)
    bias = p.mean() - r
    pts_raw = 1e5 * (1 - np.mean((p - y) ** 2) / den)
    pts_res = 1e5 * (1 - np.mean((p - bias - y) ** 2) / den)
    return dict(pts_res=float(pts_res), pts_raw=float(pts_raw), bias=float(bias), r=float(r))


def pitcher_cluster_bootstrap(p_a, p_b, y, pitcher_id, n_boot=300, seed=0, clip=True):
    """짝지은 Δ(score(p_a) - score(p_b))의 투수 단위 클러스터 부트스트랩 CI.

    i.i.d. 부트스트랩은 10배 이상 낙관적이므로 반드시 pitcher_id 단위로 리샘플.
    """
    rng = np.random.default_rng(seed)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    y = np.asarray(y, dtype=float)
    pitcher_id = np.asarray(pitcher_id)

    df = pd.DataFrame({"pid": pitcher_id, "p_a": p_a, "p_b": p_b, "y": y})
    groups = {pid: g[["p_a", "p_b", "y"]].to_numpy() for pid, g in df.groupby("pid")}
    pids = np.array(list(groups.keys()))
    n = len(pids)

    base_delta = score(p_a, y, clip=clip) - score(p_b, y, clip=clip)

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        sample_pids = rng.choice(pids, size=n, replace=True)
        stacked = np.concatenate([groups[pid] for pid in sample_pids], axis=0)
        deltas[i] = (score(stacked[:, 0], stacked[:, 2], clip=clip)
                     - score(stacked[:, 1], stacked[:, 2], clip=clip))

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return dict(delta=float(base_delta), ci_lo=float(lo), ci_hi=float(hi), n_pitchers=int(n))
