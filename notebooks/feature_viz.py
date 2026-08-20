#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LG Aimers 9기 / DACON 236743 — 투구 제구 성공 확률 예측
per-feature 시각화 + 상관/PCA/KMeans 리포트 생성기

참고 구조: Kaggle "MLB Pitcher Analysys | Clustering & PCA" (sasakitetsuya)
  - 원본은 '선수 1명 = 1행' 시즌/커리어 집계 데이터
  - 본 대회 train.csv는 '투구 1개 = 1행' 이므로, PCA/클러스터링은
    투수 단위로 집계한 뒤에만 적용한다.

출력: feature_viz_report.html (자체 완결, 이미지 base64 임베드)

실행:
    python feature_viz.py --data-dir ../data --out feature_viz_report.html
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import os
import sys
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

SEED = 42
TARGET = "control_success"

# ---------------------------------------------------------------- style
for cand in ("Noto Sans CJK JP", "NanumGothic", "Malgun Gothic", "AppleGothic"):
    try:
        matplotlib.font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.family"] = cand
        break
    except Exception:
        continue
plt.rcParams.update({
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c9ccd1",
    "axes.labelcolor": "#2b2f36",
    "axes.titlesize": 10.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": "#5a6070",
    "ytick.color": "#5a6070",
    "axes.grid": True,
    "grid.color": "#e8eaee",
    "grid.linewidth": 0.7,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})

C_BAR = "#9aa4b4"       # 분포(맥락)
C_LINE = "#2f6fdb"      # 성공률(주 신호)
C_FILL = "#c3d6f6"
C_REF = "#e4573d"       # 기준선/경고
C_ALT = "#2ea36b"
PALETTE = ["#2f6fdb", "#e4573d", "#2ea36b", "#b072dd", "#e0a12c",
           "#3ab3c9", "#d1568f", "#7d8794"]


# ---------------------------------------------------------------- utils
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fig_to_b64(fig, dpi: int = 100, quantize: bool = True) -> str:
    """figure -> base64 png (팔레트 축소로 리포트 용량 절감)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    if quantize:
        try:
            from PIL import Image
            im = Image.open(buf).convert("RGB").quantize(colors=128, method=2)
            out = io.BytesIO()
            im.save(out, format="PNG", optimize=True)
            buf = out
            buf.seek(0)
        except Exception:
            buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    """이항 비율의 Wilson score 구간. n이 작은 구간의 노이즈를 눈으로 걸러내기 위함."""
    n = np.maximum(n, 1)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return center - half, center + half


def _fmt_thousands(x, _pos):
    if x >= 1e6:
        return f"{x/1e6:.1f}M"
    if x >= 1e3:
        return f"{x/1e3:.0f}k"
    return f"{x:.0f}"


# ---------------------------------------------------------------- 단변량 BSS
def univariate_bss(df: pd.DataFrame, col: str, n_bins: int = 20,
                   m_smooth: float = 200.0,
                   fit_seasons=(2019, 2020, 2021, 2022, 2023),
                   eval_season: int = 2024) -> dict:
    """
    이 피처 하나만으로 예측했을 때의 Brier Skill Score.

    - fit: 2019~2023 구간에서 bin별 평균 성공률 (m-estimate 스무딩)
    - eval: 2024 시즌
    - baseline: fit 구간 평균 성공률을 상수로 예측
    → 시간 분할이므로 리키지 없음. 드리프트까지 반영된 '정직한' 단변량 신호량.
      대회 채점식과 동일하게 100000 * (1 - BS_model / BS_ref).
    """
    tr_mask = df["season"].isin(fit_seasons)
    ev_mask = df["season"] == eval_season
    if ev_mask.sum() == 0 or tr_mask.sum() == 0:
        return {"bss": np.nan, "n_eval": 0}

    s_tr, y_tr = df.loc[tr_mask, col], df.loc[tr_mask, TARGET].to_numpy()
    s_ev, y_ev = df.loc[ev_mask, col], df.loc[ev_mask, TARGET].to_numpy()
    prior = float(y_tr.mean())

    if pd.api.types.is_numeric_dtype(s_tr) and s_tr.nunique(dropna=True) > 30:
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.nanquantile(s_tr.to_numpy(dtype="float64"), qs))
        if len(edges) < 3:
            return {"bss": np.nan, "n_eval": int(ev_mask.sum())}
        edges[0], edges[-1] = -np.inf, np.inf
        b_tr = pd.cut(s_tr, edges, labels=False)
        b_ev = pd.cut(s_ev, edges, labels=False)
    else:
        cats = s_tr.astype("object").where(s_tr.notna(), "__NaN__")
        codes, uniq = pd.factorize(cats)
        b_tr = pd.Series(codes, index=s_tr.index).replace(-1, np.nan)
        mapping = {v: i for i, v in enumerate(uniq)}
        b_ev = s_ev.astype("object").where(s_ev.notna(), "__NaN__").map(mapping)

    b_tr = b_tr.fillna(-1).astype("int32").to_numpy()
    b_ev = b_ev.fillna(-1).astype("int32").to_numpy()

    nb = max(b_tr.max(), b_ev.max()) + 2          # -1(=NaN/미관측) 을 0으로 shift
    cnt = np.bincount(b_tr + 1, minlength=nb).astype("float64")
    pos = np.bincount(b_tr + 1, weights=y_tr, minlength=nb).astype("float64")
    p_bin = (pos + m_smooth * prior) / (cnt + m_smooth)
    p_bin[cnt == 0] = prior

    pred = p_bin[np.clip(b_ev + 1, 0, nb - 1)]
    bs = float(np.mean((pred - y_ev) ** 2))
    bs_ref = float(np.mean((prior - y_ev) ** 2))
    return {"bss": 100000.0 * (1 - bs / bs_ref), "bs": bs, "bs_ref": bs_ref,
            "n_eval": int(ev_mask.sum())}


# ---------------------------------------------------------------- 패널: 수치형
def panel_numeric(df: pd.DataFrame, col: str, n_bins: int = 20,
                  has_target: bool = True, log_x: bool = False) -> str:
    s = df[col]
    v = s.dropna().to_numpy(dtype="float64")
    miss = float(s.isna().mean())

    ncols = 2 if has_target else 1
    fig, axes = plt.subplots(1, ncols, figsize=(11.2 if has_target else 5.8, 3.1))
    axes = np.atleast_1d(axes)

    # (좌) 분포 -----------------------------------------------------------
    ax = axes[0]
    lo, hi = np.nanpercentile(v, [0.1, 99.9]) if v.size else (0, 1)
    if hi <= lo:
        hi = lo + 1e-9
    vv = v[(v >= lo) & (v <= hi)]
    ax.hist(vv, bins=60, color=C_BAR, edgecolor="none")
    ax.set_title(f"분포  (표시구간 0.1~99.9%, 결측 {miss:.1%})", loc="left")
    ax.set_xlabel(col)
    ax.set_ylabel("행 수")
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    if log_x and (v > 0).all():
        ax.set_xscale("log")
    med = float(np.median(v)) if v.size else np.nan
    ax.axvline(med, color=C_REF, lw=1.1, ls="--")
    ax.annotate(f"중앙값 {med:,.3g}", xy=(med, ax.get_ylim()[1] * 0.92),
                xytext=(4, 0), textcoords="offset points",
                color=C_REF, fontsize=8, va="top")

    if not has_target:
        return fig_to_b64(fig)

    # (우) 구간별 성공률 ---------------------------------------------------
    ax = axes[1]
    y = df[TARGET].to_numpy()
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.nanquantile(v, qs)) if v.size else np.array([0, 1])
    if len(edges) < 3:
        edges = np.unique(np.nanquantile(v, np.linspace(0, 1, 6)))
    if len(edges) >= 3:
        edges[0], edges[-1] = -np.inf, np.inf
        b = pd.cut(s, edges, labels=False)
        g = pd.DataFrame({"b": b, "y": y, "x": s}).dropna(subset=["b"])
        agg = g.groupby("b").agg(n=("y", "size"), k=("y", "sum"), xm=("x", "median"))
        lo_ci, hi_ci = wilson_ci(agg["k"].to_numpy(), agg["n"].to_numpy())
        rate = agg["k"].to_numpy() / agg["n"].to_numpy()

        # 회색 배경 막대 = 구간별 표본 수. 분위 구간이라 폭이 제각각이므로
        # 이웃 구간 간격으로 폭을 따로 잡는다(겹침·공백 방지). 극단 분위가
        # 축을 늘리지 않도록 x 범위는 구간 중앙값 기준으로 고정한다.
        xm = agg["xm"].to_numpy(dtype="float64")
        if len(xm) > 1:
            d = np.diff(xm)
            w = np.empty_like(xm)
            w[0], w[-1] = d[0], d[-1]
            if len(xm) > 2:
                w[1:-1] = np.minimum(d[:-1], d[1:])
        else:
            w = np.array([1.0])

        ax2 = ax.twinx()
        ax2.bar(xm, agg["n"].to_numpy(), width=w * 0.92, color="#eceff4",
                edgecolor="white", linewidth=0.4, zorder=0)
        ax.set_xlim(xm.min() - w[0] * 0.8, xm.max() + w[-1] * 0.8)
        ax2.set_ylabel("행 수", color="#a0a6b2")
        ax2.tick_params(colors="#a0a6b2")
        ax2.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
        ax2.grid(False)
        ax2.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_visible(False)

        ax.fill_between(agg["xm"], lo_ci, hi_ci, color=C_FILL, alpha=0.85, zorder=2)
        ax.plot(agg["xm"], rate, "-o", ms=3.2, color=C_LINE, lw=1.6, zorder=3)

    base = float(np.nanmean(y))
    ax.axhline(base, color=C_REF, lw=1.0, ls="--", zorder=4)
    ax.annotate(f"전체 평균 {base:.3f}", xy=(0.99, base), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", ha="right",
                color=C_REF, fontsize=8)

    # 결측 그룹은 별도 표시 (asof_* cold-start 확인용)
    if miss > 0:
        r_na = float(y[s.isna().to_numpy()].mean())
        ax.axhline(r_na, color=C_ALT, lw=1.0, ls=":", zorder=4)
        ax.annotate(f"결측군 {r_na:.3f}", xy=(0.01, r_na), xycoords=("axes fraction", "data"),
                    xytext=(0, -10), textcoords="offset points",
                    color=C_ALT, fontsize=8)

    ax.set_title("구간별 제구 성공률 (20분위, 음영=95% CI)", loc="left")
    ax.set_xlabel(f"{col} (구간 중앙값)")
    ax.set_ylabel("성공률")
    return fig_to_b64(fig)


# ---------------------------------------------------------------- 패널: 범주형
def panel_categorical(df: pd.DataFrame, col: str, top: int = 25,
                      has_target: bool = True) -> str:
    s = df[col].astype("object").where(df[col].notna(), "(결측)")
    vc = s.value_counts()
    keep = vc.head(top)
    other_n = int(vc.iloc[top:].sum()) if len(vc) > top else 0
    # 순서가 있는 값(시즌·월·카운트 등)은 빈도순이 아니라 값 순으로 세운다.
    if pd.api.types.is_numeric_dtype(df[col]):
        num = [k for k in keep.index if not isinstance(k, str)]
        na = [k for k in keep.index if isinstance(k, str)]
        keep = keep.reindex(sorted(num) + na)

    fig, ax = plt.subplots(figsize=(11.2, 3.1))
    x = np.arange(len(keep))
    labels = [str(i) for i in keep.index]

    ax.bar(x, keep.to_numpy(), color=C_BAR, edgecolor="none", width=0.72)
    ax.set_ylabel("행 수", color="#7c8391")
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0 if max(len(l) for l in labels) <= 4 else 45,
                       ha="center" if max(len(l) for l in labels) <= 4 else "right")
    ax.set_xlabel(col)

    ttl = f"값별 빈도 (고유값 {df[col].nunique(dropna=True):,}개"
    ttl += f", 상위 {top}개만 표시, 나머지 {other_n:,}행)" if other_n else ")"

    if has_target:
        y = df[TARGET]
        g = pd.DataFrame({"c": s, "y": y}).groupby("c").agg(n=("y", "size"), k=("y", "sum"))
        g = g.loc[keep.index]
        rate = g["k"].to_numpy() / g["n"].to_numpy()
        lo_ci, hi_ci = wilson_ci(g["k"].to_numpy(), g["n"].to_numpy())

        ax2 = ax.twinx()
        ax2.errorbar(x, rate, yerr=[rate - lo_ci, hi_ci - rate], fmt="o", ms=4.5,
                     color=C_LINE, ecolor=C_LINE, elinewidth=1.2, capsize=2.5, zorder=3)
        base = float(y.mean())
        ax2.axhline(base, color=C_REF, ls="--", lw=1.0, zorder=2)
        ax2.set_ylabel("성공률", color=C_LINE)
        ax2.tick_params(axis="y", colors=C_LINE)
        ax2.grid(False)
        span = max(hi_ci.max() - lo_ci.min(), 0.02)
        ax2.set_ylim(lo_ci.min() - span * 0.25, hi_ci.max() + span * 0.25)
        ttl += "   ·  점=성공률(95% CI), 빨간선=전체 평균"

    ax.set_title(ttl, loc="left")
    return fig_to_b64(fig)


# ---------------------------------------------------------------- 패널: 고카디널리티 ID
def panel_id(df: pd.DataFrame, col: str) -> str:
    g = df.groupby(col)[TARGET].agg(["size", "mean"])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.1))

    ax = axes[0]
    ax.hist(g["size"], bins=60, color=C_BAR, edgecolor="none")
    ax.set_xscale("log")
    ax.set_title(f"{col}별 표본 수 분포 (고유 {len(g):,}명/팀)", loc="left")
    ax.set_xlabel("해당 ID의 투구 수 (log)")
    ax.set_ylabel("ID 수")

    ax = axes[1]
    sub = g[g["size"] >= 100]
    ax.hist(sub["mean"], bins=50, color=C_LINE, alpha=0.85, edgecolor="none")
    ax.axvline(df[TARGET].mean(), color=C_REF, ls="--", lw=1.1)
    ax.set_title(f"ID별 제구 성공률 분포 (표본 100투구 이상 {len(sub):,}개)", loc="left")
    ax.set_xlabel("해당 ID의 평균 성공률")
    ax.set_ylabel("ID 수")
    sd = float(sub["mean"].std())
    ax.annotate(f"표준편차 {sd:.3f}", xy=(0.02, 0.9), xycoords="axes fraction",
                fontsize=8, color="#5a6070")
    return fig_to_b64(fig)


# ---------------------------------------------------------------- 데이터 로딩
def load_train(path: str) -> pd.DataFrame:
    log(f"train 로딩: {path}")
    df = pd.read_csv(path)
    for c in df.columns:
        if df[c].dtype == "int64":
            df[c] = pd.to_numeric(df[c], downcast="integer")
        elif df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    log(f"  shape={df.shape}  mem={df.memory_usage(deep=True).sum()/1e9:.2f} GB")
    return df


def load_trackman(path: str) -> pd.DataFrame:
    log(f"trackman 로딩: {path}")
    df = pd.read_csv(path)
    for c in df.columns:
        if df[c].dtype == "int64":
            df[c] = pd.to_numeric(df[c], downcast="integer")
        elif df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    log(f"  shape={df.shape}  mem={df.memory_usage(deep=True).sum()/1e9:.2f} GB")
    return df


# ---------------------------------------------------------------- 신호 지도
def fig_signal_map(bss_table: pd.DataFrame) -> str:
    t = bss_table.dropna(subset=["bss"]).sort_values("bss")
    fig, ax = plt.subplots(figsize=(9.4, max(4.5, 0.23 * len(t))))
    colors = [C_LINE if b > 0 else C_REF for b in t["bss"]]
    ax.barh(np.arange(len(t)), t["bss"], color=colors, height=0.72)
    ax.set_yticks(np.arange(len(t)))
    ax.set_yticklabels(t["feature"], fontsize=7.6)
    ax.axvline(0, color="#5a6070", lw=0.9)
    ax.set_xlabel("단변량 BSS  (2019–23 적합 → 2024 평가, 대회 채점식 100000·(1−BS/BS_ref))")
    ax.set_title("피처 하나만으로 얻을 수 있는 점수", loc="left", fontsize=11)
    for i, (b, n) in enumerate(zip(t["bss"], t["feature"])):
        ax.annotate(f"{b:,.0f}", xy=(b, i), xytext=(4 if b >= 0 else -4, 0),
                    textcoords="offset points", va="center",
                    ha="left" if b >= 0 else "right", fontsize=7, color="#3b4250")
    ax.margins(x=0.14)
    return fig_to_b64(fig)


# ---------------------------------------------------------------- 상관 히트맵
def fig_corr(df: pd.DataFrame, cols: list, sample: int = 300_000,
             method: str = "spearman", title: str = "") -> str:
    d = df[cols]
    if len(d) > sample:
        d = d.sample(sample, random_state=SEED)
    c = d.corr(method=method)
    n = len(cols)
    fig, ax = plt.subplots(figsize=(min(0.34 * n + 3.2, 15), min(0.34 * n + 2.6, 14)))
    im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(cols, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02).set_label(f"{method} ρ", fontsize=8)
    ax.set_title(title, loc="left", fontsize=11)
    if n <= 26:
        for i in range(n):
            for j in range(n):
                v = c.iat[i, j]
                if i != j and abs(v) >= 0.3:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if abs(v) > 0.6 else "#2b2f36")
    return fig_to_b64(fig, dpi=110)


def top_corr_pairs(df: pd.DataFrame, cols: list, k: int = 15,
                   sample: int = 300_000) -> pd.DataFrame:
    d = df[cols]
    if len(d) > sample:
        d = d.sample(sample, random_state=SEED)
    c = d.corr(method="spearman").abs()
    m = np.triu(np.ones(c.shape, dtype=bool), 1)
    s = c.where(m).stack().sort_values(ascending=False).head(k)
    return pd.DataFrame({"feature A": [i[0] for i in s.index],
                         "feature B": [i[1] for i in s.index],
                         "|ρ|": s.to_numpy().round(3)})


# ---------------------------------------------------------------- PCA + KMeans
# --- 구성비(합=1) 데이터 처리 ------------------------------------------------
def clr_transform(counts: pd.DataFrame, prefix: str = "clr_") -> pd.DataFrame:
    """
    구성비 데이터를 CLR(centered log-ratio)로 변환한다.

    왜 필요한가: 구종 비율은 합이 정확히 1이라 원자료로 PCA를 돌리면 공분산 행렬이
    특이해지고(고유값 0인 허수 축이 생긴다), 각 성분이 [0,1]에 갇혀 있어 거리 계산도
    왜곡된다. CLR = log(x_i) − mean_j log(x_j) 는 이를 실수 전체로 편다.
    0은 multiplicative replacement(δ = 0.5/총투구수)로 대체한다 — log(0)을 피하면서
    "관측되지 않았다"를 "표본이 한 번 더 있었다면 나왔을 최소치"로 다룬다.
    CLR 결과는 행 합이 0이라 자유도는 여전히 p−1이다. PCA 성분 수를 랭크로 자른다.
    """
    n = counts.sum(axis=1).to_numpy(dtype="float64")
    P = counts.to_numpy(dtype="float64") / np.maximum(n, 1)[:, None]
    d = (0.5 / np.maximum(n, 1))[:, None]
    P = np.where(P > 0, P, d)
    P = P / P.sum(axis=1, keepdims=True)
    lg = np.log(P)
    return pd.DataFrame(lg - lg.mean(axis=1, keepdims=True),
                        index=counts.index, columns=[f"{prefix}{c}" for c in counts.columns])


# --- 군집 구조가 실재하는지 검정 ---------------------------------------------
def gap_statistic(X: np.ndarray, k_max: int = 8, B: int = 20, seed: int = SEED):
    """
    Tibshirani et al. (2001) gap statistic.
    참조분포는 PCA 정렬 바운딩 박스 위 균등분포 — 데이터의 퍼짐과 방향은 그대로 두되
    '덩어리'만 없앤 null이다. 정규분포 null은 경계가 있는 데이터에서 잘못된 유의성을
    만들기 때문에 쓰지 않는다.
    선택 규칙: Gap(k) >= Gap(k+1) − s(k+1) 를 만족하는 최소 k. k=1이면 군집 구조 없음.
    """
    rng = np.random.default_rng(seed)
    pc = PCA().fit(X)
    Xp = pc.transform(X)
    lo, hi = Xp.min(0), Xp.max(0)
    ks = list(range(1, k_max + 1))

    def W(M, k):
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(M).inertia_

    logW = np.array([np.log(max(W(X, k), 1e-12)) for k in ks])
    ref = np.zeros((B, len(ks)))
    for b in range(B):
        U = rng.uniform(lo, hi, size=X.shape) @ pc.components_ + pc.mean_
        ref[b] = [np.log(max(W(U, k), 1e-12)) for k in ks]
    gap = ref.mean(0) - logW
    sk = ref.std(0) * np.sqrt(1 + 1 / B)
    k_hat = ks[-1]
    for i in range(len(ks) - 1):
        if gap[i] >= gap[i + 1] - sk[i + 1]:
            k_hat = ks[i]
            break
    return pd.DataFrame({"k": ks, "gap": gap, "s_k": sk}), int(k_hat)


@dataclass
class ClusterResult:
    X: np.ndarray
    cols: list
    pca: PCA
    Z: np.ndarray
    labels: np.ndarray
    k: int
    sil: pd.DataFrame
    profile: pd.DataFrame
    meta: pd.DataFrame
    rank: int
    gap: pd.DataFrame
    k_gap: int


def run_pca_kmeans(mat: pd.DataFrame, k_range=range(2, 9), sample_sil: int = 20000,
                   display: pd.DataFrame | None = None, gap_B: int = 20) -> ClusterResult:
    cols = list(mat.columns)
    X = StandardScaler().fit_transform(mat.to_numpy(dtype="float64"))
    # 구성비 제약 등으로 생기는 0 고유값 축은 성분에서 제외한다
    sv = np.linalg.svd(X, compute_uv=False)
    rank = int((sv > sv[0] * 1e-6).sum())
    pca = PCA(n_components=max(min(rank, len(cols)), 2), random_state=SEED).fit(X)
    Z = pca.transform(X)

    rows = []
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), min(sample_sil, len(X)), replace=False)
    best = None
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        sil = silhouette_score(X[idx], km.labels_[idx])
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        if best is None or sil > best[1]:
            best = (km, sil, k)
    km, _, k = best
    labels = km.labels_
    gap_df, k_gap = gap_statistic(X, k_max=max(k_range), B=gap_B)

    disp = mat if display is None else display
    dcols = list(disp.columns)
    prof = disp.copy()
    prof["cluster"] = labels
    zprof = ((prof.groupby("cluster")[dcols].mean() - disp.mean()) / disp.std())
    return ClusterResult(X=X, cols=cols, pca=pca, Z=Z, labels=labels, k=k,
                         sil=pd.DataFrame(rows), profile=zprof,
                         meta=prof.groupby("cluster")[dcols].mean(),
                         rank=rank, gap=gap_df, k_gap=k_gap)


def fig_scree(res: ClusterResult, title: str) -> str:
    ev = res.pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 3.3))
    ax = axes[0]
    ax.bar(np.arange(1, len(ev) + 1), ev * 100, color=C_BAR)
    ax.plot(np.arange(1, len(ev) + 1), np.cumsum(ev) * 100, "-o", ms=3.5, color=C_LINE)
    for i, c in enumerate(np.cumsum(ev) * 100, start=1):
        if i <= 5:
            ax.annotate(f"{c:.0f}%", (i, c), xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=7.5, color=C_LINE)
    ax.set_xlabel(f"주성분 (유효 랭크 {res.rank})"); ax.set_ylabel("설명 분산 (%)")
    ax.set_title("Scree — 막대=개별, 선=누적", loc="left")

    ax = axes[1]
    ax.plot(res.sil["k"], res.sil["silhouette"], "-o", ms=4, color=C_LINE, label="silhouette")
    ax.axvline(res.k, color=C_REF, ls="--", lw=1.1)
    ax.annotate(f"silhouette 최대 k={res.k}", xy=(res.k, res.sil["silhouette"].max()),
                xytext=(5, -2), textcoords="offset points", color=C_REF, fontsize=8)
    ax.set_xlabel("클러스터 수 k"); ax.set_ylabel("silhouette")
    ax2 = ax.twinx()
    ax2.plot(res.sil["k"], res.sil["inertia"], "-s", ms=3.2, color="#a0a6b2", lw=1.0)
    ax2.set_ylabel("inertia (elbow)", color="#a0a6b2")
    ax2.tick_params(colors="#a0a6b2"); ax2.grid(False)
    ax.set_title("k 후보 — 절대값은 기준이 못 된다", loc="left")

    ax = axes[2]
    g = res.gap
    ax.errorbar(g["k"], g["gap"], yerr=g["s_k"], fmt="-o", ms=4.5, color=C_LINE, capsize=3)
    ax.axvline(res.k_gap, color=C_REF, ls="--", lw=1.2)
    ax.annotate(f"Gap 선택 k={res.k_gap}" + ("  (= 군집 없음)" if res.k_gap == 1 else ""),
                xy=(res.k_gap, g["gap"].max()), xytext=(6, -2), textcoords="offset points",
                color=C_REF, fontsize=8.5)
    ax.set_xlabel("클러스터 수 k"); ax.set_ylabel("Gap 통계량")
    ax.set_title("Gap statistic — 균등박스 null 대비", loc="left")
    fig.suptitle(title, x=0.005, ha="left", fontsize=11)
    return fig_to_b64(fig, dpi=95)


def fig_biplot(res: ClusterResult, color_by: np.ndarray | None = None,
               cbar_label: str = "", title: str = "", max_pts: int = 6000,
               arrow_scale: float = 1.0) -> str:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(res.Z), min(max_pts, len(res.Z)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))

    # (좌) 클러스터
    ax = axes[0]
    for c in range(res.k):
        m = res.labels[idx] == c
        ax.scatter(res.Z[idx][m, 0], res.Z[idx][m, 1], s=7, alpha=0.55,
                   color=PALETTE[c % len(PALETTE)], edgecolors="none",
                   label=f"C{c} (n={int((res.labels==c).sum()):,})")
    ax.legend(markerscale=2.0, loc="best", fontsize=7.5)
    ax.set_xlabel(f"PC1 ({res.pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({res.pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("KMeans 군집", loc="left")

    # 로딩 화살표 (biplot)
    load = res.pca.components_[:2].T * np.sqrt(res.pca.explained_variance_[:2])
    sc = arrow_scale * 0.85 * max(np.abs(res.Z[idx][:, :2]).max(), 1e-9) / (np.abs(load).max() + 1e-9)
    for i, name in enumerate(res.cols):
        dx, dy = load[i, 0] * sc, load[i, 1] * sc
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#2b2f36", lw=1.0,
                                    alpha=0.8, shrinkA=0, shrinkB=0))
        ax.annotate(name, (dx, dy), fontsize=7, color="#12161c",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#d5d9e0",
                              lw=0.4, alpha=0.85))
    ax.margins(0.18)

    # (우) 연속값 색칠
    ax = axes[1]
    if color_by is not None:
        v = color_by[idx]
        lo, hi = np.nanpercentile(v, [2, 98])
        sc2 = ax.scatter(res.Z[idx][:, 0], res.Z[idx][:, 1], s=7, c=v, cmap="coolwarm_r",
                         vmin=lo, vmax=hi, alpha=0.75, edgecolors="none")
        fig.colorbar(sc2, ax=ax, fraction=0.04, pad=0.02).set_label(cbar_label, fontsize=8)
        ax.set_title(f"같은 평면 · {cbar_label} 로 색칠", loc="left")
    else:
        ax.axis("off")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.suptitle(title, x=0.005, ha="left", fontsize=11)
    return fig_to_b64(fig)


def fig_cluster_profile(res: ClusterResult, extra: pd.DataFrame | None = None,
                        title: str = "") -> str:
    p = res.profile
    fig, ax = plt.subplots(figsize=(min(0.55 * len(p.columns) + 3.0, 13), 0.55 * res.k + 2.0))
    im = ax.imshow(p.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(p.columns)))
    ax.set_xticklabels(p.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(res.k))
    ax.set_yticklabels([f"C{c} (n={int((res.labels==c).sum()):,})" for c in p.index], fontsize=8)
    ax.grid(False)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.iat[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 1.2 else "#2b2f36")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02).set_label("전체 대비 z-score", fontsize=8)
    ax.set_title(title, loc="left", fontsize=11)
    return fig_to_b64(fig)


# ---------------------------------------------------------------- HTML
CSS = """
:root{--fg:#12161c;--mut:#5a6070;--line:#e2e5ea;--acc:#2f6fdb;--warn:#e4573d;--bg:#ffffff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.72 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR","Malgun Gothic",sans-serif;}
.wrap{max-width:1160px;margin:0 auto;padding:48px 28px 120px;}
h1{font-size:30px;line-height:1.3;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:22px;margin:64px 0 6px;padding-top:22px;border-top:1px solid var(--line);letter-spacing:-.01em}
h3{font-size:16px;margin:34px 0 4px}
.sub{color:var(--mut);font-size:14px;margin:0 0 28px}
p{margin:10px 0 14px}
code{background:#f2f4f7;padding:1px 5px;border-radius:4px;font-size:.88em;
 font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.toc{background:#f7f8fa;border:1px solid var(--line);border-radius:10px;padding:18px 22px;margin:26px 0 8px}
.toc ol{margin:6px 0 0;padding-left:20px} .toc li{margin:3px 0}
.toc a{color:var(--fg);text-decoration:none;border-bottom:1px solid transparent}
.toc a:hover{border-color:var(--acc);color:var(--acc)}
.feat{border:1px solid var(--line);border-radius:10px;padding:14px 16px 6px;margin:16px 0;}
.feat.hi{border-color:#b9cdf3;background:#fbfcfe}
.fhead{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:2px}
.fname{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:15px;font-weight:600}
.badge{font-size:11.5px;padding:1.5px 8px;border-radius:20px;background:#eef1f5;color:var(--mut)}
.badge.b{background:#e7effc;color:#1c4fa8}.badge.w{background:#fdecea;color:#b23a25}
.badge.g{background:#e6f5ee;color:#1c7a52}
.fdesc{color:var(--mut);font-size:13.5px;margin:2px 0 10px}
img{max-width:100%;display:block;border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:14px 0}
th,td{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left}
th{background:#f7f8fa;font-weight:600;font-size:12.5px;color:var(--mut)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.note{border-left:3px solid var(--acc);background:#f6f9fe;padding:12px 16px;margin:16px 0;font-size:14px}
.note.warn{border-color:var(--warn);background:#fef7f5}
.note b{font-weight:650}
.meta{color:var(--mut);font-size:12.5px;margin-top:44px;border-top:1px solid var(--line);padding-top:14px}
"""


def h(s) -> str:
    return html.escape(str(s))


def img_tag(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" loading="lazy">'


def _fmt_cell(v) -> str:
    """정수는 천단위 구분, 소수는 유효숫자 4자리. 큰 정수가 과학표기로 새는 것을 막는다."""
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return f"{int(v):,}"
    if isinstance(v, (np.floating, float)):
        if not np.isfinite(v):
            return "—"
        if float(v).is_integer() and abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:,.4g}"
    return str(v)


def df_to_html(df: pd.DataFrame, num_cols=()) -> str:
    ths = "".join(f'<th class="{"num" if c in num_cols else ""}">{h(c)}</th>' for c in df.columns)
    trs = []
    for _, r in df.iterrows():
        tds = "".join(f'<td class="{"num" if c in num_cols else ""}">{h(_fmt_cell(r[c]))}</td>'
                      for c in df.columns)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{ths}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


# ---------------------------------------------------------------- 컬럼 사전
DESC = {
    "season": "시즌 연도. train은 2019~2024, 실제 평가는 2025.",
    "game_month": "경기 월.",
    "game_dayofweek": "경기 요일 (월=0 … 일=6).",
    "inning": "투구 직전 이닝.",
    "top_bottom": "초(T)/말(B) 구분.",
    "game_type": "경기 유형 코드 R / F. ⚠ F는 2019–2022 구간에서 reverse 미검출로 판정이 부정확하다. 아래 패널은 시즌을 뭉갠 값이라 두 체제가 섞여 있으니 1절과 함께 봐라.",
    "balls_before": "투구 직전 볼 카운트.",
    "strikes_before": "투구 직전 스트라이크 카운트.",
    "outs_before": "투구 직전 아웃 카운트.",
    "run_top_before": "투구 직전 초 공격 팀 점수.",
    "run_bot_before": "투구 직전 말 공격 팀 점수.",
    "run_total_before": "투구 직전 양 팀 합산 점수.",
    "score_diff_home": "홈 팀 기준 점수 차.",
    "score_diff_pitcher_team": "투수 소속 팀 기준 점수 차.",
    "runner_on_1b": "1루 주자 여부.",
    "runner_on_2b": "2루 주자 여부.",
    "runner_on_3b": "3루 주자 여부.",
    "num_runners_on": "출루 주자 수.",
    "base_state": "주자 상황 문자열 (___ / 1__ / _2_ / __3 / 12_ / 1_3 / _23 / 123).",
    "home_win_expectancy": "홈 팀 기대 승률 (0~100).",
    "away_win_expectancy": "원정 팀 기대 승률 (0~100).",
    "li": "상황 중요도(Leverage Index). 클수록 경기 흐름에 미치는 영향이 큼.",
    "pitcher_id": "투수 익명 ID.",
    "batter_id": "타자 익명 ID.",
    "pitcher_hand": "투수 좌우 유형 코드.",
    "batter_hand": "타자 좌우 유형 코드.",
    "pitcher_team_id": "투수 소속 팀 ID.",
    "batter_team_id": "타자 소속 팀 ID.",
    "asof_pitcher_n": "이 투구 직전까지 해당 투수의 누적 투구 수.",
    "asof_pitcher_success_rate": "직전까지 해당 투수의 제구 성공률.",
    "asof_pitcher_reverse_rate": "직전까지 해당 투수의 의도 반대성 투구 비율.",
    "asof_pitcher_middle_rate": "직전까지 해당 투수의 가운데/위험 코스 비율.",
    "asof_pitcher_ball_rate": "직전까지 해당 투수의 볼성 결과 비율.",
    "asof_pitcher_strike_rate": "직전까지 해당 투수의 스트라이크성 결과 비율.",
    "asof_pitcher_prev1_game_success_rate": "직전 1경기 제구 성공률.",
    "asof_pitcher_prev3_game_success_rate": "직전 3경기 제구 성공률.",
    "asof_pitcher_prev5_game_success_rate": "직전 5경기 제구 성공률.",
    "asof_pitcher_prev1_game_middle_rate": "직전 1경기 가운데/위험 코스 비율.",
    "asof_pitcher_prev3_game_middle_rate": "직전 3경기 가운데/위험 코스 비율.",
    "asof_pitcher_prev5_game_middle_rate": "직전 5경기 가운데/위험 코스 비율.",
    "asof_batter_n": "직전까지 해당 타자가 상대한 누적 투구 수.",
    "asof_batter_success_rate": "직전까지 해당 타자가 상대한 투구의 제구 성공률.",
    "asof_batter_middle_rate": "직전까지 해당 타자가 상대한 투구의 가운데/위험 코스 비율.",
    "asof_pitcher_pitchmix_n": "구종 사용 이력 표본 수.",
    "asof_pitcher_fastball_rate": "fastball 계열 사용 비율.",
    "asof_pitcher_breaking_rate": "breaking 계열 사용 비율.",
    "asof_pitcher_offspeed_rate": "offspeed 계열 사용 비율.",
    # trackman
    "trackman_id": "Trackman 로그 행 식별자.",
    "game_date": "경기 날짜.",
    "trackman_game_id": "Trackman 기준 경기 ID. main 데이터와 직접 대응하지 않음.",
    "pitch_no": "경기 내 투구 번호.",
    "pitch_of_pa": "해당 타석에서의 투구 순번.",
    "pitcher_trackman_id": "Trackman 기준 투수 ID (train의 pitcher_id와 다른 체계).",
    "batter_trackman_id": "Trackman 기준 타자 ID.",
    "pitcher_team": "투수 소속 팀.",
    "batter_team": "타자 소속 팀.",
    "tagged_pitch_type": "수동/태깅 기반 구종명.",
    "auto_pitch_type": "자동 분류 기반 구종명.",
    "pitch_type_group": "구종군 (fastball / breaking / offspeed / other).",
    "rel_speed": "릴리스 시점 구속. 중앙값 137, fastball 143 → 단위는 mph가 아니라 km/h.",
    "spin_rate": "회전수 (rpm). 중앙값 약 2,220.",
    "induced_vert_break": "유도 수직 무브먼트. 관측 범위 ±70, fastball이 +40대 → inch가 아니라 cm.",
    "horz_break": "수평 무브먼트 (cm). 좌완 중앙값 −17 / 우완 +22 → 손 방향으로 부호가 갈려 쌍봉 분포.",
    "extension": "릴리스 확장 거리 (m). 중앙값 1.76 (≈5.8 ft).",
    "rel_height": "릴리스 높이 (m). 중앙값 1.76.",
    "rel_side": "릴리스 좌우 위치 (m). 우완 +0.52 / 좌완 −0.57.",
    "zone_speed": "홈플레이트 부근 구속 (km/h). rel_speed보다 중앙값 기준 약 11.5 낮다.",
}

TRAIN_GROUPS = [
    ("경기 정보", ["season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type"]),
    ("카운트 · 점수 상황", ["balls_before", "strikes_before", "outs_before", "run_top_before",
                      "run_bot_before", "run_total_before", "score_diff_home",
                      "score_diff_pitcher_team"]),
    ("주자 · 상황 중요도", ["runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
                     "base_state", "home_win_expectancy", "away_win_expectancy", "li"]),
    ("선수 · 팀", ["pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
                "pitcher_team_id", "batter_team_id"]),
    ("투수 asof 이력", ["asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
                   "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
                   "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
                   "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
                   "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate"]),
    ("타자 asof 이력", ["asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate"]),
    ("구종 믹스 asof", ["asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
                   "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]),
]

TRACKMAN_GROUPS = [
    ("식별자 · 경기 맥락", ["season", "game_month", "game_dayofweek", "pitch_no", "inning",
                     "top_bottom", "balls_before", "strikes_before", "outs_before", "pitch_of_pa"]),
    ("선수 · 팀", ["pitcher_trackman_id", "batter_trackman_id", "pitcher_hand", "batter_hand",
                "pitcher_team", "batter_team"]),
    ("구종", ["tagged_pitch_type", "auto_pitch_type", "pitch_type_group"]),
    ("물리 측정값", ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
                "extension", "rel_height", "rel_side", "zone_speed"]),
]

# Trackman tagged_pitch_type 표기 정규화 (소문자 키 → 표준 구종명)
PITCH_CANON = {
    "fastball": "Fastball", "four-seam": "Fastball",
    "sinker": "Sinker", "cutter": "Cutter",
    "slider": "Slider", "sweeper": "Slider",
    "curveball": "Curveball", "slurve": "Curveball",
    "changeup": "ChangeUp", "splitter": "Splitter",
    # 제외: knuckleball(표본 극소), other, undefined, nan
}
PITCH_ORDER = ["Fastball", "Sinker", "Cutter", "Slider", "Curveball", "ChangeUp", "Splitter"]
PITCH_FAMILY = {"Fastball": "FB", "Sinker": "FB", "Cutter": "FB",
                "Slider": "BR", "Curveball": "BR",
                "ChangeUp": "OFF", "Splitter": "OFF"}

LOG_X = {"asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n", "li",
         "spin_rate", "pitch_no"}
ID_LIKE = {"pitcher_id", "batter_id", "row_id", "trackman_id",
           "pitcher_trackman_id", "batter_trackman_id", "trackman_game_id", "game_date"}


def badge_for_bss(b) -> str:
    if b is None or (isinstance(b, float) and np.isnan(b)):
        return '<span class="badge">단변량 BSS n/a</span>'
    cls = "b" if b >= 300 else ("g" if b >= 30 else ("" if b >= 0 else "w"))
    return f'<span class="badge {cls}">단변량 BSS {b:,.0f}</span>'


def feature_block(df, col, bss=None, has_target=True) -> str:
    nun = int(df[col].nunique(dropna=True))
    miss = float(df[col].isna().mean())
    if col in ID_LIKE and has_target:
        img = panel_id(df, col)
        kind = "고카디널리티 ID"
    elif (not pd.api.types.is_numeric_dtype(df[col])) or nun <= 25:
        img = panel_categorical(df, col, has_target=has_target)
        kind = "범주형/이산"
    else:
        img = panel_numeric(df, col, has_target=has_target, log_x=col in LOG_X)
        kind = "연속형"

    hi = (bss is not None and not (isinstance(bss, float) and np.isnan(bss)) and bss >= 300)
    badges = (f'<span class="badge">{kind}</span>'
              f'<span class="badge">고유값 {nun:,}</span>'
              + (f'<span class="badge w">결측 {miss:.1%}</span>' if miss > 0 else "")
              + (badge_for_bss(bss) if has_target else ""))
    return (f'<div class="feat{" hi" if hi else ""}" id="f-{h(col)}">'
            f'<div class="fhead"><span class="fname">{h(col)}</span>{badges}</div>'
            f'<div class="fdesc">{h(DESC.get(col, ""))}</div>{img_tag(img)}</div>')


# ---------------------------------------------------------------- 타깃 섹션
BAD_F_SEASONS = (2019, 2020, 2021, 2022)   # F 판정 오염 구간 (reverse 미검출)


def target_priors(df) -> dict:
    """2025 사전확률 추정 — 오염 구간을 뺀 계열로 계산."""
    isF = df["game_type"] == "F"
    t = pd.DataFrame({
        "all": df.groupby("season")[TARGET].mean(),
        "R": df[~isF].groupby("season")[TARGET].mean(),
        "F": df[isF].groupby("season")[TARGET].mean(),
        "Fshare": isF.groupby(df["season"]).mean(),
    })
    yrs = t.index.to_numpy(dtype=float)
    p_all = float(np.polyval(np.polyfit(yrs, t["all"].to_numpy(), 1), 2025))
    p_R = float(np.polyval(np.polyfit(yrs, t["R"].to_numpy(), 1), 2025))
    clean = [s for s in t.index if s not in BAD_F_SEASONS]
    p_F = float(np.polyval(np.polyfit(np.array(clean, dtype=float),
                                      t.loc[clean, "F"].to_numpy(), 1), 2025))
    w = float(t["Fshare"].iloc[-1])
    return {"tab": t, "p_all": p_all, "p_R": p_R, "p_F": p_F, "w": w,
            "p_mix": (1 - w) * p_R + w * p_F}


def fig_target(df, pri: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.4))
    ax = axes[0]
    vc = df[TARGET].value_counts().sort_index()
    ax.bar(["0 (실패)", "1 (성공)"], vc.to_numpy(), color=[C_REF, C_LINE], width=0.55)
    for i, v in enumerate(vc):
        ax.annotate(f"{v:,}\n({v/len(df):.1%})", (i, v), ha="center", va="bottom", fontsize=8.5)
    ax.set_title("control_success 전체 분포", loc="left")
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    ax.margins(y=0.18)

    ax = axes[1]
    t = pri["tab"]
    ax.axvspan(2018.6, 2022.5, color="#fdecea", zorder=0)
    ax.annotate("F 판정 오염 구간", xy=(2020.5, ax.get_ylim()[1]), xycoords=("data", "axes fraction"),
                xytext=(0, -12), textcoords="offset points", ha="center",
                color="#b23a25", fontsize=8)
    ax.plot(t.index, t["F"], "-s", color=C_REF, ms=4.5, lw=1.4, label="F 경기만")
    ax.plot(t.index, t["all"], "-o", color="#9aa4b4", ms=4.5, lw=1.4, label="전체 (F 포함)")
    ax.plot(t.index, t["R"], "-o", color=C_LINE, ms=5, lw=2.0, label="R 경기만")
    for x, y0 in zip(t.index, t["R"]):
        ax.annotate(f"{y0:.3f}", (x, y0), xytext=(0, -13), textcoords="offset points",
                    ha="center", fontsize=7.5, color=C_LINE)
    ax.plot([2024, 2025], [t["R"].iloc[-1], pri["p_R"]], ":", color=C_LINE, lw=1.3)
    ax.scatter([2025], [pri["p_R"]], color=C_LINE, s=40, marker="D", zorder=5)
    ax.annotate(f"R 2025 외삽 {pri['p_R']:.3f}", (2025, pri["p_R"]), xytext=(-6, 8),
                textcoords="offset points", ha="right", color=C_LINE, fontsize=8)
    ax.plot([2024, 2025], [t["F"].iloc[-1], pri["p_F"]], ":", color=C_REF, lw=1.3)
    ax.scatter([2025], [pri["p_F"]], color=C_REF, s=36, marker="D", zorder=5)
    ax.legend(loc="lower left", fontsize=7.5)
    ax.set_title("시즌별 성공률 — 전체 계열의 2022→2023 급락은 F 보정 인공물", loc="left")
    ax.set_xlabel("season"); ax.set_ylabel("성공률")
    ax.set_xlim(2018.6, 2025.5)
    return fig_to_b64(fig)


def fig_regime(df) -> str:
    """오염 메커니즘 증거 — F 전용 투수의 asof reverse rate가 눌려 있다."""
    isF = df["game_type"] == "F"
    pf = isF.groupby(df["pitcher_id"]).mean()
    fo, ro = set(pf[pf >= 0.95].index), set(pf[pf <= 0.05].index)
    d = df[df["asof_pitcher_n"] >= 100]
    cols = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate"]
    A = d[d["pitcher_id"].isin(fo)].groupby("season")[cols].mean()
    B = d[d["pitcher_id"].isin(ro)].groupby("season")[cols].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.4))
    for ax, c, ttl in zip(axes, cols,
                          ["reverse (포수 요구 반대) 비율 — 여기가 눌려 있다",
                           "middle (가운데 몰림) 비율 — 같은 크기의 단절은 없다"]):
        ax.axvspan(2018.6, 2022.5, color="#fdecea", zorder=0)
        ax.plot(A.index, A[c], "-s", color=C_REF, ms=5, lw=1.8,
                label=f"F 전용 투수 (n={len(fo)})")
        ax.plot(B.index, B[c], "-o", color=C_LINE, ms=5, lw=1.8,
                label=f"R 전용 투수 (n={len(ro)})")
        for x, y0 in zip(A.index, A[c]):
            ax.annotate(f"{y0:.3f}", (x, y0), xytext=(0, -13), textcoords="offset points",
                        ha="center", fontsize=7.5, color=C_REF)
        ax.set_title(ttl, loc="left")
        ax.set_xlabel("season"); ax.set_ylabel(c.replace("asof_pitcher_", ""))
        ax.legend(fontsize=7.5, loc="best")
        ax.set_xlim(2018.6, 2024.4)
    return fig_to_b64(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--out", default="feature_viz_report.html")
    ap.add_argument("--skip-trackman", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    parts: list[str] = []
    toc: list[tuple[str, str]] = []

    def sec(sid, title, body):
        toc.append((sid, title))
        parts.append(f'<h2 id="{sid}">{h(title)}</h2>{body}')

    # ===== train =====
    df = load_train(os.path.join(args.data_dir, "train.csv"))
    n_rows = len(df)
    feats = [c for g, cols in TRAIN_GROUPS for c in cols]

    log("단변량 BSS 계산")
    bss_rows = []
    for c in feats:
        r = univariate_bss(df, c)
        bss_rows.append({"feature": c, "bss": r["bss"]})
    bss_tab = pd.DataFrame(bss_rows)
    bss_map = dict(zip(bss_tab["feature"], bss_tab["bss"]))

    # 0. 개요 + 타깃
    base = float(df[TARGET].mean())
    pri = target_priors(df)
    t = pri["tab"]
    nbad = int(((df["game_type"] == "F") & df["season"].isin(BAD_F_SEASONS)).sum())
    ov = (f"<p><code>train.csv</code> — <b>{n_rows:,}행 × {df.shape[1]}컬럼</b>, "
          f"시즌 {int(df.season.min())}~{int(df.season.max())}. "
          f"전체 제구 성공률 <b>{base:.4f}</b>.</p>")
    ov += img_tag(fig_target(df, pri))
    ov += ('<div class="note warn"><b>이 데이터셋의 가장 큰 함정부터.</b> '
           f'<code>game_type=\'F\'</code> 행은 <b>{"~".join(map(str, (BAD_F_SEASONS[0], BAD_F_SEASONS[-1])))} '
           f'구간에서 제구 실패 판정이 부정확</b>하다 — 포수 요구 반대 방향으로 간 공을 '
           f'제대로 잡아내지 못해 <b>실패가 성공으로 기록</b>됐다. 2023 시즌 개막부터 정상화됐다. '
           f'해당 행은 <b>{nbad:,}행 (train의 {nbad/n_rows:.1%})</b>. '
           f'증거와 처리 방침은 <a href="#signal">1절</a>에.</div>')
    ov += ('<div class="note"><b>그래서 "하향 드리프트"는 과장돼 있다.</b> '
           f'전체 계열은 2022 {t.loc[2022,"all"]:.3f} → 2023 {t.loc[2023,"all"]:.3f}로 '
           f'{t.loc[2022,"all"]-t.loc[2023,"all"]:+.3f} 떨어지지만, '
           f'R 경기만 보면 {t.loc[2022,"R"]:.3f} → {t.loc[2023,"R"]:.3f}로 '
           f'<b>사실상 평평하다</b>. 그 급락은 야구의 변화가 아니라 F 판정이 고쳐진 인공물이다. '
           f'<b>2025 사전확률</b>은 오염 계열을 그대로 외삽(<code>{pri["p_all"]:.4f}</code>)하는 대신 '
           f'R 계열 외삽 <code>{pri["p_R"]:.4f}</code>와 정상 구간 F 외삽 '
           f'<code>{pri["p_F"]:.4f}</code>를 F 비중 {pri["w"]:.1%}로 결합한 '
           f'<b><code>{pri["p_mix"]:.4f}</code></b>가 근거가 낫다. '
           f'수치 자체는 크게 다르지 않지만({pri["p_all"]:.4f} vs {pri["p_mix"]:.4f}), '
           f'전자는 우연히 맞은 것이고 F 비중이 흔들리면 같이 흔들린다.</div>')
    ov += ('<div class="note warn"><b>읽는 법.</b> 이 리포트의 모든 우측 패널은 '
           '<i>구간별 제구 성공률</i>이다. 수평 빨간 점선(전체 평균)에서 곡선이 멀어질수록 '
           '그 피처가 타깃을 가른다. 음영은 95% Wilson 신뢰구간이고, 회색 막대는 그 구간의 '
           '표본 수다. <b>표본이 적은 양 끝단에서 곡선이 요동치는 것은 신호가 아니라 노이즈다</b> — '
           'CI 폭으로 판단하라.</div>')
    sec("overview", "0. 개요와 타깃", ov)

    # 1. 신호 지도
    body = img_tag(fig_signal_map(bss_tab))
    body += ('<div class="note"><b>단변량 BSS의 정의.</b> 피처를 20분위(또는 값)로 나눠 '
             '2019–2023 구간에서 구간별 평균 성공률을 구하고(m-estimate 스무딩, m=200), '
             '그 매핑만으로 2024 시즌을 예측했을 때의 <code>100000·(1−BS/BS_ref)</code>다. '
             '기준선은 2019–2023 평균 성공률을 상수로 찍은 예측. '
             '시간 분할이라 리키지가 없고, <b>시즌 드리프트까지 반영된 값</b>이라 '
             '실제 리더보드에서 그 피처가 얼마를 벌어줄지에 가장 가까운 추정이다. '
             '음수는 "그 피처를 단독으로 쓰면 상수 예측보다 나쁘다"는 뜻 — 드리프트로 매핑이 '
             '뒤집혔거나, <b>애초에 과거 라벨이 틀린 경우</b>다. 최하위값 '
             '<code>game_type</code>이 후자의 사례이고 바로 아래에서 해부한다.</div>')
    body += df_to_html(bss_tab.sort_values("bss", ascending=False).round(1)
                       .rename(columns={"bss": "단변량 BSS"}), num_cols=("단변량 BSS",))

    # game_type 해부 — 단변량 BSS 최하위값의 정체
    wrow = bss_tab.set_index("feature")["bss"]
    pv = df.pivot_table(index="season", columns="game_type", values=TARGET, aggfunc="mean")
    nn = df.pivot_table(index="season", columns="game_type", values=TARGET, aggfunc="size")
    tab = pd.DataFrame({"season": pv.index.astype(str)})
    for c in pv.columns:
        tab[f"{c} 성공률"] = pv[c].round(3).to_numpy()
        tab[f"{c} n"] = nn[c].to_numpy()
    body += (f"<h3>최하위값 해부 — <code>game_type</code> "
             f"(단변량 BSS {wrow.get('game_type', float('nan')):,.0f})</h3>")
    body += df_to_html(tab, num_cols=tuple(c for c in tab.columns if c != "season"))
    body += ('<div class="note warn"><b>이건 드리프트가 아니라 데이터 품질 문제다.</b> '
             f'<code>F</code>의 성공률이 2022 <b>{pv.loc[2022,"F"]:.3f}</b> → '
             f'2023 <b>{pv.loc[2023,"F"]:.3f}</b>로 끊기는 동안 <code>R</code>은 '
             f'{pv.loc[2022,"R"]:.3f} → {pv.loc[2023,"R"]:.3f}로 움직이지 않는다. '
             '월 단위로 쪼개도 시즌 중간이 아니라 <b>2023 시즌 개막에서 절단</b>된다. '
             '원인은 <b>포수 요구 반대 방향 투구(reverse)를 제대로 잡아내지 못한 것</b>이고, '
             '그래서 실패가 성공으로 기록돼 <b>성공률이 부풀려져</b> 있었다. '
             '아래 그림이 그 흔적이다.</div>')
    body += img_tag(fig_regime(df))
    isF_all = df["game_type"] == "F"
    pf_all = isF_all.groupby(df["pitcher_id"]).mean()
    d100 = df[df["asof_pitcher_n"] >= 100]
    fo = set(pf_all[pf_all >= 0.95].index)
    ro = set(pf_all[pf_all <= 0.05].index)
    rev_f = d100[d100["pitcher_id"].isin(fo)].groupby("season")["asof_pitcher_reverse_rate"].mean()
    rev_r = d100[d100["pitcher_id"].isin(ro)].groupby("season")["asof_pitcher_reverse_rate"].mean()
    mid_f = d100[d100["pitcher_id"].isin(fo)].groupby("season")["asof_pitcher_middle_rate"].mean()
    body += (f'<div class="note"><b>증거의 강도.</b> F 경기만 던진 투수들의 '
             f'<code>asof_pitcher_reverse_rate</code>는 2019–2022에 '
             f'<b>{rev_f.loc[list(BAD_F_SEASONS)].mean():.3f}</b>인데, 같은 기간 R 전용 투수는 '
             f'<b>{rev_r.loc[list(BAD_F_SEASONS)].mean():.3f}</b>다. 2024년에는 F 전용 투수도 '
             f'<b>{rev_f.loc[2024]:.3f}</b>로 올라와 R 수준({rev_r.loc[2024]:.3f})에 붙는다. '
             f'대조 성분인 <code>middle_rate</code>도 2024에 오르긴 하지만 폭이 '
             f'<b>{mid_f.loc[2024]-mid_f.loc[list(BAD_F_SEASONS)].mean():+.3f}</b>로 '
             f'reverse의 <b>{rev_f.loc[2024]-rev_f.loc[list(BAD_F_SEASONS)].mean():+.3f}</b>에 '
             f'한참 못 미치고, 2019년 값이 혼자 튀는 등 계열 자체가 불안정하다. '
             f'즉 <b>실패 3종 중 reverse에 누락이 집중됐다</b>는 방향은 지지되지만 '
             f'"reverse만"이라고 단정할 근거까지는 아니다. 사용자가 조사한 내용과 부합한다. '
             f'또 <code>asof_*</code>는 <i>누적</i>이라 2023년 값은 아직 과거 '
             f'이력에 눌려 있다. 그래서 이 그림은 누락 폭의 <b>하한</b>만 보여준다 — '
             f'reverse 결손 약 {rev_f.loc[2024]-rev_f.loc[list(BAD_F_SEASONS)].mean():.2f}는 '
             f'F 성공률 격차 {pv.loc[2022,"F"]-pv.loc[2023,"F"]:.2f}의 일부만 설명한다. '
             f'나머지가 reverse의 추가 누락인지 다른 성분인지는 이 데이터만으로 분해되지 않는다.</div>')
    body += (f'<div class="note warn"><b>처리 방침 — 앞선 판본의 권고를 철회한다.</b> '
             f'이전 버전은 "부호 반전이니 <code>season × game_type</code> 교호작용으로 '
             f'모델이 시점별로 다르게 쓰게 하라"고 썼다. <b>틀렸다.</b> 존재하지 않는 현상에 '
             f'파라미터를 배정하는 것이고, 2025는 정상 체제이므로 과거 F 매핑을 학습할 이유가 없다.'
             f'<br><br>'
             f'<b>① 라벨 오염 — 제거 대상.</b> {BAD_F_SEASONS[0]}–{BAD_F_SEASONS[-1]}의 F '
             f'<b>{int(nn.loc[list(BAD_F_SEASONS),"F"].sum()):,}행</b>은 타깃이 틀렸다. '
             f'학습에서 빼거나 강하게 다운웨이트한다. 채택은 추론이 아니라 측정으로 결정하라 — '
             f'2023까지 적합 → 2024 평가에서 포함/제외/다운웨이트를 Brier로 비교하고, '
             f'F·R 분리 점수를 같이 봐라.'
             f'<br><br>'
             f'<b>② <code>game_type</code> 컬럼 자체는 남긴다.</b> 정상 구간에서도 F는 R보다 '
             f'성공률이 낮다({pv.loc[2023:,"F"].mean():.3f} vs {pv.loc[2023:,"R"].mean():.3f}). '
             f'진짜 신호다. 게다가 평가 데이터가 245,789행인데 2024 R만 '
             f'{int(nn.loc[2024,"R"]):,}행이므로 <b>test에도 F가 들어 있을 가능성이 높다</b>. '
             f'빼면 예측해야 할 대상을 모르게 된다. 정상 구간 F {int(nn.loc[2023:,"F"].sum()):,}행이 '
             f'F 예측의 유일한 신뢰 학습 소스다.'
             f'<br><br>'
             f'<b>③ 컬럼을 지워도 체제는 샌다.</b> F 행의 <b>100%</b>가 '
             f'<code>team_id=13</code>을 한쪽에 포함한다(R 행에서는 19.9%로 정상 팀 지분과 동일). '
             f'추론: 13은 F 경기에서 상대를 특정하지 않는 플레이스홀더로 재사용된 ID로 보인다. '
             f'<code>game_type</code>만 드롭하고 팀 ID를 남기면 모델이 우회로로 같은 것을 학습한다.'
             f'<br><br>'
             f'<b>④ <code>asof_*</code> 오염은 건드리지 마라.</b> '
             f'F·R을 모두 가진 투수 233명에서 <code>max(asof_pitcher_n)</code>과 (전체 행−1)의 '
             f'차이 중앙값이 <b>0</b>이다(R 행만 세었다고 가정하면 301). 즉 이력 피처는 오염된 F '
             f'판정을 포함해 계산돼 있고, 그 편향은 해당 투수의 R 행에도 실려 있다. '
             f'그래도 재계산은 <b>안 된다</b> — 평가 서버의 <code>test.csv</code>도 운영측이 같은 '
             f'규칙으로 계산했을 것이고, 2025 이력이 우리에게 없어 애초에 재계산이 불가능하다. '
             f'train에서만 정제하면 train/serve skew가 난다. '
             f'대응이 필요하다면 투수별 F 노출도를 <b>train에서 fit → test에 transform</b>하는 '
             f'매핑으로 추가해 모델이 편향 크기를 알게 하는 쪽이다(규정 허용 범위).</div>')
    sec("signal", "1. 신호 지도 — 피처별 단변량 BSS", body)

    # 2. 피처별 패널
    for gi, (gname, cols) in enumerate(TRAIN_GROUPS, start=1):
        log(f"train 패널: {gname}")
        blocks = "".join(feature_block(df, c, bss_map.get(c)) for c in cols)
        sec(f"tg{gi}", f"2.{gi} {gname}", blocks)

    # 3. 상관
    num_cols = [c for c in feats
                if pd.api.types.is_numeric_dtype(df[c]) and c not in ID_LIKE
                and c not in ("pitcher_team_id", "batter_team_id")]
    log("상관 히트맵")
    body = img_tag(fig_corr(df, num_cols + [TARGET], method="spearman",
                            title="train.csv 수치형 피처 Spearman 상관 (30만 행 샘플)"))
    body += "<h3>상관 상위 쌍</h3>"
    body += df_to_html(top_corr_pairs(df, num_cols + [TARGET], k=18), num_cols=("|ρ|",))
    body += ('<div class="note"><b>다중공선성 경고.</b> <code>asof_pitcher_prev{1,3,5}_*</code> '
             '계열은 서로 겹치는 창(window)에서 계산되므로 구조적으로 상관이 높다. '
             'GBDT는 견디지만 선형 모델·PCA 기반 피처를 얹을 때는 대표 하나만 남기거나 '
             '차분(<code>prev1 − prev5</code>: 최근 폼의 <i>변화</i>)으로 바꿔 쓰는 편이 낫다.</div>')
    sec("corr", "3. 상관 구조", body)

    # 4. 투수 구종 공간 (train asof) — PCA
    log("투수 단위 집계 (train)")
    idx = df.groupby("pitcher_id")["asof_pitcher_n"].idxmax()
    P = df.loc[idx].set_index("pitcher_id")
    P = P[P["asof_pitcher_n"] >= 300]
    mix_cols = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
                "asof_pitcher_offspeed_rate"]
    raw = P[mix_cols].dropna()
    raw.columns = ["fastball", "breaking", "offspeed"]
    rate_by_p = df.groupby("pitcher_id")[TARGET].mean()
    n_by_p = P.loc[raw.index, "asof_pitcher_n"]
    counts = raw.mul(n_by_p, axis=0)                      # 비율 → 근사 투구 수
    mat = clr_transform(counts)
    log(f"  투수 {len(mat):,}명으로 PCA (구종 믹스 CLR)")
    res = run_pca_kmeans(mat, display=raw)
    color = rate_by_p.loc[raw.index].to_numpy()

    # 이전 판본(ball/strike 포함)과의 비교 근거
    from scipy.stats import spearmanr as _sp
    cmp_rows = []
    for c, lab in [("asof_pitcher_middle_rate", "middle_rate (타깃 성분 · 제외)"),
                   ("asof_pitcher_ball_rate", "ball_rate (이전 판본에 포함 → 제거)"),
                   ("asof_pitcher_strike_rate", "strike_rate (이전 판본에 포함 → 제거)"),
                   ("asof_pitcher_breaking_rate", "breaking_rate (구종 · 유지)"),
                   ("asof_pitcher_offspeed_rate", "offspeed_rate (구종 · 유지)"),
                   ("asof_pitcher_fastball_rate", "fastball_rate (구종 · 유지)")]:
        if c in P.columns:
            cmp_rows.append({"변수": lab,
                             "투수 성공률과 Spearman ρ": round(float(
                                 _sp(P.loc[raw.index, c], color)[0]), 3)})
    cmp_tab = pd.DataFrame(cmp_rows)

    body = ('<div class="note warn"><b>이 절은 수정판이다 — 이전 판본은 입력 변수 선정이 '
            '틀렸다.</b> 이전에는 클러스터링 입력에 <code>asof_pitcher_ball_rate</code>와 '
            '<code>asof_pitcher_strike_rate</code>를 넣었다. 두 가지가 잘못됐다. '
            '<b>①</b> 목적이 "투수가 무엇을 던지는가"인데 볼/스트라이크 비율은 '
            '<i>결과가 어땠는가</i>다 — 구종 축과 제구 축을 한 공간에 섞었다. '
            '<b>②</b> 타깃 성분이라며 <code>success/middle/reverse_rate</code>는 빼놓고 '
            '<code>ball_rate</code>는 남긴 것이 기준의 비일관 적용이다. 타깃 실패 정의 ②가 '
            '"존에서 크게 벗어난 공" = 사실상 볼이다. 아래 상관을 보라.</div>')
    body += df_to_html(cmp_tab, num_cols=("투수 성공률과 Spearman ρ",))
    body += ('<div class="note warn"><code>ball_rate</code>가 구종 변수 전부보다 타깃과 강하게 '
             '얽혀 있다. 실제로 이전 판본의 PC1(분산 41.9%)은 로딩이 '
             '<code>ball_rate</code> +0.618 / <code>strike_rate</code> −0.570으로 지배당해 '
             '<b>구종 축이 아니라 제구 축</b>이었고, 그 평면을 성공률로 색칠해 해석한 것도 '
             '그만큼 순환이었다. 원인은 참고한 Kaggle 노트북이 ERA·SO·BB 같은 <b>결과 스탯</b>으로 '
             '군집을 만드는 구조였고, 그 노트북은 예측 타깃이 없어 문제가 없었는데 이를 '
             '검토 없이 이식한 것이다.</div>')
    body += (f'<p><b>수정된 구성.</b> 각 투수의 마지막 시점 asof 프로파일에서 '
             f'<b>구종 믹스 3개만</b> 남기고, 합=1 제약은 '
             f'<b>CLR(centered log-ratio)</b>로 처리했다. CLR은 앞서 지적한 랭크 결함'
             f'(합=1 때문에 고유값 0인 허수 축이 생기던 문제)도 동시에 해결한다 — '
             f'유효 랭크가 {res.rank}로 잡혀 성분이 정확히 그만큼만 만들어진다. '
             f'표본: 누적 300투구 이상 <b>{len(mat):,}명</b>.</p>')
    body += img_tag(fig_scree(res, "구종 믹스 공간 (CLR) — scree · k 후보 · Gap"))
    body += img_tag(fig_biplot(res, color_by=color, cbar_label="투수 평균 제구 성공률",
                               title="투수 구종 공간 (PC1–PC2, CLR 좌표)"))
    body += img_tag(fig_cluster_profile(res, title="군집별 원래 구종 비율 (전체 대비 z-score)"))

    prof = res.meta.copy()
    prof.insert(0, "n", pd.Series(res.labels).value_counts().sort_index().to_numpy())
    prof["평균 성공률"] = [float(color[res.labels == c].mean()) for c in prof.index]
    prof.index = [f"C{c}" for c in prof.index]
    prof = prof.reset_index().rename(columns={"index": "cluster"})
    body += df_to_html(prof.round(4), num_cols=tuple(prof.columns[1:]))
    spread = prof["평균 성공률"].max() - prof["평균 성공률"].min()
    body += (f'<div class="note warn"><b>그런데 이 군집은 실재하지 않는다.</b> '
             f'silhouette 최대는 k={res.k}(={res.sil["silhouette"].max():.3f})지만 '
             f'<b>Gap statistic이 고르는 값은 k={res.k_gap}</b>이다'
             f'{" — 즉 나누지 말라는 뜻이다" if res.k_gap == 1 else ""}. '
             f'silhouette 절대값은 차원 수와 데이터 형태에 따라 기준이 달라져 단독으로는 '
             f'판정 근거가 못 된다. 그리고 <b>KMeans 라벨로 산점도를 색칠하면 구조가 있든 '
             f'없든 반드시 갈라져 보인다</b> — 경계는 밀도의 골짜기가 아니라 알고리즘이 그은 '
             f'Voronoi 선이다. 구조가 0인 균등난수를 같은 파이프라인에 넣어도 똑같은 그림이 나온다. '
             f'<b>색칠된 그림을 군집의 증거로 쓰지 마라.</b></div>')
    n_lo = int((raw["offspeed"] < 0.01).sum())
    n_zero = int((raw["offspeed"] == 0).sum())
    body += (f'<div class="note warn"><b>CLR 자신의 부작용도 적어둔다.</b> 이전 판본의 '
             f'silhouette 최대는 0.271이었는데 이번엔 {res.sil["silhouette"].max():.3f}로 올랐다. '
             f'구조가 새로 생긴 것이 아니라 <b>로그 변환이 왼쪽 꼬리를 늘린 것</b>이다 — '
             f'<code>offspeed</code>가 정확히 0인 투수가 {n_zero}명, 0.01 미만이 {n_lo}명이라 '
             f'이들이 log 스케일에서 멀리 밀려난다. 실제로 C1은 "오프스피드를 거의 안 쓰는 '
             f'투피치 투수" 집단이고, 이건 해석 가능한 <i>꼬리</i>이지 분리된 <i>덩어리</i>가 '
             f'아니다. Gap이 여전히 k={res.k_gap}을 고르는 것이 그 근거다. '
             f'구성비 데이터에서는 CLR의 0 대체(δ=0.5/n)가 만드는 인공 극단값을 항상 확인해야 한다.</div>')
    body += (f'<div class="note"><b>그래도 축은 쓸모가 있다.</b> 점 구름은 연속체지만 그 위에 '
             f'해석 가능한 방향이 깔려 있다. 군집 간 평균 성공률 폭은 {spread:.4f}에 불과하니 '
             f'<b>이산 라벨 대신 PC 점수를 연속값 피처로</b> 쓰는 편이 정보 손실이 없다. '
             f'클러스터를 굳이 쓴다면 용도는 하나 — <code>asof_pitcher_n</code>이 작은 신인의 '
             f'<b>cold-start fallback</b>(같은 유형 집단 평균을 사전확률로 주는 계층적 스무딩)이다.'
             f'<br><br><b>남는 한계:</b> train의 <code>asof_*</code>는 구종이 '
             f'fastball/breaking/offspeed <b>3군</b>뿐이다. 진짜 구종은 7절에서 다룬다.</div>')
    sec("pca_train", "4. 투수 구종 공간 — PCA (train asof, CLR)", body)

    del P, mat, raw
    df_pitcher_rate = rate_by_p
    del df

    # ===== trackman =====
    if not args.skip_trackman:
        tm = load_trackman(os.path.join(args.data_dir, "trackman_history.csv"))
        tfeats = [c for g, cols in TRACKMAN_GROUPS for c in cols]

        ov = (f"<p><code>trackman_history.csv</code> — <b>{len(tm):,}행 × {tm.shape[1]}컬럼</b>, "
              f"시즌 {int(tm.season.min())}~{int(tm.season.max())}. "
              "타깃 컬럼이 없으므로 이 절의 패널은 <b>분포만</b> 보여준다.</p>")
        ov += ('<div class="note"><b>단위는 미국식이 아니다.</b> 관측값으로 확인한 결과 '
               '<code>rel_speed</code> 중앙값 137·fastball 143, <code>zone_speed</code> 126 → '
               '<b>km/h</b>. <code>extension</code>·<code>rel_height</code> 중앙값 1.76, '
               '<code>rel_side</code> ±0.5 → <b>m</b>. <code>induced_vert_break</code>·'
               '<code>horz_break</code>는 ±70 범위이고 fastball IVB가 +40대 → <b>cm</b> '
               '(inch로 읽으면 물리적으로 불가능한 값이 된다). '
               'MLB Statcast 감각(mph·inch·ft)으로 임계값을 정하면 전부 틀린다. '
               f'<code>pitcher_hand</code>도 train은 정수 코드, trackman은 '
               f'<code>Right</code>/<code>Left</code> 문자열로 인코딩이 다르다.</div>')
        ov += ('<div class="note warn"><b>조인 불가.</b> <code>pitcher_trackman_id</code>는 '
               '<code>train.csv</code>의 <code>pitcher_id</code>와 다른 ID 체계다. '
               '1:1 결합 테이블이 아니라 <b>선수 단위 요약 피처의 재료</b>로만 쓸 수 있고, '
               'ID 매핑 자체가 별도 과제다. 또한 2025년 Trackman은 제공되지 않으므로 '
               '평가 시점에는 <b>2024년까지의 프로파일을 그대로 이월</b>하는 수밖에 없다.</div>')
        sec("tm_overview", "5. trackman_history 개요", ov)

        for gi, (gname, cols) in enumerate(TRACKMAN_GROUPS, start=1):
            log(f"trackman 패널: {gname}")
            blocks = "".join(feature_block(tm, c, None, has_target=False) for c in cols)
            sec(f"mg{gi}", f"5.{gi} {gname}", blocks)

        # 구종군별 물리량
        log("구종군별 물리량")
        phys = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
                "extension", "rel_height", "rel_side", "zone_speed"]
        groups = [g for g in ["fastball", "breaking", "offspeed", "other"]
                  if g in set(tm["pitch_type_group"].dropna().unique())]
        fig, axes = plt.subplots(2, 4, figsize=(15, 6.2))
        for ax, c in zip(axes.ravel(), phys):
            for i, g in enumerate(groups):
                v = tm.loc[tm["pitch_type_group"] == g, c].dropna()
                if len(v) > 200000:
                    v = v.sample(200000, random_state=SEED)
                ax.hist(v, bins=70, histtype="step", lw=1.3,
                        color=PALETTE[i], label=g, density=True)
            ax.set_title(c, loc="left", fontsize=10)
            ax.set_yticks([])
        axes.ravel()[0].legend(fontsize=8)
        fig.suptitle("구종군별 물리 측정값 분포 (density)", x=0.005, ha="left", fontsize=12)
        fig.tight_layout()
        body = img_tag(fig_to_b64(fig, dpi=95))

        # 시즌 드리프트
        fig, axes = plt.subplots(2, 4, figsize=(15, 5.6))
        for ax, c in zip(axes.ravel(), phys):
            g = tm.groupby("season")[c].median()
            ax.plot(g.index, g.to_numpy(), "-o", ms=4, color=C_LINE)
            ax.set_title(c, loc="left", fontsize=10)
            ax.set_xticks(g.index)
            ax.tick_params(axis="x", labelrotation=45)
        fig.suptitle("시즌별 중앙값 — 측정 드리프트 점검", x=0.005, ha="left", fontsize=12)
        fig.tight_layout()
        body += img_tag(fig_to_b64(fig, dpi=95))
        body += ('<div class="note"><b>왜 보는가.</b> 측정값의 시즌 중앙값이 계단식으로 움직인다면 '
                 '그것은 투수의 변화가 아니라 <b>장비·펌웨어·구장 캘리브레이션</b>의 변화일 수 있다. '
                 '그런 컬럼으로 만든 투수 프로파일을 2025 평가에 이월하면 편향이 그대로 실린다. '
                 '시즌 내 z-score 정규화로 흡수하는 편이 안전하다.</div>')
        sec("tm_phys", "6. 구종군 · 시즌별 물리량", body)

        # 투수 구종 아스널 공간 — 7종 구종 비율(CLR) × 구종별 물리량
        log("trackman 구종 아스널 집계")
        tm["_hand_sign"] = np.where(
            tm["pitcher_hand"].astype(str).str.upper().str.startswith("L"), -1.0, 1.0)
        tm["horz_break_m"] = tm["horz_break"] * tm["_hand_sign"]
        tm["rel_side_m"] = tm["rel_side"] * tm["_hand_sign"]
        tm["pt"] = tm["tagged_pitch_type"].astype(str).str.strip().str.lower().map(PITCH_CANON)
        raw_pt = tm["tagged_pitch_type"].value_counts()
        n_drop = int(tm["pt"].isna().sum())
        tmc = tm[tm["pt"].notna()].copy()
        tmc["fam"] = tmc["pt"].map(PITCH_FAMILY)

        counts = pd.crosstab(tmc["pitcher_trackman_id"], tmc["pt"])
        counts = counts.reindex(columns=PITCH_ORDER, fill_value=0)
        fam_n = pd.crosstab(tmc["pitcher_trackman_id"], tmc["fam"]).reindex(
            columns=["FB", "BR", "OFF"], fill_value=0)
        keep = (counts.sum(axis=1) >= 300) & (fam_n >= 30).all(axis=1)
        counts, fam_n = counts[keep], fam_n[keep]
        mix_clr = clr_transform(counts, prefix="")

        PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break_m",
                "extension", "rel_height", "rel_side_m"]
        g = tmc[tmc["pitcher_trackman_id"].isin(counts.index)].groupby(
            ["pitcher_trackman_id", "fam"])[PHYS].mean()
        FB = g.xs("FB", level="fam"); BR = g.xs("BR", level="fam"); OF = g.xs("OFF", level="fam")
        phys = pd.DataFrame(index=counts.index)
        for c in PHYS:
            phys[f"FB_{c}"] = FB[c]
        phys["BR_ivb"] = BR["induced_vert_break"]
        phys["BR_hb"] = BR["horz_break_m"]
        phys["BR_spin"] = BR["spin_rate"]
        phys["velo_sep_BR"] = FB["rel_speed"] - BR["rel_speed"]
        phys["velo_sep_OFF"] = FB["rel_speed"] - OF["rel_speed"]
        M = mix_clr.join(phys).dropna()
        disp = counts.div(counts.sum(axis=1), axis=0).loc[M.index].join(phys)
        log(f"  투수 {len(M):,}명 · 변수 {M.shape[1]}개")
        res2 = run_pca_kmeans(M, display=disp)

        vc = pd.DataFrame({"구종": PITCH_ORDER,
                           "투구 수": [int(counts[c].sum()) for c in PITCH_ORDER],
                           "계열": [PITCH_FAMILY[c] for c in PITCH_ORDER]})
        body = ('<p>4절은 구종이 <code>fastball/breaking/offspeed</code> 3군뿐이었다. '
                'Trackman의 <code>tagged_pitch_type</code>에는 <b>실제 구종 7종</b>이 있어 '
                '여기서 제대로 된 아스널 공간을 만든다. 대소문자·표기 변형'
                '(<code>ChangeUp</code>/<code>Changeup</code>, <code>SInker</code>, '
                '<code>Four-Seam</code>→Fastball, <code>Sweeper</code>→Slider, '
                '<code>Slurve</code>→Curveball)을 정규화했고, '
                f'<code>Undefined/Other/Knuckleball</code> {n_drop:,}투구는 제외했다.</p>')
        body += df_to_html(vc, num_cols=("투구 수",))
        body += ('<div class="note warn"><b>주의 — <code>pitch_type_group</code>이 '
                 '<code>tagged_pitch_type</code>과 어긋난다.</b> 6절에서 쓴 3군 요약은 '
                 '<code>auto_pitch_type</code> 기준으로 매겨진 것으로 보이며, tagged 기준으로는 '
                 'Slider가 fastball 군에, Fastball이 offspeed 군에 상당수 들어가 있다. '
                 '3군 요약을 쓸 거면 이 불일치를 알고 써야 한다. 이 절은 <b>tagged 기준</b>이다.</div>')
        body += (f'<p><b>행렬 구성 ({len(M):,}명 × {M.shape[1]}변수).</b> '
                 f'① 구종 7종 비율 → CLR. ② 계열별 물리량: fastball 계열의 '
                 f'{", ".join(PHYS)} 7개, breaking 계열의 무브먼트·회전 3개, '
                 f'③ 구속 분리도 <code>velo_sep_BR</code>·<code>velo_sep_OFF</code>'
                 f'(패스트볼 평균 구속 − 해당 계열 평균 구속) 2개. '
                 f'좌완은 <code>horz_break</code>·<code>rel_side</code>에 −1을 곱해 정렬했다'
                 f'(접미사 <code>_m</code>) — 이 보정 없이 돌리면 PC1이 구종이 아니라 '
                 f'<b>투수 손</b>을 찾아낸다. 표본은 총 300투구 이상이면서 FB/BR/OFF 각 계열 '
                 f'30투구 이상인 투수로 제한했다(전체 699명 중 {len(M):,}명).</p>')
        body += img_tag(fig_scree(res2, "구종 아스널 공간 — scree · k 후보 · Gap"))
        body += img_tag(fig_biplot(res2, color_by=disp["FB_rel_speed"].to_numpy(),
                                   cbar_label="fastball 평균 구속 (km/h)",
                                   title="투수 아스널 공간 (PC1–PC2)"))
        body += img_tag(fig_cluster_profile(res2, title="군집 프로파일 (z-score) — 비율은 원래 스케일"))
        p2 = res2.meta.copy()
        p2.insert(0, "n", pd.Series(res2.labels).value_counts().sort_index().to_numpy())
        p2.index = [f"C{c}" for c in p2.index]
        p2 = p2.reset_index().rename(columns={"index": "cluster"})
        body += df_to_html(p2.round(3), num_cols=tuple(p2.columns[1:]))
        ev2 = res2.pca.explained_variance_ratio_
        body += (f'<div class="note"><b>PC1이 무엇을 잡았나 — 로딩을 읽으면 팔 각도(arm slot)다.</b> '
                 f'음의 방향에 <code>Sinker</code>·<code>ChangeUp</code>·'
                 f'<code>FB_rel_side_m</code>·<code>FB_horz_break_m</code>가, 양의 방향에 '
                 f'<code>FB_rel_speed</code>·<code>FB_induced_vert_break</code>·'
                 f'<code>FB_rel_height</code>가 모여 있다. 즉 '
                 f'<b>"낮고 옆으로 벌어진 릴리스 + 싱커/횡무브"</b> ↔ '
                 f'<b>"높은 릴리스 + 라이징 패스트볼 + 빠른 구속"</b> 축이다. '
                 f'C1({int((res2.labels==1).sum())}명)은 전형적인 사이드암·스리쿼터 계열이고, '
                 f'프로파일 표에서 <code>FB_induced_vert_break</code> z=−1.5, '
                 f'<code>FB_rel_height</code> z=−1.3, <code>FB_horz_break_m</code> z=+1.2로 나타난다. '
                 f'야구 상식과 맞는 축이 자동으로 잡혔다는 점에서 이 공간 자체는 신뢰할 만하다.</div>')
        body += (f'<div class="note warn"><b>그래도 군집으로는 나누지 마라.</b> '
                 f'silhouette 최대 k={res2.k}({res2.sil["silhouette"].max():.3f}), '
                 f'Gap 선택 k={res2.k_gap}. 두 값이 어긋나면 <b>Gap 쪽을 믿어라</b> — '
                 f'silhouette은 구조가 없어도 항상 어딘가에서 최대를 만든다. '
                 f'팔 각도는 실재하지만 <b>연속체</b>다. '
                 f'또 PC1+PC2가 {(ev2[0]+ev2[1])*100:.1f}%뿐이라(유효 랭크 {res2.rank}) '
                 f'2D 산점도에서 겹쳐 보이는 두 점이 다른 축에서는 멀 수 있다. '
                 f'<b>라벨 대신 PC 점수를 연속값으로 쓰는 것이 정답이다.</b></div>')
        body += ('<div class="note"><b>이걸 대회에 어떻게 쓰나.</b> 이 좌표는 '
                 '<code>pitcher_trackman_id</code>에 붙는다. <code>train.pitcher_id</code>와 '
                 'ID 체계가 달라 그대로는 못 쓴다. 연결하려면 두 테이블에 공통으로 있는 축 — '
                 '<code>(season, 팀, 좌우, 등판 패턴, 투구 수 프로파일)</code> — 으로 '
                 '후보를 좁혀 매칭해야 하고, 이건 별도의 엔티티 해소 작업이다. '
                 '게다가 2025 Trackman은 제공되지 않아 <b>2024까지의 프로파일을 이월</b>하는 '
                 '수밖에 없다. <b>비용 대비 효과를 먼저 재라</b> — 매핑 정확도가 낮으면 '
                 '잡음만 주입된다.</div>')
        sec("pca_tm", "7. 투수 아스널 공간 — 구종 7종 × 물리량 (Trackman)", body)
        del tm, tmc

    # ===== 렌더 =====
    toc_html = '<div class="toc"><b>목차</b><ol>' + "".join(
        f'<li><a href="#{sid}">{h(t)}</a></li>' for sid, t in toc) + "</ol></div>"
    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>피처별 시각화 리포트 — 투구 제구 성공 확률 예측</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>피처별 시각화 리포트</h1>
<p class="sub">DACON 236743 · LG Aimers 9기 Phase 2 — 투구 제구 성공 확률 예측 ·
생성 {time.strftime('%Y-%m-%d %H:%M')} · 참고 구조: Kaggle
<i>MLB Pitcher Analysys | Clustering &amp; PCA</i></p>
{toc_html}
{''.join(parts)}
<p class="meta">재현: <code>python feature_viz.py --data-dir ../data</code> ·
seed={SEED} · 전체 행 사용(상관·산점도만 샘플링) · 생성 소요
{time.time()-t0:.0f}s</p>
</div></body></html>"""
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)
    log(f"완료: {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
