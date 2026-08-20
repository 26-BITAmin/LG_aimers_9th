#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
월별 구종 구성 비율 플롯.

- 출처: trackman_history.csv 의 tagged_pitch_type (train.csv에는 구종 컬럼이 없다)
- 비율은 각 월 '안에서' 계산 → 월마다 합 = 100%
- 각 점 아래에 그 월/그 구종의 투구 수와 비율을 표기 (겹치면 아래로 밀고 지시선 연결)

실행: python pitchmix_by_month.py --data-dir ../data --out pitchmix_by_month.png
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for cand in ("Noto Sans CJK JP", "NanumGothic", "Malgun Gothic", "AppleGothic"):
    try:
        matplotlib.font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.family"] = cand
        break
    except Exception:
        continue
plt.rcParams.update({
    "axes.unicode_minus": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c9ccd1", "axes.grid": True, "grid.color": "#eceef2",
    "grid.linewidth": .7, "legend.frameon": False, "savefig.bbox": "tight",
    "axes.titlesize": 12, "axes.labelsize": 10,
})

CANON = {"fastball": "Fastball", "four-seam": "Fastball", "sinker": "Sinker",
         "cutter": "Cutter", "slider": "Slider", "sweeper": "Slider",
         "curveball": "Curveball", "slurve": "Curveball",
         "changeup": "ChangeUp", "splitter": "Splitter"}
ORDER = ["Fastball", "Slider", "Curveball", "ChangeUp", "Splitter", "Sinker", "Cutter"]
COLOR = dict(zip(ORDER, ["#2f6fdb", "#e4573d", "#2ea36b", "#b072dd",
                         "#e0a12c", "#0f9aa8", "#d1568f"]))
KOR = {"Fastball": "직구", "Slider": "슬라이더", "Curveball": "커브", "ChangeUp": "체인지업",
       "Splitter": "스플리터", "Sinker": "싱커", "Cutter": "커터"}


def load_counts(path: str, by_season: bool = False) -> tuple[pd.DataFrame, int]:
    cols = ["season", "game_month", "tagged_pitch_type"]
    tm = pd.read_csv(path, usecols=cols)
    tm["pt"] = tm["tagged_pitch_type"].astype(str).str.strip().str.lower().map(CANON)
    n_drop = int(tm["pt"].isna().sum())
    tm = tm[tm["pt"].notna()]
    idx = ["season", "game_month"] if by_season else "game_month"
    ct = (tm.groupby(idx)["pt"].value_counts().unstack(fill_value=0)
          .reindex(columns=ORDER, fill_value=0))
    return ct, n_drop


def place_labels(fracs: dict, fmin: float, gap: float, first_off: float,
                 mark_pad: float = 0.013) -> dict:
    """
    점 아래에 라벨을 놓되, 위에서부터 내려오며 겹치면 더 아래로 민다.
    좌표는 축 비율(0~1)로 다룬다 — 로그축이든 선형축이든 라벨 높이가 일정해지기 때문.
    라벨끼리뿐 아니라 '다른 계열의 마커'와도 겹치지 않게 밀어낸다
    (계열 값이 가까울 때 라벨이 남의 점 위에 얹히는 것을 막는다).
    """
    out, cursor = {}, np.inf
    for name in sorted(fracs, key=lambda k: -fracs[k]):
        y = min(fracs[name] - first_off, cursor - gap)
        for _ in range(len(fracs)):                       # 마커 회피(수렴할 때까지)
            hit = [m for k, m in fracs.items()
                   if k != name and (y - gap) < m < y + mark_pad]
            if not hit:
                break
            y = min(hit) - mark_pad
        out[name] = max(y, fmin)
        cursor = out[name]
    return out


def plot(ct: pd.DataFrame, n_drop: int, out: str) -> None:
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    months = list(ct.index)
    x = np.arange(len(months))
    tot = ct.sum(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(17, 13.6),
                             gridspec_kw={"height_ratios": [2.5, 1]})

    # ── 위: 선 + 점, 점 아래 (투구 수 / 비율) ────────────────────────────────
    # y축 로그: 직구 47%와 커터 2%가 한 화면에 있으면 선형축에서는 하위 구종이
    # 바닥에 뭉쳐 라벨을 놓을 자리가 없다. 로그축이면 비율 차이가 균등하게 벌어진다.
    ax = axes[0]
    for c in ORDER:
        ax.plot(x, pct[c], "-o", ms=6.5, lw=1.9, color=COLOR[c],
                label=f"{c} ({KOR[c]})", zorder=3, mec="white", mew=1.1)
    ax.set_yscale("log")
    ax.set_ylim(1.1, 75)
    ax.set_yticks([2, 3, 5, 10, 20, 50])
    ax.set_yticklabels(["2%", "3%", "5%", "10%", "20%", "50%"])
    ax.minorticks_off()
    ax.set_xlim(-0.55, len(months) - 0.45)

    fig.canvas.draw()                       # 변환 확정 후 좌표 계산
    d2a = lambda yv: ax.transAxes.inverted().transform(
        ax.transData.transform((0, yv)))[1]
    a2d = lambda fv: ax.transData.inverted().transform(
        ax.transAxes.transform((0, fv)))[1]

    for j, m in enumerate(months):
        ys = {c: float(pct.loc[m, c]) for c in ORDER}
        fr = {c: d2a(ys[c]) for c in ORDER}
        pos = place_labels(fr, fmin=0.022, gap=0.062, first_off=0.030)
        for c in ORDER:
            yd = a2d(pos[c])
            ax.annotate(f"{int(ct.loc[m, c]):,}\n{ys[c]:.2f}%", xy=(j, yd),
                        ha="center", va="top", fontsize=7.4, color=COLOR[c],
                        linespacing=1.3, zorder=4)
            if fr[c] - pos[c] > 0.040:      # 밀린 라벨은 지시선으로 연결
                ax.plot([j, j], [a2d(fr[c] - 0.012), a2d(pos[c] + 0.006)],
                        lw=0.6, color=COLOR[c], alpha=.5, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}월\nn={int(tot.loc[m]):,}" for m in months], fontsize=9.5)
    ax.set_ylabel("해당 월 내 비율 (%)  ·  로그 축")
    ax.legend(ncol=7, loc="lower center", bbox_to_anchor=(0.5, 1.005), fontsize=9.5,
              markerscale=1.1, columnspacing=1.4, handletextpad=.5)

    # ── 아래: 100% 누적 막대 (합=100%를 눈으로 확인) ────────────────────────
    ax = axes[1]
    bottom = np.zeros(len(months))
    for c in ORDER:
        v = pct[c].to_numpy()
        ax.bar(x, v, bottom=bottom, width=.68, color=COLOR[c],
               edgecolor="white", linewidth=.8, label=c)
        for j in range(len(months)):
            if v[j] >= 3.2:
                ax.text(j, bottom[j] + v[j] / 2, f"{v[j]:.1f}%", ha="center", va="center",
                        fontsize=7.6, color="white", fontweight="bold")
        bottom += v
    ax.set_xticks(x); ax.set_xticklabels([f"{m}월" for m in months], fontsize=9.5)
    ax.set_ylim(0, 100); ax.set_ylabel("누적 비율 (%)")
    ax.set_xlim(-0.55, len(months) - 0.45)
    ax.grid(axis="x", visible=False)
    ax.set_title("같은 값의 100% 누적 막대 — 위 그림의 각 월이 정확히 100%임을 확인", loc="left")

    fig.text(0.005, 0.005,
             f"출처: trackman_history.csv (2019–2024, tagged_pitch_type). "
             f"표기 변형 정규화: Four-Seam→Fastball, Sweeper→Slider, Slurve→Curveball, "
             f"ChangeUp/Changeup, SInker/Sinker.  "
             f"Undefined·Other·Knuckleball {n_drop:,}투구 제외 (전체의 {n_drop/(n_drop+ct.to_numpy().sum())*100:.2f}%).  "
             f"train.csv에는 구종 컬럼이 없어 Trackman이 유일한 출처다.",
             fontsize=8, color="#5a6070")
    fig.suptitle("월별 구종 구성 — 각 월 안에서 100%로 정규화  ·  점 아래 = 투구 수 / 비율",
                 x=0.005, ha="left", fontsize=13.5)
    fig.tight_layout(rect=[0, 0.018, 1, 0.975])
    fig.savefig(out, dpi=130)
    print(f"saved: {out}")


def plot_by_season(ct: pd.DataFrame, out: str) -> None:
    """시즌 × 월 — 월별 패턴이 시즌마다 반복되는지 확인용."""
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    seasons = sorted({s for s, _ in pct.index})
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.4), sharex=True)
    for ax, s in zip(axes.ravel(), seasons):
        sub = pct.loc[s]
        for c in ORDER:
            ax.plot(sub.index, sub[c], "-o", ms=3.6, lw=1.5, color=COLOR[c], label=c)
        ax.set_title(f"{s} 시즌", loc="left", fontsize=10.5)
        ax.set_yscale("log")
        ax.set_yticks([2, 5, 10, 20, 50])
        ax.set_yticklabels(["2%", "5%", "10%", "20%", "50%"], fontsize=8)
        ax.set_xlabel("월", fontsize=9)
    for ax in axes.ravel()[len(seasons):]:
        ax.axis("off")
    axes.ravel()[len(seasons)].legend(*axes.ravel()[0].get_legend_handles_labels(),
                                      loc="center", fontsize=10, ncol=1)
    fig.suptitle("시즌 × 월 구종 구성 (y축 로그) — 월별 변동이 매년 반복되는가",
                 x=.005, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--out", default="pitchmix_by_month.png")
    ap.add_argument("--out-season", default="pitchmix_by_month_season.png")
    a = ap.parse_args()
    path = f"{a.data_dir}/trackman_history.csv"
    ct, nd = load_counts(path)
    plot(ct, nd, a.out)
    cts, _ = load_counts(path, by_season=True)
    plot_by_season(cts, a.out_season)
