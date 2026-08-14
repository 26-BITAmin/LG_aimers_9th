"""EDA 그림 — 내가 직접 보기 위한 PNG 렌더. 표로는 안 보이는 모양을 본다.

추가 진단 하나 (아직 그려본 적 없음):
    **투수 단위 예측 vs 실제.** 모델이 투수 축 천장의 92%에 있다는 것은 알았지만,
    나머지 8%가 "덜 퍼뜨려서"인지 "엉뚱한 투수를 틀려서"인지는 모른다.
    per-pitcher 예측 평균과 실제 성공률을 뿌리면 그 둘이 갈린다:
      · 회귀선 기울기 < 1  → **수축(under-dispersion)**. 퍼뜨리면 번다
      · 기울기 ≈ 1, 산포 큼 → 순위를 틀린 것. 퍼뜨리면 오히려 손해

실행:  .venv/Scripts/python.exe -u src/eda_plot.py
산출:  experiments/fig/*.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import foldcache
from config import EXP_DIR, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import shift_to_prior

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 9,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": .8,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#3d474d",
    "xtick.color": "#6a747a", "ytick.color": "#6a747a",
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "figure.facecolor": "white", "axes.facecolor": "white",
})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
FIG = EXP_DIR / "fig"
FIG.mkdir(parents=True, exist_ok=True)
V = json.loads((EXP_DIR / "viz.json").read_text(encoding="utf-8"))
SS = ["2019", "2020", "2021", "2022", "2023", "2024"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


# ── 1. 반응 곡선 4종 ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, col in zip(axes.ravel(), list(V["curves"])):
    c = V["curves"][col]
    x = np.arange(len(c["edges"]) + 1)
    for i, s in enumerate(SS):
        y = np.array([np.nan if v is None else v for v in c["by_season"][s]], float)
        ax.plot(x, y, color=C[i], lw=1.6, marker="o", ms=2.6, label=s)
    ax.set_title(col, fontsize=9.5)
    ax.set_xlabel("분위 빈 (공통 경계)")
    ax.set_ylabel("P(제구 성공)")
axes[0, 0].legend(ncol=3, fontsize=8, frameon=False)
fig.suptitle("타깃 반응 곡선 — 시즌별로 겹쳐 그림 (모양이 같은가, 높이만 다른가)",
             fontsize=11, fontweight="bold")
save(fig, "01_response_curves.png")

# ── 2. 같은 곡선을 시즌 평균으로 정렬 (레벨 제거) ────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, col in zip(axes.ravel(), list(V["curves"])):
    c = V["curves"][col]
    x = np.arange(len(c["edges"]) + 1)
    for i, s in enumerate(SS):
        y = np.array([np.nan if v is None else v for v in c["by_season"][s]], float)
        ax.plot(x, y - np.nanmean(y), color=C[i], lw=1.6, marker="o", ms=2.6, label=s)
    ax.axhline(0, color="#898781", lw=.8)
    ax.set_title(col, fontsize=9.5)
    ax.set_ylabel("시즌 평균 대비")
axes[0, 0].legend(ncol=3, fontsize=8, frameon=False)
fig.suptitle("같은 곡선에서 시즌 레벨만 뺐다 — 겹치면 '드리프트는 레벨뿐'이 참",
             fontsize=11, fontweight="bold")
save(fig, "02_response_centered.png")

# ── 3. 신호 드리프트 · 월별 ─────────────────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
short = {"asof_pitcher_success_rate": "success", "asof_pitcher_reverse_rate": "reverse",
         "asof_pitcher_prev5_game_success_rate": "prev5",
         "asof_pitcher_prev3_game_success_rate": "prev3",
         "asof_pitcher_ball_rate": "ball",
         "asof_pitcher_prev1_game_success_rate": "prev1"}
for i, (k, lab) in enumerate(short.items()):
    if k in V["drift"]:
        a1.plot([2021, 2022, 2023, 2024], V["drift"][k], color=C[i], lw=1.8,
                marker="o", ms=4, label=lab)
a1.axhline(0, color="#898781", lw=.8)
a1.set_title("컬럼별 단독 BSS, 검증 시즌별"); a1.set_ylabel("BSS oracle")
a1.legend(fontsize=8, frameon=False, ncol=2)
mo = V["month"]
for i, s in enumerate(SS):
    y = np.array([np.nan if v is None else v for v in mo["by_season"][s]], float)
    a2.plot(mo["months"], y, color=C[i], lw=1.8, marker="o", ms=4, label=s)
a2.set_title("시즌 내 월별 제구 성공률"); a2.set_xlabel("월")
a2.legend(fontsize=8, frameon=False, ncol=3)
save(fig, "03_drift.png")

# ── 4. 투수 단위 예측 vs 실제 (새 진단) ─────────────────────────
print("\n투수 단위 진단 계산 중…")
cached = foldcache.load([2023, 2024], verbose=False)
df = load_csv(TRAIN_CSV)
season = df["season"].to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
stats = {}
for ax, fold in zip(axes, (2023, 2024)):
    z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
    y = cached[fold][2]
    p = shift_to_prior(1.0 / (1.0 + np.exp(-z)), y.mean())
    d = df.loc[season == fold].reset_index(drop=True)
    pid = d["pitcher_id"].to_numpy()
    u, inv = np.unique(pid, return_inverse=True)
    n = np.bincount(inv).astype(float)
    act = np.bincount(inv, weights=y) / n
    pred = np.bincount(inv, weights=p) / n
    big = n >= 200
    A, P, N = act[big], pred[big], n[big]

    # 가중 최소제곱 기울기 (실제 = a + b·예측). b<1 이면 모델이 덜 퍼뜨린 것
    w = N / N.sum()
    mp, ma = np.sum(w * P), np.sum(w * A)
    b = np.sum(w * (P - mp) * (A - ma)) / np.sum(w * (P - mp) ** 2)
    a = ma - b * mp
    # 예측 산포 vs 실제 산포 (실제는 표본 노이즈 제거)
    sd_p = np.sqrt(np.sum(w * (P - mp) ** 2))
    var_a = np.sum(w * (A - ma) ** 2) - np.sum(w * A * (1 - A) / np.maximum(N - 1, 1))
    sd_a = np.sqrt(max(var_a, 0))
    stats[fold] = dict(slope=b, sd_pred=sd_p, sd_true=sd_a, n=int(big.sum()))

    ax.scatter(P, A, s=np.sqrt(N) * .9, c=C[0], alpha=.55, edgecolors="white", lw=.6)
    lo, hi = 0.38, 0.60
    ax.plot([lo, hi], [lo, hi], color="#898781", ls="--", lw=1.2, label="y = x")
    xs = np.linspace(lo, hi, 2)
    ax.plot(xs, a + b * xs, color=C[1], lw=2, label=f"적합 기울기 {b:.2f}")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("모델 예측 평균 (투수별)"); ax.set_ylabel("실제 성공률 (투수별)")
    ax.set_title(f"fold{fold} · 200투구 이상 {big.sum()}명\n"
                 f"예측 산포 {sd_p:.4f}  vs  실제 참 산포 {sd_a:.4f}")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("투수 단위 예측 vs 실제 — 기울기가 1보다 크면 모델이 '덜 퍼뜨린' 것",
             fontsize=11, fontweight="bold")
save(fig, "04_pitcher_calibration.png")

print("\n=== 투수 단위 진단 ===")
for f, s in stats.items():
    print(f"  fold{f}: 기울기 {s['slope']:.3f}   예측 산포 {s['sd_pred']:.4f}"
          f"   실제 참 산포 {s['sd_true']:.4f}"
          f"   비율 {s['sd_pred']/s['sd_true']:.3f}   n={s['n']}")
print("\n  비율 < 1 이면 모델이 투수 차이를 실제보다 좁게 본다 = 퍼뜨릴 여지가 있다.")
(EXP_DIR / "pitcher_diag.json").write_text(json.dumps(stats), encoding="utf-8")

# ── 5. 신뢰도 · 투수 분포 ───────────────────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
r = V["reliability"]
a1.plot([0.40, 0.58], [0.40, 0.58], color="#898781", ls="--", lw=1.2)
a1.plot([q["pred"] for q in r], [q["actual"] for q in r], color=C[0], lw=1.8,
        marker="o", ms=4.5)
a1.set_title("신뢰도 곡선 (배포 모델, fold2024)")
a1.set_xlabel("예측 확률"); a1.set_ylabel("실제 비율")
for k, s in enumerate(["2023", "2024"]):
    a2.hist(V["pitcher_rates"][s]["rates"], bins=np.linspace(.36, .64, 29),
            color=C[k], alpha=.62, label=f"{s} ({V['pitcher_rates'][s]['n_pitchers']}명)")
a2.set_title("투수별 시즌 제구 성공률 분포 (200투구 이상)")
a2.set_xlabel("시즌 성공률"); a2.legend(fontsize=8, frameon=False)
save(fig, "05_reliability_pitchers.png")

print(f"\n그림 저장 위치: {FIG}")
