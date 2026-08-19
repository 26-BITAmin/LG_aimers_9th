# -*- coding: utf-8 -*-
"""타자 그룹화 EDA — 활용도 / 접근 스타일 / 상대 제구 / 종합 k-means.

cluster_pitchers.py와 같은 설계:
- game_type은 군집 입력에서 배제 (행을 투구 수 가중 합산)
- 상대 제구 비율은 오염 구간(2019-22 F) 제외 행에서만 계산
- 통산 200투구 미만 제외

타자 전용 차이:
- 구종 믹스 없음, 결과 라벨은 success/middle 뿐
- first_pitch_rate는 타석당 투구 수의 역수라 제외 (공선)
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42
MIN_PITCHES = 200
HERE = Path(__file__).parent

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(HERE / "batter_season.csv")


def wavg(g, col, wcol):
    w = g[wcol]
    m = g[col].notna() & w.notna() & (w > 0)
    if m.sum() == 0:
        return np.nan
    return float(np.average(g.loc[m, col], weights=g.loc[m, wcol]))


clean = df[(df["game_type"] == "R") | (df["season"] >= 2023)]

rows = []
for pid, g in df.groupby("player_id"):
    gc = clean[clean["player_id"] == pid]
    games, pa, pitches = g["games"].sum(), g["PA"].sum(), g["pitches"].sum()
    rows.append(
        {
            "player_id": pid,
            "pitches": pitches,
            "clean_pitches": gc["pitches"].sum(),
            "games": games,
            "seasons": g["season"].nunique(),
            # 활용도 축 (전체 행)
            "pa_per_game": pa / games if games else np.nan,
            "li_mean": wavg(g, "li_mean", "pitches"),
            "on_base": wavg(g, "on_base_rate", "pitches"),
            # 접근 스타일 축 (전체 행)
            "pitch_per_pa": pitches / pa if pa else np.nan,
            "two_strike": wavg(g, "two_strike_rate", "pitches"),
            "three_ball": wavg(g, "three_ball_rate", "pitches"),
            # 상대 제구 축 (깨끗한 행만)
            "success": wavg(gc, "success_rate", "pitches"),
            "middle": wavg(gc, "middle_rate", "pitches"),
        }
    )
P = pd.DataFrame(rows)

n_all = len(P)
P = P[P["pitches"] >= MIN_PITCHES].copy()
P_ctl_ok = P["clean_pitches"] >= MIN_PITCHES
print(f"타자 {n_all}명 중 {MIN_PITCHES}투구 이상 {len(P)}명, "
      f"상대 제구 축 계산 가능 {P_ctl_ok.sum()}명")

AXES = {
    "활용도": ["pa_per_game", "li_mean", "on_base"],
    "접근": ["pitch_per_pa", "two_strike", "three_ball"],
    "상대제구": ["success", "middle"],
}
AXES["종합"] = AXES["활용도"] + AXES["접근"] + AXES["상대제구"]

results = {}
sil_curves = {}
for name, cols in AXES.items():
    sub = P if name in ("활용도", "접근") else P[P_ctl_ok]
    X = sub[cols].dropna()
    idx = X.index
    Z = StandardScaler().fit_transform(X)
    sils = {}
    labels_by_k = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED)
        lab = km.fit_predict(Z)
        sils[k] = silhouette_score(Z, lab)
        labels_by_k[k] = lab
    best_k = max(sils, key=sils.get)
    results[name] = dict(idx=idx, Z=Z, cols=cols, best_k=best_k,
                         labels=labels_by_k, sils=sils, n=len(idx))
    sil_curves[name] = sils
    top = sorted(sils.items(), key=lambda t: -t[1])[:3]
    print(f"[{name}] n={len(idx)}  silhouette: "
          + "  ".join(f"k={k}:{s:.3f}" for k, s in top))


def profile(name, k):
    r = results[name]
    sub = P.loc[r["idx"]].copy()
    sub["c"] = r["labels"][k]
    cols = r["cols"] + ["pitches", "seasons"]
    prof = sub.groupby("c")[cols].mean()
    prof["n"] = sub.groupby("c").size()
    return prof.round(3)


for name in AXES:
    k = results[name]["best_k"]
    print(f"\n=== {name} (k={k}) 클러스터 프로파일 ===")
    print(profile(name, k).to_string())

for k in (3, 4):
    if k != results["종합"]["best_k"]:
        print(f"\n=== 종합 (참고 k={k}) ===")
        print(profile("종합", k).to_string())

out = P[["player_id", "pitches", "clean_pitches", "seasons"]].copy()
for name in AXES:
    r = results[name]
    lab = pd.Series(r["labels"][r["best_k"]], index=r["idx"])
    out[f"cluster_{name}"] = lab.reindex(P.index)
out.to_csv(HERE / "batter_clusters.csv", index=False, encoding="utf-8-sig")

career = pd.read_csv(HERE / "batter_career.csv")
grp = career.drop_duplicates("player_id").set_index("player_id")["player_group"]
out["player_group"] = out["player_id"].map(grp)
print("\n=== 종합 클러스터 x 기존 R/F 3분할 (해석용 교차표) ===")
print(pd.crosstab(out["cluster_종합"], out["player_group"]).to_string())

# ---------------------------------------------------------------- 그림
fig, axes = plt.subplots(2, 3, figsize=(19, 11))

ax = axes[0, 0]
for name, sils in sil_curves.items():
    ax.plot(list(sils), list(sils.values()), marker="o", label=name)
ax.set_xlabel("k"); ax.set_ylabel("silhouette"); ax.legend()
ax.set_title("실루엣 스코어 (축별)")

r = results["종합"]; k = r["best_k"]
pca = PCA(n_components=2, random_state=SEED)
XY = pca.fit_transform(r["Z"])
ax = axes[0, 1]
ax.scatter(XY[:, 0], XY[:, 1], c=r["labels"][k], cmap="tab10", s=14, alpha=0.8)
ax.set_title(f"종합 k={k} — PCA ({pca.explained_variance_ratio_.sum():.0%} 설명)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

ax = axes[0, 2]
Zdf = pd.DataFrame(r["Z"], columns=r["cols"], index=r["idx"])
Zdf["c"] = r["labels"][k]
hm = Zdf.groupby("c").mean()
im = ax.imshow(hm.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(hm.columns)), hm.columns, rotation=45, ha="right")
ax.set_yticks(range(len(hm)), [f"C{i} (n={(Zdf['c']==i).sum()})" for i in hm.index])
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("종합 클러스터 프로파일 (z-평균)")

ax = axes[1, 0]
dendrogram(linkage(r["Z"], method="ward"), ax=ax, truncate_mode="level", p=5,
           no_labels=True, color_threshold=0)
ax.set_title("계층 군집 (Ward, 종합 공간)"); ax.set_ylabel("거리")

ru = results["활용도"]
ax = axes[1, 1]
subu = P.loc[ru["idx"]]
ax.scatter(subu["pa_per_game"], subu["li_mean"], c=ru["labels"][ru["best_k"]],
           cmap="tab10", s=14, alpha=0.8)
ax.set_xlabel("경기당 타석"); ax.set_ylabel("li_mean")
ax.set_title(f"활용도 k={ru['best_k']}")

ra = results["접근"]
ax = axes[1, 2]
suba = P.loc[ra["idx"]]
ax.scatter(suba["pitch_per_pa"], suba["three_ball"], c=ra["labels"][ra["best_k"]],
           cmap="tab10", s=14, alpha=0.8)
ax.set_xlabel("타석당 투구"); ax.set_ylabel("3볼 비율")
ax.set_title(f"접근 k={ra['best_k']}")

fig.tight_layout()
fig.savefig(HERE / "cluster_batters.png", dpi=110)
print("\n저장: batter_clusters.csv, cluster_batters.png")
