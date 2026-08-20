# -*- coding: utf-8 -*-
"""투수 그룹화 EDA — 역할 / 레퍼토리 / 제구 스타일 / 종합 k-means.

입력: pitcher_season.csv (투수 x game_type x season)
출력: pitcher_clusters.csv, cluster_pitchers.png, stdout 로그

설계:
- game_type은 군집 입력에서 배제 (시즌 x game_type 행을 투구 수 가중으로 합산)
- 제구 비율은 오염 구간(2019-22 F) 제외한 행에서만 계산 (README 참조)
- 통산 200투구 미만 제외 (start_rate 검증과 같은 기준)
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

df = pd.read_csv(HERE / "pitcher_season.csv")

# ---------------------------------------------------------------- 집계
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
    rows.append(
        {
            "player_id": pid,
            "pitches": g["pitches"].sum(),
            "clean_pitches": gc["pitches"].sum(),
            "games": g["games"].sum(),
            "seasons": g["season"].nunique(),
            # 역할 축 (전체 행)
            "start_rate": wavg(g, "start_rate", "games"),
            "entry_inning": wavg(g, "entry_inning_med", "games"),
            "ppg_med": wavg(g, "pitch_per_game_med", "games"),
            "li_mean": wavg(g, "li_mean", "pitches"),
            # 레퍼토리 축 (전체 행 — 구종 라벨은 오염 무관)
            "fastball": wavg(g, "fastball_rate", "pitches"),
            "breaking": wavg(g, "breaking_rate", "pitches"),
            "offspeed": wavg(g, "offspeed_rate", "pitches"),
            # 제구 축 (깨끗한 행만)
            "success": wavg(gc, "success_rate", "pitches"),
            "reverse": wavg(gc, "reverse_rate", "pitches"),
            "middle": wavg(gc, "middle_rate", "pitches"),
            "ball": wavg(gc, "ball_rate", "pitches"),
        }
    )
P = pd.DataFrame(rows)

mix = P[["fastball", "breaking", "offspeed"]].clip(lower=1e-9)
P["mix_entropy"] = -(mix * np.log(mix)).sum(axis=1)

n_all = len(P)
P = P[P["pitches"] >= MIN_PITCHES].copy()
n_kept = len(P)
# 제구 축은 깨끗한 투구도 충분해야 한다
P_ctl_ok = P["clean_pitches"] >= MIN_PITCHES
print(f"투수 {n_all}명 중 {MIN_PITCHES}투구 이상 {n_kept}명, "
      f"제구 축 계산 가능(깨끗한 투구 {MIN_PITCHES}+) {P_ctl_ok.sum()}명")

AXES = {
    "역할": ["start_rate", "entry_inning", "ppg_med", "li_mean"],
    "레퍼토리": ["fastball", "offspeed", "mix_entropy"],  # breaking = 1-f-o 라 2 자유도 + 균형도
    "제구": ["success", "reverse", "middle", "ball"],
}
AXES["종합"] = AXES["역할"] + AXES["레퍼토리"] + AXES["제구"]

# ---------------------------------------------------------------- k 스윕
results = {}
sil_curves = {}
for name, cols in AXES.items():
    sub = P if name in ("역할", "레퍼토리") else P[P_ctl_ok]
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

# ---------------------------------------------------------------- 프로파일
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

# 종합 축은 최적 k 외에 k=4,5도 참고 출력
for k in (4, 5):
    if k != results["종합"]["best_k"]:
        print(f"\n=== 종합 (참고 k={k}) ===")
        print(profile("종합", k).to_string())

# ---------------------------------------------------------------- 저장
out = P[["player_id", "pitches", "clean_pitches", "seasons"]].copy()
for name in AXES:
    r = results[name]
    lab = pd.Series(r["labels"][r["best_k"]], index=r["idx"])
    out[f"cluster_{name}"] = lab.reindex(P.index)
out.to_csv(HERE / "pitcher_clusters.csv", index=False, encoding="utf-8-sig")

# 기존 R/F 3분할과 교차 (입력 아님 — 해석용)
career = pd.read_csv(HERE / "pitcher_career.csv")
grp = career.drop_duplicates("player_id").set_index("player_id")["player_group"]
out["player_group"] = out["player_id"].map(grp)
print("\n=== 종합 클러스터 x 기존 R/F 3분할 (해석용 교차표) ===")
print(pd.crosstab(out[f"cluster_종합"], out["player_group"]).to_string())

# ---------------------------------------------------------------- 그림
fig, axes = plt.subplots(2, 3, figsize=(19, 11))

ax = axes[0, 0]
for name, sils in sil_curves.items():
    ax.plot(list(sils), list(sils.values()), marker="o", label=name)
ax.set_xlabel("k"); ax.set_ylabel("silhouette"); ax.legend()
ax.set_title("실루엣 스코어 (축별)")

# 종합: PCA 산점도
r = results["종합"]; k = r["best_k"]
pca = PCA(n_components=2, random_state=SEED)
XY = pca.fit_transform(r["Z"])
ax = axes[0, 1]
sc = ax.scatter(XY[:, 0], XY[:, 1], c=r["labels"][k], cmap="tab10", s=14, alpha=0.8)
ax.set_title(f"종합 k={k} — PCA ({pca.explained_variance_ratio_.sum():.0%} 설명)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

# 종합: 프로파일 히트맵 (z-score 평균)
ax = axes[0, 2]
sub = P.loc[r["idx"]]
Zdf = pd.DataFrame(r["Z"], columns=r["cols"], index=r["idx"])
Zdf["c"] = r["labels"][k]
hm = Zdf.groupby("c").mean()
im = ax.imshow(hm.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(hm.columns)), hm.columns, rotation=45, ha="right")
ax.set_yticks(range(len(hm)), [f"C{i} (n={(Zdf['c']==i).sum()})" for i in hm.index])
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("종합 클러스터 프로파일 (z-평균)")

# 덴드로그램 (Ward)
ax = axes[1, 0]
dendrogram(linkage(r["Z"], method="ward"), ax=ax, truncate_mode="level", p=5,
           no_labels=True, color_threshold=0)
ax.set_title("계층 군집 (Ward, 종합 공간)"); ax.set_ylabel("거리")

# 역할 축: start_rate vs 경기당 투구
rr = results["역할"]
ax = axes[1, 1]
subr = P.loc[rr["idx"]]
ax.scatter(subr["start_rate"], subr["ppg_med"], c=rr["labels"][rr["best_k"]],
           cmap="tab10", s=14, alpha=0.8)
ax.set_xlabel("start_rate"); ax.set_ylabel("경기당 투구 (중앙)")
ax.set_title(f"역할 k={rr['best_k']}")

# 제구 축: reverse vs middle
rc = results["제구"]
ax = axes[1, 2]
subc = P.loc[rc["idx"]]
ax.scatter(subc["reverse"], subc["middle"], c=rc["labels"][rc["best_k"]],
           cmap="tab10", s=14, alpha=0.8)
ax.set_xlabel("reverse_rate"); ax.set_ylabel("middle_rate")
ax.set_title(f"제구 k={rc['best_k']}")

fig.tight_layout()
fig.savefig(HERE / "cluster_pitchers.png", dpi=110)
print("\n저장: pitcher_clusters.csv, cluster_pitchers.png")
