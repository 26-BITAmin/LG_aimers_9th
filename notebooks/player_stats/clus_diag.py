"""rev.2 후속 진단 — '구조가 있다'와 '덩어리다'를 분리해서 검정.
① GMM BIC로 단봉/다봉  ② 경계 마진  ③ 피처 블록 제거  ④ 라벨 vs 연속축 CV AUC
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
rng=np.random.default_rng(42)
F=pd.read_pickle("rookie_clustered.pkl")
FEAT=["pitches","games","ppg_med","start_rate","entry_inn","inning_mean","li_mean",
      "F_share","first_month","span_month","clr_fb","clr_br","risp_rate"]
def prep(df,feats):
    Z=df[feats].copy()
    for c in ["pitches","games","ppg_med"]:
        if c in feats: Z[c]=np.log1p(Z[c])
    return StandardScaler().fit_transform(Z.values)
Z=prep(F,FEAT); PC=PCA().fit_transform(Z)

print("=== ② 경계 마진 ===")
for k in [2,3]:
    km=KMeans(k,n_init=20,random_state=0).fit(Z); ss=silhouette_samples(Z,km.labels_)
    d=np.sort(((Z[:,None,:]-km.cluster_centers_[None])**2).sum(-1)**.5,axis=1)
    mg=(d[:,1]-d[:,0])/d[:,1]
    print(f"  k={k}: sil<0 {(ss<0).mean():.1%} | sil<0.1 {(ss<0.1).mean():.1%} | 마진<20% {(mg<0.20).mean():.1%}")

print("\n=== ① GMM BIC (단봉/다봉) ===")
def gmmbic(x,name):
    x=np.asarray(x).reshape(-1,1)
    o=[GaussianMixture(n,n_init=10,random_state=0).fit(x).bic(x) for n in [1,2,3]]
    print(f"  {name:22s} ΔBIC(k1-k2) {o[0]-o[1]:+9.1f} → 선택 k={[1,2,3][int(np.argmin(o))]}")
gmmbic(PC[:,0],"PC1"); gmmbic(PC[:,1],"PC2"); gmmbic(F.start_rate,"start_rate")
Zs=Z.copy()
for j in range(Zs.shape[1]): Zs[:,j]=rng.permutation(Zs[:,j])
gmmbic(PCA().fit_transform(Zs)[:,0],"PC1 (셔플 대조)")

print("\n=== ③ 피처 블록 제거 ===")
def score(feats,label):
    Zx=prep(F,feats); sil=silhouette_score(Zx,KMeans(2,n_init=10,random_state=0).fit_predict(Zx))
    Zh=Zx.copy()
    for j in range(Zh.shape[1]): Zh[:,j]=rng.permutation(Zh[:,j])
    sh=silhouette_score(Zh,KMeans(2,n_init=10,random_state=0).fit_predict(Zh))
    print(f"  {label:30s} sil {sil:.3f} | 셔플 {sh:.3f} | 초과 {sil-sh:+.3f}")
score(FEAT,"전체 13피처")
score(["start_rate","entry_inn","inning_mean"],"역할 블록만")
score([f for f in FEAT if f not in ["start_rate","entry_inn","inning_mean"]],"역할 블록 제거")
score([f for f in FEAT if f!="F_share"],"F_share 제거")

print("\n=== ④ 생존 예측 CV AUC (데뷔연도 항상 포함) ===")
V=F[F._debut<=2023]; y=V._survived.values
yr=pd.get_dummies(V._debut.astype(str),prefix="y",drop_first=True).astype(float).values
def cv(X,name):
    Xs=np.column_stack([yr,StandardScaler().fit_transform(X)])
    s=cross_val_score(LogisticRegression(max_iter=3000),Xs,y,
                      cv=StratifiedKFold(5,shuffle=True,random_state=0),scoring="roc_auc")
    print(f"  {name:32s} AUC {s.mean():.3f} ± {s.std():.3f}")
cv(np.zeros((len(V),1)),"데뷔연도만")
cv(pd.get_dummies(V.k2,prefix="c",drop_first=True).astype(float).values,"k=2 라벨")
cv(V[["PC1","PC2"]].values,"PC1+PC2")
cv(V[["PC1","PC2","PC3"]].values,"PC1+PC2+PC3")
cv(pd.get_dummies(V.k3,prefix="c",drop_first=True).astype(float).values,"k=3 라벨")
Zv=V[FEAT].copy()
for c in ["pitches","games","ppg_med"]: Zv[c]=np.log1p(Zv[c])
cv(Zv.values,"원피처 13개")
