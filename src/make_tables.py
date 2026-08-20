import pandas as pd, numpy as np, os
X=pd.read_pickle("X.pkl"); OUT="out"
_k=(X.game_id.astype(str)+"|"+X.pitcher_id.astype(str)+"|"+X.batter_id.astype(str))
X["pa_id"]=(_k!=_k.shift()).cumsum()
print("[PA] 복원 타석 %s개 | 타석당 투구수 중앙값 %.1f"%(f"{X.pa_id.nunique():,}",X.groupby("pa_id").size().median())); os.makedirs(OUT,exist_ok=True)

PLAB=["p_success","p_reverse","p_middle","p_ball","p_strike"]
PMIX=["pm_fastball","pm_breaking","pm_offspeed"]
SIT=["first_pitch","two_strike","three_ball","on_base","risp"]

def build(side,keys):
    idc=f"{side}_id"; lab=PLAB+PMIX if side=="pitcher" else ["b_success","b_middle"]
    g=X.groupby(keys+[idc],observed=True)
    out=pd.DataFrame({
        "games":  g.game_id.nunique(),
        "pitches":g.size(),
        "seasons":g.season.nunique(),
        "teams":  g[f"{side}_team_id"].nunique(),
        "PA": g.pa_id.nunique(),
        "li_mean":g.li.mean(),
        "inning_mean":g.inning.mean(),
    })
    # 경기당 투구수
    pg=X.groupby(keys+[idc,"game_id"],observed=True).size().rename("n").reset_index()
    a=pg.groupby(keys+[idc],observed=True).n.agg(["mean","median","max"])
    out["pitch_per_game_mean"]=a["mean"]; out["pitch_per_game_med"]=a["median"]; out["pitch_per_game_max"]=a["max"]
    if side=="pitcher":   # 선발 프록시: 해당 경기 첫 투구의 이닝
        fi=X.sort_values("row_id").groupby(keys+[idc,"game_id"],observed=True).inning.first().rename("fi").reset_index()
        out["start_rate"]=fi.assign(s=(fi.fi==1).astype(int)).groupby(keys+[idc],observed=True).s.mean()
        out["entry_inning_med"]=fi.groupby(keys+[idc],observed=True).fi.median()
    for c in lab+SIT:
        out[c.replace("p_","").replace("pm_","").replace("b_","")+"_rate"]=g[c].mean()
    if side=="pitcher":
        out["batter_hand_mean"]=g.batter_hand.mean()
    out=out.reset_index().rename(columns={idc:"player_id"})
    out.insert(1,"role",side)
    return out

tabs={}
for side in ["pitcher","batter"]:
    tabs[f"{side}_season"]=build(side,["season","game_type"])
    tabs[f"{side}_career"]=build(side,["game_type"])

# ---- 검증: 역산 통산 성공률 vs 원 asof 마지막 값 ----
last=X.sort_values("row_id").groupby("pitcher_id").tail(1).set_index("pitcher_id")
car=tabs["pitcher_career"].groupby("player_id").apply(
    lambda d:(d.success_rate*d.pitches).sum()/d.pitches.sum(),include_groups=False)
cmp=pd.DataFrame({"역산":car,"asof마지막":last.asof_pitcher_success_rate}).dropna()
print(f"[검증] 통산 성공률 역산 vs asof 마지막값: 최대편차 {np.abs(cmp.역산-cmp['asof마지막']).max():.5f} (투수 {len(cmp)}명)")
print(f"[검증] 투구수 합계 {tabs['pitcher_career'].pitches.sum():,} = train {len(X):,}")
pc=tabs["pitcher_career"]
print("[검증] 등판 경기수 합계: R %s / F %s" % (f"{pc[pc.game_type=='R'].games.sum():,}", f"{pc[pc.game_type=='F'].games.sum():,}"))

for k,v in tabs.items():
    for c in v.columns:
        if v[c].dtype.kind=="f": v[c]=v[c].round(4)
    p=f"{OUT}/{k}.csv"; v.to_csv(p,index=False,encoding="utf-8-sig")
    print(f"[out] {p:32s} {v.shape}")
