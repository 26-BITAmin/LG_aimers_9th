"""R/F(game_type)별 선수 통계 테이블 생성.
- game_id: row 연속 블록 + 이닝/득점 단조성으로 복원
- 투구 라벨: asof 누적률 역산 (control_success와 100% 일치 검증됨)
"""
import pandas as pd, numpy as np, os
SRC="/mnt/user-data/uploads/LG aimers 야구/data/train.csv"
OUT="out"; os.makedirs(OUT,exist_ok=True)

df=pd.read_csv(SRC)
df.columns=[c.lstrip("﻿") for c in df.columns]
df=df.sort_values("row_id").reset_index(drop=True)

# ---------- 1. game_id 복원 ----------
a=np.minimum(df.pitcher_team_id,df.batter_team_id); b=np.maximum(df.pitcher_team_id,df.batter_team_id)
key=(df.season.astype(str)+"|"+df.game_month.astype(str)+"|"+df.game_dayofweek.astype(str)+"|"
     +df.game_type+"|"+a.astype(str)+"_"+b.astype(str))
newblk=(key!=key.shift())
blk0=newblk.cumsum()
# 같은 키 블록 안에서 이닝 또는 누적득점이 감소하면 다른 경기 (더블헤더/같은요일 재대결)
inn_drop=df.inning.diff()<0
run_drop=df.run_total_before.diff()<0
split=(newblk | ((~newblk)&(inn_drop|run_drop)))
df["game_id"]=split.cumsum()
G=df.groupby("game_id")
qc=pd.DataFrame({"n":G.size(),"season":G.season.first(),"gt":G.game_type.first(),
  "inn_ok":G.inning.apply(lambda s:s.diff().fillna(0).min()>=0),
  "run_ok":G.run_total_before.apply(lambda s:s.diff().fillna(0).min()>=0)})
print(f"[game] 경기 {len(qc):,}개 | 이닝 단조 {qc.inn_ok.mean():.4f} | 득점 단조 {qc.run_ok.mean():.4f}")
print("[game] 시즌×game_type 경기 수:"); print(qc.pivot_table(index="season",columns="gt",values="n",aggfunc="size").to_string())
print("[game] 경기당 투구수 중앙값:",qc.n.median())

# ---------- 2. 투구 단위 라벨 역산 ----------
def recover(prefix,nc,modes):
    o={}
    for m in modes:
        col=f"asof_{prefix}_{m}_rate"
        if col not in df.columns: continue
        cnt=df[nc]*df[col].fillna(0)
        o[m]=(cnt.groupby(df[f"{prefix}_id"],sort=False).shift(-1)-cnt).round()
    return pd.DataFrame(o)
P=recover("pitcher","asof_pitcher_n",["success","reverse","middle","ball","strike"])
PM=recover("pitcher","asof_pitcher_pitchmix_n",["fastball","breaking","offspeed"])
B=recover("batter","asof_batter_n",["success","middle"])
chk=P.success.notna()
assert (P.success[chk]==df.control_success[chk]).all(), "역산 실패"
print(f"[label] 투수 라벨 역산 검증 OK ({chk.sum():,}행, 마지막 행 {len(df)-chk.sum():,}개는 복원 불가)")
print(f"[label] 타자 라벨 역산 검증: success 일치율 {(B.success[B.success.notna()]==df.control_success[B.success.notna()]).mean():.6f}")

X=df.join(P.add_prefix("p_")).join(PM.add_prefix("pm_")).join(B.add_prefix("b_"))
X["first_pitch"]=(X.balls_before+X.strikes_before==0).astype(int)
X["two_strike"]=(X.strikes_before==2).astype(int)
X["three_ball"]=(X.balls_before==3).astype(int)
X["risp"]=((X.runner_on_2b==1)|(X.runner_on_3b==1)).astype(int)
X["on_base"]=(X.num_runners_on>0).astype(int)
X.to_pickle("X.pkl"); print("[cache] X.pkl 저장", X.shape)
