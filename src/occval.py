# -*- coding: utf-8 -*-
import sys; sys.path.insert(0,"src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import numpy as np, pandas as pd
from config import TRAIN_CSV
E = np.array([413.0, 997.0, 1843.0, 3323.0])

df = pd.read_csv(TRAIN_CSV, usecols=["season","pitcher_id","asof_pitcher_n"])
print("실측 시즌별 버킷 점유율 (동결 경계)")
print(f"  {'시즌':<6s}" + "".join(f"{l:>9s}" for l in ["<413","413-997","997-1843","1843-3323",">=3323"]) + f"{'행':>10s}")
occ = {}
for s in sorted(df.season.unique()):
    v = df.loc[df.season==s,"asof_pitcher_n"].to_numpy(float)
    o = np.bincount(np.digitize(v,E), minlength=5)/len(v)
    occ[s]=o
    print(f"  {s:<6d}" + "".join(f"{x:>9.4f}" for x in o) + f"{len(v):>10,d}")

# 더미의 노화 변환을 2023 -> "2024 예측" 으로 걸어 실제 2024 와 대조
# n' = n_off(2024) + (n - n_off(2023)),  n_off = 그 시즌 시작 시점 누적
def n_off(season):
    # 각 투수의 season 시작 시점 누적 = (season-1 까지의 마지막 asof_n + 그 행의 기여)
    # 근사: season 행들의 min(asof_n) 을 그 시즌 시작 누적으로 쓴다
    d = df[df.season==season].groupby("pitcher_id")["asof_pitcher_n"].min()
    return d
o23, o24 = n_off(2023), n_off(2024)
d23 = df[df.season==2023]
ids = d23.pitcher_id.to_numpy()
base23 = d23.pitcher_id.map(o23).to_numpy(float)
base24 = d23.pitcher_id.map(o24).to_numpy(float)     # 2023 투수 중 2024 에도 던진 사람만 유효
ok = ~np.isnan(base24)
n_new = base24[ok] + (d23.asof_pitcher_n.to_numpy(float)[ok] - base23[ok])
pred = np.bincount(np.digitize(n_new,E), minlength=5)/ok.sum()
print(f"\n더미식 노화 변환: 2023 -> 2024 예측  (2023 투수 중 2024 재등판 {ok.mean():.1%} 만 남는다)")
print("  더미 예측 " + "".join(f"{x:>9.4f}" for x in pred))
print("  실제 2024 " + "".join(f"{x:>9.4f}" for x in occ[2024]))
print("  오차      " + "".join(f"{x:>+9.4f}" for x in pred-occ[2024]))
print(f"\n  최저 버킷: 더미 예측 {pred[0]:.4f} vs 실제 {occ[2024][0]:.4f}"
      f"  ->  {pred[0]/occ[2024][0]:.2f}배")
