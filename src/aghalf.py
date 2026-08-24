# -*- coding: utf-8 -*-
import sys; sys.path.insert(0,"src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import numpy as np
import deploy29 as D
from calslope import EPS, build_layers, calib_slope
from calsub import make_buckets, EDGE_SEASON
from predict import PROB_CENTER, PROB_SCALE
from validation import bss, shift_to_prior

df, s, y = D.load_df()
edges = np.quantile(df.loc[s < EDGE_SEASON, "asof_pitcher_n"].to_numpy(float), [.2,.4,.6,.8])
print("월 분포:", df.groupby("game_month").size().to_dict())

def fit_a(yv, p, cl, n_id, bk, k, sub=None):
    m0 = np.ones(len(yv), bool) if sub is None else sub
    a = np.zeros(k); w = np.zeros(k)
    for g in range(k):
        m = m0 & (bk == g)
        w[g] = m.sum()
        if m.sum() < 1500: continue
        a[g] = calib_slope(yv[m], p[m], groups=cl[m], n_g=n_id)["a_large"]
    return a - (a*w).sum()/w.sum(), w

seq = []; occ = {}
for fold in D.FOLDS:
    st = D.state(df, s, y, fold)
    yv, cl, n_id, mm = st["yv"], st["cl"], st["n_id"], st["mm"]
    ax = st["Xax"] @ st["dep"]; ybar = yv.mean()
    bk, lab = make_buckets(df, s, fold, mm, "apn5", edges)
    # 두 fold 를 **같은 s29 층**에서 — 층 인자 0.891 을 미리 제거
    nm, zb = [t for t in build_layers(df, s, fold, mm) if t[0]=="s29+cat10"][0]
    pb = shift_to_prior(1/(1+np.exp(-(zb+ax))), ybar)
    pd_ = shift_to_prior(PROB_CENTER + PROB_SCALE*(pb-PROB_CENTER), ybar)
    mo = df[s==fold]["game_month"].to_numpy()[mm]
    a_all, w = fit_a(yv, pd_, cl, n_id, bk, len(lab))
    occ[fold] = w/w.sum()
    for h,(lo,hi) in enumerate([(0,6),(7,12)]):
        sub = (mo>=lo)&(mo<=hi)
        a,_ = fit_a(yv, pd_, cl, n_id, bk, len(lab), sub)
        seq.append((f"{fold}H{h+1}", int(sub.sum()), a))
    seq.append((f"{fold}전체", int(len(yv)), a_all))

print("\n반시즌별 a_g (전부 s29+cat10 층, 같은 경계)")
print(f"  {'구간':<10s}{'행':>9s}  " + "".join(f"{l:>12s}" for l in lab) + f"{'‖a‖':>9s}")
for nm_, n, a in seq:
    print(f"  {nm_:<10s}{n:>9,d}  " + "".join(f"{v:>12.4f}" for v in a)
          + f"{np.sqrt((a*a).sum()):>9.4f}")

print("\n연속 전환의 원점 회귀 스케일비 c (a_next ~ c·a_prev)")
half = [t for t in seq if "H" in t[0]]
for i in range(len(half)-1):
    a0, a1 = half[i][2], half[i+1][2]
    c = (a0*a1).sum()/(a0*a0).sum(); r = np.corrcoef(a0,a1)[0,1]
    print(f"  {half[i][0]} -> {half[i+1][0]:<8s} c = {c:6.3f}   corr = {r:6.3f}")
a23 = [t for t in seq if t[0]=="2023전체"][0][2]; a24 = [t for t in seq if t[0]=="2024전체"][0][2]
print(f"  2023전체 -> 2024전체   c = {(a23*a24).sum()/(a23*a23).sum():6.3f}"
      f"   corr = {np.corrcoef(a23,a24)[0,1]:6.3f}   <-- 연간 (층 일치)")

print("\n=== 항목4: 버킷 점유율 드리프트와 중심화 잔차 ===")
for f in D.FOLDS: print(f"  fold{f} 점유율 " + " ".join(f"{v:.4f}" for v in occ[f]))
drift = occ[2024]-occ[2023]
print(f"  드리프트 최대 {np.abs(drift).max():.4f}")
mean_shift = float((a23*occ[2024]).sum())    # 2023 계수를 2024 점유율에 적용
print(f"  2023 a_g 를 2024 점유율로 재가중한 가중평균 = {mean_shift:+.6f} 로짓")
print(f"  κ=0.5 적용 시 = {0.5*mean_shift:+.6f} 로짓  ≈ {0.5*mean_shift/4:+.6f} 확률")
K = 1e5/(0.4824*(1-0.4824))
print(f"  레벨 항등식 손실 K·Δ² = {K*(0.5*mean_shift/4)**2:.4f} BSS")
