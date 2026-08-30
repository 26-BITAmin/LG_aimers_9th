"""라우팅 축의 LB 앵커 닫힌식 — 판정 게이트 G1~G4 를 한 번에 실행한다.

  G1 형식 검증   라우팅은 로짓 공간 선형 블렌드다. p(t) 가 t 에 선형이면
                 Brier(t) = Brier0 + 2u*M + u^2*G 가 **근사가 아니라 항등식**이다.
                 -> 여러 t 에서 g=dp/dt 를 재고 corr / rms 비로 확인한다.
  G2 곡률        C = K*mean(g^2). **라벨이 필요 없다.**
                 PROB_SHIFT 는 확률공간 덧셈 상수라 예측 후 평균 제거 = 재보정.
  G3 방향        M 은 라벨이 필요하다 -> **같은 층의 LB 실측점 한 개**로 역산한다.
                 (2024 라벨로 재면 부호가 반대다. 그것이 이 도구가 존재하는 이유다.)
  G4 마진        예측이 뒤집히는 데 필요한 C 배율을, 실측 시즌 변동과 비교한다.

근거: reports/12_axis_reopening_report.md 라운드 23
"""

import os, re, subprocess, sys
import numpy as np, pandas as pd

SRC = "/Users/hyejin/Aimers_9th/outputs/hj09_grbc.zip"
TRAIN = "/Users/hyejin/Aimers_9th/data/train.csv"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rcache")
TS = [-0.25, -0.70, -0.90, -1.10, -1.3333]
R_RE = re.compile(r"(z\[is_r\]\s*=\s*)([0-9.]+)(\s*\*\s*z\[is_r\]\s*\+\s*)([0-9.]+)(\s*\*\s*zg\[is_r\])")
F_RE = re.compile(r"(z\[is_f\]\s*=\s*)([0-9.]+)(\s*\*\s*z\[is_f\]\s*\+\s*)([0-9.]+)(\s*\*\s*zf\[is_f\])")
os.makedirs(CACHE, exist_ok=True)

proxy = os.path.join(CACHE, "proxy.pkl"); ypath = os.path.join(CACHE, "y.npy")
if not os.path.exists(proxy):
    df = pd.read_csv(TRAIN); df.columns = [c.lstrip("﻿") for c in df.columns]
    df = df[df["season"] == 2024].reset_index(drop=True)
    np.save(ypath, df["control_success"].to_numpy(float))
    df.drop(columns=["control_success"]).to_pickle(proxy)
y = np.load(ypath)
print(f"프록시 2024 행 {len(y):,}")

P = {}
for t in TS:
    out = os.path.join(CACHE, f"p{t}.npy")
    R, F = 0.10 + t * 0.075, 0.40 + t * 0.10
    if not os.path.exists(out):
        w = os.path.join(CACHE, f"w{t}")
        subprocess.run(["unzip", "-oq", SRC, "-d", w], check=True)
        pp = os.path.join(w, "predict.py"); s = open(pp).read()
        s = R_RE.sub(lambda m: f"{m.group(1)}{1-R:.6f}{m.group(3)}{max(R,0):.6f}{m.group(5)}", s)
        s = F_RE.sub(lambda m: f"{m.group(1)}{1-F:.6f}{m.group(3)}{F:.6f}{m.group(5)}", s)
        open(pp, "w").write(s)
        code = (f"import sys,numpy as np,pandas as pd\nsys.path.insert(0,{w!r})\n"
                f"from predict import Predictor\ndf=pd.read_pickle({proxy!r})\n"
                f"p=Predictor(model_dir={os.path.join(w,'model')!r})\n"
                f"o=p.predict(df) if hasattr(p,'predict') else p.predict_proba(df)\n"
                f"np.save({out!r},np.asarray(o,dtype=np.float64))\n")
        r = subprocess.run([sys.executable, "-c", code], cwd=w,
                           env=dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="4"),
                           capture_output=True, text=True)
        if r.returncode: sys.exit(f"[중단] t={t}\n{r.stderr[-2500:]}")
    P[t] = np.load(out)
    print(f"  t={t:+.4f}  R {max(R,0):.6f} F {F:.6f}  mean {P[t].mean():.9f}")

Q = {t: v - v.mean() for t, v in P.items()}     # 평균 제거 = PROB_SHIFT 재보정
K = 1e5 / (0.486987 * (1 - 0.486987))
base = Q[-0.70]

print("\n=== p(t) 가 t 에 선형인가 (기준현 -0.70~-0.25) ===")
gref = (Q[-0.25] - base) / 0.45
print("%9s %12s %14s %10s" % ("t", "rms(g)", "corr(gref)", "비"))
for t in TS:
    if t == -0.70: continue
    g = (Q[t] - base) / (t + 0.70)
    print("%+9.4f %12.5e %14.8f %10.6f"
          % (t, np.sqrt((g**2).mean()), np.corrcoef(g, gref)[0,1],
             np.sqrt((g**2).mean())/np.sqrt((gref**2).mean())))

g = gref
C = K * (g**2).mean()
KM = (0.7907 - 0.2025 * C) / 0.9          # 라운드 20 실측 +0.7907 로 고정
print(f"\nC  = K*mean(g^2) = {C:.4f}     (라벨 불필요)")
print(f"KM = {KM:.4f}                    (LB 1점에서 역산)")
print(f"검산  ΔScore(u=+0.45) = {-(2*0.45*KM + 0.2025*C):+.4f}   (실측 -0.7907)")
KM2024 = (g * (base + P[-0.70].mean() - y)).mean() * K
print(f"[대조] 2024 라벨의 KM = {KM2024:+.4f}   부호 {'일치' if KM2024*KM>0 else '반대'}, 배율 {abs(KM2024/KM):.1f}")

print(f"\n{'t':>8} {'R':>10} {'F':>10} {'예측ΔScore':>12} {'C 4배여도':>11}")
for t in (-0.80,-0.90,-1.00,-1.10,-1.20,-1.3333):
    u = t + 0.70
    print("%+8.4f %10.6f %10.6f %+12.4f %+11.4f"
          % (t, max(0.10+t*0.075,0), 0.40+t*0.10, -(2*u*KM + u*u*C), -(2*u*KM + u*u*C*4)))
