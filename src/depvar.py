# -*- coding: utf-8 -*-
"""배포 모델(전체 train)의 emb4 시드 분산을 **라벨 없이** 더미 2025 분포에서."""
import sys, time; sys.path.insert(0,"src"); sys.path.insert(0,"submit")
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import joblib, numpy as np
from pathlib import Path
import build_submit as B, predict as P
from features import build_features
from mlp_infer import MLPPredictor

for MD, tag in ((Path("models_bak/model_m16"), "m16 (emb4 10시드)"),
                (Path("submit/model"),        "s37 배포 (emb4 5시드)")):
    bundle = joblib.load(MD / "artifacts.joblib")
    art, cols, enc = bundle["artifacts"], bundle["feature_cols"], bundle["mlp_enc"]
    dummy = B.make_dummy_test(art=art)
    X = build_features(dummy, art)
    for c in ("pitcher_id","batter_id"):
        if c in cols and c not in X.columns:
            X[c] = dummy[c].to_numpy(dtype=np.float32)
    X = X[cols]
    files = sorted(MD.glob("mlp_emb4_*.npz"))
    Z = []
    for f in files:
        with np.load(f) as npz:
            Z.append(MLPPredictor({k: npz[k] for k in npz.files}, enc).logit(X))
    Z = np.stack(Z)
    n = len(Z)
    # 각 시드를 자기 평균으로 중심화 (전역 레벨은 solve_delta 가 흡수 — §7.128)
    Zc = Z - Z.mean(axis=1, keepdims=True)
    s2 = Zc.var(axis=0, ddof=1)
    # 배포 p 로 σ' 를 만든다
    pr = P.Predictor(str(MD))
    p = pr.predict(dummy)
    dpdz2 = (p*(1-p))**2
    W_MLP, W_E = 0.675, 0.875
    UNC = 0.4824*(1-0.4824); K = 1e5/UNC
    base = W_MLP**2 * W_E**2 * float(np.mean(dpdz2*s2))
    b = base*K
    print(f"\n{tag}  멤버 {n}개  더미 {len(dummy):,}행")
    print(f"  sd(z) 시드간(중심화 후) = {float(np.sqrt(s2.mean())):.4f}")
    print(f"  ★ b(배포측, 라벨없음) = {b:7.2f}   현재 손실 = b/{n} = {b/n:6.2f} BSS")
    print(f"     fold2023 실측 b = 33.43   비 {b/33.43:.2f}")
