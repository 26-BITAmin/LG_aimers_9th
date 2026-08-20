import sys; sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, asofdelta, ctxfit, foldcache
import features as F
from config import *
from ctxsplit import resid_t
from data import load_csv, recency_weight
from validation import bss, shift_to_prior
from rolefeat import fit_role, apply_role

df = load_csv(TRAIN_CSV); s = df["season"].to_numpy()
cached = foldcache.load([2023, 2024], verbose=False)
rng = np.random.default_rng(SEED)
for fold in (2023, 2024):
    tr_m, va = s < fold, s == fold
    tr, d = df[tr_m], df[va].copy(); yv = cached[fold][2]
    z = (1-MLP_ALPHA)*cached[fold][0] + MLP_ALPHA*cached[fold][1]
    ref = int(s[tr_m].max())
    ctx = ctxfit.fit_ctx(tr, TARGET_COL, CTX_AXES, CTX_K,
                         weights=recency_weight(s[tr_m], CTX_HALF_LIFE, ref_season=ref))
    art = F.fit_artifacts(tr, ctx=ctx, contam_axes=CONTAM_AXES)
    art["asof"] = asofdelta.fit_asof_axes(tr, ASOF_AXES, ASOF_STRENGTH, ct=art.get("contam"))
    adj,_ = F.logit_adjust(d, art)
    p0 = shift_to_prior(1/(1+np.exp(-(z+adj))), yv.mean()); r = yv - p0
    role = fit_role(tr); feat, hit = apply_role(d, role)
    sr = feat["start_rate"]
    print(f"\nfold {fold}   잔차 평균 × start_rate 십분위 (선발일수록 오른쪽)")
    q = np.argsort(np.argsort(sr, kind="stable"), kind="stable")
    b = np.minimum((q*10)//len(sr), 9)
    print("   " + "".join(f"{np.mean(r[b==i])*1000:>8.2f}" for i in range(10)) + "   (×1000)")
    print("   n " + "".join(f"{np.sum(b==i)//1000:>7}k" for i in range(10)))
    # 상호작용 — 주효과를 통제군으로 같이 본다
    inn = np.clip(d["inning"].to_numpy(np.float64), 1, 12)
    outs = d["outs_before"].to_numpy(np.float64)
    bal = d["balls_before"].to_numpy(np.float64)
    run = d["num_runners_on"].to_numpy(np.float64)
    li  = np.nan_to_num(d["li"].to_numpy(np.float64), nan=1.0)
    Z = lambda v: (v - v.mean())/(v.std() or 1.0)
    print(f"   {'항':<22}{'잔차 t':>10}{'대조군 t':>11}")
    srs = sr[rng.permutation(len(sr))]
    for nm, v, vc in (("start_rate (주효과)", Z(sr), Z(srs)),
                      ("start × inning",  Z(Z(sr)*Z(inn)),  Z(Z(srs)*Z(inn))),
                      ("start × balls",   Z(Z(sr)*Z(bal)),  Z(Z(srs)*Z(bal))),
                      ("start × runners", Z(Z(sr)*Z(run)),  Z(Z(srs)*Z(run))),
                      ("start × li",      Z(Z(sr)*Z(li)),   Z(Z(srs)*Z(li))),
                      ("start × outs",    Z(Z(sr)*Z(outs)), Z(Z(srs)*Z(outs)))):
        _, t = resid_t(r, v); _, tc = resid_t(r, vc)
        print(f"   {nm:<22}{t:>10.2f}{tc:>11.2f}")
