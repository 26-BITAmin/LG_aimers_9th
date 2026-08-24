"""파트 B 참패(-247.4)가 스케일인가 신호 부재인가 — ftdiag.py 의 MLP 슬롯 판.

MLP 슬롯은 α=0.75 로 **전 행**에 섞인다. 로짓 폭이 어긋나면 이 규모가 쉽게 난다.
스케일 가설이면 작은 α 에서 양수가 나와야 하고, 신호 부재면 어떤 α 에서도 안 난다.
AUC 는 스케일 불변이라 셋째 증거가 된다.
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")
sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass
import warnings; warnings.filterwarnings("ignore")
import numpy as np

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
WORK = Path("C:/Users/강원우/AppData/Local/Temp/ftx_work")
LOG = Path(__file__).parent.parent / "experiments" / "ftdiagb.log"
FOLD = 2024

def emit(m=""):
    print(m)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(m + "\n")

def main():
    import deploy29 as D
    from config import CAT_ALPHA, FSPEC_ALPHA, MLP_ALPHA
    from ftx import prep_full, train_fspec
    from validation import bss, shift_to_prior, auc

    t0 = time.time()
    df, s, y = D.load_df()
    st = D.state(df, s, y, FOLD)
    with np.load(CACHE / f"fold{FOLD}_{D.KEY}.npz") as d0:
        z_mlp = d0["z_mlp"]
    z_lgb = np.load(CACHE / f"initdep_fold{FOLD}_ctr0.npz")["z"]
    mm, yv, dep = st["mm"], st["yv"], st["dep"]
    Xa = np.zeros((len(z_lgb), st["Xax"].shape[1])); Xa[mm] = st["Xax"]
    ax = Xa[mm] @ dep; ybar = yv.mean()
    def sc(zb):
        return bss(yv, shift_to_prior(1/(1+np.exp(-(zb[mm]+ax))), ybar))

    X, tr, va, X_ref, art = prep_full(df, FOLD)
    z_fs, _ = train_fspec(X, tr, X_ref, t0)
    is_f = va["game_type"].to_numpy().astype(str) == "F"
    z_cat = np.mean([np.load(f)["z"] for f in
                     sorted(CACHE.glob(f"catcpu_fold{FOLD}_*.npz"))], axis=0)
    z_ft = np.load(WORK / "z_B_42.npy")

    def layer(zn, a):
        z = (1-a)*z_lgb + a*zn
        z[is_f] = (1-FSPEC_ALPHA)*z[is_f] + FSPEC_ALPHA*z_fs[is_f]
        return sc((1-CAT_ALPHA)*z + CAT_ALPHA*z_cat)

    emit("="*70)
    emit(f"현행 s37 층 {layer(z_mlp, MLP_ALPHA):.1f}  (MLP 8멤버, α={MLP_ALPHA})")
    emit("")
    emit("① 로짓 스케일 (검증 253,507행)")
    for nm, z in (("z_lgb(12)", z_lgb), ("z_mlp(8)", z_mlp), ("z_ft(B,seed42)", z_ft)):
        v = z[mm]
        emit(f"   {nm:<16} std {v.std():7.3f}  평균 {v.mean():7.3f}  최대|z| {np.abs(v).max():7.2f}")
    emit("")
    emit("② α 스윕 (홀드아웃 커닝 — 스케일 가설의 검정)")
    emit(f"   {'α':>6} | {'MLP':>9} | {'FT':>9}")
    for a in [0.0, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75]:
        mk = "  <- 배포" if abs(a-MLP_ALPHA) < 1e-9 else ""
        emit(f"   {a:6.2f} | {layer(z_mlp,a):9.1f} | {layer(z_ft,a):9.1f}{mk}")
    emit("")
    emit("③ 전체 검증행 AUC — 스케일 불변")
    for nm, z in (("z_lgb", z_lgb), ("z_mlp", z_mlp), ("z_ft(B)", z_ft)):
        emit(f"   {nm:<10} {auc(yv, z[mm]):.4f}")
    emit("")
    emit(f"완료 ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
