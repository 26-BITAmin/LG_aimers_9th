"""FT-Transformer 참패(−51.8)가 **스케일**인가 **신호 부재**인가

MEMORY: [LightGBM rf 모드는 트리를 합한다] — 새 학습기는 std(z)부터 대조할 것.
        [너무 나쁜 결과도 의심하기] — 회의의 강도가 결과 방향에 따라 달라지면 편향.

두 가설이 정확히 다른 예측을 한다.
    (스케일)   z_ft 의 std 가 z_fspec 보다 크게 벗어난다. 그러면 α=0.458 은
               "같은 슬롯"이 아니다 — 부피가 다른 것을 같은 비율로 섞은 것이다.
               이 경우 **오라클 α**(홀드아웃 커닝)가 작은 값에서 양수를 낸다.
    (신호 부재) std 는 비슷한데 어떤 α 에서도 개선이 없다.

실행:  .venv-gpu/Scripts/python.exe -u src/ftdiag.py
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")
sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import warnings

warnings.filterwarnings("ignore")

import numpy as np

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
WORK = Path("C:/Users/강원우/AppData/Local/Temp/ftx_work")
LOG = Path(__file__).parent.parent / "experiments" / "ftdiag.log"
FOLD = 2024
SEEDS = [42, 7, 2024]


def emit(m=""):
    print(m)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(m + "\n")


def main():
    import deploy29 as D
    from config import CAT_ALPHA, FSPEC_ALPHA, MLP_ALPHA
    from ftx import prep_full, train_fspec
    from validation import bss, shift_to_prior

    t0 = time.time()
    df, s, y = D.load_df()
    st = D.state(df, s, y, FOLD)
    with np.load(CACHE / f"fold{FOLD}_{D.KEY}.npz") as d0:
        z_mlp = d0["z_mlp"]
    z_lgb = np.load(CACHE / f"initdep_fold{FOLD}_ctr0.npz")["z"]
    mm, yv, dep = st["mm"], st["yv"], st["dep"]
    Xa = np.zeros((len(z_lgb), st["Xax"].shape[1]))
    Xa[mm] = st["Xax"]
    ax = Xa[mm] @ dep
    ybar = yv.mean()

    def sc(zb):
        return bss(yv, shift_to_prior(1 / (1 + np.exp(-(zb[mm] + ax))), ybar))

    z_s29 = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
    X, tr, va, X_ref, art = prep_full(df, FOLD)
    z_fs, _ = train_fspec(X, tr, X_ref, t0)
    is_f = va["game_type"].to_numpy().astype(str) == "F"
    cat_files = sorted(CACHE.glob(f"catcpu_fold{FOLD}_*.npz"))
    z_cat = np.mean([np.load(f)["z"] for f in cat_files], axis=0)

    def layer(zf, a):
        z = z_s29.copy()
        z[is_f] = (1 - a) * z_s29[is_f] + a * zf[is_f]
        return sc((1 - CAT_ALPHA) * z + CAT_ALPHA * z_cat)

    base37 = layer(z_fs, FSPEC_ALPHA)
    emit("=" * 78)
    emit(f"현행 s37 층 {base37:.1f}   (fspec LightGBM, α={FSPEC_ALPHA})")
    emit("")

    # ── ① 스케일 대조 — F 행에서만 본다 (섞이는 곳이 거기다) ──
    emit("① 로짓 스케일 (F 검증행 30,010개)")
    f = is_f
    emit(f"   {'대상':<16} {'std':>8} {'평균':>8} {'|z|>6 비율':>11} {'최대|z|':>9}")

    def prof(name, z):
        v = z[f]
        emit(f"   {name:<16} {v.std():8.3f} {v.mean():8.3f}"
             f" {np.mean(np.abs(v) > 6):11.4f} {np.abs(v).max():9.2f}")

    prof("배포 블렌드", z_s29)
    prof("fspec(LGBM)", z_fs)
    zs = {}
    for sd in SEEDS:
        p = WORK / f"z_A_{sd}.npy"
        if p.exists():
            zs[sd] = np.load(p)
            prof(f"FT seed {sd}", zs[sd])

    # ── ② 오라클 α 스윕 — 커닝이지만 진단이다 ──
    emit("")
    emit("② α 스윕 (홀드아웃 커닝 — 채택 근거 아님, 스케일 가설의 검정일 뿐)")
    grid = [0.0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.458, 0.70]
    emit(f"   {'α':>6} | " + " | ".join(f"{k:>9}" for k in
                                        ["LGBM"] + [f"FT {sd}" for sd in zs]))
    for a in grid:
        row = [f"{layer(z_fs, a):9.1f}"]
        for sd in zs:
            row.append(f"{layer(zs[sd], a):9.1f}")
        mark = "  <- 배포" if abs(a - FSPEC_ALPHA) < 1e-9 else ""
        emit(f"   {a:6.3f} | " + " | ".join(row) + mark)

    # ── ③ 순위 정보는 있는가 (AUC는 스케일 불변) ──
    emit("")
    emit("③ F 행 AUC — 스케일과 무관한 순위 정보")
    from validation import auc
    yf = yv[is_f[mm]] if len(is_f) != len(yv) else yv[is_f]
    try:
        m2 = is_f & mm
        yy = yv[m2[mm]]
        emit(f"   fspec(LGBM)  {auc(yy, z_fs[m2]):.4f}")
        for sd in zs:
            emit(f"   FT seed {sd:<5} {auc(yy, zs[sd][m2]):.4f}")
        emit(f"   배포 블렌드   {auc(yy, z_s29[m2]):.4f}")
    except Exception as e:
        emit(f"   (AUC 계산 실패: {type(e).__name__} {e})")

    emit("")
    emit(f"완료 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
