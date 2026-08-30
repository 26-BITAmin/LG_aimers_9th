"""§7.210b — emb_dim **8→12** 를 챔피언 등가 fold 층에서 잰다 (사다리 다음 칸).

§7.202 가 `sy03_emb12.zip` 을 빌드했지만 *"판정이 아니라 측정"* 이라 적었다 — 챔피언 zip 은
홀드아웃이 없어서다 (§7.207). §7.210 이 그 하네스를 만들었고, 거기서 4→8 이
**f2023 −2.17 / f2024 +6.69 → LB +7.44 (전이 1.11)** 로 맞았다. 같은 자로 다음 칸을 잰다.

D12 팔만 추가한다 — D4·D8·noid 는 §7.210 캐시를 그대로 쓴다 (같은 시드·같은 데이터).

실행: .venv/Scripts/python.exe -u src/foldchamp12.py 2>&1 | tee experiments/foldchamp12.log
"""
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import deploy29 as D
import foldcache
import hillclimb as H
import mlp
import modelabel
import soften
from config import MLP_ALPHA, MLP_AUX_LAMBDA, MLP_SEEDS, SOFT_LABEL_F, TRAIN_CSV
from data import load_csv
from foldchamp import variants
from validation import bss, shift_to_prior

CACHE = D.CACHE
DIM = 12
ARM = "D12"


def train_arm():
    df = load_csv(TRAIN_CSV)
    t0 = time.time()
    for fold in (2023, 2024):
        todo = [s for s in MLP_SEEDS
                if not (CACHE / f"champ_{ARM}_f{fold}_emb4_{s}.npz").exists()]
        if not todo:
            print(f"fold {fold} 캐시 적중")
            continue
        print(f"\n=== fold {fold} · dim {DIM} · 시드 {todo} ===")
        Xtr, ytr, w, Xva, yva, art, tr = foldcache.fold_data(df, fold)
        aux, _ = modelabel.aux_labels(tr)
        if SOFT_LABEL_F:
            Xtr, ytr, w, aux, _ = soften.soften(tr, Xtr, ytr, w, aux)
        cfg, _ = variants(DIM)["emb4"]
        A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
        for s in todo:
            _, curve = mlp.train_mlp(A, C, ytr, w, B, Cv, sizes, seed=s,
                                     verbose=False, y_aux=aux,
                                     **dict(cfg, aux=MLP_AUX_LAMBDA))
            np.savez_compressed(CACHE / f"champ_{ARM}_f{fold}_emb4_{s}.npz", z=curve[-1])
            print(f"  {ARM} emb4_{s} 단독 {mlp.score(curve[-1], yva):.1f}"
                  f"  ({time.time()-t0:.0f}s)")
            gc.collect()
        del Xtr, Xva, A, C, B, Cv
        gc.collect()


def report():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y_all = df["control_success"].to_numpy(dtype=np.float64)
    print("\n" + "=" * 84)
    print("emb_dim 사다리 — 챔피언 등가 fold 층 (블렌드 = LGBM12 + MLP8 + Cat12 + δ)")
    print("=" * 84)
    for fold in (2023, 2024):
        st = D.state(df, s, y_all, fold, check=False)
        mm, ax, yv = st["mm"], st["Xax"] @ st["dep"], st["yv"]
        r = float(yv.mean())
        zl = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"][mm]
        cat = np.mean([np.load(CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz")["z"]
                       for v in H.CAT_VARIANTS for sd in H.CAT_SEEDS], axis=0)[mm]
        noid = np.load(CACHE / f"champ_noid_f{fold}.npz")["z"]
        print(f"\n  fold {fold}   행 {len(yv):,}")
        base = None
        for a in ("D4", "D8", "D12"):
            e = [np.load(CACHE / f"champ_{a}_f{fold}_emb4_{sd}.npz")["z"]
                 for sd in MLP_SEEDS]
            zm = (0.875 * np.mean(e, axis=0) + 0.125 * np.mean(noid, axis=0))[mm]
            z = ((1 - H.W_CAT) * ((1 - MLP_ALPHA) * zl + MLP_ALPHA * zm)
                 + H.W_CAT * cat + ax)
            v = bss(yv, shift_to_prior(1 / (1 + np.exp(-z)), r))
            sd_ = np.std([mlp.score(x, np.load(CACHE / f"champ_y_f{fold}.npz")["y"])
                          for x in e], ddof=1)
            if base is None:
                base = v
            print(f"    {a:<4} 블렌드 {v:>9.2f}   Δ vs D4 {v-base:>+8.2f}"
                  f"   (멤버 단독 sd {sd_:.1f})")


if __name__ == "__main__":
    train_arm()
    report()
