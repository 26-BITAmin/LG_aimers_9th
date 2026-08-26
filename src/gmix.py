"""§7.164 후속 — 사용자 질문: *"+29.0/+19.4 이거는 뭐지"*

그 칸은 **aux OFF CTRL 대비**다. 그런데 배포되는 팔은 aux ON CTRL 이다.
그리고 내가 안 잰 칸이 하나 있다: `aux` 는 **멤버별 학습 선택**이라
"GRANDE 멤버만 aux 없이, noid 는 그대로 aux ON" 이 실제로 가능한 배포다.
§7.164 의 auxOFF 팔은 emb4 와 noid 를 **둘 다** OFF 로 썼다.

학습은 없다 — 캐시된 로짓 재조합뿐이다.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass
import numpy as np
import foldcache as _fc
_fc.CENTER_ASOF = False
_fc._key()
import pandas as pd
import jointbeta as J
from config import (ASOF_AXES, CAT_ALPHA, CAT_SEEDS, CAT_VARIANTS, CTX_AXES,
                    MLP_ALPHA, MLP_SEEDS, TARGET_COL, TRAIN_CSV)
from data import load_csv
from predict import PROB_CENTER, PROB_SCALE
from validation import bss, shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
W_E, W_N = 0.875, 0.125
S5 = tuple(MLP_SEEDS[:5])


def emb(tag, fold):
    return np.mean([np.load(CACHE / f"tabm_f{fold}_{tag}_{s}.npz")["z"] for s in S5], 0)


def noid(fold, aux):
    return np.load(CACHE / f"tabm_f{fold}_noid{'aux' if aux else ''}.npz")["z"]


def main():
    rng = np.random.default_rng(42)
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy(); gt = df["game_type"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    codes, uniq = pd.factorize(df["pitcher_id"].to_numpy()); n_id = len(uniq)
    mclean = ~((gt == "F") & (s < 2023))
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])

    # (라벨, emb4 캐시 태그, noid 가 aux 인가)
    ARMS = [
        ("CTRL   emb4 auxOFF + noid OFF", "ctrl", False),
        ("CTRL   emb4 auxON  + noid ON   ← 배포", "ctrlaux", True),
        ("G32e9  GRANDE auxOFF + noid OFF", "g32e9", False),
        ("G32e9  GRANDE auxOFF + noid ON  ← 미측정 하이브리드", "g32e9", True),
        ("G32e9  GRANDE auxON  + noid ON", "g32e9a", True),
        ("N128e3 NODE auxOFF + noid ON", "n128e3", True),
    ]
    R, K = {}, {}
    for fold in (2023, 2024):
        Xax, _, _, _ = J.axis_matrix(df, s, fold)
        vi = np.where(s == fold)[0]; mm = mclean[vi]
        ax = (Xax @ dep)[mm]; yc = y[vi][mm]; ybar = yc.mean()
        z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
        z_cat = np.mean([np.load(CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz")["z"]
                         for v in CAT_VARIANTS for sd in CAT_SEEDS], 0)
        for lab, tag, na in ARMS:
            zm = W_E * emb(tag, fold) + W_N * noid(fold, na)
            z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * zm
            z = (1 - CAT_ALPHA) * z + CAT_ALPHA * z_cat
            p = shift_to_prior(1 / (1 + np.exp(-(z[mm] + ax))), ybar)
            p = shift_to_prior(PROB_CENTER + PROB_SCALE * (p - PROB_CENTER), ybar)
            R[(fold, lab)] = bss(yc, p)
            K[(fold, lab)] = (yc, z[mm] + ax, codes[vi][mm])

    BASE = "CTRL   emb4 auxON  + noid ON   ← 배포"
    bar = "=" * 78
    print(f"\n{bar}\n배포층 BSS — 슬롯 = 0.875·emb4 + 0.125·noid\n{bar}")
    print(f"  {'팔':<42}{'f2023':>9}{'f2024':>9}")
    for lab, _, _ in ARMS:
        print(f"  {lab:<42}{R[(2023, lab)]:>9.1f}{R[(2024, lab)]:>9.1f}")

    print(f"\n{bar}\n**실제 배포 팔** 대비 — 클러스터 부트스트랩 400회\n{bar}")
    for lab, _, _ in ARMS:
        if lab == BASE:
            continue
        for f in (2023, 2024):
            yv_, z1, cf = K[(f, lab)]; z0 = K[(f, BASE)][1]
            order = np.argsort(cf, kind="stable")
            bnd = np.searchsorted(cf[order], np.arange(n_id + 1))
            seen = np.where(np.diff(bnd) > 0)[0]
            bs = []
            for _ in range(400):
                idx = np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                                      rng.choice(seen, size=len(seen), replace=True)])
                yy = yv_[idx]
                bs.append(bss(yy, shift_to_prior(1 / (1 + np.exp(-z1[idx])), yy.mean()))
                          - bss(yy, shift_to_prior(1 / (1 + np.exp(-z0[idx])), yy.mean())))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"  {lab:<42} f{f}: D {R[(f, lab)] - R[(f, BASE)]:+7.1f}"
                  f"  [{lo:+7.1f}, {hi:+7.1f}]")


if __name__ == "__main__":
    main()
