"""§7.208b — 챔피언 등가 **fold** 층에서 과거 후보 축을 다시 쓸어본다.

§7.207: 챔피언 zip 은 홀드아웃이 없어 못 잰다 (위약이 후보를 이기고, LB 로 검증된
배포 축이 "제거하라"고 나온다).
§7.208(`foldchamp.py`): 그래서 **레시피를 fold 로 재학습**했다 — emb_dim 8 · 콜드 패치.

이 스크립트는 그 층 위에서
    ① D4 / D8 / D8C 세 팔의 블렌드 BSS 를 비교하고 (챔피언 변경의 **첫 정직한 홀드아웃**)
    ② 최선 팔 위에서 gap 축 β 스윕 + 엔티티 순열 위약 + 투수 클러스터 부트스트랩

블렌드는 배포 등가 B1: (1−W_CAT)(1−α)·LGBM12 + (1−W_CAT)·α·(.875 emb4 + .125 noid)
+ W_CAT·CatBoost12, 그 위에 ctx·asof δ 축.

실행: .venv/Scripts/python.exe -u src/foldchampsweep.py 2>&1 | tee experiments/foldchampsweep.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

import asofdelta
import deploy29 as D
import hillclimb as H
from config import MLP_ALPHA, MLP_SEEDS, TARGET_COL, TRAIN_CSV
from data import load_csv
from gapstrike import EXTRA_SPEC
from mixresid import GAP_SPEC, build
from validation import bss, shift_to_prior

CACHE = D.CACHE
FOLDS = (2023, 2024)
ARMS = ("D4", "D8", "D8C")
PROBED = ["gap_strike", "gap_middle", "gap_bmiddle", "gap_ball", "gap_reverse"]
BETAS = [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.03]
B_BOOT = 600


def arm_z(fold, arm):
    e = [np.load(CACHE / f"champ_{arm}_f{fold}_emb4_{s}.npz")["z"] for s in MLP_SEEDS]
    noid = np.load(CACHE / f"champ_noid_f{fold}.npz")["z"]
    return 0.875 * np.mean(e, axis=0) + 0.125 * np.mean(noid, axis=0)


def main():
    for k, v in EXTRA_SPEC.items():
        asofdelta.ASOF_SPEC.setdefault(k, v)
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y_all = df[TARGET_COL].to_numpy(dtype=np.float64)

    for fold in FOLDS:
        st = D.state(df, s, y_all, fold, check=False)
        mm, ax, yv = st["mm"], st["Xax"] @ st["dep"], st["yv"]
        r = float(yv.mean())
        z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"][mm]
        cat = np.mean([np.load(CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz")["z"]
                       for v in H.CAT_VARIANTS for sd in H.CAT_SEEDS], axis=0)[mm]

        def blend(zm):
            return ((1 - H.W_CAT) * ((1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * zm)
                    + H.W_CAT * cat + ax)

        def sc(z):
            return bss(yv, shift_to_prior(1 / (1 + np.exp(-z)), r))

        print("=" * 92)
        print(f"fold {fold}   행 {len(yv):,}   ⚠️ train < {fold} — 정직한 홀드아웃")
        print("=" * 92)
        Z, S = {}, {}
        for a in ARMS:
            Z[a] = blend(arm_z(fold, a)[mm])
            S[a] = sc(Z[a])
            print(f"  {a:<5} 블렌드 BSS {S[a]:>9.2f}"
                  + (f"   Δ vs D4 {S[a]-S['D4']:+8.2f}" if a != "D4" else ""))

        # ── 후보 축 스윕 (최선 팔 위에서)
        best = max(ARMS, key=lambda a: S[a])
        z0, base = Z[best], S[best]
        d = df.iloc[st["vi"][mm]].copy().reset_index(drop=True)
        tr = s < fold
        art = asofdelta.fit_offsets(df[tr])
        league = float(df.loc[tr, TARGET_COL].mean())
        priors = {c: float(np.nanmean(df.loc[tr, c].to_numpy(dtype=np.float64)))
                  for _, c, _ in GAP_SPEC}
        feat = build(d, art, league, priors)
        rng = np.random.default_rng(0)
        dsh = d.copy()
        dsh["pitcher_id"] = rng.permutation(d["pitcher_id"].to_numpy())
        dsh["batter_id"] = rng.permutation(d["batter_id"].to_numpy())
        fsh = build(dsh, art, league, priors)

        codes, uniq = pd.factorize(d["pitcher_id"].to_numpy())
        order = np.argsort(codes, kind="stable")
        bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
        seen = np.where(np.diff(bnd) > 0)[0]
        br = np.random.default_rng(7)
        IDX = [np.concatenate([order[bnd[j]:bnd[j+1]] for j in
                               br.choice(seen, size=len(seen), replace=True)])
               for _ in range(B_BOOT)]

        print(f"\n  축 스윕 — 기준 팔 {best} (BSS {base:.2f})")
        print(f"  {'축':<14}" + "".join(f"{b:>+9.3f}" for b in BETAS)
              + f"{'β*':>8}{'Δ':>8}{'위약':>8}{'95% 하한':>10}")
        for c in PROBED:
            if c not in feat:
                continue
            x = feat[c].astype(np.float64)
            row = [sc(z0 + b * x) - base for b in BETAS]
            i = int(np.argmax(row))
            b, dv = BETAS[i], row[i]
            pl = sc(z0 + b * fsh[c].astype(np.float64)) - base
            zc = z0 + b * x
            bs = np.array([bss(yv[j], shift_to_prior(1/(1+np.exp(-zc[j])), r))
                           - bss(yv[j], shift_to_prior(1/(1+np.exp(-z0[j])), r))
                           for j in IDX])
            lo = float(np.percentile(bs, 2.5))
            print(f"  {c:<14}" + "".join(f"{v:>+9.2f}" for v in row)
                  + f"{b:>+8.3f}{dv:>+8.2f}{pl:>+8.2f}{lo:>+10.2f}")
        print()


if __name__ == "__main__":
    main()
