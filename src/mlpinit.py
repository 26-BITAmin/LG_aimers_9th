"""§7.114 — MLP 에도 `init_score` (배포 층).

s29 는 `init_score` 를 **LGBM 에만** 걸었다. MLP 는 레벨을 여전히 스스로 배운다.
그런데 배포 블렌드가 α=0.75 라 **MLP 가 75%** 다 — LGBM(25%)만 고쳐서 배포 층 +2.8 이
나왔으니, 비중이 3배인 쪽도 볼 값어치가 있다.

구현: `mlp.train_mlp(offset=...)` 을 추가했다. 손실을 `sigmoid(z + offset)` 로 계산해
모델이 **레벨을 뺀 편차만** 배우게 하고, 반환 로짓에는 offset 을 더하지 않는다
(LGBM `raw_score` 와 같은 규약 → 블렌드가 편차 공간, 레벨은 prior shift 담당).
보조 헤드에는 offset 을 안 붙인다 — 다른 과제고 자기 레벨을 갖는다.

팔 (LGBM z 는 세 팔 모두 동일 — roundsfix 캐시의 init=1, k=320)
    H  하네스   MLP offset=None 을 **같은 새 코드로** 재학습 → 배포 캐시를 재현해야 한다
    A  배포     MLP z = 배포 캐시 (s29 상태)
    B  후보     MLP offset=init_tr (soften 후 레벨)

★ H 를 넣는 이유: `mlp.py` 학습 루프를 건드렸다. `offset=None` 이면 수식이 동일하지만
  **동일해야 한다는 것과 동일하다는 것은 다르다.** 이번 세션에서 하네스가 두 번
  결론의 부호를 바꿀 뻔했다 (§7.107 소프트라벨, §7.111 캐시 지문).

실행:  .venv/Scripts/python.exe -u src/mlpinit.py 2>&1 | tee experiments/mlpinit.log
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache as _fc

_fc.CENTER_ASOF = False
KEY = _fc._key()

import pandas as pd                                                 # noqa: E402
import mlp                                                          # noqa: E402
import modelabel                                                    # noqa: E402
import soften                                                       # noqa: E402
import jointbeta as J                                               # noqa: E402
from config import (ASOF_AXES, CONTAM_AXES, CTX_AXES, DROP_FEATURES,            # noqa: E402
                    MLP_ALPHA, MLP_AUX_LAMBDA, MLP_SEEDS, MLP_VARIANTS,
                    RECENCY_HALF_LIFE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight                           # noqa: E402
from features import build_features, fit_artifacts                  # noqa: E402
from initfix3 import level_init                                     # noqa: E402
from validation import bss, shift_to_prior                          # noqa: E402

FOLDS = [2023, 2024]
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED = 42


def train_mlps(df, fold, use_offset, t0):
    tag = "on" if use_offset else "off"
    pth = CACHE / f"mlpinit_fold{fold}_{tag}.npz"
    if pth.exists():
        print(f"    캐시 적중 {pth.name}")
        return np.load(pth)["z"]

    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xtr, Xva = X(tr), X(va)
    ss = tr["season"].to_numpy(dtype=np.int64)
    gg = tr["game_type"].astype("object").to_numpy().astype(str)
    aux, _ = modelabel.aux_labels(tr)
    h = soften.fit_h(tr)
    h_row = np.zeros(len(tr))
    for yr, v in h.items():
        h_row[(ss == yr) & (gg == "F")] = v
    dup = (h_row > 0) & (y_tr == 1)
    Xtr, y2, w2, aux2, _ = soften.soften(tr, Xtr, y_tr, w, aux)
    off = None
    if use_offset:
        off = level_init(np.concatenate([ss, ss[dup]]),
                         np.concatenate([gg, gg[dup]]), y2, w2)
        assert len(off) == len(Xtr), "offset 정렬 깨짐"

    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    zs = []
    for vname, (cfg, nseed) in MLP_VARIANTS.items():
        for sd in MLP_SEEDS[:nseed]:
            _, curve = mlp.train_mlp(A, C, y2, w2, B, Cv, sizes, seed=sd,
                                     verbose=False, y_aux=aux2, offset=off,
                                     **dict(cfg, aux=MLP_AUX_LAMBDA))
            zs.append(curve[-1])
    z = np.mean(zs, axis=0)
    np.savez_compressed(pth, z=z)
    print(f"    fold {fold} offset={tag} mlp {len(zs)}개 완료 ({time.time()-t0:.0f}s)")
    return z


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    codes, uniq = pd.factorize(df["pitcher_id"].to_numpy())
    n_id = len(uniq)
    mclean = ~((gt == "F") & (s < 2023))
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    print(f"train {df.shape}   α={MLP_ALPHA}  hl={RECENCY_HALF_LIFE}  "
          f"LGBM init=ON(고정)\n")

    R, KEEP, RAW = {}, {}, {}
    for fold in FOLDS:
        Xax, _, _, _ = J.axis_matrix(df, s, fold)
        vi = np.where(s == fold)[0]
        mm = mclean[vi]
        yv = y[vi]
        with np.load(CACHE / f"rounds_fold{fold}_r640.npz") as d:
            zl = d["k320"]                       # init_score LGBM (s29 배포 상태)
        with np.load(CACHE / f"fold{fold}_{KEY}.npz") as d:
            zm_dep = d["z_mlp"]
            assert np.allclose(d["y"], yv)
        arms = {"A 배포": zm_dep,
                "H 하네스(offset=None)": train_mlps(df, fold, False, t0),
                "B MLP init=ON": train_mlps(df, fold, True, t0)}
        for lab, zm in arms.items():
            RAW[(fold, lab)] = zm
            z = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm + Xax @ dep
            p = shift_to_prior(1 / (1 + np.exp(-z[mm])), yv[mm].mean())
            R[(fold, lab)] = bss(yv[mm], p)
            KEEP[(fold, lab)] = (yv[mm], z[mm], codes[vi][mm])
        print(f"  fold {fold}: " + "  ".join(f"{k} {R[(fold,k)]:.1f}" for k in arms))

    labs = ["A 배포", "H 하네스(offset=None)", "B MLP init=ON"]
    print(f"\n{'='*92}\n█ 배포 층 (blend α={MLP_ALPHA} + δ축 + prior shift, clean 행)\n{'='*92}")
    print(f"  {'팔':<24}" + "".join(f"{f'fold {f}':>11}" for f in FOLDS)
          + "".join(f"{f'Δ vs A ({f})':>15}" for f in FOLDS))
    for lab in labs:
        print(f"  {lab:<24}" + "".join(f"{R[(f,lab)]:>11.1f}" for f in FOLDS)
              + "".join(f"{R[(f,lab)]-R[(f,'A 배포')]:>+15.1f}" for f in FOLDS))

    print(f"\n  ★ 하네스 — H 가 A 를 재현해야 한다 (offset=None 이면 수식이 동일)")
    ok = True
    for f in FOLDS:
        d = R[(f, "H 하네스(offset=None)")] - R[(f, "A 배포")]
        c = float(np.corrcoef(RAW[(f, "H 하네스(offset=None)")], RAW[(f, "A 배포")])[0, 1])
        good = abs(d) < 3.0 and c > 0.999
        ok &= good
        print(f"     fold {f}: ΔBSS {d:+.2f}   로짓 상관 {c:.6f}   {'✅' if good else '❌'}")
    print(f"     → {'통과. B 를 읽는다.' if ok else '실패 — B 를 읽지 말 것.'}")

    print(f"\n  클러스터 부트스트랩 (투수 복원추출 200회), **H 대비** B")
    for f in FOLDS:
        yv, z1, cf = KEEP[(f, "B MLP init=ON")]
        z0 = KEEP[(f, "H 하네스(offset=None)")][1]
        order = np.argsort(cf, kind="stable")
        bnd = np.searchsorted(cf[order], np.arange(n_id + 1))
        seen = np.where(np.diff(bnd) > 0)[0]
        bs = []
        for _ in range(200):
            idx = np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                                  rng.choice(seen, size=len(seen), replace=True)])
            yy = yv[idx]
            bs.append(bss(yy, shift_to_prior(1 / (1 + np.exp(-z1[idx])), yy.mean()))
                      - bss(yy, shift_to_prior(1 / (1 + np.exp(-z0[idx])), yy.mean())))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        d = R[(f, "B MLP init=ON")] - R[(f, "H 하네스(offset=None)")]
        print(f"     fold {f}: {d:+6.1f}  [{lo:+6.1f}, {hi:+6.1f}]")

    print(f"\n{'='*92}\n■ 판정 (§7.111 개정 기준)\n{'='*92}")
    print("  어느 fold 도 0 을 **유의하게** 밑돌지 않을 것 + fold2024 가 양수.")
    print("  ⚠️ 재학습 축이라 시드 노이즈(std 4~9)가 낀다 — 부트스트랩은 표집 오차만 담는다.")
    print("  ⚠️ 기준은 **H**(같은 코드로 offset 만 끈 팔)다. A 는 낡은 코드로 만든 캐시라")
    print("     A 대비로 읽으면 코드 차이와 offset 효과가 섞인다.")


if __name__ == "__main__":
    main()
