"""s29 배포 층 홀드아웃 상태를 만드는 공용 헬퍼 (§7.118 계열이 공유).

pclip.py·tmslot.py 가 각자 복제하던 기계를 한 곳으로 모은다:
    z = (1−α)·z_lgb(initdep 팔B) + α·z_mlp(캐시) + Xax@β(δ축 4개)
    p = shift_to_prior(σ(z_clean), ȳ_clean)          clean = pre-2023 F 제외

첫 호출이 jointbeta.axis_matrix(ctx·asof·contam 적합, fold당 1~2분)를 돌리고
experiments/cache/dep29_fold{f}_{KEY}.npz 에 (z_blend, Xax, names) 를 얼린다.
이후 호출은 조회만 한다. **아티팩트 하네스 검증**: 기준 BSS가 tmslot.log 의
실측(f2023 807.2 / f2024 924.5)과 ±0.3 안에서 일치해야 한다 — 어긋나면
캐시 지문이나 config 가 바뀐 것이므로 즉시 죽는다.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import foldcache as _fc

# 프로브는 **배포 시점 베이스** 위에서 재야 한다. config 를 건드리면 (예: §7.128 의
# MLP 시드 확장) 지문이 바뀌어 캐시가 통째로 무효화되는데, 그 변경은 fold 블렌드와
# 무관한 배포 층 변경일 수 있다. `LGA_FOLD_KEY` 로 고정한다 — 틀린 캐시를 읽으면
# `EXPECT_BASE` assert 가 잡는다.
KEY = os.environ.get("LGA_FOLD_KEY") or _fc._key()

import jointbeta as J                                              # noqa: E402
from config import (ASOF_AXES, CTX_AXES, MLP_ALPHA, ROOT,          # noqa: E402
                    TARGET_COL, TRAIN_CSV)
from data import load_csv                                          # noqa: E402
from validation import bss, shift_to_prior                         # noqa: E402

FOLDS = [2023, 2024]
CACHE = ROOT / "experiments" / "cache"
EXPECT_BASE = {2023: 807.2, 2024: 924.5}       # tmslot.log 실측 — 하네스 검증용


def load_df():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    return df, s, y


def state(df, s, y, fold, check=True):
    """fold 의 배포 상태 dict 를 돌려준다 (clean 행 기준).

    키: vi(검증 행 인덱스), mm(clean 마스크), yv, z0(축 포함 로짓), p0, base(BSS),
        r(잔차), d(clean 검증 df), cl(투수 클러스터 코드), n_id,
        Xax(clean 행 × 4축 원단위), names, dep(배포 β 벡터)
    """
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    gt = df["game_type"].to_numpy()
    mclean_all = ~((gt == "F") & (s < 2023))
    codes, uniq = pd.factorize(df["pitcher_id"].to_numpy())

    f = CACHE / f"dep29_fold{fold}_{KEY}.npz"
    vi = np.where(s == fold)[0]
    if f.exists():
        with np.load(f, allow_pickle=True) as z:
            z_blend, Xax, names = z["z_blend"], z["Xax"], list(z["names"])
    else:
        Xax, names, _, _ = J.axis_matrix(df, s, fold)
        with np.load(CACHE / f"fold{fold}_{KEY}.npz") as d0:
            z_mlp, yc = d0["z_mlp"], d0["y"]
        assert np.allclose(yc, y[vi]), "캐시 y 불일치 — 지문이 틀렸다"
        z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
        z_blend = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
        np.savez_compressed(f, z_blend=z_blend, Xax=Xax, names=np.array(names))

    z = z_blend + Xax @ dep
    mm = mclean_all[vi]
    yv = y[vi][mm]
    z0 = z[mm]
    p0 = shift_to_prior(1 / (1 + np.exp(-z0)), yv.mean())
    base = bss(yv, p0)
    if check:
        assert abs(base - EXPECT_BASE[fold]) < 0.3, \
            f"기준 BSS {base:.1f} != 기대 {EXPECT_BASE[fold]} — 하네스 어긋남"
    return dict(vi=vi, mm=mm, yv=yv, z0=z0, p0=p0, base=base, r=yv - p0,
                d=df.iloc[vi[mm]], cl=codes[vi][mm], n_id=len(uniq),
                Xax=Xax[mm], names=names, dep=dep)


def rescore(st, dz):
    """clean 로짓에 dz 를 더했을 때의 ΔBSS."""
    p = shift_to_prior(1 / (1 + np.exp(-(st["z0"] + dz))), st["yv"].mean())
    return bss(st["yv"], p) - st["base"]


def cluster_t(st, dz):
    """잔차 개선의 투수 클러스터 t. d_i = (p0−y)² − (p_var−y)² 의 클러스터 평균 검정."""
    p = shift_to_prior(1 / (1 + np.exp(-(st["z0"] + dz))), st["yv"].mean())
    di = (st["p0"] - st["yv"]) ** 2 - (p - st["yv"]) ** 2
    g = np.bincount(st["cl"], weights=di, minlength=st["n_id"])
    n = np.bincount(st["cl"], minlength=st["n_id"])
    seen = n > 0
    gm = g[seen]
    mu = gm.sum() / n.sum()
    # 클러스터 합 기반 SE (CR0 동형)
    se = np.sqrt(np.sum((gm - mu * n[seen]) ** 2)) / n.sum()
    return mu / se if se > 0 else 0.0
