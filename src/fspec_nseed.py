"""F 전문가 — 멤버 수가 몇 점짜리인가 (§7.109)

왜 재는가 (사용자 지적)
    배포 경로는 전역 LightGBM 12멤버(변형 4 × 시드 3), MLP 8멤버(emb4 5 + noid 3)를
    로짓 평균한다. §7.103의 F 전문가 대조군도 3시드 평균이었다.
    그런데 **TFM 후보는 전부 1 fit**이었다 — 3 대 1로 붙인 셈이다.

    "불공정하다"는 참이다. 그러나 그것이 −5.83(TabDPT-Turbo)을 설명하는지는
    별개 질문이다. 멤버 수를 줄여가며 **핸디캡의 크기 자체**를 잰다.

읽는 법
    · 1시드 → 3시드 이득이 크면(≳3점) §7.103의 비교는 다시 해야 한다.
    · 작으면(≲1점) 멤버 수는 −5.83을 설명하지 못하고 결론은 유지된다.
    두 경우 모두 §7.103의 기각 근거를 정량적으로 갱신한다.

주의 — 이건 분산 축소의 **단독** 크기가 아니다
    MEMORY: [분산 축소 수단은 서로 겹친다]. 여기서 재는 값은 "이미 20멤버를
    앙상블한 z_dep에 α로 섞이는 전문가"의 멤버 수 효과다. 전문가를 혼자 쓸 때의
    이득과 다르며, 우리에게 필요한 건 정확히 이 섞인 상태의 값이다.

실행:  .venv-gpu/Scripts/python.exe -u src/fspec_nseed.py
"""
import gc
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

import lightgbm as lgb
import numpy as np

import foldcache
from config import (FSPEC_ALPHA, FSPEC_MIN_SEASON, FSPEC_PARAMS, MLP_ALPHA,
                    SEED_ENSEMBLE, TRAIN_CSV)
from data import load_csv
from validation import bss, shift_to_prior

LOG = Path(__file__).resolve().parents[1] / "experiments" / "fspec_nseed.log"
FOLDS = ["2023H2", 2024]
ALPHA_GRID = np.round(np.arange(0.0, 0.801, 0.05), 3)
PLACEBO_RATE = 0.118
PLACEBO_SEED = 20260820
GROUPS = ("F", "위약")
N_SEEDS = [1, 2, 3, 5]            # SEED_ENSEMBLE 앞에서부터 자른다


def emit(msg=""):
    print(msg)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def fit_spec(Xtr, ytr, tt, Xva, params, seeds):
    """§7.103 대조군과 동일. 시드 목록만 인자로 뺐다."""
    cut = np.quantile(tt, 0.75)
    itr, iva = tt <= cut, tt > cut
    if iva.sum() < 300 or len(np.unique(ytr[iva])) < 2:
        itr, iva = np.ones(len(ytr), bool), np.zeros(len(ytr), bool)
    zs = []
    for s in seeds:
        prm = dict(params, seed=s, bagging_seed=s, feature_fraction_seed=s,
                   data_random_seed=s)
        if iva.any():
            b0 = lgb.train(prm, lgb.Dataset(Xtr[itr], ytr[itr]), num_boost_round=3000,
                           valid_sets=[lgb.Dataset(Xtr[iva], ytr[iva])],
                           callbacks=[lgb.early_stopping(100, verbose=False)])
            nr = max(50, b0.best_iteration)
        else:
            nr = 200
        zs.append(lgb.train(prm, lgb.Dataset(Xtr, ytr),
                            num_boost_round=nr).predict(Xva, raw_score=True))
    return np.mean(zs, axis=0), zs


def main():
    t0 = time.time()
    emit("=" * 78)
    emit(f"§7.109 F 전문가 멤버 수 — 시드 {list(SEED_ENSEMBLE)}")

    df = load_csv(TRAIN_CSV)
    cache = foldcache.load(FOLDS)
    prng = np.random.default_rng(PLACEBO_SEED)
    placebo_tag = prng.random(len(df)) < PLACEBO_RATE

    D = {}
    for f in FOLDS:
        z_lgb, z_mlp, y, _ = cache[f]
        z_dep = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
        mtr, mva = foldcache.SPLITS[f](df.season.to_numpy(), df.game_month.to_numpy())
        tr, va = df[mtr], df[mva]
        Xtr, ytr, _, Xva, yva, _, _ = foldcache.fold_data(df, f)
        tt = (tr["season"].to_numpy(dtype=np.int64) * 12
              + tr["game_month"].to_numpy(dtype=np.int64))
        recent = tr["season"].to_numpy() >= FSPEC_MIN_SEASON
        raw = {
            "F": ((tr["game_type"].to_numpy().astype(str) == "F"),
                  (va["game_type"].to_numpy().astype(str) == "F")),
            "위약": (placebo_tag[mtr], placebo_tag[mva]),
        }
        g = {}
        for k, (m, v) in raw.items():
            m = m & recent
            g[k] = dict(Xtr=Xtr[m].to_numpy(np.float32), ytr=ytr[m], tt=tt[m],
                        Xva=Xva[v].to_numpy(np.float32), vmask=v)
        D[f] = dict(z_dep=z_dep, y=yva, g=g)
        del Xtr, Xva, ytr, yva, tr, va
        gc.collect()
    del df, cache, placebo_tag
    gc.collect()

    def score(f, z_spec, vmask, a):
        d = D[f]
        z = d["z_dep"].copy()
        z[vmask] = (1 - a) * z[vmask] + a * z_spec
        return bss(d["y"], shift_to_prior(1 / (1 + np.exp(-z)), d["y"].mean()))

    for f in FOLDS:
        emit("")
        emit(f"=== fold {f} ===")
        for group in GROUPS:
            d = D[f]["g"][group]
            # 5시드를 한 번만 학습하고, 앞에서부터 잘라 평균한다.
            _, zs = fit_spec(d["Xtr"], d["ytr"], d["tt"], d["Xva"],
                             dict(FSPEC_PARAMS), list(SEED_ENSEMBLE))
            base = score(f, zs[0], d["vmask"], 0.0)
            emit(f"  --- {group} (기준 {base:.1f}) ---")
            prev = None
            for n in N_SEEDS:
                z = np.mean(zs[:n], axis=0)
                s_dep = score(f, z, d["vmask"], FSPEC_ALPHA)
                sc = [score(f, z, d["vmask"], a) for a in ALPHA_GRID]
                k = int(np.argmax(sc))
                inc = "" if prev is None else f"   (직전 대비 {s_dep-prev:+.2f})"
                emit(f"    {n}멤버   배포 a={FSPEC_ALPHA:.2f} -> {s_dep:.1f} "
                     f"(D {s_dep-base:+.2f}){inc}      "
                     f"최적 a={ALPHA_GRID[k]:.2f} -> {sc[k]:.1f} (D {sc[k]-base:+.2f})")
                prev = s_dep

    emit("")
    emit(f"완료 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
