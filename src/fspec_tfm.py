"""F 전문가로서의 표 기반모델(TabPFN-Mix) — 위약 대조 포함 (§7.103)

배경 — §7.103 사전 점검에서 확정된 것
    · Mitra는 이 GPU(8.5GB)에서 탈락이다. ctx 3,000을 줘도 CUDA OOM으로
      max_samples_support가 8192→4096→2048→**1024**로 조용히 강등되고,
      그 상태로 fit에 5,427초(90분)가 걸린다. 25,686행의 4%만 보는 셈이다.
    · TabPFN-Mix는 ctx 6,000에서 fit 34.6초, 추론 0.3ms/행(예열 후)이다.
      30,010행 추론이 9초다. 속도는 비쟁점.
    · F 학습행은 전부 2023 한 시즌이다(25,686행). 최근성 가중으로 짜낼
      기울기가 없으므로 컨텍스트 선택은 **무작위**로 고정한다.

설계 — §7.102의 결론을 그대로 장비로 쓴다
    · 선택(컨텍스트 크기·α)은 fold "2023H2"에서만. 2024는 마지막 1회.
    · **위약 팔 필수.** 행 태그만 무작위 11.8%로 바꾼 팔을 완전히 같은 절차로
      돌린다. §7.102에서 이 대조가 튜너의 자유도와 F 고유성을 동시에 갈라줬다.
    · 대조군은 현행 LightGBM F 전문가(FSPEC_PARAMS·FSPEC_ALPHA)다.
      같은 자(fold 전체 BSS)로 잰다.

RAM 주의
    가용 RAM이 2GB대이고 AutoGluon이 +2GB를 얹는다. fspec_tune.py처럼 Xtr
    전체(1.2M x 79 = 380MB/fold)를 들고 있으면 스와핑에 들어가고, 그러면
    측정 대상인 시간이 오염된다. 태그된 행만 남기고 나머지는 버린다.

실행:  .venv-gpu/Scripts/python.exe -u src/fspec_tfm.py
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
import psutil
import truststore

truststore.inject_into_ssl()

import foldcache
from config import (FSPEC_ALPHA, FSPEC_MIN_SEASON, FSPEC_PARAMS, FSPEC_SEEDS,
                    MLP_ALPHA, SEED, TRAIN_CSV)
from data import load_csv
from validation import bss, shift_to_prior

LOG = Path(__file__).resolve().parents[1] / "experiments" / "fspec_tfm.log"
TMP = "C:/Users/강원우/AppData/Local/Temp"

FOLDS = ["2023H2", 2024]
INNER = "2023H2"
ALPHA_GRID = np.round(np.arange(0.0, 0.801, 0.05), 3)
PLACEBO_RATE = 0.118
PLACEBO_SEED = 20260820           # fspec_tune.py와 동일 — 같은 행이 태그된다
CONTEXTS = [3000, 6000, 12000, 25686]
GROUPS = ("F", "위약")
EPS = 1e-6


def emit(msg=""):
    print(msg)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def ram():
    return psutil.virtual_memory().available / 1e9


def fit_lgb(Xtr, ytr, tt, Xva, params):
    """현행 F 전문가 — fspec_tune.fit_spec과 동일 절차 (대조군)."""
    cut = np.quantile(tt, 0.75)
    itr, iva = tt <= cut, tt > cut
    if iva.sum() < 300 or len(np.unique(ytr[iva])) < 2:
        itr, iva = np.ones(len(ytr), bool), np.zeros(len(ytr), bool)
    zs = []
    for s in FSPEC_SEEDS:
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
    return np.mean(zs, axis=0)


def fit_tfm(Xtr_df, ytr, Xq_df, ctx, rng, path):
    """TabPFN-Mix. 컨텍스트는 무작위 ctx행 (시즌이 하나뿐이라 최근성 여지 없음)."""
    from autogluon.tabular import TabularPredictor

    if ctx < len(Xtr_df):
        idx = np.sort(rng.choice(len(Xtr_df), size=ctx, replace=False))
    else:
        idx = np.arange(len(Xtr_df))
    sub = Xtr_df.iloc[idx].reset_index(drop=True).copy()
    sub["__y"] = ytr[idx].astype(int)

    pred = TabularPredictor(label="__y", problem_type="binary",
                            eval_metric="log_loss", verbosity=0, path=path)
    pred.fit(sub, hyperparameters={"TABPFNMIX": {"ag.max_memory_usage_ratio": 12.0}},
             fit_weighted_ensemble=False, num_bag_folds=0, num_stack_levels=0,
             ag_args_fit={"num_gpus": 1})
    p = np.clip(pred.predict_proba(Xq_df)[1].to_numpy(), EPS, 1 - EPS)
    del pred
    gc.collect()
    return np.log(p / (1 - p))


def main():
    t0 = time.time()
    emit("=" * 78)
    emit(f"§7.103 TabPFN-Mix F 전문가 — 시작 (가용 RAM {ram():.2f}GB)")

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
        # 태그된 행만 남긴다. Xtr 전체는 fold당 380MB고 우리는 쓰지 않는다.
        g = {}
        for k, (m, v) in raw.items():
            m = m & recent
            g[k] = dict(Xtr=Xtr[m].reset_index(drop=True), ytr=ytr[m],
                        tt=tt[m], Xva=Xva[v].reset_index(drop=True), vmask=v)
        D[f] = dict(z_dep=z_dep, y=yva, g=g)
        emit(f"  fold {f}: 검증 {len(yva):,}행  "
             + "  ".join(f"{k} 학습 {len(d['ytr']):,}/검증 {int(d['vmask'].sum()):,}"
                         for k, d in g.items())
             + f"   ({time.time()-t0:.0f}s)")
        del Xtr, Xva, ytr, yva, tr, va
        gc.collect()

    del df, cache, placebo_tag
    gc.collect()
    emit(f"  프레임 해제 후 가용 RAM {ram():.2f}GB")

    def score(f, z_spec, vmask, a):
        d = D[f]
        z = d["z_dep"].copy()
        z[vmask] = (1 - a) * z[vmask] + a * z_spec
        return bss(d["y"], shift_to_prior(1 / (1 + np.exp(-z)), d["y"].mean()))

    def sweep(f, group, z):
        d = D[f]["g"][group]
        base = score(f, z, d["vmask"], 0.0)
        sc = [score(f, z, d["vmask"], a) for a in ALPHA_GRID]
        k = int(np.argmax(sc))
        return sc[k], float(ALPHA_GRID[k]), base

    # 대조군: 현행 LightGBM F 전문가
    emit("")
    emit("=== 대조군 — 현행 LightGBM F 전문가 ===")
    lgb_ref = {}
    for f in FOLDS:
        for group in GROUPS:
            d = D[f]["g"][group]
            z = fit_lgb(d["Xtr"].to_numpy(np.float32), d["ytr"], d["tt"],
                        d["Xva"].to_numpy(np.float32), dict(FSPEC_PARAMS))
            s, a, b = sweep(f, group, z)
            s_dep = score(f, z, d["vmask"], FSPEC_ALPHA)
            lgb_ref[(f, group)] = (s_dep, b)
            emit(f"  fold {f} {group:<4}  기준 {b:.1f}   "
                 f"배포 a={FSPEC_ALPHA:.2f} -> {s_dep:.1f} (D {s_dep-b:+.2f})   "
                 f"최적 a={a:.2f} -> {s:.1f} (D {s-b:+.2f})   ({time.time()-t0:.0f}s)")

    # TabPFN-Mix 컨텍스트 사다리 — 선택은 INNER에서만
    emit("")
    emit(f"=== TabPFN-Mix — 컨텍스트 사다리 (선택은 {INNER}에서만) ===")
    chosen = {}
    for group in GROUPS:
        emit(f"  --- {group} ---")
        best = None
        for ctx in CONTEXTS:
            rng = np.random.default_rng(SEED)
            d = D[INNER]["g"][group]
            try:
                z = fit_tfm(d["Xtr"], d["ytr"], d["Xva"], ctx, rng,
                            f"{TMP}/agtfm_{group}_{ctx}")
            except Exception as e:
                emit(f"    ctx {ctx:>6,}  FAIL {type(e).__name__}: {str(e)[:100]}")
                break
            s, a, b = sweep(INNER, group, z)
            emit(f"    ctx {ctx:>6,}  기준 {b:.1f} -> {s:.1f}  a={a:.2f}  "
                 f"D {s-b:+.2f}   (RAM {ram():.1f}GB, {time.time()-t0:.0f}s)")
            if best is None or s > best[0]:
                best = (s, ctx, a)
        if best is None:
            emit(f"    {group}: 전 구간 실패")
            continue
        chosen[group] = best
        emit(f"    선택: ctx {best[1]:,}  a={best[2]:.2f}  ({INNER} {best[0]:.1f})")

    # fold2024 적용 — 여기서 처음 본다
    emit("")
    emit("=== fold2024 적용 (처음 보는 곳) ===")
    for group in GROUPS:
        if group not in chosen:
            continue
        _, ctx, a = chosen[group]
        d = D[2024]["g"][group]
        rng = np.random.default_rng(SEED)
        try:
            z = fit_tfm(d["Xtr"], d["ytr"], d["Xva"], ctx, rng,
                        f"{TMP}/agtfm24_{group}_{ctx}")
        except Exception as e:
            emit(f"  {group}: FAIL {type(e).__name__}: {str(e)[:100]}")
            continue
        b = score(2024, z, d["vmask"], 0.0)
        s = score(2024, z, d["vmask"], a)
        s_lgb, _ = lgb_ref[(2024, group)]
        emit(f"  {group:<4} ctx {ctx:,} a={a:.2f}   {b:.1f} -> {s:.1f}  D {s-b:+.2f}"
             f"    (현행 LightGBM {s_lgb:.1f}, 차 {s-s_lgb:+.2f})")

    emit("")
    emit(f"완료 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
