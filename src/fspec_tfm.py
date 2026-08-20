"""F 전문가로서의 표 기반모델 — 위약 대조 포함 (§7.103)

배경 — 사전 점검(src/ag_probe.py · ag_speed.py · ag_diag.py)에서 확정된 것
    · Mitra는 이 GPU(8.5GB)에서 탈락이다. ctx 3,000을 줘도 CUDA OOM으로
      max_samples_support가 8192→4096→2048→1024로 조용히 강등되고,
      그 상태로 fit에 5,427초가 걸린다. 25,686행의 4%만 보는 셈이다.
    · TabPFN-Mix는 ctx 6,000에서 fit 34.6초, 추론 0.3ms/행(예열 후)이다.
    · F 학습행은 전부 2023 한 시즌(25,686행)이라 컨텍스트 선택에서 최근성으로
      짜낼 기울기가 없다. 무작위로 고정한다.

설계 — §7.102의 결론을 그대로 장비로 쓴다
    · 선택(컨텍스트 크기·α)은 fold "2023H2"에서만. 2024는 마지막 1회.
    · 위약 팔 필수. 행 태그만 무작위 11.8%로 바꾼 팔을 완전히 같은 절차로 돌린다.
    · 채택 기준은 "F 이득 > 0"이 아니라 **"F 이득 > 같은 실행의 위약 이득"**이다.
      §7.103 1차에서 TabPFN-Mix의 위약이 +2.08로 F(+0.68)를 넘었다 — 노이즈
      바닥은 학습기마다 다르고 모델 계열을 넘어 재사용할 수 없다.

⚠️ fit은 반드시 별도 프로세스에서 한다
    1차 다모델 실행에서 가용 RAM이 6.10GB→0.6GB로 단조 감소하며 뒤쪽 fit이
    줄줄이 터졌다. AutoGluon은 predictor를 상주시키고 torch CUDA 캐싱 할당자는
    파이썬 객체가 죽어도 메모리를 OS에 안 돌려준다. `del`+`gc.collect()`로
    부족했다.

    이 누수는 단순 노이즈가 아니라 **편향**이다 — 사다리를 작은 ctx부터 도니
    큰 ctx일수록 남은 RAM이 적고, 따라서 큰 컨텍스트가 체계적으로 불리하게
    측정된다. "컨텍스트를 키워도 안 좋아진다"가 모델 성질이 아니라 실행 순서의
    인공물이 된다. 프로세스를 격리해 OS가 회수하게 한다(종료 직후 실측 7.99GB
    회수 확인).

실행:
    .venv-gpu/Scripts/python.exe -u src/fspec_tfm.py TABDPT-TURBO NORI TABICL
"""
import gc
import json
import os
import subprocess
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

TMP = Path("C:/Users/강원우/AppData/Local/Temp")
CACHE = TMP / "fspec_tfm_cache"
EXP = Path(__file__).resolve().parents[1] / "experiments"

FOLDS = ["2023H2", 2024]
INNER = "2023H2"
ALPHA_GRID = np.round(np.arange(0.0, 0.801, 0.05), 3)
PLACEBO_RATE = 0.118
PLACEBO_SEED = 20260820           # fspec_tune.py와 동일 — 같은 행이 태그된다
CONTEXTS = [3000, 6000, 12000, 25686]
GROUPS = ("F", "위약")
EPS = 1e-6
FIT_TIME_LIMIT = 900              # 초. Mitra가 5,427초를 먹은 사고의 재발 방지장치
SEED = 42


# ─────────────────────────────────────────────────────────────
# 워커 — 프로세스 하나가 fit 하나만 하고 죽는다
# ─────────────────────────────────────────────────────────────
def worker(npz_path, model_key, ctx, out_path):
    import pandas as pd
    import truststore

    truststore.inject_into_ssl()
    from autogluon.tabular import TabularPredictor

    d = np.load(npz_path, allow_pickle=True)
    cols = [str(c) for c in d["cols"]]
    Xtr = pd.DataFrame(d["Xtr"], columns=cols)
    ytr = d["ytr"]
    Xq = pd.DataFrame(d["Xva"], columns=cols)

    rng = np.random.default_rng(SEED)
    if ctx < len(Xtr):
        idx = np.sort(rng.choice(len(Xtr), size=ctx, replace=False))
    else:
        idx = np.arange(len(Xtr))
    sub = Xtr.iloc[idx].reset_index(drop=True).copy()
    sub["__y"] = ytr[idx].astype(int)

    path = str(TMP / f"agtfm_{model_key}_{ctx}_{os.getpid()}")
    pred = TabularPredictor(label="__y", problem_type="binary",
                            eval_metric="log_loss", verbosity=1, path=path)
    pred.fit(sub, hyperparameters={model_key: {"ag.max_memory_usage_ratio": 12.0}},
             fit_weighted_ensemble=False, num_bag_folds=0, num_stack_levels=0,
             ag_args_fit={"num_gpus": 1}, time_limit=FIT_TIME_LIMIT)
    p = np.clip(pred.predict_proba(Xq)[1].to_numpy(), EPS, 1 - EPS)
    np.save(out_path, np.log(p / (1 - p)))


# ─────────────────────────────────────────────────────────────
# 부모
# ─────────────────────────────────────────────────────────────
def main(models):
    import lightgbm as lgb
    import psutil

    import foldcache
    from config import (FSPEC_ALPHA, FSPEC_MIN_SEASON, FSPEC_PARAMS,
                        FSPEC_SEEDS, MLP_ALPHA, TRAIN_CSV)
    from data import load_csv
    from validation import bss, shift_to_prior

    tag = "_".join(m.lower().replace("-", "") for m in models)
    log = EXP / f"fspec_tfm_{tag}.log"

    def emit(msg=""):
        print(msg)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    def ram():
        return psutil.virtual_memory().available / 1e9

    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    emit("=" * 78)
    emit(f"§7.103 표 기반모델 F 전문가 — {', '.join(models)}  (가용 RAM {ram():.2f}GB)")

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
            gx = dict(Xtr=Xtr[m].to_numpy(np.float32), ytr=ytr[m], tt=tt[m],
                      Xva=Xva[v].to_numpy(np.float32), vmask=v)
            # 워커가 읽을 npz. 부모는 이후 이 배열들을 다시 안 쓴다.
            npz = CACHE / f"{f}_{'F' if k == 'F' else 'PLA'}.npz"
            np.savez(npz, Xtr=gx["Xtr"], ytr=gx["ytr"], Xva=gx["Xva"],
                     cols=np.array(list(Xtr.columns), dtype=object))
            gx["npz"] = str(npz)
            g[k] = gx
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

    def run_isolated(f, group, model_key, ctx):
        """fit 하나를 별도 프로세스에서. 실패하면 (None, 사유)를 돌려준다."""
        d = D[f]["g"][group]
        out = CACHE / f"z_{model_key}_{f}_{group}_{ctx}.npy"
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--fit", d["npz"], model_key, str(ctx), str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=FIT_TIME_LIMIT + 600)
        if r.returncode != 0 or not out.exists():
            blob = (r.stderr or "") + (r.stdout or "")
            hint = ""
            for pat in ("out of memory", "Not enough memory", "OutOfMemory",
                        "CUDA", "No module named", "ImportError", "TimeLimit"):
                if pat.lower() in blob.lower():
                    hint = pat
                    break
            tail = blob.strip().splitlines()[-1][:160] if blob.strip() else "(출력 없음)"
            return None, f"{hint or 'FAIL'} | {tail}"
        z = np.load(out)
        out.unlink(missing_ok=True)
        return z, None

    # 대조군: 현행 LightGBM F 전문가
    def fit_lgb(Xtr, ytr, tt, Xva, params):
        cut = np.quantile(tt, 0.75)
        itr, iva = tt <= cut, tt > cut
        if iva.sum() < 300 or len(np.unique(ytr[iva])) < 2:
            itr, iva = np.ones(len(ytr), bool), np.zeros(len(ytr), bool)
        zs = []
        for s in FSPEC_SEEDS:
            prm = dict(params, seed=s, bagging_seed=s, feature_fraction_seed=s,
                       data_random_seed=s)
            if iva.any():
                b0 = lgb.train(prm, lgb.Dataset(Xtr[itr], ytr[itr]),
                               num_boost_round=3000,
                               valid_sets=[lgb.Dataset(Xtr[iva], ytr[iva])],
                               callbacks=[lgb.early_stopping(100, verbose=False)])
                nr = max(50, b0.best_iteration)
            else:
                nr = 200
            zs.append(lgb.train(prm, lgb.Dataset(Xtr, ytr),
                                num_boost_round=nr).predict(Xva, raw_score=True))
        return np.mean(zs, axis=0)

    emit("")
    emit("=== 대조군 — 현행 LightGBM F 전문가 ===")
    d_lgb = {}
    for f in FOLDS:
        for group in GROUPS:
            d = D[f]["g"][group]
            z = fit_lgb(d["Xtr"], d["ytr"], d["tt"], d["Xva"], dict(FSPEC_PARAMS))
            s, a, b = sweep(f, group, z)
            s_dep = score(f, z, d["vmask"], FSPEC_ALPHA)
            if f == 2024:
                d_lgb[group] = s_dep - b
            emit(f"  fold {f} {group:<4}  기준 {b:.1f}   "
                 f"배포 a={FSPEC_ALPHA:.2f} -> {s_dep:.1f} (D {s_dep-b:+.2f})   "
                 f"최적 a={a:.2f} -> {s:.1f} (D {s-b:+.2f})   ({time.time()-t0:.0f}s)")

    verdict = {}
    for model_key in models:
        emit("")
        emit(f"=== {model_key} — 컨텍스트 사다리 (선택은 {INNER}에서만) ===")
        chosen = {}
        for group in GROUPS:
            emit(f"  --- {group} ---")
            best = None
            for ctx in CONTEXTS:
                z, err = run_isolated(INNER, group, model_key, ctx)
                if z is None:
                    emit(f"    ctx {ctx:>6,}  FAIL {err}")
                    continue          # 격리했으니 위쪽 rung을 포기할 이유가 없다
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

        emit(f"  === {model_key} fold2024 적용 (처음 보는 곳) ===")
        deltas = {}
        for group in GROUPS:
            if group not in chosen:
                continue
            _, ctx, a = chosen[group]
            z, err = run_isolated(2024, group, model_key, ctx)
            if z is None:
                emit(f"    {group}: FAIL {err}")
                continue
            d = D[2024]["g"][group]
            b = score(2024, z, d["vmask"], 0.0)
            s = score(2024, z, d["vmask"], a)
            deltas[group] = s - b
            emit(f"    {group:<4} ctx {ctx:,} a={a:.2f}   {b:.1f} -> {s:.1f}  "
                 f"D {s-b:+.2f}    (현행 LightGBM D {d_lgb[group]:+.2f}, "
                 f"차 {(s-b)-d_lgb[group]:+.2f})")
        verdict[model_key] = deltas

    # 판정 — 채택 기준은 "F > 0"이 아니라 "F > 같은 실행의 위약"이다
    emit("")
    emit("=== 판정 (fold2024) — 채택 기준: F 이득 > 위약 이득 & > 현행 ===")
    emit(f"  {'현행 LightGBM':<14}  F {d_lgb['F']:+6.2f}   위약 {d_lgb['위약']:+6.2f}   "
         f"→ {'통과' if d_lgb['F'] > d_lgb['위약'] else '탈락'}")
    for model_key, dd in verdict.items():
        if "F" not in dd or "위약" not in dd:
            emit(f"  {model_key:<14}  판정 불가 (측정 결손: "
                 f"{[g for g in GROUPS if g not in dd]})")
            continue
        ok = dd["F"] > dd["위약"] and dd["F"] > d_lgb["F"]
        emit(f"  {model_key:<14}  F {dd['F']:+6.2f}   위약 {dd['위약']:+6.2f}   "
             f"→ {'채택 후보' if ok else '탈락'}  (현행 대비 {dd['F']-d_lgb['F']:+.2f})")

    emit("")
    emit(f"완료 ({time.time()-t0:.0f}s, 가용 RAM {ram():.2f}GB)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fit":
        _, _, npz, model_key, ctx, out = sys.argv
        worker(npz, model_key, int(ctx), out)
    else:
        main(sys.argv[1:] or ["TABPFNMIX"])
