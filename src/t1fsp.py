"""§7.176b — 얕은 이력 증강을 **F 전용 전문가**(`user_fspec`)에 건다

§7.176 은 코어 LGBM 에 걸어서 전체 −1.71 을 얻었다. 그런데 F행 분해가 갈렸다:

    F행   얕음 +6.1  ·  깊음 −15.0  ·  무작위 −0.4      (①−② = +21.1)
    R행   얕음 −0.5  ·  깊음  −1.3  ·  무작위 −0.7

**효과가 F행에만 있는데 코어는 모든 행에 작용한다.** R행(88%)의 작은 손해가
F행(12%)의 큰 이득을 덮었다. → **F행에만 작용하는 부품**에 걸어야 한다.

T1 에 그게 있다 — `predict.py:168`

    z[is_f] = 0.60 * z[is_f] + 0.40 * zf[is_f]      zf = user_fspec 3개 평균

| 어디에 | F행 가중 | R행 부작용 |
| --- | ---: | --- |
| 코어 LGBM (§7.176) | 0.60×0.25 = 15% | **모든 행** |
| **`user_fspec`** | **40%** | **구조적으로 0** |

F행 가중이 2.7배이고 R행 손해가 0 이다.

`user_fspec` 은 우리 것이다 — 공유본 `train_hyejin_fspec_c.py` 첫 줄이
*"Train Hyejin/Wonwoo's original clean-F specialist"*. `season>=2023 & F` 만 학습,
`FSPEC_SEEDS` 3시드, 내부 시간분할(뒤 25%) early stopping. 우리 `train_fspec` 과 같다.

팔 (전부 D_aug 코어 = s41 등가 위에서, `user_fspec` 3개만 교체)
    base    재학습·증강 없음      ← 출하본이 아니라 **재학습본**이 기준이다
                                    (출하본은 2024 를 봤다 — §7.174 함정)
    ① 얕음  + F·n<500·2023·y=0    씨앗 4,395 → 합성 4,395
    ② 깊음  + F·n>=500·2023·y=0   같은 개수   ← "얕음" 격리
    ③ 무작위 + 전문가 학습풀(2023 F)에서 아무 행  ← "표적" 격리
              ⚠️ 여기서 무작위는 **전문가 자신의 풀** 이다. 전문가는 2023+ F 만
                 보므로 학습셋 전체에서 뽑으면 그 팔만 다른 분포를 받는다.

⚠️ fold2024 하나뿐 (2023 행을 쓴다). 대조군 차감이 유일한 판정자다.

실행:  .venv/Scripts/python.exe -u src/t1fsp.py 2>&1 | tee experiments/t1fsp.log
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
T1 = ROOT / "t1work"
ASCII = Path("D:/t1eval")
sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (FSPEC_MAX_ROUNDS, FSPEC_PARAMS, FSPEC_SEEDS, TARGET_COL,
                    TRAIN_CSV)
from data import load_csv
from smotenc import smote
from t1core import WORKER
from validation import bss, shift_to_prior

VALID = 2024
ARMS = ("base", "SHALLOW", "DEEP", "RAND")
CORE = "D_aug"                      # s41 등가 코어 (§7.175, <2024 학습)


def friend_frame(df, art, feats):
    """T1 predict.py:141-149 + user_fspec_features 부분집합."""
    sys.path.insert(0, str(T1))
    for m in ("features", "contam", "team_reference", "team_reference.features"):
        sys.modules.pop(m, None)
    from team_reference.features import build_features as bff
    f = bff(df, art)
    for c in df.columns:
        if c not in f.columns:
            f[c] = df[c].to_numpy()
    f["reverse_x_success"] = (
        df["asof_pitcher_reverse_rate"].to_numpy(dtype=np.float32)
        * df["asof_pitcher_success_rate"].to_numpy(dtype=np.float32))
    sys.path.pop(0)
    return f[list(feats)]


def main():
    t0 = time.time()
    art = joblib.load(T1 / "model" / "routing_artifacts.pkl")
    feats = joblib.load(T1 / "model" / "user_fspec_features.joblib")
    df = load_csv(TRAIN_CSV)
    tr = df[df.season < VALID].reset_index(drop=True)
    print(f"user_fspec 피처 {len(feats)}개  ({time.time()-t0:.0f}s)")

    # 전문가 학습셋 = 2023+ F (fold2024 에서는 2023 F 만)
    s = tr["season"].to_numpy(); g = tr["game_type"].to_numpy().astype(str)
    y = tr[TARGET_COL].to_numpy(dtype=np.float64)
    n = np.nan_to_num(tr["asof_pitcher_n"].to_numpy(dtype=float), nan=0.0)
    pool = (g == "F") & (s >= 2023)
    F = friend_frame(tr, art, feats)
    Xf, yf = F[pool].reset_index(drop=True), y[pool]
    tt = (s[pool] * 12 + tr.loc[pool, "game_month"].to_numpy(int))
    sub = {"SHALLOW": (n[pool] < 500) & (yf == 0),
           "DEEP": (n[pool] >= 500) & (yf == 0),
           "RAND": np.ones(int(pool.sum()), bool)}
    K = int(((n[pool] < 500) & (yf == 0)).sum())          # 얕음 풀 크기 = 공통 용량
    print(f"전문가 학습 {Xf.shape} · 얕음 {K:,} · 깊음 "
          f"{int(sub['DEEP'].sum()):,} · 용량 {K:,}  ({time.time()-t0:.0f}s)")

    for arm in ARMS:
        dst = ASCII / f"FSP_{arm}"
        if (dst / "model" / f"user_fspec_{FSPEC_SEEDS[-1]}.txt").exists():
            print(f"  FSP_{arm} 재사용"); continue
        shutil.copytree(ASCII / CORE, dst, dirs_exist_ok=True)   # ← s41 등가 코어
        X2, y2, t2 = Xf, yf, tt
        if arm != "base":
            rng = np.random.default_rng(777)
            m = sub[arm]
            new, sd = smote(Xf[m].to_numpy(dtype=np.float64), K, rng)
            src = np.flatnonzero(m)[sd]
            X2 = pd.concat([Xf, pd.DataFrame(new.astype(np.float32), columns=Xf.columns)],
                           ignore_index=True)
            y2 = np.concatenate([yf, yf[src]])       # 씨앗 라벨 (RAND 는 y=1 도 섞인다)
            t2 = np.concatenate([tt, tt[src]])
            print(f"  [FSP_{arm}] 씨앗 {int(m.sum()):,} → 합성 {len(new):,} "
                  f"(학습 {len(Xf):,} → {len(X2):,}, 양성비 {y2.mean():.3f})")
        cut = np.quantile(t2, 0.75)
        itr, iva = t2 <= cut, t2 > cut
        for sd_ in FSPEC_SEEDS:
            p = dict(FSPEC_PARAMS, seed=sd_, bagging_seed=sd_,
                     feature_fraction_seed=sd_, data_random_seed=sd_)
            probe = lgb.train(p, lgb.Dataset(X2[itr], y2[itr]),
                              num_boost_round=FSPEC_MAX_ROUNDS,
                              valid_sets=[lgb.Dataset(X2[iva], y2[iva])],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
            nr = max(50, probe.best_iteration)
            b = lgb.train(p, lgb.Dataset(X2, y2), num_boost_round=nr)
            (dst / "model" / f"user_fspec_{sd_}.txt").write_text(
                b.model_to_string(), encoding="utf-8", newline="\n")   # LF (§7.173③)
            print(f"    seed {sd_} rounds {nr} ({time.time()-t0:.0f}s)")
            del b, probe
        del X2, y2

    wk = ROOT / "experiments" / "cache" / "_t1fsp_worker.py"
    wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/")),
                  encoding="utf-8", newline="\n")
    P = {}
    for arm in ARMS:
        f = ROOT / "experiments" / "cache" / f"t1fsp_{arm}.npz"
        if not f.exists():
            r = subprocess.run([sys.executable, str(wk), f"FSP_{arm}", str(f)],
                               capture_output=True, text=True)
            if not f.exists():
                print(f"  ❌ {arm}: {r.stderr.strip()[-300:]}"); return
        P[arm] = np.load(f)["p"]
    for a in ARMS[1:]:
        assert np.abs(P[a] - P["base"]).max() > 0, f"{a} 가 base 와 같다"
    print("  배관 OK — base 대비 최대|Δp|: " +
          " ".join(f"{a} {np.abs(P[a]-P['base']).max():.5f}" for a in ARMS[1:]))

    va = df[df.season == VALID].reset_index(drop=True)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    isF = va["game_type"].to_numpy().astype(str) == "F"
    codes, uniq = pd.factorize(va["pitcher_id"].to_numpy())
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    seen = np.where(np.diff(bnd) > 0)[0]
    rng = np.random.default_rng(42)
    boots = [np.concatenate([order[bnd[j]:bnd[j+1]] for j in
                             rng.choice(seen, size=len(seen), replace=True)])
             for _ in range(400)]

    def s_(p, m=None, idx=None):
        pp, yy = (p, yva) if m is None else (p[m], yva[m])
        if idx is not None:
            pp, yy = pp[idx], yy[idx]
        return bss(yy, shift_to_prior(pp, yy.mean()))

    print("\n" + "=" * 88)
    print(f"user_fspec 증강 {K:,}행 · F행 가중 40% · fold2024 · ⚠️ fold 하나뿐")
    print("=" * 88)
    print("  %-34s %9s %10s %22s %8s" % ("팔", "BSS", "Δ vs base", "95% 구간", "P(Δ>0)"))
    print("  %-34s %9.1f" % ("base (전문가 재학습·증강 없음)", s_(P["base"])))
    desc = {"SHALLOW": "① 얕음 F·n<500·2023·실패  ← 후보",
            "DEEP": "② 깊음 F·n≥500·2023·실패  [대조]",
            "RAND": "③ 무작위 (전문가 풀 안에서) [영가설]"}
    for a in ARMS[1:]:
        d = np.array([s_(P[a], idx=ix) - s_(P["base"], idx=ix) for ix in boots])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print("  %-34s %9.1f %+10.2f   [%+7.2f, %+7.2f] %8.3f"
              % (desc[a], s_(P[a]), s_(P[a]) - s_(P["base"]), lo, hi, (d > 0).mean()))
    print("\n  F행 / R행 분해  (R행은 전부 같아야 정상 — 전문가는 F행에만 들어간다)")
    for a in ARMS:
        print("    %-10s F %8.1f   R %8.1f" % (a, s_(P[a], isF), s_(P[a], ~isF)))
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
