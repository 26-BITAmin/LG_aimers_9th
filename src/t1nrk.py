"""§7.178 — s41 의 증강 **표적을 바꾼다**: pre-2023 F 전체 → **F 신인 (2019 제외)**

사용자: *"아니면 그냥 2019 제외 F 신입으로 바꿔서 해보자."*

#### 왜 2019 를 빼는가 — 좌측 절단
`train.csv` 가 2019 부터라 그 이전부터 뛰던 투수도 전부 "2019 데뷔" 로 잡힌다.
데뷔 시즌별 투수 수: **2019 355명** · 2020 110 · 2021 95 · 2022 88 · 2023 63 · 2024 81.
진짜 신인 코호트는 연 60~110 명이니 355 는 명백한 절단 산물이다. `asof_pitcher_n`
도 2019 에는 전원 0 에서 시작해 **모두 "얕은 이력" 으로 위장**된다.
2019 를 빼면 F 신인행이 49,685 → **23,899** (실패 17,010 → **8,997**). 절반이 가짜였다.

#### §7.176 과 다른 질문이다
§7.176 은 s41 **위에 더 얹었고**(이미 포화) 실패했다. 여기는 s41 의 **표적을 교체**한다 —
용량은 같고 어디를 겨누느냐만 다르다. 그리고 §7.176 이 죽은 자리는 **전문가**(25.7k행,
17% 주입이 과적합)였고 여기는 **코어**(1.29M행)다 — s41 이 실제로 작동하는 자리.

#### 팔 (전부 코어 LGBM 20, base 제외 전부 s41 증강을 **대체**)
    base    s41 등가 = pre-2023 F 실패 (씨앗 34,109 → 합성 13,357, 0.39배)  ← 학습됨
    ①a 신인   데뷔시즌·F·2020+·실패 (씨앗 8,997)  → 합성 13,357  **용량 맞춤** (1.48배)
    ①b 신인   같은 풀                              → 합성  3,509  **배율 맞춤** (0.39배)
    ②  비신인 데뷔이후·F·2020+·실패 (씨앗 30,638) → 합성 13,357  [대조] (0.44배)

⚠️ ①a 와 ② 는 용량이 같지만 **오버샘플링 배율이 1.48 vs 0.44** 로 다르다. 좁은 풀을
   겨누면 씨앗당 반복이 늘어나는 것이 내재적이라 피할 수 없다 — 그래서 ①b(배율 맞춤)
   를 같이 걸어 "신인이 나쁜가" 와 "반복이 나쁜가" 를 가른다.
⚠️ fold2024 하나뿐. 대조군 차감이 유일한 판정자다.

실행:  .venv/Scripts/python.exe -u src/t1nrk.py 2>&1 | tee experiments/t1nrk.log
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

import soften
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, N_ROUNDS,
                    RECENCY_HALF_LIFE, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from hbeta2 import season_gt_level
from smotenc import smote
from t1core import SEEDS, VARIANTS, WORKER, t1_features
from validation import bss, shift_to_prior

VALID = 2024
#  (팔, 풀 종류, 합성 개수)
ARMS = [("RK_dose", "rookie", 13357),
        ("RK_rate", "rookie", 3509),
        ("NRK", "nonrookie", 13357)]


def pools(tr, df):
    s = tr["season"].to_numpy()
    g = tr["game_type"].to_numpy().astype(str)
    y = tr[TARGET_COL].to_numpy()
    dz = tr["pitcher_id"].map(df.groupby("pitcher_id").season.min()).to_numpy()
    return {"rookie": (dz == s) & (g == "F") & (s >= 2020) & (y == 0),
            "nonrookie": (dz < s) & (g == "F") & (s >= 2020) & (y == 0)}


def build(tr, X, w, mask, k, rng):
    """soften 후 mask 풀에서 k 행 합성. **s41 증강은 하지 않는다 (표적 교체)**."""
    y = tr[TARGET_COL].to_numpy(dtype=np.float64)
    s = tr["season"].to_numpy(dtype=np.int64)
    g = tr["game_type"].astype("object").to_numpy().astype(str)
    X1, y1, w1, _, h = soften.soften(tr, X, y, np.asarray(w, np.float64))
    hr = np.zeros(len(tr))
    for yr, v in h.items():
        hr[(s == yr) & (g == "F")] = v
    dm = (hr > 0) & (y == 1)
    ss, gt = np.concatenate([s, s[dm]]), np.concatenate([g, g[dm]])
    new, sd = smote(X[mask].to_numpy(dtype=np.float64), k, rng)
    src = np.flatnonzero(mask)[sd]
    X1 = pd.concat([X1, pd.DataFrame(new.astype(np.float32), columns=X.columns)],
                   ignore_index=True)
    y1 = np.concatenate([y1, np.zeros(k)])
    w1 = np.concatenate([w1, np.asarray(w, np.float64)[src]])
    ss, gt = np.concatenate([ss, s[src]]), np.concatenate([gt, g[src]])
    return X1, y1, w1, season_gt_level(ss, gt, y1, w1)


def main():
    t0 = time.time()
    art = joblib.load(T1 / "model" / "artifacts.joblib")["artifacts"]
    cols = lgb.Booster(model_str=(T1 / "model" / "lgbm_base_42.txt")
                       .read_text(encoding="utf-8")).feature_name()
    df = load_csv(TRAIN_CSV)
    tr = df[df.season < VALID].reset_index(drop=True)
    X = t1_features(tr, art, cols)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    PL = pools(tr, df)
    print(f"피처 {X.shape} · 신인 풀 {int(PL['rookie'].sum()):,} · "
          f"비신인 풀 {int(PL['nonrookie'].sum()):,}  ({time.time()-t0:.0f}s)")

    for arm, pk, k in ARMS:
        dst = ASCII / f"NRK_{arm}"
        if (dst / "model" / f"lgbm_lowff_{SEEDS[-1]}.txt").exists():
            print(f"  {arm} 재사용"); continue
        shutil.copytree(T1, dst, dirs_exist_ok=True)
        rng = np.random.default_rng(777)
        m = PL[pk]
        X2, y2, w2, init = build(tr, X, w, m, k, rng)
        print(f"  [{arm}] 씨앗 {int(m.sum()):,} → 합성 {k:,} "
              f"({k/m.sum():.2f}배)  행 {len(X):,} → {len(X2):,}")
        for v in VARIANTS:
            for sd_ in SEEDS:
                p = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v], seed=sd_,
                         bagging_seed=sd_, feature_fraction_seed=sd_, data_random_seed=sd_)
                b = lgb.train(p, lgb.Dataset(X2, y2, weight=w2, init_score=init),
                              num_boost_round=N_ROUNDS)
                (dst / "model" / f"lgbm_{v}_{sd_}.txt").write_text(
                    b.model_to_string(), encoding="utf-8", newline="\n")
                del b
            print(f"    {v} ({time.time()-t0:.0f}s)")
        del X2, y2, w2, init
    del X, tr

    wk = ROOT / "experiments" / "cache" / "_t1nrk_worker.py"
    wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/")),
                  encoding="utf-8", newline="\n")
    P = {}
    for arm, dname in [("base", "D_aug")] + [(a, f"NRK_{a}") for a, _, _ in ARMS]:
        f = ROOT / "experiments" / "cache" / f"t1nrk_{arm}.npz"
        if not f.exists():
            r = subprocess.run([sys.executable, str(wk), dname, str(f)],
                               capture_output=True, text=True)
            if not f.exists():
                print(f"  ❌ {arm}: {r.stderr.strip()[-300:]}"); return
        P[arm] = np.load(f)["p"]
    for a, _, _ in ARMS:
        assert np.abs(P[a] - P["base"]).max() > 0, f"{a} 가 base 와 같다"
    print("  배관 OK — base 대비 최대|Δp|: " +
          " ".join(f"{a} {np.abs(P[a]-P['base']).max():.5f}" for a, _, _ in ARMS))

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

    print("\n" + "=" * 90)
    print("s41 의 증강 **표적 교체** · 코어 LGBM 20 · fold2024 · ⚠️ fold 하나뿐")
    print("=" * 90)
    print("  %-38s %9s %10s %22s %8s" % ("팔", "BSS", "Δ vs s41", "95% 구간", "P(Δ>0)"))
    print("  %-38s %9.1f" % ("base = s41 (pre-2023 F 전체, 0.39배)", s_(P["base"])))
    desc = {"RK_dose": "①a 신인 2019제외 · 13,357 (1.48배) ← 후보",
            "RK_rate": "①b 신인 2019제외 ·  3,509 (0.39배) 배율맞춤",
            "NRK":     "②  비신인       · 13,357 (0.44배) [대조]"}
    for a, _, _ in ARMS:
        d = np.array([s_(P[a], idx=ix) - s_(P["base"], idx=ix) for ix in boots])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print("  %-38s %9.1f %+10.2f   [%+7.2f, %+7.2f] %8.3f"
              % (desc[a], s_(P[a]), s_(P[a]) - s_(P["base"]), lo, hi, (d > 0).mean()))
    print("\n  F행 / R행 분해")
    for a in ["base"] + [x[0] for x in ARMS]:
        print("    %-10s F %8.1f   R %8.1f" % (a, s_(P[a], isF), s_(P[a], ~isF)))
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
