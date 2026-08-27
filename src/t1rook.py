"""§7.176 — s41 위에 **얕은 이력 F 실패행** 증강을 더한다 (사용자 제안)

사용자: *"F 리그에서 신입들의 경우 잘 못한다고 했는데, 2023부터 나온 신입들에
대해서만 증강해볼 수 있나."*

근거는 있다 — §7.79 가 `F 신인 − 2+시즌 편향 +0.0155 ± 0.0073 (t +2.1)` 로
**F 신인 과대예측**을 실측했다. 당시엔 t 가 통과선(≥3) 미달이고 F 신인이 행의 2.5%
라 δ 시프트로는 안 옮겼는데, **증강은 다른 도구**다.

#### 정의를 바꾼 이유
데뷔 연도로 자르면 fold2024 학습셋 안에 **1,739행**뿐이다 (s41 용량의 13%) —
기대 Δ +0.17 에 구간 ±0.8 이라 측정이 불가능하다. §7.79 후속(MODEL.md:13034)이
기전을 이미 적어뒀다: *"통산<500 행이 F 36.3% vs R 9.5%, F 신인의 중앙 asof_n 110
vs R 신인 679"*. **"신인" 의 실체는 데뷔 연도가 아니라 이력 깊이**이고, 그쪽이 크다.

#### 팔 — 전부 **2023 행만** (s41 이 증강하는 pre-2023 과 겹치지 않는다)

    base   D_aug (= s41 등가, <2024 학습, pre-2023 F 실패행 증강 ON)  ← 이미 학습됨
    ① 얕음  + F · asof_pitcher_n<500  · 2023 · y=0   씨앗 4,395
    ② 깊음  + F · asof_pitcher_n>=500 · 2023 · y=0   씨앗 9,144   ← "얕음" 을 격리
    ③ 무작위 + 학습셋 아무 행                          ← "표적" 을 격리

    합성 개수는 셋 다 **8,790** (얕음 풀의 2배, s41 13,357 의 66%)

#### 왜 대조군이 둘인가 (사용자 질문)
    무작위   "대상 선택이 중요한가" — §7.162d 에서 −0.51 로 이미 죽었으나
             **s41 위에서는 미측정**이라 다시 건다.
    깊음     "얕은 이력이 조건인가, 그냥 F 실패행이면 되는가" — 무작위로는 못 가른다.
    ⚠️ 깊음은 유효 성분(F 실패행)을 공유하므로 **위약이 아니라 두 번째 치료**다.
       둘 다 양수면 "얕음은 무관" 이 결론이지 위약 통과가 아니다.

⚠️ **fold2024 하나뿐이다** — 2023 행을 쓰므로 fold2023(train ≤2022)에는 존재하지
   않는다. 부호 일치 관문을 못 쓴다. 대조군 차감이 유일한 판정자다.

실행:  .venv/Scripts/python.exe -u src/t1rook.py 2>&1 | tee experiments/t1rook.log
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
from smotenc import n_needed, smote
from t1core import DROP, RATIO, SEEDS, VARIANTS, WORKER, t1_features

VALID = 2024
N_SYN = 8790                       # 얕음 풀(4,395)의 2배. 세 팔 공통
ARMS = ("SHALLOW", "DEEP", "RAND")


def extra_pool(tr, kind):
    """추가 증강 대상. 전부 **2023 행만** — s41 의 pre-2023 증강과 겹치지 않는다."""
    s = tr["season"].to_numpy()
    g = tr["game_type"].to_numpy().astype(str)
    y = tr[TARGET_COL].to_numpy()
    n = np.nan_to_num(tr["asof_pitcher_n"].to_numpy(dtype=float), nan=0.0)
    if kind == "SHALLOW":
        return (g == "F") & (n < 500) & (s == 2023) & (y == 0)
    if kind == "DEEP":
        return (g == "F") & (n >= 500) & (s == 2023) & (y == 0)
    return np.ones(len(tr), bool)          # RAND — 학습셋 전체


def build(tr, X, w, kind, rng):
    """s41 증강(pre-2023 F) + 추가 증강(kind). 순서 = [원본][soften][s41][추가]."""
    y = tr[TARGET_COL].to_numpy(dtype=np.float64)
    s = tr["season"].to_numpy(dtype=np.int64)
    g = tr["game_type"].astype("object").to_numpy().astype(str)
    X1, y1, w1, _, h = soften.soften(tr, X, y, np.asarray(w, np.float64))
    hr = np.zeros(len(tr))
    for yr, v in h.items():
        hr[(s == yr) & (g == "F")] = v
    dm = (hr > 0) & (y == 1)
    ss, gt = np.concatenate([s, s[dm]]), np.concatenate([g, g[dm]])

    def add(mask, k, lab_from_seed):
        nonlocal X1, y1, w1, ss, gt
        sub = X[mask].to_numpy(dtype=np.float64)
        idx = np.flatnonzero(mask)
        if lab_from_seed:                  # RAND — 라벨 그룹 안에서만 보간
            news, srcs, labs = [], [], []
            for lv in (0.0, 1.0):
                sel = idx[y[idx] == lv]
                kk = int(round(k * len(sel) / len(idx)))
                if kk < 1 or len(sel) < 6:
                    continue
                nw, sd = smote(X.iloc[sel].to_numpy(dtype=np.float64), kk, rng)
                news.append(nw); srcs.append(sel[sd]); labs.append(np.full(kk, lv))
            new, src, lab = np.concatenate(news), np.concatenate(srcs), np.concatenate(labs)
        else:
            new, sd = smote(sub, k, rng)
            src, lab = idx[sd], np.zeros(k)
        X1 = pd.concat([X1, pd.DataFrame(new.astype(np.float32), columns=X.columns)],
                       ignore_index=True)
        y1 = np.concatenate([y1, lab])
        w1 = np.concatenate([w1, np.asarray(w, np.float64)[src]])
        ss, gt = np.concatenate([ss, s[src]]), np.concatenate([gt, g[src]])
        return len(new)

    # ① s41 이 하는 것 — pre-2023 F & y=0
    m41 = (s <= 2022) & (g == "F") & (y == 0)
    k41 = n_needed(int(((s <= 2022) & (g == "F") & (y == 1)).sum()), int(m41.sum()), RATIO)
    n41 = add(m41, k41, False)
    # ② 추가
    pool = extra_pool(tr, kind)
    nex = add(pool, N_SYN, kind == "RAND")
    return X1, y1, w1, season_gt_level(ss, gt, y1, w1), n41, nex, int(pool.sum())


def main():
    t0 = time.time()
    bundle = joblib.load(T1 / "model" / "artifacts.joblib")
    art = bundle["artifacts"]
    cols = lgb.Booster(model_str=(T1 / "model" / "lgbm_base_42.txt")
                       .read_text(encoding="utf-8")).feature_name()
    df = load_csv(TRAIN_CSV)
    tr = df[df.season < VALID].reset_index(drop=True)
    X = t1_features(tr, art, cols)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    print(f"학습 피처 {X.shape}  ({time.time()-t0:.0f}s)")

    for kind in ARMS:
        dst = ASCII / kind
        if (dst / "model" / f"lgbm_lowff_{SEEDS[-1]}.txt").exists():
            print(f"  {kind} 재사용"); continue
        shutil.copytree(T1, dst, dirs_exist_ok=True)
        rng = np.random.default_rng(777)
        X2, y2, w2, init, n41, nex, npool = build(tr, X, w, kind, rng)
        print(f"  [{kind}] 풀 {npool:,} · s41합성 {n41:,} + 추가합성 {nex:,} "
              f"→ 행 {len(X):,} → {len(X2):,}")
        for v in VARIANTS:
            for sd in SEEDS:
                p = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v], seed=sd,
                         bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
                b = lgb.train(p, lgb.Dataset(X2, y2, weight=w2, init_score=init),
                              num_boost_round=N_ROUNDS)
                (dst / "model" / f"lgbm_{v}_{sd}.txt").write_text(
                    b.model_to_string(), encoding="utf-8", newline="\n")
                del b
            print(f"    {v} ({time.time()-t0:.0f}s)")
        del X2, y2, w2, init
    del X, tr

    wk = ROOT / "experiments" / "cache" / "_t1rook_worker.py"
    wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/")),
                  encoding="utf-8", newline="\n")
    P = {}
    for arm in ("D_aug",) + ARMS:
        f = ROOT / "experiments" / "cache" / f"t1rook_{arm}.npz"
        if not f.exists():
            r = subprocess.run([sys.executable, str(wk), arm, str(f)],
                               capture_output=True, text=True)
            if not f.exists():
                print(f"  ❌ {arm}: {r.stderr.strip()[-300:]}"); return
        P[arm] = np.load(f)["p"]
    for a in ARMS:
        d = float(np.abs(P[a] - P["D_aug"]).max())
        assert d > 0, f"{a} 가 base 와 같다 — 증강 미적용"
    print(f"  배관 OK — base 대비 최대|Δp|: " +
          " ".join(f"{a} {np.abs(P[a]-P['D_aug']).max():.5f}" for a in ARMS))

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

    print("\n" + "=" * 84)
    print(f"s41(=base) 위에 추가 증강 {N_SYN:,}행 · fold2024 · ⚠️ fold 하나뿐")
    print("=" * 84)
    print("  %-30s %9s %9s %22s %8s" % ("팔", "BSS", "Δ vs base", "95% 구간", "P(Δ>0)"))
    print("  %-30s %9.1f" % ("base = s41 등가", s_(P["D_aug"])))
    desc = {"SHALLOW": "① 얕음 F·n<500·2023·실패  ← 후보",
            "DEEP":    "② 깊음 F·n≥500·2023·실패  [대조]",
            "RAND":    "③ 무작위 아무 행           [영가설]"}
    for a in ARMS:
        d = np.array([s_(P[a], idx=ix) - s_(P["D_aug"], idx=ix) for ix in boots])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print("  %-30s %9.1f %+9.2f   [%+7.2f, %+7.2f] %8.3f"
              % (desc[a], s_(P[a]), s_(P[a]) - s_(P["D_aug"]), lo, hi, (d > 0).mean()))
    print("\n  F행 / R행 분해")
    for a in ("D_aug",) + ARMS:
        print("    %-12s F %8.1f   R %8.1f" % (a, s_(P[a], isF), s_(P[a], ~isF)))
    print(f"\n  총 {time.time()-t0:.0f}s")


from validation import bss, shift_to_prior  # noqa: E402

if __name__ == "__main__":
    main()
