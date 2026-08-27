"""§7.175 — T1 **코어 LGBM 20** 을 증강 재학습한다 (R행 22.5% 슬롯)

§7.173 은 일반 LGBM 12(유효 8.8%)를 재학습해 Δ +0.24 [−0.27, +0.75] 를 얻었다.
면적이 작아 해상도 아래였다. 코어는 **2.5배** 크다:

    R행 최종 로짓에서 LGBM 몫
      일반 LGBM   0.10                    → 유효 8.8%
      코어 LGBM   0.90 × 0.25 = 0.225     → 유효 22.5%   ← 여기

사용자가 브랜치를 보라고 해서 확인했더니 `origin/hyejin` 에 학습 코드가 있었다
(`experiments/09_strong_baseline_cv.py`, 4변형×5시드). 그런데 **그건 T1 이 쓴 계보가
아니다** — 브랜치 `src/features.py`(16KB)와 zip 의 `team_reference/features.py`(34KB)는
공통 함수가 0개다. T1 코어가 실제로 쓰는 것은 `predict.py:102` 의
`build_features(test_df, self.art)` = **`t1work/features.py`**, 즉 **우리 계보**다.

확인된 사실
    T1 코어 lgbm_base_42   피처 78 · 트리 320   ← 우리 s40 과 피처 이름·개수 동일
    T1 코어 == 우리 s37     0/20 (별도 학습 실행본)
    meta.json: use_init_score True · use_player_ids True · recency 0.5 ·
               drop [game_type, game_month, game_dayofweek] · contam (p,b) · soft_label True

→ 남의 코드가 필요 없다. **T1 자신의 features.py + artifacts.joblib** 으로 피처를
  만들고 우리 표준 LGBM 루프로 20개를 학습해 파일만 갈아 끼운다.

팔
    C_base  코어 LGBM 20, season<2024, 증강 없음
    D_aug   같은 조건 + pre-2023 F·y=0 SMOTE 13,357 행

⚠️ §7.173 과 같은 in-sample 오염이 남는다 — T1 의 나머지(MLP 8 · compact · fspec ·
   user_fspec · artifacts)는 2019~2024 를 봤다. **두 팔에 동일**하므로 절대값은
   무의미하고 차이만 읽는다. 편향은 과소추정 방향이다.
⚠️ 모델 파일은 **LF 로 쓴다** (§7.173 ③ — write_text 의 CRLF 가 model_file= 을 죽인다).

실행:  .venv/Scripts/python.exe -u src/t1core.py 2>&1 | tee experiments/t1core.log
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
from validation import bss, shift_to_prior

VALID = 2024
SEEDS = (42, 7, 2024, 1234, 31337)        # T1 meta.json 과 동일
VARIANTS = ("base", "wide", "extra", "lowff")
DROP = ["game_type", "game_month", "game_dayofweek"]   # T1 meta.json
RATIO = 0.40
ARMS = (("C_base", False), ("D_aug", True))

WORKER = r"""
import os, sys, numpy as np, pandas as pd
arm, out = sys.argv[1], sys.argv[2]
d = r"D:/t1eval/" + arm
sys.path.insert(0, d)          # ⚠️ 우리 src 를 넣지 않는다 (§7.174 사고)
va = pd.read_csv(r"{csv}")
va = va[va.season == 2024].reset_index(drop=True)
from predict import Predictor
import predict as _pm
assert _pm.__file__.replace("\\", "/").startswith(d), "T1 predict.py 가 아니다"
p = Predictor(model_dir=d + "/model")
np.savez_compressed(out, p=np.asarray(p.predict(va), dtype=np.float64))
"""


def t1_features(df, art, cols):
    """T1 자신의 features.py 로 피처를 만든다 — 추론 때와 **같은 코드·같은 아티팩트**."""
    sys.path.insert(0, str(T1))
    for m in ("features", "contam"):
        sys.modules.pop(m, None)
    import features as t1f
    assert t1f.__file__.replace("\\", "/").startswith(str(T1).replace("\\", "/")), t1f.__file__
    X = t1f.build_features(df, art)
    for c in ("pitcher_id", "batter_id"):
        X[c] = df[c].to_numpy(dtype=np.float32)
    X = X.drop(columns=[c for c in DROP if c in X.columns])
    sys.path.pop(0)
    missing = [c for c in cols if c not in X.columns]
    assert not missing, f"코어가 기대하는 피처 누락: {missing[:6]}"
    return X[list(cols)]                   # **부스터가 기대하는 순서 그대로**


def build(tr, X, w, augment, rng):
    y = tr[TARGET_COL].to_numpy(dtype=np.float64)
    s = tr["season"].to_numpy(dtype=np.int64)
    g = tr["game_type"].astype("object").to_numpy().astype(str)
    X1, y1, w1, _, h = soften.soften(tr, X, y, np.asarray(w, np.float64))
    hr = np.zeros(len(tr))
    for yr, v in h.items():
        hr[(s == yr) & (g == "F")] = v
    dm = (hr > 0) & (y == 1)
    ss, gt = np.concatenate([s, s[dm]]), np.concatenate([g, g[dm]])
    k = 0
    if augment:
        m = (s <= 2022) & (g == "F") & (y == 0)
        k = n_needed(int(((s <= 2022) & (g == "F") & (y == 1)).sum()), int(m.sum()), RATIO)
        new, sd = smote(X[m].to_numpy(dtype=np.float64), k, rng)
        src = np.flatnonzero(m)[sd]
        X1 = pd.concat([X1, pd.DataFrame(new.astype(np.float32), columns=X.columns)],
                       ignore_index=True)
        y1 = np.concatenate([y1, np.zeros(k)])
        w1 = np.concatenate([w1, np.asarray(w, np.float64)[src]])
        ss, gt = np.concatenate([ss, s[src]]), np.concatenate([gt, g[src]])
    return X1, y1, w1, season_gt_level(ss, gt, y1, w1), k


def main():
    t0 = time.time()
    bundle = joblib.load(T1 / "model" / "artifacts.joblib")
    art = bundle["artifacts"]
    cols = lgb.Booster(model_str=(T1 / "model" / "lgbm_base_42.txt")
                       .read_text(encoding="utf-8")).feature_name()
    print(f"코어가 기대하는 피처 {len(cols)}개  ({time.time()-t0:.0f}s)")

    df = load_csv(TRAIN_CSV)
    tr = df[df.season < VALID].reset_index(drop=True)
    X = t1_features(tr, art, cols)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    print(f"학습 피처 {X.shape}  ({time.time()-t0:.0f}s)")

    for arm, aug in ARMS:
        dst = ASCII / arm
        if (dst / "model" / f"lgbm_lowff_{SEEDS[-1]}.txt").exists():
            print(f"  {arm} 재사용")
            continue
        shutil.copytree(T1, dst, dirs_exist_ok=True)
        rng = np.random.default_rng(777)
        X2, y2, w2, init, k = build(tr, X, w, aug, rng)
        print(f"  [{arm}] 행 {len(X):,} → {len(X2):,} (합성 {k:,})")
        for v in VARIANTS:
            for sd in SEEDS:
                p = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v], seed=sd,
                         bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
                b = lgb.train(p, lgb.Dataset(X2, y2, weight=w2, init_score=init),
                              num_boost_round=N_ROUNDS)
                # ⚠️ LF 로 쓴다 — write_text 의 CRLF 가 model_file= 을 죽인다 (§7.173 ③)
                (dst / "model" / f"lgbm_{v}_{sd}.txt").write_text(
                    b.model_to_string(), encoding="utf-8", newline="\n")
                del b
            print(f"    {v} {len(SEEDS)}시드 ({time.time()-t0:.0f}s)")
        del X2, y2, w2, init
    del X, tr

    wk = ROOT / "experiments" / "cache" / "_t1core_worker.py"
    wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/")),
                  encoding="utf-8", newline="\n")
    P = {}
    for arm, _ in ARMS:
        f = ROOT / "experiments" / "cache" / f"t1core_{arm}.npz"
        if not f.exists():
            r = subprocess.run([sys.executable, str(wk), arm, str(f)],
                               capture_output=True, text=True)
            if not f.exists():
                print(f"  ❌ {arm}: {r.stderr.strip()[-300:]}")
                return
        P[arm] = np.load(f)["p"]
        print(f"  {arm} 예측 mean={P[arm].mean():.4f}  ({time.time()-t0:.0f}s)")

    dmax = float(np.abs(P["D_aug"] - P["C_base"]).max())
    assert dmax > 0, "두 팔이 같다 — 모델 교체 실패"
    print(f"  배관 OK — 두 팔 최대 예측차 {dmax:.6f}")

    va = df[df.season == VALID].reset_index(drop=True)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    isF = va["game_type"].to_numpy().astype(str) == "F"
    codes, uniq = pd.factorize(va["pitcher_id"].to_numpy())
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    seen = np.where(np.diff(bnd) > 0)[0]
    rng = np.random.default_rng(42)

    def s_(p, m=None, idx=None):
        pp, yy = (p, yva) if m is None else (p[m], yva[m])
        if idx is not None:
            pp, yy = pp[idx], yy[idx]
        return bss(yy, shift_to_prior(pp, yy.mean()))

    d = []
    for _ in range(400):
        ix = np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                             rng.choice(seen, size=len(seen), replace=True)])
        d.append(s_(P["D_aug"], idx=ix) - s_(P["C_base"], idx=ix))
    lo, hi = np.percentile(d, [2.5, 97.5])
    a, b = s_(P["C_base"]), s_(P["D_aug"])
    print("\n" + "=" * 80)
    print("T1 **코어 LGBM 20** 증강 · fold2024  ⚠️ 절대값 무의미(in-sample), 차이만")
    print("=" * 80)
    print(f"  C 증강 없음   BSS {a:9.1f}")
    print(f"  D 증강        BSS {b:9.1f}")
    print(f"  Δ(D−C)        {b-a:+9.2f}   95% [{lo:+.2f}, {hi:+.2f}]"
          f"   P(Δ>0)={np.mean(np.array(d) > 0):.3f}")
    print(f"  R행 {s_(P['C_base'], ~isF):.1f} → {s_(P['D_aug'], ~isF):.1f}   "
          f"F행 {s_(P['C_base'], isF):.1f} → {s_(P['D_aug'], isF):.1f}")
    print(f"  (참고: 일반 LGBM 12(8.8%) 는 +0.24 [−0.27,+0.75] 였다 — 여기는 22.5%)")
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
