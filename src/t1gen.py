"""§7.173 — T1 의 **일반 LGBM 12** 를 증강 재학습해 fold2024 홀드아웃으로 잰다

사용자 지적: *"holdout 하면 되는거 아니야?"* — 맞다. §7.171 의 "홀드아웃 불가"는
**T1 자신의 구성요소를 분해할 때**(A~D 슬롯) 참이지, **우리가 재학습하는 부품을
바꿔 끼울 때**는 아니다.

T1 predict.py:154 —  `# R routing: Hyejin+Compact 90% + the user's general LGBM 10%.`
그 12개(`general_lgb_{base,wide,extra,lowff}_{42,7,2024}`)는 **우리 것**이고,
혜진 피처 프레임 위에서 학습됐다. 재학습에 필요한 것이 전부 공유본 안에 있다:
    team_reference/features.py · model/routing_artifacts.pkl · model/general_lgb_encoder.pkl

설계 — 모델 파일만 갈아 끼우고 **T1 전체 경로를 그대로 두 번** 돌린다.
    A) 일반 LGBM 12 를 season<2024 로 학습 (증강 없음)
    B) 같은 조건 + pre-2023 F·y=0 SMOTE 증강 13,357 행
    → 각각 t1work/model 사본에 덮어쓰고 Predictor 로 2024 행 예측 → BSS 대조

⚠️ **두 가지 in-sample 오염이 남는다. 둘 다 두 팔에 완전히 동일하다.**
   ① T1 코어(혜진 LGBM/MLP·compact·user_fspec)는 2019~2024 를 학습했다.
   ② `routing_artifacts.pkl` 도 전 시즌으로 적합돼 2024 타깃 통계가 들어 있다.
   → 절대 BSS 는 부풀려지고 **의미가 없다.** 읽을 것은 두 팔의 **차이**뿐이다.
   → 편향 방향: 코어가 과도하게 정확하면 10% 부품이 기여할 여지가 **줄어든다**
      = Δ 가 **과소추정**된다. 보수적이므로 "그래도 이기면 진짜" 로 읽는다.

⚠️ 일반 LGBM 의 원래 학습 설정(가중·init_score·soft label)은 공유본에 없다.
   여기서는 우리 표준(recency + soften, init_score 없음)으로 **두 팔 동일하게** 쓴다.
   따라서 A 팔은 T1 출하본과 같지 않다 — A vs B 만 비교한다.

실행:  .venv/Scripts/python.exe -u src/t1gen.py 2>&1 | tee experiments/t1gen.log
"""
import pickle
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
T1 = ROOT / "t1work"
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(T1))

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
from smotenc import n_needed, smote
from validation import bss, shift_to_prior

VALID = 2024
SEEDS = (42, 7, 2024)                       # T1 의 general_lgb 시드와 같다
VARIANTS = ("base", "wide", "extra", "lowff")
RATIO = 0.40


def friend_frame(df, art, enc):
    """T1 predict.py:141-161 과 **같은 순서**로 일반 LGBM 입력을 만든다."""
    from team_reference.features import build_features as bff
    f = bff(df, art)
    for c in df.columns:
        if c not in f.columns:
            f[c] = df[c].to_numpy()
    f["reverse_x_success"] = (
        df["asof_pitcher_reverse_rate"].to_numpy(dtype=np.float32)
        * df["asof_pitcher_success_rate"].to_numpy(dtype=np.float32))
    gf = f[enc["features"]].copy()
    for col, mp in enc["mappings"].items():
        gf[col] = (gf[col].astype("string").fillna("__MISSING__")
                   .map(mp).fillna(-1).astype(np.int32))
    return gf.replace([np.inf, -np.inf], np.nan).fillna(-999.0).astype(np.float32)


def build_arm(tr, G, w, augment, rng):
    """soften 후 (선택적으로) 증강. s40 과 같은 조작이다."""
    y = tr[TARGET_COL].to_numpy(dtype=np.float64)
    s = tr["season"].to_numpy(dtype=np.int64)
    g = tr["game_type"].astype("object").to_numpy().astype(str)
    X, y2, w2, _, h = soften.soften(tr, G, y, np.asarray(w, np.float64))
    if not augment:
        return X, y2, w2, 0
    m = (s <= 2022) & (g == "F") & (y == 0)
    k = n_needed(int(((s <= 2022) & (g == "F") & (y == 1)).sum()), int(m.sum()), RATIO)
    new, sd = smote(G[m].to_numpy(dtype=np.float64), k, rng)
    src = np.flatnonzero(m)[sd]
    X = pd.concat([X, pd.DataFrame(new.astype(np.float32), columns=G.columns)],
                  ignore_index=True)
    return (X, np.concatenate([y2, np.zeros(k)]),
            np.concatenate([w2, np.asarray(w, np.float64)[src]]), k)


def main():
    t0 = time.time()
    art = joblib.load(T1 / "model" / "routing_artifacts.pkl")
    enc = pickle.load(open(T1 / "model" / "general_lgb_encoder.pkl", "rb"))
    df = load_csv(TRAIN_CSV)
    tr = df[df.season < VALID].reset_index(drop=True)
    va = df[df.season == VALID].reset_index(drop=True)
    print(f"train {len(tr):,} / valid {len(va):,}  ({time.time()-t0:.0f}s)")

    G = friend_frame(tr, art, enc)
    print(f"일반LGBM 입력 {G.shape}  ({time.time()-t0:.0f}s)")
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)

    for arm, aug in (("A_base", False), ("B_aug", True)):
        md = ROOT / f"t1work_{arm}" / "model"
        if (md / "general_lgb_lowff_2024.txt").exists():
            print(f"  {arm} 재사용")
            continue
        md.parent.mkdir(parents=True, exist_ok=True)
        if not md.exists():
            shutil.copytree(T1, md.parent, dirs_exist_ok=True)
        rng = np.random.default_rng(777)
        X, y2, w2, k = build_arm(tr, G, w, aug, rng)
        print(f"  [{arm}] 행 {len(G):,} → {len(X):,} (합성 {k:,})")
        for v in VARIANTS:
            for sd in SEEDS:
                p = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v], seed=sd,
                         bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
                b = lgb.train(p, lgb.Dataset(X, y2, weight=w2), num_boost_round=N_ROUNDS)
                (md / f"general_lgb_{v}_{sd}.txt").write_text(
                    b.model_to_string(), encoding="utf-8")
                del b
            print(f"    {v} 3시드 ({time.time()-t0:.0f}s)")
        del X, y2, w2

    # ── T1 전체 경로로 2024 예측 (모델 파일만 다르다) ─────────────────
    P = {}
    for arm in ("A_base", "B_aug"):
        for mod in ("predict", "features", "mlp_infer", "contam",
                    "team_reference", "team_reference.features",
                    "team_reference.mlp_infer"):
            sys.modules.pop(mod, None)
        sys.path.insert(0, str(ROOT / f"t1work_{arm}"))
        from predict import Predictor
        pr = Predictor(model_dir=str(ROOT / f"t1work_{arm}" / "model"))
        P[arm] = np.asarray(pr.predict(va), dtype=np.float64)
        sys.path.pop(0)
        del pr, Predictor
        print(f"  {arm} 예측 완료 mean={P[arm].mean():.4f}  ({time.time()-t0:.0f}s)")

    yb = yva.mean()
    codes, uniq = pd.factorize(va["pitcher_id"].to_numpy())
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    seen = np.where(np.diff(bnd) > 0)[0]
    rng = np.random.default_rng(42)

    def s_(p, yy, idx=None):
        if idx is not None:
            p, yy = p[idx], yy[idx]
        return bss(yy, shift_to_prior(p, yy.mean()))

    d = []
    for _ in range(400):
        ix = np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                             rng.choice(seen, size=len(seen), replace=True)])
        d.append(s_(P["B_aug"], yva, ix) - s_(P["A_base"], yva, ix))
    lo, hi = np.percentile(d, [2.5, 97.5])
    print("\n" + "=" * 78)
    print("T1 전체 경로 · fold2024 (⚠️ 절대값은 in-sample 오염으로 무의미, 차이만 읽는다)")
    print("=" * 78)
    print(f"  A 증강 없음   BSS {s_(P['A_base'], yva):9.1f}")
    print(f"  B 증강        BSS {s_(P['B_aug'], yva):9.1f}")
    print(f"  Δ(B−A)        {s_(P['B_aug'], yva) - s_(P['A_base'], yva):+9.2f}"
          f"   95% [{lo:+.2f}, {hi:+.2f}]   P(Δ>0)={np.mean(np.array(d) > 0):.3f}")
    print(f"  (일반 LGBM 유효 가중 = 0.10 × R행 {100*(1-(va.game_type=='F').mean()):.0f}%"
          f" = {10*(1-(va.game_type=='F').mean()):.1f}%)")
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
