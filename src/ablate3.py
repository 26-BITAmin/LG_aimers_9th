"""4차 실험 — init_score로 시즌/게임타입 레벨을 분리한다.

지금까지의 결론은 "오래된 데이터가 해롭다"였다. 하지만 해로운 것은 오래된 데이터의
**레벨**(시즌별 양성률, game_type 정의 변경)이지 **조건부 구조**가 아닐 수 있다.

    logit(p_i) = base_i + f(x_i)
    base_i = logit( 학습셋에서 관측한 (season_i, game_type_i)의 양성률 )

부스팅에 base_i를 init_score로 주면 f는 "그 시즌/구분의 평균 대비 얼마나 벗어나는가"만
학습한다. 추론 시엔 f(x)에 2025용 base 하나만 더한다.
→ 레벨 드리프트는 base가, 구조는 f가 담당하도록 분리하는 것이 목적.

실행:  .venv/Scripts/python.exe -u src/ablate3.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import lightgbm as lgb
import numpy as np

from ablate import BASE_PARAMS, r4_prior
from config import TARGET_COL, TRAIN_CSV
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import _logit, bss, shift_to_prior


def evaluate_init(df, season_rate, va_season, *, train_seasons, half_life,
                  base_by, params=BASE_PARAMS, drop_cols=(), add_ids=False,
                  k=200.0, rounds=4000):
    """base_by: None | 'season' | 'season_gt'  — init_score를 무엇 단위로 뺄 것인가."""
    tr = df[df.season.isin(train_seasons)]
    va = df[df.season == va_season]

    w = (np.ones(len(tr)) if half_life is None
         else recency_weight(tr.season.to_numpy(), half_life, ref_season=max(train_seasons)))

    art = fit_artifacts(tr, weights=w, k_pitcher=k, k_batter=k)
    Xtr, Xva = build_features(tr, art), build_features(va, art)
    if add_ids:
        for c in ("pitcher_id", "batter_id"):
            Xtr[c] = tr[c].to_numpy(dtype=np.float32)
            Xva[c] = va[c].to_numpy(dtype=np.float32)
    if drop_cols:
        Xtr, Xva = Xtr.drop(columns=list(drop_cols)), Xva.drop(columns=list(drop_cols))

    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)

    # ── base(=init_score) 산출: train에서만 계산한다 ──────────
    last = max(train_seasons)
    if base_by is None:
        init_tr = None
        base_va = 0.0
    elif base_by == "season":
        rates = tr.groupby("season")[TARGET_COL].mean()
        init_tr = _logit(tr["season"].map(rates).to_numpy(dtype=np.float64))
        base_va = float(_logit(np.array([rates[last]]))[0])
    elif base_by == "season_gt":
        rates = tr.groupby(["season", "game_type"], observed=True)[TARGET_COL].mean()
        key = list(zip(tr["season"].to_numpy(), tr["game_type"].astype("object").to_numpy()))
        init_tr = _logit(np.array([rates[k_] for k_ in key]))
        # 추론에는 game_type별 2025 레벨 추정치가 없다 → 직전 시즌 전체 레벨 하나만 사용
        base_va = float(_logit(np.array([tr[tr.season == last][TARGET_COL].mean()]))[0])
    else:
        raise ValueError(base_by)

    dtr = lgb.Dataset(Xtr, ytr, weight=w, init_score=init_tr, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr,
                      init_score=(None if init_tr is None else np.full(len(Xva), base_va)),
                      free_raw_data=False)
    bst = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])

    raw = bst.predict(Xva, num_iteration=bst.best_iteration, raw_score=True)
    p = 1.0 / (1.0 + np.exp(-(base_va + raw)))

    est = r4_prior(season_rate, train_seasons)
    return {"iter": bst.best_iteration,
            "bss_raw": bss(yva, p),
            "bss_r4": bss(yva, shift_to_prior(p, est)),
            "bss_oracle": bss(yva, shift_to_prior(p, yva.mean()))}


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    season_rate = df.groupby("season")[TARGET_COL].mean().to_dict()
    ALL = [2019, 2020, 2021, 2022, 2023]
    NG = ["game_type"]
    lean = dict(BASE_PARAMS, num_leaves=31, min_data_in_leaf=2000, lambda_l2=30.0)

    variants = {
        # 레퍼런스 (3차 최고 후보들)
        "ref  hl=1.0 -gt":          dict(half_life=1.0, base_by=None, drop_cols=NG),
        # init_score = 시즌 레벨 제거
        "S1  base=season, unif":    dict(half_life=None, base_by="season"),
        "S2  base=season, hl=1.5":  dict(half_life=1.5, base_by="season"),
        "S3  base=season, hl=1.0":  dict(half_life=1.0, base_by="season"),
        "S4  base=season -gt unif": dict(half_life=None, base_by="season", drop_cols=NG),
        # init_score = 시즌 x game_type 레벨 제거 (F 정의 변경까지 흡수)
        "T1  base=season_gt unif":  dict(half_life=None, base_by="season_gt"),
        "T2  base=season_gt hl=1.5": dict(half_life=1.5, base_by="season_gt"),
        "T3  base=season_gt hl=1.0": dict(half_life=1.0, base_by="season_gt"),
        "T4  base=season_gt +ids":  dict(half_life=1.5, base_by="season_gt", add_ids=True),
        "T5  base=season_gt lean":  dict(half_life=1.5, base_by="season_gt", params=lean),
        "T6  base=season_gt k=50":  dict(half_life=1.5, base_by="season_gt", k=50.0),
    }

    print(f"{'variant':<28s} {'iter':>5s} {'BSS raw':>9s} {'BSS R4':>9s} {'BSS oracle':>11s}")
    print("-" * 68)
    for name, cfg in variants.items():
        cfg.setdefault("train_seasons", ALL)
        r = evaluate_init(df, season_rate, 2024, **cfg)
        print(f"{name:<28s} {r['iter']:>5d} {r['bss_raw']:>9.1f} {r['bss_r4']:>9.1f} "
              f"{r['bss_oracle']:>11.1f}   ({time.time()-t0:.0f}s)")

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
