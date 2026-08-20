"""§7.81 구현 가능판 — 오라클 z̄_j·z̄_b를 **행 독립 대체물**로 갈아끼운다.

오라클(threespeed.py)은 검증 fold 행에서 투수 평균을 계산했다. 규정상 추론에선
불가. 대체물 둘:

    M_p(row)  상황·타자 피처를 뺀 **투수 전용 모델**의 행별 출력.
              그 행 자신의 asof만 보므로 행 독립이고, asof 누적을 타고
              "느린 성분"을 재현한다.
    L_j       train에서 동결한 투수별 M_p 평균 (미지 투수는 train 전역 평균).
              기존 asof/ctx 아티팩트와 같은 성격의 조회 — 규정 문제 없음.

    z' = L_all + b·(L_j − L_all) + a·(M_p − L_j) + s·(z_full − M_p)

오라클 대비 얼마나 남는지가 이 채널의 진짜 크기다.

실행:  LGA_FOLD_KEY=<지문> LGA_MODEL_DIR=submit/model_pbase \
       .venv/Scripts/python.exe -u src/pbase.py 2>&1 | tee experiments/pbase.log
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import lightgbm as lgb
import numpy as np

import foldcache
import modelabel
import soften
from clockaxis import gmean, qbucket
from config import (FINAL_LGB_PARAMS, MLP_ALPHA, N_ROUNDS, SOFT_LABEL_F, TRAIN_CSV)
from data import load_csv
from rookie import FOLDS, FOLD_KEY, deployed_pred
from validation import _logit, bss, shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
BG = [1.0, 1.1, 1.2, 1.3]
AG = [1.0, 1.2, 1.4, 1.6, 1.8]
SG = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# 투수 안에서 상수이거나 **천천히만** 변하는 컬럼 = 투수 전용.
# 타자·상황에 따라 공마다 바뀌는 것은 전부 제외한다 (그게 '빠른 성분'이다).
PITCHER_ONLY = [
    "pitcher_id", "pitcher_team_id", "pitcher_hand",
    "asof_pitcher_n", "asof_pitcher_pitchmix_n",
    "miss_pitcher_asof", "miss_prev_game",
    "asof_pitcher_success_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "form_p1_minus_career", "form_p3_minus_career", "form_p1_minus_p5",
    "form_p3_minus_p5", "form_mid_p1_minus_career",
    "fail_middle_share", "pitcher_ball_minus_strike",
    "pitchmix_entropy", "pitchmix_nonfastball",
]


def pitcher_cols(X):
    """이름 목록 + `_sm` 쌍. 실제 컬럼과 대조해 감사 로그를 남긴다."""
    want = set(PITCHER_ONLY) | {c + "_sm" for c in PITCHER_ONLY}
    keep = [c for c in X.columns if c in want]
    drop = [c for c in X.columns if c not in want]
    missing = sorted(want - set(X.columns) - {c + "_sm" for c in PITCHER_ONLY})
    return keep, drop, missing


def train_mp(fold, Xtr_o, Xtr, ytr, w, Xva, cols):
    """투수 전용 LGBM 3시드. 검증행과 **원본 학습행** 양쪽에 예측한다
    (L_j는 학습행 평균이므로 확장 전 원본에서 내야 한다)."""
    p = CACHE / f"pbase_f{fold}_{FOLD_KEY}_{len(cols)}.npz"
    if p.exists():
        d = np.load(p)
        print(f"  M_p 재사용 ({p.name})")
        return d["va"], d["tr"]
    zv = np.zeros(len(Xva)); zt = np.zeros(len(Xtr_o)); n = 0
    for s in foldcache.LGB_SEEDS:
        prm = dict(FINAL_LGB_PARAMS, seed=s, bagging_seed=s,
                   feature_fraction_seed=s, data_random_seed=s)
        bst = lgb.train(prm, lgb.Dataset(Xtr[cols], ytr, weight=w),
                        num_boost_round=N_ROUNDS)
        zv += bst.predict(Xva[cols], raw_score=True)
        zt += bst.predict(Xtr_o[cols], raw_score=True)
        n += 1
        del bst
    zv /= n; zt /= n
    np.savez_compressed(p, va=zv, tr=zt)
    return zv, zt


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    print(f"foldcache 지문: {foldcache._key()}")
    R = {}

    for fold in FOLDS:
        va, y, p = deployed_pred(df, s, fold, cached)
        z = _logit(p)
        base = bss(y, p)

        Xtr_o, ytr0, w0, Xva, yva, art, tr = foldcache.fold_data(df, fold)
        assert np.array_equal(y, yva)
        cols, dropped, missing = pitcher_cols(Xtr_o)
        if fold == FOLDS[0]:
            print(f"\n투수 전용 {len(cols)}개 / 제외 {len(dropped)}개")
            print(f"  제외: {', '.join(dropped)}")
            if missing:
                print(f"  ⚠️ 목록에 있으나 피처에 없음: {missing}")

        aux, _ = modelabel.aux_labels(tr)
        Xtr, ytr, w = Xtr_o, ytr0, w0
        if SOFT_LABEL_F:
            Xtr, ytr, w, aux, _ = soften.soften(tr, Xtr_o, ytr0, w0, aux)
        mp_va, mp_tr = train_mp(fold, Xtr_o, Xtr, ytr, w, Xva, cols)
        print(f"  M_p 단독 BSS {bss(y, shift_to_prior(1/(1+np.exp(-mp_va)), y.mean())):.1f}"
              f"   ({time.time()-t0:.0f}s)")

        # L_j — train 행에서 동결. 미지 투수는 train 전역 평균.
        tp = tr["pitcher_id"].to_numpy()
        uid, inv = np.unique(tp, return_inverse=True)
        Lmap = np.bincount(inv, weights=mp_tr) / np.bincount(inv)
        L_all = float(mp_tr.mean())
        vp = va["pitcher_id"].to_numpy()
        idx = np.searchsorted(uid, vp)
        seen = (idx < len(uid)) & (uid[np.clip(idx, 0, len(uid)-1)] == vp)
        L_j = np.where(seen, Lmap[np.clip(idx, 0, len(uid)-1)], L_all)

        # 오라클 대비 — 대체물이 얼마나 닮았나
        _, cl = np.unique(vp, return_inverse=True)
        zj = gmean(cl, z)[cl]
        g = cl.astype(np.int64) * 1000 + qbucket(cl, va["asof_pitcher_n"].to_numpy(float), 5)
        _, g = np.unique(g, return_inverse=True)
        zb = gmean(g, z)[g]
        print(f"  대체물 상관:  L_j~z̄_j {np.corrcoef(L_j, zj)[0,1]:.3f}"
              f"   M_p~z̄_b {np.corrcoef(mp_va, zb)[0,1]:.3f}"
              f"   미지 투수 행 {100*(1-seen.mean()):.1f}%")

        grid = {}
        for b in BG:
            for a in AG:
                for sc in SG:
                    zz = L_all + b*(L_j-L_all) + a*(mp_va-L_j) + sc*(z-mp_va)
                    grid[(b, a, sc)] = bss(y, shift_to_prior(
                        1/(1+np.exp(-zz)), y.mean())) - base
        k = max(grid, key=grid.get)
        print(f"  fold {fold} 기준 {base:.1f}   최적 b={k[0]} a={k[1]} s={k[2]}  Δ {grid[k]:+.1f}")
        R[fold] = grid

    print("\n" + "=" * 66)
    com = max(R[FOLDS[0]], key=lambda k: min(R[f][k] for f in FOLDS))
    v = [R[f][com] for f in FOLDS]
    print(f"두 fold 최소값 최대화: b={com[0]} a={com[1]} s={com[2]}   " +
          "   ".join(f"fold{f} {R[f][com]:+.1f}" for f in FOLDS) +
          f"   (크기비 {max(v)/max(min(v),1e-9):.2f})")
    for src in FOLDS:
        k = max(R[src], key=R[src].get)
        for dst in FOLDS:
            if dst != src:
                print(f"  교차: fold{src} 최적 (b={k[0]} a={k[1]} s={k[2]})"
                      f" → fold{dst} {R[dst][k]:+.1f}  (자체 {R[src][k]:+.1f})")


if __name__ == "__main__":
    main()
