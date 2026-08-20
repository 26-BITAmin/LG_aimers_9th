"""F 전문가 모델 — "2022 이전 F는 버리고 2023+만" (사용자 제안, TabPFN 없는 대조군)

질문
    F 행을 전역 모델에 맡기지 말고 **깨끗한 F(2023+)만으로 별도 모델**을 세우면
    F 행 성능이 오르는가. 오르면 TabPFN(소표본 특화)을 얹을 값어치가 있다.

측정 가능한 fold가 **하나뿐**이다 — 구조적 제약
    train<2024 → 깨끗한 F = 2023뿐 (25,686행)   ✅
    train<2023 → 깨끗한 F = 없음                ❌ fold2023은 이 실험이 정의되지 않는다
    따라서 CLAUDE.md §2의 "2 fold 부호 일치"를 쓸 수 없다. 대신 **영가설 팔**로
    증거를 세운다 — "최근 25k행 전문가가 자기 서브그룹에서 전역모델을 이긴다"가
    F 고유 현상인지, 아무 서브그룹에서나 되는 일반 현상인지 (R 전문가 대조).

팔 (fold2024, 검증 = 2024 F 30,010행, 3시드)
    A  전역 배포 모델 (전체 + soft)                    ← 현행. 기준선
    B  F전문가 · 2023 F만              25,686행       ← 사용자 제안
    C  F전문가 · 2019–2023 F 전부     132,994행       ← 더러운 것까지
    C2 F전문가 · 전부에서 25,686 서브샘플             ← N 통제. Δ(B−C2)가 순수 "버리기" 효과
    D  A + B 로짓 블렌드 (a 스윕)
    N  [영가설] R전문가 · 2023 R에서 25,686행 → R행 평가

n_rounds는 학습셋 **내부** 시간 분할(앞 80% / 뒤 20%)로 early stopping해서 정한다.
검증 fold를 보고 고르지 않는다.

실행:  .venv/Scripts/python.exe -u src/fspec.py 2>&1 | tee experiments/fspec.log
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

import soften as soften_mod
from config import (CONTAM_AXES, DROP_FEATURES, FINAL_LGB_PARAMS, N_ROUNDS,
                    RECENCY_HALF_LIFE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import bss, shift_to_prior

SEEDS = (42, 7, 2024)
FOLD = 2024

# 소표본(2.5만행)용 파라미터. 배포값(min_data_in_leaf=2000, 122만행 기준)을
# 그대로 쓰면 전문가 팔을 핸디캡 씌우는 것이다 — 표본당 잎 크기를 맞춘다.
SMALL = dict(objective="binary", learning_rate=0.03, num_leaves=15,
             min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7,
             bagging_freq=1, lambda_l2=10.0, max_bin=127, num_threads=6,
             verbosity=-1)
MAX_ROUNDS = 3000
BLEND_A = (0.0, 0.15, 0.30, 0.50, 0.75, 1.0)


def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_specialist(Xd, yd, month, seed):
    """학습셋 내부 시간분할로 라운드 수를 정하고 전량 재학습."""
    cut = np.quantile(month, 0.8)
    itr, iva = month <= cut, month > cut
    if iva.sum() < 500 or len(np.unique(yd[iva])) < 2:
        itr, iva = np.ones(len(yd), bool), np.zeros(len(yd), bool)
    prm = dict(SMALL, seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
               data_random_seed=seed)
    if iva.any():
        b = lgb.train(prm, lgb.Dataset(Xd[itr], yd[itr]), num_boost_round=MAX_ROUNDS,
                      valid_sets=[lgb.Dataset(Xd[iva], yd[iva])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        best = max(50, b.best_iteration)
    else:
        best = 400
    return lgb.train(prm, lgb.Dataset(Xd, yd), num_boost_round=best), best


def main():
    df = load_csv(TRAIN_CSV)
    tr = df[df.season < FOLD]
    va = df[df.season == FOLD]
    ref = int(tr.season.max())
    w_all = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=ref)
    art = fit_artifacts(tr, weights=w_all, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    gtr = tr["game_type"].to_numpy().astype(str)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    vaF = va["game_type"].to_numpy().astype(str) == "F"
    yF, yR = yva[vaF], yva[~vaF]
    XvaN = X(va).to_numpy(dtype=np.float32)
    print(f"검증 2024: F {vaF.sum():,}행 (양성률 {yF.mean():.4f}) / R {(~vaF).sum():,}행 ({yR.mean():.4f})")

    # ── A. 전역 배포 모델 ────────────────────────────────────
    t0 = time.time()
    Xa, ya, wa, _, _ = soften_mod.soften(tr, X(tr), tr[TARGET_COL].to_numpy(dtype=np.float64),
                                         w_all, None)
    pA = []
    for s in SEEDS:
        prm = dict(FINAL_LGB_PARAMS, seed=s, bagging_seed=s,
                   feature_fraction_seed=s, data_random_seed=s)
        b = lgb.train(prm, lgb.Dataset(Xa, ya, weight=wa), num_boost_round=N_ROUNDS)
        pA.append(1 / (1 + np.exp(-b.predict(XvaN, raw_score=True))))
    pA = np.array(pA)
    del Xa
    print(f"  A 전역 배포        F_self {sub(yF, pA[:, vaF]):>7.1f}   "
          f"R_self {sub(yR, pA[:, ~vaF]):>7.1f}   ({time.time()-t0:.0f}s)")

    # ── 전문가 팔들 ─────────────────────────────────────────
    mF23 = (gtr == "F") & (tr.season.to_numpy() == 2023)
    mFall = gtr == "F"
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(np.flatnonzero(mFall), size=int(mF23.sum()), replace=False)
    mFsub = np.zeros(len(tr), bool); mFsub[sub_idx] = True
    mR23 = (gtr == "R") & (tr.season.to_numpy() == 2023)
    r_idx = rng.choice(np.flatnonzero(mR23), size=int(mF23.sum()), replace=False)
    mRsub = np.zeros(len(tr), bool); mRsub[r_idx] = True

    Xtr_np = X(tr).to_numpy(dtype=np.float32)
    ytr_np = tr[TARGET_COL].to_numpy(dtype=np.float64)
    month = tr["game_month"].to_numpy(dtype=np.float64) + 12 * (tr.season.to_numpy() - 2019)

    specs = [("B  F전문가 2023만", mF23, vaF, yF),
             ("C  F전문가 2019-23", mFall, vaF, yF),
             ("C2 F전문가 서브샘플", mFsub, vaF, yF),
             ("N  [영가설] R전문가", mRsub, ~vaF, yR)]
    P = {}
    for name, m, vmask, yv in specs:
        t0 = time.time()
        ps, iters = [], []
        for s in SEEDS:
            b, it = fit_specialist(Xtr_np[m], ytr_np[m], month[m], s)
            iters.append(it)
            ps.append(1 / (1 + np.exp(-b.predict(XvaN, raw_score=True))))
        P[name] = np.array(ps)
        base = sub(yv, pA[:, vmask])
        print(f"  {name:<18} {'F_self' if vmask is vaF else 'R_self'} "
              f"{sub(yv, P[name][:, vmask]):>7.1f}  (전역 {base:>7.1f}, "
              f"Δ {sub(yv, P[name][:, vmask]) - base:>+7.1f})  "
              f"n={int(m.sum()):,} iter={iters} ({time.time()-t0:.0f}s)")

    # ── D. 블렌드 스윕 (선택은 하지 않는다 — 곡선만) ────────
    print("\n  블렌드 A+B (F행 F_self) — a=0이 전역, a=1이 전문가 단독")
    for a in BLEND_A:
        pb = 1 / (1 + np.exp(-((1 - a) * logit(pA[:, vaF]) + a * logit(P["B  F전문가 2023만"][:, vaF]))))
        print(f"    a={a:<5} {sub(yF, pb):>8.1f}")

    print("\n  쌍체 Δ (시드별)")
    for hi, lo, tag in [("B  F전문가 2023만", "C  F전문가 2019-23", "B−C  더러운 F까지 넣기"),
                        ("B  F전문가 2023만", "C2 F전문가 서브샘플", "B−C2 N통제 '2022이전 버리기'")]:
        d = np.array([sub(yF, P[hi][i:i+1, vaF]) - sub(yF, P[lo][i:i+1, vaF])
                      for i in range(len(SEEDS))])
        print(f"    {tag:<32} {d.mean():>+8.1f} ± {d.std(ddof=1)/np.sqrt(len(d)):.1f}")


def sub(y, p):
    """subBSS — 기준선은 그 부분집합 자신의 평균. p는 (seeds, n) 또는 (1, n)."""
    p = np.atleast_2d(p)
    return float(np.mean([bss(y, shift_to_prior(row, y.mean())) for row in p]))


if __name__ == "__main__":
    main()
