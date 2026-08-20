"""학습셋에서 F 행을 통째로 빼면 **R 행 자체의 성능**이 어떻게 되나 (사용자 질문)

질문
    "F 빼고 R만 학습시킨 다음에 holdout 보면 R 자체만 봤을 때 성능이 어떻게 나오지"

기존 기록의 한계
    §7.70 softlabel.py의 `drop` 팔은 **2019–22 F만** 뺐다(오염 구간). 2023–24의
    깨끗한 F는 남아 있었다. "F 전체"를 뺀 적은 없고, R행 subBSS도 전체 오라클
    시프트 후 값이라 레벨 교란이 섞여 있다.

조용한 함정 — 반드시 두 팔로 쪼갠다
    features.fit_artifacts → contam_fit.fit_contam 은 q_s = (R 반대율 − F 반대율)을
    **학습 df에서** 추정한다. F를 빼면 mF.sum()==0 이라 q={} 이고 조회 배열이 비어
    contam.adjust 가 전 행 0을 돌려준다 → §7.49 오염 보정(LB +10.48)이 통째로 증발.
    그래서 "R만 학습"의 손해가 F 제거 때문인지 보정표 소실 때문인지 갈라야 한다.

팔 (fold당 3시드, 단일 LGBM — §7.70·§7.71과 같은 스크리닝 도구)
    A  전체+soft (배포)        모든 행, soft label, art=전체
    B1 R만 · art도 R만         F 행 제거, artifacts도 R만으로 적합 → contam 소실
    B2 R만 · art는 전체        F 행 제거, artifacts는 전체 train으로 적합 (보정 유지)
    C  랜덤매칭 드랍+soft      시즌별 F와 **같은 수의 R 행**을 랜덤 제거. 표본수 영가설

    Δ(B2−A) = 사용자가 물은 총효과      Δ(C−A) = 순수 표본수 비용
    Δ(B2−C) = 표본수를 통제한 F 고유 효과

평가 — R행을 두 가지 시프트로 본다
    R_self : R 행만 뽑아 **R 자신의 prior**로 오라클 시프트 → 구조/판별력만
    R_glob : 검증셋 전체 prior로 시프트한 뒤 R 부분집합 (§7.70·71이 쓴 정의)
    R만 학습하면 레벨이 R 쪽으로 쏠리므로 두 값이 갈릴 수 있다. 둘 다 본다.

실행:  .venv/Scripts/python.exe -u src/gtdrop.py 2>&1 | tee experiments/gtdrop.log
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

FOLDS = [2024, 2023]
SEEDS = (42, 7, 2024)
DROP_SEED = 42                 # 랜덤매칭 드랍 마스크 (팔 C) — 시드 고정

ARMS = ["A 전체+soft (배포)", "B1 R만·art도R", "B2 R만·art전체", "C 랜덤매칭+soft"]
KEYS = ["ALL", "R_self", "R_glob", "F_self", "F_glob", "R_lvl"]


def make_X(art):
    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])
    return X


def matched_keep(tr, seed):
    """시즌별로 F 행 수와 **같은 수의 R 행**을 무작위 제거하는 마스크."""
    rng = np.random.default_rng(seed)
    s = tr["season"].to_numpy()
    g = tr["game_type"].to_numpy().astype(str)
    keep = np.ones(len(tr), dtype=bool)
    for yr in np.unique(s):
        n_f = int(((s == yr) & (g == "F")).sum())
        idx_r = np.flatnonzero((s == yr) & (g == "R"))
        if n_f == 0 or n_f > len(idx_r):
            continue
        keep[rng.choice(idx_r, size=n_f, replace=False)] = False
    return keep


def metrics(yva, p, is_f):
    """한 시드의 예측에서 6개 숫자. subBSS 기준선은 각 부분집합 자신의 평균."""
    p_all = shift_to_prior(p, yva.mean())
    yR, yF = yva[~is_f], yva[is_f]
    return np.array([
        bss(yva, p_all),
        bss(yR, shift_to_prior(p[~is_f], yR.mean())),      # R_self
        bss(yR, p_all[~is_f]),                             # R_glob
        bss(yF, shift_to_prior(p[is_f], yF.mean())),       # F_self
        bss(yF, p_all[is_f]),                              # F_glob
        float(p[~is_f].mean() - yR.mean()),                # R_lvl (시프트 전 레벨오차)
    ])


def run(Xtr, ytr, w, Xva, yva, is_f):
    out = []
    for s in SEEDS:
        prm = dict(FINAL_LGB_PARAMS, seed=s, bagging_seed=s,
                   feature_fraction_seed=s, data_random_seed=s)
        bst = lgb.train(prm, lgb.Dataset(Xtr, ytr, weight=w),
                        num_boost_round=N_ROUNDS)
        p = 1 / (1 + np.exp(-bst.predict(Xva, raw_score=True)))
        out.append(metrics(yva, p, is_f))
    return np.array(out)                                   # (seeds, 6)


def main():
    df = load_csv(TRAIN_CSV)
    t0, res = time.time(), {}

    for fold in FOLDS:
        print(f"\n=== fold {fold} (train < {fold}) ===")
        tr = df[df.season < fold]
        va = df[df.season == fold]
        ref = int(tr.season.max())
        yva = va[TARGET_COL].to_numpy(dtype=np.float64)
        is_f = va["game_type"].to_numpy().astype(str) == "F"
        print(f"  valid: {len(va):,}행 (F {is_f.sum():,} = {is_f.mean():.1%}), "
              f"R 양성률 {yva[~is_f].mean():.4f} / F {yva[is_f].mean():.4f}")

        w_all = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=ref)
        art_full = fit_artifacts(tr, weights=w_all, k_pitcher=SMOOTH_K_PITCHER,
                                 k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)
        X_full = make_X(art_full)

        is_r_tr = tr["game_type"].to_numpy().astype(str) == "R"
        trR = tr[is_r_tr]
        wR = recency_weight(trR.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=ref)
        art_R = fit_artifacts(trR, weights=wR, k_pitcher=SMOOTH_K_PITCHER,
                              k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)
        X_R = make_X(art_R)
        print(f"  train: 전체 {len(tr):,} → R만 {len(trR):,} ({1-len(trR)/len(tr):.1%} 제거)"
              f"   art_R의 contam q = {art_R['contam']['q']}  "
              f"(전체 q = { {k: round(v,4) for k,v in art_full['contam']['q'].items()} })")

        keep_c = matched_keep(tr, DROP_SEED)
        trC = tr[keep_c]
        wC = recency_weight(trC.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=ref)
        assert len(trC) == len(trR), (len(trC), len(trR))

        y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
        y_trR = trR[TARGET_COL].to_numpy(dtype=np.float64)
        y_trC = trC[TARGET_COL].to_numpy(dtype=np.float64)

        # ── A. 배포 (전체 + soft) ──────────────────────────────
        Xa, ya, wa, _, h = soften_mod.soften(tr, X_full(tr), y_tr, w_all, None)
        res[(ARMS[0], fold)] = run(Xa, ya, wa, X_full(va), yva, is_f)
        del Xa
        print(f"  {ARMS[0]:<18} {fmt(res[(ARMS[0], fold)])}  ({time.time()-t0:.0f}s)")

        # ── B1. R만, artifacts도 R만 (contam 소실) ─────────────
        res[(ARMS[1], fold)] = run(X_R(trR), y_trR, wR, X_R(va), yva, is_f)
        print(f"  {ARMS[1]:<18} {fmt(res[(ARMS[1], fold)])}  ({time.time()-t0:.0f}s)")

        # ── B2. R만, artifacts는 전체 train (보정 유지) ────────
        res[(ARMS[2], fold)] = run(X_full(trR), y_trR, wR, X_full(va), yva, is_f)
        print(f"  {ARMS[2]:<18} {fmt(res[(ARMS[2], fold)])}  ({time.time()-t0:.0f}s)")

        # ── C. 랜덤매칭 드랍 + soft (표본수 영가설) ────────────
        Xc, yc, wc, _, _ = soften_mod.soften(trC, X_full(trC), y_trC, wC, None)
        res[(ARMS[3], fold)] = run(Xc, yc, wc, X_full(va), yva, is_f)
        del Xc
        print(f"  {ARMS[3]:<18} {fmt(res[(ARMS[3], fold)])}  ({time.time()-t0:.0f}s)")

        del tr, va, trR, trC, art_full, art_R

    print("\n" + "=" * 96)
    print("요약 (단일 LGBM 3시드 평균)")
    print("=" * 96)
    for fold in FOLDS:
        print(f"\n  fold {fold}")
        print(f"    {'팔':<20}{'전체':>9}{'R_self':>9}{'R_glob':>9}"
              f"{'F_self':>10}{'F_glob':>10}{'R레벨오차':>11}")
        for a in ARMS:
            r = res[(a, fold)].mean(axis=0)
            print(f"    {a:<20}{r[0]:>9.1f}{r[1]:>9.1f}{r[2]:>9.1f}"
                  f"{r[3]:>10.1f}{r[4]:>10.1f}{r[5]:>+11.4f}")

    print("\n" + "=" * 96)
    print("쌍체 Δ (같은 시드끼리, 양수 = 그 팔이 이득)")
    print("=" * 96)
    pairs = [(ARMS[2], ARMS[0], "B2−A  R만 vs 배포        (총효과)"),
             (ARMS[3], ARMS[0], "C −A  랜덤매칭 vs 배포   (표본수 비용)"),
             (ARMS[2], ARMS[3], "B2−C  R만 vs 랜덤매칭    (F 고유효과)"),
             (ARMS[1], ARMS[2], "B1−B2 art R만 vs art전체 (contam 소실분)")]
    for hi, lo, tag in pairs:
        print(f"\n  {tag}")
        for fold in FOLDS:
            d = res[(hi, fold)] - res[(lo, fold)]
            m, se = d.mean(axis=0), d.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
            print(f"    f{fold}  전체 {m[0]:>+8.1f}±{se[0]:.1f}"
                  f"   R_self {m[1]:>+8.1f}±{se[1]:.1f}"
                  f"   R_glob {m[2]:>+8.1f}±{se[2]:.1f}"
                  f"   F_self {m[3]:>+9.1f}   F_glob {m[4]:>+9.1f}")
        if not all(f in FOLDS for f in (2023, 2024)):
            continue
        d24 = (res[(hi, 2024)] - res[(lo, 2024)]).mean(axis=0)
        d23 = (res[(hi, 2023)] - res[(lo, 2023)]).mean(axis=0)
        for j, nm in ((0, "전체"), (1, "R_self")):
            same = (d24[j] > 0) == (d23[j] > 0)
            ratio = max(abs(d24[j]), abs(d23[j])) / max(min(abs(d24[j]), abs(d23[j])), 1e-9)
            print(f"      {nm:<7} 부호일치 {'O' if same else 'X'}  크기비 {ratio:.1f}배")


def fmt(r):
    m = r.mean(axis=0)
    s = r.std(axis=0, ddof=1)
    return (f"전체 {m[0]:>7.1f}±{s[0]:>4.1f}  R_self {m[1]:>7.1f}  R_glob {m[2]:>7.1f}"
            f"  F_self {m[3]:>8.1f}  F_glob {m[4]:>8.1f}  R레벨 {m[5]:>+.4f}")


if __name__ == "__main__":
    main()
