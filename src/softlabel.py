"""2019-22 F 오염 라벨의 **soft label 보정** — 0단계 대조군 실험 (라벨 채널)

배경 (§7.48·§7.49·§7.69)
    2019-22 F는 reverse 검출 실패로 실패가 성공으로 기록됐다. 피처 채널은
    §7.49가 고쳤고(LB +10.48), 행 단위 δ_F는 LB에서 기각됐다(−2.81).
    남은 채널은 **학습 타깃 자체**다: 피처를 고쳐도 라벨이 계속 거짓말하면
    모델은 고쳐진 피처의 계수를 감쇠시키는 쪽으로 타협한다
    (라벨 오염은 공유 피처를 타고 무노출 행에도 번진다 — §7.69 부검).

오염의 단측 구조
    y_obs=0 행은 무결하다 (몰림/벗어남/잡힌 reverse — 제대로 기록됨).
    y_obs=1 행만 숨은 reverse가 섞여 있고, 그 비율은
        h_s = P(숨은 reverse | y_obs=1, 시즌 s) = q_s / p̄_s
        q_s = (그 시즌 R 반대율) − (그 시즌 F 반대율)   ← contam과 같은 정의
        p̄_s = 그 시즌 F의 관측 성공률                   ← 둘 다 fold train에서만
    즉 2019-22 F의 y=1 행만 라벨을 1−h_s 로 깎는다. y=0 행은 건드리지 않는다.

구현 — binary 목적함수를 유지하는 행 복제 트릭
    라벨 p인 행 = (y=1, w·p) + (y=0, w·(1−p)) 두 행. 로스가 soft target
    cross-entropy와 수학적으로 동치이고 base 팔은 비트 하나 안 바뀐다.

팔 (전부 §7.49 피처 보정 ON — 라벨 채널의 순수 증분만 잰다)
    base   현상 유지
    soft   2019-22 F & y=1 행만 soft label 1−h_s
    drop   2019-22 F 행 전체 제거 (가장 싼 대안 — soft가 이걸 못 이기면 무의미)

중단 규칙 (사용자 지시)
    fold2024(2025와 같은 레짐)에서 soft가 base보다 낮으면 즉시 중단한다.

실행:  .venv/Scripts/python.exe -u src/softlabel.py 2>&1 | tee experiments/softlabel.log
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
import pandas as pd

from config import (DROP_FEATURES, FINAL_LGB_PARAMS, N_ROUNDS, RECENCY_HALF_LIFE,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV,
                    USE_PLAYER_IDS, CONTAM_AXES)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from modelabel import recover
from validation import bss, brier, shift_to_prior

FOLDS = [2024, 2023]          # 2024 먼저 — 2025와 같은 레짐, 중단 규칙의 판정 fold
SEEDS = (42, 7, 2024)
ARMS = ["base", "soft", "drop"]


def contam_mask(d):
    return ((d["season"].to_numpy() <= 2022)
            & (d["game_type"].to_numpy().astype(str) == "F"))


def fit_h(df, train_mask):
    """h_s = q_s / p̄_s. 전부 train 행에서만. (q_s 정의는 fcontam2.fit_q와 동일)"""
    lab, _, _ = recover(df)
    rev = np.where(np.isnan(lab["reverse"]), 0.0, np.round(lab["reverse"]))
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    s = df["season"].to_numpy()
    g = df["game_type"].to_numpy().astype(str)
    h = {}
    for yr in range(2019, 2023):
        mR = train_mask & (s == yr) & (g == "R")
        mF = train_mask & (s == yr) & (g == "F")
        if mF.sum() == 0 or mR.sum() == 0:
            continue
        q = max(0.0, float(rev[mR].mean() - rev[mF].mean()))
        pbar = float(y[mF].mean())
        h[yr] = min(q / max(pbar, 1e-9), 0.95)
    return h


def prep_fold(df, valid_season):
    """base/soft가 공유하는 것들을 한 번만 만든다."""
    tr = df[df.season < valid_season]
    va = df[df.season == valid_season]
    ref = int(tr.season.max())
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=ref)
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    return tr, va, w, X


def train_eval(Xtr, ytr, w, Xva, yva, va_f, seeds):
    """3시드 학습 → (BSS, F행 subBSS, R행 subBSS) — 오라클 시프트 후."""
    refF = yva[va_f].mean() * (1 - yva[va_f].mean())
    refR = yva[~va_f].mean() * (1 - yva[~va_f].mean())
    out = []
    for s in seeds:
        p = dict(FINAL_LGB_PARAMS, seed=s, bagging_seed=s,
                 feature_fraction_seed=s, data_random_seed=s)
        bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w),
                        num_boost_round=N_ROUNDS)
        pr = shift_to_prior(1 / (1 + np.exp(-bst.predict(Xva, raw_score=True))),
                            yva.mean())
        out.append((bss(yva, pr),
                    1e5 * (1 - brier(yva[va_f], pr[va_f]) / refF),
                    1e5 * (1 - brier(yva[~va_f], pr[~va_f]) / refR)))
    return np.array(out)          # (seeds, 3)


def run_arm(arm, tr, va, w, X, h, seeds):
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    va_f = contam_mask(va) | (va["game_type"].to_numpy().astype(str) == "F")
    cm = contam_mask(tr)

    if arm == "drop":
        # 오염 행이 없는 세계 — artifacts부터 다시 만든다
        tr2 = tr[~cm]
        w2 = recency_weight(tr2.season.to_numpy(), RECENCY_HALF_LIFE,
                            ref_season=int(tr.season.max()))
        art2 = fit_artifacts(tr2, weights=w2, k_pitcher=SMOOTH_K_PITCHER,
                             k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

        def X2(d):
            x = build_features(d, art2)
            if USE_PLAYER_IDS:
                for c in ("pitcher_id", "batter_id"):
                    x[c] = d[c].to_numpy(dtype=np.float32)
            return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

        return train_eval(X2(tr2), tr2[TARGET_COL].to_numpy(dtype=np.float64),
                          w2, X2(va), yva, va_f, seeds)

    Xtr = X(tr)
    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    if arm == "base":
        return train_eval(Xtr, ytr, w, X(va), yva, va_f, seeds)

    # soft — 행 복제 트릭
    h_row = np.zeros(len(tr))
    s = tr["season"].to_numpy()
    for yr, v in h.items():
        h_row[cm & (s == yr)] = v
    dup = cm & (ytr == 1) & (h_row > 0)
    w_soft = w.copy()
    w_soft[dup] *= (1 - h_row[dup])
    Xtr2 = pd.concat([Xtr, Xtr[dup]], ignore_index=True)
    ytr2 = np.concatenate([ytr, np.zeros(int(dup.sum()))])
    w2 = np.concatenate([w_soft, w[dup] * h_row[dup]])
    print(f"    soft: 복제 {dup.sum():,}행  h = "
          + "  ".join(f"{yr}:{v:.3f}" for yr, v in sorted(h.items())))
    return train_eval(Xtr2, ytr2, w2, X(va), yva, va_f, seeds)


def paired(res, arm, fold):
    d = res[(arm, fold)][:, 0] - res[("base", fold)][:, 0]
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def main():
    df = load_csv(TRAIN_CSV)
    t0 = time.time()
    res = {}

    for fold in FOLDS:
        print(f"\n=== fold {fold} (train < {fold}) ===")
        tr, va, w, X = prep_fold(df, fold)
        h = fit_h(df, (df.season < fold).to_numpy())
        for arm in ARMS:
            r = run_arm(arm, tr, va, w, X, h, SEEDS)
            res[(arm, fold)] = r
            print(f"  {arm:<5} BSS {r[:,0].mean():>8.1f} ± {r[:,0].std(ddof=1):>5.1f}"
                  f"   F행 {r[:,1].mean():>8.1f}  R행 {r[:,2].mean():>8.1f}"
                  f"   ({time.time()-t0:.0f}s)")

            # ── 중단 규칙: 판정 fold(2024)에서 soft가 base보다 낮으면 즉시 종료 ──
            if fold == 2024 and arm == "soft":
                dm, se = paired(res, "soft", 2024)
                print(f"  → 중단 판정: Δ(soft−base) = {dm:+.1f} ± {se:.1f} (쌍체)")
                if dm < 0:
                    print("\n" + "!" * 70)
                    print("  STOP — fold2024에서 soft가 base보다 낮다. 실험 중단.")
                    print("!" * 70)
                    return
        del tr, va, w, X

    print("\n" + "=" * 74)
    print("요약 (BSS oracle, 3시드 / 쌍체 Δ vs base)")
    print("=" * 74)
    print(f"  {'팔':<6}" + "".join(f"{f:>10d}" for f in FOLDS)
          + f"{'F행2024':>10}{'R행2024':>10}")
    for arm in ARMS:
        print(f"  {arm:<6}"
              + "".join(f"{res[(arm,f)][:,0].mean():>10.1f}" for f in FOLDS)
              + f"{res[(arm,2024)][:,1].mean():>10.1f}"
              + f"{res[(arm,2024)][:,2].mean():>10.1f}")
    for arm in ["soft", "drop"]:
        d = [paired(res, arm, f) for f in FOLDS]
        same = (d[0][0] > 0) == (d[1][0] > 0)
        ratio = (max(abs(d[0][0]), abs(d[1][0]))
                 / max(min(abs(d[0][0]), abs(d[1][0])), 1e-9))
        print(f"\n  {arm} vs base: "
              + "  ".join(f"f{f} {m:+.1f}±{se:.1f}" for (m, se), f in zip(d, FOLDS))
              + f"   부호일치 {'O' if same else 'X'}  크기비 {ratio:.1f}배")


if __name__ == "__main__":
    main()
