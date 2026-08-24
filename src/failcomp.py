"""(1) 주자 컬럼이 배포 트리에서 실제로 쓰이는가  (2) 후보 B '실패 유형 구성비' 실증.

사용자 질문 둘:
  Q1  runner_on_2b/3b 는 효과가 입증되어 들어가 있는 것인가?
  Q2  후보 B — (middle, ball, reverse) → CLR 좌표 2개.

Q2 의 전제부터 센다. 문서는 세 rate 를 **실패의 3분할 조성**으로 가정한다.
그런데 data_description 상 ball_rate 는 "볼성 **결과**" 이고 strike_rate 와 짝이다.
실패 정의 ②(존에서 크게 벗어남)와 같은 물건인지 불명 — 합이 1 이 되는지로 판정한다.
조성이 아니면 CLR 은 정당한 좌표계가 아니다.

그리고 `fail_middle_share` = middle/(middle+reverse) 는 **이미 배포 피처다**
(features.py L860). §7.11 그룹 LOO 에서 '실패 구성' 2개 제거 = +4.7/+0.4 (크기비 13.2).

측정 (학습 0회. 배포 잔차에 묻는다 — 새 피처는 잔차에 먼저):
  P1  배포 LGBM gain importance 에서 주자 5컬럼의 점유율·순위
  P2  문서의 상관 주장 검증 + 조성 여부(합=1?) 검정
  P3  CLR 좌표의 신규성 — 기존 배포 피처 전체에 대한 선형 R²
  P4  배포 잔차 CR0 회귀 — 후보 vs 대조군(이미 배포된 share / 죽은 축 요일)

실행:  .venv/Scripts/python.exe -u src/failcomp.py 2>&1 | tee experiments/failcomp.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import lightgbm as lgb

import deploy29 as D
from tmslot import resid_t_cr0

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "submit" / "model"
RUNNER_COLS = ["runner_on_1b", "runner_on_2b", "runner_on_3b",
               "num_runners_on", "base_state"]


def p1_importance():
    print("=" * 96)
    print("P1. 배포 LGBM gain importance — 주자 5컬럼이 실제로 쓰이는가")
    print("=" * 96)
    tot, names = None, None
    files = sorted(MODEL_DIR.glob("lgbm_*.txt"))
    for f in files:
        # 한글 경로 → model_file 금지 (CP949 오해). model_str 로 우회.
        bst = lgb.Booster(model_str=f.read_text(encoding="utf-8"))
        g = np.asarray(bst.feature_importance("gain"), dtype=np.float64)
        if tot is None:
            tot, names = np.zeros_like(g), bst.feature_name()
        tot += g / g.sum()
    tot /= len(files)
    order = np.argsort(-tot)
    rank = {names[i]: r + 1 for r, i in enumerate(order)}
    print(f"모델 {len(files)}개 평균, 피처 {len(names)}개\n")
    print("  상위 8개:")
    for i in order[:8]:
        print(f"    {rank[names[i]]:>3d}. {names[i]:<38s}{100*tot[i]:>7.2f}%")
    print("\n  주자 계열:")
    share = 0.0
    for c in RUNNER_COLS:
        if c in rank:
            i = names.index(c)
            share += tot[i]
            print(f"    {rank[c]:>3d}. {c:<38s}{100*tot[i]:>7.2f}%")
        else:
            print(f"      —. {c:<38s}   (피처에 없음)")
    print(f"\n  주자 5컬럼 합계 gain 점유율: {100*share:>5.2f}%"
          f"   (균등 배분 기준 {100*5/len(names):.2f}%)")
    for c in ("game_month", "game_dayofweek", "fail_middle_share",
              "pitcher_ball_minus_strike"):
        if c in rank:
            print(f"  참고  {rank[c]:>3d}. {c:<36s}{100*tot[names.index(c)]:>7.2f}%")
    return set(names)


def p2_composition(df):
    print("\n" + "=" * 96)
    print("P2. 후보 B 의 전제 — (middle, ball, reverse) 는 실패의 조성인가?")
    print("=" * 96)
    c = {k: df[f"asof_pitcher_{k}_rate"].to_numpy(np.float64)
         for k in ("success", "middle", "reverse", "ball", "strike")}
    ok = ~np.isnan(c["success"])
    for k in c:
        ok &= ~np.isnan(c[k])
    print(f"  전 컬럼 비결측 행 {ok.sum():,} / {len(ok):,}")
    print("\n  합 검정 (평균 ± 표준편차):")
    combos = {
        "success + middle + reverse + ball": ["success", "middle", "reverse", "ball"],
        "success + middle + reverse":        ["success", "middle", "reverse"],
        "ball + strike":                     ["ball", "strike"],
        "middle + reverse + ball":           ["middle", "reverse", "ball"],
    }
    for label, ks in combos.items():
        v = sum(c[k][ok] for k in ks)
        print(f"    {label:<36s} {v.mean():>8.4f} ± {v.std():>6.4f}")
    print("\n  문서의 상관 주장 검증 (vs success_rate):")
    print(f"    {'컬럼':<14s}{'Pearson':>10s}{'Spearman':>10s}")
    for k in ("middle", "reverse", "ball", "strike"):
        pe = np.corrcoef(c[k][ok], c["success"][ok])[0, 1]
        sp = spearmanr(c[k][ok], c["success"][ok]).statistic
        print(f"    {k:<14s}{pe:>+10.3f}{sp:>+10.3f}")
    fms = c["middle"][ok] / np.maximum(c["middle"][ok] + c["reverse"][ok], 1e-9)
    print(f"\n  현행 fail_middle_share = middle/(middle+reverse):")
    print(f"    vs success  Pearson {np.corrcoef(fms, c['success'][ok])[0,1]:+.3f}"
          f"   Spearman {spearmanr(fms, c['success'][ok]).statistic:+.3f}"
          f"   ← 이미 '수준'이 나뉘어 있는가")
    return c, ok


def clr(parts, eps=1e-6):
    """중심 로그비. parts: (n, k) 양수 배열 → (n, k) 좌표 (합=0, 자유도 k−1)."""
    L = np.log(np.clip(parts, eps, None))
    return L - L.mean(axis=1, keepdims=True)


def main():
    names = p1_importance()

    df, s, y = D.load_df()
    p2_composition(df)

    print("\n" + "=" * 96)
    print("P3/P4. CLR 좌표의 신규성과 배포 잔차 설명력 (학습 0회)")
    print("=" * 96)
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        d = st["d"]
        m = d["asof_pitcher_middle_rate"].to_numpy(np.float64)
        r_ = d["asof_pitcher_reverse_rate"].to_numpy(np.float64)
        b = d["asof_pitcher_ball_rate"].to_numpy(np.float64)
        su = d["asof_pitcher_success_rate"].to_numpy(np.float64)
        stk = d["asof_pitcher_strike_rate"].to_numpy(np.float64)
        good = ~(np.isnan(m) | np.isnan(r_) | np.isnan(b) | np.isnan(su) | np.isnan(stk))
        fill = lambda v, dv: np.where(np.isnan(v), dv, v)
        m, r_, b = fill(m, np.nanmean(m)), fill(r_, np.nanmean(r_)), fill(b, np.nanmean(b))
        su, stk = fill(su, np.nanmean(su)), fill(stk, np.nanmean(stk))

        C = clr(np.stack([m, b, r_], axis=1))
        cand = {"clr_middle": C[:, 0], "clr_ball": C[:, 1], "clr_reverse": C[:, 2]}
        # 기존 배포 피처(원단위) — CLR 이 이들의 선형결합인지
        base_cols = np.stack([m, r_, b, su, stk,
                              m / np.maximum(m + r_, 1e-9), b - stk], axis=1)
        base_cols = np.column_stack([base_cols, np.ones(len(m))])
        print(f"\nfold {fold}   clean {len(m):,}행   기준 BSS {st['base']:.1f}"
              f"   전컬럼 관측 {100*good.mean():.1f}%")
        print(f"  {'좌표':<14s}{'기존피처 선형 R²':>18s}{'잔차 CR0 t':>14s}{'기울기·1sd':>14s}")
        rows = dict(cand)
        rows["fail_middle_share (배포중·대조)"] = m / np.maximum(m + r_, 1e-9)
        rows["요일 (죽은 축·대조)"] = d["game_dayofweek"].to_numpy(np.float64)
        for nm, v in rows.items():
            coef, *_ = np.linalg.lstsq(base_cols, v, rcond=None)
            resid = v - base_cols @ coef
            r2 = 1.0 - resid.var() / max(v.var(), 1e-12)
            vz = (v - v.mean()) / max(v.std(), 1e-12)
            sl, t = resid_t_cr0(st["r"], vz, st["cl"], st["n_id"])
            tag = "" if "대조" in nm else ("  <<<" if abs(t) > 2 else "")
            print(f"  {nm:<14s}{r2:>18.4f}{t:>+14.2f}{sl:>+14.5f}{tag}")
    print("\n판정: R²≈1 이면 기존 피처의 선형결합 — 트리에는 새 정보가 아니다.")
    print("      잔차 CR0 |t|<2 이면 배포 층이 이미 그 성분을 먹었다.")
    print("      대조군(배포중 share / 죽은 요일)이 0 근처여야 눈금이 맞는 것이다.")


if __name__ == "__main__":
    main()
