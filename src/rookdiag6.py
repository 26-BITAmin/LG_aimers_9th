"""§7.167 후속 — 편향의 `asof_pitcher_n` 단조 패턴을 **리그(R/F)로 쪼갠다**.

사용자 질문: "2024 기준으로 본다면, 리그별로는 또 어떻게 나뉘는데."

무엇을 내나 (셀 = game_type × 통산 기록 수 버킷)
    n · 행 비중 · ȳ · p̄ · 편향(p̄−ȳ) · 투수 클러스터 부트스트랩 95% CI
    편향손실 = w · 1e5 · 편향² / ref_all
        └ 그 셀의 **레벨 오차만으로** 잃는 BSS. 셀 안 순위가 완벽해도 남는 손해다.
          전부 더해도 전체 손실이 되지는 않는다 (셀 간 상쇄가 있다) —
          "이 셀을 레벨만 고치면 얼마인가"를 읽는 열이다.
    로컬BSS = 그 셀 안에서만 채점한 점수 (레벨 말고 순위 능력)

⚠️ F 는 행의 11.8% 뿐이고 저표본 칸은 더 작다. 셀이 작으면 CI 가 답을 준다.

실행:  .venv/Scripts/python.exe -u src/rookdiag6.py 2>&1 | tee experiments/rookdiag6.log
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

import rookdiag as R
from config import TRAIN_CSV
from data import load_csv
from validation import brier

B = 2000
RNG = np.random.default_rng(3)
EDGES = [1, 100, 500, 2000, 10000]
LABS = ["0", "1~99", "100~499", "500~1,999", "2,000~9,999", "10,000+"]
MIN = 150


def boot_bias(cl, n_id, y, p, m):
    """셀 편향의 투수 클러스터 부트스트랩 CI."""
    n = np.bincount(cl, weights=m, minlength=n_id)
    sd = np.bincount(cl, weights=m * (p - y), minlength=n_id)
    out = np.empty(B)
    for b in range(B):
        k = RNG.integers(0, n_id, n_id)
        a = n[k].sum()
        out[b] = sd[k].sum() / a if a > 0 else np.nan
    o = out[~np.isnan(out)]
    return np.percentile(o, [2.5, 97.5])


def block(title, y, p, cl, n_id, sel, bkt, ref_all, ntot):
    print(f"\n  {title}")
    print(f"    {'통산 기록':<14}{'n':>9}{'전체比':>8}{'ȳ':>8}{'p̄':>8}"
          f"{'편향':>9}{'95% CI':>22}{'편향손실':>10}{'로컬BSS':>10}")
    tot = 0.0
    for i, lab in enumerate(LABS):
        m = sel & (bkt == i)
        if m.sum() == 0:
            continue
        w = m.sum() / ntot
        bias = (p[m] - y[m]).mean()
        loss = w * 1e5 * bias * bias / ref_all
        tot += loss
        yg = y[m].mean()
        loc = 1e5 * (1 - brier(y[m], p[m]) / (yg * (1 - yg))) if 0 < yg < 1 else float("nan")
        ci = boot_bias(cl, n_id, y, p, m) if m.sum() >= MIN else (np.nan, np.nan)
        cis = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if m.sum() >= MIN else "표본 부족"
        print(f"    {lab:<14}{m.sum():>9,}{w*100:>7.1f}%{yg:>8.4f}{p[m].mean():>8.4f}"
              f"{bias:>+9.4f}{cis:>22}{loss:>10.2f}{loc:>10.1f}")
    b_all = (p[sel] - y[sel]).mean()
    w_all = sel.sum() / ntot
    print(f"    {'— 합계 —':<14}{sel.sum():>9,}{w_all*100:>7.1f}%"
          f"{y[sel].mean():>8.4f}{p[sel].mean():>8.4f}{b_all:>+9.4f}"
          f"{'':>22}{tot:>10.2f}")
    print(f"      (리그 전체를 한 덩어리로 보면 편향손실은"
          f" {w_all*1e5*b_all*b_all/ref_all:.2f} — 칸으로 쪼개면 {tot:.2f}."
          f" 차이가 칸 사이에서 상쇄되던 몫이다)")


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    for fold in (2024, 2023):
        va, y, p, _ = R.build(fold, df, s)
        an = va["asof_pitcher_n"].to_numpy(dtype=np.float64)
        isF = va["game_type"].to_numpy().astype(str) == "F"
        pp = R.prior_seasons(df, s, fold, "pitcher_id")
        cl, uniq = pd.factorize(va["pitcher_id"].to_numpy())
        n_id = len(uniq)
        bkt = np.where(np.isnan(an), -1, np.digitize(an, EDGES))
        ref_all = y.mean() * (1 - y.mean())
        ntot = len(y)
        tag = "★ 주 판단" if fold == 2024 else "참고 (2019~22 F 라벨 오염 시즌을 학습에 씀)"
        print("\n" + "=" * 104)
        print(f"fold {fold}   n={ntot:,}   ȳ={y.mean():.4f}   F행 {isF.mean()*100:.1f}%"
              f"   — {tag}")
        print("=" * 104)
        block("R (정규시즌)", y, p, cl, n_id, ~isF, bkt, ref_all, ntot)
        block("F (그 외)", y, p, cl, n_id, isF, bkt, ref_all, ntot)

        print(f"\n  같은 기록 구간에서 R 과 F 의 편향 차 (F − R)")
        print(f"    {'통산 기록':<14}{'R 편향':>10}{'F 편향':>10}{'차':>10}{'95% CI':>24}")
        for i, lab in enumerate(LABS):
            m1, m2 = isF & (bkt == i), (~isF) & (bkt == i)
            if m1.sum() < MIN or m2.sum() < MIN:
                continue
            mu, ci, t = R.cl_boot(cl, n_id, y, p, m1, m2, "bias")
            print(f"    {lab:<14}{(p[m2]-y[m2]).mean():>+10.4f}"
                  f"{(p[m1]-y[m1]).mean():>+10.4f}{mu:>+10.4f}"
                  f"{f'[{ci[0]:+.4f}, {ci[1]:+.4f}]':>24}")

        print(f"\n  참고 — 신인/2+ 를 리그별로 (편향)")
        print(f"    {'':<14}{'R 신인':>11}{'R 2+':>11}{'F 신인':>11}{'F 2+':>11}")
        cells = [(~isF) & (pp == 0), (~isF) & (pp >= 2), isF & (pp == 0), isF & (pp >= 2)]
        print(f"    {'n':<14}" + "".join(f"{m.sum():>11,}" for m in cells))
        print(f"    {'편향':<14}" + "".join(f"{(p[m]-y[m]).mean():>+11.4f}" for m in cells))
        print(f"    {'중앙 asof_n':<14}"
              + "".join(f"{np.nanmedian(an[m]):>11,.0f}" for m in cells))


if __name__ == "__main__":
    main()
