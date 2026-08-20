"""그림에서 본 것을 측정으로 확인 — 꼬리가 불안정하다

`02_response_centered.png`에서 눈으로 본 것:
    · 중간 빈(4~15)에서는 6시즌 곡선이 **거의 완벽하게 겹친다**
    · 양 끝 빈에서만 확 벌어진다. `asof_pitcher_success_rate` 최상위 빈은
      2021에 +0.146, 2023에 +0.000, 2024에 +0.037
    · `asof_batter_success_rate`는 **신호가 최상위 두 빈에만 있었고 최근에 사라졌다**
      (+0.158 → −0.016)
    · 빈 표본은 2,386~33,000행이라 노이즈가 아니다 (SE ≈ 0.010, 차이는 10σ 이상)

세 질문
    Q1 꼬리 불안정이 정말 꼬리에 국한되나? (중간 빈 대비 시즌간 산포)
    Q2 배포 모델이 실제로 꼬리 행에서 점수를 잃나?
    Q3 왜 꼬리가 불안정한가 — **era 오염 가설**
       `asof_*_success_rate`는 통산 누적이다. 리그 제구율이 6년간 0.08 내려갔으므로
       통산 rate가 높은 투수는 "잘하는 투수"이면서 동시에 **"고제구율 시대에 많이 던진
       투수"**다. 같은 0.632가 2019년과 2024년에 다른 것을 뜻한다.
       → 검증: 통산 길이(`asof_pitcher_n`)가 길수록 오염이 크므로,
         success ~ career_rate 기울기가 **긴 커리어 쪽에서 더 완만**해야 한다.

실행:  .venv/Scripts/python.exe -u src/eda_tail.py 2>&1 | tee experiments/eda_tail.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache
from config import MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

FEATS = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
         "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
         "asof_pitcher_prev5_game_success_rate", "asof_batter_success_rate",
         "asof_pitcher_middle_rate"]
SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
NB = 20


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)

    # ══ Q1. 꼬리 vs 중간 안정성 ═══════════════════════════════════
    print("=" * 94)
    print("Q1. 시즌간 산포 — 중간 빈(4~15) vs 꼬리 빈(0~1, 18~19)")
    print("=" * 94)
    print(f"{'컬럼':<45} {'중간':>9} {'꼬리':>9} {'배율':>7} {'꼬리 최대격차':>13}")
    for c in FEATS:
        v = df[c].to_numpy(dtype=np.float64)
        fin = np.isfinite(v)
        edges = np.unique(np.quantile(v[fin], np.linspace(0, 1, NB + 1)[1:-1]))
        b = np.searchsorted(edges, v, side="right")
        rel = []
        for ss in SEASONS:
            m = (s == ss) & fin
            cnt = np.bincount(b[m], minlength=NB).astype(float)
            tot = np.bincount(b[m], weights=y[m], minlength=NB)
            r = np.where(cnt >= 200, tot / np.maximum(cnt, 1), np.nan)
            rel.append(r - np.nanmean(r))
        R = np.array(rel)                       # (시즌, 빈)
        mid = np.nanmean(np.nanstd(R[:, 4:16], axis=0))
        tail_idx = [0, 1, NB - 2, NB - 1]
        tail = np.nanmean(np.nanstd(R[:, tail_idx], axis=0))
        gap = np.nanmax(np.nanmax(R[:, tail_idx], 0) - np.nanmin(R[:, tail_idx], 0))
        print(f"{c:<45} {mid:>9.4f} {tail:>9.4f} {tail/mid:>7.1f}x {gap:>13.4f}")
    print("\n  배율이 크면 '관계는 안정적인데 꼬리만 흔들린다'는 뜻이다.")

    # ══ Q2. 모델이 꼬리에서 잃나 ══════════════════════════════════
    print("\n" + "=" * 94)
    print("Q2. 배포 모델의 꼬리 손실 — train 분위 기준 꼬리 5% 행")
    print("=" * 94)
    cached = foldcache.load([2023, 2024], verbose=False)
    for fold in (2023, 2024):
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        yv = cached[fold][2]
        p = shift_to_prior(1.0 / (1.0 + np.exp(-z)), yv.mean())
        d = df.loc[s == fold].reset_index(drop=True)
        tr = df.loc[s < fold]
        ref = yv.mean() * (1 - yv.mean())
        print(f"\n  fold{fold}  전체 BSS {bss(yv, p):.1f}")
        print(f"    {'컬럼':<42} {'구간':>8} {'n':>9} {'실제':>8} {'예측':>8}"
              f" {'레벨손실':>9} {'내부refine':>11}")
        for c in ("asof_pitcher_success_rate", "asof_batter_success_rate"):
            vtr = tr[c].to_numpy(dtype=np.float64)
            vtr = vtr[np.isfinite(vtr)]
            lo, hi = np.quantile(vtr, [0.05, 0.95])
            v = d[c].to_numpy(dtype=np.float64)
            for nm, m in (("하위5%", v < lo), ("중간90%", (v >= lo) & (v <= hi)),
                          ("상위5%", v > hi)):
                if m.sum() < 200:
                    continue
                ys, ps = yv[m], p[m]
                lvl = 100000.0 * ((ps.mean() - ys.mean()) ** 2) / ref
                rf = 100000.0 * (ys.mean() * (1 - ys.mean())
                                 - ((ps - ys) ** 2).mean()) / ref
                print(f"    {c:<42} {nm:>8} {int(m.sum()):>9,} {ys.mean():>8.4f}"
                      f" {ps.mean():>8.4f} {lvl:>9.1f} {rf:>11.1f}")

    # ══ Q3. era 오염 ═════════════════════════════════════════════
    print("\n" + "=" * 94)
    print("Q3. era 오염 — 통산이 길수록 career_rate의 기울기가 완만해지는가")
    print("=" * 94)
    print("   기울기 = 검증 시즌에서 success ~ career_rate 를 가중 회귀한 값.")
    print("   오염이 없다면 커리어 길이와 무관하게 같아야 한다.\n")
    EDG = np.array([200, 800, 2000, 5000, 12000], dtype=np.float64)
    NAMES = ["<200", "200–800", "800–2k", "2k–5k", "5k–12k", "12k+"]
    for fold in (2023, 2024):
        m = s == fold
        d = df.loc[m]
        v = d["asof_pitcher_success_rate"].to_numpy(dtype=np.float64)
        n = d["asof_pitcher_n"].to_numpy(dtype=np.float64)
        yy = y[m]
        ok = np.isfinite(v)
        bucket = np.searchsorted(EDG, n)
        print(f"  fold{fold}")
        print(f"    {'커리어':>10} {'n행':>9} {'career_rate 평균':>17} {'기울기':>9} {'SE':>7}")
        for k in range(len(EDG) + 1):
            mm = ok & (bucket == k)
            if mm.sum() < 2000:
                continue
            x, t = v[mm], yy[mm]
            xc = x - x.mean()
            den = float((xc ** 2).sum())
            slope = float((xc * (t - t.mean())).sum() / den)
            resid = t - t.mean() - slope * xc
            se = float(np.sqrt((resid ** 2).sum() / (len(t) - 2) / den))
            print(f"    {NAMES[k]:>10} {int(mm.sum()):>9,} {x.mean():>17.4f}"
                  f" {slope:>9.3f} {se:>7.3f}")
        print()


if __name__ == "__main__":
    main()
