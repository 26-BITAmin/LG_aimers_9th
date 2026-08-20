"""§7.80 후속 — 투수내(상황) 성분을 수축했을 때의 **상한** 측정.

배경
    투수 평균을 뺀 예측 변동의 캘리브레이션 기울기가 1이 아니다
    (신인 .46/.53, 베테랑 .67/.86). 기울기가 s면 그 성분을 s배로 줄이는 것이
    Brier 최적이다. 얼마나 벌리나?

측정 대상 — **상한이지 구현안이 아니다**
    투수 평균 z̄_j를 검증 fold 행에서 직접 계산한다. 제출 규정상 추론 시엔
    이렇게 못 한다 (test 행 간 집계 금지). 여기서 나온 값은
    "이 채널에 이만큼 있다"의 천장이고, 구현은 행 독립 대체물이 필요하다.
    천장이 작으면 대체물을 설계할 이유가 없다 — 그래서 이걸 먼저 잰다.

    z' = z̄_j + s·(z − z̄_j)   (로짓 공간) → prior shift → BSS

    s를 전역 / 경력별(신인·1시즌·2+) 두 방식으로 스윕한다.

실행:  LGA_MODEL_DIR=submit/model_shrinksit .venv/Scripts/python.exe -u src/shrinksit.py
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
from config import TRAIN_CSV
from data import load_csv
from rookie import FOLDS, deployed_pred, prior_seasons
from validation import _logit, bss, shift_to_prior

B = 1000
RNG = np.random.default_rng(42)
GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def apply_shrink(z, cl, ncl, sv):
    """sv는 행별 수축 계수. z̄_j는 그 투수의 fold 내 평균 (오라클)."""
    n = np.bincount(cl, minlength=ncl)
    zb = np.bincount(cl, weights=z, minlength=ncl) / np.maximum(n, 1)
    return zb[cl] + sv * (z - zb[cl])


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    print(f"foldcache 지문 고정: {foldcache._key()}  (배포 s23)")

    for fold in FOLDS:
        va, y, p = deployed_pred(df, s, fold, cached)
        z = _logit(p)
        pid = va["pitcher_id"].to_numpy()
        _, cl = np.unique(pid, return_inverse=True)
        ncl = cl.max() + 1
        pp = prior_seasons(df, s, fold, "pitcher_id")
        base = bss(y, p)
        print(f"\n=== fold {fold}  기준 BSS {base:.1f} (투수 {ncl:,}명) ===")

        print("  [전역 s]")
        best = (None, -1e9)
        for sc in GRID:
            zz = apply_shrink(z, cl, ncl, sc)
            b = bss(y, shift_to_prior(1 / (1 + np.exp(-zz)), y.mean()))
            print(f"    s={sc:.1f}   BSS {b:>8.1f}   Δ {b-base:>+8.1f}")
            if b - base > best[1]:
                best = (sc, b - base)
        print(f"    → 최적 s={best[0]:.1f}  Δ {best[1]:+.1f}")

        # 경력별로 나눠 각각 최적 s (좌표 하강 1회면 충분 — 그룹이 서로 배타적)
        grp = np.where(pp == 0, 0, np.where(pp == 1, 1, 2))
        names = ["신인", "1시즌", "2+시즌"]
        sv = np.ones(len(z))
        picks = []
        for g in range(3):
            m = grp == g
            bb = (1.0, 0.0)
            for sc in GRID:
                t = sv.copy(); t[m] = sc
                zz = apply_shrink(z, cl, ncl, t)
                d = bss(y, shift_to_prior(1 / (1 + np.exp(-zz)), y.mean())) - base
                if d > bb[1]:
                    bb = (sc, d)
            sv[m] = bb[0]
            picks.append((names[g], m.mean(), bb[0]))
        zz = apply_shrink(z, cl, ncl, sv)
        tot = bss(y, shift_to_prior(1 / (1 + np.exp(-zz)), y.mean())) - base
        print("  [경력별 s]  " + "  ".join(f"{n}({w*100:.0f}%) s={x:.1f}" for n, w, x in picks))
        print(f"    → 합산 Δ {tot:+.1f}")

        # 채택 전 최소 조건: 전역 s가 두 fold에서 같은 방향인가 + 쌍체 SE
        zz = apply_shrink(z, cl, ncl, np.full(len(z), best[0]))
        pa = shift_to_prior(1 / (1 + np.exp(-zz)), y.mean())
        d = (p - y) ** 2 - (pa - y) ** 2
        num = np.bincount(cl, weights=d, minlength=ncl)
        cnt = np.bincount(cl, minlength=ncl)
        bt = np.empty(B)
        for b in range(B):
            i = RNG.integers(0, ncl, ncl)
            bt[b] = num[i].sum() / cnt[i].sum()
        k = 1e5 / (y.mean() * (1 - y.mean()))
        print(f"    쌍체 Δ(s={best[0]:.1f}) {k*num.sum()/cnt.sum():+.1f}"
              f" ± {k*np.nanstd(bt, ddof=1):.1f}  (투수 클러스터 부트스트랩)")


if __name__ == "__main__":
    main()
