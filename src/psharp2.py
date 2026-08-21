"""시즌 시기별 g — §7.105 파트 4의 기전을 행동으로 옮긴다 (사용자 요청).

파트 4가 남긴 것:  β(시즌 시기) 가 초반 1.3~1.5 → 9월 1.0 으로 떨어진다.
                   판정 fold(2023·2024) 둘 다 9월에 β≈1.0 — 시즌 말엔 이미 정확하다.
→ 그러면 g 도 시즌 초에 크고 말에 0 이어야 한다. `psharp.py` 는 전 구간 같은 g 를
   걸었고 그래서 두 손해(초반 근거없음 · 말 과신)가 합쳐졌을 수 있다.

⚠️ 노브가 늘어난다. 세 가지를 동시에 지킨다.

  ① **기전이 모양을 고정하는 1노브 가족**을 자유 4노브와 정면으로 붙인다.
     [전이 배율은 얼리는 파라미터 수로 갈린다] — 스칼라 1개 0.93 / 표 3.3 / 분할 68.6.
  ② **온도 대조군** — z+g·u 는 예측 분산도 키운다. 같은 std 를 만드는 전역 c 팔을 뺀다.
  ③ **월별 위약** — 월마다 g 를 바꾸면 월별 캘리브레이션 차이를 주워담을 수 있다
     (투수와 무관하게). 투수 정체만 섞은 u 에 **같은 월별 g** 를 먹여 같이 잰다.
     위약이 같이 오르면 그건 투수 효과가 아니라 월 효과다.

가족 (w = 월별 가중, g_row = g0·w(month))
    flat     w ≡ 1                          ← psharp.py 가 쟀던 것 (대조)
    early    w = 1 (≤6월) / 0 (7월~)        1노브
    linear   w = max(0, 1 − (m−3)/6)        1노브. 3월 1.00 → 9월 0.00
    free4    버킷 4개 각각 자유             4노브 (과적합 상한 확인용)

판정은 **정직한 전방 선택** — (가족, 파라미터)를 과거 fold 합산 순이득으로 고르고
미래 fold 에서만 평가한다. 오차막대는 투수 클러스터 부트스트랩.

실행:  .venv/Scripts/python.exe -u src/psharp2.py 2>&1 | tee experiments/psharp2.log
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

import jointbeta as J
from config import ASOF_AXES, CTX_AXES, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = [2021, 2022, 2023, 2024]
GG = np.linspace(0.0, 1.20, 25)
SEED = 42
BUCKETS = [(3, 4), (5, 6), (7, 8), (9, 12)]
BLAB = ["3~4월", "5~6월", "7~8월", "9월~"]


def w_family(name, month):
    if name == "flat":
        return np.ones(len(month))
    if name == "early":
        return (month <= 6).astype(float)
    if name == "linear":
        return np.clip(1.0 - (month - 3) / 6.0, 0.0, 1.0)
    raise ValueError(name)


def main():
    rng = np.random.default_rng(SEED)
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    gm = df["game_month"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    codes, uniq = pd.factorize(df["pitcher_id"].to_numpy())
    n_id = len(uniq)
    ar = df["asof_pitcher_success_rate"].to_numpy(dtype=np.float64)
    an = df["asof_pitcher_n"].to_numpy(dtype=np.float64)
    mclean = ~((gt == "F") & (s < 2023))

    cached = J.foldcache.load(FOLDS, verbose=False)
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    B = {}
    for f in FOLDS:
        X, _, _, va = J.axis_matrix(df, s, f)
        yv = cached[f][2]
        vi = np.where(s == f)[0]
        assert len(vi) == len(yv) and np.allclose(y[vi], yv)
        mm = mclean[vi]
        zb = (1 - MLP_ALPHA) * cached[f][0] + MLP_ALPHA * cached[f][1]
        zd = (zb + X @ dep)[mm]
        ii = vi[mm]
        av, nv = ar[ii], np.nan_to_num(an[ii])
        ref = float(np.nanmean(ar[mclean & (s < f)]))     # 리그평균은 train 에서 얼린다
        u = np.where(~np.isfinite(av), 0.0,
                     (nv / (nv + 200.0)) * (np.nan_to_num(av) - ref))
        B[f] = dict(y=y[ii], z=zd, u=u, m=gm[ii], c=codes[ii])
    print(f"train {df.shape}   clean 홀드아웃 " + "  ".join(
        f"{f}:{len(B[f]['y']):,}" for f in FOLDS))

    def score(f, grow, u=None, idx=None):
        """(u팔 BSS, 같은 std 를 만드는 전역 온도팔 BSS). 차이가 정체의 값어치다."""
        d = B[f]
        yy, zz = d["y"], d["z"]
        uu = d["u"] if u is None else u
        if idx is not None:
            yy, zz, uu, grow = yy[idx], zz[idx], uu[idx], grow[idx]
        zn = zz + grow * uu
        a = bss(yy, shift_to_prior(1 / (1 + np.exp(-zn)), yy.mean()))
        zc, sd0 = zz.mean(), zz.std()
        c = zn.std() / max(sd0, 1e-12)
        t = bss(yy, shift_to_prior(1 / (1 + np.exp(-(zc + c * (zz - zc)))), yy.mean()))
        return a, t

    # ── 5a — 버킷별로 직접 최적 g 를 찾는다 (진단) ───────────────────────
    print(f"\n{'='*84}\n█ 5a — 버킷별 최적 g 와 그 순이득 (진단. 각 버킷 안에서만 적용)\n{'='*84}")
    print("  파트 4의 β 곡선이 행동에서도 재현되는가. β 초반 大 → g 초반 大 여야 한다.\n")
    print(f"  {'fold':<7}" + "".join(f"{lab:>20}" for lab in BLAB))
    print(f"  {'':7}" + "".join(f"{'최적 g  순이득':>20}" for _ in BLAB))
    for f in FOLDS:
        d = B[f]
        cells = []
        for lo, hi in BUCKETS:
            sel = (d["m"] >= lo) & (d["m"] <= hi)
            best, bg = -1e9, 0.0
            for g in GG:
                grow = np.where(sel, g, 0.0)
                a, t = score(f, grow)
                if a - t > best:
                    best, bg = a - t, g
            cells.append(f"{bg:>6.2f}{best:>+9.1f}   ")
        print(f"  {f:<7}" + "".join(f"{c:>20}" for c in cells))

    # ── 5b — 가족별 fold 최적 (오라클. 상한 확인용) ──────────────────────
    print(f"\n{'='*84}\n█ 5b — 가족별 **fold 오라클** 순이득 (그 fold를 보고 고른 값 = 상한)\n{'='*84}")
    print(f"  {'fold':<7}" + "".join(f"{k:>22}" for k in ("flat", "early", "linear", "free4")))
    for f in FOLDS:
        d = B[f]
        row = []
        for fam in ("flat", "early", "linear"):
            w = w_family(fam, d["m"])
            vals = [score(f, g * w) for g in GG]
            k = int(np.argmax([a - t for a, t in vals]))
            row.append(f"g={GG[k]:.2f} {vals[k][0]-vals[k][1]:>+7.1f}")
        # free4 — 좌표하강 3회
        gs = np.zeros(4)
        for _ in range(3):
            for bi, (lo, hi) in enumerate(BUCKETS):
                sel = (d["m"] >= lo) & (d["m"] <= hi)
                best, bg = -1e9, gs[bi]
                for g in GG:
                    t_ = gs.copy(); t_[bi] = g
                    grow = np.zeros(len(d["y"]))
                    for bj, (l2, h2) in enumerate(BUCKETS):
                        grow[(d["m"] >= l2) & (d["m"] <= h2)] = t_[bj]
                    a, tt = score(f, grow)
                    if a - tt > best:
                        best, bg = a - tt, g
                gs[bi] = bg
        grow = np.zeros(len(d["y"]))
        for bj, (l2, h2) in enumerate(BUCKETS):
            grow[(d["m"] >= l2) & (d["m"] <= h2)] = gs[bj]
        a, tt = score(f, grow)
        row.append(f"[{','.join(f'{v:.2f}' for v in gs)}] {a-tt:>+6.1f}")
        print(f"  {f:<7}" + "".join(f"{v:>22}" for v in row))

    # ── 5c — ★ 정직한 판정 ──────────────────────────────────────────────
    print(f"\n{'='*84}\n█ 5c — ★ 정직한 전방 선택 (가족·g 를 과거 fold 에서만 고른다)\n{'='*84}")
    print("  위약 = 투수 정체만 뒤섞은 u 에 **같은 월별 g**. 같이 오르면 월 효과지 투수 효과가 아니다.\n")
    print(f"  {'선택 fold':<16}{'→ 가족·g':>18}{'평가':>6}{'순이득':>9}"
          f"{'95% 부트스트랩':>22}{'위약':>8}")
    for trf, ev in (((2021,), 2022), ((2021, 2022), 2023), ((2021, 2022, 2023), 2024)):
        best, bfam, bg = -1e9, "flat", 0.0
        for fam in ("flat", "early", "linear"):
            for g in GG:
                v = sum(np.subtract(*score(f, g * w_family(fam, B[f]["m"]))[::-1]) * -1
                        for f in trf)
                if v > best:
                    best, bfam, bg = v, fam, g
        d = B[ev]
        grow = bg * w_family(bfam, d["m"])
        a, t = score(ev, grow)
        # 클러스터 부트스트랩 (투수 복원추출)
        cf = d["c"]
        order = np.argsort(cf, kind="stable")
        bnd = np.searchsorted(cf[order], np.arange(n_id + 1))
        seen = np.where(np.diff(bnd) > 0)[0]
        bs = []
        for _ in range(200):
            pick = rng.choice(seen, size=len(seen), replace=True)
            idx = np.concatenate([order[bnd[j]:bnd[j + 1]] for j in pick])
            aa, tt = score(ev, grow, idx=idx)
            bs.append(aa - tt)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        # 월별 위약
        n_ = np.bincount(cf, minlength=n_id).astype(float)
        us = np.bincount(cf, weights=d["u"], minlength=n_id) / np.maximum(n_, 1)
        sn = np.where(n_ > 0)[0]
        usp = us.copy(); usp[sn] = us[rng.permutation(sn)]
        ap, tp = score(ev, grow, u=usp[cf])
        print(f"  {'·'.join(map(str,trf)):<16}{f'{bfam} g={bg:.2f}':>18}{ev:>6}"
              f"{a-t:>9.1f}{f'[{lo:+.1f}, {hi:+.1f}]':>22}{ap-tp:>8.1f}")

    print(f"\n{'='*84}\n■ 읽는 법\n{'='*84}")
    print("  5a 의 최적 g 가 왼쪽(초반)에서 크고 오른쪽에서 0 이면 파트 4 기전이 행동에서 재현된 것.")
    print("  5c 가 판정이다. 순이득이 두 평가 fold 에서 양수 + 부트스트랩 하한 > 0 + 위약 ≈ 0")
    print("  셋을 다 만족해야 후보다. 하나라도 깨지면 기각한다.")


if __name__ == "__main__":
    main()
