"""§7.108 — `n_szn` 게이트로 투수 편차 증폭 (§7.107 진단의 행동판).

`psharp3.py` 가 낸 것: β(과소확신)는 **달력이 아니라 그 투수의 올해 투구 수 n_szn** 을
따라 떨어진다.

    n_szn      0~50   50~200  200~500  500~1200   1200+
    β(2023)    1.95    1.48     1.33     1.22      0.99
    β(2024)    1.61    1.21     1.16     1.22      1.07

§7.105 는 같은 증폭을 **|u| 분위수**라는 임의 게이트로 걸었다가 정직한 전방 선택에서
전부 죽었다. 이번 게이트는 **기전이 지목한 것**이다 — 다른 자리다.

    z' = z + g·w(n_szn)·u        u = n/(n+200)·(asof_success − 리그평균)  ← 행 독립·합법

가족 (w 의 모양)
    flat    w ≡ 1                        ← §7.105 가 쟀던 것 (대조군)
    hard    w = 1[n_szn < τ]             τ ∈ {50, 200, 500}
    decay   w = 1/(1 + n_szn/n₀)         n₀ ∈ {50, 150, 400}  ← 기전이 말하는 매끄러운 모양

⚠️ 세 가지를 그대로 지킨다 (§7.105 에서 하나씩 다 필요했다)
  ① **온도 대조군** — z+g·w·u 는 예측 분산도 키운다. 같은 std 를 만드는 전역 c 팔을 뺀다.
  ② **정직한 전방 선택** — (가족, 파라미터, g) 를 과거 fold 합산 순이득으로 고르고
     미래 fold 에서만 평가한다. 평가 fold 를 보고 고르면 이 문제에서 +2.0 이 부풀려진다.
  ③ **클러스터 부트스트랩** — 효과가 투수 단위라 유효 표본이 400명이다. 행 부트스트랩은
     폭을 크게 과소추정한다 (§7.105 에서 한 fold 폭이 ±5~20 BSS 였다).
  ④ 게이트 위약 — 투수 정체만 뒤섞은 u 에 **같은 게이트**. 같이 오르면 게이트 효과다.

⚠️ `n_szn` 의 시즌시작 상수는 **train(<f) 에서만** 잡는다. 검증 시즌 행들의 최솟값을
   쓰면 test-side 통계다. 2025 에서 할 수 있는 것과 같은 형태로 만든다
   (§7.55 `asofdelta` 가 이미 쓰는 기계).

실행:  .venv/Scripts/python.exe -u src/psharp4.py 2>&1 | tee experiments/psharp4.log
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
GG = np.linspace(0.0, 1.60, 33)
FAMS = ([("flat", None)]
        + [("hard", t) for t in (50, 200, 500)]
        + [("decay", n) for n in (50, 150, 400)])
SEED = 42
EPS = 1e-6


def wgate(fam, par, nsz):
    if fam == "flat":
        return np.ones(len(nsz))
    if fam == "hard":
        return (nsz < par).astype(float)
    if fam == "decay":
        return 1.0 / (1.0 + nsz / par)
    raise ValueError(fam)


def main():
    rng = np.random.default_rng(SEED)
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
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
        X, _, _, _ = J.axis_matrix(df, s, f)
        yv = cached[f][2]
        vi = np.where(s == f)[0]
        assert len(vi) == len(yv) and np.allclose(y[vi], yv)
        mm = mclean[vi]
        ii = vi[mm]
        zb = (1 - MLP_ALPHA) * cached[f][0] + MLP_ALPHA * cached[f][1]
        z = (zb + X @ dep)[mm]

        # u — 행 자신의 컬럼만. 리그평균은 train(<f) 에서 얼린다
        av, nv = ar[ii], np.nan_to_num(an[ii])
        ref = float(np.nanmean(ar[mclean & (s < f)]))
        u = np.where(~np.isfinite(av), 0.0, (nv / (nv + 200.0)) * (np.nan_to_num(av) - ref))

        # n_szn — 시즌시작 상수를 **train(<f) 의 커리어 최댓값**으로 (test-side 금지)
        tm = s < f
        start = np.zeros(n_id)
        np.maximum.at(start, codes[tm], np.nan_to_num(an[tm]))
        nsz = np.maximum(nv - start[codes[ii]], 0.0)

        B[f] = dict(y=y[ii], z=z, u=u, n=nsz, c=codes[ii])
        print(f"  fold {f}: clean {len(ii):,}행   n_szn 중앙값 {np.median(nsz):.0f}"
              f"   n_szn<50 {float((nsz<50).mean()):.1%}   기준 BSS "
              f"{bss(y[ii], shift_to_prior(1/(1+np.exp(-z)), y[ii].mean())):.1f}")

    def net(f, fam, par, g, u=None, idx=None):
        """u팔 − 같은 std 를 만드는 전역 온도팔. 이 차이만 '정체의 값어치'다."""
        d = B[f]
        yy, zz = d["y"], d["z"]
        uu = d["u"] if u is None else u
        w = wgate(fam, par, d["n"])
        if idx is not None:
            yy, zz, uu, w = yy[idx], zz[idx], uu[idx], w[idx]
        zn = zz + g * w * uu
        a = bss(yy, shift_to_prior(1 / (1 + np.exp(-zn)), yy.mean()))
        zc, sd0 = zz.mean(), zz.std()
        c = zn.std() / max(sd0, 1e-12)
        t = bss(yy, shift_to_prior(1 / (1 + np.exp(-(zc + c * (zz - zc)))), yy.mean()))
        return a - t

    # ── 1 — 가족별 fold 오라클 (그 fold 를 보고 고른 상한) ────────────────
    print(f"\n{'='*96}\n█ 1 — 가족별 **fold 오라클** 순이득 (상한. 판정 아님)\n{'='*96}")
    print(f"  {'fold':<7}" + "".join(f"{f'{a}·{b}':>15}" for a, b in FAMS))
    for f in FOLDS:
        row = []
        for fam, par in FAMS:
            vals = [net(f, fam, par, g) for g in GG]
            k = int(np.argmax(vals))
            row.append(f"{GG[k]:.2f}:{vals[k]:+6.1f}")
        print(f"  {f:<7}" + "".join(f"{v:>15}" for v in row))

    # ── 2 — ★ 정직한 전방 선택 ──────────────────────────────────────────
    print(f"\n{'='*96}\n█ 2 — ★ 정직한 전방 선택 (가족·파라미터·g 를 과거 fold 에서만)\n{'='*96}")
    print(f"  {'선택 fold':<18}{'→ 선택':>20}{'평가':>6}{'순이득':>9}"
          f"{'95% 클러스터 부트스트랩':>26}{'위약':>8}")
    for trf, ev in (((2021,), 2022), ((2021, 2022), 2023), ((2021, 2022, 2023), 2024)):
        best, sel = -1e9, None
        for fam, par in FAMS:
            for g in GG:
                v = sum(net(f, fam, par, g) for f in trf)
                if v > best:
                    best, sel = v, (fam, par, g)
        fam, par, g = sel
        val = net(ev, fam, par, g)
        d = B[ev]
        cf = d["c"]
        order = np.argsort(cf, kind="stable")
        bnd = np.searchsorted(cf[order], np.arange(n_id + 1))
        seen = np.where(np.diff(bnd) > 0)[0]
        bs = []
        for _ in range(200):
            pick = rng.choice(seen, size=len(seen), replace=True)
            idx = np.concatenate([order[bnd[j]:bnd[j + 1]] for j in pick])
            bs.append(net(ev, fam, par, g, idx=idx))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        n_ = np.bincount(cf, minlength=n_id).astype(float)
        us = np.bincount(cf, weights=d["u"], minlength=n_id) / np.maximum(n_, 1)
        sn = np.where(n_ > 0)[0]
        usp = us.copy(); usp[sn] = us[rng.permutation(sn)]
        pl = net(ev, fam, par, g, u=usp[cf])
        lab = f"{fam}" + (f"·{par}" if par is not None else "") + f" g={g:.2f}"
        print(f"  {'·'.join(map(str,trf)):<18}{lab:>20}{ev:>6}{val:>9.1f}"
              f"{f'[{lo:+.1f}, {hi:+.1f}]':>26}{pl:>8.1f}")

    # ── 3 — 기전 고정: n₀·τ 를 기전이 준 값으로 못 박고 g 만 ──────────────
    print(f"\n{'='*96}\n█ 3 — 노브 1개만: 게이트 모양을 **기전이 준 값**으로 고정하고 g 만 고른다\n{'='*96}")
    print("  psharp3 의 β 곡선이 n₀≈150 근처의 1/(1+n/n₀) 를 닮았다. 모양을 데이터로 고르지")
    print("  않고 고정하면 자유도가 g 하나다 — [전이 배율은 얼리는 파라미터 수로 갈린다].\n")
    print(f"  {'선택 fold':<18}{'→ g':>8}{'평가':>6}{'순이득':>9}{'95% 부트스트랩':>24}{'위약':>8}")
    for trf, ev in (((2021,), 2022), ((2021, 2022), 2023), ((2021, 2022, 2023), 2024)):
        vals = [sum(net(f, "decay", 150, g) for f in trf) for g in GG]
        g = float(GG[int(np.argmax(vals))])
        val = net(ev, "decay", 150, g)
        d = B[ev]; cf = d["c"]
        order = np.argsort(cf, kind="stable")
        bnd = np.searchsorted(cf[order], np.arange(n_id + 1))
        seen = np.where(np.diff(bnd) > 0)[0]
        bs = []
        for _ in range(200):
            pick = rng.choice(seen, size=len(seen), replace=True)
            idx = np.concatenate([order[bnd[j]:bnd[j + 1]] for j in pick])
            bs.append(net(ev, "decay", 150, g, idx=idx))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        n_ = np.bincount(cf, minlength=n_id).astype(float)
        us = np.bincount(cf, weights=d["u"], minlength=n_id) / np.maximum(n_, 1)
        sn = np.where(n_ > 0)[0]
        usp = us.copy(); usp[sn] = us[rng.permutation(sn)]
        pl = net(ev, "decay", 150, g, u=usp[cf])
        print(f"  {'·'.join(map(str,trf)):<18}{g:>8.2f}{ev:>6}{val:>9.1f}"
              f"{f'[{lo:+.1f}, {hi:+.1f}]':>24}{pl:>8.1f}")

    print(f"\n{'='*96}\n■ 읽는 법\n{'='*96}")
    print("  2·3 이 판정이다. 두 평가 fold 에서 순이득 양수 + 부트스트랩 하한 > 0 + 위약 ≈ 0.")
    print("  1 은 상한일 뿐이다 — 그 fold 를 보고 고른 값이라 항상 좋아 보인다.")
    print("  3 이 2 보다 나으면 '모양을 데이터로 고른 것'이 손해였다는 뜻이다.")


if __name__ == "__main__":
    main()
