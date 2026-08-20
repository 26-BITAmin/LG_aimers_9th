"""요일 사후 로짓 보정 — `tempsig.py`가 남긴 유일한 생존 축을 조건 ①②③으로 판정한다.

`tempsig.py`가 찾은 것
----------------------
    월  : 시즌 쌍 상관 −0.131  vs 셔플 대조 −0.159   → **주기가 없다.** 기각
    요일: 시즌 쌍 상관 +0.353  vs 셔플 대조 −0.121   → 실재. 그리고 **월요일 한 칸**이다

    월요일 (레벨 뺀 값 ×1000)
      2019 −36.4  2020 −21.6  2021 −34.3  2022 −58.2  2023 **+15.4**  2024 −48.0
    fold2024 잔차: 실제 −48.0 인데 예측 −12.1 → 모델이 **1/4만** 잡는다

기전 (도메인)
-------------
KBO 월요일은 원래 휴식일이다. 월요일 경기는 대부분 **우천 순연 보충경기**라
등판 간격·불펜 운용이 비정상이고, 투구 비중도 0.9~1.8%로 극히 작다.

**왜 모델이 못 잡는가**에 대한 가설: `RECENCY_HALF_LIFE=0.5`는 유효 표본을
~1.33시즌으로 만든다. 월요일은 시즌당 투구가 2~4천 개뿐이라, 모델은 사실상
**한 시즌치 월요일**로 이 칸을 추정한다. 저빈도 범주가 recency 가중에 굶는다.
→ 그렇다면 **전 시즌을 균등 가중으로 모은 추정치**가 더 나아야 한다.

⚠️ 사전에 적어두는 반대 증거: **2023은 부호가 반대다(+15.4).** 조건 ②(양 fold
부호 일치)를 통과할 가능성이 낮다. 그래도 재는 이유는 [A] 손으로 읽은 표가
아니라 실측이 판정해야 하고, [B] 통과 못 해도 "왜 못 잡는가"의 기전 검정이
`hl`의 부작용에 대한 증거를 남기기 때문이다.

대조군
------
  `month` — `tempsig`에서 이미 죽은 축. 여기서도 0이 나와야 절차가 맞는 것이다.
  `shuffle` — 요일 라벨을 시즌 안에서 섞은 것.

실행:  .venv/Scripts/python.exe -u src/dowfit.py 2>&1 | tee experiments/dowfit.log
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
from config import (CTX_AXES, CTX_K, MLP_ALPHA, RECENCY_HALF_LIFE, TARGET_COL,
                    TRAIN_CSV)
from ctxsplit import build_delta, lookup, resid_t
from data import load_csv, recency_weight
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
K = CTX_K
BETAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def cell_offset(code, y, w, nc, prior_w):
    """시즌 레벨을 뺀 칸별 편차를 가중 평균. 수축 n/(n+k)."""
    n = np.bincount(code, weights=w, minlength=nc)
    t = np.bincount(code, weights=w * y, minlength=nc)
    base = float((w * y).sum() / w.sum())
    out = np.zeros(nc)
    ok = n > 0
    out[ok] = (n[ok] / (n[ok] + prior_w)) * (t[ok] / n[ok] - base)
    return out


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    dow = df["game_dayofweek"].to_numpy(dtype=np.int64)
    mon = df["game_month"].to_numpy(dtype=np.int64)
    pid = df["pitcher_id"].to_numpy(dtype=np.int64)
    bh = df["batter_hand"].to_numpy(dtype=np.int64)
    b = df["balls_before"].to_numpy(dtype=np.int64)
    st = df["strikes_before"].to_numpy(dtype=np.int64)
    c4 = (b >= 3).astype(np.int64) + 2 * (st >= 2).astype(np.int64)

    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(42)
    summary = {}

    for fold in FOLDS:
        tr, va = s < fold, s == fold
        z_lgb, z_mlp, yv, _ = cached[fold]
        z0 = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
        for name, beta in CTX_AXES:
            cc, m = (bh, 8) if name == "hand" else (c4, 4)
            u, d, _ = build_delta(pid[tr], cc[tr], y[tr], m, K)
            drow, _ = lookup(u, d, pid[va], cc[va], m)
            z0 = z0 + beta * drow
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean())
        base = bss(yv, p0)
        r = yv - p0

        w_flat = np.ones(int(tr.sum()))
        w_rec = recency_weight(s[tr], RECENCY_HALF_LIFE,
                               ref_season=int(s[tr].max()))

        dsh = dow.copy()
        for t_ in np.unique(s):
            m_ = s == t_
            dsh[m_] = rng.permutation(dow[m_])

        cands = {
            "요일 균등가중":   (dow, 7, w_flat),
            "요일 recency":    (dow, 7, w_rec),
            "월 (대조)":       (mon, int(mon.max()) + 1, w_flat),
            "요일셔플 (대조)": (dsh, 7, w_flat),
        }

        print("=" * 100)
        print(f"fold {fold}   valid {va.sum():,}   기준 BSS(배포상태) {base:.1f}")
        print("=" * 100)
        print(f"  {'축':<18}{'월요일 오프셋':>14}{'T2(t)':>9}   "
              + "".join(f"β{x:<7.2f}" for x in BETAS))
        for tag, (code, nc, w) in cands.items():
            off = cell_offset(code[tr], y[tr], w, nc, 0.0)
            row = off[code[va]]
            _, t2 = resid_t(r, row)
            g = []
            for beta in BETAS:
                p = shift_to_prior(1.0 / (1.0 + np.exp(-(z0 + beta * row))), yv.mean())
                g.append(bss(yv, p) - base)
            summary.setdefault(tag, {})[fold] = g
            mon_off = off[0] if nc == 7 else float("nan")
            print(f"  {tag:<18}{mon_off*1000:>13.1f}‰{t2:>+9.2f}   "
                  + "".join(f"{v:>+8.2f}" for v in g))
        print()

    print("=" * 100)
    print("요약 — 조건 ②(양 fold 부호 일치) ③(크기비 ≤3)")
    print("=" * 100)
    print(f"  {'축':<18}" + "".join(f"β{x:<11.2f}" for x in BETAS))
    for tag, per in summary.items():
        cells = ""
        for i in range(len(BETAS)):
            a, bb = per[FOLDS[0]][i], per[FOLDS[1]][i]
            ok = a > 0 and bb > 0
            ratio = max(abs(a), abs(bb)) / max(min(abs(a), abs(bb)), 1e-9)
            cells += f"{a:>+5.1f}/{bb:>+5.1f}{'✓' if ok and ratio <= 3 else ('~' if ok else ' ')}"
        print(f"  {tag:<18}{cells}")
    print("\n  ✓ = 양 fold 양수 이고 크기비 ≤3    ~ = 부호만 일치")


if __name__ == "__main__":
    main()
