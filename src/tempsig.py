"""시간 주기 신호가 **남아 있는가** — ICML'25 시간 임베딩을 구현하기 전의 관문.

문헌
----
Cai & Ye, *Understanding the Limits of Deep Tabular Methods with Temporal Shift*
(ICML 2025, arXiv:2502.20260). 딥 테이블 모델이 **주기·추세 정보를 못 잡는다**고
보고하고, 푸리에 급수 기반 시간 임베딩을 붙인다:

    ψ(t) = [ReLU(Linear(Fourier(t,T))), Trend(t)]      Fourier(t,T)=[sin,cos](2πkt/T)
    보고 이득: MLP +1.31%, 평균 +0.54%. **주 단위·연 단위 주기가 뚜렷한 데이터셋**에서 큼.

우리 파이프라인의 상태
----------------------
`game_month`(3~10)와 `game_dayofweek`(0~6)를 **표준화된 스칼라 그대로** 먹인다.
MLP 입장에서는 "월"이 **선형** 입력이다 — 굽은 계절 효과를 표현할 수 없다.
연속 시간 입력은 아예 없고, 시간 구조 전부를 **스칼라 logit shift 하나**로 처리한다.

차원 예산은 문제가 안 된다 — §7.26이 세운 법칙("입력 차원을 늘리면 단조로 진다")은
PLE +150차원 규모에서 잰 것이고, K=2 푸리에는 연/주 합쳐 **+8차원**이다 (73 → 81).

그래서 먼저 물어야 할 것은 차원이 아니라 **신호가 있는가**다.

무엇을 재는가 (학습 없음)
------------------------
  [A] **시즌 레벨을 뺀 월별 형상이 시즌을 넘어 반복되는가.**
      이건 train 안에서 완전히 식별된다 — 6시즌 × 15개 쌍 상관.
      반복되지 않으면 잡을 주기가 애초에 없다. (§7.34 F는 2 fold만 봤다)
  [B] 요일 축도 같이. §7.34 F는 요일을 본 적이 없다.
  [C] **배포 모델이 이미 잡고 있는가** — 예측의 형상 vs 실제의 형상, 그리고 잔차의 형상.
      실제엔 있는데 예측에 없으면 그게 헤드룸이다.
  [D] 형상의 크기를 BSS로 환산. 완벽히 잡았을 때의 **상한**.

대조
----
  각 시즌 안에서 행을 무작위 재배치해 만든 가짜 형상 — 표본 노이즈만의 상관.
  [A]의 상관을 이것과 비교해야 의미가 있다.

실행:  .venv/Scripts/python.exe -u src/tempsig.py 2>&1 | tee experiments/tempsig.log
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
from config import CTX_AXES, CTX_K, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from ctxsplit import build_delta, lookup
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
K = CTX_K
SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]


def shape(vals, codes, ncode, sel):
    """sel 안에서 코드별 평균 − 전체 평균. (= 레벨을 뺀 형상)"""
    v, c = vals[sel], codes[sel]
    n = np.bincount(c, minlength=ncode).astype(np.float64)
    t = np.bincount(c, weights=v, minlength=ncode)
    out = np.full(ncode, np.nan)
    ok = n > 0
    out[ok] = t[ok] / n[ok] - v.mean()
    return out, n


def pairwise_corr(mat):
    """행=시즌 형상. 결측 아닌 공통 칸으로 쌍별 상관."""
    cs = []
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            ok = ~(np.isnan(mat[i]) | np.isnan(mat[j]))
            if ok.sum() >= 4:
                cs.append(float(np.corrcoef(mat[i][ok], mat[j][ok])[0, 1]))
    return np.array(cs)


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    mon = df["game_month"].to_numpy(dtype=np.int64)
    dow = df["game_dayofweek"].to_numpy(dtype=np.int64)
    pid = df["pitcher_id"].to_numpy(dtype=np.int64)
    bh = df["batter_hand"].to_numpy(dtype=np.int64)
    b = df["balls_before"].to_numpy(dtype=np.int64)
    st = df["strikes_before"].to_numpy(dtype=np.int64)
    c4 = (b >= 3).astype(np.int64) + 2 * (st >= 2).astype(np.int64)

    rng = np.random.default_rng(42)
    print(f"train {df.shape}   월 {sorted(np.unique(mon))}   요일 {sorted(np.unique(dow))}\n")

    # ── [A][B] 레벨을 뺀 형상이 시즌을 넘어 반복되는가 ────────────
    for label, code, nc in (("월(game_month)", mon, int(mon.max()) + 1),
                            ("요일(game_dayofweek)", dow, 7)):
        print("=" * 96)
        print(f"[{label}] 시즌 레벨을 뺀 형상 (×1000)")
        print("=" * 96)
        mat, matr = [], []
        for t_ in SEASONS:
            sel = s == t_
            sh, n = shape(y, code, nc, sel)
            mat.append(sh)
            # 대조 — 같은 시즌 안에서 y를 셔플 (표본 노이즈만 남는다)
            yr = y.copy()
            yr[sel] = rng.permutation(y[sel])
            matr.append(shape(yr, code, nc, sel)[0])
            cells = " ".join(f"{v*1000:+6.1f}" if v == v else "     ." for v in sh)
            print(f"  {t_}  {cells}   (n {int(n.sum()):,})")
        mat, matr = np.array(mat), np.array(matr)
        c_real, c_rand = pairwise_corr(mat), pairwise_corr(matr)
        print(f"\n  시즌 쌍 상관 (15쌍)   실제 평균 {c_real.mean():+.3f}"
              f"  (양수 {int((c_real>0).sum())}/{len(c_real)})"
              f"   |  셔플 대조 {c_rand.mean():+.3f}")
        # 형상의 크기 → BSS 상한 (형상을 완벽히 알았을 때)
        for f in FOLDS:
            sel = s == f
            sh, n = shape(y, code, nc, sel)
            v = sh[code[sel]]
            print(f"    {f} 형상 std {np.nanstd(v):.5f}"
                  f"  → 완벽히 잡았을 때 BSS 상한 {100000*np.nanvar(v)/0.2498:+.1f}")
        print()

    # ── [C] 배포 모델이 이미 잡고 있는가 ──────────────────────────
    cached = foldcache.load(FOLDS, verbose=False)
    print("=" * 96)
    print("[C] 배포 모델이 이미 잡고 있는가 — 실제 형상 vs 예측 형상 vs 잔차 형상 (×1000)")
    print("=" * 96)
    for fold in FOLDS:
        tr, va = s < fold, s == fold
        z_lgb, z_mlp, yv, _ = cached[fold]
        z0 = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
        for name, beta in CTX_AXES:
            code_, m = (bh, 8) if name == "hand" else (c4, 4)
            u, d, _ = build_delta(pid[tr], code_[tr], y[tr], m, K)
            drow, _ = lookup(u, d, pid[va], code_[va], m)
            z0 = z0 + beta * drow
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean())
        r = yv - p0

        for label, code, nc in (("월", mon[va], int(mon.max()) + 1),
                                ("요일", dow[va], 7)):
            a_y = shape(yv, code, nc, np.ones(len(yv), bool))[0]
            a_p = shape(p0, code, nc, np.ones(len(p0), bool))[0]
            a_r = shape(r, code, nc, np.ones(len(r), bool))[0]
            ok = ~np.isnan(a_y)
            cor = float(np.corrcoef(a_y[ok], a_p[ok])[0, 1]) if ok.sum() > 3 else np.nan
            print(f"  fold{fold} {label}")
            print(f"    실제  " + " ".join(f"{v*1000:+6.1f}" if v == v else "     ." for v in a_y))
            print(f"    예측  " + " ".join(f"{v*1000:+6.1f}" if v == v else "     ." for v in a_p)
                  + f"   corr(실제,예측) {cor:+.3f}")
            print(f"    잔차  " + " ".join(f"{v*1000:+6.1f}" if v == v else "     ." for v in a_r)
                  + f"   잔차 std {np.nanstd(a_r[ok]):.5f}"
                  f"  → 남은 BSS {100000*float(np.nanvar(a_r[ok[code]] if False else a_r[code]))/0.2498:+.1f}")
        print()

    print("=" * 96)
    print("판정 규칙 (돌리기 전에 적는다)")
    print("  [A]/[B] 시즌 쌍 상관이 셔플 대조와 구별 안 되면 → 잡을 주기가 없다. 끝.")
    print("  [C] corr(실제,예측)이 높고 잔차 형상이 0에 가까우면 → 이미 잡고 있다. 끝.")
    print("  둘 다 통과할 때만 푸리에 시간 임베딩을 MLP에 붙일 값어치가 있다.")


if __name__ == "__main__":
    main()
