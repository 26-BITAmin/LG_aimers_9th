"""배포 경로가 **측정 경로와 같은 숫자**를 내는가 — 리팩터 검증.

§7.36에서 플래툰을 넣을 때 같은 로직이 네 곳에 흩어져 있었고 `verify_pipeline.py`를
빠뜨려 **검증이 그 기능 없는 모델을 통과시켰다.** `ctxcheck.py`가 그 뒤 hand 축의
비트 동일성을 확인했고, 이 파일은 시즌 진행분 축에 같은 것을 한다.

  ① `features.asof_lookup`(배포) == `mixresid.build`의 gap_success(측정)  — 표준화 후
  ② 배포 아티팩트로 계산한 fold2024 BSS 이득 == §7.55의 β 스윕 값 (+23.52 @ β=0.02)
  ③ 행 독립성 — 조각내서 계산한 것과 통짜가 **비트 단위로** 같은가

실행:  .venv/Scripts/python.exe -u src/asofcheck.py 2>&1 | tee experiments/asofcheck.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import asofdelta
import ctxfit
import foldcache
import features as F
from config import (ASOF_AXES, ASOF_STRENGTH, CTX_AXES, CTX_K, MLP_ALPHA,
                    TARGET_COL, TRAIN_CSV)
from data import load_csv
from mixresid import build
from validation import bss, shift_to_prior

FOLD = 2024


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    tr, va = s < FOLD, s == FOLD
    d = df[va].copy()

    # ── 배포 경로 ─────────────────────────────────────────────────
    BETA = dict(ASOF_AXES)["gap_success"]
    asof = asofdelta.fit_asof_axes(df[tr], [("gap_success", BETA)],
                                   ASOF_STRENGTH)[0]
    art = {"ctx": [], "asof": [asof]}
    v_dep = F.asof_lookup(d, asof)

    # ── 측정 경로 (mixresid.build + 학습행 통계로 표준화) ──────────
    off = asofdelta.fit_offsets(df[tr])
    league = float(df.loc[tr, TARGET_COL].mean())
    ftr, fva = build(df[tr], off, league), build(d, off, league)
    mu, sd = float(np.nanmean(ftr["gap_success"])), float(np.nanstd(ftr["gap_success"]))
    v_mes = np.nan_to_num((fva["gap_success"] - mu) / sd)

    print("① 배포 경로 == 측정 경로")
    dmax = float(np.abs(v_dep - v_mes).max())
    print(f"   max|배포 − 측정| = {dmax:.3e}   {'✅' if dmax < 1e-9 else '❌'}")
    print(f"   배포 mu/sd {asof['mu']:+.6f} / {asof['sd']:.6f}"
          f"   측정 mu/sd {mu:+.6f} / {sd:.6f}")

    # ── ② BSS 이득이 β 스윕 값과 같은가 ───────────────────────────
    cached = foldcache.load([FOLD], verbose=False)
    yv = cached[FOLD][2]
    z = (1 - MLP_ALPHA) * cached[FOLD][0] + MLP_ALPHA * cached[FOLD][1]
    ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
    adj, _ = F.logit_adjust(d, {"ctx": ctx})            # δ만 (asof 없음) = 기준선
    z0 = z + adj
    base = bss(yv, shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean()))

    adj2, hits = F.logit_adjust(d, {"ctx": ctx, "asof": [asof]})   # δ + asof
    got = bss(yv, shift_to_prior(1.0 / (1.0 + np.exp(-(z + adj2))), yv.mean())) - base
    print(f"\n② fold{FOLD} 이득 (배포 아티팩트 경유)  {got:+.2f}")
    print(f"   §7.55 β 스윕 β={BETA} 값        +23.52")
    print(f"   차 {got - 23.52:+.2f}   {'✅' if abs(got - 23.52) < 0.05 else '❌'}")
    print(f"   기준 BSS {base:.1f}   asof 적중 {hits.get("gap_success", 0):.1%}"
          f"   로짓 기여 평균 {(adj2 - adj).mean():+.6f}")

    # ── ③ 행 독립성 ──────────────────────────────────────────────
    print("\n③ 행 독립성 — 조각 vs 통짜 (비트 동일해야 한다)")
    whole = F.asof_lookup(d, asof)
    piece = np.concatenate([F.asof_lookup(d.iloc[i:i + 1000], asof)
                            for i in range(0, len(d), 1000)])
    e1 = float(np.abs(whole - piece).max())
    order = np.random.default_rng(0).permutation(len(d))
    sh = F.asof_lookup(d.iloc[order], asof)
    back = np.empty_like(sh)
    back[order] = sh
    e2 = float(np.abs(whole - back).max())
    print(f"   조각 1000행씩 : {e1:.3e}   {'✅' if e1 == 0.0 else '❌'}")
    print(f"   행 순서 셔플  : {e2:.3e}   {'✅' if e2 == 0.0 else '❌'}")


if __name__ == "__main__":
    main()
