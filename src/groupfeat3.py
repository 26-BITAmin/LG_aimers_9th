"""§7.69 후속 2 — fold2024의 잔여 −0.02는 「보정이 덜 걷은 오염」인가.

기존 오염 보정(contam.py)은 행마다 노출량 D/m(그 투수의 과거 이력 중 오염
투구가 남긴 결손의 비율)을 이미 계산한다. 이걸 그대로 진단 축으로 쓴다:

    · 잔차가 D/m에 비례  → 보정 확장(라벨 쪽 or 노출 δ)이 맞는 처방.
      단, 이 성분은 recency 희석으로 계속 감쇠한다.
    · D/m = 0인 행(오염 이력이 없는 투수)에서도 그룹/F행 편향이 남는다
      → 오염과 무관한 구조 효과. game_type을 드랍한 모델이 표현 못 하는
      F 행 레벨일 가능성. 이쪽은 감쇠하지 않는다.

실행:  .venv/Scripts/python.exe -u src/groupfeat3.py 2>&1 | tee experiments/groupfeat3.log
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
import contam
import ctxfit
import features as F
import foldcache
from config import (ASOF_AXES, ASOF_STRENGTH, CONTAM_AXES, CTX_AXES, CTX_HALF_LIFE,
                    CTX_K, MLP_ALPHA, SEED, TARGET_COL, TRAIN_CSV)
from ctxsplit import resid_t
from data import load_csv, recency_weight
from groupfeat import apply_group, fit_group
from groupfeat2 import GROUP_NAMES, group_codes
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(SEED)

    for fold in FOLDS:
        tr_m, va = s < fold, s == fold
        tr, d = df[tr_m], df[va].copy()
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ref = int(s[tr_m].max())
        ctx = ctxfit.fit_ctx(tr, TARGET_COL, CTX_AXES, CTX_K,
                             weights=recency_weight(s[tr_m], CTX_HALF_LIFE, ref_season=ref))
        art = F.fit_artifacts(tr, ctx=ctx, contam_axes=CONTAM_AXES)
        art["asof"] = asofdelta.fit_asof_axes(tr, ASOF_AXES, ASOF_STRENGTH,
                                              ct=art.get("contam"))
        adj, _ = F.logit_adjust(d, art)
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-(z + adj))), yv.mean())
        r = yv - p0

        expo = contam.adjust(d, art["contam"], "p")     # 행별 D/m (투수 축)
        g = fit_group(tr)
        feat, _ = apply_group(d, g)
        code = group_codes(feat)
        gt_f = d["game_type"].astype("object").to_numpy() == "F"

        print()
        print(f"fold {fold}  기준 BSS {bss(yv, p0):.1f}  "
              f"D/m>0 행 비율 {(expo > 0).mean()*100:.1f}%  "
              f"D/m 중앙값(>0 행) {np.median(expo[expo > 0]):.4f}")

        # ① 잔차 ~ 노출량 회귀 (셔플 대조군: 투수 단위로 노출을 섞는다)
        pid = d["pitcher_id"].to_numpy(np.int64)
        u, inv = np.unique(pid, return_inverse=True)
        perm = rng.permutation(len(u))
        # 투수별 평균 노출을 섞어 재배포 — 행 구조는 유지, 투수↔노출 연결만 끊는다
        pe = np.bincount(inv, weights=expo) / np.bincount(inv)
        expo_sh = pe[perm][inv]
        sl, t = resid_t(r, expo)
        _, tc = resid_t(r, expo_sh)
        print(f"  잔차~D/m:  기울기 {sl:+.3f}  t {t:+.2f}   (대조군 t {tc:+.2f})")

        # ② 그룹 × 노출 유무 — D/m=0에서도 편향이 남는가
        print(f"  {'그룹':<6}{'노출':<7}{'행':>10}{'잔차 평균':>12}")
        for gidx, name in enumerate(GROUP_NAMES):
            for has, lab in [(False, "D=0"), (True, "D>0")]:
                m = (code == gidx) & ((expo > 0) == has)
                if m.sum() < 300:
                    continue
                print(f"  {name:<6}{lab:<7}{m.sum():>10,}{r[m].mean():>+12.5f}")

        # ③ 행 유형 × 노출 유무 — F행 레벨은 오염 이력 없이도 남는가
        print(f"  {'행':<6}{'노출':<7}{'행':>10}{'잔차 평균':>12}")
        for isf, gname in [(False, "R행"), (True, "F행")]:
            for has, lab in [(False, "D=0"), (True, "D>0")]:
                m = (gt_f == isf) & ((expo > 0) == has)
                if m.sum() < 300:
                    continue
                print(f"  {gname:<6}{lab:<7}{m.sum():>10,}{r[m].mean():>+12.5f}")

        # ④ F전용 그룹 내부에서 잔차가 노출량에 비례하는가
        m = code == 2
        if m.sum() > 500:
            sl, t = resid_t(r[m], expo[m])
            print(f"  F전용 내부 잔차~D/m:  기울기 {sl:+.3f}  t {t:+.2f}  (행 {m.sum():,})")


if __name__ == "__main__":
    main()
