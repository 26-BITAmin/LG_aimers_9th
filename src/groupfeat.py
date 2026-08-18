"""§7.69 — 선수 3분할 축 (`R전용`/`둘다`/`F전용`). 사용자 제안.

EDA(notebooks/player_stats)가 이미 보여준 것:
    · F−R 성공률 격차의 30%는 구성 효과다 — F에만 나오는 투수가 다른 투수라서.
    · 정상 구간(2023–24)에서 같은 투수의 F/R 짝지은 Δ는 ±.02 안 — 경기 유형
      자체의 효과는 거의 없다.
→ 즉 이 축의 정보는 전부 **투수 정체성**이다. 그런데 모델은 raw `pitcher_id`와
  `asof_pitcher_*`를 이미 갖고 있다. 3그룹은 pitcher_id의 거친 버전(coarsening)이라
  §7.68(start_rate)보다 선험적으로 더 불리하다. 잔차로 판정한다.

후보 (전부 train에서 얼리고 검증 행은 조회만)
    f_share      그 투수의 과거 행 중 F 비율 (연속형 — 3분할의 원천)
    is_F_only    f_share == 1
    is_R_only    f_share == 0
    is_both      0 < f_share < 1
    unknown      과거 이력 없음 (신인)
    gt_F_x_share 현재 행이 F × f_share (구성 효과 상호작용)

리키지: fold 이전 시즌 행으로만 그룹을 계산한다. 2025 추론이라면
       train 전체로 fit → pitcher_id lookup과 같은 구조. 현재 투구의
       결과·test 다른 행은 쓰지 않는다.

실행:  .venv/Scripts/python.exe -u src/groupfeat.py 2>&1 | tee experiments/groupfeat.log
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
import features as F
import foldcache
from config import (ASOF_AXES, ASOF_STRENGTH, CONTAM_AXES, CTX_AXES, CTX_HALF_LIFE,
                    CTX_K, MLP_ALPHA, SEED, TARGET_COL, TRAIN_CSV)
from ctxsplit import resid_t
from data import load_csv
from data import recency_weight
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]


def fit_group(tr):
    """투수별 F 행 비율. **train에서만.**"""
    pid = tr["pitcher_id"].to_numpy(np.int64)
    is_f = (tr["game_type"].astype("object").to_numpy() == "F").astype(np.float64)
    u, inv = np.unique(pid, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    f = np.bincount(inv, weights=is_f)
    return {"keys": u, "f_share": f / n}


def apply_group(d, g):
    pid = d["pitcher_id"].to_numpy(np.int64)
    j = np.clip(np.searchsorted(g["keys"], pid), 0, len(g["keys"]) - 1)
    hit = g["keys"][j] == pid
    fs = np.where(hit, g["f_share"][j], np.nan)
    gt_f = (d["game_type"].astype("object").to_numpy() == "F").astype(np.float64)
    fs0 = np.where(hit, fs, 0.0)          # 회귀용: 미등록은 0으로 두고 unknown 지표가 흡수
    return {
        "f_share":      fs0,
        "is_F_only":    np.where(hit, (fs == 1.0).astype(np.float64), 0.0),
        "is_R_only":    np.where(hit, (fs == 0.0).astype(np.float64), 0.0),
        "is_both":      np.where(hit, ((fs > 0) & (fs < 1)).astype(np.float64), 0.0),
        "unknown":      (~hit).astype(np.float64),
        "gt_F_x_share": gt_f * fs0,
    }, hit


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

        g = fit_group(tr)
        feat, hit = apply_group(d, g)
        dsh = d.copy()
        dsh["pitcher_id"] = rng.permutation(d["pitcher_id"].to_numpy())
        fsh, _ = apply_group(dsh, g)

        print()
        print(f"fold {fold}  기준 BSS {bss(yv, p0):.1f}  행 {len(d):,}  "
              f"투수 조회 적중 {hit.mean()*100:.1f}%")

        # 그룹별 잔차 평균 — 해석용 (t와 별개)
        fs = feat["f_share"]
        grp = np.where(feat["unknown"] == 1, "신인",
              np.where(feat["is_F_only"] == 1, "F전용",
              np.where(feat["is_R_only"] == 1, "R전용", "둘다")))
        print(f"  {'그룹':<6}{'행':>10}{'잔차 평균':>12}")
        for name in ["R전용", "둘다", "F전용", "신인"]:
            m = grp == name
            if m.sum() == 0:
                continue
            print(f"  {name:<6}{m.sum():>10,}{r[m].mean():>+12.5f}")

        print(f"  {'피처':<14}{'잔차 t':>10}{'대조군 t':>11}{'차':>9}"
              f"{'배포피처 최대상관':>18}")
        X = F.build_features(d, art)
        for k, v in feat.items():
            _, t = resid_t(r, v)
            _, tc = resid_t(r, fsh[k])
            cc, cn = 0.0, ""
            for c in X.columns:
                xv = X[c].to_numpy(np.float64)
                m = np.isfinite(xv)
                if m.sum() < 1000 or np.nanstd(xv[m]) == 0:
                    continue
                cr = abs(float(np.corrcoef(v[m], xv[m])[0, 1]))
                if cr > cc:
                    cc, cn = cr, c
            mark = "  ✓" if abs(t) > 2 and abs(t) > 2 * abs(tc) else ""
            print(f"  {k:<14}{t:>10.2f}{tc:>11.2f}{t-tc:>9.2f}"
                  f"{cc:>10.3f} {cn[:16]:<16}{mark}")


if __name__ == "__main__":
    main()
