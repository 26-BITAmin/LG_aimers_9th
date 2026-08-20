"""§7.68 — 투수 **역할**(선발/불펜) 축. 사용자 제안 `start_rate`.

`game_id`가 없어 등판 단위를 셀 수 없다. 대신 `inning`으로 대리한다 —
**1회를 던지는 것은 사실상 선발뿐이다**(오프너 예외).

후보
    start_rate   그 투수의 투구 중 1회 비중            ← 제안된 것
    mean_inning  그 투수가 던지는 이닝의 평균
    inning_dev   **이 행의 이닝 − 그 투수의 평균 이닝**
                 선발이 8회에 남아 있음 / 불펜이 3회에 나옴 = 둘 다 비정상 상황.
                 역할 자체보다 **역할에서 벗어난 정도**가 새 정보일 수 있다.
    inning_z     같은 것을 투수별 sd로 표준화

⚠️ 먼저 물어야 할 것 두 가지 (§7.59의 대가로 배운 것)
    ① 모델이 이미 갖고 있나 — `inning`은 피처이고 `pitcher_id`도 피처다.
       역할의 **주효과**는 `asof_pitcher_success_rate`(그 투수의 제구 수준)에
       이미 흡수돼 있을 가능성이 높다.
    ② 기존 피처와의 상관이 0.5를 넘으면 새 정보가 아니라 재가중이다.

리키지: train에서 투수별 이닝 분포를 얼리고 test에서 조회만 한다.
       현재 투구의 결과를 안 쓰고, test 행끼리 아무것도 공유하지 않는다.

실행:  .venv/Scripts/python.exe -u src/rolefeat.py 2>&1 | tee experiments/rolefeat.log
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
from data import load_csv, recency_weight
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]


def fit_role(tr):
    """투수별 이닝 분포를 얼린다. **train에서만.**"""
    pid = tr["pitcher_id"].to_numpy(np.int64)
    inn = np.clip(tr["inning"].to_numpy(np.float64), 1, 12)
    u, inv = np.unique(pid, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    s1 = np.bincount(inv, weights=(inn == 1).astype(np.float64))
    si = np.bincount(inv, weights=inn)
    si2 = np.bincount(inv, weights=inn * inn)
    mean = si / n
    var = np.maximum(si2 / n - mean ** 2, 0.0)
    return {"keys": u, "n": n, "start": s1 / n, "mean": mean,
            "sd": np.sqrt(var), "lg_mean": float(inn.mean())}


def apply_role(d, r):
    """행 단위 조회뿐. 미등록 투수는 리그 평균으로."""
    pid = d["pitcher_id"].to_numpy(np.int64)
    inn = np.clip(d["inning"].to_numpy(np.float64), 1, 12)
    j = np.clip(np.searchsorted(r["keys"], pid), 0, len(r["keys"]) - 1)
    hit = r["keys"][j] == pid
    start = np.where(hit, r["start"][j], 0.0)
    mean = np.where(hit, r["mean"][j], r["lg_mean"])
    sd = np.where(hit, r["sd"][j], 1.0)
    return {"start_rate": start,
            "mean_inning": mean,
            "inning_dev": inn - mean,
            "inning_z": (inn - mean) / np.maximum(sd, 0.5)}, hit


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

        role = fit_role(tr)
        feat, hit = apply_role(d, role)
        dsh = d.copy()
        dsh["pitcher_id"] = rng.permutation(d["pitcher_id"].to_numpy())
        fsh, _ = apply_role(dsh, role)

        print()
        print(f"fold {fold}  기준 BSS {bss(yv, p0):.1f}  행 {len(d):,}  "
              f"투수 조회 적중 {hit.mean()*100:.1f}%")
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
