"""챔피언 `hj03` 의 **실제** 사후 보정·레벨 규격 (§7.210).

전부 `hj03_dim8_cold.zip` 안에서 직접 읽은 값이다 (`src/hjstruct.py` 가 뽑는다).
`meta.json` 도 `config.py` 도 진실이 아니다 — **`artifacts.joblib` 이 진실원**이다.

    art['ctx']   hand=1.054   count4=1.25
                 ⚠️ meta.json 은 1.5, 우리 config 는 1.875 — **셋이 전부 다르다**
    art['asof']  gap_success=0.03  gap_bsuccess=0.03  **gap_ball=-0.0038**
                 ⚠️ `config.ASOF_AXES` 에 gap_ball 이 없다 (§7.204 가 경고한 그대로)
    predict.py   p = PROB_CENTER + PROB_SCALE*(p - PROB_CENTER) + PROB_SHIFT
                 PROB_CENTER=0.4801  PROB_SCALE=1.05

`gap_ball` 은 `ASOF_SPEC` 에 없어 명세를 주입해야 한다 (`gapstrike.EXTRA_SPEC` 재사용).
그리고 hj03 의 `gap_ball` 항목에는 `cadj` 키가 **없다** — 오염 보정을 안 건다.
`asofdelta.CONTAM_OF` 에도 gap_ball 이 없으므로 그대로 두면 일치한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import asofdelta
import ctxfit
import features as F
from config import CONTAM_AXES, CTX_HALF_LIFE, CTX_K, TARGET_COL
from data import recency_weight
from gapstrike import EXTRA_SPEC

# ── 챔피언 실측 규격 ────────────────────────────────────────────
CHAMP_CTX = [("hand", 1.054), ("count4", 1.25)]
CHAMP_ASOF = [("gap_success", 0.03), ("gap_bsuccess", 0.03), ("gap_ball", -0.0038)]
CHAMP_ASOF_STRENGTH = 35.0
PROB_CENTER = 0.4801
PROB_SCALE = 1.05

for _k, _v in EXTRA_SPEC.items():
    asofdelta.ASOF_SPEC.setdefault(_k, _v)


def axis_matrix(df, s, fold, ctx_axes=None, asof_axes=None):
    """fold 검증 행 × 5축 원단위 값 + β 벡터. `jointbeta.axis_matrix` 의 챔피언판.

    ⚠️ β 만 챔피언 값을 쓰고, **δ 표 자체는 시즌 < fold 로 재적합**한다.
       배포 표를 그대로 쓰면 2024 타깃이 fold2024 검증에 들어간다.
    """
    ctx_axes = CHAMP_CTX if ctx_axes is None else ctx_axes
    asof_axes = CHAMP_ASOF if asof_axes is None else asof_axes
    tr_m, va_m = s < fold, s == fold
    tr, va = df[tr_m], df[va_m].copy()
    ref = int(s[tr_m].max())
    w = recency_weight(s[tr_m], CTX_HALF_LIFE, ref_season=ref)

    ctx = ctxfit.fit_ctx(tr, TARGET_COL, ctx_axes, CTX_K, weights=w)
    art = F.fit_artifacts(tr, ctx=ctx, contam_axes=CONTAM_AXES)
    art["asof"] = asofdelta.fit_asof_axes(tr, asof_axes, CHAMP_ASOF_STRENGTH,
                                          ct=art.get("contam"))
    cols, names, betas = [], [], []
    for ax in ctx:
        cols.append(F.ctx_lookup(va, ax))
        names.append(ax["name"])
        betas.append(float(ax["beta"]))
    for a in art["asof"]:
        cols.append(F.asof_lookup(va, a, ct=art.get("contam")))
        names.append(a["name"])
        betas.append(float(a["beta"]))
    return np.column_stack(cols), names, np.array(betas), va


def prob_transform(p, shift=0.0):
    """챔피언의 확률 아핀 변환. `predict.py` 마지막 3줄과 같다."""
    return PROB_CENTER + PROB_SCALE * (p - PROB_CENTER) + shift


def solve_shift(p_raw, target, lo=-0.5, hi=0.5, it=200):
    """아핀 변환 **후** 평균이 target 이 되는 shift 를 푼다 (PROB_SHIFT 의 역할)."""
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if prob_transform(p_raw, mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
