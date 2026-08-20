"""오염 보정이 **배포 스택**(블렌드 + δ 축)에서도 남는가 (§7.49 마무리)

§7.49의 +30.6은 **LGBM 단독**(base 740.2) 측정이다. 배포는 블렌드(+MLP 8개) 위에
δ 축 2개를 얹어 이미 799.4다. MLP나 δ가 같은 신호를 이미 먹고 있으면 이득이 줄어든다.
(§7.9의 Trackman이 정확히 그렇게 죽었다 — LGBM 단독 시절의 +13.3이 스택이 두꺼워진 뒤
잔차에서 사라졌다.)

두 캐시(base 지문 / contam 지문)를 나란히 읽어 같은 절차로 비교한다.

실행:  .venv/Scripts/python.exe -u src/fcontam3.py 2>&1 | tee experiments/fcontam3.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import ctxfit
import foldcache
from config import CTX_AXES, CTX_K, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from features import logit_adjust
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]


# base 캐시는 `_key()`에 CONTAM_AXES를 넣기 **전에** 구워졌다. 지금 None으로 되돌려
# 계산해도 지문이 달라져 30분을 다시 태운다. 이미 있는 파일을 지문으로 직접 가리킨다.
BASE_KEY = "c3d43f1a57"
CONTAM_KEY = "64aca93260"


def load_with(key):
    """지문을 고정해 해당 캐시를 읽는다. 없으면 실패시킨다 (재계산 금지)."""
    for f in FOLDS:
        p = foldcache.CACHE / f"fold{f}_{key}.npz"
        assert p.exists(), f"캐시 없음: {p.name}"
    old = foldcache._key
    foldcache._key = lambda: key
    try:
        return foldcache.load(FOLDS, verbose=False), key
    finally:
        foldcache._key = old


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()

    base, kb = load_with(BASE_KEY)
    corr, kc = load_with(CONTAM_KEY)
    print(f"base 지문 {kb}   contam 지문 {kc}\n")

    print(f"  {'fold':<8}{'구성':<22}{'base':>10}{'보정':>10}{'차이':>10}")
    for fold in FOLDS:
        tr, va = s < fold, s == fold
        d = df[va]
        yv = base[fold][2]
        assert np.array_equal(yv, corr[fold][2]), "두 캐시의 검증 행이 다르다"
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        adj, _ = logit_adjust(d, {"ctx": ctx})

        def sc(z):
            return bss(yv, shift_to_prior(1.0 / (1.0 + np.exp(-z)), yv.mean()))

        rows = [
            ("LGBM 단독", lambda c: c[0]),
            ("MLP 단독", lambda c: c[1]),
            (f"블렌드 α={MLP_ALPHA}", lambda c: (1 - MLP_ALPHA) * c[0] + MLP_ALPHA * c[1]),
            ("블렌드 + δ (배포)",
             lambda c: (1 - MLP_ALPHA) * c[0] + MLP_ALPHA * c[1] + adj),
        ]
        for nm, f in rows:
            b, g = sc(f(base[fold])), sc(f(corr[fold]))
            print(f"  {fold:<8}{nm:<22}{b:>10.1f}{g:>10.1f}{g-b:>+10.1f}")

        # 배포 구성의 쌍체 SE
        zb = (1 - MLP_ALPHA) * base[fold][0] + MLP_ALPHA * base[fold][1] + adj
        zc = (1 - MLP_ALPHA) * corr[fold][0] + MLP_ALPHA * corr[fold][1] + adj
        pb = shift_to_prior(1.0 / (1.0 + np.exp(-zb)), yv.mean())
        pc = shift_to_prior(1.0 / (1.0 + np.exp(-zc)), yv.mean())
        dd = (pb - yv) ** 2 - (pc - yv) ** 2
        se = dd.std(ddof=1) / np.sqrt(len(dd))
        bsref = yv.mean() * (1 - yv.mean())
        print(f"  {'':8}{'└ 쌍체':<22}{'':>10}{'':>10}"
              f"{1e5*dd.mean()/bsref:>+10.1f} ± {1e5*se/bsref:.1f}"
              f"   t {dd.mean()/se:+.2f}\n")


if __name__ == "__main__":
    main()
