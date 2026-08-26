"""§7.173 (2단계) — 학습된 A/B 를 **ASCII 경로**로 옮겨 T1 전체 경로 예측·대조

`t1gen.py` 는 24개 학습을 마치고 예측에서 죽었다:

    LightGBMError: Could not open D:\\projects\\LG aimers 야구\\t1work_A_base\\...

메모리 [lightgbm-windows-nonascii-path] 그대로다 — `lgb.Booster(model_file=)` 이
Windows 에서 경로를 CP949 로 오해한다. T1 의 `predict.py:47` 이 `model_file=` 을
쓰므로 **저장소가 한글 경로에 있는 한 T1 예측기는 여기서 못 돈다.**

우회: 두 팔을 ASCII 경로로 복사하고 거기서 돌린다. 코드는 손대지 않는다 —
T1 의 진짜 경로를 그대로 태우는 것이 이 실험의 요점이기 때문이다.

실행:  .venv/Scripts/python.exe -u src/t1gen2.py 2>&1 | tee experiments/t1gen2.log
"""
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
ASCII = Path("D:/t1eval")
sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

from config import TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

VALID = 2024
ARMS = ("A_base", "B_aug")


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    va = df[df.season == VALID].reset_index(drop=True)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    print(f"valid {len(va):,}  ({time.time()-t0:.0f}s)")

    P = {}
    for arm in ARMS:
        dst = ASCII / arm
        if not (dst / "model" / "general_lgb_lowff_2024.txt").exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(ROOT / f"t1work_{arm}", dst, dirs_exist_ok=True)
            print(f"  {arm} → {dst}  복사 ({time.time()-t0:.0f}s)")
        # 모듈 캐시를 비우고 그 팔의 경로에서 다시 import 한다
        for m in [k for k in list(sys.modules)
                  if k in ("predict", "features", "mlp_infer", "contam")
                  or k.startswith("team_reference")]:
            sys.modules.pop(m, None)
        sys.path.insert(0, str(dst))
        from predict import Predictor
        pr = Predictor(model_dir=str(dst / "model"))
        P[arm] = np.asarray(pr.predict(va), dtype=np.float64)
        sys.path.pop(0)
        del pr, Predictor
        print(f"  {arm} 예측 mean={P[arm].mean():.4f} std={P[arm].std():.4f}"
              f"  ({time.time()-t0:.0f}s)")

    # 두 팔이 실제로 갈렸는지 — 정확히 0 이면 같은 모델을 두 번 읽은 것이다
    dmax = float(np.abs(P["B_aug"] - P["A_base"]).max())
    assert dmax > 0, "두 팔의 예측이 완전히 같다 — 모델 교체가 안 됐다"
    print(f"  두 팔 최대 예측차 {dmax:.6f}  (0 이면 교체 실패)")

    codes, uniq = pd.factorize(va["pitcher_id"].to_numpy())
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    seen = np.where(np.diff(bnd) > 0)[0]
    rng = np.random.default_rng(42)

    def s_(p, yy, idx=None):
        if idx is not None:
            p, yy = p[idx], yy[idx]
        return bss(yy, shift_to_prior(p, yy.mean()))

    isF = va["game_type"].to_numpy().astype(str) == "F"
    d = []
    for _ in range(400):
        ix = np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                             rng.choice(seen, size=len(seen), replace=True)])
        d.append(s_(P["B_aug"], yva, ix) - s_(P["A_base"], yva, ix))
    lo, hi = np.percentile(d, [2.5, 97.5])
    a, b = s_(P["A_base"], yva), s_(P["B_aug"], yva)

    print("\n" + "=" * 80)
    print("T1 전체 경로 · fold2024  ⚠️ 절대값은 in-sample 오염으로 무의미, 차이만 읽는다")
    print("=" * 80)
    print(f"  A 증강 없음   BSS {a:9.1f}")
    print(f"  B 증강        BSS {b:9.1f}")
    print(f"  Δ(B−A)        {b-a:+9.2f}   95% [{lo:+.2f}, {hi:+.2f}]"
          f"   P(Δ>0)={np.mean(np.array(d) > 0):.3f}")
    print(f"  R행만  A {s_(P['A_base'][~isF], yva[~isF]):.1f} → "
          f"B {s_(P['B_aug'][~isF], yva[~isF]):.1f}   "
          f"(일반 LGBM 은 R행에만 10% 들어간다)")
    print(f"  F행만  A {s_(P['A_base'][isF], yva[isF]):.1f} → "
          f"B {s_(P['B_aug'][isF], yva[isF]):.1f}   (변화 없어야 정상)")
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
