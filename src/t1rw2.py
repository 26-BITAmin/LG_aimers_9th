"""§7.174b — **출하본** general LGBM 으로 R 라우팅 가중 w 스윕 (w=0.10 이 최적인가)

§7.174 는 우리가 `<2024` 로 재학습한 general LGBM 을 썼다. 그러면 **in-sample 코어 vs
out-of-sample 부품**을 겨루게 돼 최적 w 가 인위적으로 0 쪽으로 밀린다 (실제로
A 곡선이 1717 → 1398 로 폭락했다). 그 곡선은 읽을 수 없다.

여기서는 **T1 출하본의 general LGBM 12** 를 쓴다. 그것도 2019~2024 전 시즌 학습이라
코어와 **조건이 같다** — 둘 다 in-sample. 그러면 절대 점수는 여전히 부풀려져 있지만
**w 축의 모양**은 편향이 상쇄돼 읽을 수 있다.

    z[is_r] = (1−w)·core + w·zg          w = 0.10 이 현재 배포값

⚠️ 그래도 완전히 깨끗하지는 않다 — 두 팔 다 2024 를 봤으므로 w 를 올리면
   "더 많이 외운 쪽" 이 이길 수 있다. 곡선이 **평평하거나 0.10 근처에서 꺾이면**
   신뢰하고, **단조 증가하면** 과적합 신호로 읽고 LB 로 넘긴다.

읽는 법
    w* ≈ 0.10       → 이미 최적. 증강 축은 여기서 닫힌다.
    w* > 0.10       → w 를 올리는 것 자체가 이득(슬롯 B)이고, §7.174 표대로
                      그 위에서 증강 가치도 비례해 커진다. 둘이 곱해진다.
    단조 증가        → 과적합 신호. LB 프로브로만 답한다.

실행:  .venv/Scripts/python.exe -u src/t1rw2.py 2>&1 | tee experiments/t1rw2.log
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
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

WS = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60)
ARM = "T1_orig"
CACHE = ROOT / "experiments" / "cache"

WORKER = r"""
import os, sys, numpy as np, pandas as pd
arm, w, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ["T1_RW"] = w
d = r"D:/t1eval/" + arm
sys.path.insert(0, d)          # ⚠️ 우리 src 를 넣지 않는다 (§7.174 1차 사고)
va = pd.read_csv(r"{csv}")
va = va[va.season == 2024].reset_index(drop=True)
from predict import Predictor
import predict as _pm
assert _pm.__file__.replace("\\", "/").startswith(d), "T1 predict.py 가 아니다"
assert "T1_RW" in open(_pm.__file__, encoding="utf-8").read(), "T1_RW 패치 없음"
p = Predictor(model_dir=d + "/model")
np.savez_compressed(out, p=np.asarray(p.predict(va), dtype=np.float64))
"""


def main():
    wk = CACHE / "_t1rw2_worker.py"
    wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/")),
                  encoding="utf-8", newline="\n")
    df = load_csv(TRAIN_CSV)
    va = df[df.season == 2024].reset_index(drop=True)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    isF = va["game_type"].to_numpy().astype(str) == "F"

    P = {}
    for w in WS:
        f = CACHE / f"t1rw2_{ARM}_{w:.2f}.npz"
        if not f.exists():
            r = subprocess.run([sys.executable, str(wk), ARM, f"{w:.4f}", str(f)],
                               capture_output=True, text=True)
            if not f.exists():
                print(f"  ❌ w={w}: {r.stderr.strip()[-250:]}")
                continue
            print(f"  w={w:.2f} 완료")
        P[w] = np.load(f)["p"]

    assert len(P) >= 2
    dw = float(np.abs(P[max(P)] - P[min(P)]).max())
    assert dw > 0, f"w 가 예측을 안 바꾼다 (max|Δp|={dw})"
    print(f"  배관 OK — w 변화 max|Δp|={dw:.6f}")

    codes, uniq = pd.factorize(va["pitcher_id"].to_numpy())
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    seen = np.where(np.diff(bnd) > 0)[0]
    rng = np.random.default_rng(42)
    boots = [np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                             rng.choice(seen, size=len(seen), replace=True)])
             for _ in range(300)]

    def s_(p, m=None, idx=None):
        pp, yy = (p, yva) if m is None else (p[m], yva[m])
        if idx is not None:
            pp, yy = pp[idx], yy[idx]
        return bss(yy, shift_to_prior(pp, yy.mean()))

    base = 0.10
    print("\n" + "=" * 88)
    print("R 라우팅 가중 w 스윕 · **출하본** general LGBM (코어와 같은 in-sample 조건)")
    print("=" * 88)
    print("  %6s %10s %10s %22s %8s %11s"
          % ("w", "전체", "Δ vs .10", "95% 구간", "P(Δ>0)", "R행만"))
    for w in WS:
        if w not in P:
            continue
        d = np.array([s_(P[w], idx=ix) - s_(P[base], idx=ix) for ix in boots])
        lo, hi = np.percentile(d, [2.5, 97.5])
        mark = "  ← 현재" if w == base else ""
        print("  %6.2f %10.1f %10.2f   [%+7.2f, %+7.2f] %8.3f %11.1f%s"
              % (w, s_(P[w]), s_(P[w]) - s_(P[base]), lo, hi,
                 (d > 0).mean() if w != base else 0.5, s_(P[w], ~isF), mark))
    best = max(P, key=lambda w: s_(P[w]))
    print(f"\n  oracle 최적 w = {best:.2f}  (현재 {base:.2f})")
    print("  ⚠️ 단조 증가면 과적합 신호다 — 두 팔 다 2024 를 봤다. LB 로 넘긴다.")


if __name__ == "__main__":
    main()
