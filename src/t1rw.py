"""§7.174 — T1 의 **R 라우팅 가중 w** 를 스윕한다 (사용자 지적)

사용자: *"그 앙상블 비율을 조절하면 되는거 아니야?"* — 맞다. `predict.py:163` 의

    z[is_r] = 0.90 * z[is_r] + 0.10 * zg[is_r]

에서 0.10 이 곧 우리 일반 LGBM 의 몫이다. §7.173 이 w=0.10 고정에서 Δ +0.24 를
얻었는데, w 를 올리면 증강의 이득도 같이 커진다.

⚠️ **홀드아웃으로 w 자체를 고를 수는 없다.** T1 코어(혜진 LGBM/MLP·compact·
   user_fspec)는 2019~2024 를 학습해 2024 에서 **in-sample** 이고, 우리 일반 LGBM 은
   `<2024` 학습이라 **out-of-sample** 이다. 부풀려진 코어와 정직한 부품을 겨루면
   최적 w 가 인위적으로 **0 쪽으로 밀린다.** 그래서 A 팔의 w 곡선은 읽지 않는다.

   홀드아웃이 답할 수 있는 것은 **Δ(B−A) 가 w 에 따라 어떻게 커지는가** 뿐이다 —
   두 팔이 같은 코어를 쓰므로 in-sample 편향이 차분에서 상쇄된다.

   w 를 실제로 올릴지는 LB 프로브(§7.171 §4 슬롯 B)로만 정한다.

실행:  .venv/Scripts/python.exe -u src/t1rw.py 2>&1 | tee experiments/t1rw.log
"""
import os
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

WS = (0.00, 0.10, 0.20, 0.30, 0.45, 0.60)
CACHE = ROOT / "experiments" / "cache"

WORKER = r"""
import os, sys, numpy as np, pandas as pd
arm, w, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ["T1_RW"] = w
d = r"D:/t1eval/" + arm
# ⚠️ T1 경로만 sys.path 에 넣는다. 우리 src 를 같이 넣으면 `from predict import
#    Predictor` 가 **우리 predict.py** 를 잡는다 (1차 실행이 정확히 그래서 12칸이
#    전부 같은 수를 냈다 — 메모리 [assert-the-cache-key-actually-split]).
sys.path.insert(0, d)
va = pd.read_csv(r"{csv}")
va = va[va.season == 2024].reset_index(drop=True)
from predict import Predictor
import predict as _pm
assert _pm.__file__.replace("\\", "/").startswith(d), "T1 predict.py 가 아니다: " + _pm.__file__
assert "T1_RW" in open(_pm.__file__, encoding="utf-8").read(), "T1_RW 패치가 없다"
p = Predictor(model_dir=d + "/model")
np.savez_compressed(out, p=np.asarray(p.predict(va), dtype=np.float64))
"""


def main():
    wk = CACHE / "_t1rw_worker.py"
    wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/")),
                  encoding="utf-8", newline="\n")
    df = load_csv(TRAIN_CSV)
    va = df[df.season == 2024].reset_index(drop=True)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)

    P = {}
    for arm in ("A_base", "B_aug"):
        for w in WS:
            f = CACHE / f"t1rw_{arm}_{w:.2f}.npz"
            if not f.exists():
                # 별도 프로세스 — Predictor 가 모듈 전역을 잡아서 in-process 재로드가 불안하다
                r = subprocess.run([sys.executable, str(wk), arm, f"{w:.4f}", str(f)],
                                   capture_output=True, text=True)
                if not f.exists():
                    print(f"  ❌ {arm} w={w}: {r.stderr.strip()[-200:]}")
                    continue
                print(f"  {arm} w={w:.2f} 완료")
            P[(arm, w)] = np.load(f)["p"]

    # 배관 검증 — 1차 실행은 12칸이 전부 같은 수였다 (워커가 우리 predict.py 를
    # 잡았다). "여러 칸에 정확히 +0.0" 은 무효과가 아니라 **같은 것을 두 번 읽은**
    # 신호다 (메모리 [assert-the-cache-key-actually-split]).
    if ("A_base", 0.60) in P and ("A_base", 0.00) in P:
        dw = float(np.abs(P[("A_base", 0.60)] - P[("A_base", 0.00)]).max())
        assert dw > 0, f"w 가 예측을 안 바꾼다 — T1_RW 미적용 (max|Δ|={dw})"
        da = float(np.abs(P[("B_aug", 0.10)] - P[("A_base", 0.10)]).max())
        assert da > 0, f"두 팔이 같다 — 모델 교체 실패 (max|Δ|={da})"
        print(f"  배관 OK — w 변화 max|Δp|={dw:.6f} · 팔 차이 max|Δp|={da:.6f}")

    codes, uniq = pd.factorize(va["pitcher_id"].to_numpy())
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    seen = np.where(np.diff(bnd) > 0)[0]
    rng = np.random.default_rng(42)
    boots = [np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                             rng.choice(seen, size=len(seen), replace=True)])
             for _ in range(300)]

    def s_(p, idx=None):
        pp, yy = (p, yva) if idx is None else (p[idx], yva[idx])
        return bss(yy, shift_to_prior(pp, yy.mean()))

    print("\n" + "=" * 86)
    print("R 라우팅 가중 w 스윕 · fold2024")
    print("⚠️ A 의 절대 곡선은 읽지 않는다 (코어 in-sample, 부품 out-of-sample → w 를 0 쪽으로 민다)")
    print("   읽을 것은 Δ(B−A) 가 w 에 따라 어떻게 커지는가 뿐이다.")
    print("=" * 86)
    print("  %6s %10s %10s %9s %22s %8s" % ("w", "A", "B", "Δ(B−A)", "95% 구간", "P(Δ>0)"))
    for w in WS:
        if ("A_base", w) not in P or ("B_aug", w) not in P:
            continue
        a, b = P[("A_base", w)], P[("B_aug", w)]
        d = np.array([s_(b, ix) - s_(a, ix) for ix in boots])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print("  %6.2f %10.1f %10.1f %+9.2f   [%+7.2f, %+7.2f] %8.3f"
              % (w, s_(a), s_(b), s_(b) - s_(a), lo, hi, (d > 0).mean()))
    print("\n  (w=0.00 은 일반 LGBM 을 완전히 끈 것 — Δ 가 정확히 0 이어야 배관이 맞다)")


if __name__ == "__main__":
    main()
