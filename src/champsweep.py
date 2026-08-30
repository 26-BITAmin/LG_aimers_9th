"""§7.207 — **챔피언(hj03) 층 위에서** 과거 후보 축을 전부 다시 쓸어본다.

사용자 지시: *"방금 그 챔피언을 바탕으로, soft_label2 제외하고 이전에 실험해봤던 것들을
다시 다 해보라."*

왜 지금까지 못 했나 — 과거 판정의 층이 제각각이었다.
    §7.57 gap_*      : 블렌드 + δ2축 (s15 시절)
    §7.204 gap_strike: 우리 fold 배포등가층 (s29 계보)
    §7.59 r3_ent     : s18 시절
    챔피언 hj03      : T1 계보 + dim8 + 콜드패치 — **한 번도 이 축들을 안 재봤다**

이 스크립트가 하는 것
    ① hj03 의 최종 확률을 2024 행에서 받아 **아핀을 역산해 z(로짓)** 를 복원한다.
       p = C + S·(σ(z) − C) + SHIFT   →   σ(z) = (p − C − SHIFT)/S + C
       역산 정확도를 재적용으로 검증한다 (max|Δ| < 1e-12).
    ② 후보 축을 β 로 z 에 더하고 같은 아핀을 다시 걸어 **쌍체 ΔBSS** 를 잰다.
    ③ 투수 클러스터 부트스트랩 구간 + 엔티티 순열 위약을 같이 낸다.

⚠️ **2024 는 챔피언에게 in-sample 이다** (전 시즌 학습). 절대 BSS 는 무의미하고,
   잔차가 이미 줄어 있어 **잔차 교정형 축의 효과는 0 쪽으로 편향**된다 (§7.174).
   즉 여기서 나오는 양수는 **하한**으로 읽는다. 음수는 강한 증거가 아니다.
   같은 성질을 안고도 §7.181 은 이 층에서 +1.72 를 재 LB +1.02 를 맞췄다.

실행: .venv/Scripts/python.exe -u src/champsweep.py 2>&1 | tee experiments/champsweep.log
"""
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

import asofdelta
from config import ROOT, TARGET_COL, TRAIN_CSV
from data import load_csv
from gapstrike import EXTRA_SPEC
from mixresid import GAP_SPEC, build
from validation import bss

ASCII = Path("D:/t1eval")
ZIP = ROOT / "hj03_dim8_cold.zip"
DST = ASCII / "HJ03"
CACHE = ROOT / "experiments" / "cache"
PC, PS, PSH = 0.4801, 1.05, 0.000172669          # hj03/predict.py 실측값
VALID = 2024
B_BOOT = 1000

WORKER = r"""
import sys, numpy as np, pandas as pd
d, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, d)
va = pd.read_csv(r"{csv}")
va = va[va.season == {vs}].reset_index(drop=True)
from predict import Predictor
import predict as _pm
assert _pm.__file__.replace("\\", "/").startswith(d), "챔피언 predict.py 가 아니다"
p = Predictor(model_dir=d + "/model")
np.savez_compressed(out, p=np.asarray(p.predict(va), dtype=np.float64))
"""

# 후보 축. 배포에 이미 있는 것은 **증분 β**(추가분)로 읽는다.
PROBED = ["gap_strike", "gap_middle", "gap_bmiddle", "gap_ball", "gap_reverse",
          "gap_success", "gap_bsuccess"]
BETAS = [-0.03, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.03]


def champion_z():
    f = CACHE / "champ_hj03_p2024.npz"
    if not f.exists():
        if not (DST / "script.py").exists():
            DST.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(ZIP) as z:
                z.extractall(DST)
        wk = CACHE / "_champ_worker.py"
        wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/"))
                            .replace("{vs}", str(VALID)),
                      encoding="utf-8", newline="\n")
        r = subprocess.run([sys.executable, str(wk), DST.as_posix(), str(f)],
                           capture_output=True, text=True)
        if not f.exists():
            print(r.stdout[-800:], r.stderr[-1200:])
            sys.exit(1)
    return np.load(f)["p"]


def affine(z):
    return PC + PS * (1 / (1 + np.exp(-z)) - PC) + PSH


def main():
    for k, v in EXTRA_SPEC.items():
        asofdelta.ASOF_SPEC.setdefault(k, v)

    p = champion_z()
    q = np.clip((p - PC - PSH) / PS + PC, 1e-9, 1 - 1e-9)
    z0 = np.log(q / (1 - q))
    back = affine(z0)
    print(f"아핀 역산 검증  max|Δp| = {np.abs(back - p).max():.3e}")
    assert np.abs(back - p).max() < 1e-9

    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    tr, va = s < VALID, s == VALID
    d = df[va].copy().reset_index(drop=True)
    y = d[TARGET_COL].to_numpy(dtype=np.float64)
    assert len(y) == len(p), f"행 불일치 {len(y)} vs {len(p)}"
    base = bss(y, p)
    print(f"챔피언 2024 행 {len(y):,}   BSS {base:.1f}  ⚠️ in-sample — 절대값 무의미\n")

    art = asofdelta.fit_offsets(df[tr])
    league = float(df.loc[tr, TARGET_COL].mean())
    priors = {c: float(np.nanmean(df.loc[tr, c].to_numpy(dtype=np.float64)))
              for _, c, _ in GAP_SPEC}
    feat = build(d, art, league, priors)
    rng = np.random.default_rng(0)
    dsh = d.copy()
    dsh["pitcher_id"] = rng.permutation(d["pitcher_id"].to_numpy())
    dsh["batter_id"] = rng.permutation(d["batter_id"].to_numpy())
    fsh = build(dsh, art, league, priors)

    codes, uniq = pd.factorize(d["pitcher_id"].to_numpy())
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    seen = np.where(np.diff(bnd) > 0)[0]
    br = np.random.default_rng(7)
    IDX = [np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                           br.choice(seen, size=len(seen), replace=True)])
           for _ in range(B_BOOT)]

    def delta(x, b):
        return bss(y, affine(z0 + b * x)) - base

    print(f"  {'축':<14}" + "".join(f"{b:>+9.3f}" for b in BETAS) + f"{'최적β':>9}{'Δ':>9}")
    best = {}
    for c in PROBED:
        if c not in feat:
            continue
        row = [delta(feat[c].astype(np.float64), b) for b in BETAS]
        i = int(np.argmax(row))
        best[c] = (BETAS[i], row[i])
        print(f"  {c:<14}" + "".join(f"{v:>+9.2f}" for v in row)
              + f"{BETAS[i]:>+9.3f}{row[i]:>+9.2f}")

    print("\n" + "=" * 92)
    print("최적 β 칸 — 클러스터 부트스트랩 + 엔티티 순열 위약")
    print("=" * 92)
    print(f"  {'축':<14}{'β*':>8}{'Δ':>9}{'95% 구간':>22}{'P(Δ>0)':>9}"
          f"{'위약 Δ':>10}{'판정':>7}")
    for c, (b, dv) in sorted(best.items(), key=lambda kv: -kv[1][1]):
        x = feat[c].astype(np.float64)
        xs = fsh[c].astype(np.float64)
        bs = np.array([bss(y[i], affine(z0 + b * x)[i]) - bss(y[i], p[i]) for i in IDX])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        pl = delta(xs, b)
        ok = "✅" if (lo > 0 and dv > abs(pl)) else "·"
        print(f"  {c:<14}{b:>+8.3f}{dv:>+9.2f}   [{lo:>+7.2f}, {hi:>+7.2f}]"
              f"{np.mean(bs > 0):>9.3f}{pl:>+10.2f}{ok:>7}")
    print("\n  ⚠️ in-sample 층이라 양수는 하한, 음수는 약한 증거다 (§7.174·§7.181).")


if __name__ == "__main__":
    main()
