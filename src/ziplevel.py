"""제출 zip 들의 **도달 레벨**을 같은 입력에서 대조한다.

왜 필요한가
    `src/zipdiff.py` 가 `hj03_dim8_cold.zip` 과 우리 `sy04_cold.zip` 의 유일한 소스 차이가
    `predict.py` 의 한 줄임을 찾았다:

        hj03        PROB_SHIFT = 0.000172669
        sy04_cold   PROB_SHIFT = 0.005580239        (sy02_c4 는 0.005482311)

    30배 차이다. 그런데 두 빌드의 emb4 가중치가 **서로 다른 학습 draw** 이므로
    (W0 max|Δ| 0.05 · b1 0.12 · emb0[1:] 0.063) 자연 예측 평균도 다르다.
    → PROB_SHIFT 만 보고 "레벨이 다르다" 고 말할 수 없다. **도달 평균을 재야 한다.**

    이게 중요한 이유는 MODEL.md §0 이다:
        BSS 손실 ≈ −400,700 × (예측평균 − 실제평균)²
    0.005 어긋나면 10 BSS 다. hj03(1131.4503) 과 sy02_c4(1127.1558) 의 차이가 +4.29 이니
    레벨 하나로 부호가 갈릴 수 있는 규모다.

무엇을 재나
    각 zip 을 **자기 `predict.py` 로** 같은 2024 행 위에서 돌린다 (모듈 격리 = 서브프로세스).
      · p̄ (도달 레벨) · sd · 쌍별 corr · mean|Δp|
      · 마스킹 프록시(UNK 0.20) 판 — 콜드스타트 패치가 겨냥한 바로 그 행들
    ⚠️ 2024 는 이 빌드들의 **학습 안**이다. BSS 는 의미 없고, 여기서 읽을 것은
       **레벨과 행별 차이**뿐이다.

실행:  .venv/Scripts/python.exe -u src/ziplevel.py 2>&1 | tee experiments/ziplevel.log
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
ASCII = Path("D:/t1eval")
sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

from config import TRAIN_CSV

CACHE = ROOT / "experiments" / "cache"
UNK = 0.20

ZIPS = [
    ("sy02_c4",   "sy02_c4.zip",         "LB 1127.1558  우리 챔피언 (신영 C-4, dim8)"),
    ("hj03",      "hj03_dim8_cold.zip",  "LB 1131.4503  팀 챔피언 (혜진 dim8 + 콜드)"),
    ("sy04cold",  "sy04_cold.zip",       "미제출  sy02_c4 + 우리 콜드 패치"),
    ("sy05coldrt", "sy05_coldrt.zip",    "미제출  거기에 라우팅 t=−0.45"),
]

WORKER = r"""
import sys, numpy as np, pandas as pd
arm, out, mask_mode = sys.argv[1], sys.argv[2], sys.argv[3]
d = r"D:/t1eval/" + arm
sys.path.insert(0, d)                       # 우리 src 를 넣지 않는다 (§7.174 사고)
va = pd.read_csv(r"{csv}")
va = va[va.season == 2024].reset_index(drop=True)
mask = np.zeros(len(va), dtype=bool)
if mask_mode == "mask":
    rng = np.random.RandomState(0)          # coldbuild.py 와 같은 시드·같은 절차
    pids = rng.permutation(np.sort(va["pitcher_id"].unique()))
    cnt = va["pitcher_id"].value_counts()
    take, tot, goal = [], 0, {unk} * len(va)
    for pid in pids:
        if tot >= goal:
            break
        take.append(pid); tot += int(cnt[pid])
    mask = va["pitcher_id"].isin(take).to_numpy()
    va.loc[mask, "pitcher_id"] = -1
from predict import Predictor
import predict as _pm
assert _pm.__file__.replace("\\", "/").startswith(d), "그 zip 의 predict.py 가 아니다"
p = Predictor(model_dir=d + "/model")
np.savez_compressed(out, p=np.asarray(p.predict(va), dtype=np.float64), mask=mask,
                    y=va["control_success"].to_numpy(dtype=np.float64))
"""


def unpack(tag, zipname):
    d = ASCII / f"ZL_{tag}"
    if (d / "predict.py").exists():
        return d
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    with zipfile.ZipFile(ROOT / zipname) as z:
        z.extractall(d)
    return d


def run(tag, mode):
    f = CACHE / f"ziplevel_{tag}_{mode}.npz"
    if not f.exists():
        wk = CACHE / "_ziplevel_worker.py"
        wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/"))
                            .replace("{unk}", repr(UNK)),
                      encoding="utf-8", newline="\n")
        r = subprocess.run([sys.executable, str(wk), f"ZL_{tag}", str(f), mode],
                           capture_output=True, text=True)
        if not f.exists():
            print(f"  [실패] {tag}/{mode}:\n{r.stdout.strip()[-800:]}\n{r.stderr.strip()[-1200:]}")
            sys.exit(1)
    z = np.load(f)
    return z["p"], z["mask"], z["y"]


def main():
    for tag, zn, _d in ZIPS:
        assert (ROOT / zn).exists(), zn
        unpack(tag, zn)
    print("압축 해제 완료. 2024 행에서 각 zip 의 자기 predict.py 로 추론한다.")
    print("⚠️ 2024 는 학습 안이다 — 읽을 것은 **레벨과 행별 차이**뿐, BSS 가 아니다.\n")

    for mode, label in (("plain", "무마스킹 2024"), ("mask", f"마스킹 프록시 (UNK {UNK})")):
        P, Y, M = {}, None, None
        for tag, _zn, desc in ZIPS:
            p, m, y = run(tag, mode)
            P[tag] = p
            Y, M = y, m
        print("=" * 96)
        print(f"■ {label}   n={len(Y):,}   실제 양성률 ȳ = {Y.mean():.6f}"
              + (f"   미지 행 {M.mean():.1%}" if mode == "mask" else ""))
        print("=" * 96)
        print(f"  {'tag':<11}{'p̄':>11}{'p̄ − ȳ':>11}{'sd':>9}"
              f"{'레벨 BSS 손실':>14}   설명")
        for tag, _zn, desc in ZIPS:
            p = P[tag]
            d = p.mean() - Y.mean()
            print(f"  {tag:<11}{p.mean():>11.6f}{d:>+11.6f}{p.std():>9.5f}"
                  f"{-400700*d*d:>14.1f}   {desc}")

        print(f"\n  ── 쌍별 (기준 = hj03) ──")
        print(f"  {'상대':<11}{'corr':>12}{'mean|Δp|':>11}{'max|Δp|':>10}{'Δp̄':>12}")
        h = P["hj03"]
        for tag, _zn, _d in ZIPS:
            if tag == "hj03":
                continue
            q = P[tag]
            print(f"  {tag:<11}{np.corrcoef(h, q)[0,1]:>12.7f}"
                  f"{np.abs(h-q).mean():>11.6f}{np.abs(h-q).max():>10.5f}"
                  f"{q.mean()-h.mean():>+12.6f}")

        if mode == "mask":
            print(f"\n  ── 미지 행 vs 알려진 행 (콜드 패치가 겨냥한 자리) ──")
            print(f"  {'상대':<11}{'미지 Δp̄':>12}{'알려진 Δp̄':>13}")
            for tag, _zn, _d in ZIPS:
                if tag == "hj03":
                    continue
                q = P[tag] - h
                print(f"  {tag:<11}{q[M].mean():>+12.6f}{q[~M].mean():>+13.6f}")
        print()

    print("=" * 96)
    print("읽는 법")
    print("=" * 96)
    print("  · p̄ 가 서로 가까우면 → PROB_SHIFT 가 30배 달라도 **도달 레벨은 같다**.")
    print("    그러면 hj03 과 우리 빌드의 LB 차이는 레벨이 아니라 refinement(학습 draw) 몫이다.")
    print("  · p̄ 가 벌어지면 → 그 차이가 곧 LB 차이의 후보다 (−400,700·Δ²).")
    print("  · 알려진 행 Δp̄ ≈ 0 이고 미지 행만 움직이면 → 두 콜드 패치가 같은 일을 한다.")


if __name__ == "__main__":
    main()
