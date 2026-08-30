"""콜드스타트 대체 벡터의 **스케일 적합** — `emb[0] <- s · mean(emb[1:])`.

사용자: *"베타 감마 적합도 해봤어?"* 안 했다. 혜진 라운드 18 도 안 했다.
`mean` 은 **s = 1 을 못박은 것**이지 적합한 값이 아니다.

    emb0[0] <- β · mean(emb0[1:])      투수. 질량 19.9%. 챔피언 `hj03` 이 **이미 β=1 로 배포**
    emb1[0] <- γ · mean(emb1[1:])      타자. 질량  9.3%. `hj07_bcold` 가 γ=1

    -> 투수 쪽 β* ≠ 1 이면 **챔피언 자체에 공짜 이득**이 있다. 재학습이 없으니 비용은 0.

왜 s=1 이 최적일 이유가 없나
    `mean` 은 임베딩 공간의 중심이지 **Brier 최적점이 아니다.** 임베딩 -> 예측 사상이
    비선형이라 중심이 그대로 최적 예측을 주지 않는다. 그리고 미지 개체는 정보가 없으므로
    개체별 신호를 **더 줄이는 편**(s<1, 원점 쪽)이 나을 수 있다 — 학습된 중심의 노름이
    이미 0.04~0.15 로 작지만 0 은 아니다. 반대로 평균이 "평균적 개체" 를 과소 대표하면
    s>1 이 나을 수도 있다. **부호를 사전에 못 정한다 -> 재야 한다.**

평가는 §7.207 과 같은 UNK-vs-arm 마스킹 설계다 (2024·2023 둘 다 배포 in-sample 이므로
절대 BSS 를 안 읽고 **같은 행 위 두 팔의 차이**만 읽는다). 15 마스킹 시드 · 두 fold.

    기준선(0점)  투수 = 현 배포 β=1        타자 = 미패치 UNK (γ 없음)
    ⚠️ 투수는 `hj03` 에서 원래 초기값이 이미 덮여 있어 UNK 를 복원할 수 없다.
       그래서 투수 곡선은 **β=1 대비 Δ** 로 읽는다 (그게 배포 판단에 필요한 값이다).

⚠️ 2D(β, γ) 상호작용은 안 잰다. 서로 다른 임베딩 표라 1차로는 분리되지만, 같은 MLP 로
   들어가므로 상호작용이 0 이라는 보장은 없다 — **1D 두 개로 읽고 그 한계를 적어둔다**
   ([[telescoping-chains-are-not-evidence]]).

사용
    python tools_hj/cold_scale_sweep.py --src hj03_dim8_cold.zip --channel emb1
    python tools_hj/cold_scale_sweep.py --src hj03_dim8_cold.zip --channel emb0
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
from scipy import stats

TRAIN_CSV = "D:/projects/LG aimers 야구/data/train.csv"
SEEDS = tuple(range(15))
SEASONS = (2023, 2024)
SCALES = (0.0, 0.5, 1.0, 1.5, 2.0)
CH = {"emb0": ("pitcher_id", 0.199), "emb1": ("batter_id", 0.093)}

WORKER = r'''
import sys, numpy as np, pandas as pd
d, out, season = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, d)
df = pd.read_csv(r"{csv}", encoding="utf-8-sig")
df = df[df["season"] == season].reset_index(drop=True)
y = df["control_success"].to_numpy(dtype=np.float64)
X = df.drop(columns=["control_success"])
from predict import Predictor
import predict as _pm
assert _pm.__file__.replace("\\", "/").startswith(d.replace("\\", "/"))
p = Predictor(model_dir=d + "/model")
res = {}
for seed in {seeds}:
    rng = np.random.RandomState(seed)
    ids = rng.permutation(np.sort(X["{idcol}"].unique()))
    cnt = X["{idcol}"].value_counts()
    take, tot, goal = [], 0, {frac} * len(X)
    for b in ids:
        if tot >= goal: break
        take.append(b); tot += int(cnt[b])
    m = X["{idcol}"].isin(take).to_numpy()
    sub = X[m].copy()
    sub["{idcol}"] = -1
    res[f"p{seed}"] = np.asarray(p.predict(sub), dtype=np.float64)
    res[f"y{seed}"] = y[m]
np.savez_compressed(out, **res)
'''


def bss(y, p):
    r = y.mean()
    return 1e5 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--channel", required=True, choices=list(CH))
    a = ap.parse_args()
    idcol, frac = CH[a.channel]
    knob = "β" if a.channel == "emb0" else "γ"

    os.makedirs("D:/t1eval", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="scale_", dir="D:/t1eval")
    base = os.path.join(tmp, "base")
    with zipfile.ZipFile(a.src) as z:
        z.extractall(base)

    # 스케일별 팔 디렉터리 (+ 기준선 = 원본)
    arms = {"BASE": base}
    for s in SCALES:
        d = os.path.join(tmp, f"s{s:.1f}")
        shutil.copytree(base, d)
        for f in sorted(glob.glob(os.path.join(d, "model", "mlp_emb4_*.npz"))):
            dd = dict(np.load(f))
            e = dd[a.channel]
            dd[a.channel] = np.vstack([(s * e[1:].mean(0))[None, :], e[1:]]).astype(e.dtype)
            np.savez(f, **dd)
        arms[f"{s:.1f}"] = d
    print(f"소스 {os.path.basename(a.src)} · 채널 {a.channel} ({idcol}, 질량 {frac:.1%})")
    print(f"{knob} 격자 {SCALES} + 기준선(원본) · 시드 {len(SEEDS)} · fold {SEASONS}")
    print(f"⚠️ 기준선 = zip 원본. emb0 은 이미 mean(β=1)이므로 '1.0' 팔과 같아야 한다 (검산).\n")

    res = {}
    for season in SEASONS:
        for name, d in arms.items():
            out = os.path.join(tmp, f"{name}_{season}.npz")
            wk = os.path.join(tmp, f"w_{name}_{season}.py")
            open(wk, "w", encoding="utf-8", newline="\n").write(
                WORKER.replace("{csv}", TRAIN_CSV).replace("{seeds}", repr(list(SEEDS)))
                      .replace("{idcol}", idcol).replace("{frac}", repr(frac)))
            r = subprocess.run([sys.executable, wk, d, out, str(season)],
                               capture_output=True, text=True,
                               env=dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE",
                                        OMP_NUM_THREADS="6"))
            if not os.path.exists(out):
                raise SystemExit(f"[중단] {name} f{season}\n{r.stdout[-1200:]}\n{r.stderr[-1200:]}")
            z = np.load(out)
            res[(name, season)] = np.array([bss(z[f"y{s}"], z[f"p{s}"]) for s in SEEDS])
            print(f"  f{season} {name:>5}  BSS {res[(name,season)].mean():>8.2f}")

    print("\n" + "=" * 80)
    print(f"{knob} 곡선 — Δ vs 기준선(zip 원본), 마스킹 행 위, 시드 쌍체")
    print("=" * 80)
    print(f"  {'fold':<7}" + "".join(f"{knob}={s:.1f}".rjust(16) for s in SCALES))
    best = {}
    for season in SEASONS:
        b = res[("BASE", season)]
        line, stat = f"  {season:<7}", {}
        for s in SCALES:
            d = res[(f"{s:.1f}", season)] - b
            se = d.std(ddof=1) / np.sqrt(len(d))
            stat[s] = (d.mean(), se)
            line += f"{d.mean():+8.2f}±{se:.2f}".rjust(16)
        print(line)
        best[season] = stat
    print()
    for season in SEASONS:
        st = best[season]
        bs = max(SCALES, key=lambda s: st[s][0])
        m, se = st[bs]
        tc = stats.t.ppf(0.975, len(SEEDS) - 1)
        edge = " ⚠️ 격자 끝" if bs in (SCALES[0], SCALES[-1]) else ""
        print(f"  fold {season} 최적 {knob} = {bs:.1f}   Δ {m:+.2f} ± {se:.2f}"
              f"   95% CI [{m-tc*se:+.2f}, {m+tc*se:+.2f}]"
              f"   전체환산 {m*frac:+.3f}{edge}")
    a1 = max(SCALES, key=lambda s: best[SEASONS[0]][s][0])
    a2 = max(SCALES, key=lambda s: best[SEASONS[1]][s][0])
    print(f"\n  두 fold 최적 일치: {'✅' if a1 == a2 else f'❌ {a1:.1f} vs {a2:.1f}'}")
    print(f"  ※ 1.0 팔과 기준선의 차이는 emb0 에서 0 이어야 한다 (이미 mean 패치됨).")
    print(f"     emb1 에서는 기준선이 미패치 UNK 이므로 1.0 팔의 Δ = hj07 의 이득이다.")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
