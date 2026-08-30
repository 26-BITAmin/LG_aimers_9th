"""`batter_id` 콜드스타트 홀드아웃 — 혜진 `coldstart_probe.py` 의 UNK vs COLD 설계.

⚠️ 왜 이 설계가 필요한가
    2024 행은 배포 모델의 **in-sample** 이다 (학습에 2024 가 들어갔다). 그래서
    절대 BSS 를 읽으면 안 된다. 대신 타자 일부를 **인위적으로 미지 처리**하고
    같은 행에서 두 팔을 비교한다.

        UNK   마스킹된 행 -> emb1[0] (미학습 초기값)   = 현 배포가 미지 타자에 하는 것
        COLD  마스킹된 행 -> emb1[1:] 의 평균          = 제안 (hj07_bcold)
        TRUE  마스킹 안 함 (그 타자 자신의 벡터)        = 상한. in-sample 이라 부풀어 있다

    **판정은 UNK vs COLD 다.** 둘 다 그 타자 자신의 벡터를 안 쓰므로 in-sample
    부풀림이 두 팔에 똑같이 걸려 차이에서 상쇄된다. TRUE 는 회수율 계산용 참고값이다.

    회수율 = (COLD − UNK) / (TRUE − UNK)
    혜진이 pitcher 에서 잰 값: dim8 **23.5%** · dim12 5.5% (차원이 오르면 회수가 무너진다)
    `emb1` 은 dim **4** 라 그보다 높을 수 있다 — 그것이 이 측정의 핵심 미지수다.

⚠️ 시드는 **마스킹 뽑기**다 (모델은 고정). 혜진이 pitcher 에서 5시드 t 1.42 ->
   15시드 t 2.81~18.94 로 뒤집었다 — 표본 부족이 판정을 죽이던 자리다. 여기선 처음부터 15.

★ 행 독립성이 검증돼 있으므로(verify_zip 6번, 비트 0) **마스킹된 행만 추론해도**
  전체를 추론한 것과 값이 같다. 15시드 × 3팔을 그래서 몇 분에 돌린다.

사용
    python tools_hj/bcold_probe.py --unk hj03_dim8_cold.zip --cold hj07_bcold.zip
"""
import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

TRAIN_CSV = "D:/projects/LG aimers 야구/data/train.csv"
UNK_FRAC = 0.093
SEEDS = tuple(range(15))
SEASONS = (2023, 2024)      # 저장소 규약: 두 fold 를 따로 보고한다

WORKER = r'''
import sys, numpy as np, pandas as pd
d, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, d)
df = pd.read_csv(r"{csv}", encoding="utf-8-sig")
df = df[df["season"] == {season}].reset_index(drop=True)
y = df["control_success"].to_numpy(dtype=np.float64)
X = df.drop(columns=["control_success"])
from predict import Predictor
import predict as _pm
assert _pm.__file__.replace("\\", "/").startswith(d.replace("\\", "/")), "predict.py 가 그 팔이 아니다"
p = Predictor(model_dir=d + "/model")
res = {}
for seed in {seeds}:
    rng = np.random.RandomState(seed)
    bids = rng.permutation(np.sort(X["batter_id"].unique()))
    cnt = X["batter_id"].value_counts()
    take, tot, goal = [], 0, {frac} * len(X)
    for b in bids:
        if tot >= goal: break
        take.append(b); tot += int(cnt[b])
    m = X["batter_id"].isin(take).to_numpy()
    sub = X[m].copy()
    if mode == "mask":
        sub["batter_id"] = -1
    res[f"p{seed}"] = np.asarray(p.predict(sub), dtype=np.float64)
    res[f"y{seed}"] = y[m]
np.savez_compressed(out, **res)
'''


def run(arm_dir, out, mode, tmp, season):
    wk = os.path.join(tmp, f"w_{os.path.basename(arm_dir)}_{mode}_{season}.py")
    open(wk, "w", encoding="utf-8", newline="\n").write(
        WORKER.replace("{csv}", TRAIN_CSV).replace("{seeds}", repr(list(SEEDS)))
              .replace("{frac}", repr(UNK_FRAC)).replace("{season}", str(season)))
    r = subprocess.run([sys.executable, wk, arm_dir, out, mode],
                       capture_output=True, text=True,
                       env=dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="6"))
    if not os.path.exists(out):
        raise SystemExit(f"[중단] {arm_dir} {mode}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return np.load(out)


def bss(y, p):
    r = y.mean()
    return 1e5 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unk", required=True)
    ap.add_argument("--cold", required=True)
    a = ap.parse_args()

    os.makedirs("D:/t1eval", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="bprobe_", dir="D:/t1eval")
    dirs = {}
    for name, zp in (("UNK", a.unk), ("COLD", a.cold)):
        d = os.path.join(tmp, name)
        with zipfile.ZipFile(zp) as z:
            z.extractall(d)
        dirs[name] = d
    print(f"UNK  = {a.unk}\nCOLD = {a.cold}")
    print(f"마스킹 시드 {len(SEEDS)}개 · 목표 미지 비율 {UNK_FRAC:.1%}\n")

    from scipy import stats
    out = {}
    for season in SEASONS:
        print(f"\n{'#'*70}\n#  fold {season}\n{'#'*70}")
        print("추론 UNK (마스킹) …")
        unk = run(dirs["UNK"], os.path.join(tmp, f"unk{season}.npz"), "mask", tmp, season)
        print("추론 COLD (마스킹) …")
        cold = run(dirs["COLD"], os.path.join(tmp, f"cold{season}.npz"), "mask", tmp, season)
        print("추론 TRUE (마스킹 없음, 상한 참고) …")
        true = run(dirs["UNK"], os.path.join(tmp, f"true{season}.npz"), "plain", tmp, season)

        rows = []
        print(f"\n{'시드':>4}{'행':>9}{'UNK':>10}{'COLD':>10}{'TRUE':>10}"
              f"{'COLD−UNK':>11}{'회수율':>9}")
        for s in SEEDS:
            y = unk[f"y{s}"]
            u, c, t = bss(y, unk[f"p{s}"]), bss(y, cold[f"p{s}"]), bss(y, true[f"p{s}"])
            rec = (c - u) / (t - u) if abs(t - u) > 1e-9 else float("nan")
            rows.append((u, c, t, c - u, rec))
            print(f"{s:>4}{len(y):>9,}{u:>10.1f}{c:>10.1f}{t:>10.1f}{c-u:>+11.2f}{rec:>8.1%}")

        d = np.array([r[3] for r in rows])
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        tc = stats.t.ppf(0.975, n - 1)
        out[season] = (d, se, np.array([r[4] for r in rows]),
                       np.mean([r[2] - r[0] for r in rows]))
        print(f"\n  fold {season}  Δ = {d.mean():+.2f} ± {se:.2f}  t {d.mean()/se:+.2f}"
              f"  95% CI [{d.mean()-tc*se:+.2f}, {d.mean()+tc*se:+.2f}]"
              f"  양수 {int((d>0).sum())}/{n}")

    print("\n" + "=" * 78)
    print("판정 — 두 fold (저장소 규약)")
    print("=" * 78)
    print(f"  {'fold':<8}{'Δ(마스킹행)':>14}{'SE':>7}{'t':>8}{'95% CI':>22}"
          f"{'양수':>7}{'천장':>9}{'회수율':>9}{'전체환산':>10}")
    for season in SEASONS:
        d, se, rec, ceil = out[season]
        tc = stats.t.ppf(0.975, len(d) - 1)
        print(f"  {season:<8}{d.mean():>+14.2f}{se:>7.2f}{d.mean()/se:>+8.2f}"
              f"   [{d.mean()-tc*se:+7.2f},{d.mean()+tc*se:+7.2f}]"
              f"{int((d>0).sum()):>4}/{len(d)}{ceil:>9.1f}{np.nanmean(rec):>9.1%}"
              f"{d.mean()*UNK_FRAC:>+10.3f}")
    a, b = out[SEASONS[0]][0].mean(), out[SEASONS[1]][0].mean()
    sa, sb = out[SEASONS[0]][1], out[SEASONS[1]][1]
    print(f"\n  부호 일치: {'✅ 두 fold 양수' if a > 0 and b > 0 else '❌ 갈림'}")
    print(f"  fold 차 = {b-a:+.2f} ± {np.sqrt(sa**2+sb**2):.2f}"
          f"   z {(b-a)/np.sqrt(sa**2+sb**2):+.2f}")
    print("\n  ※ Δ 는 마스킹된 행 위에서만 잰 값이다. 전체 환산은 × 미지 비율"
          f" {UNK_FRAC:.3f} (ref_S/ref_all ≈ 1 가정).")
    print("  ※ 두 fold 다 배포 모델의 in-sample 이다 — 절대 BSS 가 아니라")
    print("     **같은 행 위 두 팔의 차이**만 읽는다. 부풀림은 양쪽에 똑같이 걸려 상쇄된다.")


if __name__ == "__main__":
    main()
