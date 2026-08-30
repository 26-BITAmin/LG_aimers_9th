"""§7.204c — `gap_strike` 기울기가 **표류하는가 진동하는가**. 4시즌.

사용자 반박: *"§7.204+b 의 폐쇄 논거가 개소리다."* 맞다. 두 가지가 틀렸다.

    ① 표의 "문제" 열에 내 행만 "✅ 닫는다" 를 적었다 — 남의 폐쇄는 감사하고
       내 것만 감사에서 뺐다.
    ② **논거 자체가 틀렸다.** "단일 β 가 2023 과 2024 를 동시에 도울 수 없다" 로 닫았는데
       **2023 을 도울 이유가 없다.** 배포 대상은 2025 다. fold 는 2025 의 추정량이지
       목적함수가 아니다. 게다가 §7.200 자신의 결론이 *"fold2024 를 믿어라"* 였고
       LB 가 2/2 로 그것을 따랐다 (s10 −4.97 · sy02_c4 +7.44).

그러면 진짜 질문이 바뀐다.

    ❌ "부호가 갈렸나"            <- 2시즌으로 답할 수 있지만 잘못된 질문
    ✅ "기울기가 **표류**하는가 **진동**하는가"

    표류라면 f2023 −0.285 -> f2024 +0.218 은 2025 에 대해 **양수를 가리킨다**.
    진동이라면 두 값은 평균 0 주위의 뽑기이고 2025 는 동전 던지기다.
    **2점으로는 두 가설이 구별되지 않는다** ([[n1-transfer-test-gives-mean-not-variance]]).

    -> fold 를 2021·2022 까지 넓힌다. 캐시가 이미 있다 (키 5488ef1b18, 4시즌 전부).

⚠️ 이 실험이 답하지 못하는 것
    fold 가 이르면 학습 시즌이 줄어 기저 모델이 달라진다. 그래서 기울기의 **절대 크기**는
    시즌 간 직접 비교가 위험하다. 우리가 읽는 것은 **부호의 시계열 패턴**이다.
    ([[raw-transfer-confounds-level-with-structure]] — 레벨과 구조를 섞지 말 것)

기준선은 §7.204 B팔(실제 배포: ctx 2축 + asof gap_success/gap_bsuccess/gap_ball)과 동일.
SE 는 §7.204b 와 같은 **투수 클러스터 robust**.

실행:  .venv/Scripts/python.exe -u src/gapstrike3.py > experiments/gapstrike3.log 2>&1
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
import foldcache
from config import SEED, TARGET_COL, TRAIN_CSV
from data import load_csv
from gapstrike import DEPLOYED_BASE, EXTRA_SPEC, GAP_BALL, baseline
from gapstrike2 import cluster_t
from mixresid import GAP_SPEC, build

FOLDS = [2021, 2022, 2023, 2024]
AXIS = "gap_strike"


def main():
    for k, v in EXTRA_SPEC.items():
        asofdelta.ASOF_SPEC.setdefault(k, v)

    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(SEED)

    print(f"§7.204c  {AXIS} 기울기 시계열 — 표류인가 진동인가")
    print("  기준선 = 실제 배포 (ctx 2축 + asof gap_success/gap_bsuccess/gap_ball)")
    print("  SE = 투수 클러스터 robust\n")

    out = {}
    for fold in FOLDS:
        tr, va = s < fold, s == fold
        d = df[va].copy()
        yv = cached[fold][2]
        pid = d["pitcher_id"].to_numpy(dtype=np.int64)

        art = asofdelta.fit_offsets(df[tr])
        league = float(df.loc[tr, TARGET_COL].mean())
        priors = {c: float(np.nanmean(df.loc[tr, c].to_numpy(dtype=np.float64)))
                  for _, c, _ in GAP_SPEC}
        feat = build(d, art, league, priors)
        dsh = d.copy()
        dsh["pitcher_id"] = rng.permutation(pid)
        dsh["batter_id"] = rng.permutation(d["batter_id"].to_numpy())
        fsh = build(dsh, art, league, priors)

        rB, _ = baseline(df, tr, d, yv, cached[fold], DEPLOYED_BASE + [GAP_BALL])
        sl, t, se, G = cluster_t(rB, feat[AXIS], pid)
        slc, tc, _, _ = cluster_t(rB, fsh[AXIS], pid)
        out[fold] = (sl, se, t, tc, G, len(d))
        print(f"  fold {fold}  행 {len(d):,}  투수 {G}  학습시즌 {int(s[tr].min())}~{fold-1}"
              f"   기울기 {sl:+.5f} ± {se:.5f}   t {t:+.2f}   (대조군 t {tc:+.2f})")

    print("\n" + "=" * 78)
    print("시계열")
    print("=" * 78)
    print(f"  {'fold':<8}{'기울기':>12}{'SE':>10}{'클러스터 t':>12}{'95% 구간':>26}{'0배제':>7}")
    for f in FOLDS:
        sl, se, t, tc, G, n = out[f]
        lo, hi = sl - 1.96 * se, sl + 1.96 * se
        print(f"  {f:<8}{sl:>+12.5f}{se:>10.5f}{t:>12.2f}   [{lo:+.5f}, {hi:+.5f}]"
              f"{'예' if not (lo < 0 < hi) else '-':>7}")

    sls = np.array([out[f][0] for f in FOLDS])
    ses = np.array([out[f][1] for f in FOLDS])
    yrs = np.array(FOLDS, dtype=float)

    # 가중 선형 추세 (역분산 가중)
    w = 1.0 / ses ** 2
    X = np.stack([np.ones_like(yrs), yrs - yrs.mean()], axis=1)
    W = np.diag(w)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ sls)
    cov = np.linalg.inv(X.T @ W @ X)
    slope_t = beta[1] / np.sqrt(cov[1, 1])

    # 가중 평균 (추세 없음 가정)
    mu = float((w * sls).sum() / w.sum())
    se_mu = float(np.sqrt(1.0 / w.sum()))
    # 이질성 Q (진동 = 과분산)
    Q = float((w * (sls - mu) ** 2).sum())

    print("\n" + "=" * 78)
    print("표류 vs 진동")
    print("=" * 78)
    print(f"  가중 추세 기울기/년 = {beta[1]:+.5f}   t = {slope_t:+.2f}"
          f"   {'유의 (표류)' if abs(slope_t) > 1.96 else '유의하지 않다'}")
    print(f"  가중 평균           = {mu:+.5f} ± {se_mu:.5f}   t = {mu/se_mu:+.2f}")
    print(f"  이질성 Q            = {Q:.2f}  (df {len(FOLDS)-1}, 임계 7.81)"
          f"   {'과분산 = 시즌마다 다르다' if Q > 7.81 else '균질'}")
    print(f"  부호                = {''.join('+' if v > 0 else '-' for v in sls)}")
    print(f"  2025 외삽 (추세)    = {beta[0] + beta[1]*(2025 - yrs.mean()):+.5f}")

    print("\n  읽는 법")
    print("    추세 t 유의 + 부호가 단조   -> 표류. 최근 fold 를 믿고 2025 에 베팅 가능")
    print("    Q 과분산 + 부호가 왔다갔다  -> 진동. 2025 는 동전 던지기. 닫는다")
    print("    둘 다 아니고 평균 t 유의    -> 안정된 축인데 fold 하나가 튄 것")


if __name__ == "__main__":
    main()
