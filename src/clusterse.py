"""§7.133 후속 4 — **행 단위 SE 가 얼마나 과소평가인가**를 내 팔에서 직접 잰다. 학습 없음.

발단
★ 결과 먼저 — **가설은 틀렸다. 비율은 1.1~1.5 배(평균 1.28)다.**
   featdrop 계열의 행 단위 SE 는 과소평가가 **아니었다.** §7.93 의 t -3.9 는 클러스터로
   읽어도 -3.0 이고 여전히 유의하다. **§7.93 의 기각도 계열 종결 선언도 그대로 선다.**

   내 오류의 정체: §7.133 의 "클러스터 폭 ±7~9" 는 **95% 구간의 반폭**(≈2·SE)이었고,
   §7.93 의 "±1.4~1.6" 은 **SE 그 자체**였다. **구간과 SE 를 맞대놓고 4~5 배라고 했다.**
   같은 단위로 놓으면 §7.93 의 95% 반폭은 2.8~3.1 이고 내 basenf(같은 1컬럼 드랍)의
   클러스터 SE 3.27/2.78 과 같은 자릿수다 — 애초에 불일치가 없었다.

   그리고 **왜 클러스터 보정이 거의 무의미했는지**가 남는 소득이다:
   이 비교들은 전부 **쌍체**다(같은 행·같은 y 에서 두 모델의 차분). 투수 수준
   성분은 차분에서 대부분 상쇄되므로 행 간 상관이 SE 를 거의 안 부풀린다.
   `rookie2.py` 가 클러스터로 갈아탄 맥락은 **그룹 비교**(신인 vs 베테랑)로 쌍체가
   아니었다 — 거기서는 클러스터가 지배한다. **쌍체냐 아니냐가 갈림이지 행이냐
   클러스터냐가 아니다.** featdrop 계열은 처음부터 옳은 SE 를 쓰고 있었다.

--- 이하는 실행 전에 세운 (틀린) 가설의 기록 ---

    §7.92~§7.94(`featdrop*.py`)는 SE 를 **행 단위**로 계산했다:

        featdrop2.py:59   return k * d.mean(), k * d.std(ddof=1) / np.sqrt(len(d))

    한 투수의 수천 행은 독립이 아니다. `rookie2.py` 가 같은 지적을 하고 클러스터
    부트스트랩으로 갈아탔는데 featdrop 계열은 행 단위로 남아 있었고, §7.93 은
    그 t (-3.9 / -2.6)로 계열 전체를 **종결 선언**했다("현 구성이 이미 국소 최적").

⚠️ 폐기한 접근 — 이 파일의 첫 판은 featdrop 의 저장된 z 로 §7.92·93·94 를 **직접
   재판정**하려 했다. 실패했다: featdrop 이 쓴 ctl foldcache 를 특정할 수 없다.
   길이로 고르면 fold 마다 다른 키(26bcbefa56 / 5488ef1b18)가 걸리고, ctl 은 0.7 이내로
   맞는데 drop 팔이 ±2 어긋난다. 재구성된 점추정이 기록(-5.57/-4.07)과 안 맞으면
   그것으로 원 판정을 뒤집을 수 없다 — **틀린 ctl 로 계산한 SE 는 SE 가 아니다.**
   -> 그 결과는 폐기한다. 대신 **내가 통제하는 팔**에서 비율만 잰다.

무엇을 하나 (재학습 0)
    이번 라운드에서 내가 만든 팔들에 대해 같은 Δ 에 두 가지 SE 를 붙인다.
        ① 행 단위 쌍체 SE   (featdrop 계열의 방식)
        ② 투수 클러스터 부트스트랩
    비율 ②/① 이 곧 "featdrop 의 t 를 얼마로 나눠 읽어야 하는가"의 추정치다.
    조작 종류별로 나눠 본다 — 컬럼 수를 바꾸는지가 §7.133 에서 폭을 갈랐다.

    ★ `dropmiss2` 가 이 조사의 앵커다. §7.94 기전대로 **z_lgb 가 base 와 비트 동일**
      임을 확인했으므로 LGBM 쪽 참효과가 정확히 0 이다. 그런 팔의 Δ 가 -4.0/+3.9 로
      나온다는 것 자체가 노이즈 바닥의 직접 측정이고, 두 SE 중 어느 쪽이 그것을
      제대로 덮는지가 곧 판정이다.

실행:  .venv/Scripts/python.exe -u src/clusterse.py 2>&1 | tee experiments/clusterse.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

import jointbeta as J                                                # noqa: E402
from config import (ASOF_AXES, CTX_AXES, MLP_ALPHA, TARGET_COL, TRAIN_CSV)  # noqa: E402
from data import load_csv                                            # noqa: E402
from validation import bss, shift_to_prior                           # noqa: E402

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
FOLDS = [2023, 2024]
SEED = 42
NBOOT = 400

# (표시명, 파일 prefix, mode, 조작 종류)
ARMS = [
    ("basenf     (컬럼 -1)", "histdepth", "basenf", "컬럼 수 변경"),
    ("dropmiss2  (컬럼 -2)", "dupdrop", "dropmiss2", "컬럼 수 변경"),
    ("dupsig     (컬럼 +1)", "dupdrop", "dupsig", "컬럼 수 변경"),
    ("depth      (컬럼 +1)", "histdepth", "depth", "컬럼 수 변경"),
    ("depthflag  (값만)", "histflag", "depthflag", "컬럼 수 불변"),
    ("dfplacebo  (값만)", "histflag", "dfplacebo", "컬럼 수 불변"),
    ("carry      (1.98% 행)", "histdepth", "carry", "컬럼 수 불변"),
]


def load(pref, fold, mode):
    with np.load(CACHE / f"{pref}_fold{fold}_{mode}.npz") as d:
        return d["z_lgb"], d["z_mlp"]


def main():
    rng = np.random.default_rng(SEED)
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    y_all = df[TARGET_COL].to_numpy(dtype=np.float64)
    codes, _ = pd.factorize(df["pitcher_id"].to_numpy())
    mclean = ~((gt == "F") & (s < 2023))
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    a = MLP_ALPHA
    print(f"a={a}  부트스트랩 {NBOOT}회  배포층(delta축 + prior shift, clean 행)")
    print(f"재학습 없음 — 이번 라운드에서 저장한 z 만 읽는다\n")

    rows = []
    for fold in FOLDS:
        Xax, _, _, _ = J.axis_matrix(df, s, fold)
        vi = np.where(s == fold)[0]
        mm = mclean[vi]
        yv = y_all[vi][mm]
        cf = codes[vi][mm]
        ax = (Xax @ dep)[mm]
        n_id = cf.max() + 1
        order = np.argsort(cf, kind="stable")
        bnd = np.searchsorted(cf[order], np.arange(n_id + 1))
        seen = np.where(np.diff(bnd) > 0)[0]

        zl0, zm0 = load("histdepth", fold, "base")
        z0 = ((1 - a) * zl0 + a * zm0)[mm] + ax

        for lab, pref, mode, kind in ARMS:
            zl1, zm1 = load(pref, fold, mode)
            z1 = ((1 - a) * zl1 + a * zm1)[mm] + ax

            def sc(z, idx):
                yy = yv[idx]
                return bss(yy, shift_to_prior(1 / (1 + np.exp(-z[idx])), yy.mean()))

            full = np.arange(len(yv))
            delta = sc(z1, full) - sc(z0, full)

            # ① 행 단위 쌍체 SE (featdrop 계열의 방식 그대로)
            p0 = shift_to_prior(1 / (1 + np.exp(-z0)), yv.mean())
            p1 = shift_to_prior(1 / (1 + np.exp(-z1)), yv.mean())
            d_row = (p0 - yv) ** 2 - (p1 - yv) ** 2
            k = 1e5 / (yv.mean() * (1 - yv.mean()))
            se_row = k * d_row.std(ddof=1) / np.sqrt(len(d_row))

            # ② 투수 클러스터 부트스트랩
            bs = [sc(z1, i) - sc(z0, i) for i in
                  (np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                                   rng.choice(seen, size=len(seen), replace=True)])
                   for _ in range(NBOOT))]
            se_cl = float(np.std(bs, ddof=1))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append((fold, lab, kind, delta, se_row, se_cl, lo, hi))

    print(f"  {'fold':<6}{'팔':<24}{'Δ':>8}{'① 행SE':>9}{'t행':>7}"
          f"{'② 클러스터SE':>13}{'t클':>7}{'②/①':>7}   95% 구간")
    for fold, lab, kind, d, sr, sc_, lo, hi in rows:
        print(f"  {fold:<6}{lab:<24}{d:>+8.1f}{sr:>9.2f}{d/sr:>+7.1f}"
              f"{sc_:>13.2f}{d/sc_:>+7.1f}{sc_/sr:>7.1f}   [{lo:+6.1f},{hi:+6.1f}]")

    print(f"\n  조작 종류별 ②/① 평균")
    for kind in ("컬럼 수 변경", "컬럼 수 불변"):
        v = [sc_ / sr for _, _, k2, _, sr, sc_, _, _ in rows if k2 == kind]
        w = [sr for _, _, k2, _, sr, _, _, _ in rows if k2 == kind]
        c = [sc_ for _, _, k2, _, _, sc_, _, _ in rows if k2 == kind]
        print(f"     {kind:<14} ②/① = {np.mean(v):.2f}배   "
              f"(행SE 평균 {np.mean(w):.2f}, 클러스터SE 평균 {np.mean(c):.2f})")

    print(f"\n  ★ 앵커 — dropmiss2 는 z_lgb 가 base 와 **비트 동일**이다 (§7.94 기전, 양 fold 확인).")
    print(f"     LGBM 쪽 참효과가 정확히 0 인 팔이므로, 그 Δ 는 전량 MLP 재학습 노이즈다.")
    print(f"     두 SE 중 그 |Δ| 를 덮는 쪽이 옳은 SE 다.")
    for fold, lab, _, d, sr, sc_, lo, hi in rows:
        if "dropmiss2" in lab:
            print(f"     fold {fold}: |Δ|={abs(d):.1f}  행SE {sr:.2f} -> "
                  f"{'덮는다' if abs(d) <= 2*sr else '**못 덮는다 (거짓 유의)**'}   "
                  f"클러스터SE {sc_:.2f} -> {'덮는다' if abs(d) <= 2*sc_ else '못 덮는다'}")

    print(f"\n  ⚠️ 이 결과는 featdrop 계열의 **점추정을 재현하지 않는다** (ctl 특정 실패, 위 폐기 기록).")
    print(f"     말할 수 있는 것은 '같은 하네스에서 행 SE 가 클러스터 SE 보다 N 배 좁다' 뿐이고,")
    print(f"     그것이 §7.92~94 의 t 를 얼마로 나눠 읽어야 하는지의 추정치다.")


if __name__ == "__main__":
    main()
