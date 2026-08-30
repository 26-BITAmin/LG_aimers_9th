"""제출 전 LB 점수 예측 — **레벨 이동에 한해서만** 정확하다.

원리
    점수 = 1e5 · (1 − Brier / (r(1−r))),  r = 평가 라벨 평균.
    모든 예측에 상수 Δ 를 더하면 Brier 가 정확히 다음만큼 변한다.

        ΔScore = −K · [2Δ·b + Δ²]        K = 1e5/(r(1−r)),  b = p̄ − r

    Δ 와 p̄ 는 로컬에서 잰다. r 은 §4-17 에서 **LB 두 점으로 역산**됐고 두 독립 측정이
    2.6e−05 안에서 일치했다 (0.487000 / 0.486974). 라운드 4 에서 이 항등식이 실측과
    **소수 넷째 자리까지** 맞았다 (−38.7302 vs −38.7302).

⚠️ 쓸 수 있는 경우 / 없는 경우

    ✅ 두 zip 이 **전 예측에 같은 상수를 더한 것만** 다르다  (PROB_SHIFT · prior 목표)
    ⚠️ 스케일(아핀 c) 변경 — Var 와 Cov(p,y) 가 같이 바뀌어 근사만 된다
    ❌ 모델·라우팅·피처가 바뀌었다 — **예측 불가.** 행별로 예측이 달라지면
       Cov(p,y) 를 로컬에서 알 방법이 없다

    ❌ 사례가 실제로 있었다: `hj07`(트랙맨 잔차 층)은 홀드아웃 순효과 +2.84 였는데
       LB 는 **−5.93** 이었다 (배율 −2.09). 모델 변경은 홀드아웃으로도 못 맞힌다.

    p̄ 는 2024 프록시로 잰다 — 실제 test 가 아니라 대리값이라 스택마다 미세한 간격이
    있다 (§4-17 이 그 간격을 0.1224 점으로 정량화했다).
"""
import argparse
import os
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

TRAIN_CSV = "/Users/hyejin/Aimers_9th/data/train.csv"
R_EVAL = 0.486987          # §4-17 두 독립 역산의 평균 (0.487000 / 0.486974)


def K(r=R_EVAL):
    return 1e5 / (r * (1 - r))


def run(zip_path, pkl):
    tmp = tempfile.mkdtemp(prefix="plb_")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    out = os.path.join(tmp, "p.npy")
    code = ("import sys, numpy as np, pandas as pd\n"
            f"sys.path.insert(0, {tmp!r})\n"
            "from predict import Predictor\n"
            f"df = pd.read_pickle({pkl!r})\n"
            f"p = Predictor(model_dir={os.path.join(tmp,'model')!r})\n"
            f"np.save({out!r}, np.asarray(p.predict(df), dtype=np.float64))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp, capture_output=True, text=True,
                       env=dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="4"))
    if r.returncode:
        raise SystemExit(f"[중단] 추론 실패\n{r.stdout[-1500:]}\n{r.stderr[-2000:]}")
    return np.load(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="LB 점수를 아는 기준 zip")
    ap.add_argument("--ref-score", type=float, required=True)
    ap.add_argument("--cand", required=True, help="예측할 zip")
    ap.add_argument("--r", type=float, default=R_EVAL)
    a = ap.parse_args()

    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    df = df[df.season == 2024].drop(columns=["control_success"]).reset_index(drop=True)
    tmp = tempfile.mkdtemp(); pkl = os.path.join(tmp, "px.pkl"); df.to_pickle(pkl)
    print(f"프록시 2024 {len(df):,}행\n")

    pr = run(a.ref, pkl)
    pc = run(a.cand, pkl)
    d = pc - pr
    dm, ds, dmax = float(d.mean()), float(d.std()), float(np.abs(d).max())
    print(f"{'기준':<8} p̄ {pr.mean():.6f}   std {pr.std():.6f}")
    print(f"{'후보':<8} p̄ {pc.mean():.6f}   std {pc.std():.6f}")
    print(f"{'차이':<8} Δmean {dm:+.6g}   Δstd {ds:.3e}   max|Δ| {dmax:.3e}\n")

    # 순수 레벨 이동인가 — 행별 편차가 평균 대비 무시할 만한가
    pure = ds < 0.05 * abs(dm) if dm != 0 else ds < 1e-9
    b = pr.mean() - a.r
    k = K(a.r)
    dscore = -k * (2 * dm * b + dm ** 2)
    print(f"r = {a.r:.6f}   K = {k:,.0f}   b = p̄_ref − r = {b:+.6f}")
    print(f"예측 ΔScore = −K[2Δb + Δ²] = {dscore:+.4f}")
    print(f"예측 LB     = {a.ref_score:.4f} {dscore:+.4f} = **{a.ref_score + dscore:.4f}**\n")

    if pure:
        print("✅ 순수 레벨 이동으로 보인다 (Δstd 가 Δmean 의 5% 미만) — 예측이 유효하다.")
    else:
        print("❌ **순수 레벨 이동이 아니다.** 행별로 예측이 달라졌다"
              f" (Δstd {ds:.3e} vs Δmean {abs(dm):.3e}).")
        print("   위 숫자는 **레벨 성분만** 설명한다. 모델·라우팅 변경분은 로컬에서 예측 불가다.")
        print("   실사례: hj07 은 홀드아웃 +2.84 였는데 LB −5.93 이었다.")


if __name__ == "__main__":
    main()
