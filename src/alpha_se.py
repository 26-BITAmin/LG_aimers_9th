"""α 비교의 **표준오차**를 실측한다 — 스토리를 만들기 전에 노이즈부터 재는 용도.

s09(α .40)가 s08(α .30) 대비 LB +6.7을 냈는데 홀드아웃 예측은 +1.0이었다.
"MLP 비중을 늘리는 변경은 홀드아웃이 과소평가한다"는 결론으로 가기 전에
**+1.0과 +6.7의 차이가 노이즈보다 큰지** 확인해야 한다.

BSS 스케일이 10만 배라 눈으로는 큰 숫자처럼 보이지만, 두 예측이 상관 0.99+로
붙어 있으면 차이의 SE도 같이 작아진다. 어느 쪽인지는 재봐야 안다.

쌍체 비교의 표준오차:
    d_i = (p_A,i - y_i)^2 - (p_B,i - y_i)^2 = (p_A,i - p_B,i)(p_A,i + p_B,i - 2 y_i)
    SE(mean d) = std(d_i) / sqrt(n)
    BSS 단위로 환산: SE / BS_ref * 100000

⚠️ 배포 모델은 2019~2024로 학습됐으므로 2024에 대해 **in-sample**이다.
   따라서 여기서 얻는 BSS 수준값은 낙관적이고 의미가 없다. 필요한 건 수준이 아니라
   **(p_A - p_B)의 스케일**이며, 이건 in-sample에서도 대표성이 있다.
   유의성 판정에만 쓰고 성능 판정에는 쓰지 않는다.

실행:  .venv/Scripts/python.exe -u src/alpha_se.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "submit"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

from config import TARGET_COL, TRAIN_CSV
from data import load_csv
from predict import Predictor

N_ROWS = 250_000          # 평가 테스트셋(245,789)과 비슷하게 맞춘다
ALPHAS = (0.30, 0.40, 0.50, 0.60)


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    df = df[df.season == 2024]
    if len(df) > N_ROWS:
        df = df.iloc[:N_ROWS]
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    print(f"표본 {len(df):,}행 (2024)  ({time.time()-t0:.0f}s)")

    P = Predictor(model_dir=str(Path(__file__).parent.parent / "submit" / "model"))

    # predict()는 블렌드된 확률만 돌려주므로 두 성분을 여기서 따로 만든다.
    # 배포 파일(predict.py)은 검증을 통과한 상태라 건드리지 않는다.
    from features import build_features
    X = build_features(df, P.art)
    if P.art.get("use_player_ids"):
        for c in ("pitcher_id", "batter_id"):
            X[c] = df[c].to_numpy(dtype=np.float32)
    X = X[P.cols]

    z_lgb = np.zeros(len(X))
    for bst in P.boosters:
        z_lgb += bst.predict(X, raw_score=True)
    z_lgb /= len(P.boosters)

    z_mlp = np.zeros(len(X))
    for m in P.mlps:
        z_mlp += m.logit(X)
    z_mlp /= len(P.mlps)
    print(f"예측 완료  lgb {len(P.boosters)}개 / mlp {len(P.mlps)}개"
          f"  ({time.time()-t0:.0f}s)")

    d = z_mlp - z_lgb
    print(f"\nstd(z_lgb)={z_lgb.std():.4f}  std(z_mlp)={z_mlp.std():.4f}"
          f"  corr={np.corrcoef(z_lgb, z_mlp)[0,1]:.4f}")
    print(f"std(z_mlp - z_lgb) = {d.std():.4f}")

    bs_ref = y.mean() * (1 - y.mean())

    def probs(a):
        z = (1 - a) * z_lgb + a * z_mlp + P.delta
        return 1.0 / (1.0 + np.exp(-z))

    print(f"\n{'비교':<16s} {'ΔBrier':>12s} {'ΔBSS':>9s} {'SE(BSS)':>9s} {'t':>7s}")
    print("-" * 58)
    base = probs(ALPHAS[0])
    for a in ALPHAS[1:]:
        p = probs(a)
        di = (p - y) ** 2 - (base - y) ** 2
        se = di.std(ddof=1) / np.sqrt(len(di))
        print(f"α {ALPHAS[0]:.2f} → {a:.2f}     {di.mean():>+12.3e}"
              f" {-di.mean()/bs_ref*1e5:>+9.2f} {se/bs_ref*1e5:>9.2f}"
              f" {abs(di.mean()/se):>7.1f}")

    print("\n  ※ ΔBSS 수준값은 in-sample이라 의미 없다. 볼 것은 **SE(BSS)** 한 칸이다.")
    print("     홀드아웃 +1.0 과 LB +6.7 의 차이(5.7)가 이 SE의 몇 배인지가 핵심.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
