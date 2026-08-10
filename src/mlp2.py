"""MLP 재시도 — **천장 측정**. 제대로 조인 MLP가 통과선(-80) 근처라도 오는가.

1차 결과 (src/mlp.py): 핸디캡 -4670. 학습 손실 0.696 -> 0.641로 계속 떨어지는데
valid는 무너졌다. 이 문제에서 일반화 가능한 신호의 총량이 logloss 약 0.004 nat인데
MLP는 0.048 nat을 팠다 — 있지도 않은 신호의 12배.

그런데 버리기 전에 하나 남았다: **로짓 상관이 0.35~0.45다.**
lgbm<->xgb는 0.9+였다. 탈상관 보너스가 xgb의 +15.5보다 훨씬 클 수 있다.

    블렌드 점수 ~= 가중평균(단독 점수) + 보너스

1차는 보너스를 잴 수조차 없었다(MLP가 너무 망가져서). 조인 MLP로 다시 잰다.

⚠️ 이 실험은 **valid에서 최적 에폭을 고른다.** 즉 낙관적인 상한이다.
   묻는 질문이 "천장이 통과선에 닿기는 하는가"이므로 그게 옳은 질문이다.
   천장이 못 미치면 그대로 기각. 넘으면 그때 정직한 조기중단 규칙을 세워
   2 fold로 다시 판정한다.

fold 2024만 쓴다 — 기각 방향으로는 이걸로 충분하고 2배 빠르다.

실행:  .venv/Scripts/python.exe -u src/mlp2.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import mlp
from config import TRAIN_CSV
from data import load_csv

VALID = 2024
E = 10

# 1차의 실패 모드는 투수/타자 임베딩의 암기로 의심된다.
# 정규화 축을 하나씩 흔들어 어느 쪽이 듣는지 본다.
VARIANTS = {
    "1차 설정 (참고)": dict(epochs=E),
    "ID임베딩 제거": dict(epochs=E, use_id_emb=False),
    "ID임베딩 4차원+wd": dict(epochs=E, use_id_emb=True, wd=1e-2, dropout=0.3, hid=128,
                          emb_dim=dict(mlp.EMB_DIM, pitcher_id=4, batter_id=4)),
    "소형+강한 정규화": dict(epochs=E, use_id_emb=False, hid=64, dropout=0.4, wd=3e-2),
}


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    Xtr, ytr, w, Xva, yva, art = mlp.fold_data(df, VALID)
    A, Ctr, B, Cva, num_cols, sizes = mlp.encode(Xtr, Xva, art)
    print(f"valid={VALID}  수치 {len(num_cols)}개  ({time.time()-t0:.0f}s)")

    z_lgb = mlp.run_lgbm(Xtr, ytr, w, Xva, yva)
    s_lgb = mlp.score(z_lgb, yva)
    print(f"기준 lgbm  BSS = {s_lgb:.1f}   ({time.time()-t0:.0f}s)\n")

    rows = []
    for name, cfg in VARIANTS.items():
        print(f"--- {name} ---")
        _, curve = mlp.train_mlp(A, Ctr, ytr, w, B, Cva, sizes, yva=yva, **cfg)
        scores = [mlp.score(z, yva) for z in curve]
        k = int(np.argmax(scores))
        z, s = curve[k], scores[k]

        # 최적 에폭에서 블렌드. 보너스 = 블렌드 - 점수의 가중평균
        best = max(((a, mlp.score((1 - a) * z_lgb + a * z, yva)) for a in
                    (0.1, 0.2, 0.3, 0.4, 0.5)), key=lambda t: t[1])
        a, sb = best
        bonus = sb - ((1 - a) * s_lgb + a * s)
        corr = float(np.corrcoef(z_lgb, z)[0, 1])
        rows.append((name, k + 1, s, s - s_lgb, a, sb, sb - s_lgb, bonus, corr))
        print(f"  최적 ep{k+1}  단독 {s:.1f} (핸디캡 {s-s_lgb:+.1f})  "
              f"최적블렌드 a={a} → {sb:.1f} ({sb-s_lgb:+.1f})  "
              f"보너스 {bonus:+.1f}  corr {corr:.3f}   ({time.time()-t0:.0f}s)\n")

    print(f"{'변형':<18s} {'ep':>3s} {'단독':>8s} {'핸디캡':>8s} {'a':>4s} "
          f"{'블렌드':>8s} {'대비':>7s} {'보너스':>7s} {'corr':>6s}")
    print("-" * 76)
    print(f"{'lgbm (기준)':<18s} {'':>3s} {s_lgb:>8.1f}")
    for r in sorted(rows, key=lambda r: -r[6]):
        print(f"{r[0]:<18s} {r[1]:>3d} {r[2]:>8.1f} {r[3]:>+8.1f} {r[4]:>4.1f} "
              f"{r[5]:>8.1f} {r[6]:>+7.1f} {r[7]:>+7.1f} {r[8]:>6.3f}")

    print("\n판정:")
    print("  · '대비'가 양수인 변형이 하나도 없으면 → MLP 기각. 피처 작업에 자원을 돌린다.")
    print("  · 양수가 있으면 → 정직한 조기중단 규칙을 세워 2 fold x 3시드로 재판정.")
    print("    (여기서 고른 에폭은 valid를 보고 고른 것이라 그대로 쓸 수 없다.)")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
