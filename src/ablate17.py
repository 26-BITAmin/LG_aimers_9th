"""17차 실험 — MLP 성분을 더 키울 수 있는가.

s07이 LB +29.5를 냈다 (홀드아웃 예측 +11.4의 2.6배). **모델 다양성이 이 문제에서
가장 생산적인 축**이라는 증거다. 그렇다면 MLP 쪽을 더 밀어볼 값이 있다.

세 가지를 묻는다:

  ① **시드 수.** 현재 MLP는 3멤버인데 LightGBM은 20부스터다. MLP 쪽 분산이
     훨씬 크다. 5시드로 늘리면 블렌드가 좋아지는가. (순수 분산 축소)

  ② **두 번째 아키텍처.** 80개 피처 위에서 조건 ③(크기비 ≤3배)을 통과한 것은
     emb4@9 하나뿐이었다. 다른 후보가 통과하는가.
       emb8 — 용량을 키우고 정규화도 같이 키운다 (hid 256, emb 8, wd 2e-2)
       noid — 투수/타자 임베딩을 아예 뺀다. 정보 사용 방식이 달라 탈상관이 클 수 있다

  ③ **alpha.** 사전 고정값 0.3이 여전히 맞는가. 시드를 늘려 노이즈를 줄이고 재확인.

⚠️ **LB를 보고 alpha를 조정하지 않는다.** s07의 +29.5는 fold 2023 값(+27.9)에
   가까웠지만, 그걸 근거로 alpha를 올리면 리더보드 역추론이다. 판정은 두 fold로만.

⚠️ 채택 기준은 그대로 — 부호 일치 + **크기비 3배 이내**. 이 조건은 지금 4전 4중이다
   (Trackman 5.8배 ✗ / 피처51개 4.9배 ✗ / 앙상블 1.1배 ✓ / MLP 2.5배 ✓).

실행:  .venv/Scripts/python.exe -u src/ablate17.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import lightgbm as lgb
import numpy as np

import mlp
from config import ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, N_ROUNDS, TRAIN_CSV
from data import load_csv

mlp.THREADS = 8

FOLDS = [2023, 2024]
LGB_SEEDS = (42, 7, 2024)
MLP_SEEDS = (42, 7, 2024, 1234, 31337)
ALPHAS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)

EMB4 = {"pitcher_id": 4, "batter_id": 4, "pitcher_team_id": 4, "batter_team_id": 4,
        "base_state": 4, "hand_matchup": 4, "count_state": 4}
EMB8 = dict(EMB4, pitcher_id=8, batter_id=8)

# (설정, 시드 수)
ARCHS = {
    "emb4": (dict(epochs=9, hid=128, lr=2e-3, dropout=0.3, wd=1e-2,
                  use_id_emb=True, emb_dim=EMB4), 5),
    "emb8": (dict(epochs=9, hid=256, lr=2e-3, dropout=0.35, wd=2e-2,
                  use_id_emb=True, emb_dim=EMB8), 3),
    "noid": (dict(epochs=6, hid=128, lr=2e-3, dropout=0.3, wd=1e-2,
                  use_id_emb=False, emb_dim=EMB4), 3),
}


def lgb_ensemble(Xtr, ytr, w, Xva, t0):
    z, n = np.zeros(len(Xva)), 0
    for over in ENSEMBLE_VARIANTS.values():
        for s in LGB_SEEDS:
            p = dict(FINAL_LGB_PARAMS, **over, seed=s, bagging_seed=s,
                     feature_fraction_seed=s, data_random_seed=s)
            bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=N_ROUNDS)
            z += bst.predict(Xva, raw_score=True)
            n += 1
    print(f"  lgbm {n}부스터  ({time.time()-t0:.0f}s)")
    return z / n


def verdict(d):
    """부호 일치 + 크기비 ≤3 → 채택."""
    same = bool(np.all(d > 0) or np.all(d < 0))
    sp = np.abs(d).max() / max(np.abs(d).min(), 1e-9) if same else np.inf
    return same, sp, (same and d.mean() > 0 and sp <= 3)


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    Z = {}          # Z[fold] = {"lgb": z, "emb4": [z per seed], ...}
    Y = {}

    for f in FOLDS:
        print(f"\n=== valid={f} ===")
        Xtr, ytr, w, Xva, yva, art = mlp.fold_data(df, f)
        A, C, B, Cv, num_cols, sizes = mlp.encode(Xtr, Xva, art)
        Y[f] = yva
        Z[f] = {"lgb": lgb_ensemble(Xtr, ytr, w, Xva, t0)}
        print(f"  lgb_ens BSS = {mlp.score(Z[f]['lgb'], yva):.1f}")

        for name, (cfg, nseed) in ARCHS.items():
            zs = []
            for s in MLP_SEEDS[:nseed]:
                _, curve = mlp.train_mlp(A, C, ytr, w, B, Cv, sizes, seed=s,
                                         verbose=False, **cfg)
                zs.append(curve[-1])          # 마지막 에폭 (argmax 금지)
            Z[f][name] = zs
            zm = np.mean(zs, axis=0)
            print(f"  {name:<5s} {nseed}시드  단독 {mlp.score(zm, yva):>7.1f}  "
                  f"corr {np.corrcoef(Z[f]['lgb'], zm)[0,1]:.3f}   ({time.time()-t0:.0f}s)")

    def blend(f, names, a, k=None):
        zs = [z for n in names for z in (Z[f][n][:k] if k else Z[f][n])]
        return mlp.score((1 - a) * Z[f]["lgb"] + a * np.mean(zs, axis=0), Y[f])

    base = np.array([mlp.score(Z[f]["lgb"], Y[f]) for f in FOLDS])
    print(f"\n기준 lgb_ens: " + "  ".join(f"{f}={b:.1f}" for f, b in zip(FOLDS, base)))

    def report(title, rows):
        print(f"\n=== {title} ===")
        print(f"{'설정':<26s} " + " ".join(f"{f:>8d}" for f in FOLDS)
              + f" {'기준대비':>9s} {'크기비':>7s}")
        print("-" * 62)
        for label, v in rows:
            d = v - base
            same, sp, ok = verdict(d)
            print(f"{label:<26s} " + " ".join(f"{x:>8.1f}" for x in v)
                  + f" {d.mean():>+9.1f} "
                  + (f"{sp:>7.1f}" if same else f"{'inf':>7s}")
                  + ("  <<< 채택 조건 통과" if ok else ""))

    # ① 시드 수 (emb4, alpha 0.3 고정)
    report("① MLP 시드 수 (emb4, a=0.30)",
           [(f"emb4 x{k}시드", np.array([blend(f, ["emb4"], 0.30, k) for f in FOLDS]))
            for k in (1, 2, 3, 5)])

    # ② 아키텍처 (각자 전체 시드, alpha 0.3 고정)
    combos = [["emb4"], ["emb8"], ["noid"], ["emb4", "emb8"], ["emb4", "noid"],
              ["emb4", "emb8", "noid"]]
    report("② 아키텍처 (a=0.30)",
           [("+".join(c), np.array([blend(f, c, 0.30) for f in FOLDS])) for c in combos])

    # ③ alpha (emb4 전체 시드)
    report("③ alpha (emb4 5시드)",
           [(f"a={a:.2f}", np.array([blend(f, ["emb4"], a) for f in FOLDS]))
            for a in ALPHAS])

    print("\n  ※ LB를 보고 alpha를 고르지 않는다. 두 fold 평균과 크기비로만 판정한다.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
