"""잎이 **상수가 아니라 선형**인 트리 — §7.21 가설을 트리 쪽에서 직접 친다.

가설의 출처는 이 저장소다 (§7.21 / SUBMISSIONS.md 전이표)
------------------------------------------------------
    "매끄러운 함수는 드리프트에 강하다. 트리는 학습 분포에 맞춘 임계값으로 공간을
     쪼개고 **잎마다 고정값을 박는다**. 분포가 밀린 곳에서 그 잎 값은 낡은 지역
     평균이 된다. 매끄러운 함수는 같은 상황에서 더 완만하게 무너진다."

이 가설로 MLP 도입(+29.5, 홀드아웃의 2.6배)과 α 이동(+6.7, 6.7배)을 설명했다.
그런데 **트리에서 "잎마다 고정값"을 떼어내 본 적은 없다.**

`linear_tree=True` (Shi, Li, Ke et al., *Gradient Boosting With Piece-Wise Linear
Regression Trees*, IJCAI 2019)는 각 잎에 상수 대신 **선형 모델**을 적합한다.
가설이 지목한 그 성질만 정확히 제거한다.

왜 이건 노브가 아닌가
---------------------
§7.18의 9종 재튜닝은 전부 **같은 함수족 안에서** 용량·정규화를 옮긴 것이다.
piecewise-constant → piecewise-linear는 **함수족이 바뀐다.** 전이표에서 유일하게
1배를 넘긴 계열이 "새 함수족"이다 (MLP 도입 2.6배).

배포 비용도 거의 없다 — LightGBM 텍스트 포맷이 잎 계수를 직렬화하고 `predict`가
그대로 처리한다. `mlp_infer` 같은 새 추론 코드가 필요 없다.

결정 기준 (이 저장소가 이미 쓰던 것)
------------------------------------
    CatBoost   corr 0.985 → 블렌드 보너스 +3.5뿐  → 기각
    MLP        corr 0.90~0.93                    → 채택 (+29.5)
**단독 점수가 아니라 기존 z_lgb와의 상관**이 갈랐다. 여기서도 그걸 먼저 본다.
corr가 0.98대면 그 자리에서 끝이다 (학습을 더 돌릴 이유가 없다).

실행:  .venv/Scripts/python.exe -u src/lintree.py 2>&1 | tee experiments/lintree.log
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

import foldcache
from config import (CTX_AXES, CTX_K, FINAL_LGB_PARAMS, MLP_ALPHA, N_ROUNDS,
                    TARGET_COL, TRAIN_CSV)
from ctxsplit import build_delta, lookup
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
SEEDS = [42, 7]
K = CTX_K

# linear_lambda = 잎 선형계수의 L2. 0이면 25만 행 × 80 피처에서 발산하기 쉽다.
# 문헌 기본값 대역(0.1~10)에서 세 점만 본다 — 스윕이 아니라 **감도 확인**이다.
LIN_LAMBDAS = [1.0, 10.0]


def train_z(X, y, w, Xva, params, seeds):
    z, n = np.zeros(len(Xva)), 0
    for s in seeds:
        p = dict(params, seed=s, bagging_seed=s, feature_fraction_seed=s,
                 data_random_seed=s)
        bst = lgb.train(p, lgb.Dataset(X, y, weight=w), num_boost_round=N_ROUNDS)
        z += bst.predict(Xva, raw_score=True)
        n += 1
        del bst
    return z / n


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    pid = df["pitcher_id"].to_numpy(dtype=np.int64)
    bh = df["batter_hand"].to_numpy(dtype=np.int64)
    b = df["balls_before"].to_numpy(dtype=np.int64)
    st = df["strikes_before"].to_numpy(dtype=np.int64)
    c4 = (b >= 3).astype(np.int64) + 2 * (st >= 2).astype(np.int64)

    cached = foldcache.load(FOLDS, verbose=False)

    for fold in FOLDS:
        t0 = time.time()
        tr, va = s < fold, s == fold
        Xtr, ytr, w, Xva, yva, art = foldcache.fold_data(df, fold)
        nan_frac = float(np.isnan(Xtr.to_numpy(dtype=np.float64)).mean())

        z_lgb, z_mlp, yv, _ = cached[fold]
        adj = np.zeros_like(z_lgb)
        for name, beta in CTX_AXES:
            code, m = (bh, 8) if name == "hand" else (c4, 4)
            u, d, _ = build_delta(pid[tr], code[tr], y[tr], m, K)
            drow, _ = lookup(u, d, pid[va], code[va], m)
            adj = adj + beta * drow

        def blend(zl, a=MLP_ALPHA):
            z = (1 - a) * zl + a * z_mlp + adj
            return bss(yv, shift_to_prior(1.0 / (1.0 + np.exp(-z)), yv.mean()))

        base_blend = blend(z_lgb)
        print("=" * 96)
        print(f"fold {fold}   train {tr.sum():,} → valid {va.sum():,}"
              f"   피처 {Xtr.shape[1]}   NaN 비율 {nan_frac:.4f}")
        print(f"  배포 z_lgb(12개) 단독 {bss(yv, shift_to_prior(1/(1+np.exp(-z_lgb)), yv.mean())):.1f}"
              f"   배포 블렌드 {base_blend:.1f}")
        print("=" * 96)

        # 대조 — **같은 시드 수**의 상수잎 base. 시드 수를 맞춰야 공정하다
        z_ctl = train_z(Xtr, ytr, w, Xva, FINAL_LGB_PARAMS, SEEDS)
        s_ctl = bss(yv, shift_to_prior(1 / (1 + np.exp(-z_ctl)), yv.mean()))
        print(f"\n  {'설정':<22}{'단독BSS':>9}{'corr(z_lgb)':>13}{'corr(z_mlp)':>13}"
              f"{'교체 블렌드':>12}{'추가 블렌드':>12}{'초':>7}")
        print(f"  {'상수잎 base(대조)':<22}{s_ctl:>9.1f}"
              f"{np.corrcoef(z_ctl, z_lgb)[0,1]:>13.4f}"
              f"{np.corrcoef(z_ctl, z_mlp)[0,1]:>13.4f}"
              f"{blend(z_ctl):>12.1f}{blend(0.5*(z_lgb+z_ctl)):>12.1f}"
              f"{time.time()-t0:>7.0f}")

        for lam in LIN_LAMBDAS:
            t1 = time.time()
            p = dict(FINAL_LGB_PARAMS, linear_tree=True, linear_lambda=lam)
            try:
                z_lin = train_z(Xtr, ytr, w, Xva, p, SEEDS)
            except Exception as e:                       # noqa: BLE001
                print(f"  linear λ={lam}: 실패 — {type(e).__name__}: {e}")
                continue
            print(f"  {f'선형잎 λ={lam}':<22}"
                  f"{bss(yv, shift_to_prior(1/(1+np.exp(-z_lin)), yv.mean())):>9.1f}"
                  f"{np.corrcoef(z_lin, z_lgb)[0,1]:>13.4f}"
                  f"{np.corrcoef(z_lin, z_mlp)[0,1]:>13.4f}"
                  f"{blend(z_lin):>12.1f}{blend(0.5*(z_lgb+z_lin)):>12.1f}"
                  f"{time.time()-t1:>7.0f}")
        print()

    print("=" * 96)
    print("읽는 법 — **corr(z_lgb)가 먼저다**")
    print("  0.98대  → CatBoost와 같은 자리. 블렌드 보너스가 안 나온다. 그 자리에서 끝.")
    print("  0.95 이하 → 새 함수족일 가능성. 그때 시드를 늘려 조건 ①②③로 간다.")
    print("  '교체'는 z_lgb를 대체, '추가'는 z_lgb와 반반 섞은 것. 추가 쪽이 실전 후보다.")


if __name__ == "__main__":
    main()
