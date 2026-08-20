"""F 전문가 블렌드의 **정직한** 이득 — 배포 스택 위에서, 가중치를 미리 고정해서

검증할 주장 (fspec.py, §7.101)
    2023+ F만으로 만든 LGBM F 전문가를 전역 모델과 섞으면 F행 +54.6, 전체 환산 +6.4.

그 숫자의 두 가지 결함
    ① 기준선이 **단일 LGBM**이었다. 실제 배포는 LGBM 20 + MLP 8이라 이미 훨씬 좋고,
       분산이 이미 줄어든 위에 얹으면 이득이 작아지는 게 보통이다
       (MODEL.md: 단일 모델에서 잰 이득은 앙상블 시스템에 대한 예측이 아니다).
    ② 섞는 비율 a=0.5를 **평가 fold(2024)를 보고** 골랐다 → 낙관 편향.

이 스크립트가 고치는 방법
    기준선  foldcache의 배포 스택 예측 (1-MLP_ALPHA)·z_lgb + MLP_ALPHA·z_mlp
    a 선택  **fold "2023H2"** 에서 고른다 (2023 7~9월 검증, 그 이전만 학습).
            2024는 쳐다보지 않는다.
    평가    그 a를 2024에 그대로 적용. 이게 정직한 추정치다.
            비교용으로 2024 오라클 최적 a도 같이 찍어 낙관 편향의 크기를 보인다.

덤으로 얻는 것
    "2023H2"는 2024와 **겹치지 않는 두 번째 검증 구간**이다. 이 아이디어는 깨끗한 F가
    2023년부터라 fold가 하나뿐이라고 했는데, 시즌 내부 분할로 하나를 더 만들 수 있다.
    두 구간에서 부호가 같은지가 채택 조건이다 (CLAUDE.md §2).

주의
    foldcache의 fold_data는 δ 보정축(ctx) 없이 artifacts를 만든다. 즉 캐시 예측은
    배포 스택의 **근사**다 (이 저장소의 다른 스크립트들과 같은 관례).

실행:  .venv/Scripts/python.exe -u src/fspec_verify.py 2>&1 | tee experiments/fspec_verify.log
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
from config import MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = ["2023H2", 2024]
SEEDS = (42, 7, 2024)
N_BOOT = 400
GRID = np.round(np.arange(0.0, 0.801, 0.05), 3)

SMALL_LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15,
                 min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7,
                 bagging_freq=1, lambda_l2=10.0, max_bin=127, num_threads=6,
                 verbosity=-1)


def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def specialist(Xtr_f, ytr_f, month_f, Xva):
    """2023+ F만으로 학습. 라운드 수는 학습셋 내부 시간분할로 정한다."""
    cut = np.quantile(month_f, 0.75)
    itr, iva = month_f <= cut, month_f > cut
    if iva.sum() < 300 or len(np.unique(ytr_f[iva])) < 2:
        itr, iva = np.ones(len(ytr_f), bool), np.zeros(len(ytr_f), bool)
    zs = []
    for s in SEEDS:
        prm = dict(SMALL_LGB, seed=s, bagging_seed=s, feature_fraction_seed=s,
                   data_random_seed=s)
        if iva.any():
            b0 = lgb.train(prm, lgb.Dataset(Xtr_f[itr], ytr_f[itr]), num_boost_round=3000,
                           valid_sets=[lgb.Dataset(Xtr_f[iva], ytr_f[iva])],
                           callbacks=[lgb.early_stopping(100, verbose=False)])
            nr = max(50, b0.best_iteration)
        else:
            nr = 200
        b = lgb.train(prm, lgb.Dataset(Xtr_f, ytr_f), num_boost_round=nr)
        zs.append(b.predict(Xva, raw_score=True))
    return np.mean(zs, axis=0)


def main():
    df = load_csv(TRAIN_CSV)
    cache = foldcache.load(FOLDS)
    t0 = time.time()

    P = {}
    for f in FOLDS:
        z_lgb, z_mlp, y, _ = cache[f]
        z_dep = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp     # 배포 스택 로짓

        mtr, mva = foldcache.SPLITS[f](df.season.to_numpy(), df.game_month.to_numpy())
        tr, va = df[mtr], df[mva]
        assert len(va) == len(y) and np.allclose(va[TARGET_COL].to_numpy(), y), "행 정렬 불일치"

        Xtr, ytr, _, Xva, yva, _, _ = foldcache.fold_data(df, f)

        gtr = tr["game_type"].to_numpy().astype(str)
        gva = va["game_type"].to_numpy().astype(str)
        # 전문가 학습셋: **2023년 이후 F만** (오염 구간 제외)
        mF = (gtr == "F") & (tr["season"].to_numpy() >= 2023)
        vF = gva == "F"

        z_spec = specialist(Xtr[mF].to_numpy(dtype=np.float32),
                            ytr[mF],
                            tr["game_month"].to_numpy(dtype=np.float64)[mF],
                            Xva.to_numpy(dtype=np.float32))
        P[f] = dict(z_dep=z_dep, z_spec=z_spec, y=yva, vF=vF)
        print(f"  fold {f}: 검증 {len(yva):,}행 (F {vF.sum():,}) · "
              f"전문가 학습 {int(mF.sum()):,}행  ({time.time()-t0:.0f}s)")

    def score(f, a):
        """F행만 전문가와 섞은 뒤 **전체** fold BSS (오라클 시프트)."""
        d = P[f]
        z = d["z_dep"].copy()
        z[d["vF"]] = (1 - a) * z[d["vF"]] + a * d["z_spec"][d["vF"]]
        return bss(d["y"], shift_to_prior(1 / (1 + np.exp(-z)), d["y"].mean()))

    def score_F(f, a):
        """F행만 떼어 그 자신의 prior로 시프트 — 서브그룹 진단용."""
        d = P[f]
        zf = (1 - a) * d["z_dep"][d["vF"]] + a * d["z_spec"][d["vF"]]
        yf = d["y"][d["vF"]]
        return bss(yf, shift_to_prior(1 / (1 + np.exp(-zf)), yf.mean()))

    print("\n  a 스윕 — 전체 fold BSS (a=0이 배포 스택 그대로)")
    print(f"  {'a':>6}" + "".join(f"{str(f):>12}" for f in FOLDS)
          + "".join(f"{'F행 ' + str(f):>13}" for f in FOLDS))
    for a in GRID:
        print(f"  {a:>6.2f}" + "".join(f"{score(f, a):>12.1f}" for f in FOLDS)
              + "".join(f"{score_F(f, a):>13.1f}" for f in FOLDS))

    inner = "2023H2"
    a_star = float(GRID[int(np.argmax([score(inner, a) for a in GRID]))])
    print(f"\n  {inner}에서 고른 a = {a_star:.2f}   (2024는 보지 않았다)")

    print("\n" + "=" * 66)
    print("  정직한 추정치 — a를 2024에 그대로 적용")
    print("=" * 66)
    for f in FOLDS:
        b0, b1 = score(f, 0.0), score(f, a_star)
        f0, f1 = score_F(f, 0.0), score_F(f, a_star)
        print(f"  fold {str(f):<8} 전체 {b0:>8.1f} → {b1:>8.1f}  ({b1-b0:>+7.1f})"
              f"    F행 {f0:>8.1f} → {f1:>8.1f}  ({f1-f0:>+7.1f})")

    a_or = float(GRID[int(np.argmax([score(2024, a) for a in GRID]))])
    b_or = score(2024, a_or) - score(2024, 0.0)
    b_hon = score(2024, a_star) - score(2024, 0.0)
    print(f"\n  낙관 편향의 크기: 2024 오라클 최적 a={a_or:.2f} → {b_or:+.1f}"
          f"   vs 정직 a={a_star:.2f} → {b_hon:+.1f}   차이 {b_or-b_hon:+.1f}")

    same = (score("2023H2", a_star) - score("2023H2", 0.0) > 0) == (b_hon > 0)
    print(f"  두 구간 부호 일치: {'O' if same else 'X'}")

    # ── 오차막대 — 쌍체 부트스트랩 (같은 행을 다시 뽑아 Δ를 반복 계산) ──
    print()
    print("=" * 66)
    print(f"  오차막대 — 행 재표집 {N_BOOT}회, a={a_star:.2f} 고정")
    print("=" * 66)
    rb = np.random.default_rng(7)
    for f in FOLDS:
        d = P[f]
        y = d["y"]
        z0 = d["z_dep"]
        z1 = z0.copy()
        z1[d["vF"]] = (1 - a_star) * z0[d["vF"]] + a_star * d["z_spec"][d["vF"]]
        # 시프트는 각 재표집 안에서 다시 푼다 — 레벨 보정도 표본에 딸린 것이므로
        n = len(y)
        boot = np.empty(N_BOOT)
        for t in range(N_BOOT):
            j = rb.integers(0, n, n)
            yj = y[j]
            p0 = shift_to_prior(1 / (1 + np.exp(-z0[j])), yj.mean())
            p1 = shift_to_prior(1 / (1 + np.exp(-z1[j])), yj.mean())
            boot[t] = bss(yj, p1) - bss(yj, p0)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        verdict = "0 제외" if lo > 0 else "0 포함 — 판별 불가"
        print(f"  fold {str(f):<8} Δ {boot.mean():>+7.2f}   95% [{lo:>+7.2f}, {hi:>+7.2f}]   {verdict}")


if __name__ == "__main__":
    main()
