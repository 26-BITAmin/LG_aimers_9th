"""`MLP_ALPHA` 스윕 — s25 스택(F 전문가 켬) 위에서. §7.65의 s24 재측정 + 상호작용.

이 스크립트가 답하는 질문은 **"α를 얼마로 할까"가 아니다.**
그건 오늘 LB가 이미 답했고(α=0.60이 s24 대비 +6.98), 홀드아웃은 이 축에서
**부호를 틀린다**는 것이 s26에서 확인됐다. 홀드아웃으로 α를 고르면 안 된다.

답하는 질문은 하나다:

    **F 전문가를 켜면 α 최적점이 움직이는가?**

움직이면 → p_a60에서 이긴 α를 s25에 그대로 옮길 수 없다. s25 기반 재빌드 필요.
안 움직이면 → 두 층이 독립. 옮겨도 된다.

이건 **상대적** 질문이라 홀드아웃의 레벨·부호 편향이 상당 부분 상쇄된다.
(같은 fold·같은 로짓 위에서 fa=0과 fa=0.15의 곡선을 나란히 비교한다.)

설계
    Arm A  fa=0.00 (s24 스택)   folds 2023 · 2024 · 2023H2   ← §7.65 재현 + 0.85까지 확장
    Arm B  fa=0.15 (s25 스택)   folds 2023H2 · 2024
           ⚠️ fold 2023은 학습셋이 season<2023이라 "2023+ F"가 비어 있다 →
              F 전문가를 만들 수 없다. fspec_verify.py가 이 fold를 뺀 이유와 같다.

    상호작용 = (Arm B의 α*) − (Arm A의 α*)   공유 fold 2개에서

블렌드 순서는 train.py·predict.py와 같다: MLP 블렌드 → F 전문가 → (δ 보정 생략) → 시프트.
foldcache는 δ 보정축 없이 artifacts를 만든다 — 이 저장소 다른 스크립트와 같은 관례다.

실행:  .venv/Scripts/python.exe -u src/alpha_s25.py 2>&1 | tee experiments/alpha_s25.log
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
from config import FSPEC_ALPHA, FSPEC_MIN_SEASON, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

FOLDS_A = [2023, 2024, "2023H2"]      # fa=0 — 전문가 불필요
FOLDS_B = ["2023H2", 2024]            # fa=0.15 — 전문가 필요
ALPHA = np.round(np.arange(0.10, 0.851, 0.05), 3)
SEEDS = (42, 7, 2024)
N_BOOT = 400

SMALL_LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15,
                 min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7,
                 bagging_freq=1, lambda_l2=10.0, max_bin=127, num_threads=6,
                 verbosity=-1)


def specialist(Xtr_f, ytr_f, month_f, Xva):
    """fspec_verify.py와 **같은 함수**다. 여기서 바꾸면 §7.101과 비교가 깨진다."""
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
        zs.append(lgb.train(prm, lgb.Dataset(Xtr_f, ytr_f),
                            num_boost_round=nr).predict(Xva, raw_score=True))
    return np.mean(zs, axis=0)


def main():
    t0 = time.time()
    print(f"배포값 MLP_ALPHA={MLP_ALPHA}  FSPEC_ALPHA={FSPEC_ALPHA}  "
          f"FSPEC_MIN_SEASON={FSPEC_MIN_SEASON}")
    print(f"캐시 키 {foldcache._key()}\n")

    df = load_csv(TRAIN_CSV)
    cache = foldcache.load(sorted(set(FOLDS_A) | set(FOLDS_B), key=str))

    P = {}
    for f in FOLDS_A:
        z_lgb, z_mlp, y, _ = cache[f]
        mtr, mva = foldcache.SPLITS[f](df.season.to_numpy(), df.game_month.to_numpy())
        va = df[mva]
        assert len(va) == len(y) and np.allclose(va[TARGET_COL].to_numpy(), y), \
            f"fold {f}: 행 정렬 불일치"
        vF = va["game_type"].to_numpy().astype(str) == "F"
        P[f] = dict(z_lgb=z_lgb, z_mlp=z_mlp, y=y, vF=vF, z_spec=None)
        print(f"  fold {str(f):<8} 검증 {len(y):,}행 (F {vF.sum():,} = {vF.mean():.1%})"
              f"   로짓 상관(lgb,mlp) {np.corrcoef(z_lgb, z_mlp)[0, 1]:.3f}")

    print()
    for f in FOLDS_B:
        mtr, _ = foldcache.SPLITS[f](df.season.to_numpy(), df.game_month.to_numpy())
        tr = df[mtr]
        Xtr, ytr, _, Xva, _, _, _ = foldcache.fold_data(df, f)
        gtr = tr["game_type"].to_numpy().astype(str)
        mF = (gtr == "F") & (tr["season"].to_numpy() >= FSPEC_MIN_SEASON)
        assert mF.sum() > 1000, f"fold {f}: F 전문가 학습셋이 {mF.sum()}행뿐"
        P[f]["z_spec"] = specialist(Xtr[mF].to_numpy(dtype=np.float32), ytr[mF],
                                    tr["game_month"].to_numpy(dtype=np.float64)[mF],
                                    Xva.to_numpy(dtype=np.float32))
        print(f"  fold {str(f):<8} F 전문가 학습 {int(mF.sum()):,}행  ({time.time() - t0:.0f}s)")

    def blend(f, a, fa):
        d = P[f]
        z = (1 - a) * d["z_lgb"] + a * d["z_mlp"]
        if fa > 0:
            if d["z_spec"] is None:
                return None
            z = z.copy()
            z[d["vF"]] = (1 - fa) * z[d["vF"]] + fa * d["z_spec"][d["vF"]]
        return z

    def score(f, a, fa, rows=None):
        z = blend(f, a, fa)
        if z is None:
            return None
        y = P[f]["y"]
        if rows is not None:
            z, y = z[rows], y[rows]
        return bss(y, shift_to_prior(1 / (1 + np.exp(-z)), y.mean()))

    # ── 스윕 ────────────────────────────────────────────────────────
    for fa, folds, tag in ((0.0, FOLDS_A, "Arm A · fa=0.00 (s24 스택)"),
                           (FSPEC_ALPHA, FOLDS_B, f"Arm B · fa={FSPEC_ALPHA} (s25 스택)")):
        print("\n" + "=" * 78)
        print(f"  {tag}   —   전체 fold BSS (오라클 시프트)")
        print("=" * 78)
        base = {f: score(f, MLP_ALPHA, fa) for f in folds}
        print(f"  {'a':>6}" + "".join(f"{str(f):>13}" for f in folds)
              + "   |" + "".join(f"{'d vs .40':>11}" for f in folds))
        for a in ALPHA:
            row = [score(f, a, fa) for f in folds]
            mark = "  <- 배포" if abs(a - MLP_ALPHA) < 1e-9 else (
                   "  <- LB 승자" if abs(a - 0.60) < 1e-9 else "")
            print(f"  {a:>6.2f}" + "".join(f"{v:>13.1f}" for v in row)
                  + "   |" + "".join(f"{v - base[f]:>+11.2f}"
                                     for v, f in zip(row, folds)) + mark)
        for f in folds:
            vals = [score(f, a, fa) for a in ALPHA]
            print(f"    fold {str(f):<8} a* = {ALPHA[int(np.argmax(vals))]:.2f}"
                  f"   (배포 0.40 대비 {max(vals) - base[f]:+.1f})")

    # ── 상호작용: α* 가 움직이는가 ────────────────────────────────
    print("\n" + "=" * 78)
    print("  상호작용 — F 전문가를 켜면 a* 가 움직이는가")
    print("=" * 78)
    print(f"  {'fold':<10}{'a* (fa=0)':>12}{'a* (fa=.15)':>14}{'이동':>8}"
          f"{'dfspec@.40':>13}{'dfspec@.60':>13}")
    for f in FOLDS_B:
        vA = [score(f, a, 0.0) for a in ALPHA]
        vB = [score(f, a, FSPEC_ALPHA) for a in ALPHA]
        aA, aB = ALPHA[int(np.argmax(vA))], ALPHA[int(np.argmax(vB))]
        d40 = score(f, 0.40, FSPEC_ALPHA) - score(f, 0.40, 0.0)
        d60 = score(f, 0.60, FSPEC_ALPHA) - score(f, 0.60, 0.0)
        print(f"  {str(f):<10}{aA:>12.2f}{aB:>14.2f}{aB - aA:>+8.2f}"
              f"{d40:>+13.2f}{d60:>+13.2f}")
    print("\n  dfspec@.40 ~= dfspec@.60  이면 두 층은 독립 — a를 옮겨도 된다.")
    print("  dfspec@.60 이 뚜렷이 작으면 a↑가 F 전문가의 이득을 미리 먹은 것이다.")

    # ── F행/R행 분해 — 간섭이 있다면 F행에만 있어야 한다 ────────────
    print("\n" + "=" * 78)
    print("  F행 / R행 분해 (각 부분집합 자기 prior로 시프트)")
    print("=" * 78)
    for f in FOLDS_B:
        vF = P[f]["vF"]
        print(f"\n  fold {f}   F {vF.sum():,}행 / R {(~vF).sum():,}행")
        print(f"    {'a':>6}{'F행 fa=0':>12}{'F행 fa=.15':>13}{'d':>9}"
              f"{'R행 (fa 무관)':>16}")
        for a in ALPHA:
            f0 = score(f, a, 0.0, rows=vF)
            f1 = score(f, a, FSPEC_ALPHA, rows=vF)
            r0 = score(f, a, 0.0, rows=~vF)
            print(f"    {a:>6.2f}{f0:>12.1f}{f1:>13.1f}{f1 - f0:>+9.1f}{r0:>16.1f}")

    # ── 오차막대 — α .40→.60 의 쌍체 부트스트랩 (fa=.15 위에서) ────
    print("\n" + "=" * 78)
    print(f"  오차막대 — a 0.40 -> 0.60, fa={FSPEC_ALPHA} 고정, 행 재표집 {N_BOOT}회")
    print("=" * 78)
    rb = np.random.default_rng(7)
    for f in FOLDS_B:
        y = P[f]["y"]
        z0, z1 = blend(f, 0.40, FSPEC_ALPHA), blend(f, 0.60, FSPEC_ALPHA)
        n = len(y)
        boot = np.empty(N_BOOT)
        for t in range(N_BOOT):
            j = rb.integers(0, n, n)
            yj = y[j]
            boot[t] = (bss(yj, shift_to_prior(1 / (1 + np.exp(-z1[j])), yj.mean()))
                       - bss(yj, shift_to_prior(1 / (1 + np.exp(-z0[j])), yj.mean())))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  fold {str(f):<8} d {boot.mean():>+8.2f}   95% [{lo:>+8.2f}, {hi:>+8.2f}]"
              f"   {'0 제외' if lo > 0 or hi < 0 else '0 포함 — 판별 불가'}")

    print(f"\n총 {time.time() - t0:.0f}s")
    print("\n[주의] 이 표로 a를 고르지 않는다. s26에서 홀드아웃이 이 축의 부호를")
    print("       틀린다는 것이 LB로 확인됐다. 여기서 읽을 것은 **상호작용**뿐이다.")


if __name__ == "__main__":
    main()
