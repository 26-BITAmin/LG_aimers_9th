"""`FSPEC_ALPHA` × `MLP_ALPHA` 2D 스윕 — 현재 배포(s27, α=0.60) 위에서.

왜 다시 재는가
    `FSPEC_ALPHA=0.15`는 §7.101에서 **α=0.40 시절 s24 위에서** 골랐다. 지금은 α=0.60이다.
    오늘 §7.65가 두 세대 낡아 부호가 뒤집힌 것을 겪었다 — 스택이 바뀌면 다시 잰다(§7.64).
    그리고 fspec.log의 블렌드 곡선(정점 a=0.50)은 **평가 fold 2024를 보고 고른 것**이라
    낙관 편향이 있다. 그게 배포값이 0.15인 이유다.

정직한 선택 규약 (§7.101과 동일하게 유지한다 — 여기서 바꾸면 비교가 깨진다)
    선택  fold "2023H2" 에서만 고른다. 2024는 쳐다보지 않는다.
    평가  그 값을 2024에 그대로 적용한다.
    참고  2024 오라클 최적도 같이 찍어 **낙관 편향의 크기**를 보인다.

2D인 이유
    "75%로 올리는 게 낫다"가 `MLP_ALPHA`를 가리키는지 `FSPEC_ALPHA`를 가리키는지
    모호하다. 둘을 같이 훑으면 한 번에 답이 나오고, 상호작용도 같이 보인다.

실행:  .venv/Scripts/python.exe -u src/fspec_a.py 2>&1 | tee experiments/fspec_a.log
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

INNER, OUTER = "2023H2", 2024          # 선택 / 평가
FOLDS = [INNER, OUTER]
MA = (0.40, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
FA = np.round(np.arange(0.0, 0.901, 0.05), 3)
SEEDS = (42, 7, 2024)
N_BOOT = 400

SMALL_LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15,
                 min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7,
                 bagging_freq=1, lambda_l2=10.0, max_bin=127, num_threads=6,
                 verbosity=-1)


def specialist(Xtr_f, ytr_f, month_f, Xva):
    """fspec_verify.py·alpha_s25.py와 **같은 함수**. 바꾸면 §7.101과 비교가 깨진다."""
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
    print(f"배포 MLP_ALPHA={MLP_ALPHA}  FSPEC_ALPHA={FSPEC_ALPHA}   캐시 {foldcache._key()}")
    df = load_csv(TRAIN_CSV)
    cache = foldcache.load(FOLDS)

    P = {}
    for f in FOLDS:
        z_lgb, z_mlp, y, _ = cache[f]
        mtr, mva = foldcache.SPLITS[f](df.season.to_numpy(), df.game_month.to_numpy())
        tr, va = df[mtr], df[mva]
        assert len(va) == len(y) and np.allclose(va[TARGET_COL].to_numpy(), y)
        vF = va["game_type"].to_numpy().astype(str) == "F"
        Xtr, ytr, _, Xva, _, _, _ = foldcache.fold_data(df, f)
        mF = ((tr["game_type"].to_numpy().astype(str) == "F")
              & (tr["season"].to_numpy() >= FSPEC_MIN_SEASON))
        z_spec = specialist(Xtr[mF].to_numpy(dtype=np.float32), ytr[mF],
                            tr["game_month"].to_numpy(dtype=np.float64)[mF],
                            Xva.to_numpy(dtype=np.float32))
        P[f] = dict(z_lgb=z_lgb, z_mlp=z_mlp, y=y, vF=vF, z_spec=z_spec)
        print(f"  fold {str(f):<8} 검증 {len(y):,}행 (F {vF.sum():,})  "
              f"전문가 학습 {int(mF.sum()):,}행  ({time.time() - t0:.0f}s)")

    def blend(f, ma, fa):
        d = P[f]
        z = (1 - ma) * d["z_lgb"] + ma * d["z_mlp"]
        if fa > 0:
            z = z.copy()
            z[d["vF"]] = (1 - fa) * z[d["vF"]] + fa * d["z_spec"][d["vF"]]
        return z

    def score(f, ma, fa, rows=None):
        z, y = blend(f, ma, fa), P[f]["y"]
        if rows is not None:
            z, y = z[rows], y[rows]
        return bss(y, shift_to_prior(1 / (1 + np.exp(-z)), y.mean()))

    for f in FOLDS:
        star = "  ← 선택 fold" if f == INNER else "  ← 평가 fold (선택에 쓰지 않음)"
        print("\n" + "=" * 84)
        print(f"  fold {f}   전체 BSS{star}")
        print("=" * 84)
        print(f"  {'fa\\ma':>7}" + "".join(f"{m:>11.2f}" for m in MA))
        for fa in FA:
            row = [score(f, m, fa) for m in MA]
            mk = "  ← 배포" if abs(fa - FSPEC_ALPHA) < 1e-9 else ""
            print(f"  {fa:>7.2f}" + "".join(f"{v:>11.1f}" for v in row) + mk)

    # ── 정직한 선택: INNER 에서 고르고 OUTER 에 적용 ──────────────
    grid = [(m, fa) for m in MA for fa in FA]
    m_s, fa_s = max(grid, key=lambda g: score(INNER, *g))
    base = score(OUTER, MLP_ALPHA, FSPEC_ALPHA)

    print("\n" + "=" * 84)
    print("  정직한 추정 — 선택은 2023H2, 평가는 2024")
    print("=" * 84)
    print(f"  선택된 (ma, fa) = ({m_s:.2f}, {fa_s:.2f})")
    print(f"  {'설정':<28}{INNER:>12}{str(OUTER):>12}{'Δ vs 배포(2024)':>18}")
    cands = [(MLP_ALPHA, FSPEC_ALPHA, f"현 config ({MLP_ALPHA:.2f}, {FSPEC_ALPHA:.2f})"),
             (m_s, fa_s, "2023H2 선택"),
             (0.60, FSPEC_ALPHA, "ma=0.60 (s27, LB 1077.02)"),
             (0.75, FSPEC_ALPHA, "ma=0.75 (s28)"),
             (0.60, 0.75, "fa=0.75 (ma=0.60)"),
             (0.60, 0.50, "fa=0.50 (fspec.log 정점)")]
    for m, fa, tag in cands:
        print(f"  {tag:<28}{score(INNER, m, fa):>12.1f}{score(OUTER, m, fa):>12.1f}"
              f"{score(OUTER, m, fa) - base:>+18.2f}")

    m_o, fa_o = max(grid, key=lambda g: score(OUTER, *g))
    print(f"\n  2024 오라클 최적 = ({m_o:.2f}, {fa_o:.2f}) → {score(OUTER, m_o, fa_o) - base:+.2f}")
    print(f"  낙관 편향 크기   = {score(OUTER, m_o, fa_o) - score(OUTER, m_s, fa_s):+.2f}")

    # ── 오차막대 — 배포 대비 선택안, 쌍체 부트스트랩 ────────────
    print("\n" + "=" * 84)
    print(f"  오차막대 — 배포(0.60,0.15) → 선택안({m_s:.2f},{fa_s:.2f}), 재표집 {N_BOOT}회")
    print("=" * 84)
    rb = np.random.default_rng(7)
    for f in FOLDS:
        y = P[f]["y"]
        z0, z1 = blend(f, MLP_ALPHA, FSPEC_ALPHA), blend(f, m_s, fa_s)
        n, boot = len(y), np.empty(N_BOOT)
        for t in range(N_BOOT):
            j = rb.integers(0, n, n)
            yj = y[j]
            boot[t] = (bss(yj, shift_to_prior(1 / (1 + np.exp(-z1[j])), yj.mean()))
                       - bss(yj, shift_to_prior(1 / (1 + np.exp(-z0[j])), yj.mean())))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        tag = "선택 fold — 참고용" if f == INNER else "평가 fold — 이게 판정자"
        print(f"  fold {str(f):<8} Δ {boot.mean():>+7.2f}  95% [{lo:>+7.2f}, {hi:>+7.2f}]"
              f"  {'0 제외' if lo > 0 or hi < 0 else '0 포함'}   {tag}")

    print(f"\n총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
