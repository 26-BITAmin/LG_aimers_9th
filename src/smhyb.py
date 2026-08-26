"""§7.162c — 하이브리드: **LGBM 에만** 증강, MLP/CatBoost 는 원래대로 (사용자 지적)

§7.162b 는 증강을 세 학습기 **전부**에 걸었다. 그건 사용자가 말한 구성이 아니다.
단독 로짓이 그 차이를 설명한다 — LGBM base 917.5 → SMOTE 926.0 (**+8.5**) 인데
MLP 는 917.3 → 905.2 (**−12.1**) 다. 증강은 LGBM 을 돕고 MLP 를 망친다.
멤버별 학습 선택은 실제로 가능한 배포다 (gmix.py 의 hybrid 와 같은 형태).

두 가지를 잰다.

① **위약 하이브리드** — 관문이다. §7.162 에서 위약(무결한 pre-2023 R)이 후보보다
   높았다. 하이브리드에서도 같은 +1.3 이 나오면 이득은 F 와 무관하다.

② **α 재측정** (사용자 지적) — `MLP_ALPHA=0.745` 는 **증강 없는 LGBM** 에 맞춰
   **LB 로** 확정한 값이다(홀드아웃 oracle 은 0.57~0.82 로 빗나갔다). LGBM 팔의
   품질과 MLP 와의 상관이 바뀌면 최적 α 가 이동한다.
   ⚠️ 하지만 **평가 fold 에서 α 를 고르면 안 된다** — §7.101 이 그 낙관 편향을
      +2.0 으로 실측했다. 여기서는 곡선을 **보여주기만** 하고, 판정은 고정
      α=0.745 에서 한다. α 변경은 별도 결정이고 별도 LB 프로브가 필요하다.

실행:  .venv/Scripts/python.exe -u src/smhyb.py 2>&1 | tee experiments/smhyb.log
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

import jointbeta as J
from config import (ASOF_AXES, CAT_ALPHA, CTX_AXES, FSPEC_ALPHA, MLP_ALPHA,
                    MLP_SEEDS, MLP_VARIANTS, TARGET_COL, TRAIN_CSV)
from data import load_csv
from hbeta2 import train_fspec
from predict import PROB_CENTER, PROB_SCALE
from smotenc import prep
from validation import bss, shift_to_prior

FOLD = 2024
C = Path(__file__).parent.parent / "experiments" / "cache"
ALPHAS = (0.50, 0.60, 0.65, 0.70, 0.745, 0.80, 0.85, 0.90)
B_BOOT = 2000          # §7.161 메모: 경계선 게이트는 B 에 민감하다. 넉넉히 준다.


def main():
    df = load_csv(TRAIN_CSV)
    tr, va, w, X, art = prep(df, FOLD)
    Xva = X(va)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    isF = va["game_type"].to_numpy().astype(str) == "F"
    z_fs, _ = train_fspec(tr, X, Xva)
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    Xax, _, _, _ = J.axis_matrix(df, df["season"].to_numpy(), FOLD)
    ax = Xax @ dep

    def ld(k, n):
        p = C / f"sfull_f{FOLD}_{k}_{n}.npz"
        return np.load(p)["z"] if p.exists() else None

    mb = np.mean([ld("base", f"mlp_{v}_{s}") for v, (c, n) in MLP_VARIANTS.items()
                  for s in MLP_SEEDS[:n]], axis=0)
    cb = ld("base", "cat")
    arms = {k: ld(k, "lgb") for k in ("base", "SMOTE", "PLACEBO")}
    arms = {k: v for k, v in arms.items() if v is not None}
    print("LGBM 팔: " + ", ".join(arms) + f"   (MLP/CatBoost 는 base = 원래대로)")

    def zf(zl, alpha):
        """배포 순서 그대로. MLP·CatBoost 는 증강 없는 base 를 쓴다."""
        z = (1 - CAT_ALPHA) * ((1 - alpha) * zl + alpha * mb) + CAT_ALPHA * cb
        z = z.copy()
        z[isF] = (1 - FSPEC_ALPHA) * z[isF] + FSPEC_ALPHA * z_fs[isF]
        return z + ax

    def sc(z, yy, idx=None):
        if idx is not None:
            z, yy = z[idx], yy[idx]
        m = yy.mean()
        p = shift_to_prior(1 / (1 + np.exp(-z)), m)
        return bss(yy, shift_to_prior(PROB_CENTER + PROB_SCALE * (p - PROB_CENTER), m))

    # ── ① 고정 α=MLP_ALPHA 에서의 판정 + 클러스터 부트스트랩 ───────────
    codes, uniq = pd.factorize(va["pitcher_id"].to_numpy())
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    seen = np.where(np.diff(bnd) > 0)[0]
    rng = np.random.default_rng(42)
    boots = [np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                             rng.choice(seen, size=len(seen), replace=True)])
             for _ in range(B_BOOT)]

    z0 = zf(arms["base"], MLP_ALPHA)
    base0 = sc(z0, yva)
    print(f"\n{'='*80}\n① 고정 α={MLP_ALPHA} (배포값) — base {base0:.1f}"
          f"   투수 {len(seen)}명 · 부트스트랩 {B_BOOT}회\n{'='*80}")
    print("  %-12s %9s %8s %22s %10s" % ("LGBM 팔", "BSS", "Δ", "95% 구간", "P(Δ>0)"))
    for k, zl in arms.items():
        if k == "base":
            continue
        z1 = zf(zl, MLP_ALPHA)
        d = np.array([sc(z1, yva, ix) - sc(z0, yva, ix) for ix in boots])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print("  %-12s %9.1f %+8.2f   [%+7.2f, %+7.2f] %10.3f"
              % (k, sc(z1, yva), sc(z1, yva) - base0, lo, hi, (d > 0).mean()))

    # ── ② α 곡선 — **보여주기만** 한다 (평가 fold 에서 고르지 않는다) ──
    print(f"\n{'='*80}\n② α 곡선 ⚠️ 평가 fold 에서 α 를 고르면 §7.101 의 낙관 편향(+2.0)을 먹는다"
          f"\n{'='*80}")
    print("  %-8s" % "α" + "".join("%12s" % k for k in arms))
    best = {k: (-1e9, None) for k in arms}
    for a in ALPHAS:
        row = "  %-8.3f" % a
        for k, zl in arms.items():
            v = sc(zf(zl, a), yva)
            row += "%12.1f" % v
            if v > best[k][0]:
                best[k] = (v, a)
        print(row + ("   ← 배포값" if a == MLP_ALPHA else ""))
    print("\n  oracle 최적:  " + "  ".join(f"{k} α={b[1]:.3f}({b[0]:.1f})"
                                         for k, b in best.items()))

    # 로짓 상관 — α 이동의 기전
    print("\n  corr(z_lgb, z_mlp):  " + "  ".join(
        f"{k} {np.corrcoef(zl, mb)[0,1]:.4f}" for k, zl in arms.items()))


if __name__ == "__main__":
    main()
