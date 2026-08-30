"""§7.210 — 콜드 패치를 **챔피언 규격 그대로** 재채점한다.

§7.209b 가 dim 8 은 맞췄지만 사후 보정·레벨을 우리 config 로 썼다.
`src/hjstruct.py` 가 hj03 을 직접 열어 찾아낸 불일치:

    δ hand β    hj03 **1.054**  ≠  우리 config 1.875   (meta.json 은 1.5 — 셋 다 다름)
    asof 축     hj03 3개 (**gap_ball −0.0038 포함**)  ≠  우리 2개
    레벨        hj03 `p = 0.4801 + 1.05*(p−0.4801) + shift`  ≠  우리 shift_to_prior

여기서는 그 셋을 전부 챔피언 값으로 맞춘다. **재학습은 없다** — dim 8 가중치
(`bcoldw_d8_*`)를 그대로 쓰고 축·레벨만 교체한다.

⚠️ β 만 챔피언 값이고 **δ 표는 시즌 < fold 로 재적합**한다. 배포 표를 쓰면
   2024 타깃이 fold2024 검증에 들어간다.

⚠️ 아직 안 맞는 것 (재학습·미보유 필요): LGBM 5시드(챔피언 20 vs 우리 12) ·
   라우팅 계층 29파일(general_lgb·compact·user_fspec·routing_artifacts).

실행:  .venv/Scripts/python.exe -u src/bcoldchamp.py 2>&1 | tee experiments/bcoldchamp.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import champspec as CS
import deploy29 as D
from bcoldfold import ARMS, EMB4_SEEDS, patched_z, prep
from config import MLP_ALPHA, ROOT, TARGET_COL, TRAIN_CSV
from data import load_csv
from config import SEED_ENSEMBLE
from tmeval import W_CAT, load_family

# 챔피언은 LGBM 4변형 x **5시드 = 20멤버** (hjstruct 실측). 우리 관례는 3시드다.
LGB5_SEEDS = SEED_ENSEMBLE[:5]
LGB_VARIANTS5 = ("base", "wide", "extra", "lowff")


def lgb20(fold):
    """챔피언과 같은 20멤버 평균 (initdepm 캐시)."""
    zs = []
    for v in LGB_VARIANTS5:
        for sd in LGB5_SEEDS:
            q = CACHE / f"initdepm_fold{fold}_{v}_{sd}.npz"
            assert q.exists(), f"없음 {q.name} — src/lgb5seed.py 를 먼저"
            zs.append(np.load(q)["z"])
    assert len(zs) == 20, len(zs)
    return np.mean(zs, axis=0)
from validation import bss

CACHE = ROOT / "experiments" / "cache"
FOLDS = [2023, 2024]
P_DIM = 8
RNG_SEED = 42
B_BOOT = 4000


def boot(yv, cl, n_id, ybar, p_new, p_ref, rng, B=B_BOOT):
    ref = ybar * (1 - ybar)
    n = np.bincount(cl, minlength=n_id)
    s0 = np.bincount(cl, weights=(p_ref - yv) ** 2, minlength=n_id)
    s1 = np.bincount(cl, weights=(p_new - yv) ** 2, minlength=n_id)
    seen = np.where(n > 0)[0]
    draws = np.empty(B)
    for b in range(B):
        pick = rng.choice(seen, size=len(seen), replace=True)
        N = n[pick].sum()
        draws[b] = 1e5 * (s0[pick].sum() - s1[pick].sum()) / (N * ref)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return lo, hi, float(np.mean(draws > 0))


def main():
    rng = np.random.default_rng(RNG_SEED)
    print("§7.210 — 콜드 패치, **챔피언 규격**으로 재채점")
    print(f"  δ ctx  {CS.CHAMP_CTX}")
    print(f"  δ asof {CS.CHAMP_ASOF}")
    print(f"  레벨   p = {CS.PROB_CENTER} + {CS.PROB_SCALE}*(p-{CS.PROB_CENTER}) + shift")
    print("  LGBM 20멤버 (4변형 x 5시드) — 챔피언과 동일")
    print("  ⚠️ 미반영: 라우팅 계층 29파일\n")

    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()

    for fold in FOLDS:
        st = D.state(df, s, y=df[TARGET_COL].to_numpy(dtype=np.float64), fold=fold,
                     check=False)
        mm, yv = st["mm"], st["yv"]
        ybar = float(yv.mean())
        cl, n_id = st["cl"], st["n_id"]

        # ── 챔피언 δ 축 (β 는 챔피언, 표는 시즌<fold 재적합) ──────
        Xax, names, betas, _va = CS.axis_matrix(df, s, fold)
        ax = (Xax[mm] @ betas)
        print("=" * 92)
        print(f"■ fold {fold}   축 {names}  β {np.round(betas, 4).tolist()}")
        print("=" * 92)

        # ── dim8 가중치 재사용 (재학습 없음) ─────────────────────
        _art, _Xtr, _ytr, _w2, _aux, Xva = prep(df, fold)
        import mlp as _mlp
        enc = _mlp.fit_encoder(_Xtr, _art)
        wts = []
        for sd in EMB4_SEEDS:
            p = CACHE / f"bcoldw_d{P_DIM}_f{fold}_emb4_{sd}.npz"
            assert p.exists(), f"dim8 가중치 없음: {p.name} — src/bcoldfold.py 를 먼저"
            wts.append(dict(np.load(p, allow_pickle=True)))
        del _Xtr

        cached = load_family("base", "mlp", fold)
        z_lgb12 = load_family("base", "lgb", fold).mean(axis=1)   # 12멤버 (대조)
        z_lgb = lgb20(fold)                                       # 20멤버 (챔피언)
        print(f"  LGBM 20멤버 사용 (12멤버 대비 corr "
              f"{np.corrcoef(z_lgb, z_lgb12)[0,1]:.6f})")
        z_cat = load_family("base", "cat", fold).mean(axis=1)
        z_noid = cached[:, 5:].mean(axis=1)

        def score(zemb4, mode):
            zm = 0.875 * zemb4.mean(axis=1) + 0.125 * z_noid
            z = (1 - W_CAT) * ((1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * zm) + W_CAT * z_cat
            p_raw = 1.0 / (1.0 + np.exp(-(z[mm] + ax)))
            if mode == "champ":                      # 아핀 + 레벨을 ȳ 로 고정
                sh = CS.solve_shift(p_raw, ybar)
                p = np.clip(CS.prob_transform(p_raw, sh), 1e-6, 1 - 1e-6)
            else:                                    # 아핀 없음 (§7.209b 와 같은 처리)
                from validation import shift_to_prior
                p = shift_to_prior(p_raw, ybar)
            return bss(yv, p), p

        Z = {a: (np.stack([patched_z(w, enc, Xva, a) for w in wts], axis=1))
             for a in ARMS}
        for mode, lab in (("champ", "챔피언 아핀(1.05)"), ("plain", "아핀 없음")):
            b_s, b_p = score(Z["base"], mode)
            print(f"  [{lab}]  base {b_s:.2f}")
            print(f"    {'팔':<8}{'BSS':>10}{'Δ':>9}{'95% CI':>21}{'P(Δ>0)':>9}")
            for a in ARMS[1:]:
                sc, p = score(Z[a], mode)
                lo, hi, pg = boot(yv, cl, n_id, ybar, p, b_p, rng)
                print(f"    {a:<8}{sc:>10.2f}{sc-b_s:>+9.2f}   [{lo:+7.2f},{hi:+7.2f}]"
                      f"{pg:>9.3f}")
        del Xva, wts, Z
        print()

    print("=" * 92)
    print("§7.209b 대비 무엇이 달라졌나")
    print("=" * 92)
    print("  · δ hand β 1.875 → **1.054** (챔피언 아티팩트 실측)")
    print("  · asof 축 2개 → **3개** (gap_ball −0.0038 추가)")
    print("  · 레벨: shift_to_prior → **아핀(center .4801, scale 1.05) + shift**")
    print("  · 두 모드를 같이 찍어 아핀이 Δ 를 얼마나 키우는지 분리한다 (이론상 ×1.05)")


if __name__ == "__main__":
    main()
