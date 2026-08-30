"""층별 **재학습 뽑기 잡음**을 배포 등가 블렌드에서 잰다 (§7.198 확장).

§7.198 은 emb4 5시드만 흔들었다. 그런데 역대 채택의 다수는 **LGBM 코어 재학습**
(s40·s41·s45·s47류)이고 CatBoost 이식(s37·s43)도 있다. §7.128 은 *"노이즈의 99.5%가
MLP"* 라고 했지만 그건 멤버 수가 다른 층들의 합산 장부였다 — **변경이 건드리는 층마다
Δ 에 붙는 잡음이 다르다.** 층별로 따로 재야 과거 판정에 정직한 오차막대를 붙일 수 있다.

하네스: `hillclimb.load_library` 와 같은 32멤버 (LGBM 4변형×3시드 · MLP emb4 5+noid 3 ·
CatBoost 4변형×3시드), 가중은 B1 = 배포 등가 (.875/.125 변형가중, W_CAT .10).
캐시만 읽는다 — 재학습 0회.

배포 시드 수는 층마다 다르다: LGBM 5/변형(=20), CatBoost 3/변형(=12), MLP emb4 5·noid 3.
관측이 3시드뿐인 층은 변형 내 부트스트랩으로 k=배포 로 환산한다.

실행: .venv/Scripts/python.exe -u src/layerdisp.py 2>&1 | tee experiments/layerdisp.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import deploy29 as D                                            # noqa: E402
import hillclimb as H                                           # noqa: E402
from config import MLP_ALPHA                                    # noqa: E402
from validation import bss, shift_to_prior                      # noqa: E402

B = 1200

# (층, 변형 목록, 관측 시드, 배포 시드 수, 층 가중)
LAYERS = {
    "LGBM": dict(var=H.LGB_VARIANTS, seeds=H.LGB_SEEDS, k_dep=5,
                 w=(1 - H.W_CAT) * (1 - MLP_ALPHA)),
    "CatBoost": dict(var=H.CAT_VARIANTS, seeds=H.CAT_SEEDS, k_dep=3, w=H.W_CAT),
    "MLP": dict(var=("emb4", "noid"), seeds=None, k_dep=None,
                w=(1 - H.W_CAT) * MLP_ALPHA),
}


def members(fold):
    """{(층, 변형): [시드별 z]} — 캐시만."""
    M = {}
    for v in H.LGB_VARIANTS:
        M[("LGBM", v)] = [np.load(D.CACHE / f"initdepm_fold{fold}_{v}_{sd}.npz")["z"]
                          for sd in H.LGB_SEEDS]
    for v in H.CAT_VARIANTS:
        M[("CatBoost", v)] = [np.load(D.CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz")["z"]
                              for sd in H.CAT_SEEDS]
    with np.load(D.CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
        mem = d0["members"]
    M[("MLP", "emb4")] = [mem[i] for i in range(5)]
    M[("MLP", "noid")] = [mem[i] for i in range(5, 8)]
    return M


def layer_z(M, layer, pick=None):
    """층 로짓 = 변형 평균(변형 안은 시드 평균). pick[(층,변형)] 이 주어지면 그 인덱스로."""
    if layer == "MLP":
        e = np.mean([M[("MLP", "emb4")][i] for i in
                     (pick[("MLP", "emb4")] if pick else range(5))], axis=0)
        n = np.mean([M[("MLP", "noid")][i] for i in
                     (pick[("MLP", "noid")] if pick else range(3))], axis=0)
        return 0.875 * e + 0.125 * n            # B1 변형가중
    vs = []
    for v in LAYERS[layer]["var"]:
        idx = pick[(layer, v)] if pick else range(len(M[(layer, v)]))
        vs.append(np.mean([M[(layer, v)][i] for i in idx], axis=0))
    return np.mean(vs, axis=0)


def main():
    df, s, y = D.load_df()
    rng = np.random.default_rng(0)
    summary = {}
    for fold in (2023, 2024):
        st = D.state(df, s, y, fold)
        mask, ax, yv = st["mm"], st["Xax"] @ st["dep"], st["yv"]
        ybar = float(yv.mean())

        def score(zfull):                       # Newton — shift_to_prior 와 동치
            z = zfull[mask] + ax
            d = 0.0
            for _ in range(40):
                p = 1 / (1 + np.exp(-(z + d)))
                step = (ybar - p.mean()) / np.mean(p * (1 - p))
                d += step
                if abs(step) < 1e-13:
                    break
            return bss(yv, 1 / (1 + np.exp(-(z + d))))

        M = members(fold)
        base_parts = {L: layer_z(M, L) for L in LAYERS}
        zfull = sum(LAYERS[L]["w"] * base_parts[L] for L in LAYERS)
        base = score(zfull)
        ref = bss(yv, shift_to_prior(1 / (1 + np.exp(-(zfull[mask] + ax))), ybar))
        print(f"\n=== fold {fold}  n={len(yv):,}  B1 기준 {base:.2f} "
              f"(shift_to_prior 대조 {ref:.2f}, Δ {base-ref:+.2e}) ===")
        assert abs(base - ref) < 1e-6

        # 배관: LGBM 12멤버 평균이 배포 캐시와 같은가
        mad = float(np.abs(layer_z(M, "LGBM")
                           - np.load(D.CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]).max())
        print(f"  LGBM 12멤버 재구성 max|Δ| = {mad:.1e}")

        print(f"  {'층':<9}{'k(관측)':>8}{'k(배포)':>8}{'sd@k=1':>9}"
              f"{'sd@배포':>9}{'SD(Δ)':>8}   (B={B})")
        for L, cfg in LAYERS.items():
            vlist = cfg["var"]
            nobs = {v: len(M[(L, v)]) for v in vlist}
            kdep = {v: (5 if v == "emb4" else 3) for v in vlist} if L == "MLP" \
                else {v: cfg["k_dep"] for v in vlist}

            def draw(k_of):
                pick = {(L, v): rng.integers(0, nobs[v], k_of[v]) for v in vlist}
                parts = dict(base_parts)
                parts[L] = layer_z(M, L, pick)
                return score(sum(LAYERS[x]["w"] * parts[x] for x in LAYERS))

            sd1 = np.std([draw({v: 1 for v in vlist}) for _ in range(B // 3)], ddof=1)
            sdk = np.std([draw(kdep) for _ in range(B)], ddof=1)
            print(f"  {L:<9}{min(nobs.values()):>8}{min(kdep.values()):>8}"
                  f"{sd1:>9.2f}{sdk:>9.2f}{np.sqrt(2)*sdk:>8.2f}")
            summary[(fold, L)] = (sd1, sdk)

    print("\n=== 요약: 그 층을 재학습한 변경의 Δ 에 붙는 잡음 SD (BSS) ===")
    print(f"  {'층':<9}{'f2023':>10}{'f2024':>10}{'2fold 결합':>12}")
    for L in LAYERS:
        a = np.sqrt(2) * summary[(2023, L)][1]
        b = np.sqrt(2) * summary[(2024, L)][1]
        c = 1 / np.sqrt(1 / a**2 + 1 / b**2)
        print(f"  {L:<9}{a:>10.2f}{b:>10.2f}{c:>12.2f}")


if __name__ == "__main__":
    main()
