"""다중 과제 보조 헤드 — **배포 구성 그대로** 재측정 (채택 판정용)

ablate39/39b가 준 것 (emb4 3시드 풀, 블렌드 기준)
    mode 2.0   +34.05 / +13.45   크기비 2.53   ①②③ 통과
    rand 2.0   +32.87 / **−6.60**  ← 부호 반전. 정규화 성분은 채택 대상이 아니다
    → fold2023에서 mode·rand가 안 갈린 것은 그 시즌 MLP 단독 기준선이 −72.3
      (상수보다 나쁨)이라 아무 정규화나 크게 듣기 때문이다 (§7.21).
      신뢰할 수 있는 fold2024에서는 +13.45 대 −6.60으로 깨끗이 갈린다.

여기서 고치는 것 — 앞의 숫자는 배포 구성이 아니다
    배포 MLP 풀 = **emb4 5시드 + noid 3시드 = 8멤버**, 가중 {emb4 .625, noid .375}
    앞 실험은 emb4 3시드뿐이었다. 분산 축소가 덜 됐고 noid가 아예 빠져 있다.

    그리고 결정해야 할 것: **noid에도 보조 헤드를 다는가.**
      · emb4만  — noid는 그대로 두어 풀의 다양성을 유지한다
      · 둘 다   — 이득이 크면 더 크겠지만 두 멤버가 닮아져 앙상블 이득이 준다
    §7.24가 배운 것: `noid`는 핸디캡 멤버가 아니라 **더 센** 멤버다. 함부로 안 건드린다.

⚠️ GPU 전용. 절대 점수를 CPU 기록과 대조하지 말 것 — 같은 실행 안의 A/B만 읽는다.

실행:  MLP_DEVICE=cuda .venv-gpu/Scripts/python.exe -u src/ablate40.py
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

import ctxfit
import foldcache
import mlp
from config import (CTX_AXES, CTX_K, MLP_ALPHA, MLP_SEEDS, MLP_VARIANT_WEIGHT,
                    MLP_VARIANTS, TARGET_COL, TRAIN_CSV)
from data import load_csv
from features import logit_adjust
from mlp_infer import encode_apply
from modelabel import recover
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
# λ는 인자로 받는다. 배포 구성 λ=2.0에서 크기비 3.09로 게이트를 아슬아슬하게 넘겼고,
# 정규화 성분이 λ에 비례하므로(rand fold2023: λ2.0 +32.9 → λ4.0 +50.2)
# **λ를 낮추면 fold2023의 부풀림이 fold2024의 실질보다 빨리 줄어야 한다.**
# 그 예측을 검증하는 것이 λ 인자의 목적이다.
LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
# 이름 → 어느 변형에 보조 헤드를 다는가
CONFIGS = [("None", set()), ("emb4만", {"emb4"}), ("둘 다", {"emb4", "noid"})]


def pool_z(A, Ctr, y, w, B, Cva, sizes, aux, with_aux):
    """배포와 같은 8멤버 풀의 가중 평균 로짓."""
    tot, acc = 0.0, np.zeros(len(B))
    for vname, (vcfg, nseed) in MLP_VARIANTS.items():
        wt = MLP_VARIANT_WEIGHT[vname] / nseed        # 멤버당 가중
        cfg = dict(vcfg, aux=(LAM if vname in with_aux else None))
        for seed in MLP_SEEDS[:nseed]:
            _, curve = mlp.train_mlp(A, Ctr, y, w, B, Cva, sizes, seed=seed,
                                     verbose=False, yva=None,
                                     y_aux=(aux if vname in with_aux else None),
                                     **cfg)
            acc += wt * curve[-1]
            tot += wt
    return acc / tot


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    season = df["season"].to_numpy()
    lab, _, _ = recover(df)
    mid, rev, suc = (np.round(lab[k]) for k in ("middle", "reverse", "success"))
    okl = ~(np.isnan(mid) | np.isnan(rev) | np.isnan(suc))
    cached = foldcache.load(FOLDS, verbose=False)
    print(f"라벨 {okl.mean():.1%}  device={mlp._device()}  λ={LAM}"
          f"  풀={MLP_VARIANT_WEIGHT}  ({time.time()-t0:.0f}s)")

    store = {}
    for fold in FOLDS:
        Xtr, y, w, Xva, yva, art = mlp.fold_data(df, fold)
        enc = mlp.fit_encoder(Xtr, art)
        A, Ctr = encode_apply(Xtr, enc)
        B, Cva = encode_apply(Xva, enc)
        m_tr = season < fold
        aux = np.stack([mid[m_tr], rev[m_tr]], 1)
        aux[~okl[m_tr]] = 0.0

        z_lgb = cached[fold][0]
        ctx = ctxfit.fit_ctx(df[m_tr], TARGET_COL, CTX_AXES, CTX_K)
        d_adj, _ = logit_adjust(df[season == fold], {"ctx": ctx})

        print(f"\n{'='*78}\nfold {fold}   검증 {len(B):,}   ({time.time()-t0:.0f}s)"
              f"\n{'='*78}")
        print(f"  {'설정':<8}{'MLP풀':>9}{'블렌드':>9}{'vs None':>10}")
        res = {}
        for name, with_aux in CONFIGS:
            zm = pool_z(A, Ctr, y, w, B, Cva, enc["sizes"], aux, with_aux)
            solo = bss(yva, shift_to_prior(1 / (1 + np.exp(-zm)), yva.mean()))
            zb = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * zm + d_adj
            blend = bss(yva, shift_to_prior(1 / (1 + np.exp(-zb)), yva.mean()))
            res[name] = (solo, blend)
            d = blend - res["None"][1] if "None" in res else 0.0
            print(f"  {name:<8}{solo:>9.1f}{blend:>9.1f}{d:>+10.2f}"
                  f"   ({time.time()-t0:.0f}s)")
        store[fold] = res

    print("\n" + "=" * 78)
    print("판정 — 블렌드 이득 (배포 구성: 8멤버 풀 + LGBM + δ)")
    print("=" * 78)
    from decide import decide
    print(f"  {'설정':<8}{'f2023':>10}{'f2024':>10}{'크기비':>8}   판정")
    for name, _ in CONFIGS:
        if name == "None":
            continue
        a = store[2023][name][1] - store[2023]["None"][1]
        b = store[2024][name][1] - store[2024]["None"][1]
        v, why, _ = decide(name, float(a), float(b))
        r = max(abs(a), abs(b)) / max(min(abs(a), abs(b)), 1e-9)
        print(f"  {name:<8}{a:>+10.2f}{b:>+10.2f}{r:>8.2f}   {v}  {why.splitlines()[0]}")
    print("\n  ⚠️ 블렌드는 풀 평균 하나뿐이라 **쌍체 SE가 없다.** 조건 ①은 MLP 단독의"
          "\n     쌍체 t(ablate38에서 12.9~14.0)로 대신 읽는다.")
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
