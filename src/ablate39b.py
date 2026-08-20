"""ablate39의 빠진 조각 — fold2024의 rand 대조군

ablate39를 중단했을 때 fold2023은 전부 나왔고 fold2024는 mode만 나왔다.
```
fold2023 쌍체:  mode2.0 +117.2 → (8.0 +137.1)   rand2.0 +124.2   rand4.0 +159.0
fold2024      :  None 649.2/793.5   2.0 709.8/807.0   4.0 707.7/806.7   8.0 695.5/804.6
```
**fold2023에서는 rand가 mode보다 크다.** 정규화 효과가 λ와 함께 커지고,
2023은 망가진 시즌이라 정규화가 특히 잘 듣는다. 그러니 fold2024에서 갈리는지가
채택의 마지막 관문이다 — 거기서 멈췄다.

여기서는 fold2024만, 같은 실행 안에서 None / mode2.0 / rand2.0 / rand4.0 을 돌린다.
(시드 쌍이 정확히 맞아야 하므로 None도 다시 계산한다 — GPU는 실행 간 비트
동일성이 보장되지 않는다.)

실행:  MLP_DEVICE=cuda .venv-gpu/Scripts/python.exe -u src/ablate39b.py
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
from config import CTX_AXES, CTX_K, MLP_ALPHA, MLP_VARIANTS, TARGET_COL, TRAIN_CSV
from data import load_csv
from features import logit_adjust
from mlp_infer import encode_apply
from modelabel import recover
from validation import bss, shift_to_prior

FOLD = 2024
SEEDS = [42, 7, 2024]
VARIANT = dict(MLP_VARIANTS["emb4"][0])
CONFIGS = [("None", None, None), ("mode2.0", 2.0, "mode"),
           ("rand2.0", 2.0, "rand"), ("rand4.0", 4.0, "rand")]

# ablate39 로그에서 가져온 fold2023 쌍체 결과 (같은 코드·같은 시드)
F2023 = {"mode2.0": 117.17, "rand2.0": 124.24, "rand4.0": 158.99}


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    season = df["season"].to_numpy()
    lab, _, _ = recover(df)
    mid, rev, suc = (np.round(lab[k]) for k in ("middle", "reverse", "success"))
    okl = ~(np.isnan(mid) | np.isnan(rev) | np.isnan(suc))

    Xtr, y, w, Xva, yva, art = mlp.fold_data(df, FOLD)
    enc = mlp.fit_encoder(Xtr, art)
    A, Ctr = encode_apply(Xtr, enc)
    B, Cva = encode_apply(Xva, enc)
    m_tr = season < FOLD

    aux_mode = np.stack([mid[m_tr], rev[m_tr]], 1)
    aux_mode[~okl[m_tr]] = 0.0
    rng = np.random.default_rng(12345)
    auxes = {"mode": aux_mode, "rand": aux_mode[rng.permutation(len(aux_mode))]}

    cached = foldcache.load([FOLD], verbose=False)
    z_lgb = cached[FOLD][0]
    ctx = ctxfit.fit_ctx(df[m_tr], TARGET_COL, CTX_AXES, CTX_K)
    d_adj, _ = logit_adjust(df[season == FOLD], {"ctx": ctx})

    print(f"fold {FOLD}  학습 {len(A):,}  검증 {len(B):,}  device={mlp._device()}"
          f"  ({time.time()-t0:.0f}s)")
    print(f"\n  {'설정':<9}{'MLP단독':>10}{'블렌드':>10}   시드별 MLP")

    res = {}
    for name, lam, kind in CONFIGS:
        zs = []
        for seed in SEEDS:
            _, curve = mlp.train_mlp(A, Ctr, y, w, B, Cva, enc["sizes"],
                                     seed=seed, verbose=False, yva=yva,
                                     y_aux=(None if kind is None else auxes[kind]),
                                     **dict(VARIANT, aux=lam))
            zs.append(curve[-1])
        solo = np.array([bss(yva, shift_to_prior(1 / (1 + np.exp(-z)), yva.mean()))
                         for z in zs])
        zb = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * np.mean(zs, 0) + d_adj
        blend = bss(yva, shift_to_prior(1 / (1 + np.exp(-zb)), yva.mean()))
        res[name] = {"solo": solo, "blend": blend}
        print(f"  {name:<9}{solo.mean():>10.1f}{blend:>10.1f}   "
              + " ".join(f"{v:>7.1f}" for v in solo) + f"   ({time.time()-t0:.0f}s)")

    base = res["None"]
    print(f"\n  {'설정':<9}{'MLP 쌍체':>11}{'SE':>7}{'t':>6}{'블렌드':>9}")
    for name, lam, kind in CONFIGS:
        if name == "None":
            continue
        d = res[name]["solo"] - base["solo"]
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {name:<9}{d.mean():>+11.2f}{se:>7.2f}"
              f"{d.mean()/max(se,1e-9):>6.1f}"
              f"{res[name]['blend'] - base['blend']:>+9.2f}")

    # ── 핵심 판정: 모드 몫이 양 fold에서 살아 있는가 ─────────────────
    print("\n" + "=" * 80)
    print("모드 몫 = mode − rand (같은 λ). 정규화 효과를 뺀 순수 모드 지식")
    print("=" * 80)
    dm = (res["mode2.0"]["solo"] - base["solo"]).mean()
    dr = (res["rand2.0"]["solo"] - base["solo"]).mean()
    print(f"  {'':<10}{'mode2.0':>10}{'rand2.0':>10}{'모드 몫':>10}")
    print(f"  {'fold2023':<10}{F2023['mode2.0']:>+10.2f}{F2023['rand2.0']:>+10.2f}"
          f"{F2023['mode2.0'] - F2023['rand2.0']:>+10.2f}")
    print(f"  {'fold2024':<10}{dm:>+10.2f}{dr:>+10.2f}{dm - dr:>+10.2f}")
    a, b = F2023["mode2.0"] - F2023["rand2.0"], dm - dr
    print(f"\n  → 모드 몫 부호: fold2023 {'+' if a > 0 else '−'} / "
          f"fold2024 {'+' if b > 0 else '−'}")
    if a <= 0 or b <= 0:
        print("     **한쪽이 음수다. 모드 라벨이 아니라 정규화가 본체다.**")
        print("     그렇다면 보조 헤드가 아니라 더 단순한 노이즈 주입으로 얻어야 한다.")
    else:
        print(f"     둘 다 양수. 크기비 {max(abs(a),abs(b))/max(min(abs(a),abs(b)),1e-9):.2f}")
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
