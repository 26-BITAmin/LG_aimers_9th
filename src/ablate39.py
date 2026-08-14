"""다중 과제 MLP — λ 고원 끝까지 + **블렌드 수준** 측정

ablate38이 준 것 (3시드, 쌍체)
    λ         f2023    f2024   크기비
    0.5      +66.3    +39.8    1.67
    1.0      +91.8    +52.4    1.75
    2.0     +117.2    +60.7    1.93   ← 아직 안 꺾였다
    rand0.5  +53.6    +17.1    3.13   ❌ 조건 ③ 실패 (망가진 시즌에서만 큼)
    dup0.5   −24.2    −10.6     —     ❌ 양 fold 음수

    → 모드 몫 = mode − rand = **+12.7 / +22.7**. 좋은 fold에서 더 크다.
      정규화 효과와 모드 지식이 겹쳐 있었고 대조군이 분리했다.

여기서 재는 것 — 앞의 숫자로는 채택 판단을 할 수 없다
    ① **λ 고원의 끝.** 2.0에서 안 꺾였으니 4·8까지 본다. 끝을 모르면
       "통과 구간의 가장자리를 피한다"(§7.37)를 적용할 수 없다.
    ② **블렌드 수준 이득.** 배포는 0.6·LGBM + 0.4·MLP + δ 다.
       MLP 단독 +60이 블렌드에서 얼마가 되는지가 실제 채택 근거다.
       (§7.16에서 MLP 멤버를 늘렸을 때 단독 이득의 대부분이 블렌드에서 사라졌다.)
    ③ 고른 λ에서의 **rand 대조군**을 다시 — 0.5에서 잰 것을 그대로 쓰면 안 된다.

블렌드 계산
    z_blend = (1−α)·z_lgb(캐시) + α·mean(새 MLP 시드들) + Σβ·δ
    LGBM 쪽은 `foldcache`의 것을 그대로 쓴다 (이 실험이 안 건드리는 부분이다).
    MLP 풀은 배포가 8멤버인데 여기서는 3시드 평균이라 **분산 축소가 덜 된 상태**다 —
    즉 여기서 나오는 블렌드 이득은 보수적인(작게 나오는) 추정이다.

⚠️ GPU 전용. 절대 점수를 CPU 기록과 대조하지 말 것 — 같은 실행 안의 A/B만 읽는다.

실행:  MLP_DEVICE=cuda .venv-gpu/Scripts/python.exe -u src/ablate39.py
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
from config import (CTX_AXES, CTX_K, MLP_ALPHA, MLP_VARIANTS, TARGET_COL,
                    TRAIN_CSV)
from data import load_csv
from features import logit_adjust
from mlp_infer import encode_apply
from modelabel import recover
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
SEEDS = [42, 7, 2024]
VARIANT = dict(MLP_VARIANTS["emb4"][0])
CONFIGS = [("None", None, None), ("2.0", 2.0, "mode"), ("4.0", 4.0, "mode"),
           ("8.0", 8.0, "mode"), ("rand2.0", 2.0, "rand"), ("rand4.0", 4.0, "rand")]


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    season = df["season"].to_numpy()
    lab, _, _ = recover(df)
    mid, rev, suc = (np.round(lab[k]) for k in ("middle", "reverse", "success"))
    okl = ~(np.isnan(mid) | np.isnan(rev) | np.isnan(suc))
    cached = foldcache.load(FOLDS, verbose=False)
    print(f"라벨 {okl.mean():.1%}  device={mlp._device()}  α={MLP_ALPHA}"
          f"  ({time.time()-t0:.0f}s)")

    store = {}
    for fold in FOLDS:
        Xtr, y, w, Xva, yva, art = mlp.fold_data(df, fold)
        enc = mlp.fit_encoder(Xtr, art)
        A, Ctr = encode_apply(Xtr, enc)
        B, Cva = encode_apply(Xva, enc)
        m_tr = season < fold

        aux_mode = np.stack([mid[m_tr], rev[m_tr]], 1)
        aux_mode[~okl[m_tr]] = 0.0
        rng = np.random.default_rng(12345)
        auxes = {"mode": aux_mode, "rand": aux_mode[rng.permutation(len(aux_mode))]}

        # ── 블렌드용 고정 성분 (이 실험이 안 건드린다) ──────────────
        z_lgb = cached[fold][0]
        ctx = ctxfit.fit_ctx(df[m_tr], TARGET_COL, CTX_AXES, CTX_K)
        d_adj, _ = logit_adjust(df[season == fold], {"ctx": ctx})
        assert len(z_lgb) == len(yva)

        print(f"\n{'='*88}\nfold {fold}   학습 {len(A):,}  검증 {len(B):,}"
              f"   ({time.time()-t0:.0f}s)\n{'='*88}")
        print(f"  {'설정':<9}{'MLP단독':>10}{'블렌드':>10}   시드별 MLP")

        res = {}
        for name, lam, kind in CONFIGS:
            zs = []
            for seed in SEEDS:
                _, curve = mlp.train_mlp(A, Ctr, y, w, B, Cva, enc["sizes"],
                                         seed=seed, verbose=False, yva=yva,
                                         y_aux=(None if kind is None else auxes[kind]),
                                         **dict(VARIANT, aux=lam))
                zs.append(curve[-1])
            solo = [bss(yva, shift_to_prior(1 / (1 + np.exp(-z)), yva.mean()))
                    for z in zs]
            zm = np.mean(zs, 0)                       # MLP 풀 (3시드 평균)
            zb = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * zm + d_adj
            blend = bss(yva, shift_to_prior(1 / (1 + np.exp(-zb)), yva.mean()))
            res[name] = {"solo": np.array(solo), "blend": blend}
            print(f"  {name:<9}{np.mean(solo):>10.1f}{blend:>10.1f}   "
                  + " ".join(f"{v:>7.1f}" for v in solo)
                  + f"   ({time.time()-t0:.0f}s)")
        store[fold] = res

        base = res["None"]
        print(f"\n  {'설정':<9}{'MLP 쌍체':>11}{'SE':>7}{'t':>6}{'블렌드 이득':>12}")
        for name, lam, kind in CONFIGS:
            if name == "None":
                continue
            d = res[name]["solo"] - base["solo"]
            se = d.std(ddof=1) / np.sqrt(len(d))
            print(f"  {name:<9}{d.mean():>+11.2f}{se:>7.2f}{d.mean()/max(se,1e-9):>6.1f}"
                  f"{res[name]['blend'] - base['blend']:>+12.2f}")

    print("\n" + "=" * 88)
    print("판정 — **블렌드 이득**으로 (이게 배포에서 실제로 얻는 값이다)")
    print("=" * 88)
    from decide import decide
    print(f"  {'설정':<9}{'f2023':>10}{'f2024':>10}{'크기비':>8}   판정")
    for name, lam, kind in CONFIGS:
        if name == "None":
            continue
        a = store[2023][name]["blend"] - store[2023]["None"]["blend"]
        b = store[2024][name]["blend"] - store[2024]["None"]["blend"]
        v, why, _ = decide(name, float(a), float(b))
        r = max(abs(a), abs(b)) / max(min(abs(a), abs(b)), 1e-9)
        print(f"  {name:<9}{a:>+10.2f}{b:>+10.2f}{r:>8.2f}   {v}  {why.splitlines()[0]}")
    print("\n  ⭐ 채택 λ는 **최댓값이 아니라 통과 구간 한가운데**로 고른다 (§7.37).")
    print("     그리고 같은 λ의 rand 대조군보다 확실히 커야 모드 지식이 근거가 된다.")
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
