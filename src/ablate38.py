"""다중 과제 MLP — **대조군**과 λ 고원 (`ablate37`이 양성을 냈다)

ablate37 결과 (2시드, GPU)
    λ        fold2023          fold2024
    None      −61.2             648.8
    0.2       −12.6 (+48.6)     666.1 (+17.3)
    0.5       +18.0 (+79.1)     683.9 (+35.1)
    → 4/4 전부 단조 증가. 쌍체 산포 6.9 / 3.7 로 매우 일관.

왜 이대로 채택하면 안 되나
    λ=0.5면 보조 손실이 주 손실의 **절반**이다. 그 정도면 모드 지식이 아니라
    **그냥 정규화**로 작동할 수 있다 — emb4는 9에폭·dropout 0.3짜리 작은 모델이다.
    플래툰(§7.36)을 살린 것이 정확히 이 대조군이었다. 여기서도 같은 것이 필요하다.

    dup    보조 헤드가 `success`를 한 번 더 맞힌다 — **새 정보 0**, 정규화 효과만
    rand   모드 라벨을 **행 단위로 섞는다** — 주변분포 유지, 대응만 파괴

    둘 중 하나라도 같은 이득을 주면 모드 라벨은 무관하고, 발견한 것은
    "MLP에 정규화가 더 필요하다"이다 (그것도 쓸모는 있지만 다른 얘기다).

같이 재는 것
    λ 고원 — 0.2 / 0.5 / 1.0 / 2.0. 끝에서 꺾이지 않으면 더 볼 값이 있다.
    §7.37에서 배운 대로 **통과 구간의 가장자리를 밟지 않는 값**을 고른다.

⚠️ GPU(`.venv-gpu`) 전용이다. 절대 점수를 CPU 기록과 대조하지 말 것 —
   **같은 실행 안의 A/B 차이만** 읽는다.

실행:  MLP_DEVICE=cuda .venv-gpu/Scripts/python.exe -u src/ablate38.py
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

import mlp
from config import MLP_VARIANTS, TRAIN_CSV
from data import load_csv
from mlp_infer import encode_apply
from modelabel import recover
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
SEEDS = [42, 7, 2024]
VARIANT = dict(MLP_VARIANTS["emb4"][0])
# (이름, λ, 보조 타깃 종류)
CONFIGS = [("None", None, None), ("0.2", 0.2, "mode"), ("0.5", 0.5, "mode"),
           ("1.0", 1.0, "mode"), ("2.0", 2.0, "mode"),
           ("dup0.5", 0.5, "dup"), ("rand0.5", 0.5, "rand")]


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    lab, _, _ = recover(df)
    mid, rev, suc = (np.round(lab[k]) for k in ("middle", "reverse", "success"))
    okl = ~(np.isnan(mid) | np.isnan(rev) | np.isnan(suc))
    print(f"라벨 복원 {okl.mean():.1%}   device={mlp._device()}   ({time.time()-t0:.0f}s)")

    store = {}
    for fold in FOLDS:
        Xtr, y, w, Xva, yva, art = mlp.fold_data(df, fold)
        enc = mlp.fit_encoder(Xtr, art)
        A, Ctr = encode_apply(Xtr, enc)
        B, Cva = encode_apply(Xva, enc)
        m_tr = (df.season < fold).to_numpy()

        aux_mode = np.stack([mid[m_tr], rev[m_tr]], 1)
        aux_mode[~okl[m_tr]] = 0.0
        # dup — 새 정보 0. 같은 타깃을 두 번 맞힌다
        aux_dup = np.stack([y, y], 1)
        # rand — 주변분포는 그대로, 행 대응만 파괴 (시드 고정)
        rng = np.random.default_rng(12345)
        aux_rand = aux_mode[rng.permutation(len(aux_mode))]

        auxes = {"mode": aux_mode, "dup": aux_dup, "rand": aux_rand}
        print(f"\n{'='*84}\nfold {fold}   학습 {len(A):,}  검증 {len(B):,}"
              f"   ({time.time()-t0:.0f}s)\n{'='*84}")

        res = {}
        for name, lam, kind in CONFIGS:
            for seed in SEEDS:
                cfg = dict(VARIANT, aux=lam)
                ya = None if kind is None else auxes[kind]
                _, curve = mlp.train_mlp(A, Ctr, y, w, B, Cva, enc["sizes"],
                                         seed=seed, verbose=False, yva=yva,
                                         y_aux=ya, **cfg)
                p = 1.0 / (1.0 + np.exp(-curve[-1]))
                res.setdefault(name, {})[seed] = bss(yva, shift_to_prior(p, yva.mean()))
            v = np.array([res[name][s] for s in SEEDS])
            print(f"  {name:<8} " + "  ".join(f"{res[name][s]:>8.1f}" for s in SEEDS)
                  + f"   평균 {v.mean():>8.1f}   ({time.time()-t0:.0f}s)")
        store[fold] = res

        # ── 쌍체 차이 (시드가 짝지어지므로 레벨 노이즈가 상쇄된다) ──────
        print(f"\n  {'설정':<8}{'쌍체 평균':>10}{'쌍체 SE':>10}{'t':>7}   시드별 차이")
        for name, lam, kind in CONFIGS:
            if name == "None":
                continue
            d = np.array([res[name][s] - res["None"][s] for s in SEEDS])
            se = d.std(ddof=1) / np.sqrt(len(d))
            print(f"  {name:<8}{d.mean():>+10.2f}{se:>10.2f}{d.mean()/max(se,1e-9):>7.1f}"
                  + "   " + " ".join(f"{x:>+7.1f}" for x in d))

    # ── 판정 ────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("판정 — 양 fold 쌍체 차이 (src/decide.py 조건 ①②③)")
    print("=" * 84)
    from decide import decide
    print(f"  {'설정':<8}{'f2023':>10}{'f2024':>10}{'크기비':>8}   판정")
    for name, lam, kind in CONFIGS:
        if name == "None":
            continue
        a = np.mean([store[2023][name][s] - store[2023]["None"][s] for s in SEEDS])
        b = np.mean([store[2024][name][s] - store[2024]["None"][s] for s in SEEDS])
        v, why, _ = decide(name, float(a), float(b))
        r = max(abs(a), abs(b)) / max(min(abs(a), abs(b)), 1e-9)
        print(f"  {name:<8}{a:>+10.2f}{b:>+10.2f}{r:>8.2f}   {v}  {why.splitlines()[0]}")

    print("\n  ⭐ 읽는 법: **dup / rand 가 mode 와 비슷하면 모드 라벨은 무관**하고,")
    print("     발견한 것은 '정규화가 더 필요하다'이다. mode만 크면 모드 지식이 진짜다.")
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
