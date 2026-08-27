"""MLP 정규화 노브 — `dropout` · `weight_decay` (사용자 제안).

§7.173(`regloose2.py`)은 블렌드의 **25%**인 트리 팔을 쟀다. `MLP_ALPHA = 0.75` 이므로
나머지 75%는 MLP고, `dropout 0.3` · `wd 1e-2` 는 그쪽 노브다. 같은 로짓 변화량이면
MLP 쪽이 블렌드를 **3배** 크게 움직인다 — 잠재 레버리지는 이쪽이 크다.

§7.172 가 남긴 두 교훈을 설계에 박는다.
  ① **배포 등가 경로에서 잰다** — `foldcache.fold_data` + `soften` + `aux_labels`.
  ② **양성 대조군을 넣는다** — `MLP_AUX_LAMBDA = 0`. 보조 헤드는 §7.41 에서 LB +2.66 로
     채택된 부품이다. 이 팔이 뚜렷한 음수여야 나머지 널이 의미를 갖는다.

**사전 등록 — 검출력.** §7.172 에서 ρ≈0.9 를 가정했다가 실측 0.18 로 4배 낙관했다.
`dropout`·`wd` 는 MLP 가중을 통째로 바꾸는 **구조 변경** 처리이므로 이번에도 ρ 는 낮게
(0.2 안팎) 잡는다. 복제 단위가 단일 emb4 멤버라 5멤버 평균보다 산포가 크다 —
기준 팔의 SD 를 먼저 찍고 최소검출을 계산한 뒤 팔들을 읽는다.

z_lgb 는 팔 사이에서 고정이므로 한 번만 학습해 캐시한다 (쌍체 차분에서 상쇄).

실행:  .venv/Scripts/python.exe -u src/mlpreg.py 2024
"""
import gc
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
import mlp
import modelabel
import soften
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA, MLP_AUX_LAMBDA,
                    N_ROUNDS, TRAIN_CSV)
from data import load_csv

mlp.THREADS = 8
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEEDS = (42, 7, 2024, 1234, 31337, 777, 31415)
EMB4 = {"pitcher_id": 4, "batter_id": 4, "pitcher_team_id": 4, "batter_team_id": 4,
        "base_state": 4, "hand_matchup": 4, "count_state": 4}
BASE = dict(epochs=9, hid=128, lr=2e-3, dropout=0.3, wd=1e-2, use_id_emb=True, emb_dim=EMB4)

ARMS = {
    "기준 do.3 wd.01":   dict(cfg={},                  aux=MLP_AUX_LAMBDA),
    "dropout 0.15":      dict(cfg=dict(dropout=0.15),  aux=MLP_AUX_LAMBDA),
    "dropout 0.45 [대조]": dict(cfg=dict(dropout=0.45),  aux=MLP_AUX_LAMBDA),
    "wd 0.003":          dict(cfg=dict(wd=3e-3),       aux=MLP_AUX_LAMBDA),
    "wd 0.03 [대조]":     dict(cfg=dict(wd=3e-2),       aux=MLP_AUX_LAMBDA),
    "aux 0 [양성대조]":    dict(cfg={},                  aux=0.0),
}


def get_lgb(X, y, w, Xva, yva, fold, t0):
    f = CACHE / f"mlpreg_lgb_f{fold}.npz"
    if f.exists():
        z = np.load(f)["z"]
        print(f"  [{time.time()-t0:.0f}s] LGBM 재사용  단독 {mlp.score(z, yva):.1f}")
        return z
    z, n = np.zeros(len(Xva)), 0
    for vover in ENSEMBLE_VARIANTS.values():
        for s in (42, 7, 2024):
            p = dict(FINAL_LGB_PARAMS, **vover, seed=s, bagging_seed=s,
                     feature_fraction_seed=s, data_random_seed=s)
            bst = lgb.train(p, lgb.Dataset(X, y, weight=w), num_boost_round=N_ROUNDS)
            z += bst.predict(Xva, raw_score=True); n += 1
            del bst
    z /= n
    np.savez_compressed(f, z=z)
    print(f"  [{time.time()-t0:.0f}s] LGBM {n}개  단독 {mlp.score(z, yva):.1f}")
    return z


def main(fold):
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    Xtr0, ytr0, w0, Xva, yva, art, tr = foldcache.fold_data(df, fold)
    del df; gc.collect()
    aux0, cov = modelabel.aux_labels(tr)
    X, y, w, aux, h = soften.soften(tr, Xtr0, ytr0, w0, aux0)
    print(f"[{time.time()-t0:.0f}s] fold={fold}  soft {len(Xtr0):,}→{len(X):,}행"
          f"  보조라벨 {cov:.1%}")

    z_lgb = get_lgb(X, y, w, Xva, yva, fold, t0)
    A, C, B, Cv, _, sizes = mlp.encode(X, Xva, art)
    del Xtr0, X; gc.collect()

    res = {}
    for name, spec in ARMS.items():
        f = CACHE / f"mlpreg_arm_f{fold}_{name.split()[0]}_{name.split()[1] if len(name.split())>1 else ''}.npz"
        if f.exists():
            res[name] = np.load(f)["s"]
            print(f"[{time.time()-t0:.0f}s] {name:<18} 재사용 {res[name].mean():8.2f}")
            continue
        per_seed = []
        for s in SEEDS:
            _, curve = mlp.train_mlp(A, C, y, w, B, Cv, sizes, seed=s, verbose=False,
                                     y_aux=aux, **dict(BASE, **spec["cfg"], aux=spec["aux"]))
            zb = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * curve[-1]
            per_seed.append(mlp.score(zb, yva))
            gc.collect()
        res[name] = np.array(per_seed)
        np.savez_compressed(f, s=res[name])
        print(f"[{time.time()-t0:.0f}s] {name:<18} 시드별 {res[name].mean():8.2f}"
              f" ± {res[name].std(ddof=1):.2f}")

    base = res["기준 do.3 wd.01"]
    sd = base.std(ddof=1)
    print(f"\n기준 팔 SD = {sd:.2f}  (단일 emb4 멤버 복제)")
    print(f"{'팔':<18}{'쌍체평균Δ':>11}{'SE':>8}{'t':>8}{'양수':>7}{'ρ':>7}  95% CI")
    print("-" * 76)
    for name, s in res.items():
        if name.startswith("기준"):
            continue
        d = s - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se > 0 else 0.0
        rho = np.corrcoef(base, s)[0, 1]
        print(f"{name:<18}{d.mean():>11.2f}{se:>8.2f}{t:>8.2f}{f'{(d>0).sum()}/7':>7}"
              f"{rho:>7.2f}  [{d.mean()-2.447*se:+.2f}, {d.mean()+2.447*se:+.2f}]")
    print("\n※ 판정 순서: ① aux0(양성대조)이 뚜렷한 음수인가 → 아니면 나머지 널은 무효.")
    print("             ② 완화(do .15 / wd .003) 와 조임(do .45 / wd .03) 의 부호가 대칭인가.")
    np.savez_compressed(CACHE / f"mlpreg_f{fold}.npz", **res)
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2024)
