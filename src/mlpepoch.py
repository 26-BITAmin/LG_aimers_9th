"""MLP 에폭 곡선 — 배포 예산(ep9)이 정점을 지났는가 (사용자 제안, §7.196 후속 1단계).

■ 왜
§7.196 의 P5 는 {9, 15} 만 걸었다 — **미수렴만 의심하고 과학습을 안 봤다.** 결과가
−27.9/−30.2 로 강한 음수였는데, 그게 "9 가 정점이라 양쪽이 나쁘다" 인지 "정점이 9 보다
아래인데 오른쪽만 쟀다" 인지 구분할 데이터가 없다. §7.196 의 깊이 결론이 그 구멍 위에 있다.

전례가 이미 있다 — §7.163(NODE) 4차 관문:
    N64 f2023  518.7 591.2 [603.0] 585.5 570.0 566.7 560.4 557.8 557.4
    ep3 정점 후 단조 하락, 최종이 정점보다 f2023 −46 / f2024 −15.
그리고 그 절의 결론: **"용량-반응이 에폭 예산으로 뒤집힌다 — 이 계열에서 용량과 에폭은
같은 축이다"** (ep9 에서 N128 < N64 인데 ep3 에서 N128e3 ≥ N64e3).
깊이는 용량이다. 그러니 §7.196 의 깊이 사다리는 **예산 한 점에서 잰 용량-반응**이다.

§7.163 이 남긴 교훈을 이번엔 지킨다:
> ★ 양방향으로 읽히는 진단을 넣은 것이 값을 냈다. 한쪽 가설(과소학습)만 검정하도록
>   설계했으면 못 봤다.

■ 왜 런 내부 곡선으로 충분한가 (그리고 어디까지만 쓸 수 있나)
`train_mlp` 는 에폭별 valid 로짓(`curve`)을 이미 돌려준다. `OneCycleLR(total_steps=steps)`
라 9에폭 런의 ep3 지점은 **아직 안 식은 고 LR 지점**이고, 제대로 스케줄된 3에폭 런보다
불리하다. 따라서
  · **과학습 탐지에는 보수적이다** — ep_k 가 ep9 를 이기면 진짜 k에폭 런은 그보다 낫다.
  · **미수렴 탐지에는 못 쓴다** — 불리한 지점이라 낮게 나오는 게 당연하다.
여기서 묻는 것은 전자뿐이다. 후자는 §7.196 P5 가 이미 답했다(ep15 가 더 나쁨).

■ 설계
배포 예산 그대로 **epochs=9** 로 돌리고 ep1~9 곡선을 읽는다 (12에폭으로 늘리면 스케줄이
달라져 §7.196 과 비교가 끊긴다).
  · **정합성 게이트(공짜)** — `curve[-1]` 점수가 `mlpdepth` 캐시의 같은 시드 값과
    일치해야 한다. 같은 호출·같은 시드라 비트 단위로 같아야 정상이다.
  · 정점 에폭은 **시드 평균 곡선**에서 고르고, 그 **고정 에폭**에서 시드별 쌍체차를 낸다.
    시드마다 argmax 를 따로 잡으면 낙관 편향이 생긴다.
  · 이건 **스크린**이다. 정점이 9 보다 앞이면 2단계에서 **그 에폭에 제대로 스케줄된 런**을
    5시드로 돌려 채점한다. 이 스크립트 숫자로 채택 판정하지 않는다
    (§7.25 가 지적한 valid argmax 반칙, §7.163 ⚠️ 의 1시드 오독).

■ 팔
  d2p — 배포 설정 그 자체. `MLP_ALPHA=0.75` 라 블렌드의 75% 다. **최우선 표적.**
  d4r — §7.196 생존 후보
  d6p / d6r — §7.196 깊이 벌점이 에폭 불일치 아티팩트인지

실행:  .venv/Scripts/python.exe -u src/mlpepoch.py 2024
       .venv/Scripts/python.exe -u src/mlpepoch.py 2023
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
                    MLP_VARIANTS, N_ROUNDS, TRAIN_CSV)
from data import load_csv

mlp.THREADS = 8
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEEDS = (42, 7)                      # mlpdepth SEEDS 의 앞 2개 -> 정합성 게이트가 걸린다
TCRIT = 12.71                        # t(0.975, df=1) — n=2 라 구간이 매우 넓다. 스크린임을 잊지 말 것.

EMB4 = {"pitcher_id": 4, "batter_id": 4, "pitcher_team_id": 4, "batter_team_id": 4,
        "base_state": 4, "hand_matchup": 4, "count_state": 4}
BASE = dict(epochs=9, hid=128, lr=2e-3, dropout=0.3, wd=1e-2, use_id_emb=True,
            emb_dim=EMB4)
assert BASE == MLP_VARIANTS["emb4"][0], "BASE 가 배포 emb4 설정과 다르다"

ARMS = {
    "d2p": dict(depth=2, resid=False),       # 배포 설정 — 최우선
    "d4r": dict(depth=4, resid=True),        # §7.196 생존 후보
    "d6p": dict(depth=6, resid=False),
    "d6r": dict(depth=6, resid=True),
}
N_EP = BASE["epochs"]


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
            z += bst.predict(Xva, raw_score=True)
            n += 1
            del bst
    z /= n
    np.savez_compressed(f, z=z)
    print(f"  [{time.time()-t0:.0f}s] LGBM {n}개  단독 {mlp.score(z, yva):.1f}")
    return z


def main(fold):
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    Xtr0, ytr0, w0, Xva, yva, art, tr = foldcache.fold_data(df, fold)
    del df
    gc.collect()
    aux0, cov = modelabel.aux_labels(tr)
    X, y, w, aux, h = soften.soften(tr, Xtr0, ytr0, w0, aux0)
    print(f"[{time.time()-t0:.0f}s] fold={fold}  soft {len(Xtr0):,}→{len(X):,}행"
          f"  보조라벨 {cov:.1%}  에폭 곡선 ep1~{N_EP}, 시드 {SEEDS}")

    z_lgb = get_lgb(X, y, w, Xva, yva, fold, t0)
    A, C, B, Cv, _, sizes = mlp.encode(X, Xva, art)
    del Xtr0, X
    gc.collect()

    CU = {}
    for name, cfg in ARMS.items():
        f = CACHE / f"mlpepoch_f{fold}_{name}.npz"
        if f.exists():
            CU[name] = np.load(f)["c"]
            print(f"[{time.time()-t0:.0f}s] {name:<5} 재사용")
            continue
        rows = []
        for s in SEEDS:
            _, curve = mlp.train_mlp(A, C, y, w, B, Cv, sizes, seed=s, verbose=False,
                                     y_aux=aux, **dict(BASE, **cfg, aux=MLP_AUX_LAMBDA))
            rows.append([mlp.score((1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * zk, yva)
                         for zk in curve])
            del curve
            gc.collect()
        CU[name] = np.array(rows)
        np.savez_compressed(f, c=CU[name])
        print(f"[{time.time()-t0:.0f}s] {name:<5} " +
              " ".join(f"{v:6.1f}" for v in CU[name].mean(0)))

    # ── 정합성 게이트 — 마지막 에폭이 mlpdepth 캐시와 같아야 한다 ──────────────
    print(f"\n[정합성 게이트] curve[-1] vs mlpdepth 캐시 (같은 호출·같은 시드)")
    okall = True
    for name in ARMS:
        g = CACHE / f"mlpdepth_f{fold}_{name}.npz"
        if not g.exists():
            print(f"  {name:<5} mlpdepth 캐시 없음 — 건너뜀")
            continue
        ref = np.load(g)["s"][:len(SEEDS)]
        d = np.abs(CU[name][:, -1] - ref).max()
        okall &= d < 1e-6
        print(f"  {name:<5} max|diff| = {d:.3e}  {'OK' if d < 1e-6 else '❌ 불일치'}")
    print(f"  → {'통과 — 하네스 동일' if okall else '❌ 실패. 아래 곡선을 믿지 말 것.'}")

    # ── 곡선 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*92}\n[fold {fold}] 에폭 곡선 (시드 {len(SEEDS)}개 평균, 배포 스케줄 ep{N_EP})"
          f"\n{'='*92}")
    print(f"{'팔':<6}" + "".join(f"{'ep'+str(k+1):>8}" for k in range(N_EP))
          + f"{'정점':>7}{'정점−최종':>11}")
    print("-" * 92)
    peak = {}
    for name in ARMS:
        m = CU[name].mean(0)
        k = int(np.argmax(m))                       # 시드 평균 곡선에서 정점 (시드별 argmax 아님)
        peak[name] = k
        mark = "".join(f"{v:>8.1f}" if i != k else f"{('['+f'{v:.1f}'+']'):>8}"
                       for i, v in enumerate(m))
        print(f"{name:<6}{mark}{'ep'+str(k+1):>7}{m[k]-m[-1]:>11.2f}")

    # ── 정점 − 최종, 시드별 쌍체 ────────────────────────────────────────────
    print(f"\n[정점 − 최종]  정점 에폭은 위 평균 곡선에서 고정 (시드별 argmax 금지)")
    print(f"{'팔':<6}{'정점':>6}{'Δ':>9}{'시드별':>22}{'부호':>7}")
    print("-" * 92)
    for name in ARMS:
        k = peak[name]
        d = CU[name][:, k] - CU[name][:, -1]
        print(f"{name:<6}{'ep'+str(k+1):>6}{d.mean():>9.2f}"
              + f"{'  '.join(f'{v:+.1f}' for v in d):>22}"
              + f"{f'{int((d > 0).sum())}/{len(d)}':>7}")

    np.savez_compressed(CACHE / f"mlpepoch_f{fold}.npz", **CU)
    print("\n※ 이건 스크린이다. 정점이 ep9 보다 앞인 팔만 2단계(그 에폭에 제대로 스케줄된")
    print("   5시드 런)로 채점한다. 런 내부 곡선은 과학습 탐지에만 유효하다.")
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2024)
