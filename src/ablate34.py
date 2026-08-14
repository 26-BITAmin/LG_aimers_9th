"""파이프라인 v2 — EDA가 지목한 두 가지 변경 (`src/eda*.py`, `notebooks/eda_report.html`)

무엇을 왜 바꾸나

  ① 꼬리 winsorize (`clip2` / `clip5`)
     `02_response_centered.png`에서 눈으로 보이는 것: 6시즌 반응 곡선이 중간 빈에서는
     거의 완벽히 겹치는데 **양 끝 빈에서만 확 벌어진다.** 측정으로 확인했다
     (`eda_tail.py` Q1): 시즌간 산포가 중간 0.007 vs 꼬리 0.023 — **3.2배.**

     그리고 배포 모델이 실제로 거기서 잃는다 (Q2, 레벨손실):
         fold2023  batter 상위5%  실제 .4767 → 예측 .6380   10,412
         fold2024  batter 상위5%  실제 .4710 → 예측 .5167      837
     **양 fold 같은 부호다** (조건 ②). 전체 BSS 기여로 환산하면 fold2023 −179,
     fold2024 −8.0.

     → 학습 분위로 양끝을 눌러 앉힌다. 행을 버리는 게 아니라 **레버리지를 자른다.**

  ② 타자 이력 블록 제거 (`nobatstat`)
     `batter_id`의 천장이 **−25.7**이다 (§7.34 G) — 검증 시즌 자기 데이터로 오라클을
     써도 상수 모델보다 나쁘다. 타자 개인차에는 in-season으로도 신호가 없다.
     단변량도 일관된다: `asof_batter_success_rate` −132.2 / `asof_batter_n` −253.2.
     그리고 꼬리 손실 1위가 이 블록이다.

     제거 대상은 **이력 통계만**이다. `batter_hand`·`hand_matchup`은 남긴다 —
     손잡이 축은 천장 111.3 / 전이 91.1로 살아 있는 몇 안 되는 축이다.

⚠️ 이 저장소는 피처 제거로 한 번 크게 데였다 (s06, 51개 피처, 홀드아웃 +13.3 → LB −2.13).
   그때의 사후 진단 셋 중 하나가 "뺀 것에 cold-start 신호(`miss_*`, `asof_*_n`)가
   섞여 있었다"였다. 여기서는 `miss_batter_asof`·`asof_batter_n`도 같이 빠지므로
   **같은 함정을 밟을 수 있다.** 그래서 조건 ①②③을 그대로 적용한다.

측정 구조
    배포는 트리 20부스터 + MLP 8멤버다. 여기서는 **모든 변형에 똑같이** 축소 구조를
    쓴다 — 트리 3시드(base) + MLP 3시드(emb4), α=0.40. 변형 간 차이만 보면 되고
    대조군도 같은 구조로 새로 학습하므로 비교는 유효하다. (절대 점수는 배포와 다르다.)

    1단계  트리만 4변형 × 3시드 × 2fold (~12분) → 스크리닝
    2단계  살아남은 변형만 MLP까지 (~변형당 16분)
    ⚠️ 1단계는 **스크리닝이지 채택 판정이 아니다** (§7.14: 단일 모델 측정은
       앙상블 시스템에 대한 예측이 아니다).

사전 예측 (실행 전에 박아둔다)
    P1  `clip2` 트리 단독이 양 fold에서 ≥ 0        내 예측: **fold2023만 크게 +,
        fold2024는 ±2 이내**. 꼬리 손실이 fold2023에 179, fold2024에 8뿐이라
        fold2024에서는 잴 수 있는 크기가 아니다.
    P2  `clip5` < `clip2` (너무 자르면 신호도 자른다)   내 예측: 그렇다
    P3  `nobatstat`이 조건 ①②③ 통과              내 예측: **아니다** —
        s06과 같은 프로필(cold-start 신호 동반 제거)이다.
    P4  통과 후보 수                               내 예측: **0~1개**

실행:  .venv/Scripts/python.exe -u src/ablate34.py 2>&1 | tee experiments/ablate_run34.log
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

import clipfit
import foldcache
import mlp
from config import (DROP_FEATURES, FINAL_LGB_PARAMS, MLP_ALPHA, MLP_SEEDS, MLP_VARIANTS,
                    N_ROUNDS, RECENCY_HALF_LIFE, SEED_ENSEMBLE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from foldcache import CACHE, _key
from mlp_infer import encode_apply
from validation import bss, shift_to_prior

THREADS = 5
mlp.THREADS = THREADS
FOLDS = [2023, 2024]
TSEEDS = SEED_ENSEMBLE[:3]
MSEEDS = MLP_SEEDS[:3]
EMB4 = MLP_VARIANTS["emb4"][0]

# 타자 **이력 통계**만 뺀다. 손잡이·팀·id는 남긴다 (손잡이 축은 살아 있다).
BATTER_STATS = ["asof_batter_success_rate", "asof_batter_success_rate_sm",
                "asof_batter_middle_rate", "asof_batter_middle_rate_sm",
                "asof_batter_n", "miss_batter_asof",
                "matchup_success_edge", "matchup_middle_edge"]

VARIANTS = {
    "ctl":       dict(q=None, drop=[]),
    "clip2":     dict(q=0.02, drop=[]),
    "clip5":     dict(q=0.05, drop=[]),
    "nobatstat": dict(q=None, drop=BATTER_STATS),
}


def ck(fold, kind, name, seed):
    return CACHE / f"v2_{kind}_{name}_f{fold}_{_key()}_{seed}.npz"


def paired(pA, pB, y):
    c = 100000.0 / (y.mean() * (1 - y.mean()))
    d = (pB - y) ** 2 - (pA - y) ** 2
    return c * d.mean(), c * d.std(ddof=1) / np.sqrt(len(d))


def prob(z, y):
    return shift_to_prior(1.0 / (1.0 + np.exp(-z)), y.mean())


def fold_data(df, fold, cfg):
    season = df["season"].to_numpy()
    tr, va = df[season < fold], df[season == fold]
    w = recency_weight(tr["season"].to_numpy(), RECENCY_HALF_LIFE, ref_season=fold - 1)
    # ⚠️ 경계도 **학습 가중을 반영해** 뽑는다. hl=0.5면 사실상 직전 시즌 분포이고,
    #    그게 추론 시점 분포에 가장 가깝다. 검증셋은 절대 안 본다.
    clip = clipfit.fit_clip(tr, cfg["q"], weights=w) if cfg["q"] else None
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, clip=clip)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        drop = [c for c in DROP_FEATURES + cfg["drop"] if c in x.columns]
        return x.drop(columns=drop)

    Xtr, Xva = X(tr), X(va)
    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    del tr, va
    gc.collect()
    return Xtr, ytr, w, Xva, yva, art, clip


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    df = load_csv(TRAIN_CSV)
    print(f"train {df.shape}  ({time.time()-t0:.0f}s)")

    Z, Y = {}, {}
    ncols = {}
    for fold in FOLDS:
        for name, cfg in VARIANTS.items():
            need = [s for s in TSEEDS if not ck(fold, "lgb", name, s).exists()]
            if need:
                Xtr, ytr, w, Xva, yva, art, clip = fold_data(df, fold, cfg)
                ncols[(fold, name)] = Xtr.shape[1]
                if clip and name == "clip2" and fold == 2024:
                    k = "asof_batter_success_rate"
                    print(f"    [{name}] {k} 경계 {clip[k][0]:.4f} ~ {clip[k][1]:.4f}")
            for s in TSEEDS:
                p = ck(fold, "lgb", name, s)
                if p.exists():
                    Z[(fold, name, "lgb", s)] = np.load(p)["z"]
                    continue
                par = dict(FINAL_LGB_PARAMS, seed=s, bagging_seed=s,
                           feature_fraction_seed=s, data_random_seed=s,
                           num_threads=THREADS)
                bst = lgb.train(par, lgb.Dataset(Xtr, ytr, weight=w),
                                num_boost_round=N_ROUNDS)
                z = bst.predict(Xva, raw_score=True)
                del bst
                np.savez_compressed(p, z=z)
                Z[(fold, name, "lgb", s)] = z
            if need:
                Y[fold] = yva
                print(f"  fold{fold} {name:<10} 피처 {Xtr.shape[1]:>3}개  트리 3시드 "
                      f"평균 BSS {mlp.score(np.mean([Z[(fold,name,'lgb',s)] for s in TSEEDS],0), yva):>8.1f}"
                      f"  ({time.time()-t0:.0f}s)")
                del Xtr, Xva
                gc.collect()
        if fold not in Y:
            Y[fold] = df.loc[df["season"].to_numpy() == fold,
                             TARGET_COL].to_numpy(dtype=np.float64)

    def tz(fold, name):
        return np.mean([Z[(fold, name, "lgb", s)] for s in TSEEDS], axis=0)

    # ── 1단계: 트리 스크리닝 ────────────────────────────────────
    print("\n" + "=" * 92)
    print("1단계 — 트리 단독 (3시드 평균, 대조군 `ctl` 대비, 쌍체 SE)")
    print("=" * 92)
    print("⚠️ 스크리닝이다. 채택 판정은 2단계(블렌드)에서 한다.")
    print(f"\n{'변형':>11} " + " ".join(f"{'fold'+str(f):>9}" for f in FOLDS)
          + " " + " ".join(f"{'Δ'+str(f)+'±SE':>21}" for f in FOLDS))
    screen = {}
    for name in VARIANTS:
        row, dl = [], []
        for f in FOLDS:
            y = Y[f]
            row.append(mlp.score(tz(f, name), y))
            if name == "ctl":
                dl.append("          —          ")
            else:
                m_, se = paired(prob(tz(f, name), y), prob(tz(f, "ctl"), y), y)
                dl.append(f"{m_:>+10.2f} ± {se:<5.2f} (t{m_/se:>+5.1f})")
                screen.setdefault(name, []).append((m_, se))
        print(f"{name:>11} " + " ".join(f"{v:>9.1f}" for v in row)
              + " " + " ".join(f"{v:>21}" for v in dl))

    alive = [n for n, v in screen.items()
             if np.sign(v[0][0]) == np.sign(v[1][0]) and max(abs(x[0]) for x in v) > 2]
    print(f"\n  부호 일치 + 최소 한 fold에서 |Δ|>2 → 2단계 진출: {alive or '없음'}")

    # ── 2단계: MLP 블렌드 ───────────────────────────────────────
    stage2 = ["ctl"] + alive
    if len(stage2) > 1:
        print("\n" + "=" * 92)
        print(f"2단계 — 배포 구조 축소판 (트리 3시드 + MLP emb4 3시드, α={MLP_ALPHA})")
        print("=" * 92)
        for fold in FOLDS:
            for name in stage2:
                need = [s for s in MSEEDS if not ck(fold, "mlp", name, s).exists()]
                if need:
                    Xtr, ytr, w, Xva, yva, art, _ = fold_data(df, fold, VARIANTS[name])
                    enc = mlp.fit_encoder(Xtr, art)
                    A, C = encode_apply(Xtr, enc)
                    B, Cv = encode_apply(Xva, enc)
                    del Xtr, Xva
                    gc.collect()
                for s in MSEEDS:
                    p = ck(fold, "mlp", name, s)
                    if p.exists():
                        Z[(fold, name, "mlp", s)] = np.load(p)["z"]
                        continue
                    m, _ = mlp.train_mlp(A, C, ytr, w, A[:512], C[:512],
                                         enc["sizes"], seed=s, verbose=False, **EMB4)
                    z = mlp.infer(m, B, Cv)
                    del m
                    np.savez_compressed(p, z=z)
                    Z[(fold, name, "mlp", s)] = z
                    print(f"  fold{fold} {name} mlp_{s} 완료 ({time.time()-t0:.0f}s)")
                if need:
                    del A, C, B, Cv
                    gc.collect()

        def blend(fold, name):
            zm = np.mean([Z[(fold, name, "mlp", s)] for s in MSEEDS], axis=0)
            return (1 - MLP_ALPHA) * tz(fold, name) + MLP_ALPHA * zm

        print(f"\n{'변형':>11} " + " ".join(f"{'fold'+str(f):>9}" for f in FOLDS)
              + " " + " ".join(f"{'Δ'+str(f)+'±SE':>21}" for f in FOLDS))
        res = {}
        for name in stage2:
            row, dl = [], []
            for f in FOLDS:
                y = Y[f]
                row.append(mlp.score(blend(f, name), y))
                if name == "ctl":
                    dl.append("          —          ")
                else:
                    m_, se = paired(prob(blend(f, name), y), prob(blend(f, "ctl"), y), y)
                    dl.append(f"{m_:>+10.2f} ± {se:<5.2f} (t{m_/se:>+5.1f})")
                    res.setdefault(name, []).append((m_, se))
            print(f"{name:>11} " + " ".join(f"{v:>9.1f}" for v in row)
                  + " " + " ".join(f"{v:>21}" for v in dl))

        print("\n  채택 조건 — ① |Δ| > 2·SE  ② 양 fold 부호 일치  ③ fold간 크기비 ≤ 3배")
        for name, v in res.items():
            (m23, s23), (m24, s24) = v
            c1 = abs(m23) > 2 * s23 and abs(m24) > 2 * s24
            c2 = np.sign(m23) == np.sign(m24)
            ratio = max(abs(m23), abs(m24)) / max(min(abs(m23), abs(m24)), 1e-9)
            c3 = ratio <= 3
            print(f"    {name:<11} ① {'O' if c1 else 'X'}  ② {'O' if c2 else 'X'}"
                  f"  ③ {'O' if c3 else 'X'} (크기비 {ratio:.1f})"
                  f"   → {'채택 검토' if (c1 and c2 and c3) else '기각'}")

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
