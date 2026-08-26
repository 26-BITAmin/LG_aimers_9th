"""§7.162 — **TabM** (Gorishniy et al., ICLR 2025, arXiv:2410.24210) 1단계 측정.

이 저장소에서 유일하게 남아 있던 미측정 문헌 항목이다 (§7.32 말미, §7.26 각주).
기각된 적은 없고 **우선순위가 낮다고 판단해 미뤄둔 것**인데, 그 판단 근거 세 개가
이후 실험으로 전부 약해졌다:

    "이미 MLP 8멤버 명시 앙상블이라 같은 축이고 분산축소 수단끼리 겹친다"
        → §7.140 이 **앙상블은 분산만이 아니라 편향도 줄인다** 를 실측했다
          (CatBoost 핸디캡 −39.8 → −9.1, 12멤버). 가중치 공유는 1/k 분산축소가
          아니라 **다른 함수족**이다.
    "배포 경로가 는다"
        → 추론 29.7s / 600s. TabM-mini 의 numpy 순전파는 독립 8세트보다 **더 싸다**.
    "MLP 쪽 노브는 전부 fold2024 노이즈 대역"
        → 맞다. 그런데 그 MLP 가 로짓 가중 0.675 를 갖고 시드 분산의 99.5% 를 낸다
          (§7.161b). 노브가 죽은 것과 팔이 죽은 것은 다르다.

### ★ 왜 §7.25·§7.26 의 용량 법칙에 안 걸리나

법칙: "입력이든 은닉이든 파라미터를 늘리면 진다" (신호 총량 ≈ 0.004 nat, §7.13).
파라미터를 센다 (hid=128 · depth=2 · d≈73):

    독립 emb4 1멤버         ≈  34,000
    배포 emb4 5멤버         ≈ 170,000
    TabM-mini k=32          ≈  32,600      ← 멤버 6배, 파라미터 1/5

**이 저장소 첫 "앙상블 폭↑ · 파라미터↓" 후보다.** 같은 법칙이 PLE(§7.26)는 죽이고
이것은 살린다 — PLE 는 입력차원을 73→381 로 **늘리는** 변경이었다.

### 팔 (emb4 슬롯만 갈아끼운다. noid 3멤버는 전 팔 공통·고정)

    z_mlp = 0.875·mean(emb4 슬롯) + 0.125·mean(noid 3멤버)      §7.120 LB 확정 가중

    CTRL   emb4 독립 5멤버                     배포 구성 (aux 만 OFF)
    T5     TabM-mini k=5,  3 draw             ← **k 를 맞춘 대조.** 여기서 갈리면
                                                 공유 효과, 안 갈리면 그냥 k 효과
    T32    TabM-mini k=32, 3 draw             ← 원논문 기본값. 진짜 후보

⚠️ **aux 헤드는 전 팔 OFF** 다 (TabM 쪽 멤버별 aux 헤드가 아직 없다). 배포는 aux ON
   이라 절대 BSS 가 배포와 다르다 — 이 표는 **팔 사이 Δ 만** 읽는다. aux ON/OFF
   자체는 §7.118-E 가 이미 쟀다 (현행 유지).

### 사전 등록 판정 (결과 보기 **전에** 적는다)

  1차 관문 — **단독 점수**. §7.24 가 정한 조건이고 §7.26 이 정확히 여기서 죽었다.
             T* 의 emb4 슬롯 단독 BSS 가 CTRL 을 양 fold 에서 이겨야 한다.
  2차 관문 — **배포 층**. 전체 스택(LGBM + CatBoost + δ축 + prior shift)에서
             양 fold Δ, 클러스터 부트스트랩 구간의 **하한**을 같이 본다.
             ⚠️ 부호가 엇갈려도 자동 기각하지 않는다 — 구간이 0 을 포함하면
                부호는 증거가 아니다 (§7.152 전례: 그렇게 기각한 변경이 LB +5.30).
  3차 관문 — **draw 분포**. 3 draw 의 sd 를 Δ 와 맞댄다. §7.161 의 확장 기댓값
             +2.28 vs draw sd 1.71 — 평균이 sd 만하면 그건 추첨이다.
  ✋ 죽는 조건 — T5 ≈ CTRL 인데 T32 만 좋으면 그건 **k 효과(분산축소)** 다.
             §7.161(b) 장부가 이미 `b(1/5 − 1/32) = +5.6` 으로 값을 갖고 있고,
             그 경우 emb4 를 20멤버로 늘리는 쪽이 싸다. TabM 을 쓸 이유가 없다.

실행 (GPU):
    MLP_DEVICE=cuda .venv-gpu/Scripts/python.exe -u src/tabm.py 2>&1 | tee experiments/tabm.log
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

import foldcache as _fc

_fc.CENTER_ASOF = False
KEY = _fc._key()

import pandas as pd                                                 # noqa: E402
import mlp                                                          # noqa: E402
import soften                                                       # noqa: E402
import jointbeta as J                                               # noqa: E402
# ⚠️ catens 를 import 하면 catboost 가 딴려온다 — .venv-gpu 에는 없다.
#    여기서 필요한 건 CatBoost 로짓 캐시를 읽을 키뿐이고, 그 상수는
#    config 에 같은 값으로 있다 (catens.SEEDS/VARIANTS == config.CAT_SEEDS/CAT_VARIANTS).
from config import (ASOF_AXES, CAT_ALPHA, CAT_SEEDS, CAT_VARIANTS,  # noqa: E402
                    CONTAM_AXES, CTX_AXES,
                    DROP_FEATURES, MLP_ALPHA, MLP_SEEDS, MLP_VARIANTS,
                    RECENCY_HALF_LIFE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight                           # noqa: E402
from features import build_features, fit_artifacts                  # noqa: E402
from predict import PROB_CENTER, PROB_SCALE                         # noqa: E402
from validation import bss, shift_to_prior                          # noqa: E402

FOLDS = [2023, 2024]
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED = 42
W_E, W_N = 0.875, 0.125            # §7.120 — emb4 / noid 몫 (LB 확정)
DRAWS = (42, 7, 2024)              # TabM 팔의 draw 시드 3개

# (라벨, tabm cfg 또는 None, 시드들)
ARMS = [
    ("CTRL  emb4 독립 5멤버", None, tuple(MLP_SEEDS[:5])),
    ("T5    TabM-mini k=5", dict(k=5), DRAWS),
    ("T32   TabM-mini k=32", dict(k=32), DRAWS),
]


def fold_inputs(df, fold):
    """배포와 같은 전처리. 반환 (A, C, B, Cv, sizes, y2, w2)."""
    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xtr, Xva = X(tr), X(va)
    Xtr, y2, w2, _, _ = soften.soften(tr, Xtr, y_tr, w, None)      # 배포와 동일
    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    return A, C, B, Cv, sizes, y2, w2


def _train(inp, variant, seed, tabm, tag, fold, t0):
    """멤버(또는 TabM draw) 하나. 로짓 (N,) 을 캐시하고 돌려준다."""
    pth = CACHE / f"tabm_f{fold}_{tag}_{seed}.npz"
    if pth.exists():
        print(f"      캐시 {pth.name}")
        return np.load(pth)["z"]
    A, C, B, Cv, sizes, y2, w2 = inp
    cfg = dict(MLP_VARIANTS[variant][0], aux=None)
    if tabm is not None:
        cfg["tabm"] = tabm
    _, curve = mlp.train_mlp(A, C, y2, w2, B, Cv, sizes, seed=seed,
                             verbose=False, y_aux=None, offset=None, **cfg)
    z = curve[-1]
    # ★ 멤버 상관 — 이 한 줄이 "TabM 이 앟상블로 작동하는가" 를 가른다.
    #   corr ≈ 0.9x  멤버가 같은 신호를 잡고 오차만 갈린다 (정상 앟상블)
    #   corr ≈ 0     공유 몸통이 일을 안 하고 헤드가 따로 노이즈를 적합 중
    #   corr ≈ 1     사실상 1멤버 — r_in 이 다양성을 못 만들었다
    cc = float("nan")
    if z.ndim == 2 and z.shape[1] > 1:
        k_ = z.shape[1]
        cc = float(np.corrcoef(z.T)[np.triu_indices(k_, 1)].mean())
    # TabM 은 (N, k) 를 준다 → **로짓평균.** 배포 블렌드가 로짓공간이라서다
    # (predict.py: z = (1−α)z_lgb + α z_mlp). 확률평균으로 바꾸면 블렌드 층과
    # 축약 순서가 어긋나 이 표가 배포를 대표하지 않는다.
    z = z.mean(1) if z.ndim == 2 else z
    np.savez_compressed(pth, z=z, memcorr=cc)
    print(f"      {tag}/{seed}  ({time.time()-t0:.0f}s)"
          + ("" if cc != cc else f"   멤버상관 {cc:.4f}"))
    return z


def noid_z(inp, fold, t0):
    """noid 3멤버 — 전 팔 공통 상수. aux OFF 로 맞춘다."""
    pth = CACHE / f"tabm_f{fold}_noid.npz"
    if pth.exists():
        print(f"      캐시 {pth.name}")
        return np.load(pth)["z"]
    z = np.mean([_train(inp, "noid", sd, None, "noidmem", fold, t0)
                 for sd in MLP_SEEDS[:3]], axis=0)
    np.savez_compressed(pth, z=z)
    return z


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    codes, uniq = pd.factorize(df["pitcher_id"].to_numpy())
    n_id = len(uniq)
    mclean = ~((gt == "F") & (s < 2023))
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    print(f"train {df.shape}   a_mlp={MLP_ALPHA} a_cat={CAT_ALPHA}   "
          f"device={mlp._device()}\n")

    R, SOLO, DRAW, KEEP = {}, {}, {}, {}
    for fold in FOLDS:
        print(f"  -- fold {fold} --")
        inp = fold_inputs(df, fold)
        Xax, _, _, _ = J.axis_matrix(df, s, fold)
        vi = np.where(s == fold)[0]
        mm, yv = mclean[vi], y[vi]
        ax = (Xax @ dep)[mm]
        yc = yv[mm]
        ybar = yc.mean()
        z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
        z_cat = np.mean([np.load(CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz")["z"]
                         for v in CAT_VARIANTS for sd in CAT_SEEDS], axis=0)
        z_noid = noid_z(inp, fold, t0)

        def solo(zm):
            return bss(yc, shift_to_prior(1 / (1 + np.exp(-zm[mm])), ybar))

        for lab, tabm, seeds in ARMS:
            tag = "ctrl" if tabm is None else f"k{tabm['k']}"
            per = [_train(inp, "emb4", sd, tabm, tag, fold, t0) for sd in seeds]
            ze = np.mean(per, axis=0)
            zm = W_E * ze + W_N * z_noid
            SOLO[(fold, lab)] = solo(zm)
            # draw 분포 — draw 하나짜리 슬롯의 단독 점수. CTRL 은 멤버 하나씩이라
            # 의미가 다르지만(개별 멤버 = 1멤버 풀) 같은 자로 찍어 대조한다.
            DRAW[(fold, lab)] = np.array([solo(W_E * z + W_N * z_noid) for z in per])
            z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * zm
            z = (1 - CAT_ALPHA) * z + CAT_ALPHA * z_cat
            p = shift_to_prior(1 / (1 + np.exp(-(z[mm] + ax))), ybar)
            p = shift_to_prior(PROB_CENTER + PROB_SCALE * (p - PROB_CENTER), ybar)
            R[(fold, lab)] = bss(yc, p)
            KEEP[(fold, lab)] = (yc, z[mm] + ax, codes[vi][mm])
        print("     " + "  ".join(f"{l.split()[0]} {R[(fold, l)]:.1f}"
                                  for l, _, _ in ARMS))

    labs = [l for l, _, _ in ARMS]
    bar = "=" * 88
    print(f"\n{bar}\n[1차 관문] emb4 슬롯 **단독** — §7.24 조건, §7.26 이 여기서 죽었다\n{bar}")
    print(f"  {'팔':<24}" + "".join(f"{f'fold {f}':>12}" for f in FOLDS))
    for l in labs:
        print(f"  {l:<24}" + "".join(f"{SOLO[(f, l)]:>12.1f}" for f in FOLDS))

    print(f"\n{bar}\n[2차 관문] 배포 층 — LGBM + CatBoost + delta축 + prior shift\n{bar}")
    print(f"  {'팔':<24}" + "".join(f"{f'fold {f}':>12}" for f in FOLDS))
    for l in labs:
        print(f"  {l:<24}" + "".join(f"{R[(f, l)]:>12.1f}" for f in FOLDS))

    print("\n  클러스터 부트스트랩 (투수 복원추출 400회) — CTRL 대비")
    for l in labs[1:]:
        for f in FOLDS:
            yv_, z1, cf = KEEP[(f, l)]
            z0 = KEEP[(f, labs[0])][1]
            order = np.argsort(cf, kind="stable")
            bnd = np.searchsorted(cf[order], np.arange(n_id + 1))
            seen = np.where(np.diff(bnd) > 0)[0]
            bs = []
            for _ in range(400):
                idx = np.concatenate(
                    [order[bnd[j]:bnd[j + 1]]
                     for j in rng.choice(seen, size=len(seen), replace=True)])
                yy = yv_[idx]
                bs.append(
                    bss(yy, shift_to_prior(1 / (1 + np.exp(-z1[idx])), yy.mean()))
                    - bss(yy, shift_to_prior(1 / (1 + np.exp(-z0[idx])), yy.mean())))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"     {l:<24} fold {f}: D {R[(f, l)] - R[(f, labs[0])]:+7.1f}"
                  f"  [{lo:+7.1f}, {hi:+7.1f}]")

    print(f"\n{bar}\n[3차 관문] draw 분포 — 평균이 sd 만하면 그건 추첨이다\n{bar}")
    print(f"  {'팔':<24}{'fold':>7}{'n':>4}{'평균':>10}{'sd':>8}{'최소':>10}{'최대':>10}")
    for l in labs:
        for f in FOLDS:
            v = DRAW[(f, l)]
            print(f"  {l:<24}{f:>7}{len(v):>4}{v.mean():>10.1f}"
                  f"{v.std(ddof=1):>8.2f}{v.min():>10.1f}{v.max():>10.1f}")

    print(f"\n{bar}\n[판정]\n{bar}")
    print("  X  T5 ~= CTRL 인데 T32 만 좋으면 -> k 효과(분산축소)다. §7.161(b) 장부가")
    print("     이미 b(1/5 - 1/32) = +5.6 으로 값을 갖고 있다 -> emb4 확장이 더 싸다.")
    print("  O  T5 > CTRL 이면 -> 가중치 공유가 일한다. 그때만 배포를 논의한다.")
    print("  ! 부호가 엇갈려도 구간이 0 을 포함하면 자동 기각하지 않는다 (§7.152).")
    print(f"  총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
