"""§7.162 3단계 — **2×2 (TabM × aux) + `r_init` 위약.** 여기가 진짜 판정이다.

1·2단계는 전 팔 `aux=None` 이었다. 그런데 §7.118-E 실측이 **aux 제거 = −15.6 / −32.4**
이고, §7.42 는 aux 이득의 대부분이 **정규화 성분**이라고 쟀다 (`rand` 가 `mode` 의 81%).
TabM 도 정규화다. **둘은 대체재 후보다.**

저장소에 같은 사고가 두 번 있다:
    §7.109  소프트라벨 × init_score 를 대체재로 판정 (같이 켜서 −727)
    §7.161⑤ 위약·부호·기전 다 통과한 β 후보가 init_score 를 켜자 **네 칸 전부 부호 반전**

→ 주효과 +21.5 는 증거가 아니다. **2×2 상호작용**을 봐야 갈린다.

### 팔 (emb4 슬롯. noid 는 같은 aux 조건으로 따로 학습해 맞춘다)

              aux OFF                aux ON (배포)
    CTRL      독립 5멤버              독립 5멤버
    T5        TabM k=5 sign, 3 draw  TabM k=5 sign, 3 draw     <- 판정 열은 오른쪽
    P5        TabM k=5 **ones**      TabM k=5 **ones**         <- 위약

### 위약 설계 — `r_init="ones"` 가 왜 옳은 영가설인가

TabM 의 이득에는 최소 두 경로가 있다:
    (a) 멤버 다양성      r_i 가 서로 달라 k 개의 다른 함수가 앙상블된다
    (b) 최적화 부작용    손실이 k 멤버 평균이라 공유 가중치의 경사 분산이 1/k 로 줄고,
                        BatchNorm 이 (B*k) 를 보므로 통계가 안정된다
                        -> "그냥 배치를 키운 것"과 구분이 안 된다

`ones` 는 **(b) 를 전부 남기고 (a) 만 지운다** — 같은 몸통, 같은 BN 기하, 같은 경사
평균, 같은 헤드 수인데 입력 어댑터만 전부 1 이라 멤버가 헤드에서만 갈린다.
memory: "위약은 공유 항을 남겨야 한다" · "분산을 키우는 조작에는 온도 대조군".

    P5 ~= T5   -> 이득은 (b). 배치/에폭 노브로 더 싸게 살 수 있다 -> TabM 불필요
    P5 <<  T5  -> 이득은 (a). 가중치 공유 앙상블이 진짜다

⚠️ `ones` 분기는 RNG 를 안 쓴다 (sign 은 k*d 개를 뽑는다) -> `head_w` 초기값이
   갈린다. 그건 draw 노이즈 안이고 3 draw 로 평균한다.

### 사전 등록 판정

  판정 열 = **Δ(aux ON) = T5/on − CTRL/on**, 양 fold + 부트스트랩 구간.
  상호작용 = Δ(ON) − Δ(OFF).  |상호작용| 이 Δ(OFF) 의 절반을 넘으면 §7.161⑤ 재현이다.
  ✋ Δ(aux ON) 이 양 fold 에서 0 을 포함하면 -> 대체재. 축을 닫는다.
  ✋ P5 가 T5 의 70% 이상이면 -> 다양성 채널이 아니다. TabM 을 쓸 이유가 없다.

실행 (GPU):
    MLP_DEVICE=cuda .venv-gpu/Scripts/python.exe -u src/tabm3.py 2>&1 | tee experiments/tabm3.log
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

import tabm as T1
import mlp
import modelabel
import soften
import jointbeta as J
from config import (ASOF_AXES, CAT_ALPHA, CAT_SEEDS, CAT_VARIANTS, CONTAM_AXES,
                    CTX_AXES, DROP_FEATURES, MLP_ALPHA, MLP_AUX_LAMBDA,
                    MLP_SEEDS, MLP_VARIANTS, RECENCY_HALF_LIFE,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV,
                    USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from predict import PROB_CENTER, PROB_SCALE
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
CACHE = T1.CACHE
W_E, W_N = T1.W_E, T1.W_N
DRAWS = T1.DRAWS
S5 = tuple(MLP_SEEDS[:5])

# (라벨, tabm cfg, aux 켜기, 시드들, 캐시 태그)
ARMS = [
    ("CTRL  독립 5멤버", None, False, S5, "ctrl"),
    ("T5    TabM sign", dict(k=5, r_init="sign"), False, DRAWS, "k5"),
    ("P5    TabM ones", dict(k=5, r_init="ones"), False, DRAWS, "p5"),
    ("CTRL  독립 5멤버", None, True, S5, "ctrlaux"),
    ("T5    TabM sign", dict(k=5, r_init="sign"), True, DRAWS, "k5aux"),
    ("P5    TabM ones", dict(k=5, r_init="ones"), True, DRAWS, "p5aux"),
]


def fold_inputs(df, fold):
    """T1.fold_inputs 와 같은데 **보조 라벨 aux2 까지** 돌려준다."""
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
    aux, _ = modelabel.aux_labels(tr)                  # 배포와 같은 경로 (mlpinit)
    Xtr, y2, w2, aux2, _ = soften.soften(tr, Xtr, y_tr, w, aux)
    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    return A, C, B, Cv, sizes, y2, w2, aux2


def train(inp, variant, seed, tabm, use_aux, tag, fold, t0):
    pth = CACHE / f"tabm_f{fold}_{tag}_{seed}.npz"
    if pth.exists():
        return np.load(pth)["z"]
    A, C, B, Cv, sizes, y2, w2, aux2 = inp
    cfg = dict(MLP_VARIANTS[variant][0], aux=(MLP_AUX_LAMBDA if use_aux else None))
    if tabm is not None:
        cfg["tabm"] = tabm
    _, curve = mlp.train_mlp(A, C, y2, w2, B, Cv, sizes, seed=seed, verbose=False,
                             y_aux=(aux2 if use_aux else None), offset=None, **cfg)
    z = curve[-1]
    cc = (float(np.corrcoef(z.T)[np.triu_indices(z.shape[1], 1)].mean())
          if z.ndim == 2 and z.shape[1] > 1 else float("nan"))
    z = z.mean(1) if z.ndim == 2 else z
    np.savez_compressed(pth, z=z, memcorr=cc)
    print(f"      {tag}/{seed}  ({time.time() - t0:.0f}s)"
          + ("" if cc != cc else f"   멤버상관 {cc:.4f}"))
    return z


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    import pandas as pd
    codes, uniq = pd.factorize(df["pitcher_id"].to_numpy())
    n_id = len(uniq)
    mclean = ~((gt == "F") & (s < 2023))
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    print(f"2x2 (TabM x aux) + ones 위약   device={mlp._device()}\n")

    R, KEEP, MC = {}, {}, {}
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
        # noid 도 aux 조건마다 따로 — 팔 안에서 조건이 섞이면 2x2 가 깨진다
        noid = {}
        for ua, tg in ((False, "noid"), (True, "noidaux")):
            p = CACHE / f"tabm_f{fold}_{tg}.npz"
            if p.exists():
                noid[ua] = np.load(p)["z"]
            else:
                noid[ua] = np.mean([train(inp, "noid", sd, None, ua,
                                          f"{tg}mem", fold, t0)
                                    for sd in MLP_SEEDS[:3]], axis=0)
                np.savez_compressed(p, z=noid[ua])

        for lab, tabm, ua, seeds, tag in ARMS:
            per = [train(inp, "emb4", sd, tabm, ua, tag, fold, t0) for sd in seeds]
            MC[(fold, tag)] = [float(np.load(CACHE / f"tabm_f{fold}_{tag}_{sd}.npz")
                                     ["memcorr"]) for sd in seeds]
            zm = W_E * np.mean(per, axis=0) + W_N * noid[ua]
            z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * zm
            z = (1 - CAT_ALPHA) * z + CAT_ALPHA * z_cat
            p = shift_to_prior(1 / (1 + np.exp(-(z[mm] + ax))), ybar)
            p = shift_to_prior(PROB_CENTER + PROB_SCALE * (p - PROB_CENTER), ybar)
            R[(fold, tag)] = bss(yc, p)
            KEEP[(fold, tag)] = (yc, z[mm] + ax, codes[vi][mm])
        print("     " + "  ".join(f"{t} {R[(fold, t)]:.1f}"
                                  for _, _, _, _, t in ARMS))

    bar = "=" * 84
    print(f"\n{bar}\n[2x2] 배포 층 BSS\n{bar}")
    print(f"  {'fold':>6}  {'팔':<18}{'aux OFF':>10}{'aux ON':>10}{'aux 값어치':>12}")
    for fold in FOLDS:
        for lab, o, n in (("CTRL", "ctrl", "ctrlaux"), ("T5 sign", "k5", "k5aux"),
                          ("P5 ones", "p5", "p5aux")):
            print(f"  {fold:>6}  {lab:<18}{R[(fold, o)]:>10.1f}{R[(fold, n)]:>10.1f}"
                  f"{R[(fold, n)] - R[(fold, o)]:>+12.1f}")

    print(f"\n{bar}\n[판정 열] Δ = 팔 − CTRL, **같은 aux 조건 안에서**\n{bar}")
    print(f"  {'fold':>6}  {'팔':<10}{'Δ(aux OFF)':>13}{'Δ(aux ON)':>13}"
          f"{'상호작용':>11}")
    for fold in FOLDS:
        for lab, o, n in (("T5 sign", "k5", "k5aux"), ("P5 ones", "p5", "p5aux")):
            do = R[(fold, o)] - R[(fold, "ctrl")]
            dn = R[(fold, n)] - R[(fold, "ctrlaux")]
            print(f"  {fold:>6}  {lab:<10}{do:>+13.1f}{dn:>+13.1f}{dn - do:>+11.1f}")

    print(f"\n{bar}\n[부트스트랩] 투수 복원추출 400회 — **aux ON 조건에서만**\n{bar}")
    for tag, lab in (("k5aux", "T5 sign"), ("p5aux", "P5 ones")):
        for fold in FOLDS:
            yv_, z1, cf = KEEP[(fold, tag)]
            z0 = KEEP[(fold, "ctrlaux")][1]
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
            d = R[(fold, tag)] - R[(fold, "ctrlaux")]
            print(f"     {lab:<10} fold {fold}: Δ {d:>+7.1f}  [{lo:>+7.1f}, {hi:>+7.1f}]"
                  f"   {'0 배제' if lo * hi > 0 else '0 포함'}")

    print(f"\n{bar}\n[위약 비율] P5 / T5 — 70% 넘으면 다양성 채널이 아니다\n{bar}")
    for fold in FOLDS:
        dt = R[(fold, "k5aux")] - R[(fold, "ctrlaux")]
        dp = R[(fold, "p5aux")] - R[(fold, "ctrlaux")]
        r = dp / dt if abs(dt) > 1e-9 else float("nan")
        print(f"     fold {fold}: T5 {dt:+.1f}  P5 {dp:+.1f}   비 {r:>6.1%}"
              f"   {'!! 위약이 먹었다' if r > 0.70 else 'OK'}")

    print(f"\n{bar}\n[멤버상관]\n{bar}")
    for fold in FOLDS:
        for tag in ("k5", "k5aux", "p5", "p5aux"):
            v = [c for c in MC[(fold, tag)] if c == c]
            if v:
                print(f"     fold {fold} {tag:<8} {np.mean(v):.4f}")
    print(f"\n  총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
