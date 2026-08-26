"""§7.119 — 부분 중심화(5컬럼) + **μ 를 컬럼으로도 준다**. 2×2 대조.

사용자 제안: *"pitcher_success_rate, reverse_rate, prev1/3/5 이것들만 가장 최근에 했던
센터링처럼 해봐. 대신 추가 입력으로 각 feature의 평균 값도 같이 넣어줄거야."*

= §7.117 lag 중심화를 (a) **5개 컬럼에만** 걸고, (b) 빼낸 μ 를 **피처로 되돌려준다**.

(b)가 핵심이다. 중심화 v ← v − μ 는 정보를 **버리는** 변환이라 되돌릴 수 없었다.
μ 를 같이 주면 모델이 v = (v−μ) + μ 로 복원할 수 있다 — **가역 변환**이 된다.
즉 "정보를 버리지 않고 표현만 바꾼다". 이러면 §7.106~117 의 손실원 중 '정보 손실'
계열은 전부 무력화되고, 남는 것은 "표현이 학습을 돕느냐" 하나다.

⚠️ **그런데 μ 는 (시즌 × game_type) 의 결정적 함수 = 사실상 시즌 지시자다.**
   그래서 한 팔만 돌리면 §7.116 ⑩·§7.117 이 반복해서 걸린 함정에 또 걸린다 —
   이득/손실이 **중심화** 때문인지 **시즌 지시자 추가** 때문인지 못 가른다.
   → 2×2 로 짠다. (MEMORY: 새 교정이 뭘 두 번 빼는지 세라 / 짝지은 대조 없이 해석 금지)

    팔          중심화(5col)   +5컬럼      설명
    A 현 배포        ✗          ✗         캐시 재사용 (공짜)
    lag5mu          ✓          μ         **사용자 제안**
    mu5             ✗          μ         **대조군** — §7.116 이 빠뜨린 그 대조군의 일반화
    mu5dead         ✗          죽은 축   **위약** — 같은 이름·개수·구조, 축만 요일(§7.78)
    lag5            ✓          ✗         **대조군** — 대상 축소만 (§7.117 은 16컬럼)

읽는 법 (Δ = 팔 − A):
    I = Δ(lag5mu) − Δ(mu5) − Δ(lag5)          상호작용 (컬럼수 성분 c 는 여기서 상쇄)
    μ의 순수 정보값  =  Δ(mu5) − Δ(mu5dead)   (컬럼수 성분을 뺀 값)
    H1 "μ 가 중심화가 버린 것을 복원한다"  →  Δ(lag5)<0, Δ(lag5mu)≈0, I>0
    H2 "μ 자체가 유해한 시즌 지시자다"     →  Δ(mu5) < Δ(mu5dead)
    H3 "둘은 무관하다"                     →  I≈0

★ 위약이 필요한 이유 (병행 세션 §7.120): 거기서 **내용이 0인 컬럼 1개짜리 위약이
  후보와 같은 크기로 움직였다** (위약 −2.1 [−8.0, +4.4] / 후보 −1.4 [−7.2, +4.2]).
  ⚠️ −2.1 은 **구간이 0 을 포함해 부호조차 미확정**이라 "컬럼 1개 = −2.1" 로 인용할
     수 없다. 확립된 것은 방향이 아니라 **"컬럼 수가 다른 팔끼리 비교하면 그 성분이
     효과 추정치에 섞인다"** 는 사실 하나다. 그 섞이는 양은 **미지수**이고, 이 실험이
     그걸 처음 잰다 — 1개에서 잰 값을 5개로 외삽하지 않는 이유이기도 하다
     ([노이즈 바닥은 학습기마다 다르다] / [구간이 0을 포함하면 부호는 증거가 아니다]).
  위약 없이는 Δ(lag5mu) 의 음수를 "μ 가 무용" 으로 잘못 읽는다.
  ⚠️ 위약의 한계: 요일은 train·test 에 같은 수준이 다 있어 μ 의 **범위 밖 외삽**
     성분은 재현하지 않는다. 컬럼 수 성분만 격리한다.

사전 회의론 (기록 기반, 판정에 쓰지 않고 사전 확률로만):
    · 중심화 축 5구현 5패 (§7.106/111/115/116×2/117). 다만 [전적은 근거가 아니다].
    · **컬럼 추가는 4전 4패** (MEMORY: 전이 배율은 얼리는 파라미터 수로 갈린다 —
      트리 분할로 값이 정해지는 컬럼 추가가 전이 손실 배율 68.6 으로 가장 나쁘다).
      μ 컬럼은 정확히 그 유형이다.
    · 검증/2025 의 μ 는 lag 규칙상 **전체 누적 상수**로 학습 최소값보다 약간 아래 →
      트리는 최좌측 잎. 드리프트가 단조라 "최좌측 = 가장 최근 시즌"이라 방향은 맞다.
      §7.116 의 외삽 실패(범위를 크게 벗어남)와 다르다. 아래 verify 가 수치를 찍는다.

채택 기준 (§7.111 개정): 어느 fold 도 0 을 유의하게 밑돌지 않을 것 + fold2024 양수.

★ **검정력 사전 선언** (병행 세션 지적). 이 축의 시드 노이즈는 std 4~9 다
  (config.py:14, §0-4). 그래서:
  · 팔 쌍의 부트스트랩은 **같은 검증 행·같은 투수 리샘플**에서 짝지어 잰다
    → 행 표집 분산은 거의 상쇄된다. 아래 `boot()` 가 A 대비가 아니라 **팔 쌍**을
      직접 부트스트랩하는 이유다 (독립 구간 두 개를 빼면 노이즈가 √2 배가 된다).
  · 그래도 **시드/컬럼추첨 성분은 안 없어진다.** 컬럼 집합이 달라지면 같은 시드라도
    `feature_fraction` 이 다른 부분집합을 뽑는다 — 짝지어도 남는 오차다.
  · 따라서 **|Δ| ≲ 10 인 대비는 이 설계로 판정 불가**로 보고한다. 특히
    `μ의 순수 정보값 = Δ(mu5) − Δ(mu5dead)` 가 그 대역이면 "μ 가 무용" 이라고
    쓰지 않고 **"미판정"** 이라고 쓴다 ([구간이 0을 포함하면 부호는 증거가 아니다]).
  · 판정이 걸린 것은 **Δ(lag5mu) 하나**다 — 채택 여부는 그것만으로 결정된다.
    나머지 셋은 기전 해석용이고, 미판정으로 끝날 수 있음을 미리 인정한다.

실행:  .venv/Scripts/python.exe -u src/asofctr7.py 2>&1 | tee experiments/asofctr7.log
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

import lightgbm as lgb                                              # noqa: E402
import pandas as pd                                                 # noqa: E402
import features as F                                                # noqa: E402
import mlp                                                          # noqa: E402
import modelabel                                                    # noqa: E402
import soften                                                       # noqa: E402
import jointbeta as J                                               # noqa: E402
from config import (ASOF_AXES, CONTAM_AXES, CTX_AXES, DROP_FEATURES,            # noqa: E402
                    ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA,
                    MLP_AUX_LAMBDA, MLP_SEEDS, MLP_VARIANTS, N_ROUNDS,
                    RECENCY_HALF_LIFE, SEED_ENSEMBLE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight                           # noqa: E402
from features import build_features, fit_artifacts                  # noqa: E402
from initfix3 import level_init                                     # noqa: E402
from validation import bss, shift_to_prior                          # noqa: E402

FOLDS = [2023, 2024]
LGB_SEEDS = SEED_ENSEMBLE[:3]
MODES = ["lag5mu", "mu5", "mu5dead", "lag5"]   # 사용자 제안 먼저 — 중단돼도 그건 건진다
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED = 42
VCOL = "asof_pitcher_success_rate"
OUTSIDE = "asof_batter_success_rate"     # 부분 중심화 대상이 **아닌** 컬럼 (불변 확인용)


def verify(art, Xtr, Xva, tr, va, fold, mode, ncol_base):
    """팔이 설계대로 갈렸는지 **학습 전에** 못 박는다.

    MEMORY: 캐시 키가 진짜 갈렸는지 검정하라 / 간이 재구현은 알려진 수치 재현부터.
    여기서 걸러야 4팔 × 2 fold ≈ 140분을 헛돌리지 않는다.
    """
    ac = art["asof_center"]
    cols = list(ac["table"])
    assert len(cols) == 5, f"대상이 5개가 아니다: {cols}"
    assert set(cols) == set(F.CENTER_SUBSET5), f"대상 목록 불일치: {cols}"
    assert OUTSIDE not in cols, "비대상 컬럼이 섞였다"

    # ① 표가 시즌 μ 표와 실제로 다른가 (lag 계열만) — 같으면 zero 를 돌린 것
    if mode in ("lag5", "lag5mu"):
        d = [abs(ac["lag"][VCOL][k] - ac["table"][VCOL][k])
             for k in ac["lag"][VCOL] if k in ac["table"][VCOL]]
        assert max(d) > 1e-4, "lag 표 == 시즌 μ 표 — 팔이 안 갈렸다"

    # ② priors: 중심화한 5개만 0, 나머지는 그대로여야 한다
    want0 = mode in ("lag5", "lag5mu")
    for c in cols:
        p = art["priors"][c]
        assert (p == 0.0) == want0, f"priors[{c}]={p} (want0={want0})"
    assert art["priors"][OUTSIDE] > 0.3, "비대상 컬럼의 priors 가 오염됐다"

    # ③ μ 컬럼 개수 — mu 계열만 +5 (위약도 **같은 개수·같은 이름**이어야 한다)
    want_mu = mode in F._MU_FEATURE_MODES
    extra = len(Xtr.columns) - ncol_base
    assert extra == (5 if want_mu else 0), f"컬럼 증가 {extra} (want_mu={want_mu})"

    # ④ 값이 실제로 중심화됐는가 / 안 됐는가
    s2 = tr["season"].to_numpy()
    g2 = tr["game_type"].astype(str).to_numpy()
    v = Xtr[VCOL].to_numpy(dtype=np.float64)
    m19 = (s2 == 2019) & (g2 == "R") & np.isfinite(v)
    lvl = v[m19].mean()
    if want0:
        assert abs(lvl) < 0.02, f"중심화 안 됐다 (2019 평균 {lvl:.4f})"
    else:
        assert 0.3 < lvl < 0.7, f"중심화되면 안 되는 팔인데 됐다 ({lvl:.4f})"

    line = []
    for yy in sorted(set(s2.tolist())):
        m = (s2 == yy) & (g2 == "R") & np.isfinite(v)
        line.append(f"{yy} {v[m].mean():+.4f}")
    vv = Xva[VCOL].to_numpy(dtype=np.float64)
    mv = (va["game_type"].astype(str).to_numpy() == "R") & np.isfinite(vv)
    print(f"    · 값(R): {'  '.join(line)} | va{fold} {vv[mv].mean():+.4f}")

    # ⑤ 위약 팔이면 죽은 축 표가 실제로 쓰였는지 못 박는다 (팔 분리 검정)
    if mode == "mu5dead":
        assert ac.get("dead") is not None, "위약인데 죽은 축 표가 없다"
        mu_v = Xtr[VCOL + "_mu"].to_numpy(dtype=np.float64)
        s_tr = tr["season"].to_numpy()
        by_szn = [mu_v[s_tr == yy].mean() for yy in sorted(set(s_tr.tolist()))]
        spread = max(by_szn) - min(by_szn)
        print(f"    · 위약 μ 의 시즌 간 산포 {spread:.5f} (μ 팔이면 0.02+ 여야 정상)")
        assert spread < 0.005, f"위약이 시즌 정보를 담고 있다 (산포 {spread:.5f})"
        return

    # ⑥ μ 컬럼의 검증 시즌 값이 학습 범위에서 얼마나 벗어나는가 (사전 등록한 우려)
    #    ⚠️ 검증 μ 는 **game_type 별 상수**다 (lag fallback 이 gt 별로 걸린다).
    #       전체가 한 상수일 거라고 가정했다가 스모크에서 걸렸다.
    if want_mu:
        mu_tr = Xtr[VCOL + "_mu"].to_numpy(dtype=np.float64)
        mu_va = Xva[VCOL + "_mu"].to_numpy(dtype=np.float64)
        gtr = tr["game_type"].astype(str).to_numpy()
        gva = va["game_type"].astype(str).to_numpy()
        assert len(np.unique(mu_va)) <= len(np.unique(gva)), \
            "검증 μ 가 gt 별 상수가 아니다 — lag fallback 이 안 걸렸다"
        for gtv in sorted(set(gva.tolist())):
            mt, mv2 = mu_tr[gtr == gtv], mu_va[gva == gtv]
            assert np.ptp(mv2) == 0.0, f"검증 μ({gtv}) 가 상수가 아니다"
            gap = (mt.min() - mv2[0]) / max(np.std(mu_tr), 1e-9)
            print(f"    · μ({VCOL}) [{gtv}] 학습 [{mt.min():.5f}, {mt.max():.5f}]"
                  f"  검증 {mv2[0]:.5f}  (학습 최소보다 {gap:+.2f}σ 아래)")


def compute(df, fold, mode, t0, ncol_base):
    pth = CACHE / f"ctr7_fold{fold}_{mode}.npz"
    if pth.exists():
        d = np.load(pth)
        print(f"    캐시 적중 {pth.name}")
        return d["z_lgb"], d["z_mlp"]

    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                        center_asof=mode)
    print(f"    fold {fold} [{mode}] 아티팩트 완료 ({time.time()-t0:.0f}s)")

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xtr, Xva = X(tr), X(va)
    verify(art, Xtr, Xva, tr, va, fold, mode, ncol_base)
    ss = tr["season"].to_numpy(dtype=np.int64)
    gg = tr["game_type"].astype("object").to_numpy().astype(str)
    aux, _ = modelabel.aux_labels(tr)
    h = soften.fit_h(tr)
    h_row = np.zeros(len(tr))
    for yr, v in h.items():
        h_row[(ss == yr) & (gg == "F")] = v
    dup = (h_row > 0) & (y_tr == 1)
    Xtr, y2, w2, aux2, _ = soften.soften(tr, Xtr, y_tr, w, aux)
    init_tr = level_init(np.concatenate([ss, ss[dup]]),
                         np.concatenate([gg, gg[dup]]), y2, w2)
    assert len(init_tr) == len(Xtr), "정렬 깨짐"

    z_lgb, n = np.zeros(len(Xva)), 0
    for over in ENSEMBLE_VARIANTS.values():
        for sd in LGB_SEEDS:
            p = dict(FINAL_LGB_PARAMS, **over, seed=sd, bagging_seed=sd,
                     feature_fraction_seed=sd, data_random_seed=sd)
            bst = lgb.train(p, lgb.Dataset(Xtr, y2, weight=w2, init_score=init_tr),
                            num_boost_round=N_ROUNDS)
            z_lgb += bst.predict(Xva, raw_score=True)
            n += 1
            del bst
    z_lgb /= n
    print(f"    fold {fold} [{mode}] lgbm {n}개 ({time.time()-t0:.0f}s)")

    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    zs = []
    for vname, (cfg, nseed) in MLP_VARIANTS.items():
        for sd in MLP_SEEDS[:nseed]:
            _, curve = mlp.train_mlp(A, C, y2, w2, B, Cv, sizes, seed=sd,
                                     verbose=False, y_aux=aux2,
                                     **dict(cfg, aux=MLP_AUX_LAMBDA))
            zs.append(curve[-1])
    z_mlp = np.mean(zs, axis=0)
    print(f"    fold {fold} [{mode}] mlp {len(zs)}개 ({time.time()-t0:.0f}s)")
    np.savez_compressed(pth, z_lgb=z_lgb, z_mlp=z_mlp)
    return z_lgb, z_mlp


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

    # 기준 컬럼 수 — 중심화 없는 아티팩트에서 한 번만 잰다 (verify ③ 의 기준선)
    tr0 = df[s < FOLDS[0]]
    w0 = recency_weight(tr0.season.to_numpy(), RECENCY_HALF_LIFE,
                        ref_season=int(tr0.season.max()))
    art0 = fit_artifacts(tr0, weights=w0, k_pitcher=SMOOTH_K_PITCHER,
                         k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)
    x0 = build_features(tr0.head(1000), art0)
    if USE_PLAYER_IDS:
        for c in ("pitcher_id", "batter_id"):
            x0[c] = tr0.head(1000)[c].to_numpy(dtype=np.float32)
    ncol_base = len(x0.drop(columns=[c for c in DROP_FEATURES
                                     if c in x0.columns]).columns)
    del art0, x0
    print(f"train {df.shape}   α={MLP_ALPHA}  init_score=ON  모드 {MODES}")
    print(f"기준 컬럼 수 {ncol_base}   중심화 대상 {F.CENTER_SUBSET5}\n")

    labs = ["A 현 배포"] + list(MODES)
    R, KEEP = {}, {}
    for fold in FOLDS:
        Xax, _, _, _ = J.axis_matrix(df, s, fold)
        vi = np.where(s == fold)[0]
        mm = mclean[vi]
        yv = y[vi]
        with np.load(CACHE / f"rounds_fold{fold}_r640.npz") as d:
            zl_a = d["k320"]
        with np.load(CACHE / f"fold{fold}_{KEY}.npz") as d:
            zm_a = d["z_mlp"]
            assert np.allclose(d["y"], yv)
        arms = {"A 현 배포": (zl_a, zm_a)}
        for m in MODES:
            arms[m] = compute(df, fold, m, t0, ncol_base)
        for lab, (zl, zm) in arms.items():
            z = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm + Xax @ dep
            p = shift_to_prior(1 / (1 + np.exp(-z[mm])), yv[mm].mean())
            R[(fold, lab)] = bss(yv[mm], p)
            KEEP[(fold, lab)] = (yv[mm], z[mm], codes[vi][mm])
        print(f"  fold {fold}: " + "  ".join(f"{k} {R[(fold,k)]:.1f}" for k in labs))

    print(f"\n{'='*84}\n█ 판정 — 배포 층 (α={MLP_ALPHA} + δ축 + prior shift, clean 행)\n{'='*84}")
    print(f"  {'팔':<14}" + "".join(f"{f'fold {f}':>11}" for f in FOLDS)
          + "".join(f"{f'Δ ({f})':>12}" for f in FOLDS))
    for lab in labs:
        print(f"  {lab:<14}" + "".join(f"{R[(f,lab)]:>11.1f}" for f in FOLDS)
              + "".join(f"{R[(f,lab)]-R[(f,'A 현 배포')]:>+12.1f}" for f in FOLDS))

    def boot(f, lab1, lab0):
        """팔 **쌍**을 같은 행·같은 투수 리샘플에서 짝지어 부트스트랩한다.

        A 대비 구간 두 개를 빼는 것보다 노이즈가 작다 (공통 성분이 상쇄된다).
        단 시드/컬럼추첨 성분은 여기서 안 잡힌다 — 독스트링의 검정력 선언 참조.
        """
        yv, z1, cf = KEEP[(f, lab1)]
        z0 = KEEP[(f, lab0)][1]
        order = np.argsort(cf, kind="stable")
        bnd = np.searchsorted(cf[order], np.arange(n_id + 1))
        seen = np.where(np.diff(bnd) > 0)[0]
        bs = []
        for _ in range(200):
            idx = np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                                  rng.choice(seen, size=len(seen), replace=True)])
            yy = yv[idx]
            bs.append(bss(yy, shift_to_prior(1 / (1 + np.exp(-z1[idx])), yy.mean()))
                      - bss(yy, shift_to_prior(1 / (1 + np.exp(-z0[idx])), yy.mean())))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return R[(f, lab1)] - R[(f, lab0)], lo, hi

    print(f"\n  클러스터 부트스트랩 vs A (투수 복원추출 200회, 짝지음)")
    for lab in labs[1:]:
        cells = [f"{d:+6.1f} [{lo:+6.1f},{hi:+6.1f}]"
                 for d, lo, hi in (boot(f, lab, "A 현 배포") for f in FOLDS)]
        print(f"     {lab:<12}" + "".join(f"{c:>24}" for c in cells))

    print(f"\n  █ 기전 대비 — **팔 쌍 직접 부트스트랩** (구간이 0 을 포함하면 미판정)")
    CONTRASTS = [
        ("μ의 순수 정보값", "mu5", "mu5dead"),
        ("사용자팔 − 위약", "lag5mu", "mu5dead"),
        ("μ의 한계값(중심화 위)", "lag5mu", "lag5"),
        ("중심화의 한계값(μ 위)", "lag5mu", "mu5"),
    ]
    for name, a, b in CONTRASTS:
        cells = [f"{d:+6.1f} [{lo:+6.1f},{hi:+6.1f}]"
                 for d, lo, hi in (boot(f, a, b) for f in FOLDS)]
        print(f"     {name:<22}" + "".join(f"{c:>24}" for c in cells))
    print(f"     (시드 노이즈 std 4~9 → |Δ| ≲ 10 이면 부호를 증거로 쓰지 않는다)")

    print(f"\n  █ 2×2 상호작용  I = Δ(lag5mu) − Δ(mu5) − Δ(lag5)   (컬럼수 성분은 상쇄)")
    for f in FOLDS:
        d = {m: R[(f, m)] - R[(f, "A 현 배포")] for m in MODES}
        print(f"     fold {f}:  중심화만 {d['lag5']:+6.1f}   μ만 {d['mu5']:+6.1f}"
              f"   둘다 {d['lag5mu']:+6.1f}   →  I = {d['lag5mu']-d['mu5']-d['lag5']:+6.1f}")

    print(f"\n  █ 위약 대조 — 컬럼 5개를 붙이는 것 자체의 비용")
    for f in FOLDS:
        d = {m: R[(f, m)] - R[(f, "A 현 배포")] for m in MODES}
        print(f"     fold {f}:  위약(죽은축) {d['mu5dead']:+6.1f}   μ {d['mu5']:+6.1f}"
              f"   →  μ의 순수 정보값 = {d['mu5']-d['mu5dead']:+6.1f}")
        print(f"                사용자 팔 {d['lag5mu']:+6.1f}"
              f"   위약 보정 후 {d['lag5mu']-d['mu5dead']:+6.1f}")

    print(f"\n  ★ 대조 (같은 배포 층, init_score 위에서)")
    print(f"     §7.117  lag 16컬럼            −12.0 / −21.9")
    print(f"     §7.116  zero 16컬럼 (원안)    −30.8 / −56.5")
    print(f"     §7.111  μ_global 혼종          −4.1 / −12.3")
    print(f"\n  채택 기준(§7.111 개정): 어느 fold 도 0 을 유의하게 밑돌지 않을 것 + fold2024 양수.")


if __name__ == "__main__":
    main()
