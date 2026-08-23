"""§7.133 — 이력 깊이(`hist_depth`)와 prev 결측 채움 방식. 사용자 지시로 **전 팔 동시**.

발단 (사용자)
    "결측치가 나오기 이전 시즌에서 등장했던 신인들 통계만 써서 채울 수 없나"
    → 그 제안 자체는 **학습 없이 기각**됐다 (아래 ①). 다만 그 조사 과정에서
      진짜 표적이 하나 나왔고 (②), 이 스크립트는 그것과 앞서 미룬 2x2 를 한 번에 잰다.

① 신인 테이블 기각 (측정만으로 결론, 이 스크립트의 대상이 아님)
    투수별 첫 non-NaN prev 값(= 데뷔 다음 등판에 갓 생긴 값)을 gt 별로 모으면
    현재 prior 대비 F +0.0887 (t +8.9) / R -0.0139 (t -2.25) 로 커 보인다.
    그런데 **채워질 행 수로 가중**하고 자기 (시즌 x gt) 층 평균을 빼면:
        F 전체 +0.0196+-0.0082   F 2023+ -0.0296+-0.0184   <- 시대 간 부호 반전
        R 전체 -0.0068+-0.0062   R 2023+ -0.0091+-0.0129
    신인성 항은 전부 1.6 시그마 미만이고, 오염 F 를 빼면 n_eff 가 42/47 로 떨어져
    추정량 노이즈가 효과와 같은 크기가 된다. 남는 것은 신인성이 아니라 **층 레벨**
    이고 그건 이미 init_score 가 수십만 행으로 추정한다.
    상한도 작다: b<=0.009, f=0.014 -> f*b^2 = 1.1e-6 -> +0.46 BSS.

② 진짜 표적 — 결측 표현이 이진인데 실제 조건은 서수다
    prev{1,3,5} 는 완료 경기가 적으면 서로 같은 값이 된다. train 실측:
        완료 0 개 (prev 전부 결측)  29,185 행  1.98%   <- miss_prev_game 이 덮는 유일한 칸
        완료 1 개 (p1==p3==p5)      29,637 행  2.01%
        완료 2~3 개 (p3==p5)        57,183 행  3.88%
    즉 `form_p3_minus_p5` 가 구조적으로 고정값인 행이 **7.86%** 인데 모델은 그것을
    "폼 변화 없음"과 구별할 수단이 없다. 표적 행 수가 기존 플래그의 4 배다.

③ 미뤘던 2x2 — 채움 방식이 플래그와 대체재인가
    현행은 prev 결측을 리그 prior 로 채운다. 그러면
        form_p1_minus_career = prior - career_i
    가 되어 **커리어에 대한 가짜 단조감소 기울기**가 깔린다 (실측: 결측행에서
    범위 [-0.110,+0.038], std 0.0172 — 상수가 아니다). 그 투수 커리어로 채우면
    정확히 0 이 되어 사라지지만, 그러면 `miss_prev_game` 과 완전 공선이 된다.
    주효과만 보면 안 갈린다 -> 2x2 로 상호작용을 본다
    ([[count-what-a-new-correction-double-subtracts]]).

팔 (전부 같은 코드 경로. 바뀌는 것은 `features.HIST_MODE` 와 드랍 컬럼뿐)
    A         현 배포 캐시 (하네스 앵커. base 가 여기 붙는지가 코드 검증이다)
    base      HIST_MODE=None,     플래그 유지     <- 2x2 의 (prior, flag) 칸
    depth     HIST_MODE="depth"                   <- ② 후보. 컬럼 +1
    depthnan  HIST_MODE="depthnan"                <- ② 강한 판. 구조적 고정값 -> NaN
    dplacebo  depth 컬럼을 (시즌 x gt) 안에서 치환  <- 영가설. 주변분포 동일,
                                                     '이력 깊이'라는 내용만 끊는다.
                                                     컬럼 +1 자체의 효과를 뺀다
                                                     ([[noise-floor-is-learner-specific]])
    carry     HIST_MODE="carry",  플래그 유지     <- 2x2 의 (career, flag)
    carrynf   HIST_MODE="carry",  플래그 드랍     <- 2x2 의 (career, noflag)
    basenf    HIST_MODE=None,     플래그 드랍     <- 2x2 의 (prior, noflag)

    ⚠️ carry 팔에는 hist_depth 를 **넣지 않는다** — 넣으면 한 팔에 변경이 둘이 되어
       2x2 칸의 의미가 깨진다 (features.py 에 주석으로 박아뒀다).

사전 검증 (전체 train 에서 학습 없이 통과 확인 완료)
    hist_depth 0/1/2/3 = 1.979% / 2.009% / 3.877% / 92.136%,  depth==0 == miss_prev_game
    carry 팔은 결측행 form_* 5 개가 max|v| = 0.000e+00
    비결측 행은 base 와 **비트 단위로 동일** (depth·carry 둘 다)

채택 기준 (실행 전에 박아둔다)
    ②  (1) depth-base 가 양 fold 양수  (2) 양 fold 에서 depth-dplacebo > 0
        (3) 두 fold 이득 크기비 <= 3  (4) 클러스터 부트스트랩 구간이 0 을 크게 밑돌지 않을 것
    ③  상호작용 (carry-base) - (carrynf-basenf) 의 부호와 크기.
        |상호작용| 이 주효과보다 크면 둘은 대체재다 -> 같이 쓰면 안 된다.
    !! 재학습 축이라 시드 노이즈(std 4~9)가 낀다. |D| < 10 은 무승부로 읽는다.
    !! 부호가 엇갈려도 각 fold 구간을 보고 판단한다
       ([[sign-isnt-evidence-when-the-interval-spans-zero]] — 부호 엇갈려 기각한
       변경이 LB +5.30 신기록이었다).

내 예측: depth 는 (1) 을 통과하지만 dplacebo 와의 차가 작아 (2) 에서 애매해질 것.
        2x2 상호작용은 음수(대체재)가 가장 그럴듯하다.

실행:  .venv/Scripts/python.exe -u src/histdepth.py 2>&1 | tee experiments/histdepth.log
"""
import hashlib
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
from initfix3 import level_init                                     # noqa: E402
from validation import bss, shift_to_prior                          # noqa: E402

FOLDS = [2023, 2024]
LGB_SEEDS = SEED_ENSEMBLE[:3]
MODES = ["base", "depth", "depthnan", "dplacebo", "carry", "carrynf", "basenf"]
HIST = {"base": None, "depth": "depth", "depthnan": "depthnan",
        "dplacebo": "depth", "carry": "carry", "carrynf": "carry", "basenf": None}
DROPCOL = {"carrynf": ["miss_prev_game"], "basenf": ["miss_prev_game"]}
PERM = {"dplacebo"}
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED = 42
# 하네스 앵커 — §7.115/§7.117/§7.120 이 같은 캐시로 찍은 A 팔 값.
# base 가 여기서 +-15 이상 벗어나면 팔이 아니라 코드가 문제다
# ([[simplified-reimplementations-need-a-known-number-check]]).
ANCHOR = {2023: 807.2, 2024: 924.5}

FP = {}          # 팔이 진짜 갈렸는지 증명하는 지문


def _sig(v):
    """순서까지 반영하는 지문. 두 팔이 같은 캐시를 읽으면 여기서 걸린다."""
    return hashlib.sha1(np.ascontiguousarray(v, dtype=np.float64).tobytes()).hexdigest()[:10]


def _permute_within(x, d, seed):
    """(시즌 x gt) 안에서 `hist_depth` 를 치환한다.

    층별 주변분포가 **정확히** 보존되고 '이력 깊이'라는 내용만 끊긴다. 순수 난수
    컬럼이 아니라 실제 값의 재배열이라, 4 수준 서수라는 형태·희소성까지 같다.
    """
    rng = np.random.default_rng(seed)
    v = x["hist_depth"].to_numpy().copy()
    key = (d["season"].to_numpy(dtype=np.int64) * 2
           + (d["game_type"].astype("object").to_numpy().astype(str) == "F"))
    for k in np.unique(key):
        idx = np.where(key == k)[0]
        v[idx] = v[rng.permutation(idx)]
    x = x.copy()
    x["hist_depth"] = v
    return x


def compute(df, fold, mode, t0):
    pth = CACHE / f"histdepth_fold{fold}_{mode}.npz"
    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = F.fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                          k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

    F.HIST_MODE = HIST[mode]                     # <- 팔을 가르는 스위치 (1/2)

    def X(d, seed):
        x = F.build_features(d, art)
        if mode in PERM:
            x = _permute_within(x, d, seed)      # <- 팔을 가르는 스위치 (2/2)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        drop = [c for c in DROP_FEATURES if c in x.columns] + DROPCOL.get(mode, [])
        return x.drop(columns=drop)

    Xva = X(va, SEED)
    # ── 팔 검증 ──────────────────────────────────────────────
    has_h = "hist_depth" in Xva.columns
    assert has_h == (mode in ("depth", "depthnan", "dplacebo")), \
        f"{mode}: hist_depth 유무가 설계와 다르다"
    assert ("miss_prev_game" in Xva.columns) == (mode not in DROPCOL), \
        f"{mode}: 플래그 드랍이 설계와 다르다"
    FP[(fold, mode)] = {
        "ncol": len(Xva.columns),
        "hist": (_sig(Xva["hist_depth"]),
                 tuple(np.bincount(Xva["hist_depth"].to_numpy().astype(int),
                                   minlength=4).tolist())) if has_h else None,
        "form": _sig(Xva["form_p1_minus_career"]),
        "fp5": _sig(np.nan_to_num(Xva["form_p3_minus_p5"].to_numpy(), nan=-9.0)),
    }
    print(f"    [{mode:9s}] 피처 {len(Xva.columns)}개  form {FP[(fold,mode)]['form']}"
          + (f"  hist {FP[(fold,mode)]['hist'][0]} {FP[(fold,mode)]['hist'][1]}"
             if has_h else ""))

    if pth.exists():
        print(f"    캐시 적중 {pth.name}")
        with np.load(pth) as d:
            out = (d["z_lgb"], d["z_mlp"])
        F.HIST_MODE = None
        return out

    Xtr = X(tr, SEED + 1)
    print(f"    fold {fold} [{mode}] 피처 완료 ({time.time()-t0:.0f}s)")
    ss = tr["season"].to_numpy(dtype=np.int64)
    gg = tr["game_type"].astype("object").to_numpy().astype(str)
    aux, _ = modelabel.aux_labels(tr)
    h = soften.fit_h(tr)
    h_row = np.zeros(len(tr))
    for yr, v2 in h.items():
        h_row[(ss == yr) & (gg == "F")] = v2
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
    F.HIST_MODE = None
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
    print(f"train {df.shape}   a={MLP_ALPHA}  init_score=ON  팔 {MODES}\n")

    labs = ["A"] + MODES
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
        arms = {"A": (zl_a, zm_a)}
        for m in MODES:
            arms[m] = compute(df, fold, m, t0)

        # ── 팔이 진짜 갈렸는가 ([[assert-the-cache-key-actually-split]]) ──
        f = FP
        assert f[(fold, "depth")]["hist"][1] == f[(fold, "dplacebo")]["hist"][1], \
            "위약의 주변분포가 후보와 다르다"
        assert f[(fold, "depth")]["hist"][0] != f[(fold, "dplacebo")]["hist"][0], \
            "위약이 후보와 같은 배열이다 (치환이 안 걸렸다)"
        assert f[(fold, "base")]["form"] != f[(fold, "carry")]["form"], \
            "carry 가 base 와 같은 form 을 만들었다"
        assert f[(fold, "base")]["form"] == f[(fold, "depth")]["form"], \
            "depth 가 form 을 건드렸다 (컬럼 추가만이어야 한다)"
        assert f[(fold, "base")]["fp5"] != f[(fold, "depthnan")]["fp5"], \
            "depthnan 이 form_p3_minus_p5 를 안 바꿨다"
        assert f[(fold, "base")]["ncol"] == f[(fold, "carry")]["ncol"] \
            == f[(fold, "basenf")]["ncol"] + 1 == f[(fold, "depth")]["ncol"] - 1, \
            "컬럼 수가 설계와 다르다"

        for lab, (zl, zm) in arms.items():
            z = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm + Xax @ dep
            p = shift_to_prior(1 / (1 + np.exp(-z[mm])), yv[mm].mean())
            R[(fold, lab)] = bss(yv[mm], p)
            KEEP[(fold, lab)] = (yv[mm], z[mm], codes[vi][mm])
        print(f"  fold {fold}: " + "  ".join(f"{k} {R[(fold,k)]:.1f}" for k in labs))
        d0 = R[(fold, "base")] - ANCHOR[fold]
        print(f"    앵커 검증: base - A(기록 {ANCHOR[fold]}) = {d0:+.1f}  "
              f"{'OK' if abs(d0) < 15 else '!! 앵커 이탈 — 코드 문제 의심'}")

    print(f"\n{'='*100}\n█ 판정 — 배포 층 (a={MLP_ALPHA} + delta축 + prior shift, clean 행)\n{'='*100}")
    print(f"  {'팔':<10}" + "".join(f"{f'fold {f}':>11}" for f in FOLDS)
          + "".join(f"{f'Δ vs base ({f})':>17}" for f in FOLDS))
    for lab in labs:
        print(f"  {lab:<10}" + "".join(f"{R[(f,lab)]:>11.1f}" for f in FOLDS)
              + "".join(f"{R[(f,lab)]-R[(f,'base')]:>+17.1f}" for f in FOLDS))

    # ── 클러스터 부트스트랩 ─────────────────────────────────
    print(f"\n  클러스터 부트스트랩 (투수 복원추출 200회, 기준 = base)")
    boot = {}
    for lab in MODES[1:]:
        cells = []
        for f in FOLDS:
            yv, z1, cf = KEEP[(f, lab)]
            z0 = KEEP[(f, "base")][1]
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
            boot[(f, lab)] = (lo, hi)
            cells.append(f"{R[(f,lab)]-R[(f,'base')]:+6.1f} [{lo:+6.1f},{hi:+6.1f}]")
        print(f"     {lab:<10}" + "".join(f"{c:>26}" for c in cells))

    print(f"\n  ② 후보 - 위약 (컬럼 하나 더 붙는 효과를 뺀 순수 내용 성분)")
    for f in FOLDS:
        print(f"     fold {f}:  depth-dplacebo {R[(f,'depth')]-R[(f,'dplacebo')]:+6.1f}"
              f"    depthnan-dplacebo {R[(f,'depthnan')]-R[(f,'dplacebo')]:+6.1f}")

    print(f"\n  ③ 2x2 — 채움 방식(prior/career) x 플래그(유지/드랍)")
    print(f"     {'fold':<8}{'prior+flag':>12}{'prior-flag':>12}{'career+flag':>13}"
          f"{'career-flag':>13}{'주효과(채움)':>15}{'주효과(플래그)':>16}{'상호작용':>12}")
    for f in FOLDS:
        a, b = R[(f, "base")], R[(f, "basenf")]
        c, d = R[(f, "carry")], R[(f, "carrynf")]
        me_fill = ((c + d) - (a + b)) / 2          # career - prior
        me_flag = ((a + c) - (b + d)) / 2          # flag 유지 - 드랍
        inter = (c - a) - (d - b)
        print(f"     {f:<8}{a:>12.1f}{b:>12.1f}{c:>13.1f}{d:>13.1f}"
              f"{me_fill:>+15.1f}{me_flag:>+16.1f}{inter:>+12.1f}")
    print(f"     상호작용 음수 = 둘이 대체재(같이 쓰면 이득이 겹친다).")
    print(f"     |상호작용| > |주효과| 면 주효과만 보고 채택하면 안 된다.")

    print(f"\n  채택 기준: ② (1) depth-base 양 fold 양수 (2) depth-dplacebo 양 fold 양수"
          f" (3) 크기비 <=3 (4) 부트스트랩 구간이 0 을 크게 밑돌지 않을 것")
    print(f"  !! 재학습 축이라 시드 노이즈(std 4~9)가 낀다. |Δ| < 10 은 무승부로 읽는다.")
    print(f"  !! 부호가 엇갈려도 각 fold 구간을 보고 판단한다 (자동 기각 금지).")
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
