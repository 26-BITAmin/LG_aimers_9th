"""§7.163 — **NODE** (Popov et al., ICLR 2020) 1단계 측정.

§7.162(TabM) 하네스의 직계다. 바뀌는 것은 **emb4 슬롯의 몸통 하나**고
noid 3멤버·LGBM·CatBoost·δ축·prior shift 는 전부 캐시에서 그대로 온다.
`tabm_f*` 캐시 파일명을 **일부러 그대로 쓴다** — CTRL 과 T5 를 재학습 없이 읽는다.

### 대조군은 CTRL 이 아니라 **T5** 다 (사용자 결정)

§7.162 에서 TabM-mini k=5 가 이 슬롯을 이겼다 (배포층 +21.5 / +19.8, 양 fold
하한 양수). 그러니 물어야 할 질문은 "MLP 를 이기나" 가 아니라 **"TabM 을 이기나"** 다.
CTRL 은 눈금으로만 남긴다.

### ⚠️ 멤버 수 핸디캡을 장부에 적고 시작한다

    CTRL   emb4 독립 5멤버                    → 유효 5
    T5     TabM-mini k=5 · 3 draw            → 유효 15
    N*     NODE 독립 5 draw                   → 유효 5

NODE 는 T5 대비 분산축소에서 진 채로 시작한다. §7.161(b) 장부 b = 33.43 로
크기를 매기면  b(1/5 − 1/15) = **+4.46 BSS** 가 T5 쪽 공짜 이득이다.
→ 판정할 때 이 값을 **더해 주지 않는다** (그건 §7.159 가 데인 외삽이다).
  다만 Δ 가 −4.5 근처의 음수면 "졌다" 가 아니라 **"멤버 수만큼 졌다"** 로 읽는다.

### 팔 (전부 aux=None. §7.162 와 같은 이유 — NODE 쪽 aux 헤드가 없다)

    CTRL   emb4 독립 5멤버                 (캐시)   눈금
    T5     TabM-mini k=5, 3 draw          (캐시)   ★ 대조군
    N64    NODE m=64,  D=4, entmax        5 draw   파라미터 20k < MLP 몸통 26k
    N128   NODE m=128, D=6, entmax        5 draw   파라미터 66k = 2.5배 ← 용량-반응
    NSIG   NODE m=64,  D=4, **sigmoid**   5 draw   게이트 대조군

  N64 vs N128 이 §7.151 이 요구한 **용량-반응**이다 (단일 지점이면 또 "미판정").
  N64 vs NSIG 는 **이득의 출처**를 가른다 — 트리 구조냐, 그냥 곱셈 게이팅이냐.

### 사전 등록 판정 (결과 보기 **전에** 적는다)

  1차 관문 — 단독 점수. N* 의 emb4 슬롯 단독 BSS 가 **양 fold 에서 T5 를** 이겨야
             한다. §7.24 조건이고 §7.26 이 정확히 여기서 죽었다.
  2차 관문 — 배포 층. 양 fold Δ 와 클러스터 부트스트랩 **하한**을 같이 본다.
             ⚠️ 부호가 엇갈려도 구간이 0 을 포함하면 자동 기각 금지 (§7.152).
  3차 관문 — draw 분포. 5 draw 의 sd 를 Δ 와 맞댄다 (§7.161).
  4차 관문 — **에폭 곡선**. 이 하네스는 팔마다 seed 42 의 에폭별 단독 BSS 를
             찍는다. ep9 에서 아직 오르는 중이면 그 팔의 음수는 "NODE 가 졌다"
             가 아니라 **"9 에폭이 모자랐다"** 다. 판정을 유보하고 epochs 를 연다.
  ✋ 죽는 조건 — N64 ≈ NSIG 면 트리 구조가 한 일이 없다는 뜻이다. 그러면
             "미분가능 트리" 라는 기전이 통째로 틀린 것이고 GRANDE 도 안 돌린다.

실행 (GPU):
    MLP_DEVICE=cuda .venv-gpu/Scripts/python.exe -u src/node.py 2>&1 | tee experiments/node.log
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
import modelabel                                                    # noqa: E402
import soften                                                       # noqa: E402
import jointbeta as J                                               # noqa: E402
from config import (ASOF_AXES, CAT_ALPHA, CAT_SEEDS, CAT_VARIANTS,  # noqa: E402
                    CONTAM_AXES, CTX_AXES, MLP_AUX_LAMBDA,
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
TABM_DRAWS = (42, 7, 2024)         # §7.162 가 쓴 draw 시드 — 캐시 키를 맞춘다
NODE_SEEDS = tuple(MLP_SEEDS[:5])  # CTRL 과 **같은 5개**. 멤버 수를 맞춘다

# (라벨, 추가 cfg, 시드들, 캐시 태그, epochs 덮어쓰기)
#   추가 cfg 는 `MLP_VARIANTS["emb4"]` 위에 그대로 얹힌다 — {} 면 순수 MLP,
#   {"tabm": ...} / {"node": ...} / {"grande": ...} 면 그 몸통으로 갈아끼운다.
#   이 형태라야 grande.py 가 `node.ARMS` 만 갈아끼우고 평가 코드를 통째로
#   재사용할 수 있다 (평가 코드가 두 벌이면 조용히 어긋난다 — MODEL.md §5 전례).
#   캐시 태그는 §7.162 와 같은 문자열을 써야 CTRL/T5 가 재학습 없이 읽힌다.
#
# ── ep3 팔은 **1차 실행의 4차 관문이 만든 것**이다 (하네스 결함 수정)
#    `epochs=9` 는 MLP_VARIANTS["emb4"] 에서 물려받은 값인데 NODE 는 ep3 에서
#    정점을 찍고 단조 하락했다 (최종이 정점보다 f2023 −46 / f2024 −15).
#    ⚠️ ep3 은 **같은 검증 fold 위에서 고른 값**이라 그만큼 부풀어 있다. 다만
#       incumbent 의 epochs=9·6 도 역사적으로 같은 방식으로 골랐으므로 계열마다
#       자기 예산을 주는 것이 공정하다. 부풀림은 양쪽에 같이 걸려 있다.
#    ⚠️ fold 별 정점이 다르다 (2023 → ep3, 2024 → ep6). ep3 을 고른 근거는
#       2024 에서 ep3(751.6) 이 ep6(756.1) 과 4.5 밖에 차이 안 나기 때문이다.
ARMS = [
    ("CTRL  emb4 독립 5멤버", {}, tuple(MLP_SEEDS[:5]), "ctrl", None),
    ("T5    TabM-mini k=5", {"tabm": dict(k=5)}, TABM_DRAWS, "k5", None),
    ("N64   NODE m64 D4 entmax",
     {"node": dict(m=64, depth=4, gate="entmax")}, NODE_SEEDS, "n64", None),
    ("N128  NODE m128 D6 entmax",
     {"node": dict(m=128, depth=6, gate="entmax")}, NODE_SEEDS, "n128", None),
    ("NSIG  NODE m64 D4 sigmoid",
     {"node": dict(m=64, depth=4, gate="sigmoid")}, NODE_SEEDS, "nsig", None),
    ("N64e3 NODE m64 D4 entmax ep3",
     {"node": dict(m=64, depth=4, gate="entmax")}, NODE_SEEDS, "n64e3", 3),
    ("N128e3 NODE m128 D6 ep3",
     {"node": dict(m=128, depth=6, gate="entmax")}, NODE_SEEDS, "n128e3", 3),
    ("NSIGe3 NODE m64 D4 sig ep3",
     {"node": dict(m=64, depth=4, gate="sigmoid")}, NODE_SEEDS, "nsige3", 3),
]
BASE = "T5    TabM-mini k=5"        # ★ 대조군. CTRL 은 눈금으로만 쓴다

# 팔마다 **다른** 대조군을 쓰고 싶을 때 (2×2). {팔 라벨: 대조군 라벨}.
# 비어 있으면 전부 BASE 에 붙인다. aux 2×2 에서는 ON 팔을 CTRL_ON 에, OFF 팔을
# CTRL_OFF 에 붙여야 부트스트랩이 **aux 를 통제한 Δ** 를 준다 — 섞으면 aux 효과와
# 몸통 효과가 한 숫자에 뭉친다.
PAIR = {}


def fold_inputs(df, fold):
    """배포와 같은 전처리. 반환 (A, C, B, Cv, sizes, y2, w2, aux2).

    ⚠️ `aux` 를 넘겨도 `X2·y2·w2` 는 안 바뀐다 (`soften.soften` 이 복제 행을 y·w
       기준으로만 만든다) — 그래서 §7.163 의 aux OFF 캐시가 그대로 유효하다.
    """
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
    Xtr, y2, w2, aux2, _ = soften.soften(tr, Xtr, y_tr, w, aux)    # 배포와 동일
    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    return A, C, B, Cv, sizes, y2, w2, aux2


def _train(inp, variant, seed, extra, tag, fold, t0, curve_score=None,
           epochs=None):
    """멤버(또는 draw) 하나. 로짓 (N,) 을 캐시하고 돌려준다.

    `curve_score(z) -> float` 를 주면 **에폭별** 점수를 같이 저장한다 (4차 관문).
    """
    pth = CACHE / f"tabm_f{fold}_{tag}_{seed}.npz"
    if pth.exists():
        d_ = np.load(pth)
        return d_["z"], (d_["curve"] if "curve" in d_ else None)
    A, C, B, Cv, sizes, y2, w2, aux2 = inp
    # 기본은 aux OFF (§7.163 팔 전부). 팔이 {"aux": MLP_AUX_LAMBDA} 를 주면 켜진다.
    # ⚠️ `dict(base, aux=None, **extra)` 로 쓰면 extra 가 "aux" 를 가질 때
    #    "multiple values for keyword argument" 로 죽는다. 병합은 update 로.
    cfg = dict(MLP_VARIANTS[variant][0], aux=None)
    cfg.update(extra or {})
    use_aux = bool(cfg.get("aux"))
    if epochs is not None:
        cfg["epochs"] = int(epochs)
    _, curve = mlp.train_mlp(A, C, y2, w2, B, Cv, sizes, seed=seed,
                             verbose=False, offset=None,
                             y_aux=(aux2 if use_aux else None), **cfg)
    cs = (np.array([curve_score(zz) for zz in curve])
          if curve_score is not None else None)
    z = curve[-1]
    cc = float("nan")
    if z.ndim == 2 and z.shape[1] > 1:
        k_ = z.shape[1]
        cc = float(np.corrcoef(z.T)[np.triu_indices(k_, 1)].mean())
    z = z.mean(1) if z.ndim == 2 else z
    # ★ 로짓 스케일 감시. LightGBM rf 모드 사고(합 vs 평균, BSS −83,000) 자리다.
    #   위/아래를 **다르게** 다룬다:
    #     위쪽  축약이 합이면 m 배(64·128배)로 튄다 → 즉시 세운다. 계속 돌려봐야
    #           나머지 멤버도 같은 사고를 반복할 뿐이다.
    #     아래쪽 "학습이 안 됐다" 는 신호지 코드 사고가 아니다. NODE 초기 std 는
    #           설계상 0.025~0.05 라(nodenet 자기검사 ⑤) 하한에 붙어 있고,
    #           여기서 세우면 30멤버 중 7번째에서 GPU 를 버린다.
    #           → 크게 찍고 계속 간다. 4차 관문(에폭 곡선)이 판정한다.
    sd = float(np.std(z))
    if sd > 5.0:
        raise RuntimeError(f"{tag}/{seed} std(z) = {sd:.4f} — 축약이 합인지 확인하라 "
                           f"(nodenet._selftest 의 sum/mean 비율 검사)")
    flag = "   ⚠️ std 낮다 — 학습이 안 됐는지 에폭 곡선을 보라" if sd < 0.06 else ""
    out = dict(z=z, memcorr=cc)
    if cs is not None:
        out["curve"] = cs
    np.savez_compressed(pth, **out)
    print(f"      {tag}/{seed}  ({time.time()-t0:.0f}s)  std(z) {sd:.3f}"
          + ("" if cc != cc else f"  멤버상관 {cc:.4f}") + flag)
    return z, cs


def noid_z(inp, fold, t0, use_aux=False):
    """noid 3멤버 — 슬롯의 고정 몫. §7.162·§7.162-3 캐시를 그대로 읽는다.

    ⚠️ **aux ON/OFF 로 값이 다르다.** noid 멤버도 보조 헤드를 달고 학습하므로
       `noid.npz` 와 `noidaux.npz` 가 별개다. 2×2 에서 aux ON 팔에 aux OFF noid
       를 물리면 슬롯의 12.5% 가 다른 레짐 값이 되어 상호작용이 오염된다.
       (`W_N = 0.125` 이라 작아 보이지만, 재는 Δ 자체가 한 자릿수다.)
    """
    sfx = "aux" if use_aux else ""
    pth = CACHE / f"tabm_f{fold}_noid{sfx}.npz"
    if pth.exists():
        print(f"      캐시 {pth.name}")
        return np.load(pth)["z"]
    ex = {"aux": MLP_AUX_LAMBDA} if use_aux else {}
    z = np.mean([_train(inp, "noid", sd, ex, f"noid{sfx}mem", fold, t0)[0]
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

    R, SOLO, DRAW, KEEP, CURVE = {}, {}, {}, {}, {}
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
        _nc = {}

        def noidz(use_aux):
            if use_aux not in _nc:
                _nc[use_aux] = noid_z(inp, fold, t0, use_aux)
            return _nc[use_aux]

        def solo(zm):
            return bss(yc, shift_to_prior(1 / (1 + np.exp(-zm[mm])), ybar))

        def ep_score(zz, zn):
            """에폭 곡선용 — 슬롯 단독 점수 (TabM 은 (N,k) 라 먼저 로짓평균)."""
            z1 = zz.mean(1) if zz.ndim == 2 else zz
            return solo(W_E * z1 + W_N * zn)

        for lab, extra, seeds, tag, eps in ARMS:
            z_noid = noidz(bool((extra or {}).get("aux")))
            per, curves = [], []
            for sd in seeds:
                zz, cs = _train(inp, "emb4", sd, extra, tag, fold, t0,
                                curve_score=((lambda zz, _zn=z_noid: ep_score(zz, _zn))
                                             if sd == seeds[0] else None),
                                epochs=eps)
                per.append(zz)
                if cs is not None:
                    curves.append(cs)
            ze = np.mean(per, axis=0)
            zm = W_E * ze + W_N * z_noid
            SOLO[(fold, lab)] = solo(zm)
            DRAW[(fold, lab)] = np.array([solo(W_E * z + W_N * z_noid)
                                          for z in per])
            if curves:
                CURVE[(fold, lab)] = curves[0]
            z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * zm
            z = (1 - CAT_ALPHA) * z + CAT_ALPHA * z_cat
            p = shift_to_prior(1 / (1 + np.exp(-(z[mm] + ax))), ybar)
            p = shift_to_prior(PROB_CENTER + PROB_SCALE * (p - PROB_CENTER), ybar)
            R[(fold, lab)] = bss(yc, p)
            KEEP[(fold, lab)] = (yc, z[mm] + ax, codes[vi][mm])
        print("     " + "  ".join(f"{l.split()[0]} {R[(fold, l)]:.1f}"
                                  for l, _, _, _, _ in ARMS))

    labs = [l for l, _, _, _, _ in ARMS]
    bar = "=" * 92
    print(f"\n{bar}\n[1차 관문] emb4 슬롯 **단독** — §7.24 조건, §7.26 이 여기서 죽었다\n{bar}")
    print(f"  {'팔':<30}" + "".join(f"{f'fold {f}':>12}" for f in FOLDS))
    for l in labs:
        mark = "  <- 대조군" if l == BASE else ""
        print(f"  {l:<30}" + "".join(f"{SOLO[(f, l)]:>12.1f}" for f in FOLDS) + mark)

    print(f"\n{bar}\n[2차 관문] 배포 층 — LGBM + CatBoost + delta축 + prior shift\n{bar}")
    print(f"  {'팔':<30}" + "".join(f"{f'fold {f}':>12}" for f in FOLDS))
    for l in labs:
        mark = "  <- 대조군" if l == BASE else ""
        print(f"  {l:<30}" + "".join(f"{R[(f, l)]:>12.1f}" for f in FOLDS) + mark)

    print("\n  클러스터 부트스트랩 (투수 복원추출 400회)"
          + ("  — 팔마다 짝지은 대조군" if PAIR else f" — {BASE.split()[0]} 대비"))
    for l in labs:
        # PAIR 가 있으면 팔마다 다른 대조군에 붙인다 (aux 2×2 — 층을 맞춰야
        # Δ 가 aux 를 통제한 값이 된다). 없으면 전부 BASE.
        b_ = PAIR.get(l, BASE)
        if l == b_:
            continue
        for f in FOLDS:
            yv_, z1, cf = KEEP[(f, l)]
            z0 = KEEP[(f, b_)][1]
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
            print(f"     {l:<30} fold {f}: D {R[(f, l)] - R[(f, b_)]:+7.1f}"
                  f"  [{lo:+7.1f}, {hi:+7.1f}]   vs {b_.split()[0]}")

    print(f"\n{bar}\n[3차 관문] draw 분포 — 평균이 sd 만하면 그건 추첨이다\n{bar}")
    print(f"  {'팔':<30}{'fold':>7}{'n':>4}{'평균':>10}{'sd':>8}{'최소':>10}{'최대':>10}")
    for l in labs:
        for f in FOLDS:
            v = DRAW[(f, l)]
            print(f"  {l:<30}{f:>7}{len(v):>4}{v.mean():>10.1f}"
                  f"{v.std(ddof=1):>8.2f}{v.min():>10.1f}{v.max():>10.1f}")

    print(f"\n{bar}\n[4차 관문] 에폭 곡선 (seed {MLP_SEEDS[0]}, 슬롯 단독)\n{bar}")
    print("  ep9 에서 아직 오르는 중이면 음수는 'NODE 가 졌다' 가 아니라 '9 에폭이 모자랐다' 다")
    for l in labs:
        for f in FOLDS:
            v = CURVE.get((f, l))
            if v is None:
                continue
            print(f"  {l:<30}{f:>7}  " + " ".join(f"{x:7.1f}" for x in v))

    print(f"\n{bar}\n[판정]\n{bar}")
    print(f"  대조군 = {BASE.split()[0]} (§7.162 채택 후보). CTRL 은 눈금이다.")
    print("  ! 멤버 수 핸디캡: N* 유효 5 vs T5 유효 15 -> b(1/5-1/15) = +4.46 이 T5 쪽")
    print("    공짜 이득. 더해 주지는 않되(§7.159), -4.5 근처 음수는 '멤버 수만큼 졌다'.")
    print("  X  N64 ~= NSIG 이면 트리 구조가 한 일이 없다 -> 기전이 틀렸다. GRANDE 도 취소.")
    print("  X  N128 > N64 이면 용량 법칙(§7.25)이 이 계열에서는 안 걸린다는 뜻 -> 재설계.")
    print("  ! 부호가 엇갈려도 구간이 0 을 포함하면 자동 기각하지 않는다 (§7.152).")
    print(f"  총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
