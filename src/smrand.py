"""§7.162d — **진짜 영가설**: 아무 행이나 증강해도 같은가 (사용자 지적)

§7.162b·c 의 `PLACEBO` 는 위약이 아니라 **짝지은 대조군**이었다 —
pre-2023 · R · y=0 · 시즌별 행 수 매칭으로 "F냐 R이냐" 만 빼고 전부 붙들었다.
그건 *"F 오염 때문인가"* 는 검정하지만 *"대상 선택이 조금이라도 중요한가"* 는
검정하지 못한다. 사용자 지적: *"나라면 증강 대상을 타겟하지 않고 그냥 아무렇게
막 잡을 것 같은데. 결국 PLACEBO나 내가 제안한 거나 다 효과가 있다는 거 아님?"*

맞다. 그래서 **대상을 완전히 푼** 팔을 넣는다.

    RANDOM      학습셋 **전체**에서 34,109 행을 균등 추출해 풀로 삼는다.
                시즌·game_type·라벨 전부 무제한. SMOTE 는 라벨 그룹 **안에서**
                보간하고(교차 클래스 보간은 정의되지 않는다) 합성 행은 씨앗 라벨.
    RANDOM23    같은데 풀을 **2023+ 행**으로 제한. "낡은 행" 이 조건인지 가른다.

비교 대상 (전부 하이브리드 = **LGBM 에만** 증강, MLP/CatBoost 는 base):
    base 935.8 · SMOTE +1.36 [+0.32,+2.40] · PLACEBO +1.33 [+0.04,+2.67]

세 갈래로 읽는다
    RANDOM 도 ≈+1.3   → 대상은 **아무 상관 없다.** 남는 건 "교란 행 13k 개" 뿐.
                        그러면 후보는 원안이 아니라 **일반 증강**이고, 튜닝 축은
                        대상이 아니라 **개수·교란 크기**다.
    RANDOM ≈ 0        → "pre-2023 실패 행" 이 진짜 조건이다 (F/R 은 무관해도).
                        원안의 절반이 맞았던 셈 — 오염이 아니라 **낡음+소수클래스**.
    RANDOM23 만 0     → "낡은 행" 이 조건. recency 가중이 눌러둔 구간을 되살리는 것.

실행:  .venv/Scripts/python.exe -u src/smrand.py 2>&1 | tee experiments/smrand.log
"""
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
import pandas as pd

import jointbeta as J
import modelabel
import soften
from config import (ASOF_AXES, CAT_ALPHA, CTX_AXES, ENSEMBLE_VARIANTS,
                    FINAL_LGB_PARAMS, FSPEC_ALPHA, MLP_ALPHA, MLP_SEEDS,
                    MLP_VARIANTS, N_ROUNDS, SEED_ENSEMBLE, TARGET_COL, TRAIN_CSV)
from data import load_csv
from hbeta2 import season_gt_level, train_fspec
from predict import PROB_CENTER, PROB_SCALE
from smotenc import n_needed, prep, smote, target_mask
from validation import bss, shift_to_prior

FOLD = 2024
LGB_SEEDS = SEED_ENSEMBLE[:3]
C = Path(__file__).parent.parent / "experiments" / "cache"
B_BOOT = 400   # 시간 압박 — B=600/2000 이 일치했으므로 400 으로 줄인다          # §7.162c 에서 B=600 과 2000 이 일치했다 — 1000 이면 충분
POOL_N = 34109         # pre-2023 F & y=0 과 같은 풀 크기
ARMS = ("BOTH2X", "UNION", "PRE_ANY", "RECENT_NEG")


def make_pool(tr, kind, rng):
    """증강 풀 마스크. 대상을 **완전히 푼다** (라벨·시즌·game_type 무제한)."""
    n = len(tr)
    ss = tr["season"].to_numpy()
    yy = tr[TARGET_COL].to_numpy()
    cand = {                                 # 2×2 분해 — 낡음 × 실패
        "RANDOM":     np.arange(n),                              # 무표적 (진짜 영가설)
        "RANDOM23":   np.flatnonzero(ss >= 2023),                # 최근 · 아무 라벨
        "PRE_ANY":    np.flatnonzero(ss <= 2022),                # 낡음 · 아무 라벨
        "RECENT_NEG": np.flatnonzero((ss >= 2023) & (yy == 0)),  # 최근 · 실패만
        # 사용자 질문 — F 와 R 을 **둘 다** 증강한다
        "UNION":  np.flatnonzero((ss <= 2022) & (yy == 0)),      # 낡음 · 실패 (F+R 합집합)
        "BOTH2X": np.flatnonzero((ss <= 2022) & (yy == 0)),      # 같은 풀, **2배 용량**
    }[kind]
    m = np.zeros(n, bool)
    pn = POOL_N * (2 if kind in ("UNION", "BOTH2X") else 1)
    m[rng.choice(cand, size=min(pn, len(cand)), replace=False)] = True
    return m


def build_lgb_arm(tr, Xtr, w, kind, rng):
    """[원본][soften 복제][합성]. 합성은 **라벨 그룹 안에서** 보간한다."""
    y = tr[TARGET_COL].to_numpy(dtype=np.float64)
    s = tr["season"].to_numpy(dtype=np.int64)
    g = tr["game_type"].astype("object").to_numpy().astype(str)
    aux, _ = modelabel.aux_labels(tr)

    X1, y1, w1, _, h = soften.soften(tr, Xtr, y, np.asarray(w, np.float64), aux)
    hr = np.zeros(len(tr))
    for yr, v in h.items():
        hr[(s == yr) & (g == "F")] = v
    dm = (hr > 0) & (y == 1)
    ss1 = np.concatenate([s, s[dm]])
    gt1 = np.concatenate([g, g[dm]])

    nF1 = int(((s <= 2022) & (g == "F") & (y == 1)).sum())
    nF0 = int(((s <= 2022) & (g == "F") & (y == 0)).sum())
    n_syn = n_needed(nF1, nF0, 0.40)          # 13,357 — 다른 팔과 같은 수
    if kind == "BOTH2X":
        n_syn *= 2        # F 13,357 + R 13,357 = 26,714. **용량 축**이라 별도로 읽는다
    pool = make_pool(tr, kind, rng)

    news, srcs, labs = [], [], []
    idx_pool = np.flatnonzero(pool)
    for lab in (0.0, 1.0):                     # 라벨 그룹 안에서만 보간
        sel = idx_pool[y[idx_pool] == lab]
        k = int(round(n_syn * len(sel) / len(idx_pool)))
        if k < 1 or len(sel) < 6:
            continue
        sub = Xtr.iloc[sel].to_numpy(dtype=np.float64)
        nw, sd = smote(sub, k, rng)
        news.append(nw)
        srcs.append(sel[sd])
        labs.append(np.full(k, lab))
    new = np.concatenate(news)
    src = np.concatenate(srcs)
    lab = np.concatenate(labs)

    X1 = pd.concat([X1, pd.DataFrame(new.astype(np.float32), columns=Xtr.columns)],
                   ignore_index=True)
    y1 = np.concatenate([y1, lab])
    w1 = np.concatenate([w1, np.asarray(w, np.float64)[src]])
    ss1 = np.concatenate([ss1, s[src]])
    gt1 = np.concatenate([gt1, g[src]])
    init = season_gt_level(ss1, gt1, y1, w1)
    return X1, y1, w1, init, len(new), float(lab.mean())


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    tr, va, w, X, art = prep(df, FOLD)
    Xtr, Xva = X(tr), X(va)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    isF = va["game_type"].to_numpy().astype(str) == "F"
    z_fs, _ = train_fspec(tr, X, Xva)
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    Xax, _, _, _ = J.axis_matrix(df, df["season"].to_numpy(), FOLD)
    ax = Xax @ dep
    ld = lambda k, n: np.load(C / f"sfull_f{FOLD}_{k}_{n}.npz")["z"]
    mb = np.mean([ld("base", f"mlp_{v}_{s}") for v, (c, n) in MLP_VARIANTS.items()
                  for s in MLP_SEEDS[:n]], axis=0)
    cb = ld("base", "cat")

    def zf(zl):
        z = (1 - CAT_ALPHA) * ((1 - MLP_ALPHA) * zl + MLP_ALPHA * mb) + CAT_ALPHA * cb
        z = z.copy()
        z[isF] = (1 - FSPEC_ALPHA) * z[isF] + FSPEC_ALPHA * z_fs[isF]
        return z + ax

    def sc(z, yy, idx=None):
        if idx is not None:
            z, yy = z[idx], yy[idx]
        m = yy.mean()
        p = shift_to_prior(1 / (1 + np.exp(-z)), m)
        return bss(yy, shift_to_prior(PROB_CENTER + PROB_SCALE * (p - PROB_CENTER), m))

    Z = {k: ld(k, "lgb") for k in ("base", "SMOTE", "PLACEBO")}
    for kind in ARMS:
        f = C / f"smrand_f{FOLD}_{kind}_lgb.npz"
        if f.exists():
            Z[kind] = np.load(f)["z"]
            print(f"  {kind} 재사용")
            continue
        rng = np.random.default_rng(2024)
        X2, y2, w2, init, ns, pr = build_lgb_arm(tr, Xtr, w, kind, rng)
        print(f"  {kind}: 합성 {ns:,}행 (양성비 {pr:.3f})  학습 중...")
        z, n = np.zeros(len(Xva)), 0
        for over in ENSEMBLE_VARIANTS.values():
            for sd in LGB_SEEDS:
                p = dict(FINAL_LGB_PARAMS, **over, seed=sd, bagging_seed=sd,
                         feature_fraction_seed=sd, data_random_seed=sd)
                b = lgb.train(p, lgb.Dataset(X2, y2, weight=w2, init_score=init),
                              num_boost_round=N_ROUNDS)
                z += b.predict(Xva, raw_score=True)
                n += 1
                del b
        z /= n
        np.savez_compressed(f, z=z)
        Z[kind] = z
        print(f"    lgbm {n}개 ({time.time()-t0:.0f}s)")
        del X2, y2, w2, init

    codes, uniq = pd.factorize(va["pitcher_id"].to_numpy())
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    seen = np.where(np.diff(bnd) > 0)[0]
    rng = np.random.default_rng(42)
    boots = [np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                             rng.choice(seen, size=len(seen), replace=True)])
             for _ in range(B_BOOT)]

    z0 = zf(Z["base"])
    b0 = sc(z0, yva)
    print(f"\n{'='*84}\n하이브리드 (LGBM 에만 증강) — base {b0:.1f}"
          f"   부트스트랩 {B_BOOT}회 · 투수 {len(seen)}명\n{'='*84}")
    print("  %-32s %9s %8s %22s %9s" % ("팔", "BSS", "Δ", "95% 구간", "P(Δ>0)"))
    desc = {"SMOTE": "pre-2023 F & y=0 (원안)",
            "PLACEBO": "pre-2023 R & y=0 (짝지은 대조)",
            "RANDOM": "아무 행이나 (진짜 영가설)",
            "RANDOM23": "최근(2023+) · 아무 라벨",
            "PRE_ANY": "낡음(<=2022) · 아무 라벨   <- 분해",
            "RECENT_NEG": "최근(2023+) · 실패만    <- 분해",
            "UNION":  "낡음·실패 F+R 합집합 (같은 용량)  <- 둘 다",
            "BOTH2X": "낡음·실패 F+R **2배 용량**       <- 둘 다"}
    for k in ("SMOTE", "PLACEBO", "UNION", "BOTH2X", "PRE_ANY", "RECENT_NEG", "RANDOM23", "RANDOM"):
        if k not in Z:
            continue
        z1 = zf(Z[k])
        d = np.array([sc(z1, yva, ix) - sc(z0, yva, ix) for ix in boots])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print("  %-32s %9.1f %+8.2f   [%+7.2f, %+7.2f] %9.3f"
              % (desc[k], sc(z1, yva), sc(z1, yva) - b0, lo, hi, (d > 0).mean()))
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
