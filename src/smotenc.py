"""§7.162 — 사용자 원안 실측: pre-2023 F **실패 행**에 SMOTE(NC) 증강

§7.161 은 사용자 제안(SMOTENC)을 **논증으로** 기각하고 β 스윕으로 갈아탔다.
그건 다른 실험이다. 원안을 그대로 잰다.

원안: "F 리그에서 2022 이때까지는 데이터가 이상하니, F 리그에서 실패한 행들에
      대해서 SMOTENC 같은 데이터 증강을 쓰는 건 어떤가."
      (+ "원래는 평균 성공률이 0.5도 안 나오는데 늘려서 낮추기라도 해야 하지 않나")

대상: pre-2023 F & y=0 = **34,109행** (vs y=1 71,199행, 불균형비 1:2.09).
      50:50 으로 맞추려면 합성 **37,090행**.

팔 (전부 배포 설정 위에서 — β=1.0 soft label ON · init_score ON · fspec 사후블렌드)
    base       현 배포
    SMOTE r    설계행렬(76컬럼 float32) 보간. r = 목표 실패비율.
    ROS  r     **같은 수만큼 진짜 실패 행을 복제**. SMOTE 의 보간이 실제로
               무언가를 하는지, 아니면 그냥 실패 행 가중을 올린 것뿐인지 가른다.
               → 이 대조가 없으면 SMOTE 의 이득/손해를 잘못 귀속한다.
    NCRAW r    **원시 컬럼**에서 SMOTENC (범주형은 이웃 최빈값) 후 피처 재빌드.
               설계행렬은 전부 float32 라 NC 가 작동하지 않는다 — 진짜 NC 는 여기다.
    PLACEBO r  같은 조작을 pre-2023 **R** 의 y=0 행에 (시즌별 행 수 매칭).

⚠️ 합성 행의 (season, game_type) 은 씨앗 행의 것을 물려받는다 → recency 가중과
   init_score 레벨에 그대로 반영된다. 그게 실제로 일어날 일이므로 피하지 않는다.
⚠️ soften 은 y=1 행만 건드리므로 합성 행(y=0)과 겹치지 않는다.

실행:  .venv/Scripts/python.exe -u src/smotenc.py 2>&1 | tee experiments/smotenc.log
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

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import soften
from config import (CONTAM_AXES, DROP_FEATURES, FINAL_LGB_PARAMS, FSPEC_ALPHA,
                    N_ROUNDS, RECENCY_HALF_LIFE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    SOFT_LABEL_F, TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from hbeta2 import score, season_gt_level, train_fspec
from validation import bss

FOLDS = [2024, 2023]
SEEDS = (42, 7, 2024)
K_NN = 5
RATIOS = (0.40, 0.50)          # 목표 실패비율. 원본 .324 → .40(부분) / .50(완전균형)
CACHE = Path(__file__).parent.parent / "experiments" / "cache" / "smotenc.npz"


def smote(Xmin, n_new, rng, cat_idx=None, cat_pen=None):
    """SMOTE / SMOTENC. cat_idx 가 없으면 순수 SMOTE(연속 보간)다.

    SMOTENC(Chawla 2002 §6.1): 거리에 범주 불일치마다 **연속축 표준편차의 중앙값**을
    더하고, 합성 행의 범주값은 k 이웃의 **최빈값**으로 정한다.
    """
    n = len(Xmin)
    D = Xmin.astype(np.float64, copy=True)

    # ── 결측 처리 — 이 문제에서는 **NaN 이 정보다** ────────────────────────
    # `asof_pitcher_prev{1,3,5}_game_*` 는 시즌 첫 등판·신인에서 비고(대상 풀의
    # 6.35% 행 · 16/78 컬럼), LightGBM 은 그걸 **독자 분기**로 쓴다. kNN 은 NaN 을
    # 못 받으므로 **거리 계산에만** 중앙값을 채우고, 합성값은 **부모 중 하나라도
    # NaN 이면 NaN 으로 전파**한다. 채운 값을 그대로 내보내면 "경력 없음" 이
    # "평균적 경력" 으로 둔갑해 cold-start 신호가 조용히 사라진다.
    nanm = np.isnan(D)
    if nanm.any():
        med = np.nanmedian(D, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        D = np.where(nanm, med[None, :], D)

    if cat_idx is not None and len(cat_idx):
        con = np.setdiff1d(np.arange(D.shape[1]), cat_idx)
        med = float(np.median(D[:, con].std(axis=0))) if len(con) else 1.0
        cat_pen = med if cat_pen is None else cat_pen
        # 범주축을 med 배로 스케일하면 "값이 다르면 med, 같으면 0" 에 근사한다
        Dm = D.copy()
        Dm[:, cat_idx] = Dm[:, cat_idx] * (cat_pen * 1e3)
    else:
        con, Dm = np.arange(D.shape[1]), D

    nn = NearestNeighbors(n_neighbors=min(K_NN + 1, n)).fit(Dm)
    _, idx = nn.kneighbors(Dm)
    idx = idx[:, 1:]                                   # 자기 자신 제외

    seed = rng.integers(0, n, size=n_new)
    pick = idx[seed, rng.integers(0, idx.shape[1], size=n_new)]
    gap = rng.random((n_new, 1))
    out = D[seed] + gap * (D[pick] - D[seed])          # 연속축 보간

    if cat_idx is not None and len(cat_idx):
        # 범주축: 씨앗의 k 이웃 중 최빈값 (동률이면 작은 코드)
        for j in cat_idx:
            nb = D[idx[seed], j].astype(np.int64)      # (n_new, k)
            mx = int(nb.max()) + 1
            cnt = np.zeros((len(seed), mx), dtype=np.int32)
            for c in range(nb.shape[1]):
                cnt[np.arange(len(seed)), nb[:, c]] += 1
            out[:, j] = cnt.argmax(axis=1)

    if nanm.any():
        out[nanm[seed] | nanm[pick]] = np.nan       # 결측 전파 (위 설명 참조)
    return out, seed


def n_needed(n_pos, n_neg, target):
    """실패비율을 target 으로 만들려면 필요한 합성 실패 행 수."""
    return max(0, int(round((target * n_pos - (1 - target) * n_neg) / (1 - target))))


def prep(df, valid_season):
    tr = df[df.season < valid_season].reset_index(drop=True)
    va = df[df.season == valid_season]
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

    return tr, va, w, X, art


def target_mask(tr, slice_):
    """증강 대상 = pre-2023 (F 또는 R) 의 y=0 행."""
    s = tr["season"].to_numpy()
    g = tr["game_type"].to_numpy().astype(str)
    y = tr[TARGET_COL].to_numpy()
    return (s <= 2022) & (g == slice_) & (y == 0)


def build_arm(tr, Xtr, w, kind, ratio, rng, art, X):
    """(X2, y2, w2, ss2, gt2, n_syn). soften 은 여기서 적용한다 (y=1 만 건드림)."""
    y = tr[TARGET_COL].to_numpy(dtype=np.float64)
    s = tr["season"].to_numpy(dtype=np.int64)
    g = tr["game_type"].astype("object").to_numpy().astype(str)

    Xa, ya, wa, ssa, gta = Xtr, y, np.asarray(w, dtype=np.float64), s, g
    n_syn = 0
    if kind != "base":
        # ⚠️ 합성 행 수는 **항상 F 블록 기준**이다. 1차에서 위약의 n_pos 에
        #    pre-2023 R 의 y=1 전체(~45만)를 넘겨 269,505 행을 주입했다 —
        #    SMOTE 의 13,357 대비 20배라 기하가 깨졌고 그 팔은 무효였다.
        nF1 = int(((s <= 2022) & (g == "F") & (y == 1)).sum())
        nF0 = int(((s <= 2022) & (g == "F") & (y == 0)).sum())
        n_syn = n_needed(nF1, nF0, ratio)

        if kind == "PLACEBO":            # 무결 슬라이스(R) — 시즌별 행 수 매칭
            mF = target_mask(tr, "F")
            mR = target_mask(tr, "R")
            m = np.zeros(len(tr), bool)
            for yr in range(2019, 2023):
                cand = np.flatnonzero(mR & (s == yr))
                m[rng.choice(cand, size=min(len(cand), int((mF & (s == yr)).sum())),
                             replace=False)] = True
            lab = 0.0
        elif kind == "SMOTEPOS":         # **다수 클래스**(성공)를 대신 증강한다
            m = (s <= 2022) & (g == "F") & (y == 1)
            lab = 1.0
        else:
            m = target_mask(tr, "F")
            lab = 0.0

        if n_syn:
            sub = Xtr[m].to_numpy(dtype=np.float64)
            if kind == "ROS":
                pick = rng.integers(0, len(sub), size=n_syn)
                new, seed = sub[pick], pick
            elif kind == "NCRAW":
                new, seed = _ncraw(tr, m, n_syn, rng, art, X)
            elif kind == "JITTER":
                # SMOTE 와 **같은 크기**의 교란을, 이웃 구조 없이 등방으로 준다.
                # 통과하면 이득의 정체는 "다양체를 따라간 보간" 이 아니라 그냥 노이즈다.
                ref, rseed = smote(sub, min(n_syn, 20000), rng)
                sd = np.nanstd(ref - sub[rseed], axis=0)
                pick = rng.integers(0, len(sub), size=n_syn)
                new = sub[pick] + rng.normal(size=(n_syn, sub.shape[1])) * sd[None, :]
                new[np.isnan(sub[pick])] = np.nan
                seed = pick
            elif kind == "SMOTENOID":
                # ID 축만 씨앗값으로 고정. §4 는 pitcher_id 를 **데뷔 시점 프록시**로
                # 쓴다 — 보간이 만드는 '가짜 중간 ID' 가 일하는 건지 가른다.
                new, seed = smote(sub, n_syn, rng)
                for c in ("pitcher_id", "batter_id"):
                    if c in Xtr.columns:
                        j = list(Xtr.columns).index(c)
                        new[:, j] = sub[seed, j]
            else:
                new, seed = smote(sub, n_syn, rng)
            src = np.flatnonzero(m)[seed]
            Xa = pd.concat([Xtr, pd.DataFrame(new.astype(np.float32),
                                              columns=Xtr.columns)],
                           ignore_index=True)
            ya = np.concatenate([y, np.full(n_syn, lab)])
            wa = np.concatenate([wa, wa[src]])       # 씨앗 행의 recency 가중 상속
            ssa = np.concatenate([s, s[src]])
            gta = np.concatenate([g, g[src]])

    if SOFT_LABEL_F:
        # soften 은 df 정렬을 요구한다 → 원본 부분에만 걸고 합성 부분은 뒤에 붙인다
        n0 = len(tr)
        X0, y0, w0, _, _ = soften.soften(tr, Xa.iloc[:n0], ya[:n0], wa[:n0])
        dup = len(X0) - n0
        Xa = pd.concat([X0, Xa.iloc[n0:]], ignore_index=True)
        ya = np.concatenate([y0, ya[n0:]])
        wa = np.concatenate([w0, wa[n0:]])
        h = soften.fit_h(tr)
        hr = np.zeros(n0)
        for yr, v in h.items():
            hr[(s == yr) & (g == "F")] = v
        dm = (hr > 0) & (y == 1)
        ssa = np.concatenate([s, s[dm], ssa[n0:]])
        gta = np.concatenate([g, g[dm], gta[n0:]])
        assert len(Xa) == len(ya) == len(wa) == len(ssa) == n0 + dup + n_syn
    return Xa, ya, wa, ssa, gta, n_syn


def _ncraw(tr, m, n_syn, rng, art, X):
    """원시 컬럼에서 SMOTENC → 피처 재빌드. 범주형이 실제로 살아 있는 유일한 경로."""
    sub = tr[m]
    num = [c for c in sub.columns
           if c not in (TARGET_COL,) and pd.api.types.is_numeric_dtype(sub[c])]
    cat = [c for c in sub.columns
           if c not in (TARGET_COL,) and not pd.api.types.is_numeric_dtype(sub[c])]
    A = sub[num].to_numpy(dtype=np.float64)
    A = np.nan_to_num(A, nan=0.0)
    codes, cats = [], {}
    for c in cat:
        cc = pd.Categorical(sub[c].astype(str))
        codes.append(cc.codes.astype(np.float64))
        cats[c] = cc.categories
    M = np.column_stack([A] + codes) if codes else A
    ci = np.arange(A.shape[1], M.shape[1])
    new, seed = smote(M, n_syn, rng, cat_idx=ci)

    out = sub.iloc[seed].copy().reset_index(drop=True)
    for j, c in enumerate(num):
        out[c] = new[:, j]
    for j, c in enumerate(cat):
        out[c] = pd.Categorical.from_codes(
            np.clip(new[:, A.shape[1] + j].round().astype(int), 0, len(cats[c]) - 1),
            cats[c]).astype(str)
    out[TARGET_COL] = 0
    return X(out).to_numpy(dtype=np.float64), seed


def main():
    df = load_csv(TRAIN_CSV)
    t0 = time.time()
    cache = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        cache = {tuple(k.split("|")): z[k] for k in z.files}
        print("재개 캐시 %d칸" % len(cache))
    res = {}

    # r=0.40 에서 SMOTE 가 두 fold 부호 일치(+10.2/+23.6, 2.3배)를 냈다.
    # 추가 대조군은 **그 지점에만** 건다 — 설명 후보를 하나씩 벗기는 것이 목적이다.
    arms = ([("base", None)]
            + [(k, r) for r in RATIOS for k in ("SMOTE", "ROS", "NCRAW", "PLACEBO")]
            + [(k, 0.40) for k in ("JITTER", "SMOTENOID", "SMOTEPOS")])

    for fold in FOLDS:
        print("\n" + "=" * 92)
        print("=== fold %d ===" % fold)
        print("=" * 92)
        tr, va, w, X, art = prep(df, fold)
        Xtr, Xva = X(tr), X(va)
        yva = va[TARGET_COL].to_numpy(dtype=np.float64)
        va_f = va["game_type"].to_numpy().astype(str) == "F"
        z_fs, n_fs = train_fspec(tr, X, Xva)
        m0 = target_mask(tr, "F")
        print("  증강 대상 pre-2023 F & y=0 : %s행  (y=1 %s행)"
              % ("{:,}".format(int(m0.sum())),
                 "{:,}".format(int(((tr.season.to_numpy() <= 2022)
                                    & (tr.game_type.to_numpy().astype(str) == "F")
                                    & (tr[TARGET_COL].to_numpy() == 1)).sum()))))

        for kind, ratio in arms:
            rng = np.random.default_rng(777)
            X2, y2, w2, ss2, gt2, ns = build_arm(tr, Xtr, w, kind, ratio, rng, art, X)
            init = season_gt_level(ss2, gt2, y2, w2)
            name = "%s%s" % (kind, "" if ratio is None else " r=%.2f" % ratio)
            fp = hashlib.sha1(np.ascontiguousarray(w2).tobytes()
                              + np.ascontiguousarray(y2).tobytes()
                              + np.ascontiguousarray(init).tobytes()
                              + np.ascontiguousarray(
                                  X2.to_numpy(dtype=np.float32)[-1000:]).tobytes()
                              ).hexdigest()[:8]
            ck = (str(fold), fp)
            if ck in cache:
                z, tag = cache[ck], "cache"
            else:
                z = []
                for sd in SEEDS:
                    p = dict(FINAL_LGB_PARAMS, seed=sd, bagging_seed=sd,
                             feature_fraction_seed=sd, data_random_seed=sd)
                    bst = lgb.train(p, lgb.Dataset(X2, y2, weight=w2, init_score=init),
                                    num_boost_round=N_ROUNDS)
                    z.append(bst.predict(Xva, raw_score=True))
                    del bst
                z = np.stack(z)
                cache[ck] = z
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                np.savez(CACHE, **{"|".join(k): v for k, v in cache.items()})
                tag = "%.0fs" % (time.time() - t0)
            r = score(z, yva, va_f)
            res[(fold, name, False)] = r
            line = ("  %-16s 합성%7s  BSS %8.1f +-%4.1f  F행 %8.1f  R행 %7.1f"
                    % (name, "{:,}".format(ns), r[:, 0].mean(),
                       r[:, 0].std(ddof=1), r[:, 1].mean(), r[:, 2].mean()))
            if z_fs is not None:
                zb = z.copy()
                zb[:, va_f] = (1 - FSPEC_ALPHA) * zb[:, va_f] + FSPEC_ALPHA * z_fs[va_f]
                rb = score(zb, yva, va_f)
                res[(fold, name, True)] = rb
                line += "   |+fspec %8.1f  F행 %8.1f" % (rb[:, 0].mean(), rb[:, 1].mean())
            print(line + "  [%s] (%s)" % (fp, tag))
        del tr, va, w, Xtr, Xva

    print("\n" + "=" * 92)
    print("요약 — Δ 는 base(현 배포) 대비 쌍체 3시드.  ROS 는 'SMOTE 의 보간'을 벗기는 대조군.")
    print("=" * 92)
    print("  %-16s %10s %9s %7s %10s %9s %7s  부호"
          % ("팔", "BSS2024", "Δ2024", "±SE", "BSS2023", "Δ2023", "±SE"))
    for kind, ratio in arms:
        name = "%s%s" % (kind, "" if ratio is None else " r=%.2f" % ratio)
        row, ds = "  %-16s" % name, []
        for fold in FOLDS:
            k = (fold, name, True) if (fold, name, True) in res else (fold, name, False)
            if k not in res:
                row += " %10s %9s %7s" % ("-", "-", "-")
                ds.append(None)
                continue
            b = (fold, "base", True) if (fold, "base", True) in res else (fold, "base", False)
            d = res[k][:, 0] - res[b][:, 0]
            row += " %10.1f %+9.1f %7.1f" % (res[k][:, 0].mean(), d.mean(),
                                             d.std(ddof=1) / np.sqrt(len(d)))
            ds.append(d.mean())
        ok = "-" if (name == "base" or ds[0] is None or ds[1] is None) else \
             ("O" if ds[0] * ds[1] > 0 else "X")
        print(row + "  %4s" % ok)
    print("\n  ※ BSS 는 fspec 블렌드 후(fold2024) / 전(fold2023, 앵커 없음).")
    print("  총 %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
