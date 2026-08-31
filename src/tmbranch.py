"""§7.240 **TrackMan CatBoost 잔차 브랜치** — 타팀 B2 레시피의 챔피언 잔차 이식 (`src/tmbranch.py`).

base: `hj16_psucc.zip` 등가 fold 층 (§7.239 와 동일 하네스, 아핀 ON).

■ §7.239 (기각) 와 무엇이 다른가
    §7.239 는 트랙맨 프로필을 **선형 δ 스칼라**로 증류했다 (전이 배율 0.93 가족) — 죽었다.
    타팀이 LB +20 을 실측한 형태는 **트리 학습기에 프로필 39컬럼 투입** (분할 전이
    68.6배 가족)이다. 이번 실험은 그 형태 그대로: CatBoost 잔차 회귀에
    공식 numeric + TM 프로필 29 + 범주형 8 을 넣는다.

■ 층 문제 (§7.207) — 타팀은 leak-free 풀스택 OOF(v1_44)가 있어 잔차 학습이 가능했다.
    우리 등가물 = champfold 챔피언 등가 fold 층:
      정직 칸: fold2023 잔차로 학습 → fold2024 평가 (시간 순방향)
      sanity : fold2024 잔차로 학습 → fold2023 평가 (시간 역방향, 참고)

■ 2×2 대조 (메모리: 위약 통과 ≠ 증분 · 증류 실험의 위약은 자기증류)
    Arm A: 공식 피처만 (numeric+cat) — 타팀 8층 제네릭 스태커의 등가물
    Arm B: 공식 + TM 프로필 29
    **트랙맨 증분 = B − A** 를 따로 잰다. A 자체의 이득은 별도 채널 (제네릭 재적합).

■ 타팀 레시피 준수: R행 ∧ covered 만 학습/적용 · RMSE·iters 850·depth 8·lr .03·l2 25
    · 시드 3 배깅 · 결측 median fill (train-fold 에서 동결 → eval 적용, skew 없음)
    · `season` 컬럼은 제외 (외삽 + §6 버그 2 전례) · `game_type` 제외 (§3.1a)

■ 판정: γ 스윕 (p += γ·corr, 아핀 재보정) · 정직 칸 D>0 · B−A 증분의
    투수 클러스터 부트스트랩 · f2023 함정 감시 (이득이 sanity 칸에만 몰리면 기각)

실행: .venv/Scripts/python.exe -u src/tmbranch.py 2>&1 | tee experiments/tmbranch.log
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
import pandas as pd

import asofdelta
import champfold as CF
import features as F
import hillclimb as H
from colaxis import ANCHOR, ANCHOR_BETA, ANCHOR_EXPECT, std_axis
from config import (CONTAM_AXES, MLP_ALPHA, MLP_SEEDS, RECENCY_HALF_LIFE, ROOT,
                    SEED, SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from gapstrike import EXTRA_SPEC
from tmaxis import (TM_PARQ, build_tm_features, champ_col_axes, make_sc, to_prob)
from validation import bss, shift_to_prior

CHAMP = ROOT / "hj16_psucc.zip"
DIRS = [(2023, 2024, "정직: 학습 f2023 → 평가 f2024"),
        (2024, 2023, "sanity: 학습 f2024 → 평가 f2023 (시간 역방향)")]
SEEDS_CB = [2026, 2027, 2028]
N_TRAIN_SUB = 400_000
B_BOOT = 400
GAMMAS = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                   0.65, 0.80, 1.00])
CAT_COLS = ["pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
            "pitcher_team_id", "batter_team_id", "top_bottom", "base_state"]
DROP_NUM = {"season", "row_id"}
CB_PARAMS = dict(loss_function="RMSE", iterations=850, depth=8,
                 learning_rate=0.03, l2_leaf_reg=25.0, verbose=False,
                 task_type="CPU", thread_count=6, allow_writing_files=False)


def official_numeric_cols(df):
    out = []
    for c in df.columns:
        if c in DROP_NUM or c == TARGET_COL or c in CAT_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def build_matrix(d, T, num_cols, use_tm, fill=None):
    """CatBoost 입력. fill 이 None 이면 (train) median 을 새로 동결해 반환."""
    X = pd.DataFrame(index=range(len(d)))
    for c in num_cols:
        X[c] = pd.to_numeric(d[c].to_numpy(), errors="coerce")
    if use_tm:
        for c in T.columns:
            X[c] = T[c].to_numpy(np.float64)
    X = X.replace([np.inf, -np.inf], np.nan)
    if fill is None:
        fill = {c: float(m) if np.isfinite(m := X[c].median()) else 0.0
                for c in X.columns}
    for c in X.columns:
        X[c] = X[c].fillna(fill[c])
    for c in CAT_COLS:
        X[c] = pd.Series(d[c].to_numpy()).astype("string").fillna(
            "MISSING").astype(str).to_numpy()
    return X, fill


def fold_ctx(df, s, y_all, fold, jt, deployed, rng):
    """챔피언 등가 z0 → p0, resid + TM 피처. §7.239 와 동일 경로."""
    import contam_fit
    import modelabel
    st, mm, yv, d, core, cat, noid, ax, tr, w = CF.fold_state(df, s, y_all, fold)
    r = float(yv.mean())
    E = [np.load(CF.CACHE / f"champ_D8C_f{fold}_emb4_{sd}.npz")["z"]
         for sd in MLP_SEEDS]
    zm = (0.875 * np.mean(E, axis=0) + 0.125 * np.mean(noid, axis=0))[mm]
    z0 = ((1 - H.W_CAT) * ((1 - MLP_ALPHA) * core + MLP_ALPHA * zm)
          + H.W_CAT * cat + ax)
    lab, _, _ = modelabel.recover(tr)
    rev = np.where(np.isnan(lab["reverse"]), 0.0, np.round(lab["reverse"]))
    ct = contam_fit.fit_contam(tr, rev)
    a_r = asofdelta.fit_asof_axes(tr, [(ANCHOR, 1.0)], CF.ASOF_S, ct=ct)[0]
    v_r = F.asof_lookup(d, a_r, ct=ct).astype(np.float64)
    pr0 = 1 / (1 + np.exp(-z0))
    prA = 1 / (1 + np.exp(-(z0 + ANCHOR_BETA * v_r)))
    d_ns = bss(yv, shift_to_prior(prA, r)) - bss(yv, shift_to_prior(pr0, r))
    err = abs(d_ns - ANCHOR_EXPECT[fold]) / abs(ANCHOR_EXPECT[fold])
    print(f"fold {fold}  G0 앵커 D {d_ns:+.2f} (오차 {err:.2%}) "
          f"{'OK' if err < 0.02 else 'FAIL'}")
    assert err < 0.02, "G0 실패"
    z0 = z0 + ANCHOR_BETA * v_r
    wt = recency_weight(tr["season"].to_numpy(), RECENCY_HALF_LIFE,
                        ref_season=int(tr["season"].max()))
    art = F.fit_artifacts(tr, weights=wt, k_pitcher=SMOOTH_K_PITCHER,
                          k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)
    Xv = F.build_features(d, art)
    idx = np.sort(rng.choice(len(tr), size=min(N_TRAIN_SUB, len(tr)),
                             replace=False))
    Xt = F.build_features(tr.iloc[idx], art)
    for nm, b in deployed:
        t = np.asarray(Xt[nm], dtype=np.float64)
        t = t[np.isfinite(t)]
        z0 = z0 + b * std_axis(np.asarray(Xv[nm], dtype=np.float64),
                               float(t.mean()), float(t.std()))
    sc = make_sc(yv, r)
    base = sc(z0)
    p0 = to_prob(1 / (1 + np.exp(-z0)), r)
    T, covered = build_tm_features(jt, d, fold)
    is_r = d["game_type"].astype(str).eq("R").to_numpy()
    print(f"  기준선 {base:.2f} · TM covered {covered.mean():.1%} · "
          f"학습가능(R∧cov) {(is_r & covered).sum():,}행")
    return dict(d=d, yv=yv, r=r, p0=p0, base=base, sc=sc, T=T,
                covered=covered, is_r=is_r)


def main():
    from catboost import CatBoostRegressor
    for k, v in EXTRA_SPEC.items():
        asofdelta.ASOF_SPEC.setdefault(k, v)
    deployed = champ_col_axes(CHAMP)
    print(f"§7.240  TrackMan CatBoost 잔차 브랜치 — {CHAMP.name} 등가 층, 아핀 ON")
    print(f"  2×2: ArmA 공식피처만 / ArmB +TM29 · 시드 {SEEDS_CB} · "
          f"CB {CB_PARAMS['iterations']}it d{CB_PARAMS['depth']} · "
          f"학습/적용 = R∧covered\n")

    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y_all = df[TARGET_COL].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(SEED)

    jt = pd.read_parquet(TM_PARQ)
    jt["row_id"] = jt["row_id"].astype(str)
    base_cols = ["row_id", "season", "pitcher_id", "batter_hand",
                 "balls_before", "strikes_before"]
    key_src = df[base_cols].copy()
    key_src["row_id"] = key_src["row_id"].astype(str)
    jt = jt.merge(key_src, on="row_id", how="inner")
    from tmaxis import ctx_keys
    jt = pd.concat([jt.drop(columns=base_cols[2:]), ctx_keys(jt)], axis=1)

    num_cols = official_numeric_cols(df)
    print(f"  공식 numeric {len(num_cols)} + cat {len(CAT_COLS)} · "
          f"TM 29 (ArmB)\n" + "=" * 118)

    ctx = {}
    for fold in sorted({f for pair in DIRS for f in pair[:2]}):
        ctx[fold] = fold_ctx(df, s, y_all, fold, jt, deployed, rng)
    print("=" * 118)

    # ── 채점 규약 (v2) ────────────────────────────────────────
    # p0 는 이미 아핀(PROB_SCALE) 적용본이다. 후보는 p0 + γ·corr 뒤 **평균만**
    # prior 재정합(shift_to_prior) — to_prob 를 다시 적용하면 PROB_SCALE 이중
    # 적용으로 후보만 ~−8 벌점을 받는다 (v1 곡선의 결함, 부트스트랩과 모순으로 발견).
    cache_p = ROOT / "experiments" / "tmbranch_corr.npz"
    cache = dict(np.load(cache_p)) if cache_p.exists() else {}
    results = {}
    for tr_f, ev_f, label in DIRS:
        c_tr, c_ev = ctx[tr_f], ctx[ev_f]
        m_tr = c_tr["is_r"] & c_tr["covered"]
        m_ev = c_ev["is_r"] & c_ev["covered"]
        resid_tr = (c_tr["yv"] - c_tr["p0"])[m_tr]
        base_p = bss(c_ev["yv"], shift_to_prior(c_ev["p0"], c_ev["r"]))
        print(f"\n[{label}]  기준선(shift 정합) {base_p:.2f} "
              f"(하네스 기준선 {c_ev['base']:.2f})")
        for arm, use_tm in (("A", False), ("B", True)):
            key = f"{tr_f}_{ev_f}_{arm}"
            t0 = time.time()
            if key in cache:
                corr = cache[key]
                print(f"  Arm{arm}: 캐시 재사용")
            else:
                Xtr, fill = build_matrix(c_tr["d"][m_tr].reset_index(drop=True),
                                         c_tr["T"][m_tr].reset_index(drop=True),
                                         num_cols, use_tm)
                Xev, _ = build_matrix(c_ev["d"][m_ev].reset_index(drop=True),
                                      c_ev["T"][m_ev].reset_index(drop=True),
                                      num_cols, use_tm, fill=fill)
                cat_idx = [Xtr.columns.get_loc(c) for c in CAT_COLS]
                preds = []
                for sd in SEEDS_CB:
                    mdl = CatBoostRegressor(random_seed=sd, **CB_PARAMS)
                    mdl.fit(Xtr, resid_tr, cat_features=cat_idx)
                    preds.append(mdl.predict(Xev))
                corr = np.zeros(len(c_ev["yv"]))
                corr[m_ev] = np.mean(preds, axis=0)
                cache[key] = corr
                np.savez_compressed(cache_p, **cache)
            results[(tr_f, ev_f, arm)] = corr
            dt = time.time() - t0
            row = []
            for g in GAMMAS:
                p_ = np.clip(c_ev["p0"] + g * corr, 1e-6, 1 - 1e-6)
                row.append(bss(c_ev["yv"],
                               shift_to_prior(p_, c_ev["r"])) - base_p)
            results[(tr_f, ev_f, arm, "curve")] = np.array(row)
            j = int(np.argmax(row))
            print(f"  Arm{arm} ({'공식+TM' if use_tm else '공식만':>6})  "
                  f"{dt/60:.1f}분  corr sd {corr[m_ev].std():.5f}  "
                  f"γ* {GAMMAS[j]:.2f}  D* {row[j]:+.2f}")
            print("     γ:  " + "  ".join(f"{g:.2f}:{v:+.2f}"
                                          for g, v in zip(GAMMAS, row)))
        cA = results[(tr_f, ev_f, "A", "curve")]
        cB = results[(tr_f, ev_f, "B", "curve")]
        print(f"  TM 증분 (B−A, γ 짝지음): "
              + "  ".join(f"{g:.2f}:{b - a:+.2f}"
                          for g, a, b in zip(GAMMAS, cA, cB)))

    # ── 정직 칸 부트스트랩: fold2024 에서 B−A 와 B−base ────────
    c_ev = ctx[2024]
    yv, r, p0 = c_ev["yv"], c_ev["r"], c_ev["p0"]
    cA = results[(2023, 2024, "A", "curve")]
    cB = results[(2023, 2024, "B", "curve")]
    gA = float(GAMMAS[int(np.argmax(cA))])
    gB = float(GAMMAS[int(np.argmax(cB))])
    pa = shift_to_prior(np.clip(p0 + gA * results[(2023, 2024, "A")],
                                1e-6, 1 - 1e-6), r)
    pb = shift_to_prior(np.clip(p0 + gB * results[(2023, 2024, "B")],
                                1e-6, 1 - 1e-6), r)
    pz = shift_to_prior(p0, r)
    codes, uniq = pd.factorize(c_ev["d"]["pitcher_id"].to_numpy())
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    seen = np.where(np.diff(bnd) > 0)[0]
    br_ = np.random.default_rng(SEED)
    dBb, dBA = [], []
    for _ in range(B_BOOT):
        i = np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                            br_.choice(seen, size=len(seen), replace=True)])
        sb = bss(yv[i], pb[i])
        dBb.append(sb - bss(yv[i], pz[i]))
        dBA.append(sb - bss(yv[i], pa[i]))
    for nm, dd in (("B−base", dBb), ("B−A(TM증분)", dBA)):
        lo, hi = np.percentile(dd, [2.5, 97.5])
        dd = np.asarray(dd)
        print(f"\n  G3 {nm}  γA {gA:.2f}/γB {gB:.2f}  중앙값 {np.median(dd):+.2f}  "
              f"95% [{lo:+.2f}, {hi:+.2f}]  P(>0) {(dd > 0).mean():.3f}")


if __name__ == "__main__":
    main()
