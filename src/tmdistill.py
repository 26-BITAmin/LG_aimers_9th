"""Lever 3 · 2.75단계 — **미니 student 직접 측정** (§7.159).

§7.158 의 외삽(B_opt +272/+600, 멱법칙 +27/+294)은 시끄럽고 두 추정량이 문턱을
사이에 두고 갈렸다. **여기서는 같은 양을 직접 잰다** — soft 타깃으로 student 를
실제로 학습해 hard 라벨 팔과 쌍체 비교한다. 외삽이 통째로 필요 없어진다.

### 사전 등록 (결과 보기 전 고정)

  · **teacher = 팔 C** (`x + x*_post`). B≈C 였으므로 정보 손실 ~0 이고,
    `pitch_of_pa`·`pitch_no` 를 빼면 "이득의 원천이 순수 privileged" 라는 해석이
    오염 없이 보존된다.
  · **안쪽 OOF**: 창 내부 무작위 K=5, **경기 단위 그룹**. 같은 등판 행이 teacher
    학습에 남으면 등판 수준 노이즈를 외워 `s_i` 로 되돌려준다.
  · **student**: 팔 A 피처(`x` 만) + soft 타깃, **매칭 행 전용 · λ=1**
    (= `m` 이 최대인 모서리 = 최대 감도 설계)
  · **게이트**: 쌍체 Δ 의 CI 하한 > 0 **그리고** 점추정 ≥ **+15**
  · **fold 판정 분리**: 채택은 양 fold 동시 통과를 요구하지 **않는다** —
    요구하는 것은 "양 fold 부호 양수 + 한쪽 이상 게이트 통과"다.
    (2023 은 체제전환 fold 라 분산이 크다. §7.158 에서도 CI 가 [−17, +60] 이었다.
     s38 교훈은 "부호 일치 없이 크기만 보지 마라"였지 "양쪽 유의를 요구하라"가 아니다.)
  · **λ 규칙**: 통과하면 배포 단계에서 λ ∈ {0.5, 0.75, 1} 을 holdout 스윕한다.
    **λ=1 로 실패하면 λ 스윕으로 재도전하지 않는다** — 최대 감도 설계가 실패한
    방향을 배합비로 되살리는 것은 §7.45 류 노브 사냥이다.
  · 구획(R/F/cold-start) Δ 는 **부산물**이다. 게이트는 전체 Δ 로만 판정한다.

### ⚠️ 목적함수를 양 팔에 같게 맞춘다

soft 타깃은 `objective="cross_entropy"`(연속 라벨)로 먹인다. **hard 팔도 같은
목적함수를 쓴다** — 안 그러면 라벨 종류와 목적함수가 같이 바뀌어 쌍체가 아니다.
`binary` 로 돌린 hard 팔을 대조군으로 하나 더 두어 **목적함수 교체 자체가 중립인지**
확인한다 (§7.157 팔 A 와 대조 가능해야 한다).

실행:  .venv/Scripts/python.exe -u src/tmdistill.py 2>&1 | tee experiments/tmdistill.log
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import lightgbm as lgb
import numpy as np
import pandas as pd

import deploy29 as D
import trackman_map as TM
from calsub import EDGE_SEASON
from config import (CENTER_ASOF, CONTAM_AXES, DROP_FEATURES, FINAL_LGB_PARAMS,
                    N_ROUNDS, RECENCY_HALF_LIFE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, USE_PLAYER_IDS)
from data import recency_weight
from features import build_features, fit_artifacts
from tmteach import CATCOL, XSTAR_POST, XSTAR_PRE, with_star
from validation import bss, brier, shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED, K_INNER, B_BOOT = 42, 5, 4000
GATE_PT = 15.0


def fit(X, target, w, Xva, obj="cross_entropy"):
    p = dict(FINAL_LGB_PARAMS, objective=obj, seed=SEED, bagging_seed=SEED,
             feature_fraction_seed=SEED, data_random_seed=SEED)
    bst = lgb.train(p, lgb.Dataset(X, target, weight=w), num_boost_round=N_ROUNDS)
    z = bst.predict(Xva, raw_score=True)
    del bst
    return z


def paired_boot(y, p0, p1, cl, n_cl, rng, B=B_BOOT):
    ybar = y.mean()
    ref = ybar * (1 - ybar)
    n = np.bincount(cl, minlength=n_cl)
    s0 = np.bincount(cl, weights=(p0 - y) ** 2, minlength=n_cl)
    s1 = np.bincount(cl, weights=(p1 - y) ** 2, minlength=n_cl)
    seen = np.flatnonzero(n > 0)
    d = 1e5 * (s0.sum() - s1.sum()) / (n.sum() * ref)
    dr = np.empty(B)
    for i in range(B):
        k = rng.choice(seen, size=len(seen), replace=True)
        dr[i] = 1e5 * (s0[k].sum() - s1[k].sum()) / (n[k].sum() * ref)
    lo, hi = np.percentile(dr, [2.5, 97.5])
    return d, lo, hi, float(np.mean(dr > 0))


def main():
    t0 = time.time()
    df, s, y = D.load_df()
    jn = np.load(CACHE / "tmrow_join.npz")
    tm_idx, trust = jn["tm_idx"], jn["trust"]

    tcols = sorted(set(XSTAR_POST + XSTAR_PRE))
    t = TM.load_tm(extra_cols=["balls_before", "strikes_before", "outs_before"]
                   + [c for c in tcols if c not in TM.TM_COLS]).reset_index(drop=True)
    for c in CATCOL:
        t[c] = pd.Categorical(t[c].astype(str)).codes.astype(np.float32)
    XS = pd.DataFrame(index=df.index, columns=tcols, dtype=np.float32)
    ok = trust & (tm_idx >= 0)
    XS.loc[ok, tcols] = t.loc[tm_idx[ok], tcols].to_numpy(dtype=np.float32)

    gt = df["game_type"].to_numpy()
    clean = ~((gt == "F") & (s < 2023))
    edges = np.quantile(df.loc[s < EDGE_SEASON, "asof_pitcher_n"].to_numpy(float),
                        [.2, .4, .6, .8])
    apn = df["asof_pitcher_n"].to_numpy(dtype=np.float64)
    pair = (np.minimum(df.pitcher_team_id, df.batter_team_id).astype(np.int64) * 100
            + np.maximum(df.pitcher_team_id, df.batter_team_id))
    key = df.season.astype(np.int64) * 10000 + df.game_month * 100 + df.game_dayofweek
    half = df.inning.astype(np.int64) * 2 + (df.top_bottom == "B")
    game = (((pair != pair.shift()) | (key != key.shift())
             | (half < half.shift())).cumsum()).to_numpy()

    rng = np.random.default_rng(20260824)
    verdict = {}
    for fold in D.FOLDS:
        mtr = (s < fold) & clean & trust
        mva = (s == fold) & clean & trust
        tr, va = df[mtr], df[mva]
        w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                           ref_season=int(tr.season.max()))
        art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                            k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                            center_asof=CENTER_ASOF)

        def X(d):
            x = build_features(d, art)
            if USE_PLAYER_IDS:
                for c in ("pitcher_id", "batter_id"):
                    x[c] = d[c].to_numpy(dtype=np.float32)
            return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

        Xtr, Xva = X(tr), X(va)
        ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
        yva = va[TARGET_COL].to_numpy(dtype=np.float64)
        Str = XS[mtr]
        Xtr_c = with_star(Xtr, Str, XSTAR_POST)          # teacher 입력
        gtr = game[mtr]

        print(f"\n{'='*98}\nfold{fold} 라인   학습 {len(tr):,}행 · 평가 {len(va):,}행"
              f"   teacher = C(x+x*_post)\n{'='*98}")

        # ── 안쪽 OOF: 창 내부 경기-그룹 K=5 ─────────────────────────
        ug = np.unique(gtr)
        lab = rng.permutation(len(ug)) % K_INNER
        gk = pd.Series(lab, index=ug).reindex(gtr).to_numpy()
        soft = np.empty(len(tr))
        for k in range(K_INNER):
            trn, hol = np.flatnonzero(gk != k), np.flatnonzero(gk == k)
            z = fit(Xtr_c.iloc[trn], ytr[trn], w[trn], Xtr_c.iloc[hol])
            soft[hol] = 1.0 / (1.0 + np.exp(-z))
            print(f"    OOF {k+1}/{K_INNER}  학습 {len(trn):,} → {len(hol):,}"
                  f"   ({time.time()-t0:.0f}s)")
        print(f"  soft label 품질:  평균 {soft.mean():.5f} (y {ytr.mean():.5f})"
              f"  ·  sd {soft.std():.4f}  ·  corr(s,y) {np.corrcoef(soft, ytr)[0,1]:.4f}"
              f"  ·  OOF Brier {brier(ytr, soft):.5f} (상수 {ytr.mean()*(1-ytr.mean()):.5f})")

        # ── student 두 팔 (같은 행·같은 목적함수, 라벨만 다르다) ────
        z_hard = fit(Xtr, ytr, w, Xva)
        z_soft = fit(Xtr, soft, w, Xva)
        z_bin = fit(Xtr, ytr, w, Xva, obj="binary")      # 목적함수 중립성 대조군
        ybar = yva.mean()
        P = {k: shift_to_prior(1 / (1 + np.exp(-z)), ybar)
             for k, z in (("hard(xent)", z_hard), ("soft(xent)", z_soft),
                          ("hard(binary)", z_bin))}
        for k, p_ in P.items():
            print(f"  [BSS] {k:<14s} {bss(yva, p_):8.1f}")
        print(f"  ⇒ 목적함수 중립성: xent−binary = "
              f"{bss(yva, P['hard(xent)']) - bss(yva, P['hard(binary)']):+.1f}"
              f"  (0 에 가까워야 쌍체가 라벨 종류만 가른다)")

        isf = va["game_type"].to_numpy().astype(str) == "F"
        cold = apn[mva] < edges[0]
        cl, uq = pd.factorize(game[mva])
        n_cl = len(uq)
        print(f"\n  {'구획':<12s}{'행':>9s}{'쌍체 Δ':>10s}{'2.5%':>9s}"
              f"{'97.5%':>9s}{'P>0':>7s}{'판정':>8s}")
        for lab_, m in (("전체", np.ones(len(yva), bool)), ("R", ~isf), ("F", isf),
                        ("cold-start", cold)):
            if m.sum() < 500:
                continue
            d, lo, hi, pg = paired_boot(yva[m], P["hard(xent)"][m], P["soft(xent)"][m],
                                        cl[m], n_cl, rng)
            mark = ""
            if lab_ == "전체":
                mark = "✅통과" if (lo > 0 and d >= GATE_PT) else "❌미달"
                verdict[fold] = (d, lo, hi)
            print(f"  {lab_:<12s}{m.sum():>9,d}{d:>10.1f}{lo:>9.1f}{hi:>9.1f}"
                  f"{pg:>7.2f}{mark:>8s}   {'(부산물)' if lab_ != '전체' else ''}")

    print(f"\n{'='*98}\n사전 등록 판정 — 양 fold 부호 양수 + 한쪽 이상 게이트 통과\n{'='*98}")
    pos = all(v[0] > 0 for v in verdict.values())
    anyp = any(v[1] > 0 and v[0] >= GATE_PT for v in verdict.values())
    for f_, (d, lo, hi) in verdict.items():
        print(f"  fold{f_}  Δ {d:+.1f} [{lo:+.1f}, {hi:+.1f}]"
              f"   {'게이트 통과' if (lo > 0 and d >= GATE_PT) else '게이트 미달'}")
    print(f"  양 fold 부호 양수: {pos}   ·   한쪽 이상 통과: {anyp}")
    print(f"  ⇒ **{'채택 — 배포 단계로 (λ ∈ {0.5, 0.75, 1} 스윕)' if (pos and anyp) else '폐쇄'}**")


if __name__ == "__main__":
    main()
