"""Lever 3 · 2단계 — **teacher gap** = distillation 이득의 상한 (§7.157).

사용자 사양 그대로. 측정 대상은 "teacher − 배포 스택"이 **아니다** — 그 차이에는
모델 계열·피처 파이프라인·fold 구성이 섞인다. 상한의 올바른 추정량은
**쌍둥이 teacher 의 쌍체 비교**다:

    gap ≡ BSS[T(x, x*)] − BSS[T(x)]

두 팔은 `x*` 컬럼 유무만 다르고 나머지 전부(모델 계열·하이퍼·fold 절단·행 집합)
동일하다. 배포 스택 대비 수치는 **참고용 부차 출력**으로만 낸다.

### fold 구조 — 바깥은 시간, 안쪽은 (3단계에서) 경기-그룹 무작위

바깥 절단은 시간이고 학생 평가 경계와 같다. "teacher 는 배포 안 되니 시간 누출이
규정 위반은 아니다"는 절반만 맞다 — **규정이 아니라 측정 오염 문제**다. fold2023
평가용 teacher 가 2023 행을 봤다면 그 정보가 파라미터를 거쳐 ≤2022 행의 `s_i` 에
스며들고 holdout 점수가 낙관 편향된다. 그 누출은 판정하려는 효과 크기(한 자릿수~
수십 BSS)와 **같은 자릿수**로 들어온다.

    fold2023 라인 → teacher 는 ≤2022 만
    fold2024 라인 → teacher 는 ≤2023 만

안쪽 OOF(창 내부 경기-그룹 K=5)는 **3단계 student 용**이다. 2단계의 판정 수치는
창→holdout 예측이므로 여기서는 만들지 않는다 ("격차가 유의미할 때만" 규칙).

### 라벨 정책 (2단계)

teacher 학습에서 **pre-2023 F 오염 행은 제외**한다 (soften 이 아니라 제외 —
teacher 의 목적은 참 `p*` 추정이고 오염 라벨은 `x*→y` 관계 자체를 왜곡한다).
gap 평가는 holdout 2023/2024 에서만 = 정의상 깨끗한 구간.

⚠️ **사전 고정**: fold2023 라인의 teacher 창(≤2022)은 오염 제외로 F 가 거의 비므로
`gap_F` 는 **fold2024 라인만** 신뢰한다.

### ★ 사양에 팔을 하나 더한다 — privileged 와 "빠진 피처"를 가른다

trackman 30컬럼 중 `pitch_of_pa`·`pitch_no` 는 **투구 전에 알 수 있다.** 이것이
gap 에 섞이면 privileged information 이득이 아니라 **그냥 빠진 피처**를 재는 것이고,
그 경우 옳은 처방은 distillation 이 아니라 컬럼 추가다. 팔을 갈라 놓는다:

    A  x            (기준)
    B  x + x*_all   (사양 그대로 — 30컬럼 원본)
    C  x + x*_post  (투구 후에만 알 수 있는 측정값만) ← **판정은 이쪽**

실행:  .venv/Scripts/python.exe -u src/tmteach.py 2>&1 | tee experiments/tmteach.log
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
from calslope import build_layers
from calsub import EDGE_SEASON
from config import (CENTER_ASOF, CONTAM_AXES, DROP_FEATURES, FINAL_LGB_PARAMS,
                    N_ROUNDS, RECENCY_HALF_LIFE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, USE_PLAYER_IDS)
from data import recency_weight
from features import build_features, fit_artifacts
from predict import PROB_CENTER, PROB_SCALE
from validation import bss, shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED = 42
B_BOOT = 4000

# 투구 **후에만** 알 수 있는 측정값 — 진짜 privileged information
XSTAR_POST = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
              "extension", "rel_height", "rel_side", "zone_speed",
              "tagged_pitch_type", "auto_pitch_type", "pitch_type_group"]
# 투구 **전에** 알 수 있는 것 — 섞으면 "빠진 피처"를 privileged 로 오독한다
XSTAR_PRE = ["pitch_of_pa", "pitch_no"]
CATCOL = ["tagged_pitch_type", "auto_pitch_type", "pitch_type_group"]


def with_star(Xbase, S, cols):
    """x* 컬럼을 **위치 기반**으로 붙인다.

    ⚠️ `pd.concat(axis=1)` 을 쓰면 안 된다 — `build_features` 는 새 RangeIndex 를
    돌려주고 `XS[mask]` 는 원본 인덱스를 유지하므로 인덱스 정렬이 합집합을 만든다
    (LightGBM 이 "Length of labels differs" 로 잡았다).
    """
    assert len(Xbase) == len(S), (len(Xbase), len(S))
    out = Xbase.copy()
    v = S[cols].to_numpy(dtype=np.float32)
    for i, c in enumerate(cols):
        out[c] = v[:, i]
    return out


def fit_arm(Xtr, ytr, w, Xva, tag, t0):
    p = dict(FINAL_LGB_PARAMS, seed=SEED, bagging_seed=SEED,
             feature_fraction_seed=SEED, data_random_seed=SEED)
    bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=N_ROUNDS)
    z = bst.predict(Xva, raw_score=True)
    print(f"    {tag:<14s} 피처 {Xtr.shape[1]:>3d}  ({time.time()-t0:.0f}s)")
    del bst
    return z


def paired_boot(y, p0, p1, cl, n_cl, rng, B=B_BOOT):
    """경기 클러스터 쌍체 부트스트랩 ΔBSS."""
    ybar = y.mean()
    ref = ybar * (1 - ybar)
    e0, e1 = (p0 - y) ** 2, (p1 - y) ** 2
    n = np.bincount(cl, minlength=n_cl)
    s0 = np.bincount(cl, weights=e0, minlength=n_cl)
    s1 = np.bincount(cl, weights=e1, minlength=n_cl)
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
    print("=== 로드 ===")
    df, s, y = D.load_df()
    jn = np.load(CACHE / "tmrow_join.npz")
    tm_idx, trust = jn["tm_idx"], jn["trust"]
    print(f"  train {len(df):,}  신뢰 조인 {int(trust.sum()):,} ({trust.mean():.1%})")

    tcols = sorted(set(XSTAR_POST + XSTAR_PRE))
    t = TM.load_tm(extra_cols=["balls_before", "strikes_before", "outs_before"]
                   + [c for c in tcols if c not in TM.TM_COLS])
    t = t.reset_index(drop=True)
    for c in CATCOL:
        t[c] = pd.Categorical(t[c].astype(str)).codes.astype(np.float32)

    # 신뢰 행에만 x* 를 붙인다. 나머지는 NaN (LightGBM 이 결측으로 처리)
    XS = pd.DataFrame(index=df.index, columns=tcols, dtype=np.float32)
    ok = trust & (tm_idx >= 0)
    XS.loc[ok, tcols] = t.loc[tm_idx[ok], tcols].to_numpy(dtype=np.float32)

    gt = df["game_type"].to_numpy()
    clean = ~((gt == "F") & (s < 2023))
    edges = np.quantile(df.loc[s < EDGE_SEASON, "asof_pitcher_n"].to_numpy(float),
                        [.2, .4, .6, .8])
    apn = df["asof_pitcher_n"].to_numpy(dtype=np.float64)
    # 경기 경계 (부트스트랩 클러스터) — §4 규칙
    pair = (np.minimum(df.pitcher_team_id, df.batter_team_id).astype(np.int64) * 100
            + np.maximum(df.pitcher_team_id, df.batter_team_id))
    key = df.season.astype(np.int64) * 10000 + df.game_month * 100 + df.game_dayofweek
    half = df.inning.astype(np.int64) * 2 + (df.top_bottom == "B")
    game = (((pair != pair.shift()) | (key != key.shift())
             | (half < half.shift())).cumsum()).to_numpy()

    rng = np.random.default_rng(20260824)
    for fold in D.FOLDS:
        print(f"\n{'='*100}\nfold{fold} 라인   teacher 창 = 시즌 ≤ {fold-1}\n{'='*100}")
        mtr = (s < fold) & clean & trust           # 두 팔 동일 행 집합
        mva = (s == fold) & clean & trust
        tr, va = df[mtr], df[mva]
        print(f"  teacher 학습 {len(tr):,}행 · 평가 {len(va):,}행"
              f"  (평가 시즌 전체 {int(((s == fold) & clean).sum()):,} 중 "
              f"{len(va)/max(int(((s == fold) & clean).sum()),1):.1%})")

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
        Str, Sva = XS[mtr], XS[mva]

        arms = {}
        arms["A x"] = fit_arm(Xtr, ytr, w, Xva, "A  x", t0)
        allc = list(Str.columns)
        arms["B x+all"] = fit_arm(with_star(Xtr, Str, allc), ytr, w,
                                  with_star(Xva, Sva, allc), "B  x+x*_all", t0)
        arms["C x+post"] = fit_arm(with_star(Xtr, Str, XSTAR_POST), ytr, w,
                                   with_star(Xva, Sva, XSTAR_POST), "C  x+x*_post", t0)

        ybar = yva.mean()
        P = {k: shift_to_prior(1 / (1 + np.exp(-z)), ybar) for k, z in arms.items()}

        # 참고 — 같은 행에서의 배포 스택 (선택편향 점검용)
        st = D.state(df, s, y, fold)
        mm = st["mm"]
        ax = st["Xax"] @ st["dep"]
        _, zb = build_layers(df, s, fold, mm)[-1]
        pb = shift_to_prior(1 / (1 + np.exp(-(zb + ax))), st["yv"].mean())
        pdep_all = shift_to_prior(PROB_CENTER + PROB_SCALE * (pb - PROB_CENTER),
                                  st["yv"].mean())
        sel = trust[st["vi"][mm]]
        print(f"\n  [참고] 배포 스택 BSS  전체 {bss(st['yv'], pdep_all):.1f}"
              f"  ·  신뢰 행만 {bss(st['yv'][sel], pdep_all[sel]):.1f}"
              f"  ·  비신뢰 행 {bss(st['yv'][~sel], pdep_all[~sel]):.1f}"
              f"   ← 선택편향 점검")
        for k, p_ in P.items():
            print(f"  [참고] teacher {k:<10s} BSS {bss(yva, p_):8.1f}")

        isf = va["game_type"].to_numpy().astype(str) == "F"
        cold = apn[mva] < edges[0]
        cl, n_cl = pd.factorize(game[mva])[0], int(pd.factorize(game[mva])[1].size)
        print(f"\n  {'구획':<12s}{'행':>9s}{'팔':<10s}{'쌍체 gap':>10s}"
              f"{'2.5%':>9s}{'97.5%':>9s}{'P>0':>7s}{'sd(Δp)':>9s}")
        for lab, m in (("전체", np.ones(len(yva), bool)), ("R", ~isf), ("F", isf),
                       ("cold-start", cold)):
            if m.sum() < 500:
                continue
            for arm in ("B x+all", "C x+post"):
                d, lo, hi, pg = paired_boot(yva[m], P["A x"][m], P[arm][m],
                                            cl[m], n_cl, rng)
                sd = float(np.std(P[arm][m] - P["A x"][m]))
                note = ""
                if lab == "F" and fold == 2023:
                    note = "  ⚠️사전고정: 창에 깨끗한 F 없음 — 신뢰 안 함"
                print(f"  {lab if arm=='B x+all' else '':<12s}"
                      f"{m.sum() if arm=='B x+all' else '':>9}"
                      f"{arm:<10s}{d:>10.1f}{lo:>9.1f}{hi:>9.1f}{pg:>7.2f}"
                      f"{sd:>9.4f}{note}")


if __name__ == "__main__":
    main()
