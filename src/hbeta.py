"""soft label 크기 h_s 의 **스칼라 배율 β** 스윕 (§7.161) — §7.70 이 안 연 문

배경
    §7.70 은 h_s = q_s/p̄_s 를 정하자마자 투수별 세분화(부호 불일치로 기각)로
    갔고, **크기 자체는 한 번도 스윕되지 않았다.** q_s = (R 반대율)−(F 반대율)
    은 "F 의 진짜 reverse 성향 = R 의 성향" 을 가정한다. 2023+ 실측은
    F(.300/.288) > R(.250/.240) 이라 그 가정이 h 를 **과소추정**할 수 있다.

사전분포 (증거 아님)
    F 도 R 처럼 연 −.012 드리프트했다고 보고 각 시즌을 추세선에 앉히는 배율:
    2019 2.626(질량 1.0%) · 2020 1.220(3.7%) · 2021 1.210(16.7%) · 2022 1.063(78.5%)
    → 가중평균 β ≈ 1.11.  스윕 범위를 [0.8, 2.0] 으로 잡는 근거일 뿐이다.

팔
    lin    β·h 를 [0,.95] 로 clip.  β=0 은 소프트 라벨 OFF (= §7.70 의 base).
    logit  σ(β·logit h).  h<.5 라 **β>1 이 h 를 줄인다** — lin 과 방향이 반대다.
           그래서 보고는 β 가 아니라 **결과 h(가중평균)** 를 공통 축으로 한다.
    target 깨끗한 2023 F 를 앵커로 추세 외삽. fold2023 은 앵커가 없어 **불가**.
    PLACEBO-R  같은 h 를 pre-2023 **R** 의 y=1 행에 건다 (시즌별 행 수 매칭).
           β>1 은 라벨 노이즈를 늘리므로 그 자체가 정규화다. 위약이 같은
           곡선을 그리면 이득은 "오염 보정" 이 아니라 **일반 스무딩**이다.
           기하 일치: 같은 시즌 분포 · 같은 행 수 · 같은 recency · 같은 y=1 제한.

기준 팔
    Δ 는 **β=1.0 (현 배포)** 대비로 읽는다. 판정은 "1.0 에서 옮길 이유가 있나".
    β=0 도 같이 실어 §7.70 의 +17.7/+494 가 재현되는지 본다 (하네스 검증).

⚠️ 이 하네스는 §7.70 과 같은 **단일 LGBM** 이다 — init_score·MLP·fspec 없음.
   §7.70 은 단독 +17.7 이 전체 스택에서 +39.8 로 커졌다. 절대 크기가 아니라
   **β 축의 모양**을 보는 것이 목적이고, init_score 와의 상호작용(§7.109 는
   소프트라벨과 init_score 가 대체재라 했다)은 2단계에서 따로 잰다.

실행:  .venv/Scripts/python.exe -u src/hbeta.py 2>&1 | tee experiments/hbeta.log
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

import soften
from config import (CONTAM_AXES, DROP_FEATURES, FINAL_LGB_PARAMS, N_ROUNDS,
                    RECENCY_HALF_LIFE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import bss, brier, shift_to_prior

FOLDS = [2024, 2023]           # 2024 먼저 — 2025 와 같은 레짐, 판정 fold
SEEDS = (42, 7, 2024)
PLACEBO_SEED = 909             # 위약 R 행 추첨 (고정)

#  (라벨, 모드, β)  — β=1.0 lin 이 현 배포. β=0 이 소프트 라벨 OFF.
#  1차 부분 실행(hbeta_run1_partial.log)에서 fold2024 정점이 h̄≈.22~.24 로 나왔다.
#  → 저구간(β 0.5·0.65, logit 1.5)을 넣어 정점을 가둔다. 확실히 죽은 위쪽
#    (lin 2.0, logit 0.6 = h̄ .36 으로 lin 1.2~1.4 와 중복)은 뺀다.
#  위약은 **정점 근처(0.8)와 먼 쪽(1.6)** 에 건다 — 승자를 직접 검정해야 한다.
ARMS = ([("lin b=0.0 [OFF]", "lin", 0.0)]
        + [("lin b=%.2f" % b, "lin", b) for b in (0.5, 0.65, 0.8, 1.0, 1.1, 1.2, 1.4, 1.6)]
        + [("logit b=%.1f" % b, "logit", b) for b in (0.8, 1.2, 1.5)]
        + [("target", "target", None)]
        + [("PLACEBO-R b=%.2f" % b, "placebo", b) for b in (0.8, 1.6)])
REF_ARM = "lin b=1.00"

CACHE = Path(__file__).parent.parent / "experiments" / "cache" / "hbeta.npz"


def prep_fold(df, valid_season):
    """모든 팔이 공유하는 것 — artifacts·설계행렬은 fold 당 한 번만 만든다."""
    tr = df[df.season < valid_season].reset_index(drop=True)
    va = df[df.season == valid_season]
    ref = int(tr.season.max())
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=ref)
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    return tr, va, w, X(tr), X(va)


def placebo_mask(tr, h, rng):
    """pre-2023 R 의 y=1 행 중 **시즌별로 F 와 같은 수**를 뽑는다 (기하 일치)."""
    s = tr["season"].to_numpy()
    g = tr["game_type"].to_numpy().astype(str)
    y = tr[TARGET_COL].to_numpy()
    m = np.zeros(len(tr), dtype=bool)
    for yr in h:
        n_f = int(((s == yr) & (g == "F") & (y == 1)).sum())
        cand = np.flatnonzero((s == yr) & (g == "R") & (y == 1))
        if len(cand) < n_f:
            raise ValueError("%d: R 후보 %d < F %d" % (yr, len(cand), n_f))
        m[rng.choice(cand, size=n_f, replace=False)] = True
    return m


def make_arm(tr, Xtr, w, h, mode, rng):
    """(X2, y2, w2, 복제수) — 복제 트릭. mode='placebo' 면 R 행에 건다."""
    y = tr[TARGET_COL].to_numpy(dtype=np.float64)
    s = tr["season"].to_numpy()
    g = tr["game_type"].to_numpy().astype(str)
    h_row = np.zeros(len(tr))
    if mode == "placebo":
        pm = placebo_mask(tr, h, rng)
        for yr, v in h.items():
            h_row[pm & (s == yr)] = v
    else:
        for yr, v in h.items():
            h_row[(s == yr) & (g == "F")] = v
    dup = (h_row > 0) & (y == 1)
    if not dup.any():
        return Xtr, y, np.asarray(w, dtype=np.float64), 0
    X2 = pd.concat([Xtr, Xtr[dup]], ignore_index=True)
    y2 = np.concatenate([y, np.zeros(int(dup.sum()))])
    w2 = np.asarray(w, dtype=np.float64).copy()
    w2[dup] *= (1 - h_row[dup])
    w2 = np.concatenate([w2, np.asarray(w)[dup] * h_row[dup]])
    return X2, y2, w2, int(dup.sum())


def train_eval(X2, y2, w2, Xva, yva, va_f):
    """3시드 → (BSS, F행 subBSS, R행 subBSS). 오라클 시프트 후 (§7.70 과 동일)."""
    refF = yva[va_f].mean() * (1 - yva[va_f].mean())
    refR = yva[~va_f].mean() * (1 - yva[~va_f].mean())
    out = []
    for sd in SEEDS:
        p = dict(FINAL_LGB_PARAMS, seed=sd, bagging_seed=sd,
                 feature_fraction_seed=sd, data_random_seed=sd)
        bst = lgb.train(p, lgb.Dataset(X2, y2, weight=w2), num_boost_round=N_ROUNDS)
        pr = shift_to_prior(1 / (1 + np.exp(-bst.predict(Xva, raw_score=True))),
                            yva.mean())
        out.append((bss(yva, pr),
                    1e5 * (1 - brier(yva[va_f], pr[va_f]) / refF),
                    1e5 * (1 - brier(yva[~va_f], pr[~va_f]) / refR)))
    return np.array(out)


def hbar(tr, h):
    """recency·행수 가중 평균 h — 모드가 달라도 비교 가능한 공통 축."""
    s = tr["season"].to_numpy()
    g = tr["game_type"].to_numpy().astype(str)
    y = tr[TARGET_COL].to_numpy()
    w = recency_weight(s, RECENCY_HALF_LIFE, ref_season=int(s.max()))
    num = den = 0.0
    for yr in range(2019, 2023):
        m = (s == yr) & (g == "F") & (y == 1)
        num += h.get(yr, 0.0) * w[m].sum()
        den += w[m].sum()
    return num / max(den, 1e-12)


def load_cache():
    """(fold, 입력지문) → 시드별 결과. 지문은 실제 (w2, y2) 해시라 팔 정의가
    바뀌면 자동으로 미스가 난다 — 낡은 수를 새 설정으로 읽는 사고를 막는다."""
    if not CACHE.exists():
        return {}
    z = np.load(CACHE, allow_pickle=True)
    return {tuple(k.split("|")): z[k] for k in z.files}


def save_cache(cache):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **{"|".join(k): v for k, v in cache.items()})


def main():
    df = load_csv(TRAIN_CSV)
    t0 = time.time()
    res, meta = {}, {}
    cache = load_cache()
    if cache:
        print("재개 캐시 %d칸 로드" % len(cache))

    for fold in FOLDS:
        print("\n" + "=" * 78)
        print("=== fold %d (train < %d) ===" % (fold, fold))
        print("=" * 78)
        tr, va, w, Xtr, Xva = prep_fold(df, fold)
        yva = va[TARGET_COL].to_numpy(dtype=np.float64)
        va_f = va["game_type"].to_numpy().astype(str) == "F"
        rng = np.random.default_rng(PLACEBO_SEED)
        fps = {}
        print("  준비 완료 (%.0fs)  train %s  valid %s  (F %s)"
              % (time.time() - t0, "{:,}".format(len(tr)),
                 "{:,}".format(len(va)), "{:,}".format(int(va_f.sum()))))

        for name, mode, beta in ARMS:
            if mode == "target" and fold == 2023:
                print("  %-18s SKIP — fold2023 은 2023+ F 앵커가 없다" % name)
                continue
            hm = "lin" if mode == "placebo" else mode
            h = soften.fit_h(tr, beta=beta, mode=hm)
            X2, y2, w2, nd = make_arm(tr, Xtr, w, h, mode, rng)

            # 캐시 지문 — 팔마다 실제로 갈렸는지 확인한다 (같은 수 두 번 읽기 방지)
            fp = hashlib.sha1(np.ascontiguousarray(w2).tobytes()
                              + np.ascontiguousarray(y2).tobytes()).hexdigest()[:8]
            assert fp not in fps, "%s 이 %s 과 같은 입력이다!" % (name, fps[fp])
            fps[fp] = name

            ck = (str(fold), fp)
            if ck in cache:
                r, tag = cache[ck], "cache"
            else:
                r = train_eval(X2, y2, w2, Xva, yva, va_f)
                cache[ck] = r
                save_cache(cache)          # 팔 하나 끝날 때마다 즉시 — 세션이 죽어도 산다
                tag = "%.0fs" % (time.time() - t0)
            res[(name, fold)] = r
            meta[(name, fold)] = (hbar(tr, h), nd, h)
            print("  %-18s h=%.4f 복제%7s  BSS %8.1f +-%5.1f  F행 %8.1f  R행 %7.1f  [%s] (%s)"
                  % (name, hbar(tr, h), "{:,}".format(nd), r[:, 0].mean(),
                     r[:, 0].std(ddof=1), r[:, 1].mean(), r[:, 2].mean(),
                     fp, tag))
        del tr, va, w, Xtr, Xva

    # ── 요약 ① 곡선 — **h̄ 오름차순**. 모드가 달라도 h̄ 이 같으면 같은 조작이다 ──
    print("\n" + "=" * 92)
    print("① 곡선 (h 오름차순) — Δ 는 %s (현 배포) 대비 쌍체, 3시드" % REF_ARM)
    print("=" * 92)
    print("  %-18s %7s %9s %9s %6s %9s %9s %6s  부호"
          % ("팔", "h", "BSS2024", "D2024", "+-SE", "BSS2023", "D2023", "+-SE"))
    order = sorted([a for a in ARMS if (a[0], 2024) in res or (a[0], 2023) in res],
                   key=lambda a: (a[1] == "placebo",
                                  meta.get((a[0], 2024), meta.get((a[0], 2023), (9,)))[0]))
    for name, mode, beta in order:
        hb = meta.get((name, 2024), meta.get((name, 2023), (float("nan"),)))[0]
        row = "  %-18s %7.4f" % (name, hb)
        ds = []
        for fold in FOLDS:
            if (name, fold) not in res:
                row += " %9s %9s %6s" % ("-", "-", "-")
                ds.append(None)
                continue
            d = res[(name, fold)][:, 0] - res[(REF_ARM, fold)][:, 0]
            row += " %9.1f %+9.1f %6.1f" % (res[(name, fold)][:, 0].mean(), d.mean(),
                                            d.std(ddof=1) / np.sqrt(len(d)))
            ds.append(d.mean())
        ok = "-" if (ds[0] is None or ds[1] is None or name == REF_ARM) else \
             ("O" if ds[0] * ds[1] > 0 else "X")
        print(row + "  %4s" % ok)

    # ── 요약 ② F행 / R행 분해 — 최적점이 갈리는 **기전**이 여기 있다 ──
    print("\n" + "=" * 92)
    print("② F행 / R행 분해 (fold2024) — 소프트 라벨은 F 를 사고 R 을 판다")
    print("=" * 92)
    print("  %-18s %7s %10s %10s %10s %10s"
          % ("팔", "h", "전체", "F행", "R행", "F-R"))
    for name, mode, beta in order:
        if (name, 2024) not in res:
            continue
        r = res[(name, 2024)]
        hb = meta[(name, 2024)][0]
        print("  %-18s %7.4f %10.1f %10.1f %10.1f %10.1f"
              % (name, hb, r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean(),
                 r[:, 1].mean() - r[:, 2].mean()))
    print("\n  총 %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
