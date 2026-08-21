"""§7.107 — 중심화는 **타깃 쪽 탈레벨(init_score)과 짝지어야** 하는가 (§7.106 ④의 예측).

§7.106 기전 가설: 원시 `asof_*` 레벨이 두 가지를 겸하고 있었다.
    (a) 그 투수의 상대적 실력        ← 중심화가 살리려던 것
    (b) **암묵적 시대 지시자**        ← 중심화가 지운 것
`season`이 피처가 아니라 (b)가 모델의 유일한 시대 핸들이었고, 지우니 −13.6 이 났다.

→ 예측: `init_score`로 (시즌×game_type) 레벨을 **타깃 쪽에서** 빼서 (b)의 필요를
  없애면, 중심화의 부호가 바뀌거나 최소한 손실이 크게 줄어야 한다.

사용자 질문 "그러면 다시 시즌을 넣어보면 되는 거 아님?" 에 대한 답이기도 하다 —
`season`을 **피처로** 넣으면 2025가 학습에 없던 값이라 트리가 외삽을 못 하고
`season≥2024` 잎으로 떨어져 2024 레벨을 2025에 그대로 쏜다 (§2의 `game_type` 제거가
+172였던 것과 같은 함정). `init_score`는 같은 정보를 **외삽 가능한 형태**로 넣는다.

설계 2×2   init_score {off, on} × center_asof {off, on},  fold {2023, 2024}
  ★ **하네스 검증**: init=off 열의 중심화 Δ가 §7.106의 −12.3 / −15.0 (lgb 층)을
    재현해야 한다. 재현 안 되면 init=on 열의 숫자도 못 믿는다.

⚠️ 소프트 라벨은 **켠다.** 처음엔 껐다가 fold2023 하네스가 −1255 로 깨졌다
   (학습셋 2019~2022 가 전부 F 오염 구간이라 오염 처리 없이는 fold 자체가 무너진다).
⚠️ 이 실험은 **LGBM 단독**이다 (§7.106에서 손실이 lgb −13.6 / mlp −3.9 로 트리에
   몰렸고, fold당 20분이 6분이 된다). 소프트 라벨·보조 헤드는 끄므로 절대 BSS는
   배포 스택과 다르다 — **팔 간 Δ만 읽는다.**
⚠️ `jointbeta`는 import하지 않는다 (foldcache._key 를 못 박는다 — §7.106 ⑤).

실행:  .venv/Scripts/python.exe -u src/asofctr3.py 2>&1 | tee experiments/asofctr3.log
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

import soften
from config import (CONTAM_AXES, DROP_FEATURES, ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS,
                    N_ROUNDS, RECENCY_HALF_LIFE, SEED_ENSEMBLE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, SOFT_LABEL_F, TARGET_COL, TRAIN_CSV,
                    USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
LGB_SEEDS = SEED_ENSEMBLE[:3]
EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def level_table(tr, y):
    """(시즌, game_type) → 양성률. + gt별 **다음 시즌 외삽** 규칙.

    외삽은 연차분 평균 1스텝 — `TARGET_PRIOR_2025`·`fit_asof_center`와 같은 규칙이다.
    검증 시즌은 항상 표 밖이므로 **이 경로가 2025와 같은 조건에서 평가된다.**
    """
    ss = tr["season"].to_numpy(dtype=np.int64)
    gg = tr["game_type"].astype("object").to_numpy().astype(str)
    tab, fb = {}, {}
    for gtv in np.unique(gg):
        pts = []
        for yy in sorted(set(ss[gg == gtv].tolist())):
            m = (ss == yy) & (gg == gtv)
            if m.sum() >= 500:
                tab[(int(yy), gtv)] = float(y[m].mean())
                pts.append((int(yy), float(y[m].mean())))
        if len(pts) >= 2:
            fb[gtv] = (pts[-1][0], pts[-1][1], float(np.mean(np.diff([p[1] for p in pts]))))
        elif pts:
            fb[gtv] = (pts[-1][0], pts[-1][1], 0.0)
    return tab, fb


def level_lookup(d, tab, fb, gmean):
    ss = d["season"].to_numpy(dtype=np.int64)
    gg = d["game_type"].astype("object").to_numpy().astype(str)
    ug = np.unique(gg)
    k = ss * len(ug) + np.searchsorted(ug, gg)
    out = np.full(len(k), gmean, dtype=np.float64)
    for kk in np.unique(k):
        yy, c = int(kk // len(ug)), int(kk % len(ug))
        gtv = str(ug[c])
        v = tab.get((yy, gtv))
        if v is None:
            f = fb.get(gtv)
            v = (f[1] + f[2] * (yy - f[0])) if f else gmean
        out[k == kk] = v
    return np.clip(out, 0.02, 0.98)


def run(df, fold, center, init, t0):
    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    y_va = va[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                        center_asof=center)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xtr, Xva = X(tr), X(va)

    if init:
        tab, fb = level_table(tr, y_tr)
        init_tr = _logit(level_lookup(tr, tab, fb, float(y_tr.mean())))
        base_va = _logit(level_lookup(va, tab, fb, float(y_tr.mean())))
    else:
        init_tr, base_va = None, 0.0

    # ── 소프트 라벨 (§7.106 하네스 검증 실패의 원인) ────────────────────
    # 처음엔 껐다가 fold2023 이 620.2 → 131.3 으로 내려앉았다. fold2023 의 학습셋은
    # 2019~2022 = **전부 F 오염 구간**이라 오염 처리를 빼면 통째로 무너진다.
    # soften 은 복제 행을 **끝에 덧붙이므로** init_score 도 같은 마스크로 늘리면
    # 정렬이 유지된다 (soften.py:62-66 과 같은 dup 정의를 쓴다).
    if SOFT_LABEL_F:
        h = soften.fit_h(tr)
        s_ = tr["season"].to_numpy()
        g_ = tr["game_type"].to_numpy().astype(str)
        h_row = np.zeros(len(tr))
        for yr, v in h.items():
            h_row[(s_ == yr) & (g_ == "F")] = v
        dup = (h_row > 0) & (y_tr == 1)
        n0 = len(Xtr)
        Xtr, y_tr, w, _, _ = soften.soften(tr, Xtr, y_tr, w)
        if init_tr is not None:
            init_tr = np.concatenate([init_tr, init_tr[dup]])
        assert len(Xtr) == len(y_tr) == len(w) == (
            len(init_tr) if init_tr is not None else len(Xtr)), "soften 정렬 깨짐"
        print(f"      soft label: {n0:,} → {len(Xtr):,}행")

    z = np.zeros(len(Xva))
    n = 0
    for over in ENSEMBLE_VARIANTS.values():
        for sd in LGB_SEEDS:
            p = dict(FINAL_LGB_PARAMS, **over, seed=sd, bagging_seed=sd,
                     feature_fraction_seed=sd, data_random_seed=sd)
            ds = lgb.Dataset(Xtr, y_tr, weight=w, init_score=init_tr)
            bst = lgb.train(p, ds, num_boost_round=N_ROUNDS)
            z += bst.predict(Xva, raw_score=True)
            n += 1
            del bst
    z = z / n + base_va          # init_score를 썼으면 기준 로짓을 되돌려 준다

    gt = va["game_type"].to_numpy()
    sv = va["season"].to_numpy()
    keep = ~((gt == "F") & (sv < 2023))          # clean 행 (§7.101)
    pv = shift_to_prior(1 / (1 + np.exp(-z[keep])), y_va[keep].mean())
    sc = bss(y_va[keep], pv)
    print(f"    fold {fold}  center={int(center)} init={int(init)}  "
          f"BSS {sc:8.1f}   ({time.time()-t0:.0f}s)")
    return sc


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    print(f"train {df.shape}   LGBM 단독 ({len(ENSEMBLE_VARIANTS)}변형 × {len(LGB_SEEDS)}시드)")
    print(f"소프트 라벨 {'ON' if SOFT_LABEL_F else 'OFF'} · MLP/보조헤드 없음"
          " → 절대 BSS는 배포와 다르다. **팔 간 Δ만 읽는다.**\n")
    R = {}
    for fold in FOLDS:
        for init in (False, True):
            for center in (False, True):
                R[(fold, init, center)] = run(df, fold, center, init, t0)

    print(f"\n{'='*76}\n█ 판정 — 중심화 Δ (center=1 − center=0), clean 행 BSS\n{'='*76}")
    print(f"  {'fold':<8}{'init':<7}{'center=0':>11}{'center=1':>11}{'Δ(중심화)':>12}")
    for fold in FOLDS:
        for init in (False, True):
            a, b = R[(fold, init, False)], R[(fold, init, True)]
            print(f"  {fold:<8}{int(init):<7}{a:>11.1f}{b:>11.1f}{b-a:>+12.1f}")
    print(f"\n  {'':8}{'init 효과 (center=0)':<24}"
          + "  ".join(f"{f}:{R[(f,True,False)]-R[(f,False,False)]:+.1f}" for f in FOLDS))
    d0 = [R[(f, False, True)] - R[(f, False, False)] for f in FOLDS]
    d1 = [R[(f, True, True)] - R[(f, True, False)] for f in FOLDS]
    print(f"\n  ★ 하네스 검증 — init=0 의 중심화 Δ: "
          + "  ".join(f"{f}:{v:+.1f}" for f, v in zip(FOLDS, d0)))
    print(f"     §7.106 lgb 층 실측:            2023:-12.3  2024:-15.0")
    print(f"     → 같은 자릿수·같은 부호면 하네스 통과. 아니면 아래 숫자도 못 믿는다.\n")
    print(f"  ★ 가설 검정 — init=1 의 중심화 Δ: "
          + "  ".join(f"{f}:{v:+.1f}" for f, v in zip(FOLDS, d1)))
    print(f"     부호가 바뀌거나 손실이 크게 줄면 §7.106 ④의 기전 가설이 지지된다.")
    print(f"     그대로면 기전 가설은 틀렸고, 중심화는 시대 지시자 때문이 아니다.")


if __name__ == "__main__":
    main()
