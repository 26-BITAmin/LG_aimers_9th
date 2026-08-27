"""§7.179 — soft label h 의 행 단위화 ("noise swap"): R 에서 숨은-reverse 조건부를 배워 pre-2023 F 에 재배분

사용자: *"2023 이전 F 이상한 데이터 문제 있는 부분만 R 하고 noise swap 해서
        R 에서 그거를 학습하고, F 에서 나중에 적용하는 거는 어때"*

무엇을 재나
    배포 soften(§7.70)은 pre-2023 F·y=1 행에 **시즌 상수** h_s 를 건다. 제안은
    그 상수를 **행 단위 h(x)** 로 바꾸는 것 — R 행(reverse 기준이 늘 적용돼 라벨
    무결)에서
        r(x) = P(reverse-only | y=1 ∪ reverse-only, x)
    를 학습해 F 의 오염 행에 적용한다. 숨은 reverse = "reverse 기준에만 걸렸을"
    행이므로 조건부의 분모는 (성공 ∪ reverse-only) 가 맞다 (몰림+반대 행은
    F 에서도 몰림 기준이 잡아 y=0 으로 기록됐다).
    **시즌별 총질량은 보존한다** (h̄ = h_s). §7.161 이 크기 축은 이미 최적이라
    답했으므로 여기서는 재배분 축만 연다.

닫힌 문과의 관계 (§7.70 1단계)
    투수별 세분화 h_i = h_s·r_i/r̄ 가 f2023 −3.6±2.3 부호 불일치로 기각됐고
    "전용 reverse 모델은 같은 축" 이라며 2단계를 열지 않았다. 다시 여는 근거:
    ① r_i 는 **투수 정체** 축이었다. §7.40 실측 — reverse 는 **카운트** 축으로
       1.7배 움직인다 (0-2 반대 .162 vs 3-1 .276). 상황 축 재배분은 잰 적 없다.
    ② 당시에도 f2024 는 **+4.0±1.4 양수**였다. 기각한 fold2023 은 학습 F 가
       100% 오염 구간 + recency 만점이라 재배분 오차가 극대화되는 fold 다.
    ③ 메모리 [[ask-if-the-correction-constant-must-vary-per-entity]] ·
       [[revisit-closed-diagnoses-with-new-tools]].

팔 (전부 코어 LGBM 20 · s41 SMOTE 증강 동일 · rng 777 → 합성행 동일 → 쌍체)
    base  s41 등가 (h 시즌 상수)                          ← t1nrk_base.npz 재사용
    TBL   h_i ∝ (balls,strikes) 테이블 r — 12셀. 파라미터가 적어 전이가 안전한 축
          ([[transfer-ratio-depends-on-when-value-is-fixed]])
    MDL   h_i ∝ LGBM r(x) (코어 78피처 그대로)
    PLB   MDL 의 h_i 를 시즌 안에서 셔플 [위약 — 질량·분산 보존, 정체만 삭제.
          [[placebo-must-keep-shared-terms]] · 라벨 분산 팽창 채널 통제]

사전 등록 판정 (⚠️ fold2024 하나뿐 — 최종 판정자는 LB 슬롯)
    후보 = Δ vs base 양수 · P(Δ>0) ≥ 0.9 · (Δ_후보 − Δ_PLB) 쌍체 양수.
    PLB 가 후보만큼 오르면 이득은 재배분이 아니라 노이즈/분산 팽창이다.

배관 검증 (사전)
    ⓐ 상수 h 로 내 경로를 돌리면 t1core.build(soften 경로)와 y·w·init 이 일치
    ⓑ 시즌별 h̄ 가 clip 후에도 h_s 와 1e-3 이내
    ⓒ dup 행 수가 base 와 동일 (재배분은 행 집합을 못 바꾼다)
    ⓓ 각 팔 vs base 최대|Δp| > 0

실행:  .venv/Scripts/python.exe -u src/t1swp.py 2>&1 | tee experiments/t1swp.log
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
T1 = ROOT / "t1work"
ASCII = Path("D:/t1eval")
sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

import soften
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, N_ROUNDS,
                    RECENCY_HALF_LIFE, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from hbeta2 import season_gt_level
from modelabel import recover
from smotenc import n_needed, smote
from t1core import RATIO, SEEDS, VARIANTS, WORKER, build as t1core_build, t1_features
from validation import bss, shift_to_prior

VALID = 2024
ARMS = ("TBL", "MDL", "PLB")
H_CAP = soften.H_CAP

R_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                min_data_in_leaf=100, verbose=-1, seed=42)


def fit_r(tr, X):
    """R 행에서 r(x) 를 학습하고, 오염 행(pre-2023 F·y=1)의 r 예측 두 벌을 돌려준다."""
    lab, _, _ = recover(tr)
    ok = ~np.isnan(lab["reverse"]) & ~np.isnan(lab["middle"])
    rev = np.where(ok, np.round(lab["reverse"]), 0.0)
    mid = np.where(ok, np.round(lab["middle"]), 0.0)
    y = tr[TARGET_COL].to_numpy(dtype=np.float64)
    s = tr["season"].to_numpy(dtype=np.int64)
    g = tr["game_type"].astype("object").to_numpy().astype(str)

    revonly = (y == 0) & (rev == 1) & (mid == 0)
    pop = ok & (g == "R") & ((y == 1) | revonly)     # 분모 = 성공 ∪ reverse-only
    t = revonly[pop].astype(np.float64)
    aff = (s <= 2022) & (g == "F") & (y == 1)        # 적용 대상 = 오염 y=1 행
    print(f"  r 학습 풀(R) {int(pop.sum()):,}행 · 양성률 {t.mean():.4f} · "
          f"적용 대상(pre-2023 F·y=1) {int(aff.sum()):,}행")

    # ── MDL: 코어 피처 그대로 작은 LGBM
    b = lgb.train(R_PARAMS, lgb.Dataset(X[pop], t), num_boost_round=400)
    r_mdl = np.asarray(b.predict(X[aff]), dtype=np.float64)

    # ── TBL: (balls,strikes) 12셀 테이블
    bl = tr["balls_before"].to_numpy(dtype=np.int64)
    st = tr["strikes_before"].to_numpy(dtype=np.int64)
    cell = bl * 10 + st
    tab = pd.Series(t, index=cell[pop]).groupby(level=0).mean()
    r_tbl = pd.Series(cell[aff]).map(tab).fillna(t.mean()).to_numpy(dtype=np.float64)
    print(f"  카운트 테이블 r: {tab.min():.3f}~{tab.max():.3f} ({tab.max()/tab.min():.2f}배) · "
          f"corr(TBL, MDL)={np.corrcoef(r_tbl, r_mdl)[0, 1]:.3f}")
    return aff, {"TBL": r_tbl, "MDL": r_mdl}


def make_h(tr, aff, r_aff):
    """행 단위 h. 시즌별 총질량 보존 → clip → 재정규화 3회."""
    h_const = soften.fit_h(tr)
    s = tr["season"].to_numpy(dtype=np.int64)
    h_row = np.zeros(len(tr))
    idx_all = np.flatnonzero(aff)
    for yr, hs in h_const.items():
        m = s[idx_all] == yr
        v = np.maximum(r_aff[m], 1e-6)
        for _ in range(3):                            # clip 이 깎은 질량을 되채운다
            v = np.clip(v * (hs / v.mean()), 1e-6, H_CAP)
        assert abs(v.mean() - hs) < 1e-3, (yr, v.mean(), hs)
        h_row[idx_all[m]] = v
    return h_row, h_const


def build_rowh(tr, X, w, h_row, rng):
    """t1core.build 와 동일 경로 — soften 상수 h 만 h_row 로 대체. s41 SMOTE 포함."""
    y = tr[TARGET_COL].to_numpy(dtype=np.float64)
    s = tr["season"].to_numpy(dtype=np.int64)
    g = tr["game_type"].astype("object").to_numpy().astype(str)
    dup = (h_row > 0) & (y == 1)

    X1 = pd.concat([X, X[dup]], ignore_index=True)
    y1 = np.concatenate([y, np.zeros(int(dup.sum()))])
    w1 = np.asarray(w, dtype=np.float64).copy()
    w1[dup] *= (1.0 - h_row[dup])
    w1 = np.concatenate([w1, np.asarray(w, np.float64)[dup] * h_row[dup]])
    ss, gt = np.concatenate([s, s[dup]]), np.concatenate([g, g[dup]])

    m = (s <= 2022) & (g == "F") & (y == 0)
    k = n_needed(int(((s <= 2022) & (g == "F") & (y == 1)).sum()), int(m.sum()), RATIO)
    new, sd = smote(X[m].to_numpy(dtype=np.float64), k, rng)
    src = np.flatnonzero(m)[sd]
    X1 = pd.concat([X1, pd.DataFrame(new.astype(np.float32), columns=X.columns)],
                   ignore_index=True)
    y1 = np.concatenate([y1, np.zeros(k)])
    w1 = np.concatenate([w1, np.asarray(w, np.float64)[src]])
    ss, gt = np.concatenate([ss, s[src]]), np.concatenate([gt, g[src]])
    return X1, y1, w1, season_gt_level(ss, gt, y1, w1), int(dup.sum())


def main():
    t0 = time.time()
    art = joblib.load(T1 / "model" / "artifacts.joblib")["artifacts"]
    cols = lgb.Booster(model_str=(T1 / "model" / "lgbm_base_42.txt")
                       .read_text(encoding="utf-8")).feature_name()
    df = load_csv(TRAIN_CSV)
    tr = df[df.season < VALID].reset_index(drop=True)
    X = t1_features(tr, art, cols)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    print(f"피처 {X.shape}  ({time.time()-t0:.0f}s)")

    aff, R = fit_r(tr, X)
    s = tr["season"].to_numpy(dtype=np.int64)

    H, h_const = {}, None
    for a in ("TBL", "MDL"):
        H[a], h_const = make_h(tr, aff, R[a])
    rng_p = np.random.default_rng(12345)              # PLB: MDL h 를 시즌 안 셔플
    hp = H["MDL"].copy()
    for yr in h_const:
        ix = np.flatnonzero(aff & (s == yr))
        hp[ix] = hp[ix[rng_p.permutation(len(ix))]]
    H["PLB"] = hp
    for a in ARMS:
        v = H[a][aff]
        print(f"  h[{a}]  mean {v.mean():.4f}  std {v.std():.4f}  "
              f"q05 {np.quantile(v, .05):.4f}  q95 {np.quantile(v, .95):.4f}")

    # ── 배관 ⓐ: 상수 h 로 내 경로 == t1core.build(soften 경로)
    hc_row = np.zeros(len(tr))
    g_ = tr["game_type"].astype("object").to_numpy().astype(str)
    for yr, v in h_const.items():
        hc_row[(s == yr) & (g_ == "F")] = v
    _, y1a, w1a, ini_a, nd = build_rowh(tr, X, w, hc_row, np.random.default_rng(777))
    _, y1b, w1b, ini_b, _ = t1core_build(tr, X, w, True, np.random.default_rng(777))
    assert np.array_equal(y1a, y1b) and np.allclose(w1a, w1b) and np.allclose(ini_a, ini_b), \
        "배관 ⓐ 실패 — 상수 h 등가가 깨졌다"
    print(f"  배관 ⓐ OK — 상수 h 에서 soften 경로와 y·w·init 일치 (dup {nd:,})  "
          f"({time.time()-t0:.0f}s)")
    del y1a, w1a, ini_a, y1b, w1b, ini_b

    for a in ARMS:
        dst = ASCII / f"SWP_{a}"
        if (dst / "model" / f"lgbm_lowff_{SEEDS[-1]}.txt").exists():
            print(f"  {a} 재사용"); continue
        shutil.copytree(T1, dst, dirs_exist_ok=True)
        X2, y2, w2, init, nd2 = build_rowh(tr, X, w, H[a], np.random.default_rng(777))
        assert nd2 == nd, f"배관 ⓒ 실패 — dup {nd2} != base {nd}"
        print(f"  [{a}] 행 {len(X):,} → {len(X2):,}")
        for v in VARIANTS:
            for sd_ in SEEDS:
                p = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v], seed=sd_,
                         bagging_seed=sd_, feature_fraction_seed=sd_, data_random_seed=sd_)
                b = lgb.train(p, lgb.Dataset(X2, y2, weight=w2, init_score=init),
                              num_boost_round=N_ROUNDS)
                # ⚠️ LF — write_text 의 CRLF 가 model_file= 을 죽인다 (§7.173 ③)
                (dst / "model" / f"lgbm_{v}_{sd_}.txt").write_text(
                    b.model_to_string(), encoding="utf-8", newline="\n")
                del b
            print(f"    {v} ({time.time()-t0:.0f}s)")
        del X2, y2, w2, init
    del X, tr

    wk = ROOT / "experiments" / "cache" / "_t1swp_worker.py"
    wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/")),
                  encoding="utf-8", newline="\n")
    P = {}
    for a, dname in [("base", "D_aug")] + [(x, f"SWP_{x}") for x in ARMS]:
        f = ROOT / "experiments" / "cache" / ("t1nrk_base.npz" if a == "base"
                                              else f"t1swp_{a}.npz")
        if not f.exists():
            r = subprocess.run([sys.executable, str(wk), dname, str(f)],
                               capture_output=True, text=True)
            if not f.exists():
                print(f"  ❌ {a}: {r.stderr.strip()[-300:]}"); return
        P[a] = np.load(f)["p"]
    print("  배관 ⓓ OK — base 대비 최대|Δp|: " +
          " ".join(f"{a} {np.abs(P[a]-P['base']).max():.5f}" for a in ARMS))
    for a in ARMS:
        assert np.abs(P[a] - P["base"]).max() > 0, f"{a} 가 base 와 같다"

    va = df[df.season == VALID].reset_index(drop=True)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    isF = va["game_type"].to_numpy().astype(str) == "F"
    codes, uniq = pd.factorize(va["pitcher_id"].to_numpy())
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    seen = np.where(np.diff(bnd) > 0)[0]
    rng = np.random.default_rng(42)
    boots = [np.concatenate([order[bnd[j]:bnd[j+1]] for j in
                             rng.choice(seen, size=len(seen), replace=True)])
             for _ in range(400)]

    def s_(p, m=None, idx=None):
        pp, yy = (p, yva) if m is None else (p[m], yva[m])
        if idx is not None:
            pp, yy = pp[idx], yy[idx]
        return bss(yy, shift_to_prior(pp, yy.mean()))

    print("\n" + "=" * 92)
    print("soft label h 행 단위화 (noise swap) · 코어 LGBM 20 · fold2024 · ⚠️ fold 하나뿐")
    print("=" * 92)
    print("  %-34s %9s %10s %22s %8s" % ("팔", "BSS", "Δ vs s41", "95% 구간", "P(Δ>0)"))
    print("  %-34s %9.1f" % ("base = s41 (h 시즌 상수)", s_(P["base"])))
    D = {}
    desc = {"TBL": "TBL  h∝카운트 테이블 (12셀)",
            "MDL": "MDL  h∝LGBM r(x) (78피처)",
            "PLB": "PLB  MDL 시즌내 셔플 [위약]"}
    for a in ARMS:
        D[a] = np.array([s_(P[a], idx=ix) - s_(P["base"], idx=ix) for ix in boots])
        lo, hi = np.percentile(D[a], [2.5, 97.5])
        print("  %-34s %9.1f %+10.2f   [%+7.2f, %+7.2f] %8.3f"
              % (desc[a], s_(P[a]), s_(P[a]) - s_(P["base"]), lo, hi, (D[a] > 0).mean()))
    print("\n  위약 차감 (쌍체, 같은 부트스트랩 draw)")
    for a in ("TBL", "MDL"):
        d = D[a] - D["PLB"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print("    Δ(%s−PLB) %+7.2f   [%+7.2f, %+7.2f]   P(>0)=%.3f"
              % (a, (s_(P[a]) - s_(P["PLB"])), lo, hi, (d > 0).mean()))
    print("\n  F행 / R행 분해")
    for a in ["base"] + list(ARMS):
        print("    %-6s F %8.1f   R %8.1f" % (a, s_(P[a], isF), s_(P[a], ~isF)))
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
