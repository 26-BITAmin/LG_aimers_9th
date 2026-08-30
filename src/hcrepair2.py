"""§7.195d — 생성적 오염 수리: 행 유지 + 라벨 재구성 (GEN soft / HRD hard)

사용자: *"삭제하자는 게 아니라 regression 기반이든 모델 기반이든 생성 관점에서
해결해 보자."* — §7.195c 의 FIX(삭제)는 순위 안정성(ρ .33→.57)과 전이(−5.7→−2.2,
−1.7→+0.35)를 사면서 B1 을 −4.7/−8.9 에 팔았다. 행을 유지하며 오염만 도려내는
수리가 그 트레이드를 깰 수 있는지 실측한다.

팔 (LGBM 12멤버 × 2 fold 재학습, 나머지 경로는 hillclimb.train_lgb_members 동일)
    CUR   시즌 상수 soft label (배포 현행) — initdepm 캐시
    FIX   2019~22 F 삭제 (§7.195c) — initdepx 캐시
    GEN   행 단위 soft: h_s 를 r(x) 로 재배분, 시즌 질량 보존 (§7.179 MDL 동형)
    HRD   생성적 hard 재라벨: 시즌별 숨은 reverse 기대 개수 round(h_s·n₁) 만큼
          r(x) 상위 오염 y=1 행을 y=0 으로 뒤집는다 (행·가중 유지, dup 없음)

r(x) = P(reverse-only | 성공 ∪ reverse-only, x) — fold 학습창의 R 행에서 학습
(§7.179 fit_r recipe, 분모 근거 §7.179 참조). 피처는 멤버와 같은 행렬.

⚠️ §7.179 경고: 질량 보존 soft 재배분은 배포층 |Δp| ~ .002 였다. GEN 이 CUR 와
구분 불가로 나오면 그것이 답이다 — |Δz| 를 같이 보고한다.

사전 등록 판정 (§7.195c 의 ①②③ + B1, 4팔 비교)
    수리 성립 = B1 이 CUR ±1 안 (삭제형 비용 없음) & ρ 상승 & ③ 전이 개선.
    힐클라임 재개봉 후보 = 위에 더해 ③ 전이 Δ > 0 이고 절대 점수(B1+Δ)가
    CUR B1 이상. 이 마지막 조건이 §7.195c 에서 FIX 를 죽인 조건이다.

실행:  .venv/Scripts/python.exe -u src/hcrepair2.py 2>&1 | tee experiments/hcrepair2.log
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

import deploy29 as D
import hillclimb as H
from config import (CONTAM_AXES, DROP_FEATURES, ENSEMBLE_VARIANTS,
                    FINAL_LGB_PARAMS, N_ROUNDS, RECENCY_HALF_LIFE,
                    SEED_ENSEMBLE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    TARGET_COL, USE_PLAYER_IDS)
from hcrepair import disagreement, family_greedy, solo_bss, spearman

CACHE = D.CACHE
FOLDS = (2023, 2024)
LGB_SEEDS = SEED_ENSEMBLE[:3]
LGB_VARIANTS = ("base", "wide", "extra", "lowff")
H_CAP = 0.95
R_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                min_data_in_leaf=100, verbose=-1, seed=42)   # §7.179 동일
ARM_TAG = {"GEN": "g", "HRD": "h"}


def train_gen_members(df, fold, t0):
    """GEN·HRD 두 팔의 12멤버 z 를 돌려준다 (피처 빌드는 한 번)."""
    paths = {arm: {(v, sd): CACHE / f"initdep{ARM_TAG[arm]}_fold{fold}_{v}_{sd}.npz"
                   for v in LGB_VARIANTS for sd in LGB_SEEDS}
             for arm in ("GEN", "HRD")}
    if all(p.exists() for a in paths.values() for p in a.values()):
        return {a: {k: np.load(p)["z"] for k, p in paths[a].items()}
                for a in ("GEN", "HRD")}

    import lightgbm as lgb
    import pandas as pd
    import soften
    from data import recency_weight
    from features import build_features, fit_artifacts
    from initfix3 import level_init
    from modelabel import recover

    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                        center_asof=False)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xtr, Xva = X(tr), X(va)
    ss = tr["season"].to_numpy(dtype=np.int64)
    gg = tr["game_type"].astype("object").to_numpy().astype(str)

    # ── r(x): fold 학습창 R 행에서 (§7.179 fit_r recipe) ─────────
    lab, _, _ = recover(tr)
    ok = ~np.isnan(lab["reverse"]) & ~np.isnan(lab["middle"])
    rev = np.where(ok, np.round(lab["reverse"]), 0.0)
    mid = np.where(ok, np.round(lab["middle"]), 0.0)
    revonly = (y_tr == 0) & (rev == 1) & (mid == 0)
    pop = ok & (gg == "R") & ((y_tr == 1) | revonly)
    t = revonly[pop].astype(np.float64)
    aff = (ss <= 2022) & (gg == "F") & (y_tr == 1)
    b = lgb.train(R_PARAMS, lgb.Dataset(Xtr[pop], t), num_boost_round=400)
    r_aff = np.asarray(b.predict(Xtr[aff]), dtype=np.float64)
    del b
    print(f"  fold{fold} r 학습 풀(R) {int(pop.sum()):,}행 양성률 {t.mean():.4f}"
          f" · 오염 y=1 {int(aff.sum()):,}행 · r q05/q95 "
          f"{np.percentile(r_aff, 5):.3f}/{np.percentile(r_aff, 95):.3f}")

    # ── GEN: h_row (시즌 질량 보존, §7.179 make_h 동형) ──────────
    h_const = soften.fit_h(tr)
    h_row = np.zeros(len(tr))
    idx_all = np.flatnonzero(aff)
    for yr, hs in h_const.items():
        m = ss[idx_all] == yr
        v = np.maximum(r_aff[m], 1e-6)
        for _ in range(3):
            v = np.clip(v * (hs / v.mean()), 1e-6, H_CAP)
        assert abs(v.mean() - hs) < 1e-3, (yr, float(v.mean()), hs)
        h_row[idx_all[m]] = v
    dup = (h_row > 0) & (y_tr == 1)
    Xg = pd.concat([Xtr, Xtr[dup]], ignore_index=True)
    yg = np.concatenate([y_tr, np.zeros(int(dup.sum()))])
    wg = np.asarray(w, dtype=np.float64).copy()
    wg[dup] *= (1 - h_row[dup])
    wg = np.concatenate([wg, np.asarray(w)[dup] * h_row[dup]])
    ssg = np.concatenate([ss, ss[dup]])
    ggg = np.concatenate([gg, gg[dup]])
    init_g = level_init(ssg, ggg, yg, wg)
    print(f"  fold{fold} GEN dup {int(dup.sum()):,}행 · h_row std "
          f"{h_row[aff].std():.4f} (상수라면 0 근처)")

    # ── HRD: 시즌별 round(h_s·n₁) 개를 r(x) 상위로 hard flip ────
    y_h = y_tr.copy()
    for yr, hs in h_const.items():
        m = ss[idx_all] == yr
        k = int(round(hs * m.sum()))
        top = idx_all[m][np.argsort(r_aff[m])[::-1][:k]]
        y_h[top] = 0.0
        print(f"  fold{fold} HRD {yr}: {k:,}/{int(m.sum()):,}행 flip "
              f"(h_s={hs:.3f})")
    init_h = level_init(ss, gg, y_h, w)

    out = {"GEN": {}, "HRD": {}}
    jobs = [("GEN", Xg, yg, wg, init_g), ("HRD", Xtr, y_h, w, init_h)]
    for arm, X2, y2, w2, init2 in jobs:
        for v in LGB_VARIANTS:
            for sd in LGB_SEEDS:
                p = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v], seed=sd,
                         bagging_seed=sd, feature_fraction_seed=sd,
                         data_random_seed=sd)
                bst = lgb.train(p, lgb.Dataset(X2, y2, weight=w2,
                                               init_score=init2),
                                num_boost_round=N_ROUNDS)
                z = bst.predict(Xva, raw_score=True)
                del bst
                np.savez_compressed(paths[arm][(v, sd)], z=z)
                out[arm][(v, sd)] = z
                print(f"    lgb-{arm} fold{fold} [{v}/{sd}] "
                      f"({time.time()-t0:.0f}s)")
    return out


def main():
    t0 = time.time()
    print(f"§7.195d 생성적 수리 (GEN/HRD)   지문 {D.KEY}")
    df, s, y = D.load_df()
    ST = {f: H.make_state(df, s, y, f, t0) for f in FOLDS}
    NEW = {f: train_gen_members(df, f, t0) for f in FOLDS}
    del df

    LIB = {}
    order = [(v, sd) for v in LGB_VARIANTS for sd in LGB_SEEDS]
    for f in FOLDS:
        st = ST[f]
        LIB[f] = {"CUR": st["Z"]}
        for arm in ("FIX", "GEN", "HRD"):
            Zc = st["Z"].copy()
            for j, (v, sd) in enumerate(order):
                assert st["names"][j] == f"lgb_{v}_{sd}"
                if arm == "FIX":
                    Zc[:, j] = np.load(
                        CACHE / f"initdepx_fold{f}_{v}_{sd}.npz")["z"]
                else:
                    Zc[:, j] = NEW[f][arm][(v, sd)]
            LIB[f][arm] = Zc

    # |Δz| vs CUR — 조작 크기 (§7.179 의 |Δp| 대응물)
    print(f"\n[조작 크기] 12멤버 평균 |Δz| vs CUR (clean 평가행)")
    for f in FOLDS:
        st = ST[f]
        row = []
        for arm in ("FIX", "GEN", "HRD"):
            dz = np.abs(LIB[f][arm][:, :12] - st["Z"][:, :12])[st["mask"]]
            row.append(f"{arm} {dz.mean():.4f}")
        print(f"  fold{f}  " + "   ".join(row))

    print(f"\n[①] lgb 12멤버 분산 (clean 평가행 로짓)")
    for f in FOLDS:
        st = ST[f]
        row = []
        for arm in ("CUR", "FIX", "GEN", "HRD"):
            a_all, _ = disagreement(st, LIB[f][arm][:, :12])
            row.append(f"{arm} {a_all:.4f}")
        print(f"  fold{f}  " + "   ".join(row))

    print(f"\n[②] solo BSS 순위의 fold 간 안정성")
    for arm in ("CUR", "FIX", "GEN", "HRD"):
        solo = {f: np.array([solo_bss(ST[f], LIB[f][arm][:, j])
                             for j in range(12)]) for f in FOLDS}
        rho = spearman(solo[2023], solo[2024])
        print(f"  {arm}: f2023 평균 {solo[2023].mean():.1f}"
              f"   f2024 평균 {solo[2024].mean():.1f}   ρ={rho:+.3f}")

    print(f"\n[③] lgb 가족 내 greedy — 각 팔 자신의 B1 기준   ({time.time()-t0:.0f}s)")
    for arm in ("CUR", "FIX", "GEN", "HRD"):
        family_greedy(ST[2023], ST[2024], LIB[2023][arm], LIB[2024][arm],
                      ST[2023]["fam"], arm)

    print(f"\n[B1 절대] (lgb 균등 + MLP/cat B1 고정)")
    for f in FOLDS:
        row = []
        for arm in ("CUR", "FIX", "GEN", "HRD"):
            _, sc = H.eval_w(ST[f], LIB[f][arm],
                             H.baseline_w(ST[f]["fam"], "B1"))
            row.append(f"{arm} {sc:.2f}")
        print(f"  fold{f}  " + "   ".join(row))

    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
