"""§7.195c — 오염 데이터 수리 실측: 분산 채널 검정 (LGBM 12멤버 재학습)

사용자 반론: "(b) '오염은 공통 상수라 시드 우열을 못 뒤집는다'는 틀렸다.
시드 수준으로도 변할 수 있다." — 맞다. 오염 라벨은 입력으로는 공통이어도
확률적 학습을 통과한 효과는 시드별이다. 라벨 노이즈가 멤버 간 **분산을 팽창**
시키면 selection 이 쫓을 노이즈 자체가 커진다. 이 채널은 §7.195 [A](손실 위치)
와 [B](선택 목적함수)가 못 닫는다 — 수리해서 재는 수밖에 없다.

수리 = 2019~22 F 행 **삭제** (최대 수리; soft label 이 하던 일을 삭제가 대체,
asof 피처 오염은 운영측 계산이라 남고 §7.49 contam 보정은 그대로).
LGBM 12멤버(4변형×3시드)만 재학습 — MLP·cat 재학습은 비용상 불가, 둘은
양 팔 모두 B1 고정 (스코프 한계로 명시).

사전 등록 판정
  ① 분산: FIX 의 멤버 간 std(z) 가 CUR 대비 유의하게 줄어야 채널 실재.
  ② 순위 안정성: 멤버 solo BSS 의 fold2023↔2024 순위 상관(Spearman)이
     FIX 에서 CUR 보다 올라야 "선택이 쫓는 것이 진짜 신호"에 가까워진 것.
  ③ 전이: lgb 가족 내 greedy(LOCK 동형, 타 가족 B1 고정)의 sel→eval 전이 Δ
     (각 팔 자신의 B1 기준) 가 FIX 에서 개선. ③이 0 이상으로 살아야만
     힐클라임 재개봉 후보.
  ①·② 없이 ③만 좋아지는 것은 해석 불가(다른 원인), ①·②만 좋고 ③이
  그대로면 "분산 채널은 실재하나 수리로 부족" — 축 폐쇄 유지.

실행:  .venv/Scripts/python.exe -u src/hcrepair.py 2>&1 | tee experiments/hcrepair.log
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
from validation import bss, shift_to_prior

CACHE = D.CACHE
FOLDS = (2023, 2024)
LGB_SEEDS = SEED_ENSEMBLE[:3]
LGB_VARIANTS = ("base", "wide", "extra", "lowff")
MARKS = (5, 10, 20, 40)
STEPS = 40


def train_fix_members(df, fold, t0):
    """hillclimb.train_lgb_members 와 동일 경로, 단 학습셋에서 오염 F 를 삭제.

    삭제 후 soften 은 자동 무연산(h 를 추정할 2019~22 F 행이 없다) — soft label
    의 일을 삭제가 대체한다는 설계 그 자체. init_score 는 남은 행으로 적합.
    """
    paths = {(v, sd): CACHE / f"initdepx_fold{fold}_{v}_{sd}.npz"
             for v in LGB_VARIANTS for sd in LGB_SEEDS}
    if all(p.exists() for p in paths.values()):
        return {k: np.load(p)["z"] for k, p in paths.items()}

    import lightgbm as lgb
    import soften
    from data import recency_weight
    from features import build_features, fit_artifacts
    from initfix3 import level_init

    s = df["season"].to_numpy()
    gt = df["game_type"].astype("object").to_numpy().astype(str)
    keep = (s < fold) & ~((s < 2023) & (gt == "F"))
    tr, va = df[keep], df[s == fold]
    n_all = int((s < fold).sum())
    print(f"  fold{fold} FIX 학습셋 {keep.sum()}행 (원 {n_all}, "
          f"오염 F {n_all-keep.sum()} 삭제)  F 잔존 "
          f"{int(((gt=='F') & keep).sum())}행")

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
    Xtr, y2, w2, _, h = soften.soften(tr, Xtr, y_tr, w)
    assert h == {} and len(y2) == len(y_tr), \
        f"삭제 후 soften 이 무연산이 아니다: h={h}"
    ss = tr["season"].to_numpy(dtype=np.int64)
    gg = tr["game_type"].astype("object").to_numpy().astype(str)
    init_tr = level_init(ss, gg, y2, w2)

    out = {}
    for v in LGB_VARIANTS:
        for sd in LGB_SEEDS:
            p = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v], seed=sd,
                     bagging_seed=sd, feature_fraction_seed=sd,
                     data_random_seed=sd)
            bst = lgb.train(p, lgb.Dataset(Xtr, y2, weight=w2,
                                           init_score=init_tr),
                            num_boost_round=N_ROUNDS)
            z = bst.predict(Xva, raw_score=True)
            del bst
            np.savez_compressed(paths[(v, sd)], z=z)
            out[(v, sd)] = z
            print(f"    lgb-FIX fold{fold} [{v}/{sd}] ({time.time()-t0:.0f}s)")
    return out


def solo_bss(st, z_member):
    zc = z_member[st["mask"]] + st["ax"]
    p = shift_to_prior(1 / (1 + np.exp(-zc)), st["ybar"])
    return bss(st["yv"], p)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def disagreement(st, Z12):
    """clean 평가행에서의 멤버 간 산포."""
    Zc = Z12[st["mask"]]
    sd_all = float(Zc.std(axis=1).mean())
    sd_seed = []
    for i in range(0, 12, 3):
        sd_seed.append(Zc[:, i:i+3].std(axis=1).mean())
    return sd_all, float(np.mean(sd_seed))


def family_greedy(a, b, Zlib_a, Zlib_b, fam, tag):
    """lgb 가족 내 greedy (타 가족 B1 고정) — LOCK 동형. 각 팔 자신의 B1 기준."""
    w_base = H.baseline_w(fam, "B1")
    lgb_ix = np.where(fam == "lgb")[0]
    others = np.setdiff1d(np.arange(len(fam)), lgb_ix)
    alpha = w_base[lgb_ix].sum()

    out = {}
    for st_sel, st_ev, Za, Zb in ((a, b, Zlib_a, Zlib_b),
                                  (b, a, Zlib_b, Zlib_a)):
        Zf = Za[st_sel["mask"]]
        zfix = Zf[:, others] @ w_base[others]
        hist, _ = H.greedy(Zf[:, lgb_ix], st_sel["ax"], st_sel["yv"],
                           np.arange(12), STEPS, scale=alpha, z_fixed=zfix)
        p1, s1_sel = H.eval_w(st_sel, Za, w_base)
        p1e, s1_ev = H.eval_w(st_ev, Zb, w_base)
        rows_in, rows_tr = [], []
        for k in MARKS:
            w = w_base.copy()
            w[lgb_ix] = alpha * hist[k - 1] / k
            rows_in.append(H.eval_w(st_sel, Za, w)[1] - s1_sel)
            rows_tr.append(H.eval_w(st_ev, Zb, w)[1] - s1_ev)
        out[st_sel is a] = (rows_in, rows_tr, s1_sel)
    for sel_is_23, (ri, rt, s1) in out.items():
        sf, ef = (2023, 2024) if sel_is_23 else (2024, 2023)
        print(f"  [{tag}] sel={sf}→{ef}  자기 B1={s1:.2f}   k:"
              + "".join(f"{k:>8d}" for k in MARKS))
        print("    in-fold Δ " + "".join(f"{d:>+8.2f}" for d in ri))
        print("    전이 Δ    " + "".join(f"{d:>+8.2f}" for d in rt))
    return out


def main():
    t0 = time.time()
    print(f"§7.195c 오염 수리 분산 채널 검정   지문 {D.KEY}")
    df, s, y = D.load_df()
    ST = {f: H.make_state(df, s, y, f, t0) for f in FOLDS}

    FIXZ = {f: train_fix_members(df, f, t0) for f in FOLDS}
    del df

    LIB = {}
    for f in FOLDS:
        st = ST[f]
        Zfix = st["Z"].copy()
        for j, (v, sd) in enumerate([(v, sd) for v in LGB_VARIANTS
                                     for sd in LGB_SEEDS]):
            assert st["names"][j] == f"lgb_{v}_{sd}"
            Zfix[:, j] = FIXZ[f][(v, sd)]
        LIB[f] = Zfix

    # ── ① 멤버 간 분산 ────────────────────────────────────────────
    print(f"\n[①] lgb 12멤버 분산 (clean 평가행 로짓)   ({time.time()-t0:.0f}s)")
    for f in FOLDS:
        st = ST[f]
        c_all, c_seed = disagreement(st, st["Z"][:, :12])
        x_all, x_seed = disagreement(st, LIB[f][:, :12])
        print(f"  fold{f}  전체 std  CUR {c_all:.4f} → FIX {x_all:.4f} "
              f"({(x_all/c_all-1)*100:+.1f}%)   시드내 std  CUR {c_seed:.4f} "
              f"→ FIX {x_seed:.4f} ({(x_seed/c_seed-1)*100:+.1f}%)")

    # ── ② solo BSS 와 순위 안정성 ────────────────────────────────
    print(f"\n[②] 멤버 solo BSS 순위의 fold 간 안정성")
    solo = {}
    for arm, Zget in (("CUR", lambda f: ST[f]["Z"]), ("FIX", lambda f: LIB[f])):
        solo[arm] = {f: np.array([solo_bss(ST[f], Zget(f)[:, j])
                                  for j in range(12)]) for f in FOLDS}
        rho = spearman(solo[arm][2023], solo[arm][2024])
        print(f"  {arm}: fold2023 평균 {solo[arm][2023].mean():.1f} "
              f"(min {solo[arm][2023].min():.1f} max {solo[arm][2023].max():.1f})"
              f"   fold2024 평균 {solo[arm][2024].mean():.1f}"
              f"   순위 상관 ρ={rho:+.3f}")
    names12 = ST[2023]["names"][:12]
    for f in FOLDS:
        b_c = names12[int(np.argmax(solo["CUR"][f]))]
        b_x = names12[int(np.argmax(solo["FIX"][f]))]
        print(f"  fold{f} 최고 멤버  CUR {b_c}   FIX {b_x}")

    # ── ③ 가족 내 greedy 전이 (LOCK 동형) ────────────────────────
    print(f"\n[③] lgb 가족 내 greedy — 각 팔 자신의 B1 기준   ({time.time()-t0:.0f}s)")
    family_greedy(ST[2023], ST[2024], ST[2023]["Z"], ST[2024]["Z"],
                  ST[2023]["fam"], "CUR")
    family_greedy(ST[2023], ST[2024], LIB[2023], LIB[2024],
                  ST[2023]["fam"], "FIX")

    # 참고: FIX B1 vs CUR B1 절대 레벨 (수리가 기본 블렌드에 이득인가)
    print(f"\n[참고] B1 절대 (lgb 균등 + MLP/cat B1 고정)")
    for f in FOLDS:
        _, sc = H.eval_w(ST[f], ST[f]["Z"], H.baseline_w(ST[f]["fam"], "B1"))
        _, sx = H.eval_w(ST[f], LIB[f], H.baseline_w(ST[f]["fam"], "B1"))
        print(f"  fold{f}  CUR {sc:.2f} → FIX {sx:.2f}  (Δ {sx-sc:+.2f})")

    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
