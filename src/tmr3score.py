"""Trackman 재구축 3단계 — **홀드아웃 2023 · 2024 BSS 실측. 판정자는 이것뿐이다.**

사용자 지시: *"기각 함부로 하지 말고, 무조건 holdout 2023 2024 결과를 보여줘."*
→ t 통계량·상관·형질성은 전부 **설계 근거**일 뿐이고, 여기서 나오는 BSS 가 판정이다.

앞 단계가 만든 것
    1단계  레짐 단절 실측 (`extension` 2021→2022 −0.145 = 0.95 SD) · E1 독립 재현
    2단계  변형 5종 + 예측 상관 — `within`/`recent` 가 현행 `deploy` 를 일관되게 이긴다.
           ★★ §7.9 가 고른 릴리스포인트 산포(ρ .15~.30)가 **운반 가능성 최하위**였고,
              한 번도 단독 시험된 적 없는 `ivb_std`·`hb_std`(ρ .85)가 최상위다.

⚠️ 공유 자원 오염 방지 (이게 이 파일에서 제일 위험한 부분이다)
    `foldcache._key()` 에는 **tm_table 이 안 들어간다.** 그래서 trackman 을 켠 채
    `foldcache.load()` 를 부르면 **같은 키로 배포 베이스라인 캐시를 덮어쓴다** —
    다른 세션의 기준선이 조용히 오염된다. 그래서 이 파일은
      · `foldcache.load()` / `foldcache.compute()` 를 **호출하지 않는다**
      · 자체 캐시 네임스페이스 `experiments/cache_tmr/` 를 쓰고 키에 변형명을 박는다

★ 하네스 재현 관문 (`none` 팔)
    변형 `none` 은 tm_table=None 이므로 배포 베이스라인과 **비트 단위로 같아야 한다.**
    기존 `experiments/cache/fold{f}_<key>.npz` 의 z 와 대조해서
    max|Δz| / sd(z) > 0.1% 면 **하네스가 틀린 것**이고 뒤 숫자는 전부 무효다.
    (간이 재구현은 알려진 수치 재현부터 — 이 저장소가 이미 한 번 데였다.)

보고 형식 — 팔마다 fold 2023 · 2024 를 **따로** 찍는다
    LGBM 단독 / MLP 단독 / 블렌드(α) / 블렌드+δ(배포 상태)
    그리고 `none` 대비 쌍체 Δ 를 같이 낸다 (같은 시드·같은 분할).

실행:
    .venv/Scripts/python.exe -u src/tmr3score.py none    2>&1 | tee experiments/tmr3_none.log
    .venv/Scripts/python.exe -u src/tmr3score.py within  2>&1 | tee experiments/tmr3_within.log
    .venv/Scripts/python.exe -u src/tmr3score.py drift   2>&1 | tee experiments/tmr3_drift.log
    .venv/Scripts/python.exe -u src/tmr3score.py recent  2>&1 | tee experiments/tmr3_recent.log
    .venv/Scripts/python.exe -u src/tmr3score.py --report
"""
import gc
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

import ctxfit
import foldcache
import mlp
import modelabel
import soften
from config import (CTX_AXES, CTX_K, DROP_FEATURES, ENSEMBLE_VARIANTS,
                    FINAL_LGB_PARAMS, MLP_ALPHA, MLP_AUX_LAMBDA, MLP_AUX_MODES,
                    MLP_SEEDS, MLP_VARIANTS, N_ROUNDS, RECENCY_HALF_LIFE, ROOT,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, SOFT_LABEL_F, TARGET_COL,
                    TRAIN_CSV, USE_PLAYER_IDS, CENTER_ASOF, CONTAM_AXES)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts, logit_adjust
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
CACHE = ROOT / "experiments" / "cache_tmr"
TMR_DIR = ROOT / "experiments" / "tmr"
VARIANTS = ["none", "deploy", "within", "drift", "regime", "recent", "both", "focus"]
LGB_SEEDS = foldcache.LGB_SEEDS          # 배포 실험 관행: fold 실험은 3시드
GATE_TOL = 0.001                          # 재현 관문 — max|Δz|/sd(z)


def tm_table(variant):
    if variant == "none":
        return None
    p = TMR_DIR / f"tm_{variant}.csv"
    if not p.exists():
        raise SystemExit(f"없음: {p}  — 먼저 src/tmr2build.py 를 돌려라")
    return pd.read_csv(p)


def fold_data(df, fold, variant):
    """foldcache.fold_data 와 **같은 구조**. 다른 것은 tm_table 하나뿐이다."""
    mtr, mva = foldcache.SPLITS[fold](df.season.to_numpy(), df.game_month.to_numpy())
    tr, va = df[mtr], df[mva]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                        center_asof=CENTER_ASOF, tm_table=tm_table(variant))

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    return (X(tr), tr[TARGET_COL].to_numpy(dtype=np.float64), w,
            X(va), va[TARGET_COL].to_numpy(dtype=np.float64), art, tr)


def part(variant, f, kind, tag):
    return CACHE / f"{variant}_f{f}_{foldcache._key()}_{kind}_{tag}.npz"


def final(variant, f):
    return CACHE / f"{variant}_f{f}_{foldcache._key()}.npz"


def compute(variant, f, df, t0):
    """foldcache.compute 와 같은 순서·같은 시드. 캐시 경로만 분리돼 있다."""
    Xtr, ytr, w, Xva, yva, art, tr = fold_data(df, f, variant)
    ntm = len([c for c in Xtr.columns if c.startswith("tm")])
    print(f"  피처 {Xtr.shape[1]}개 (tm 계열 {ntm}개)  train {len(Xtr):,} → valid {len(Xva):,}")

    aux, cov = modelabel.aux_labels(tr)
    if SOFT_LABEL_F:
        Xtr, ytr, w, aux, soft_h = soften.soften(tr, Xtr, ytr, w, aux)
    print(f"  보조 라벨 {cov:.1%}  soft label 후 {len(Xtr):,}행")

    pl = part(variant, f, "lgb", "all")
    if pl.exists():
        z_lgb = np.load(pl)["z"]
        print(f"  lgbm 재사용   단독 {mlp.score(z_lgb, yva):>8.1f}  ({time.time()-t0:.0f}s)")
    else:
        z_lgb, n = np.zeros(len(Xva)), 0
        for over in ENSEMBLE_VARIANTS.values():
            for s in LGB_SEEDS:
                p = dict(FINAL_LGB_PARAMS, **over, seed=s, bagging_seed=s,
                         feature_fraction_seed=s, data_random_seed=s)
                bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w),
                                num_boost_round=N_ROUNDS)
                z_lgb += bst.predict(Xva, raw_score=True)
                n += 1
                del bst
        z_lgb /= n
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pl, z=z_lgb)
        print(f"  lgbm {n}개   단독 {mlp.score(z_lgb, yva):>8.1f}  ({time.time()-t0:.0f}s)")

    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    del Xtr, Xva
    gc.collect()

    zs = []
    for vname, (cfg, nseed) in MLP_VARIANTS.items():
        for s in MLP_SEEDS[:nseed]:
            pm = part(variant, f, "mlp", f"{vname}_{s}")
            if pm.exists():
                zs.append(np.load(pm)["z"])
                continue
            _, curve = mlp.train_mlp(A, C, ytr, w, B, Cv, sizes, seed=s,
                                     verbose=False, y_aux=aux,
                                     **dict(cfg, aux=MLP_AUX_LAMBDA))
            zs.append(curve[-1])
            np.savez_compressed(pm, z=zs[-1])
            print(f"  mlp {vname}_{s}   단독 {mlp.score(zs[-1], yva):>8.1f}"
                  f"  ({time.time()-t0:.0f}s)")
    z_mlp = np.mean(zs, axis=0)
    print(f"  mlp {len(zs)}개 평균  단독 {mlp.score(z_mlp, yva):>8.1f}  ({time.time()-t0:.0f}s)")
    np.savez_compressed(final(variant, f), z_lgb=z_lgb, z_mlp=z_mlp, y=yva,
                        members=np.stack(zs))
    for p in CACHE.glob(f"{variant}_f{f}_{foldcache._key()}_*.npz"):
        if p != final(variant, f):
            p.unlink()
    return z_lgb, z_mlp, yva


def gate(variant, f, z_lgb, z_mlp):
    """`none` 팔은 배포 베이스라인 캐시를 재현해야 한다."""
    if variant != "none":
        return
    p = foldcache.CACHE / f"fold{f}_{foldcache._key()}.npz"
    if not p.exists():
        print(f"  ⚠️ 재현 관문 생략 — 베이스라인 캐시 없음 ({p.name})")
        return
    with np.load(p) as d:
        for nm, mine, ref in (("z_lgb", z_lgb, d["z_lgb"]), ("z_mlp", z_mlp, d["z_mlp"])):
            rel = float(np.abs(mine - ref).max() / (ref.std() + 1e-12))
            ok = "✅" if rel <= GATE_TOL else "❌"
            print(f"  {ok} 재현 관문 {nm}: max|Δz|/sd(z) = {rel:.5%}  (임계 {GATE_TOL:.1%})")
            if rel > GATE_TOL:
                raise SystemExit("하네스가 배포 베이스라인을 재현하지 못한다 — 중단.")


def deployed_bss(df, fold, z_lgb, z_mlp, yv):
    """네 층에서 BSS 를 낸다. 배포 상태 = 블렌드 + δ(hand) + δ(count4)."""
    s = df["season"].to_numpy()
    tr, va = s < fold, s == fold
    z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
    ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
    adj, _ = logit_adjust(df[va], {"ctx": ctx})
    f = lambda zz: bss(yv, shift_to_prior(1.0 / (1.0 + np.exp(-zz)), yv.mean()))
    return {"LGBM 단독": f(z_lgb), "MLP 단독": f(z_mlp),
            f"블렌드 α{MLP_ALPHA}": f(z), "블렌드+δ (배포)": f(z + adj)}


def report():
    print("=" * 100)
    print("Trackman 재구축 — **홀드아웃 BSS 실측** (fold 2023 · 2024)")
    print("=" * 100)
    have, res = [], {}
    for v in VARIANTS:
        if all(final(v, f).exists() for f in FOLDS):
            have.append(v)
    if "none" not in have:
        print("  기준 팔 `none` 이 없다. 먼저 돌려라."); return
    df = load_csv(TRAIN_CSV)
    for v in have:
        res[v] = {}
        for f in FOLDS:
            with np.load(final(v, f)) as d:
                res[v][f] = deployed_bss(df, f, d["z_lgb"], d["z_mlp"], d["y"])

    layers = list(res["none"][FOLDS[0]].keys())
    for layer in layers:
        print(f"\n── {layer}")
        print(f"  {'변형':<10}{'f2023':>10}{'f2024':>10}{'Δ2023':>10}{'Δ2024':>10}"
              f"{'양fold':>8}   판정")
        b23 = res["none"][2023][layer]
        b24 = res["none"][2024][layer]
        for v in have:
            a, b = res[v][2023][layer], res[v][2024][layer]
            d23, d24 = a - b23, b - b24
            if v == "none":
                print(f"  {v:<10}{a:>10.1f}{b:>10.1f}{'—':>10}{'—':>10}{'':>8}   기준")
                continue
            both = "○" if (d23 > 0 and d24 > 0) else ("●" if (d23 < 0 and d24 < 0) else "×")
            note = {"○": "양 fold 양수", "●": "양 fold 음수", "×": "부호 분열"}[both]
            print(f"  {v:<10}{a:>10.1f}{b:>10.1f}{d23:>+10.1f}{d24:>+10.1f}"
                  f"{both:>8}   {note}")
    print("\n" + "=" * 100)
    print("  ⚠️ 홀드아웃 이득은 채택 근거가 아니다 — §7.9 가 +2.3/+13.3 을 보고 채택해")
    print("     LB −13.6 을 두 번 맞았다. 여기서 양수가 나오면 다음 관문은 **시즌 전이**다.")
    print("  ⚠️ 시드 잡음: MLP 층 SD(Δ) 4.8~7.1 / LGBM 0.22~0.40 (§7.199). 3시드 기준.")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "--report":
        return report()
    variant = sys.argv[1]
    if variant not in VARIANTS:
        raise SystemExit(f"variant 은 {VARIANTS} 중 하나")
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    df = load_csv(TRAIN_CSV)
    print(f"변형 **{variant}**   캐시 {CACHE}")
    for f in FOLDS:
        if final(variant, f).exists():
            with np.load(final(variant, f)) as d:
                z_lgb, z_mlp, yv = d["z_lgb"], d["z_mlp"], d["y"]
            print(f"\n=== fold {f} — 완료본 재사용 ===")
        else:
            print(f"\n=== fold {f} ===")
            z_lgb, z_mlp, yv = compute(variant, f, df, t0)
        gate(variant, f, z_lgb, z_mlp)
        for k, v in deployed_bss(df, f, z_lgb, z_mlp, yv).items():
            print(f"    {k:<18}{v:>10.1f}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
