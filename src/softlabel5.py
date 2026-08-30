"""§7.205c — `soft2` 를 **각 팔의 자기 최적 β 에서** 비교한다. `init_score` 포함.

사용자: *"베타 적합을 하고 평가해야 하는 거 아니냐."* 맞다. §7.205b 는 안 했다.

무엇이 틀렸었나 (두 겹)

  ① §7.205b 는 두 팔을 **`soft` 의 최적 β(=1.00)** 에서 비교했다. β 는 `soft` 의
     시즌 상수 형태에 대해 적합된 값이다 (§7.161). `ablate17` 이 emb8 을 emb4 의
     시드 예산으로 재서 −2.9 핸디캡을 준 것과 같은 구조다 — **후보를 대조군에
     맞춘 설정에서 평가했다.**

     ⚠️ 부분 방어는 있었다: 내 soft2 산식 `r_i·(h_s/r̄_s)` 은 시즌별 **h̄ 를 정확히
     보존**하고 (H_CAP 0.95 에 안 걸렸다 — 최대 0.444), §7.161 ③ 은 `lin`/`logit`/
     `target` 세 모드가 **h̄ 축에서 겹친다**고 실측했다. 그러나 그 셋은 전부 시즌
     상수의 재모수화다. soft2 는 h 에 **분산**을 넣으므로 그 가족 밖이고,
     분산이 라벨 노이즈로 작용하면 **최적 h̄ 자체가 이동**한다.
     β 곡률이 3점대(β 1.00 → 0.87 이 −3.5)인데 §7.205b 가 잰 Δ 는 −0.29 다.
     **곡률이 효과보다 10배 크다** — 최적점이 조금만 밀려도 결론이 뒤집힌다.

  ② `init_score` 가 빠져 있었다. §7.161 ⑤ 가 `init_score` 를 켜자 **β 네 칸 전부
     부호가 뒤집혔다**고 실측했다 (init OFF +11.9 / +10.7 → ON −3.5 / −3.2).
     둘은 대체재다 (§7.109: 같이 켜서 −727). init OFF 로 β 를 적합하면 β 가
     `init_score` 의 일을 대신 떠맡아 **엉뚱한 최적점**이 나온다.
     ([[ablation-harness-omitting-a-substitute-shifts-the-optimum]] — 내 메모리에
     이미 적혀 있던 사고를 §7.205b 에서 그대로 재현했다.)

설계

    fold 2024 · `init_score` ON — **배포(`train.py:606`)와 같은 구성**:
    `_season_gt_level(ss, gt, y, w)` 을 **soften 뒤** 가중치로 부른다(= 탈오염 레벨).
    · LGBM 4변형 × 3시드 = 12 · MLP emb4×5 + noid×3 = 8 · 블렌드 α = 0.75

    ⚠️ 이 하네스는 배포의 **부분집합**이다 — CatBoost(α .10) · ctx/asof δ · fspec ·
       확률공간 아핀이 없다. 배포 fold2024 BSS 는 ~924 인데 여기는 그보다 낮게 나온다.
       팔 간 **쌍체 차이**를 읽는 용도이지 절대 수준을 읽는 용도가 아니다.

    β ∈ {0.85, 1.00, 1.15} × 팔 {soft, soft2} = **6칸**. 두 팔이 같은 시드(쌍체).
    각 팔의 곡선 정점을 찾고 **정점끼리** 비교한다.

    β 는 `soften.fit_h(tr, beta=β, mode="lin")` 로 h_s 를 스케일한 뒤
    soft2 는 그 h_s 를 투수별로 재배분한다 (h̄ 보존). 즉 β 는 **크기**,
    soft2 는 **모양** — 두 축이 직교하게 걸린다.

사전 등록
    · 채택 방향 = max_β Δ(soft2) − max_β Δ(soft) 의 **점추정 양수** (§7.201).
    · soft 의 정점이 β=1.00 이 아니면 그 자체가 보고 대상이다 (§7.161 은 다른
      하네스였다 — 거기엔 MLP 가 없었고 fspec 이 있었다).
    · 전망 = Δ × 전이 배율 0.175.
    · 참고: §7.205 단독 LGBM 10시드 −1.45±1.03 · §7.205b init 없는 블렌드 −0.29.

실행:  .venv/Scripts/python.exe -u src/softlabel5.py > experiments/softlabel5.log 2>&1
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

import foldcache
import mlp
import modelabel
import soften
from train import _logit, _season_gt_level
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA,
                    MLP_AUX_LAMBDA, MLP_SEEDS, MLP_VARIANTS, N_ROUNDS,
                    TARGET_COL, TRAIN_CSV)
from data import load_csv
from softlabel2 import rev_propensity
from softlabel4 import soften_with
from validation import bss, shift_to_prior

FOLD = 2024
BETAS = (0.85, 1.00, 1.15)
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
XFER = 3.09 / 17.7


def h_rows(tr, arm, r_sm, beta):
    """β 로 크기를, arm 으로 모양을 정한다. 두 축은 직교 — soft2 는 h̄ 를 보존한다."""
    h = soften.fit_h(tr, beta=beta, mode="lin")
    s = tr["season"].to_numpy()
    g = tr["game_type"].to_numpy().astype(str)
    out = np.zeros(len(tr))
    for yr, v in h.items():
        m = (s == yr) & (g == "F")
        if not m.any():
            continue
        out[m] = v if arm == "soft" else np.clip(
            r_sm[m] * (v / max(r_sm[m].mean(), 1e-9)), 0.0, soften.H_CAP)
    return out, h


def run_cell(arm, beta, tr, X0, y0, w0, Xva, yva, art, aux0, r_sm,
             ss0, gt0, base_va, t0):
    hr, h = h_rows(tr, arm, r_sm, beta)
    m = hr > 0
    hbar = float(hr[m].mean())
    Xtr, ytr, w, aux, ndup = soften_with(tr, X0, y0, w0, aux0, hr)
    dup = (hr > 0) & (y0 == 1)
    # ⚠️ 배포(train.py:606)와 같이 **soften 뒤** 가중 레벨로 만든다 = 탈오염 레벨.
    #    asofctr3 처럼 soften 전 표를 복제하면 β 가 레벨을 못 움직이고,
    #    그러면 soft label 과 init_score 의 대체 관계(§7.109)가 하네스에서 사라진다.
    ss_lv = np.concatenate([ss0, ss0[dup]])
    gt_lv = np.concatenate([gt0, gt0[dup]])
    init_tr = _season_gt_level(ss_lv, gt_lv, ytr, w)
    assert len(Xtr) == len(ytr) == len(w) == len(init_tr), "init_score 정렬 깨짐"
    print(f"\n--- {arm} β={beta:.2f} ---")
    print(f"  h̄ {hbar:.4f}  중앙 {np.median(hr[m]):.3f}  q90 {np.quantile(hr[m],.9):.3f}"
          f"  최대 {hr[m].max():.3f}  복제 {ndup:,}행")

    z_lgb, n = np.zeros(len(Xva)), 0
    for over in ENSEMBLE_VARIANTS.values():
        for s in foldcache.LGB_SEEDS:
            p = dict(FINAL_LGB_PARAMS, **over, seed=s, bagging_seed=s,
                     feature_fraction_seed=s, data_random_seed=s)
            bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w, init_score=init_tr),
                            num_boost_round=N_ROUNDS)
            z_lgb += bst.predict(Xva, raw_score=True)
            n += 1
            del bst
    z_lgb = z_lgb / n + base_va      # init_score 기준 로짓 복원 (asofctr3:159)
    print(f"  lgbm {n}개  단독 {mlp.score(z_lgb, yva):.2f}  ({time.time()-t0:.0f}s)")

    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    del Xtr
    gc.collect()
    zs = []
    for vname, (cfg, nseed) in MLP_VARIANTS.items():
        for s in MLP_SEEDS[:nseed]:
            f = CACHE / f"sl5_{arm}_b{beta:.2f}_f{FOLD}_{vname}_{s}.npz"
            if f.exists():
                zs.append(np.load(f)["z"])
                continue
            _, curve = mlp.train_mlp(A, C, ytr, w, B, Cv, sizes, seed=s,
                                     verbose=False, y_aux=aux,
                                     **dict(cfg, aux=MLP_AUX_LAMBDA))
            zs.append(curve[-1])
            np.savez_compressed(f, z=zs[-1])
    z_mlp = np.mean(zs, axis=0)
    print(f"  mlp 8개 평균  단독 {mlp.score(z_mlp, yva):.2f}  ({time.time()-t0:.0f}s)")

    z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
    sc = bss(yva, shift_to_prior(1.0 / (1.0 + np.exp(-z)), yva.mean()))
    print(f"  ★ 블렌드 BSS {sc:.2f}")
    del A, C, B, Cv
    gc.collect()
    return sc, hbar, z_lgb


def main():
    t0 = time.time()
    print(f"§7.205c  soft2 — 팔별 β 적합 · init_score ON · fold {FOLD}")
    print(f"  β 격자 {BETAS} × 팔 (soft, soft2) = {2*len(BETAS)}칸, 두 팔 같은 시드\n")

    df = load_csv(TRAIN_CSV)
    X0, y0, w0, Xva, yva, art, tr = foldcache.fold_data(df, FOLD)
    aux0, cov = modelabel.aux_labels(tr)
    r_sm = rev_propensity(tr, art)

    ss0 = tr["season"].to_numpy(dtype=np.int64)
    gt0 = tr["game_type"].astype("object").to_numpy().astype(str)
    # 검증 기준 로짓 — 검증 시즌은 레벨 표 밖이다. 배포 추론이 REF_SEASON 레벨
    # 스칼라를 쓰는 것과 같은 규약으로, 학습 마지막 시즌 R 의 가중 양성률을 쓴다.
    mref = (ss0 == ss0.max()) & (gt0 == "R")
    base_va = float(_logit(np.array([np.average(y0[mref], weights=w0[mref])]))[0])
    print(f"fold {FOLD}  학습 {len(X0):,} · 검증 {len(Xva):,} · 보조라벨 {cov:.1%}")
    print(f"  기준 로짓 base_va = {base_va:+.5f}  ({int(ss0.max())} R 가중 양성률)")

    res = {}
    for arm in ("soft", "soft2"):
        for b in BETAS:
            res[(arm, b)] = run_cell(arm, b, tr, X0, y0, w0, Xva, yva, art, aux0,
                                     r_sm, ss0, gt0, base_va, t0)

    for b in BETAS:
        assert not np.array_equal(res[("soft", b)][2], res[("soft2", b)][2]), \
            f"β={b} 에서 두 팔의 LGBM 이 같다 — 모양이 안 갈렸다"

    print("\n" + "=" * 78)
    print("β 곡선 (블렌드 BSS)")
    print("=" * 78)
    print(f"  {'β':>6}{'h̄':>9}{'soft':>10}{'soft2':>10}{'soft2−soft':>12}")
    for b in BETAS:
        s0, hb0, _ = res[("soft", b)]
        s2, hb2, _ = res[("soft2", b)]
        print(f"  {b:>6.2f}{hb0:>9.4f}{s0:>10.2f}{s2:>10.2f}{s2-s0:>+12.2f}")

    best0 = max(BETAS, key=lambda b: res[("soft", b)][0])
    best2 = max(BETAS, key=lambda b: res[("soft2", b)][0])
    v0, v2 = res[("soft", best0)][0], res[("soft2", best2)][0]
    d = v2 - v0
    print("\n" + "=" * 78)
    print("판정 — 각 팔의 자기 최적 β 끼리")
    print("=" * 78)
    print(f"  soft   최적 β {best0:.2f}  BSS {v0:.2f}"
          f"{'   ⚠️ 격자 끝 — 정점이 밖일 수 있다' if best0 in (BETAS[0], BETAS[-1]) else ''}")
    print(f"  soft2  최적 β {best2:.2f}  BSS {v2:.2f}"
          f"{'   ⚠️ 격자 끝' if best2 in (BETAS[0], BETAS[-1]) else ''}")
    print(f"  ★ Δ = {d:+.2f}   →  전망 LB {d*XFER:+.2f}")
    print(f"  판정: {'채택 후보 (점추정 양수)' if d > 0 else '기각 (점추정 음수)'}")
    print(f"\n  참고: §7.205 단독LGBM −1.45±1.03 · §7.205b init 없는 블렌드 −0.29")
    print(f"  §7.161 은 배포 설정에서 soft 최적 β=1.00 (0.87 −3.5 · 0.65 −3.7) 이었다")
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
