"""§7.205b — `soft2`(투수별 세분 soft label)를 **배포 스택**에서 잰다.

왜 다시 (§7.205 를 버린다)

    §7.205 는 `softlabel2.train_eval` 을 재사용했는데 그건 `lgb.train` 만 돈다.
    배포 블렌드는 `(1−α)·z_lgbm + α·z_mlp`, **α = 0.75** — 즉 내가 잰 것은
    가중 **0.25** 짜리 소수 채널이다. 그리고 이 저장소에는 같은 조작이
    학습기마다 부호가 갈린 실측이 있다 ([[augmentation-targets-learners-not-just-learners]]:
    LGBM +8.5 / MLP −12.1). **LGBM 단독값으로 스택 부호를 말할 수 없다.**

    §7.70 자신도 같은 것을 보여줬다 — soft label 본체가 단독 하네스 **+17.7**,
    배포 스택 **+39.8**. 하네스가 크기도 2.2배 틀렸다.

    ⚠️ 나는 `train_eval` 이 LGBM 전용이라는 것을 §7.205 스크립트의 docstring 에
    직접 적어놓고 그대로 실행했다. 관문을 쓰는 것과 그 관문이 무엇을 재는지
    확인하는 것은 다른 일이다.

이 스크립트가 재는 것

    fold 2024 · `foldcache.compute` 와 **같은 순서**로 배포 경로를 복제한다:
        fold_data -> aux_labels -> **soften(팔별)** -> encode -> LGBM 12 + MLP 8 -> 블렌드

    두 팔은 `h_row` 하나만 다르다.
        soft   h_row = 시즌 상수 h_s                    (= 현 배포)
        soft2  h_row = clip(h_s · r_i/r̄_s, 0, 0.95)    (§7.70 1단계 산식)
        r_i = 보정(§7.49 D/m) + 스무딩(k=200) 된 투수 reverse 성향

    ★ 두 팔이 **같은 시드**(LGBM 42/7/2024 · MLP 42/7/2024/1234/31337)를 쓴다 —
      쌍체(CRN). `ablate17` 이 emb4 5시드 vs emb8 3시드로 비교해 후보에 −2.9
      핸디캡을 준 것과 정반대로 간다. 예산은 축이 아니다.

    ★ 캐시는 `sl4_` 접두로 완전히 분리한다. `foldcache.part` 를 쓰면 배포 캐시를
      오염시킨다 ([[assert-the-cache-key-actually-split]] — 두 팔이 같은 z 를
      돌려주면 assert 로 잡는다).

사전 등록 — 실행 전에 정한다
    · 판정 fold 는 **2024** 하나. (fold2023 은 비용이 2배이고 §7.200 이 이 층에
      예산을 쓰지 말라 했다. 배포 판단은 fold2024 가 한다 — LB 일치 2/2.)
    · 채택 방향 = Δ(soft2 − soft) 의 **점추정이 양수**일 것 (§7.201 결정 규칙).
      유의성은 요구하지 않는다 — 이 층에서 2σ 를 요구하면 챔피언도 기각된다.
    · 크기 전망 = Δ × 전이 배율 **0.175** (soft label 본체 실측: f2024 +17.7 -> LB +3.09).
    · 참고값: 단독 LGBM 하네스에서는 10시드로 **−1.45 ± 1.03** 이었다 (§7.205).

실행:  .venv/Scripts/python.exe -u src/softlabel4.py > experiments/softlabel4.log 2>&1
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

import foldcache
import mlp
import modelabel
import soften
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA,
                    MLP_AUX_LAMBDA, MLP_AUX_MODES, MLP_SEEDS, MLP_VARIANTS,
                    N_ROUNDS, TARGET_COL, TRAIN_CSV)
from data import load_csv
from softlabel2 import rev_propensity
from validation import bss, shift_to_prior

FOLD = 2024
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
XFER = 3.09 / 17.7


def h_rows(tr, y, arm, r_sm):
    """`soften.soften` 과 **같은 마스크 정의**로 h_row 를 만든다.

    soften: h_row[(season==yr) & (game_type=='F')] = h_s,  dup = (h_row>0) & (y==1)
    두 팔의 차이는 그 자리에 상수를 넣느냐 투수별 값을 넣느냐 **하나뿐**이다.
    """
    h = soften.fit_h(tr)
    s = tr["season"].to_numpy()
    g = tr["game_type"].to_numpy().astype(str)
    out = np.zeros(len(tr))
    for yr, v in h.items():
        m = (s == yr) & (g == "F")
        if not m.any():
            continue
        if arm == "soft":
            out[m] = v
        else:
            out[m] = np.clip(r_sm[m] * (v / max(r_sm[m].mean(), 1e-9)), 0.0, soften.H_CAP)
    return out, h


def soften_with(df, X, y, w, aux, h_row):
    """`soften.soften` 의 본체 — h_row 만 밖에서 받는다 (그 외는 바이트 동일 논리)."""
    dup = (h_row > 0) & (y == 1)
    X2 = pd.concat([X, X[dup]], ignore_index=True)
    y2 = np.concatenate([y, np.zeros(int(dup.sum()))])
    w2 = np.asarray(w, dtype=np.float64).copy()
    w2[dup] *= (1 - h_row[dup])
    w2 = np.concatenate([w2, np.asarray(w)[dup] * h_row[dup]])
    a_dup = np.zeros_like(aux[dup])
    if "reverse" in MLP_AUX_MODES:
        a_dup[:, tuple(MLP_AUX_MODES).index("reverse")] = 1.0
    return X2, y2, w2, np.concatenate([aux, a_dup]), int(dup.sum())


def run_arm(arm, tr, Xtr0, ytr0, w0, Xva, yva, art, aux0, r_sm, t0):
    hr, h = h_rows(tr, ytr0, arm, r_sm)
    m = hr > 0
    print(f"\n--- 팔 {arm} ---")
    print(f"  h_s = { {k: round(v,4) for k,v in h.items()} }")
    print(f"  h_row>0 {int(m.sum()):,}행  q10 {np.quantile(hr[m],.1):.3f}"
          f"  중앙 {np.median(hr[m]):.3f}  q90 {np.quantile(hr[m],.9):.3f}  최대 {hr[m].max():.3f}")
    Xtr, ytr, w, aux, ndup = soften_with(tr, Xtr0, ytr0, w0, aux0, hr)
    print(f"  복제 {ndup:,}행 → 학습 {len(Xtr):,}행")

    # ── LGBM 12개 (4변형 × 3시드) ─────────────────────────────────
    z_lgb, n = np.zeros(len(Xva)), 0
    for over in ENSEMBLE_VARIANTS.values():
        for s in foldcache.LGB_SEEDS:
            p = dict(FINAL_LGB_PARAMS, **over, seed=s, bagging_seed=s,
                     feature_fraction_seed=s, data_random_seed=s)
            bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=N_ROUNDS)
            z_lgb += bst.predict(Xva, raw_score=True)
            n += 1
            del bst
    z_lgb /= n
    print(f"  lgbm {n}개  단독 {mlp.score(z_lgb, yva):.2f}  ({time.time()-t0:.0f}s)")

    # ── MLP 8개 (emb4×5 + noid×3) ────────────────────────────────
    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    del Xtr
    gc.collect()
    zs = []
    for vname, (cfg, nseed) in MLP_VARIANTS.items():
        for s in MLP_SEEDS[:nseed]:
            f = CACHE / f"sl4_{arm}_f{FOLD}_{vname}_{s}.npz"
            if f.exists():
                zs.append(np.load(f)["z"])
                continue
            _, curve = mlp.train_mlp(A, C, ytr, w, B, Cv, sizes, seed=s,
                                     verbose=False, y_aux=aux,
                                     **dict(cfg, aux=MLP_AUX_LAMBDA))
            zs.append(curve[-1])
            np.savez_compressed(f, z=zs[-1])
            print(f"    mlp {vname}_{s}  단독 {mlp.score(zs[-1], yva):>8.2f}"
                  f"  ({time.time()-t0:.0f}s)")
    z_mlp = np.mean(zs, axis=0)
    print(f"  mlp {len(zs)}개 평균  단독 {mlp.score(z_mlp, yva):.2f}  ({time.time()-t0:.0f}s)")

    z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
    p = shift_to_prior(1.0 / (1.0 + np.exp(-z)), yva.mean())
    score = bss(yva, p)
    print(f"  ★ 블렌드 α={MLP_ALPHA}  BSS {score:.2f}")
    del A, C, B, Cv
    gc.collect()
    return score, z_lgb, z_mlp, np.stack(zs)


def main():
    t0 = time.time()
    print(f"§7.205b  soft2 — 배포 스택 (fold {FOLD})")
    print(f"  LGBM {len(ENSEMBLE_VARIANTS)}변형 × {len(foldcache.LGB_SEEDS)}시드"
          f" · MLP {sum(v[1] for v in MLP_VARIANTS.values())}멤버 · α {MLP_ALPHA}")
    print(f"  두 팔 같은 시드 (쌍체). 차이는 h_row 하나.\n")

    df = load_csv(TRAIN_CSV)
    Xtr0, ytr0, w0, Xva, yva, art, tr = foldcache.fold_data(df, FOLD)
    aux0, cov = modelabel.aux_labels(tr)
    r_sm = rev_propensity(tr, art)
    print(f"fold {FOLD}  학습 {len(Xtr0):,}행 · 검증 {len(Xva):,}행 · 보조라벨 복원 {cov:.1%}")
    print(f"  r_sm  평균 {r_sm.mean():.4f}  sd {r_sm.std():.4f}"
          f"  [{r_sm.min():.4f}, {r_sm.max():.4f}]")

    out = {}
    for arm in ("soft", "soft2"):
        out[arm] = run_arm(arm, tr, Xtr0, ytr0, w0, Xva, yva, art, aux0, r_sm, t0)

    assert not np.array_equal(out["soft"][1], out["soft2"][1]), "두 팔의 LGBM 이 같다 — h_row 가 안 갈렸다"
    assert not np.array_equal(out["soft"][2], out["soft2"][2]), "두 팔의 MLP 가 같다"

    d = out["soft2"][0] - out["soft"][0]
    # 멤버 쌍체 — 같은 시드 인덱스끼리 MLP 단독 점수 차
    dm = np.array([mlp.score(a, yva) - mlp.score(b, yva)
                   for a, b in zip(out["soft2"][3], out["soft"][3])])
    print("\n" + "=" * 78)
    print(f"판정 (사전 등록: 점추정 양수면 채택 후보)")
    print("=" * 78)
    print(f"  soft   블렌드 BSS {out['soft'][0]:.2f}")
    print(f"  soft2  블렌드 BSS {out['soft2'][0]:.2f}")
    print(f"  ★ Δ(soft2−soft) = {d:+.2f}   →  전망 LB {d*XFER:+.2f}")
    print(f"  MLP 멤버 쌍체 Δ: {np.array2string(dm, precision=1, floatmode='fixed')}")
    print(f"     평균 {dm.mean():+.2f} ± {dm.std(ddof=1)/np.sqrt(len(dm)):.2f}  양수 {int((dm>0).sum())}/{len(dm)}")
    print(f"\n  참고 — 단독 LGBM 하네스 10시드 (§7.205): −1.45 ± 1.03")
    print(f"  판정: {'채택 후보 (점추정 양수)' if d > 0 else '기각 (점추정 음수)'}")
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
