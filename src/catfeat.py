"""§7.138 — LightGBM `categorical_feature` A/B. §7.137 의 외삽을 측정으로 대체한다.

§7.137 은 순서 비용을 **root 단일 분할에서** 쟀다 (팀 1.00/1.24배, base_state 3.30배).
거기서 "트리 전체에서도 작을 것"으로 외삽해 "하지 마세요"라고 권고했다.
그 외삽의 약점은 명확하다 — 깊은 노드의 최적 그룹핑, 부스팅 라운드에 따른
gradient 재분포, 팀x상황 상호작용은 root 계산이 원리적으로 못 본다.
**11분짜리 실험 앞에서 외삽을 쓸 이유가 없다.** 측정한다.

팔 (바뀌는 것은 lgb.Dataset 의 categorical_feature 하나뿐. X 는 비트 단위 동일)
    현행   앵커. 캐시된 initdep_fold{f}_ctr0.npz. base 재현이 하네스 검증이다.
    A      base_state + top_bottom              (8종·2종, 순서비용 3.30배)
    B      A + pitcher_team_id + batter_team_id (13·13종, 순서비용 1.24·1.00배)

측정 층: 배포 블렌드 (z = .25*z_lgb + .75*z_mlp + Xax@beta, prior shift, clean 행).
        **solo 도 같이 찍는다** — solo 이득이 blend 에서 얼마나 희석되는지가
        "MLP 가 이미 임베딩으로 처리 중"이라는 §7.136-3 주장의 검증이다.

채택 기준 — **실행 전에 박는다**
   (1) 블렌드 Delta 가 **양 fold 양수**
   (2) 시드 노이즈 초과 — 재학습 축이라 std 4~9. |Delta| < 10 은 무승부로 읽는다
   (3) 크기비 <= 3
   (4) 투수 클러스터 부트스트랩 구간이 0 을 유의하게 밑돌지 않을 것
   alpha/beta/N_ROUNDS 는 **배포값 고정**. fold 를 보고 다시 고르면 +2.0 이 부푼다.

사전 예측 (§7.137 외삽): A·B 모두 |Delta| < 5. B 가 A 보다 클 이유는 없다
(팀 순서비용이 base_state 보다 작으므로). 이 예측이 틀리면 root 외삽이 틀린 것이다.

실행:  .venv/Scripts/python.exe -u src/catfeat.py 2>&1 | tee experiments/catfeat.log
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
import lightgbm as lgb

import deploy29 as D
import soften
from config import (CONTAM_AXES, DROP_FEATURES, ENSEMBLE_VARIANTS,
                    FINAL_LGB_PARAMS, MLP_ALPHA, N_ROUNDS, RECENCY_HALF_LIFE,
                    SEED_ENSEMBLE, SMOOTH_K_PITCHER, SMOOTH_K_BATTER,
                    TARGET_COL, USE_PLAYER_IDS)
from data import recency_weight
from features import build_features, fit_artifacts
from initfix3 import level_init
from validation import bss, shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
LGB_SEEDS = SEED_ENSEMBLE[:3]
SEED = 42
ARMS = [
    ("A base_state+top_bottom", "A", ["base_state", "top_bottom"]),
    ("B A+team_id 2개", "B", ["base_state", "top_bottom",
                              "pitcher_team_id", "batter_team_id"]),
]


def train_lgb(df, fold, cat_cols, tag, t0):
    pth = CACHE / f"catfeat_fold{fold}_{tag}.npz"
    if pth.exists():
        print(f"    캐시 적중 {pth.name}")
        return np.load(pth)["z"]

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
    Xtr, y2, w2, _, _ = soften.soften(tr, Xtr, y_tr, w)
    h = soften.fit_h(tr)
    h_row = np.zeros(len(tr))
    for yr, v in h.items():
        h_row[(ss == yr) & (gg == "F")] = v
    dup = (h_row > 0) & (y_tr == 1)
    ss2, gg2 = np.concatenate([ss, ss[dup]]), np.concatenate([gg, gg[dup]])
    init_tr = level_init(ss2, gg2, y2, w2)
    assert len(init_tr) == len(Xtr)

    # 유일한 변경점: 범주형 컬럼을 정수로 캐스팅하고 이름으로 선언한다.
    # 음수(-1 = 미지값)는 LightGBM 이 결측으로 처리한다.
    cats = [c for c in cat_cols if c in Xtr.columns]
    assert len(cats) == len(cat_cols), f"컬럼 누락: {set(cat_cols) - set(cats)}"
    for c in cats:
        Xtr[c] = Xtr[c].to_numpy(np.float64).round().astype(np.int32)
        Xva[c] = Xva[c].to_numpy(np.float64).round().astype(np.int32)
    desc = ", ".join(f"{c}({Xtr[c].nunique()}종)" for c in cats)
    print(f"    범주형 {len(cats)}개: {desc}")

    z, n = np.zeros(len(Xva)), 0
    for over in ENSEMBLE_VARIANTS.values():
        for sd in LGB_SEEDS:
            p = dict(FINAL_LGB_PARAMS, **over, seed=sd, bagging_seed=sd,
                     feature_fraction_seed=sd, data_random_seed=sd)
            ds = lgb.Dataset(Xtr, y2, weight=w2, init_score=init_tr,
                             categorical_feature=cats)
            bst = lgb.train(p, ds, num_boost_round=N_ROUNDS)
            z += bst.predict(Xva, raw_score=True)
            n += 1
            del bst, ds
    z /= n
    np.savez_compressed(pth, z=z)
    print(f"    fold {fold} [{tag}] LGBM {n}개 완료 ({time.time() - t0:.0f}s)")
    return z


def score(z_lgb, z_mlp, Xa, dep, mm, yv, solo=False):
    z = z_lgb if solo else (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
    z = z[mm] + (0.0 if solo else Xa[mm] @ dep)
    p = shift_to_prior(1 / (1 + np.exp(-z)), yv.mean())
    return bss(yv, p)


def boot(z0, z1, z_mlp, Xa, dep, mm, yv, cl, n_id, rng, B=200):
    """투수 클러스터 복원추출로 Delta BSS 구간."""
    def pv(zl):
        z = (1 - MLP_ALPHA) * zl + MLP_ALPHA * z_mlp
        z = z[mm] + Xa[mm] @ dep
        return shift_to_prior(1 / (1 + np.exp(-z)), yv.mean())
    p0, p1 = pv(z0), pv(z1)
    e0, e1 = (p0 - yv) ** 2, (p1 - yv) ** 2
    n = np.bincount(cl, minlength=n_id)
    s0 = np.bincount(cl, weights=e0, minlength=n_id)
    s1 = np.bincount(cl, weights=e1, minlength=n_id)
    seen = np.where(n > 0)[0]
    ref = yv.mean() * (1 - yv.mean())
    out = []
    for _ in range(B):
        pick = rng.choice(seen, size=len(seen), replace=True)
        N = n[pick].sum()
        out.append(1e5 * (s0[pick].sum() - s1[pick].sum()) / (N * ref))
    return np.percentile(out, [2.5, 97.5])


def main():
    t0 = time.time()
    df, s, y = D.load_df()
    rng = np.random.default_rng(SEED)
    res = {}
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        with np.load(CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
            z_mlp = d0["z_mlp"]
        z_cur = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
        mm, yv, dep = st["mm"], st["yv"], st["dep"]
        # deploy29.state 는 Xax 를 이미 mm 으로 잘라 돌려준다 -> 전체 길이로 되돌린다
        Xa = np.zeros((len(z_cur), st["Xax"].shape[1]))
        Xa[mm] = st["Xax"]

        b_blend = score(z_cur, z_mlp, Xa, dep, mm, yv)
        b_solo = score(z_cur, z_mlp, Xa, dep, mm, yv, solo=True)
        assert abs(b_blend - D.EXPECT_BASE[fold]) < 0.3, \
            f"앵커 실패 {b_blend:.1f} != {D.EXPECT_BASE[fold]}"
        print(f"\nfold {fold}  앵커 OK — 블렌드 {b_blend:.1f} "
              f"(기대 {D.EXPECT_BASE[fold]})  ·  LGBM solo {b_solo:.1f}")
        res[(fold, "현행")] = (b_blend, b_solo, None)

        for name, tag, cats in ARMS:
            zc = train_lgb(df, fold, cats, tag, t0)
            v_b = score(zc, z_mlp, Xa, dep, mm, yv)
            v_s = score(zc, z_mlp, Xa, dep, mm, yv, solo=True)
            ci = boot(z_cur, zc, z_mlp, Xa, dep, mm, yv, st["cl"], st["n_id"], rng)
            res[(fold, name)] = (v_b, v_s, ci)
            print(f"    -> 블렌드 {v_b:.1f} ({v_b - b_blend:+.1f})   "
                  f"solo {v_s:.1f} ({v_s - b_solo:+.1f})   "
                  f"부트스트랩 [{ci[0]:+.1f}, {ci[1]:+.1f}]")

    print("\n" + "=" * 100)
    print("판정 — 배포 블렌드 층 (alpha=0.75 MLP, delta축 4개, prior shift, clean 행)")
    print("=" * 100)
    print(f"  {'팔':<26s}{'f2023':>10s}{'f2024':>10s}"
          f"{'D2023':>9s}{'D2024':>9s}{'Dsolo23':>10s}{'Dsolo24':>10s}")
    b23, s23, _ = res[(2023, "현행")]
    b24, s24, _ = res[(2024, "현행")]
    print(f"  {'현행 (앵커)':<26s}{b23:>10.1f}{b24:>10.1f}"
          f"{0.0:>+9.1f}{0.0:>+9.1f}{0.0:>+10.1f}{0.0:>+10.1f}")
    for name, _tag, _c in ARMS:
        v23, o23, c23 = res[(2023, name)]
        v24, o24, c24 = res[(2024, name)]
        print(f"  {name:<26s}{v23:>10.1f}{v24:>10.1f}"
              f"{v23 - b23:>+9.1f}{v24 - b24:>+9.1f}"
              f"{o23 - s23:>+10.1f}{o24 - s24:>+10.1f}")
        print(f"  {'':<26s}부트스트랩 2023 [{c23[0]:+.1f}, {c23[1]:+.1f}]   "
              f"2024 [{c24[0]:+.1f}, {c24[1]:+.1f}]")
    print("\n  채택: (1) 양 fold 양수 (2) |D|>10 (재학습 시드 노이즈 std 4~9)")
    print("        (3) 크기비<=3 (4) 부트스트랩이 0을 유의하게 밑돌지 않을 것")
    print("  사전 예측(§7.137 외삽): A·B 모두 |D|<5, B<=A. 빗나가면 root 외삽이 틀린 것이다.")


if __name__ == "__main__":
    main()
