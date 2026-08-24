"""§7.139 — CatBoost 재평가. **두 질문이 겹쳐 있고 둘 다 미측정이다.**

Q1  저카디널리티 `cat_features` — `CAT_ID_COLS = ["pitcher_id","batter_id"]` 뿐이라
    팀 ID·base_state 는 CatBoost 에도 한 번도 안 들어갔다. §7.12 의 −101.4 는
    "pitcher_id CTR 이 어느 시대에 던졌나를 인코딩"하는 기전이고, 전 시즌 존재하는
    13개 팀에는 적용되지 않는다.
Q2  **§7.14 는 §7.16(MLP 도입, LB +29.5) 이전이다.** corr 0.985 는 corr(z_cat, z_lgb)
    였고 — 트리 대 트리다. 지금 스택은 MLP 75% / LGBM 25%. corr(z_cat, z_blend) 를
    잰 적이 없다. §7.14 바로 앞 문단의 저장소 자체 규칙:
        "기준은 핸디캡 대비 탈상관 보너스의 비율이고, corr 을 반드시 같이 봐야 한다"
        "핸디캡 −223 짜리 MLP(corr 0.83)가 −81 짜리 xgb(corr 0.9+)를 이겼다"
    -> 핸디캡 −27.9 는 그 자체로 사망 선고가 아니다.

사전 등록 예측 (§7.138 기전으로부터)
    · lowcard 팔의 팀 ID 성분은 **죽거나 해로울 것**. 팀 성공률의 풀링->다음시즌
      전이가 +0.390/−0.391 (pitcher), −0.464/+0.157 (batter) 로 부호가 뒤집힌다.
      이건 학습기 속성이 아니라 데이터 속성이라 CatBoost 에도 그대로 간다.
    · allid 팔은 **크게 음수**여야 정상 (§7.12 −101 재현). 안 나오면 하네스가 틀렸다.
    · Q2 가 진짜 열린 질문이다. corr(z_cat, z_blend) 가 0.985 보다 유의하게 낮으면
      블렌드 보너스 여지가 생긴다.

백엔드: GPU (CatBoost 4.6배). ⚠️ GPU 는 CPU 와 **다른 모델**이다 (corr 0.9957) 하고
`rsm` 을 못 쓴다. 그래서 **모든 Δ 는 같은 백엔드 앵커(gpu_none) 대비**로 잰다.
CPU 재현은 목표가 아니다 — "CatBoost 가 지금 스택에 도움이 되나"가 질문이다.

공정성: LGBM 과 **같은 X, 같은 soften, 같은 init_score(baseline)** 을 준다.
        init_score 없이 재면 CatBoost 가 시즌 레벨을 스스로 학습해야 해서 불리하다
        (§7.111 이 LGBM 에 준 것과 같은 것을 준다).

실행:  .venv/Scripts/python.exe -u src/catgpu.py 2>&1 | tee experiments/catgpu.log
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
from catboost import CatBoostClassifier, Pool

import deploy29 as D
import soften
from config import (CONTAM_AXES, DROP_FEATURES, MLP_ALPHA, RECENCY_HALF_LIFE,
                    SMOOTH_K_PITCHER, SMOOTH_K_BATTER, TARGET_COL, USE_PLAYER_IDS)
from data import recency_weight
from features import build_features, fit_artifacts
from initfix3 import level_init
from validation import bss, shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEEDS = (42, 7, 2024)
ITERS = 250                 # §7.12: loss_num 곡선이 150~350 에서 평평. fold 를 안 본다
WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.30)

# §7.12 CAT_PARAMS 에서 rsm 만 제거 (GPU 미지원). 나머지는 그대로.
CAT_PARAMS = dict(learning_rate=0.03, l2_leaf_reg=30.0, border_count=128,
                  grow_policy="Lossguide", max_leaves=31, min_data_in_leaf=2000,
                  loss_function="Logloss", verbose=0, allow_writing_files=False,
                  task_type="GPU", devices="0")

LOWCARD = ["base_state", "top_bottom", "pitcher_team_id", "batter_team_id"]
ARMS = [
    ("gpu_none  (기준선)", "none", []),
    ("gpu_lowcard", "lowcard", LOWCARD),
    ("gpu_allid  (죽은축 대조)", "allid", LOWCARD + ["pitcher_id", "batter_id"]),
]


def prep(df, fold):
    """LGBM 과 비트 단위로 같은 X / y / w / init_score 를 만든다."""
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
    return Xtr, Xva, y2, w2, init_tr


def train_cat(Xtr, Xva, y2, w2, init_tr, cats, tag, fold, t0):
    pth = CACHE / f"catgpu_fold{fold}_{tag}.npz"
    if pth.exists():
        print(f"    캐시 적중 {pth.name}")
        return np.load(pth)["z"]
    Xtr, Xva = Xtr.copy(), Xva.copy()
    idx = None
    if cats:
        miss = [c for c in cats if c not in Xtr.columns]
        assert not miss, f"컬럼 누락: {miss}"
        for c in cats:
            Xtr[c] = Xtr[c].to_numpy(np.float64).round().astype(np.int64).astype(str)
            Xva[c] = Xva[c].to_numpy(np.float64).round().astype(np.int64).astype(str)
        idx = [Xtr.columns.get_loc(c) for c in cats]
        print(f"    cat_features {len(cats)}개: "
              + ", ".join(f"{c}({Xtr[c].nunique()})" for c in cats))
    z, n = np.zeros(len(Xva)), 0
    for sd in SEEDS:
        m = CatBoostClassifier(iterations=ITERS, random_seed=sd, **CAT_PARAMS)
        m.fit(Pool(Xtr, y2, weight=w2, baseline=init_tr, cat_features=idx))
        z += m.predict(Pool(Xva, cat_features=idx), prediction_type="RawFormulaVal")
        n += 1
        del m
    z /= n
    np.savez_compressed(pth, z=z)
    print(f"    fold {fold} [{tag}] CatBoost {n}시드 완료 ({time.time() - t0:.0f}s)")
    return z


def main():
    t0 = time.time()
    df, s, y = D.load_df()
    out = {}
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        with np.load(CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
            z_mlp = d0["z_mlp"]
        z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
        mm, yv, dep = st["mm"], st["yv"], st["dep"]
        Xa = np.zeros((len(z_lgb), st["Xax"].shape[1]))
        Xa[mm] = st["Xax"]
        z_blend = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp

        def sc(zb, solo=False):
            z = zb[mm] + (0.0 if solo else Xa[mm] @ dep)
            return bss(yv, shift_to_prior(1 / (1 + np.exp(-z)), yv.mean()))

        base = sc(z_blend)
        assert abs(base - D.EXPECT_BASE[fold]) < 0.3, f"앵커 실패 {base:.1f}"
        solo_lgb = sc(z_lgb, solo=True)
        print(f"\nfold {fold}  앵커 OK 블렌드 {base:.1f}  ·  LGBM solo {solo_lgb:.1f}")

        Xtr, Xva, y2, w2, init_tr = prep(df, fold)
        print(f"    X {Xtr.shape} 준비 ({time.time() - t0:.0f}s)")
        for name, tag, cats in ARMS:
            zc = train_cat(Xtr, Xva, y2, w2, init_tr, cats, tag, fold, t0)
            solo_cat = sc(zc, solo=True)
            r_lgb = np.corrcoef(zc[mm], z_lgb[mm])[0, 1]
            r_bl = np.corrcoef(zc[mm], z_blend[mm])[0, 1]
            deltas = [sc((1 - wt) * z_blend + wt * zc) - base for wt in WEIGHTS]
            out[(fold, name)] = (solo_cat, solo_cat - solo_lgb, r_lgb, r_bl, deltas)
            print(f"    {name:<24s} solo {solo_cat:7.1f} (핸디캡 {solo_cat-solo_lgb:+7.1f})"
                  f"  corr(lgb) {r_lgb:.3f}  corr(blend) {r_bl:.3f}")
            print(f"    {'':<24s} 블렌드 Δ: "
                  + "  ".join(f"w={wt:.2f}:{d:+6.1f}" for wt, d in zip(WEIGHTS, deltas)))

    print("\n" + "=" * 104)
    print("판정 — CatBoost 를 현 배포 블렌드(MLP 75% / LGBM 25%)에 얹었을 때")
    print("=" * 104)
    print(f"  {'팔':<26s}{'fold':>6s}{'solo':>9s}{'핸디캡':>9s}"
          f"{'corr(lgb)':>11s}{'corr(blend)':>13s}{'최선 Δ':>10s}{'@w':>6s}")
    for name, _t, _c in ARMS:
        for fold in D.FOLDS:
            sc_, hc, rl, rb, ds = out[(fold, name)]
            i = int(np.argmax(ds))
            print(f"  {name:<26s}{fold:>6d}{sc_:>9.1f}{hc:>+9.1f}"
                  f"{rl:>11.3f}{rb:>13.3f}{ds[i]:>+10.1f}{WEIGHTS[i]:>6.2f}")
    print("\n  §7.14 재현 기준: corr(lgb) ~ 0.985 이면 그때와 같은 자리다.")
    print("  채택: 양 fold 같은 w 에서 Δ>0 이고 시드 노이즈(std 4~9) 초과.")
    print("  사전 예측: lowcard 는 팀 CTR 이 전이 안 돼 죽거나 해롭다(§7.138).")
    print("            allid 는 크게 음수여야 정상(§7.12 −101 재현). 아니면 하네스 오류.")
    print("            진짜 열린 칸은 corr(blend) 다 — 0.985 보다 낮으면 여지가 있다.")


if __name__ == "__main__":
    main()
