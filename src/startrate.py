"""§7.211 — `start_rate` (§7.68) 를 **학습 경로**로 처음 잰다. 챔피언 등가 fold 층.

§7.68 판정: 잔차 t **f2023 −0.36 / f2024 +3.12** (대조군 −1.17/−0.91), corr 0.55 with
`asof_pitcher_n` → *"커리어 투구 수가 곧 역할을 인코딩한다. 모델이 이미 갖고 있다"* 로 폐쇄.
문서에 *"판정에 학습이 한 번도 필요 없었고 1분이 들었다"* 고 적혀 있다.

★ 재개봉 사유 셋
  ① **학습 경로가 없다.** 잔차 프로브는 **주변 상관**만 본다 — 이미 쓰이는 피처와의
     **교호에서만** 값을 내는 축은 t 가 0 이어도 재학습하면 이득이 난다
     ([[test-new-feature-against-model-residual]]).
  ② 닫은 논거가 **정보 중복**인데 그 논증 종류를 §7.136 이 반증했다 — `base_state` 와
     주자 3플래그는 정확한 전단사인데 트리 표현 비용이 1회 vs 7회로 갈렸다. §7.93 은
     **비트 동일** 중복 컬럼을 뺐다가 −5.57 을 맞았다
     ([[information-redundancy-isnt-representation-redundancy]]).
  ③ f2023 −0.36 / f2024 +3.12 — 비대칭비 0.12 로, 이긴 사례들(0.32)보다 좋다.

⚠️ 반대 근거: 이건 **컬럼 추가**다. `config.py` 기록 — 전이 배율이 얼려지는 파라미터 수에
   단조로 붙고 (β 0.93 · δ표 3.3 · 트리 분할 68.6), **컬럼 추가는 4전 4패**다.

두 경로를 같이 잰다
  [A] 사후 β — 재학습 0회. 중심화한 `start_rate` 를 로짓에 β 로 더한다. β × α 격자.
  [B] 학습 경로 — 컬럼을 넣고 LGBM 12 + MLP 8(dim 8) 을 fold 로 재학습, 짝지어 비교 + α 스윕.
      A 가 0 인데 B 가 양수면 **정확히 교호 성분**이다 (이 실험의 요점).

실행: .venv/Scripts/python.exe -u src/startrate.py 2>&1 | tee experiments/startrate.log
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

import deploy29 as D
import foldcache
import hillclimb as H
import mlp
import modelabel
import rolefeat
import soften
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA,
                    MLP_AUX_LAMBDA, MLP_SEEDS, N_ROUNDS, SOFT_LABEL_F, TRAIN_CSV)
from data import load_csv
from foldchamp import variants
from initfix3 import level_init
from validation import bss, shift_to_prior

CACHE = D.CACHE
ARM = {2023: "D4", 2024: "D8"}          # §7.210 각 fold 최선 팔 (기준선)
DIM = {2023: 4, 2024: 8}
ALPHAS = [0.65, 0.70, 0.75, 0.80, 0.85]
BETAS = [-1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5]


def parts(fold, df, s, y_all):
    """기준 블렌드 부품 (배포 등가 B1)."""
    st = D.state(df, s, y_all, fold, check=False)
    mm, ax, yv = st["mm"], st["Xax"] @ st["dep"], st["yv"]
    zl = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"][mm]
    cat = np.mean([np.load(CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz")["z"]
                   for v in H.CAT_VARIANTS for sd in H.CAT_SEEDS], axis=0)[mm]
    noid = np.load(CACHE / f"champ_noid_f{fold}.npz")["z"]
    return st, mm, ax, yv, zl, cat, noid


def mlp_pool(fold, arm, noid, mm):
    e = [np.load(CACHE / f"champ_{arm}_f{fold}_emb4_{sd}.npz")["z"] for sd in MLP_SEEDS]
    return (0.875 * np.mean(e, axis=0) + 0.125 * np.mean(noid, axis=0))[mm]


def post_hoc(df, s, y_all):
    print("=" * 92)
    print("[A] 사후 β 경로 — 재학습 0회 (주변 성분만 산다)")
    print("=" * 92)
    for fold in (2023, 2024):
        st, mm, ax, yv, zl, cat, noid = parts(fold, df, s, y_all)
        r = float(yv.mean())
        zm = mlp_pool(fold, ARM[fold], noid, mm)
        d = df.iloc[st["vi"][mm]].copy().reset_index(drop=True)
        role = rolefeat.fit_role(df[s < fold])
        x = rolefeat.apply_role(d, role)[0]["start_rate"].astype(np.float64)
        x = x - x.mean()
        rng = np.random.default_rng(0)
        dsh = d.copy()
        dsh["pitcher_id"] = rng.permutation(d["pitcher_id"].to_numpy())
        xs = rolefeat.apply_role(dsh, role)[0]["start_rate"].astype(np.float64)
        xs = xs - xs.mean()

        def blend(a):
            return (1 - H.W_CAT) * ((1 - a) * zl + a * zm) + H.W_CAT * cat + ax

        def sc(z):
            return bss(yv, shift_to_prior(1 / (1 + np.exp(-z)), r))

        print(f"\nfold {fold}  팔 {ARM[fold]}  sd(start_rate) {x.std():.4f}")
        head = "  alpha\\beta"
        print(head + "".join(f"{b:>8.2f}" for b in BETAS) + f"{'기준':>10}")
        for a in ALPHAS:
            z0 = blend(a)
            b0 = sc(z0)
            cells = "".join(f"{sc(z0 + b * x) - b0:>+8.2f}" for b in BETAS)
            print(f"  {a:<10.2f}" + cells + f"{b0:>10.2f}")
        z0 = blend(MLP_ALPHA)
        b0 = sc(z0)
        row = [sc(z0 + b * x) - b0 for b in BETAS]
        i = int(np.argmax(row))
        pl = sc(z0 + BETAS[i] * xs) - b0
        print(f"  배포 alpha: beta*={BETAS[i]:+.2f}  Δ={row[i]:+.2f}  위약 {pl:+.2f}")


def learn_path(df, s):
    print("\n" + "=" * 92)
    print("[B] 학습 경로 — `start_rate` 컬럼 추가 후 LGBM 12 + MLP 8 재학습")
    print("=" * 92)
    t0 = time.time()
    for fold in (2023, 2024):
        pl = CACHE / f"sr_lgb_f{fold}.npz"
        V = variants(DIM[fold])
        done = all((CACHE / f"sr_mlp_f{fold}_{v}_{sd}.npz").exists()
                   for v, (_, n) in V.items() for sd in MLP_SEEDS[:n])
        if pl.exists() and done:
            print(f"  fold {fold} 캐시 적중")
            continue
        print(f"\n  === fold {fold} · dim {DIM[fold]} ===")
        Xtr, ytr, w, Xva, yva, art, tr = foldcache.fold_data(df, fold)
        role = rolefeat.fit_role(tr)
        Xtr = Xtr.copy()
        Xva = Xva.copy()
        Xtr["start_rate"] = rolefeat.apply_role(tr, role)[0]["start_rate"].astype(np.float32)
        Xva["start_rate"] = rolefeat.apply_role(df[s == fold], role)[0][
            "start_rate"].astype(np.float32)
        print(f"    컬럼 {Xtr.shape[1]}개 (기존 +1)")
        aux, _ = modelabel.aux_labels(tr)
        y_raw = tr["control_success"].to_numpy(dtype=np.float64)
        ss = tr["season"].to_numpy(dtype=np.int64)
        gg = tr["game_type"].astype("object").to_numpy().astype(str)
        if SOFT_LABEL_F:
            Xtr, ytr, w, aux, _ = soften.soften(tr, Xtr, ytr, w, aux)
        # soften 이 복제한 행만큼 (season, game_type) 도 늘려야 init_score 가 맞는다
        hh = soften.fit_h(tr)
        h_row = np.zeros(len(tr))
        for yr, v in hh.items():
            h_row[(ss == yr) & (gg == "F")] = v
        dup = (h_row > 0) & (y_raw == 1)
        ss2 = np.concatenate([ss, ss[dup]])
        gg2 = np.concatenate([gg, gg[dup]])

        if not pl.exists():
            # ⚠️ 기준선 `initdep_fold{f}_ctr0.npz` 는 **init_score(soften 후 레벨)** 로
            #    학습됐다 (`initdeploy.train_lgb:107`). 그걸 빼면 두 팔이 레시피가 달라져
            #    `start_rate` 효과에 init_score 제거분이 섞인다
            #    ([[training-side-correction-cleaner-than-inference-is-skew]]).
            #    `hillclimb.train_lgb_members` 와 같은 경로로 맞춘다.
            init_tr = level_init(ss2, gg2, ytr, w)
            assert len(init_tr) == len(Xtr)
            z, n = np.zeros(len(Xva)), 0
            for over in ENSEMBLE_VARIANTS.values():
                for sd in foldcache.LGB_SEEDS:
                    p = dict(FINAL_LGB_PARAMS, **over, seed=sd, bagging_seed=sd,
                             feature_fraction_seed=sd, data_random_seed=sd)
                    b = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w,
                                                 init_score=init_tr),
                                  num_boost_round=N_ROUNDS)
                    z += b.predict(Xva, raw_score=True)
                    n += 1
                    del b
            np.savez_compressed(pl, z=z / n)
            print(f"    lgbm {n}개 단독 {mlp.score(z / n, yva):.1f} ({time.time()-t0:.0f}s)")

        A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
        for vname, (cfg, nseed) in V.items():
            for sd in MLP_SEEDS[:nseed]:
                f = CACHE / f"sr_mlp_f{fold}_{vname}_{sd}.npz"
                if f.exists():
                    continue
                _, curve = mlp.train_mlp(A, C, ytr, w, B, Cv, sizes, seed=sd,
                                         verbose=False, y_aux=aux,
                                         **dict(cfg, aux=MLP_AUX_LAMBDA))
                np.savez_compressed(f, z=curve[-1])
                print(f"    mlp {vname}_{sd} 단독 {mlp.score(curve[-1], yva):.1f}"
                      f"  ({time.time()-t0:.0f}s)")
                gc.collect()
        del Xtr, Xva, A, C, B, Cv
        gc.collect()


def report(df, s, y_all):
    print("\n" + "=" * 92)
    print("[B] 판정 — 블렌드 Δ (start_rate 있음 − 없음) · alpha 스윕")
    print("=" * 92)
    for fold in (2023, 2024):
        st, mm, ax, yv, zl, cat, noid = parts(fold, df, s, y_all)
        r = float(yv.mean())
        zl2 = np.load(CACHE / f"sr_lgb_f{fold}.npz")["z"][mm]
        e2 = [np.load(CACHE / f"sr_mlp_f{fold}_emb4_{sd}.npz")["z"] for sd in MLP_SEEDS]
        n2 = [np.load(CACHE / f"sr_mlp_f{fold}_noid_{sd}.npz")["z"]
              for sd in MLP_SEEDS[:3]]
        zm2 = (0.875 * np.mean(e2, axis=0) + 0.125 * np.mean(n2, axis=0))[mm]
        zm1 = mlp_pool(fold, ARM[fold], noid, mm)

        def sc(z):
            return bss(yv, shift_to_prior(1 / (1 + np.exp(-z)), r))

        print(f"\n  fold {fold}")
        print(f"    {'alpha':>7}{'기준':>10}{'+start_rate':>13}{'Δ':>9}")
        for a in ALPHAS:
            b1 = sc((1 - H.W_CAT) * ((1 - a) * zl + a * zm1) + H.W_CAT * cat + ax)
            b2 = sc((1 - H.W_CAT) * ((1 - a) * zl2 + a * zm2) + H.W_CAT * cat + ax)
            mk = "  <- 배포" if a == MLP_ALPHA else ""
            print(f"    {a:>7.2f}{b1:>10.2f}{b2:>13.2f}{b2 - b1:>+9.2f}{mk}")
        print(f"    LGBM 단독  기준 {mlp.score(zl, yv):.1f}"
              f"  +start_rate {mlp.score(zl2, yv):.1f}"
              f"   Δ {mlp.score(zl2, yv) - mlp.score(zl, yv):+.2f}")
        print(f"    MLP  단독  기준 {mlp.score(zm1, yv):.1f}"
              f"  +start_rate {mlp.score(zm2, yv):.1f}"
              f"   Δ {mlp.score(zm2, yv) - mlp.score(zm1, yv):+.2f}")


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y_all = df["control_success"].to_numpy(dtype=np.float64)
    post_hoc(df, s, y_all)
    learn_path(df, s)
    report(df, s, y_all)


if __name__ == "__main__":
    main()
