"""§7.101 7차 — 임베딩을 **여기서** 학습한다 (사용자 지적)

지적
    "MLP의 임베딩을 써야 하는 게 아니라 여기에서 또 학습을 따로 시켜줘야 하는 거 아님?"

맞는 부분
    전역 MLP의 투수 4차원은 **전역 과제**(R+F 122만행, 2019–23)에 최적화된 표현이다.
    F 전문가가 필요로 하는 표현과 다를 수 있고, 남의 표현을 빌리면 "TabPFN이 좋은가"와
    "MLP 임베딩이 좋은가"가 섞인다.

충돌하는 제약
    F 전문가의 데이터는 25,686행 · 투수 793명 = **투수당 32투구**. 거기서 4차원을
    학습하면 과적합 구간이다. 과제를 맞추려고 데이터를 48배 버리는 셈.
        emb_G  전역 MLP 1,221,585행   데이터 풍부 · 과제 불일치   ← 6차(idrep.py)에서 측정
        emb_F  2023 F만    25,686행   과제 일치 · 데이터 기아     ← 여기

스택 함정과 그 처리
    emb_F를 2023 F로 학습해 같은 2023 F 전문가에게 먹이면, **학습 행은 자기 라벨을 본
    임베딩**을 받고 검증 행(2024 F)은 정직한 임베딩을 받는다. 리키지는 아니지만
    train/serve 불일치라 스택 피처가 무너지는 전형적 경로다. 그래서 두 갈래:

    (a) MLP F전문가       전문가가 임베딩을 **직접** 학습. 스택 자체가 없다.
    (b) TabPFN + emb_F(OOF)  2023 F를 5폴드로 갈라 각 폴드 행에는 그 폴드를 안 본
                             임베딩을 준다. 검증 행에는 전량 학습 임베딩.

전부 같은 8,000행(2024 F)에서 LGBM F전문가 raw와 **쌍체** 비교.

실행:  .venv-gpu/Scripts/python.exe -u src/idrep2.py 2>&1 | tee experiments/idrep2.log
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
import truststore

truststore.inject_into_ssl()

import mlp as mlp_mod
from config import (CONTAM_AXES, DROP_FEATURES, MLP_VARIANTS, RECENCY_HALF_LIFE,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV,
                    USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import bss, shift_to_prior

ACC_N = 8000
SEEDS = (42, 7, 2024)
K_OOF = 5
CTX = 10_000
BIG_COLS = ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n")
ID_COLS = ("pitcher_id", "batter_id")

SMALL_LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15,
                 min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7,
                 bagging_freq=1, lambda_l2=10.0, max_bin=127, num_threads=6,
                 verbosity=-1)


def emb_tables(A, C, y, w, B, Cv, sizes, cfg, epochs, seed=42):
    """MLP 하나 학습 → (투수표, 타자표). 여기서만 임베딩이 만들어진다."""
    m, _ = mlp_mod.train_mlp(A, C, y, w, B, Cv, sizes, seed=seed, verbose=False,
                             **dict(cfg, epochs=epochs))
    wt = mlp_mod.export_weights(m)
    ec = [str(x) for x in wt["ecols"]]
    return wt[f"emb{ec.index('pitcher_id')}"], wt[f"emb{ec.index('batter_id')}"], m


def main():
    from tabpfn import TabPFNClassifier

    df = load_csv(TRAIN_CSV)
    tr = df[df.season < 2024]
    va = df[df.season == 2024]
    w_all = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=2023)
    art = fit_artifacts(tr, weights=w_all, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ID_COLS:
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    trF = tr[(tr.season == 2023) & (tr.game_type.astype(str) == "F")]
    vaF = va[va.game_type.astype(str) == "F"]
    rng = np.random.default_rng(42)
    acc_i = rng.permutation(len(vaF))[:ACC_N]

    Xtr_df, Xq_df = X(trF), X(vaF).iloc[acc_i]
    cols = list(Xtr_df.columns)
    Xtr, Xq = Xtr_df.to_numpy(dtype=np.float32), Xq_df.to_numpy(dtype=np.float32)
    ytr = trF[TARGET_COL].to_numpy(dtype=np.int64)
    ya = vaF[TARGET_COL].to_numpy(dtype=np.float64)[acc_i]
    ref = ya.mean() * (1 - ya.mean())
    n = len(Xtr)

    # 인코더는 **F 전문가의 세계**(2023 F)로 적합한다 — 이 모델이 아는 선수가 그게
    # 전부다. 2024 F에만 나오는 선수는 코드 0(미지)으로 떨어진다.
    Atr, Ctr, Aq, Cq, _, sizes = mlp_mod.encode(Xtr_df, Xq_df, art)
    wtr = np.ones(n)
    cfg = dict(MLP_VARIANTS["emb4"][0])
    cfg.pop("aux", None)

    month = trF["game_month"].to_numpy(dtype=np.float64)
    cut = np.quantile(month, 0.8)
    itr, iva = month <= cut, month > cut

    keep = [i for i, c in enumerate(cols) if c not in ID_COLS]
    big = [j for j, i in enumerate(keep) if cols[i] in BIG_COLS]
    med = np.nanmedian(Xtr[:, keep], axis=0)

    def hyg(Z):
        W = Z[:, keep].copy()
        W[:, big] = np.log1p(np.maximum(np.nan_to_num(W[:, big], nan=0.0), 0.0))
        bad = ~np.isfinite(W)
        if bad.any():
            W[bad] = np.take(med, np.nonzero(bad)[1])
        return W

    def lgb_run(Ztr, Zq):
        ps = []
        for s in SEEDS:
            prm = dict(SMALL_LGB, seed=s, bagging_seed=s, feature_fraction_seed=s,
                       data_random_seed=s)
            b0 = lgb.train(prm, lgb.Dataset(Ztr[itr], ytr[itr].astype(np.float64)),
                           num_boost_round=3000,
                           valid_sets=[lgb.Dataset(Ztr[iva], ytr[iva].astype(np.float64))],
                           callbacks=[lgb.early_stopping(100, verbose=False)])
            b = lgb.train(prm, lgb.Dataset(Ztr, ytr.astype(np.float64)),
                          num_boost_round=max(50, b0.best_iteration))
            ps.append(1 / (1 + np.exp(-b.predict(Zq, raw_score=True))))
        return np.mean(ps, axis=0)

    out = {}
    t0 = time.time()
    out["LGBM F전문가 raw (기준)"] = shift_to_prior(lgb_run(Xtr, Xq), ya.mean())
    print(f"  기준 준비 {time.time()-t0:.0f}s")

    # ── (a) MLP F전문가 — 임베딩을 자기가 학습 ───────────────
    # train_mlp은 (모델, [에폭별 valid 로짓])을 준다. 예측 함수를 따로 부르지 않고
    # 평가셋을 valid로 넘겨 마지막 에폭 로짓을 쓴다 (배포 경로와 같은 관례).
    t0 = time.time()
    _, curves = mlp_mod.train_mlp(Atr[itr], Ctr[itr], ytr[itr].astype(np.float64),
                                  wtr[itr], Atr[iva], Ctr[iva], sizes, seed=42,
                                  verbose=False, **cfg)
    yin = ytr[iva].astype(np.float64)
    best_ep = 1 + int(np.argmax([bss(yin, shift_to_prior(1 / (1 + np.exp(-z)), yin.mean()))
                                 for z in curves]))
    ps = []
    for s_ in SEEDS:
        _, cv = mlp_mod.train_mlp(Atr, Ctr, ytr.astype(np.float64), wtr, Aq, Cq, sizes,
                                  seed=s_, verbose=False, **dict(cfg, epochs=best_ep))
        ps.append(1 / (1 + np.exp(-cv[-1])))
    out[f"(a) MLP F전문가 (ep={best_ep})"] = shift_to_prior(np.mean(ps, axis=0), ya.mean())
    print(f"  (a) MLP F전문가 {time.time()-t0:.0f}s   내부검증 최적 에폭 {best_ep}")

    # ── (b) TabPFN + emb_F (OOF) ────────────────────────────
    t0 = time.time()
    fold = rng.permutation(n) % K_OOF
    Ep_full, Eb_full, _ = emb_tables(Atr, Ctr, ytr.astype(np.float64), wtr, Aq, Cq,
                                     sizes, cfg, best_ep)
    Etr = np.zeros((n, Ep_full.shape[1] + Eb_full.shape[1]), dtype=np.float32)
    for k in range(K_OOF):
        m_ = fold != k
        Ep, Eb, _ = emb_tables(Atr[m_], Ctr[m_], ytr[m_].astype(np.float64), wtr[m_],
                               Atr[~m_], Ctr[~m_], sizes, cfg, best_ep)
        pc = np.clip(Ctr[~m_][:, 0].astype(np.int64), 0, len(Ep) - 1)
        bc = np.clip(Ctr[~m_][:, 1].astype(np.int64), 0, len(Eb) - 1)
        Etr[~m_] = np.hstack([Ep[pc], Eb[bc]])
    pcq = np.clip(Cq[:, 0].astype(np.int64), 0, len(Ep_full) - 1)
    bcq = np.clip(Cq[:, 1].astype(np.int64), 0, len(Eb_full) - 1)
    Eq = np.hstack([Ep_full[pcq], Eb_full[bcq]]).astype(np.float32)
    print(f"  OOF 임베딩 {time.time()-t0:.0f}s")

    Ztr, Zq = np.hstack([hyg(Xtr), Etr]), np.hstack([hyg(Xq), Eq])
    i = np.random.default_rng(42).choice(n, size=min(CTX, n), replace=False)
    c = TabPFNClassifier(device="cuda", n_estimators=1, random_state=42,
                         inference_precision="autocast", ignore_pretraining_limits=True)
    c.fit(Ztr[i], ytr[i])
    out["(b) TabPFN + emb_F(OOF)"] = shift_to_prior(c.predict_proba(Zq)[:, 1], ya.mean())
    out["(b') LGBM + emb_F(OOF)"] = shift_to_prior(lgb_run(Ztr, Zq), ya.mean())

    base = out["LGBM F전문가 raw (기준)"]
    print(f"\n  {'팔':<30}{'subBSS':>10}{'Δ vs 기준':>12}{'쌍체SE':>9}{'t':>7}")
    for k, p in out.items():
        d = (base - ya) ** 2 - (p - ya) ** 2
        m_, se = 1e5 * d.mean() / ref, 1e5 * d.std(ddof=1) / np.sqrt(len(d)) / ref
        tag = "(기준)" if k.startswith("LGBM F전문가 raw") else f"{m_:+.1f}"
        t = "" if k.startswith("LGBM F전문가 raw") else f"{m_/max(se,1e-9):.2f}"
        print(f"  {k:<30}{bss(ya, p):>10.1f}{tag:>12}{('' if not t else f'{se:.1f}'):>9}{t:>7}")


if __name__ == "__main__":
    main()
