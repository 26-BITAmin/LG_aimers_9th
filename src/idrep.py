"""§7.101 6차 — ID 표현 3종 (사용자 지적: "ID를 임베딩하고 넣어야 하는 거 아닌가")

지적의 요지
    4차의 "정제" 팔은 pitcher_id/batter_id를 **그냥 버렸다**. 그건 위생이지 표현이
    아니다. 배포 MLP는 ID 임베딩(emb4)에서 실제로 이득을 보고 있으므로, 드랍은
    신호를 버리는 조작이다. 트랜스포머에 맞는 표현은 **임베딩**이다.

왜 원시 ID가 트랜스포머에서 죽는가
    TabPFN/TabICL은 수치 피처 벡터를 받는다. pitcher_id=23143은 "23,143 크기의
    1차원 축"으로 읽히고, 선수 번호 순서에는 의미가 없다. LGBM은 축 정렬 분기로
    ID 구간을 격리할 수 있어 같은 입력에도 멀쩡하다 — 그래서 4차에서 ID를 빼자
    TabPFN만 +52.3 올랐다.

⚠️ 리키지 주의 — 배포 npz를 쓰면 안 된다
    submit/model/mlp_emb4_*.npz의 임베딩은 2019~2024 **전체**로 학습됐다.
    검증이 2024인 이 실험에 쓰면 임베딩이 2024 라벨을 본 상태가 된다.
    그래서 train<2024로 MLP를 **새로 학습해** emb 테이블을 뽑는다.

설계 — ID 표현 × 학습기 (2×3, 쌍체)
    표현  raw   원시 정수 ID 2열            (LGBM의 현행 방식)
          drop  ID 2열 제거
          emb   fold2024 MLP의 4+4차원 임베딩으로 치환
    학습기 LGBM F전문가 / TabPFN ctx10k est1

    각 학습기를 **자기 최적 형태**로 둔다: LGBM은 원본 수치(NaN을 자체 처리하므로
    대치하면 오히려 손해 — §7.92의 결측 플래그 교훈), TabPFN은 위생 적용
    (카운트 3열 log1p + NaN 중앙값 대치). 표현만 3×로 바꾼다.

    핵심 대조: emb가 TabPFN만 올리면 "표현 문제였다". **둘 다 올리면** 그건
    표현이 아니라 그냥 더 나은 피처고, 그럼 배포 LGBM에 넣는 게 답이다.

실행:  .venv-gpu/Scripts/python.exe -u src/idrep.py 2>&1 | tee experiments/idrep.log
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
from config import (CONTAM_AXES, DROP_FEATURES, MLP_AUX_LAMBDA, MLP_VARIANTS,
                    RECENCY_HALF_LIFE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from validation import bss, shift_to_prior

ACC_N = 8000
SEEDS = (42, 7, 2024)
BIG_COLS = ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n")
ID_COLS = ("pitcher_id", "batter_id")
CTX = 10_000

SMALL_LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15,
                 min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7,
                 bagging_freq=1, lambda_l2=10.0, max_bin=127, num_threads=6,
                 verbosity=-1)


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

    Xtr_df, XvaF_df = X(trF), X(vaF).iloc[acc_i]
    cols = list(Xtr_df.columns)
    Xtr = Xtr_df.to_numpy(dtype=np.float32)
    Xq = XvaF_df.to_numpy(dtype=np.float32)
    ytr = trF[TARGET_COL].to_numpy(dtype=np.int64)
    ya = vaF[TARGET_COL].to_numpy(dtype=np.float64)[acc_i]
    ref = ya.mean() * (1 - ya.mean())

    # ── fold2024 임베딩 학습 (train<2024 전체, 리키지 없음) ──
    t0 = time.time()
    Xfull = X(tr)
    A, C, B, Cq, _, sizes = mlp_mod.encode(Xfull, XvaF_df, art)
    Atr_f, Ctr_f, _, Ctrf, _, _ = mlp_mod.encode(Xfull, Xtr_df, art)
    cfg = dict(MLP_VARIANTS["emb4"][0], aux=MLP_AUX_LAMBDA)
    cfg.pop("aux", None)                      # 보조 헤드는 임베딩 추출에 불필요
    m, _ = mlp_mod.train_mlp(A, C, tr[TARGET_COL].to_numpy(dtype=np.float64),
                             w_all, B, Cq, sizes, seed=42, verbose=False, **cfg)
    wts = mlp_mod.export_weights(m)
    ec = [str(x) for x in wts["ecols"]]
    ip, ib = ec.index("pitcher_id"), ec.index("batter_id")
    Ep, Eb = wts[f"emb{ip}"], wts[f"emb{ib}"]
    print(f"  임베딩 학습 완료  투수 {Ep.shape}  타자 {Eb.shape}  ({time.time()-t0:.0f}s)")
    del Xfull, A, B

    def emb_of(Ccat):
        p = np.clip(Ccat[:, ip].astype(np.int64), 0, len(Ep) - 1)
        b = np.clip(Ccat[:, ib].astype(np.int64), 0, len(Eb) - 1)
        return np.hstack([Ep[p], Eb[b]]).astype(np.float32)

    keep = [i for i, c in enumerate(cols) if c not in ID_COLS]
    big = [j for j, i in enumerate(keep) if cols[i] in BIG_COLS]
    med = np.nanmedian(Xtr[:, keep], axis=0)

    def variant(Z, Ccat, rep, hygiene):
        W = Z[:, keep].copy()
        if hygiene:
            W[:, big] = np.log1p(np.maximum(np.nan_to_num(W[:, big], nan=0.0), 0.0))
            bad = ~np.isfinite(W)
            if bad.any():
                W[bad] = np.take(med, np.nonzero(bad)[1])
        if rep == "raw":
            W = np.hstack([W, Z[:, [cols.index(c) for c in ID_COLS]]])
        elif rep == "emb":
            W = np.hstack([W, emb_of(Ccat)])
        return W

    def lgb_run(Ztr, Zq):
        month = trF["game_month"].to_numpy(dtype=np.float64)
        cut = np.quantile(month, 0.8); itr, iva = month <= cut, month > cut
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

    def pfn_run(Ztr, Zq):
        i = np.random.default_rng(42).choice(len(Ztr), size=CTX, replace=False)
        c = TabPFNClassifier(device="cuda", n_estimators=1, random_state=42,
                             inference_precision="autocast", ignore_pretraining_limits=True)
        c.fit(Ztr[i], ytr[i])
        return c.predict_proba(Zq)[:, 1]

    base = None
    print(f"\n  {'학습기':<10}{'ID 표현':<8}{'열':>5}{'subBSS':>10}{'Δ vs LGBM raw':>16}{'SE':>8}{'소요':>8}")
    for kind, fn, hyg in (("LGBM", lgb_run, False), ("TabPFN", pfn_run, True)):
        for rep in ("raw", "drop", "emb"):
            Ztr = variant(Xtr, Ctrf, rep, hyg)
            Zq = variant(Xq, Cq, rep, hyg)
            t0 = time.time()
            p = shift_to_prior(fn(Ztr, Zq), ya.mean())
            el = time.time() - t0
            s = bss(ya, p)
            if base is None:
                base, base_p = s, p
                print(f"  {kind:<10}{rep:<8}{Ztr.shape[1]:>5}{s:>10.1f}{'(기준)':>16}{'':>8}{el:>7.0f}s")
                continue
            d = (base_p - ya) ** 2 - (p - ya) ** 2
            mm = 1e5 * d.mean() / ref
            se = 1e5 * d.std(ddof=1) / np.sqrt(len(d)) / ref
            print(f"  {kind:<10}{rep:<8}{Ztr.shape[1]:>5}{s:>10.1f}{mm:>+16.1f}{se:>8.1f}{el:>7.0f}s")


if __name__ == "__main__":
    main()
