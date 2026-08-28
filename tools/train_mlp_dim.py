"""배포 MLP `emb4` 멤버를 임의의 `pitcher_id` 임베딩 차원으로 재학습한다.

경로 선택 (원우 §7.175 `t1core.py` 의 방침을 그대로 따른다)
    "남의 코드가 필요 없다. **T1 자신의 features.py + artifacts.joblib** 으로
    피처를 만든다." — 그래서 피처·인코더는 **출하 zip 것을 그대로** 쓰고,
    학습 루프만 원우 `src/mlp.py` 에서 가져온다.

⚠️ 인코더를 다시 적합하지 않는다
    `bundle["mlp_enc"]` 를 그대로 쓴다. 재적합하면 범주 인덱스가 어긋나 배포
    임베딩과 행이 안 맞고, 그 순간 비교 자체가 무의미해진다. 바꾸는 것은
    `emb_dim["pitcher_id"]` **하나뿐**이다.

학습 레시피 (meta.json + artifacts.joblib 교차검증)
    전 시즌 2019~2024 · recency_half_life 0.5 (ref 2024)
    보조 라벨 middle·reverse 복원 (modelabel) · aux_lambda 1.0
    soft label F (soften) · init_score = (시즌 x game_type) 가중 양성률 로짓
    emb4: epochs 9 · hid 128 · lr 2e-3 · dropout 0.3 · wd 1e-2
    seeds 42, 7, 2024, 1234, 31337

⚠️ meta.json 은 믿지 않는다 — `emb_dim.pitcher_id` 가 4 로 적혀 있으나 실제 npz 는 8
   이고, `mlp_variant_weight` 도 어긋난다 (원우 §7.193 이 T1 계보에서 고친 그 문제가
   재발했다). 값은 `artifacts.joblib` 과 npz 로 확인하고 쓴다.
"""
import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

TRAIN_CSV = "/Users/hyejin/Aimers_9th/data/train.csv"


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def season_gt_level(ss, gg, y, w, min_n=500):
    """(시즌 x game_type) 가중 양성률의 로짓 — 행별 init_score (원우 train.py:377)."""
    ug = np.unique(gg)
    k = ss * len(ug) + np.searchsorted(ug, gg)
    out = np.empty(len(k), dtype=np.float64)
    gm = float(np.average(y, weights=w))
    for kk in np.unique(k):
        m = k == kk
        out[m] = float(np.average(y[m], weights=w[m])) if m.sum() >= min_n else gm
    return _logit(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="학습 모듈 + 출하 features.py 가 있는 디렉터리")
    ap.add_argument("--zip-dir", required=True, help="풀어 둔 챔피언 zip")
    ap.add_argument("--dim", type=int, required=True, help="pitcher_id 임베딩 차원")
    ap.add_argument("--out", required=True, help="npz 를 쓸 디렉터리")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    sys.path.insert(0, a.work)
    os.makedirs(a.out, exist_ok=True)

    import mlp
    import modelabel
    import soften
    from data import recency_weight
    from features import build_features
    from mlp_infer import encode_apply

    mlp.THREADS = a.threads

    t0 = time.time()
    bundle = joblib.load(os.path.join(a.zip_dir, "model", "artifacts.joblib"))
    art, cols, enc = bundle["artifacts"], bundle["feature_cols"], bundle["mlp_enc"]
    # 학습 레시피는 meta.json 에만 있다 (artifacts.joblib 은 추론 상수만 담는다).
    # meta 의 **배포 상수**(emb_dim · variant_weight · fspec_alpha · ctx_axes)는 낡았지만,
    # 학습 레시피 항목은 C-4 가 건드리지 않았으므로 유효하다. art 에 있으면 art 를 우선한다.
    import json
    meta = json.load(open(os.path.join(a.zip_dir, "model", "meta.json"), encoding="utf-8"))
    def cfgv(k, default=None):
        return art[k] if k in art else meta.get(k, default)

    # 배포 emb4 설정을 npz 에서 읽어 확인한다 (meta.json 이 아니라)
    ref = np.load(os.path.join(a.zip_dir, "model", "mlp_emb4_42.npz"))
    print(f"배포 emb4 임베딩 모양: {[ref[f'emb{i}'].shape for i in range(len(ref['ecols']))]}")

    print("데이터 적재 …")
    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    y = df["control_success"].to_numpy(dtype=np.float32)
    w = recency_weight(df["season"].to_numpy(), cfgv("recency_half_life"),
                       ref_season=cfgv("ref_season", 2024))
    print(f"  {len(df):,}행 (배포 기록 train_rows = {art.get('train_rows')})")

    print("피처 생성 (출하 features.py + 동결 artifacts) …")
    X = build_features(df, art)
    if art.get("use_player_ids"):
        for c in ("pitcher_id", "batter_id"):
            X[c] = df[c].to_numpy(dtype=np.float32)
    X = X[cols]
    print(f"  {X.shape[1]}피처 (배포 기록 n_features = {art.get('n_features')})")

    print("보조 라벨 복원 …")
    aux, cov = modelabel.aux_labels(df)
    print(f"  복원률 {cov:.1%}  축 {aux.shape[1]}개")

    ss = df["season"].to_numpy(dtype=np.int64)
    gt = df["game_type"].astype("object").to_numpy().astype(str)
    if cfgv("soft_label_f"):
        n0 = len(y)
        h_pre = soften.fit_h(df)
        h_row = np.zeros(len(df))
        for yr, v in h_pre.items():
            h_row[(ss == yr) & (gt == "F")] = v
        dup = (h_row > 0) & (y == 1)
        X, y, w, aux, soft_h = soften.soften(df, X, y, w, aux)
        ss = np.concatenate([ss, ss[dup]])
        gt = np.concatenate([gt, gt[dup]])
        print(f"  soft label +{len(y)-n0:,}행 복제")
        ref_h = cfgv("soft_h") or {}
        for k in sorted(soft_h):
            r = ref_h.get(k, ref_h.get(str(k)))
            flag = "일치" if r is not None and abs(soft_h[k] - r) < 1e-9 else "⚠️ 불일치"
            print(f"    h[{k}] = {soft_h[k]:.12f}   배포 {r}   {flag}")

    offset = season_gt_level(ss, gt, y, w) if cfgv("use_init_score") else None
    print(f"  init_score {'ON' if offset is not None else 'OFF'}")

    print("인코딩 (동결 mlp_enc 재사용 — 재적합 안 함) …")
    A, C = encode_apply(X, enc)
    sizes = list(enc["sizes"])
    print(f"  A {A.shape}  C {C.shape}  sizes {sizes}")

    cfg = dict(epochs=9, hid=128, lr=2e-3, dropout=0.3, wd=1e-2, use_id_emb=True,
               emb_dim={"pitcher_id": a.dim, "batter_id": 4, "pitcher_team_id": 4,
                        "batter_team_id": 4, "base_state": 4, "hand_matchup": 4,
                        "count_state": 4})
    seeds = list(cfgv("mlp_seeds") or (42, 7, 2024, 1234, 31337))
    print(f"\npitcher_id 임베딩 차원 = {a.dim} · 시드 {seeds}")

    val = slice(0, 20000)     # 곡선 반환용 더미. 에폭은 9 고정이라 선택에 쓰지 않는다
    for sd in seeds:
        ts = time.time()
        m, _curve = mlp.train_mlp(A, C, y, w, A[val], C[val], sizes, seed=sd,
                                  verbose=False, y_aux=aux, offset=offset,
                                  **dict(cfg, aux=cfgv("mlp_aux_lambda", 1.0)))
        Wd = mlp.export_weights(m)
        p = os.path.join(a.out, f"mlp_emb4_{sd}.npz")
        np.savez(p, **Wd)
        shp = [Wd[f"emb{i}"].shape for i in range(len(Wd["ecols"]))]
        print(f"  seed {sd:<6} {time.time()-ts:6.0f}s  emb {shp[0]}  -> {os.path.basename(p)}")

    print(f"\n완료 {time.time()-t0:.0f}s  ->  {a.out}")


if __name__ == "__main__":
    main()
