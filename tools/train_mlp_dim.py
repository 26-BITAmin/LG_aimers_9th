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
    ap.add_argument("--unk-dropout", type=float, default=0.0,
                    help="학습 시 이 비율의 행을 pitcher_id 미지값(index 0)으로 보낸다. "
                         "0 이면 배포 원본과 같다 — index 0 이 그래디언트를 못 받는다.")
    ap.add_argument("--aux-pitchmix", action="store_true",
                    help="구종군 3열을 보조 타깃에 추가한다 (§7.116 E4). 헤드 수는 타깃 모양이 정한다")
    ap.add_argument("--aux-tmdev", default=None,
                    help="트랙맨 릴리스 이탈 타깃 pkl (tm_align_targets.py 산출). "
                         "논문 Manzi 2021 이 지목한 제구 일관성 변수. 미정렬 행은 0.5(무정보)")
    ap.add_argument("--aux-lambda", type=float, default=None,
                    help="보조 손실 가중. 생략하면 meta 값(1.0). 보조 손실은 헤드별 **평균**"
                         "이므로 헤드를 늘리면 기존 헤드가 희석된다 — 보존하려면 헤드비만큼 올린다")
    ap.add_argument("--holdout-season", type=int, default=None,
                    help="이 시즌 미만으로 학습하고 이 시즌에서 MLP 앙상블 BSS 를 잰다")
    ap.add_argument("--only-seeds", default=None,
                    help="쉼표 구분. 이 시드만 학습한다 (중단분 이어서 채울 때)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    # ⚠️ 이 모듈들은 저장소에 없다 — `--work` 에 조립해 둔 것들이다 (정적 분석 불가).
    #    원우 `src/` (mlp · modelabel · soften · data) + 출하 zip (features · mlp_infer).
    need = ["mlp.py", "modelabel.py", "soften.py", "data.py", "features.py", "mlp_infer.py"]
    miss = [n for n in need if not os.path.exists(os.path.join(a.work, n))]
    if miss:
        raise SystemExit(f"[중단] --work 에 모듈이 없다: {miss}\n"
                         f"  원우 src/*.py 를 복사한 뒤 features.py · contam.py · mlp_infer.py 를\n"
                         f"  제출 zip 것으로 덮어써야 한다 (배포와 같은 피처 경로).")
    sys.path.insert(0, a.work)
    import importlib
    mlp = importlib.import_module("mlp")
    modelabel = importlib.import_module("modelabel")
    soften = importlib.import_module("soften")
    recency_weight = importlib.import_module("data").recency_weight
    build_features = importlib.import_module("features").build_features
    _mi = importlib.import_module("mlp_infer")
    encode_apply, EMB_COLS = _mi.encode_apply, _mi.EMB_COLS

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
    print(f"  복원률 {cov:.1%}  축 {aux.shape[1]}개 (middle · reverse)")
    if a.aux_tmdev:
        tmt = pd.read_pickle(a.aux_tmdev).set_index("row_id")["dev_hi"]
        v = df["row_id"].map(tmt).to_numpy(dtype=np.float64)
        cov = np.isfinite(v).mean()
        # 정렬 안 된 행은 0.5 — BCE 에서 기울기가 헤드 편향으로만 가고 방향 정보를 안 준다
        v = np.where(np.isfinite(v), v, 0.5).astype(np.float32)
        aux = np.concatenate([aux, v[:, None]], axis=1)
        print(f"  + 트랙맨 릴리스 이탈 1축  정렬 커버리지 {cov:.1%}  →  보조 축 {aux.shape[1]}개")
    if a.aux_pitchmix:
        sys.path.insert(0, "/Users/hyejin/Aimers_9th/tools")
        import pitchmix_labels
        pm, pcov, _ = pitchmix_labels.aux_labels(df)
        aux = np.concatenate([aux, pm], axis=1).astype(np.float32)
        print(f"  + 구종군 3축 복원률 {pcov:.1%}  →  보조 축 {aux.shape[1]}개")

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

    # ── UNK 드롭아웃 ──────────────────────────────────────────
    # 배포 원본에서 index 0 은 **학습 중 한 행도 받지 않는다** (LUT 를 학습 데이터로
    # 적합했으므로 미지값이 나올 수 없다). 그래서 초기값에 머물고, 그 난수 벡터가
    # test 행의 ~20% 를 받는다 (§18). 여기서는 일부러 일부 행을 index 0 으로 보내
    # **"정체를 모르는 투수" 표현을 학습시킨다** (NLP 의 UNK 드롭아웃과 같은 착상).
    #
    # ⚠️ 투수 단위가 아니라 **행 단위**로 가린다. 투수를 통째로 가리면 그 투수의
    #    임베딩이 학습을 전혀 못 받는데, 배포 시점에 그 투수는 **알려진 투수**라
    #    벡터가 비어 있으면 손해다. 행 단위면 각 투수가 (1-p) 의 행으로 자기 벡터를
    #    학습하고 index 0 은 p 의 행으로 주변화된 표현을 얻는다.
    PCOL = EMB_COLS.index("pitcher_id")

    hv = None
    if a.holdout_season:
        # soften 이 행을 복제하므로 시즌 벡터도 그때 함께 늘어난 ss 를 쓴다
        tr_m = ss < a.holdout_season
        va_m = ss == a.holdout_season
        print(f"  홀드아웃: 학습 {tr_m.sum():,}행 (<{a.holdout_season}) · 평가 {va_m.sum():,}행")
        hv = (np.flatnonzero(tr_m), np.flatnonzero(va_m))

    cfg = dict(epochs=9, hid=128, lr=2e-3, dropout=0.3, wd=1e-2, use_id_emb=True,
               emb_dim={"pitcher_id": a.dim, "batter_id": 4, "pitcher_team_id": 4,
                        "batter_team_id": 4, "base_state": 4, "hand_matchup": 4,
                        "count_state": 4})
    seeds = list(cfgv("mlp_seeds") or (42, 7, 2024, 1234, 31337))
    if a.only_seeds:
        want = [int(x) for x in a.only_seeds.split(",")]
        bad = [x for x in want if x not in seeds]
        if bad:
            raise SystemExit(f"[중단] 배포 시드 목록에 없다: {bad} (가능: {seeds})")
        seeds = want
    print(f"\npitcher_id 임베딩 차원 = {a.dim} · 시드 {seeds}"
          + (f" · aux_lambda {a.aux_lambda}" if a.aux_lambda is not None else "")
          + (f" · UNK 드롭아웃 {a.unk_dropout:.0%}" if a.unk_dropout > 0 else ""))

    val = slice(0, 20000)     # 곡선 반환용 더미. 에폭은 9 고정이라 선택에 쓰지 않는다
    zsum, yv = None, None
    for sd in seeds:
        ts = time.time()
        Cs = C
        if a.unk_dropout > 0:
            # 마스크는 **멤버 시드에 묶는다** — 멤버마다 다른 행이 가려져 앙상블
            # 다양성이 늘고, 특정 행이 모든 멤버에서 빠지는 일이 없다.
            Cs = C.copy()
            drop = np.random.RandomState(10_000 + sd).rand(len(Cs)) < a.unk_dropout
            Cs[drop, PCOL] = 0
            print(f"    UNK 마스크 적용 {drop.mean():.1%} · Cs[:,PCOL]==0 인 행 "
                  f"{(Cs[:, PCOL] == 0).mean():.1%}")
        if hv is None:
            Ai, Ci, yi, wi, oi = A, Cs, y, w, offset
        else:
            ti = hv[0]
            Ai, Ci, yi, wi = A[ti], Cs[ti], y[ti], w[ti]
            oi = None if offset is None else offset[ti]
        m, _curve = mlp.train_mlp(Ai, Ci, yi, wi, Ai[val], Ci[val], sizes, seed=sd,
                                  verbose=False, y_aux=(aux if hv is None else aux[hv[0]]),
                                  offset=oi,
                                  **dict(cfg, aux=(a.aux_lambda if a.aux_lambda is not None else cfgv("mlp_aux_lambda", 1.0))))
        Wd = mlp.export_weights(m)
        p = os.path.join(a.out, f"mlp_emb4_{sd}.npz")
        np.savez(p, **Wd)
        shp = [Wd[f"emb{i}"].shape for i in range(len(Wd["ecols"]))]
        print(f"  seed {sd:<6} {time.time()-ts:6.0f}s  emb {shp[0]}  -> {os.path.basename(p)}")

    if hv is not None:
        vi = hv[1]
        from mlp_infer import MLPPredictor  # noqa
        zs = np.zeros(len(vi))
        import glob as _g
        for f in sorted(_g.glob(os.path.join(a.out, "mlp_emb4_*.npz"))):
            z = np.load(f)
            nt = MLPPredictor({k: z[k] for k in z.files}, enc)
            Av, Cv = A[vi], C[vi]
            h = np.concatenate([Av] + [nt.emb[m2][Cv[:, nt.eidx[m2]]] for m2 in range(len(nt.emb))], axis=1)
            def _g2(x): return 0.5*x*(1.0+np.tanh(0.7978845608028654*(x+0.044715*x**3)))
            h = _g2(h @ nt.W[0] + nt.b[0]); h = _g2(h @ nt.W[1] + nt.b[1])
            zs += (h @ nt.W[2] + nt.b[2]).ravel()
        zs /= len(seeds)
        pv = 1.0/(1.0+np.exp(-zs)); yv2 = y[vi]
        r = yv2.mean(); bss = 1e5*(1-((pv-yv2)**2).mean()/(r*(1-r)))
        print(f"\n홀드아웃 {a.holdout_season}  MLP {len(seeds)}멤버 앙상블 BSS = {bss:.3f}"
              f"   (보조 축 {aux.shape[1]}개)")

    print(f"\n완료 {time.time()-t0:.0f}s  ->  {a.out}")


if __name__ == "__main__":
    main()
