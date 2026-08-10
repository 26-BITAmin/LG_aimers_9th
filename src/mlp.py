"""엔티티 임베딩 MLP — 트리와 **다른 종류의 오차**를 만들 수 있는지 본다.

왜 MLP인가
    §5에서 확인된 사실: lgbm+xgb 블렌드는 두 단독 점수의 평균보다 **+15.5** 높았다.
    오차가 실제로 탈상관돼 있다는 뜻이다. 그런데 xgb의 단독 핸디캡이 −81이라 순액이
    마이너스였다. **앙상블은 구성원 실력이 비슷할 때만 이긴다.**

    MLP는 xgb보다 트리와 훨씬 다르다 — 축 정렬 분할이 아니라 매끄러운 함수를 만든다.
    탈상관 보너스가 15.5보다 클 가능성이 높다. 대략:

        블렌드 ≈ 평균(단독 점수) + 보너스
        이기려면  보너스 > 핸디캡/2

    보너스를 40으로 잡으면 핸디캡 80까지는 버틴다. 그래서 **단독으로 −80 이내**가
    이 실험의 통과선이다. (xgb는 −81로 정확히 그 선에 걸려 실패했다.)

임베딩
    `pitcher_id`는 지금 **연대순 정수**로 들어간다 (데뷔 시점 프록시, +12).
    트리는 그걸 구간으로 쪼개는 것밖에 못 한다. 임베딩은 투수별 벡터를 과제와 함께
    학습하므로 표현력이 근본적으로 다르다.

    ⚠️ CatBoost의 −214가 임베딩을 반증하지 **않는다.** CatBoost가 한 건 target
       statistic(= 타깃 평균 스칼라)이고, 임베딩은 다차원 표현을 손실로부터 학습한다.
       기전이 다르다. 단, 드리프트 위험은 같다 — 2019~2024로 배운 투수 벡터는 2025에
       낡았고 신인은 벡터가 없다. 그래서 미지 ID는 0번(평균 초기화)으로 떨어뜨린다.

배포
    **torch는 학습에만 쓴다.** 가중치를 npz로 내보내면 추론은 numpy 순전파 몇 줄이다
    (245,789 x 80 → 1초 미만). `requirements.txt`에 torch가 안 들어가므로
    오프라인 설치 10분 / pickle 버전 / CUDA 유무 위험이 전부 사라진다.

실행:  .venv/Scripts/python.exe -u src/mlp.py
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

from config import (DROP_FEATURES, FINAL_LGB_PARAMS, N_ROUNDS, RECENCY_HALF_LIFE, SEED,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV,
                    USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import PREV_COLS, SMOOTH_SPEC, build_features, fit_artifacts
from mlp_infer import EMB_COLS, encode_apply
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]

# 임베딩 컬럼 목록은 mlp_infer가 단일 진실 소스다 (제출 zip에 복사되는 쪽).
EMB_DIM = {"pitcher_id": 16, "batter_id": 16, "pitcher_team_id": 4,
           "batter_team_id": 4, "base_state": 4, "hand_matchup": 4, "count_state": 4}

# LightGBM은 NaN을 스스로 처리하지만 MLP는 못 한다.
# 원본 rate의 NaN은 이미 `_sm` 버전이 prior로 채운 값을 갖고 있으므로 그걸 쓴다.
NAN_COLS = [c for c, _ in SMOOTH_SPEC] + PREV_COLS

BATCH = 8192
THREADS = 4          # 16코어를 ablate13/14 · 사용자 작업과 나눠 쓴다

# ⚠️ 학습 루프는 `train_mlp` 한 곳에만 둔다.
#    예전에 train.py와 verify_pipeline.py에 루프가 중복돼 검증이 배포 설정을
#    테스트하지 않는 사고가 있었다 (MODEL.md §5). 변형은 cfg로만 준다.
DEFAULT = dict(hid=256, epochs=25, lr=2e-3, dropout=0.10, wd=1e-4,
               emb_dim=EMB_DIM, use_id_emb=True, loss="bce")


def fold_data(df, valid_season):
    tr = df[df.season < valid_season]
    va = df[df.season == valid_season]
    ref = int(tr.season.max())
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=ref)
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    return (X(tr), tr[TARGET_COL].to_numpy(dtype=np.float64), w,
            X(va), va[TARGET_COL].to_numpy(dtype=np.float64), art)


def fit_encoder(Xtr, art):
    """학습 데이터로 인코더 상수를 만든다. 추론 때는 이 딕셔너리만 있으면 된다.

    ⚠️ **적용 로직은 `mlp_infer.encode_apply` 한 곳에만 둔다.** 여기서도 그걸 호출한다.
       학습용/추론용 인코더를 따로 짜면 조용히 어긋난다 (features.py의 `_col64`
       사고와 같은 종류 — 1 ULP 차이가 예측을 가른다).
    """
    num_cols = [c for c in Xtr.columns if c not in EMB_COLS]
    nan_plan = [(c, (c + "_sm") if (c + "_sm") in num_cols else None,
                 float(art["priors"].get(c, 0.5)))
                for c in NAN_COLS if c in num_cols]

    cat_luts = {}
    for c in EMB_COLS:
        keys = np.unique(Xtr[c].to_numpy(dtype=np.float64))     # 정렬됨
        cat_luts[c] = (keys, np.arange(1, len(keys) + 1, dtype=np.int64))

    enc = {"num_cols": num_cols, "nan_plan": nan_plan, "cat_luts": cat_luts,
           "mu": np.zeros(len(num_cols)), "sd": np.ones(len(num_cols)),
           "sizes": [len(cat_luts[c][0]) + 1 for c in EMB_COLS]}

    A, _ = encode_apply(Xtr, enc)          # mu=0, sd=1 → 결측만 채운 원본
    assert np.isfinite(A).all(), "NaN이 남았다"
    mu, sd = A.mean(0).astype(np.float64), A.std(0).astype(np.float64)
    sd[sd < 1e-6] = 1.0
    enc["mu"], enc["sd"] = mu, sd
    return enc


def encode(Xtr, Xva, art):
    """실험용 편의 함수. 배포 경로는 fit_encoder + encode_apply를 직접 쓴다."""
    enc = fit_encoder(Xtr, art)
    A, Ctr = encode_apply(Xtr, enc)
    B, Cva = encode_apply(Xva, enc)
    return A, Ctr, B, Cva, enc["num_cols"], enc["sizes"]


def train_mlp(A, Ctr, y, w, B, Cva, sizes, seed=SEED, verbose=True, yva=None, **cfg):
    """에폭마다 valid 로짓을 남긴다. 최적 에폭은 호출자가 고른다.

    반환: (모델, [에폭별 valid 로짓]) — 곡선을 봐야 조기중단 지점을 알 수 있다.
    """
    import torch
    import torch.nn as nn

    c = dict(DEFAULT, **cfg)
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)

    use_id = c["use_id_emb"]
    ecols = [x for x in EMB_COLS if use_id or x not in ("pitcher_id", "batter_id")]
    keep = [EMB_COLS.index(x) for x in ecols]
    esizes = [sizes[i] for i in keep]
    edim = c["emb_dim"]

    # ⚠️ C를 **미리 자르지 않는다.** 예전에는 여기서 `Ctr[:, keep]`으로 잘랐는데,
    #    그러면 학습된 모델이 "몇 번째 컬럼을 쓰는지"를 잊어버린다. 나중에 전체
    #    7컬럼 C를 넘기면 tiny 변형(임베딩 5개)이 pitcher_id 인덱스(~470)를
    #    pitcher_team_id 테이블(크기 12)에 넣어 IndexError를 낸다.
    #    → 모델이 자기 컬럼을 스스로 고르게 한다. 그러면 학습과 추론이 똑같이
    #      전체 C를 받으므로 `mlp_infer.MLPPredictor`와 의미가 정확히 일치한다.
    class Net(nn.Module):
        def __init__(self, n_num):
            super().__init__()
            self.keep = keep
            self.emb = nn.ModuleList([nn.Embedding(s, edim[k])
                                      for s, k in zip(esizes, ecols)])
            d = n_num + sum(edim[k] for k in ecols)
            h, p = c["hid"], c["dropout"]
            self.net = nn.Sequential(
                nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(approximate="tanh"), nn.Dropout(p),
                nn.Linear(h, h), nn.BatchNorm1d(h), nn.GELU(approximate="tanh"), nn.Dropout(p),
                nn.Linear(h, 1))

        def forward(self, x, cc):
            e = [emb(cc[:, self.keep[i]]) for i, emb in enumerate(self.emb)]
            return self.net(torch.cat([x] + e, 1)).squeeze(1)

    m = Net(A.shape[1])
    m.ecols = ecols                    # export_weights가 npz에 같이 넣는다
    opt = torch.optim.AdamW(m.parameters(), lr=c["lr"], weight_decay=c["wd"])
    n, ep_n = len(A), c["epochs"]
    steps = ep_n * ((n + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=steps)
    # 이 대회 지표는 **Brier**인데 기본 손실은 logloss다. 둘 다 proper scoring rule이지만
    # 최적화 대상이 달라 적합되는 함수가 다르다 — logloss는 극단 확률에 큰 벌점을 줘
    # 예측을 중앙으로 민다. loss="mse"는 지표를 직접 최소화한다.
    if c["loss"] == "mse":
        def lossf(z, y):
            return (torch.sigmoid(z) - y) ** 2
    else:
        lossf = nn.BCEWithLogitsLoss(reduction="none")

    tA, tC = torch.from_numpy(A), torch.from_numpy(Ctr)
    ty = torch.from_numpy(y.astype(np.float32))
    tw = torch.from_numpy((w / w.mean()).astype(np.float32))
    g = torch.Generator().manual_seed(seed)

    t0, curve = time.time(), []
    for ep in range(ep_n):
        m.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:                       # BatchNorm은 배치 1을 못 받는다
                continue
            opt.zero_grad(set_to_none=True)
            l = (lossf(m(tA[idx], tC[idx]), ty[idx]) * tw[idx]).mean()
            l.backward()
            opt.step()
            sched.step()
            tot += float(l.detach()) * len(idx)
        z = infer(m, B, Cva)
        curve.append(z)
        if verbose:
            s = "" if yva is None else f"  validBSS={score(z, yva):>8.1f}"
            print(f"    ep{ep+1:>2d}  loss={tot/n:.5f}  ({time.time()-t0:.0f}s){s}")
    return m, curve


def export_weights(m):
    """torch 모델 → numpy 가중치 딕셔너리. **BatchNorm을 Linear에 접어 넣는다.**

    eval 모드의 BatchNorm은 학습된 상수 변환이므로 앞 Linear와 합칠 수 있다:

        y = ((W a + b) - mu) / sqrt(var+eps) * gamma + beta
          = s·W a + [(b - mu)·s + beta]          (s = gamma / sqrt(var+eps))

    numpy 쪽은 `h @ W + b` 형태(W가 (in, out))이고 torch Linear는 (out, in)이므로
    전치한다. 이렇게 하면 추론 코드에 BatchNorm이 아예 없다.
    """
    # 어떤 임베딩 컬럼을 쓰는 모델인지 함께 저장한다 — 변형마다 다르다
    # (`use_id_emb=False`면 투수/타자 임베딩이 없어 5개뿐이다).
    out = {"ecols": np.array(m.ecols)}
    for i, e in enumerate(m.emb):
        out[f"emb{i}"] = e.weight.detach().numpy().astype(np.float32)

    # net = [Linear, BN, GELU, Dropout, Linear, BN, GELU, Dropout, Linear]
    for j, (li, bi) in enumerate(((0, 1), (4, 5))):
        lin, bn = m.net[li], m.net[bi]
        W = lin.weight.detach().numpy()
        b = lin.bias.detach().numpy()
        s = bn.weight.detach().numpy() / np.sqrt(
            bn.running_var.detach().numpy() + bn.eps)
        out[f"W{j}"] = (W.T * s).astype(np.float32)
        out[f"b{j}"] = ((b - bn.running_mean.detach().numpy()) * s
                        + bn.bias.detach().numpy()).astype(np.float32)

    out["W2"] = m.net[8].weight.detach().numpy().T.astype(np.float32)
    out["b2"] = m.net[8].bias.detach().numpy().astype(np.float32)
    return out


def infer(m, B, Cva, batch=65536):
    import torch
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(B), batch):
            out.append(m(torch.from_numpy(B[i:i + batch]),
                         torch.from_numpy(Cva[i:i + batch])).numpy())
    return np.concatenate(out).astype(np.float64)


def run_lgbm(Xtr, ytr, w, Xva, yva, seed=SEED):
    import lightgbm as lgb
    p = dict(FINAL_LGB_PARAMS, seed=seed, bagging_seed=seed,
             feature_fraction_seed=seed, data_random_seed=seed)
    bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=N_ROUNDS)
    return bst.predict(Xva, raw_score=True)


def score(z, y):
    return bss(y, shift_to_prior(1 / (1 + np.exp(-z)), y.mean()))


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    res = {}

    for f in FOLDS:
        print(f"\n=== valid={f} ===")
        Xtr, ytr, w, Xva, yva, art = fold_data(df, f)
        A, Ctr, B, Cva, num_cols, sizes = encode(Xtr, Xva, art)
        print(f"  수치 {len(num_cols)}개 + 임베딩 {dict(zip(EMB_COLS, sizes))}"
              f"   ({time.time()-t0:.0f}s)")

        z_lgb = run_lgbm(Xtr, ytr, w, Xva, yva)
        print(f"  lgbm 단독  BSS = {score(z_lgb, yva):>8.1f}   ({time.time()-t0:.0f}s)")

        _, curve = train_mlp(A, Ctr, ytr, w, B, Cva, sizes, yva=yva)
        z_mlp = curve[-1]

        # 블렌드는 **로짓 공간**에서. 확률 평균은 0.5쪽으로 수축해 캘리브레이션을 흐린다.
        row = {"lgbm": score(z_lgb, yva), "mlp": score(z_mlp, yva)}
        for a in (0.1, 0.2, 0.3, 0.4, 0.5):
            row[f"blend{a:.1f}"] = score((1 - a) * z_lgb + a * z_mlp, yva)
        row["corr"] = float(np.corrcoef(z_lgb, z_mlp)[0, 1])
        res[f] = row
        print("  " + "  ".join(f"{k}={v:.1f}" if k != "corr" else f"corr={v:.3f}"
                               for k, v in row.items()))

    keys = list(res[FOLDS[0]])
    print(f"\n{'':<10s} " + " ".join(f"{f:>9d}" for f in FOLDS) + f" {'평균':>9s}")
    print("-" * 42)
    for k in keys:
        v = [res[f][k] for f in FOLDS]
        fmt = "{:>9.3f}" if k == "corr" else "{:>9.1f}"
        print(f"{k:<10s} " + " ".join(fmt.format(x) for x in v)
              + " " + fmt.format(float(np.mean(v))))

    h = np.mean([res[f]["mlp"] - res[f]["lgbm"] for f in FOLDS])
    print(f"\nMLP 단독 핸디캡: {h:+.1f}   (통과선 −80)")
    print("  · 핸디캡이 −80 이내이고 blend가 lgbm을 넘으면 → 앙상블 멤버로 채택 검토")
    print("  · 넘더라도 2 fold 부호 일치 + 크기비 3배 이내를 따로 확인해야 한다")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
