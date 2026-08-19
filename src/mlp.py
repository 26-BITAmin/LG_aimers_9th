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
import os
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
import numemb
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

# ── 학습 디바이스 ───────────────────────────────────────────────────────
# **기본은 CPU다.** 환경변수 MLP_DEVICE=cuda 일 때만 GPU를 쓴다.
#   · 배포(`train.py`)는 절대 이 변수를 켜지 않는다 → 기존 경로가 비트 동일하다
#   · CUDA 빌드는 부동소수점 누산 순서와 RNG가 달라 같은 시드로도 다른 가중치가 나온다.
#     그래서 GPU에서 잰 절대 점수를 CPU 기록과 직접 대조하면 안 된다.
#     **같은 실행 안의 A/B 차이만** 읽는다 (ablation은 원래 그렇게 쓴다).
#   · 실험은 별도 환경 `.venv-gpu`에서 돌린다. `.venv`(CPU)가 제출물의 정본이다.
def _device():
    # torch는 이 모듈에서 **지연 임포트**된다 (import 비용 때문). 여기서도 지역 임포트.
    import torch
    d = os.environ.get("MLP_DEVICE", "cpu").lower()
    if d.startswith("cuda") and torch.cuda.is_available():
        return torch.device(d)
    return torch.device("cpu")

# ⚠️ 학습 루프는 `train_mlp` 한 곳에만 둔다.
#    예전에 train.py와 verify_pipeline.py에 루프가 중복돼 검증이 배포 설정을
#    테스트하지 않는 사고가 있었다 (MODEL.md §5). 변형은 cfg로만 준다.
# `scale`은 RealMLP(Holzmüller et al., NeurIPS 2024)의 스케일링 층이다.
#   None                        끄기 (기본. 예전 경로와 비트 단위로 같다)
#   {"emb": False, "wd": True}  수치 입력에 학습되는 대각 스케일 + weight decay
#   {"emb": True,  "wd": True}  임베딩 블록마다 스칼라도 하나씩 (soft noid)
#   {"emb": False, "wd": False} 스케일에만 wd를 빼서 기전을 가른다
# ⚠️ 표현력은 하나도 안 는다 — Linear(diag(s)·x)는 그냥 다른 행렬이다.
#    바뀌는 것은 (a) 곱셈 파라미터화의 경사 동역학, (b) wd의 의미다.
#    ‖W‖²+‖s‖² 를 W·diag(s)=M 제약에서 최소화하면 2Σ_i‖M_:,i‖ — 즉 **입력 피처
#    단위의 group lasso**가 된다. 배포 wd가 1e-2로 세므로 이 전환이 실제로 작동한다.
DEFAULT = dict(hid=256, epochs=25, lr=2e-3, dropout=0.10, wd=1e-4,
               emb_dim=EMB_DIM, use_id_emb=True, drop_emb=(), loss="bce",
               depth=2, resid=False, numemb=None, scale=None, aux=None,
               act="gelu")


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


def train_mlp(A, Ctr, y, w, B, Cva, sizes, seed=SEED, verbose=True, yva=None,
              y_aux=None, **cfg):
    """에폭마다 valid 로짓을 남긴다. 최적 에폭은 호출자가 고른다.

    반환: (모델, [에폭별 valid 로짓]) — 곡선을 봐야 조기중단 지점을 알 수 있다.
    """
    import torch
    import torch.nn as nn

    c = dict(DEFAULT, **cfg)
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)

    # `use_id_emb`은 "임베딩 블록을 빼서 오차를 다르게 만든다"의 한 사례였다 (noid, 채택됨).
    # `drop_emb`은 그 일반형이다 — 기본값이 비어 있어 기존 변형의 동작은 비트 단위로 같다.
    # 추론 쪽은 `export_weights`가 `ecols`를 npz에 넣고 `MLPPredictor`가 그걸 읽으므로
    # 임베딩 구성이 달라져도 규약이 깨지지 않는다 (§7.16 함정 4의 수정 덕이다).
    use_id = c["use_id_emb"]
    drop_e = set(c["drop_emb"]) | (set() if use_id else {"pitcher_id", "batter_id"})
    ecols = [x for x in EMB_COLS if x not in drop_e]
    keep = [EMB_COLS.index(x) for x in ecols]
    esizes = [sizes[i] for i in keep]
    edim = c["emb_dim"]

    # 보조 헤드 개수는 **타깃이 정한다.** 예전에는 2로 하드코딩돼 있었다
    # (몰림·반대). `modelabel`이 볼/스트라이크도 같은 방식으로 복원하므로
    # 축을 늘릴 수 있어야 한다. aux=None이면 헤드를 아예 안 만드니 기존 경로는
    # 비트 단위로 그대로다.
    n_aux = 2 if y_aux is None else int(y_aux.shape[1])

    # ⚠️ C를 **미리 자르지 않는다.** 예전에는 여기서 `Ctr[:, keep]`으로 잘랐는데,
    #    그러면 학습된 모델이 "몇 번째 컬럼을 쓰는지"를 잊어버린다. 나중에 전체
    #    7컬럼 C를 넘기면 tiny 변형(임베딩 5개)이 pitcher_id 인덱스(~470)를
    #    pitcher_team_id 테이블(크기 12)에 넣어 IndexError를 낸다.
    #    → 모델이 자기 컬럼을 스스로 고르게 한다. 그러면 학습과 추론이 똑같이
    #      전체 C를 받으므로 `mlp_infer.MLPPredictor`와 의미가 정확히 일치한다.
    class Res(nn.Module):
        """x + body(x). 항등 경로가 있으면 깊어져도 gradient가 죽지 않는다."""

        def __init__(self, body):
            super().__init__()
            self.body = body

        def forward(self, x):
            return x + self.body(x)

    class Net(nn.Module):
        def __init__(self, n_num):
            super().__init__()
            self.keep = keep
            self.emb = nn.ModuleList([nn.Embedding(s, edim[k])
                                      for s, k in zip(esizes, ecols)])
            # 수치 임베딩(numemb.py)은 **배치 안에서** 만든다. 미리 펼치면 fold2024
            # 학습셋이 2.3GB다 (이 기계 RAM 15.3GB). 공식 구현도 affine+clamp 형태다.
            self.ne = c["numemb"]
            if self.ne is not None:
                for k_ in ("idx", "raw", "w", "b"):
                    if self.ne[k_] is not None:
                        self.register_buffer(f"ne_{k_}", torch.as_tensor(self.ne[k_]))
                n_num = numemb.out_dim(self.ne, n_num)

            d = n_num + sum(edim[k] for k in ecols)
            h, p = c["hid"], c["dropout"]

            # depth=2 / resid=False 면 예전 코드와 **완전히 같은 Sequential**이 나온다:
            #   [Linear, BN, GELU, Dropout, Linear, BN, GELU, Dropout, Linear]  (인덱스 0..8)
            # export_weights가 그 인덱스를 하드코딩하고 있으므로 기본 경로를 깨면 안 된다.
            def block(i):
                # act 모듈 생성은 RNG를 소비하지 않으므로 기본(gelu) 경로는
                # 비트 단위로 예전과 같다. relu는 §7.25 후속 activation 실험용.
                a = (nn.ReLU() if c["act"] == "relu"
                     else nn.GELU(approximate="tanh"))
                return [nn.Linear(h if i else d, h), nn.BatchNorm1d(h),
                        a, nn.Dropout(p)]

            layers = block(0)
            for i in range(1, c["depth"]):
                # 잔차는 stem 이후 블록에만 건다 (입출력 폭이 h로 같아야 더할 수 있다)
                layers += [Res(nn.Sequential(*block(i)))] if c["resid"] else block(i)
            self.net = nn.Sequential(*layers, nn.Linear(h, 1))

            # 보조 헤드 (다중 과제). 몸통을 공유하고 실패 모드를 같이 맞힌다.
            # GBDT 다중분류가 진 이유가 **표현 공유가 없어서**였다 (MODEL.md §7.40) —
            # 여기서는 실제로 공유되므로 그 기전을 처음으로 시험한다.
            # ⚠️ aux=None이면 이 Linear를 **아예 만들지 않는다.** 만들면 RNG를 소비해
            #    기존 경로의 비트 동일성이 깨진다. 추론에서는 버린다(export 무관).
            # 이 Linear는 **맨 나중에** 만들어진다(스케일 층은 RNG를 안 쓴다).
            # 그래서 폭을 2→4로 바꿔도 임베딩과 몸통의 초기값은 비트 단위로 같다 —
            # 갈리는 것은 보조 헤드 자신의 초기값과 거기서 흘러드는 경사뿐이다.
            # 보조 축 개수를 비교하는 실험이 **초기화 차이가 아니라 학습 신호 차이**를
            # 재게 되므로 유리하다. (측정: 1에폭 뒤 max|Δz| 9.3e-05)
            self.aux = nn.Linear(h, n_aux) if c["aux"] else None

            # ⚠️ 스케일링 층은 **맨 마지막에** 만든다. torch.ones는 RNG를 소비하지
            #    않지만 생성 순서가 어긋나면 위 Linear/Embedding 초기화가 달라져
            #    scale=None 경로의 비트 동일성이 깨진다.
            self.n_num = n_num
            self.snum = self.semb = None
            if c["scale"] is not None:
                self.snum = nn.Parameter(torch.ones(n_num))
                if c["scale"].get("emb"):
                    self.semb = nn.Parameter(torch.ones(len(ecols)))

        def numemb_apply(self, x):
            """`numemb.apply_np`의 torch 판. **두 구현이 같아야 한다** —
            학습은 이쪽, 배포는 numpy 쪽을 쓰므로 어긋나면 조용히 성능만 깎인다
            (features.py의 `_col64` 사고와 같은 종류). ablate27이 이 둘을 대조한다."""
            if self.ne is None:
                return x
            v = x[:, self.ne_idx].unsqueeze(-1)
            if self.ne["kind"] == "ple":
                z = v * self.ne_w + self.ne_b
                # clamp가 비대칭이다 — 양끝을 열어둬야 학습 범위 밖 값이 살아남는다
                z = torch.cat([z[..., :1].clamp_max(1.0),
                               z[..., 1:-1].clamp(0.0, 1.0),
                               z[..., -1:].clamp_min(0.0)], -1)
            else:
                a = 2.0 * np.pi * v * self.ne_w
                z = torch.cat([torch.cos(a), torch.sin(a)], -1)
            return torch.cat([x[:, self.ne_raw], z.flatten(1)], 1)

        def forward(self, x, cc):
            x = self.numemb_apply(x)
            if self.snum is not None:
                x = x * self.snum
            e = [emb(cc[:, self.keep[i]]) for i, emb in enumerate(self.emb)]
            if self.semb is not None:
                e = [ei * self.semb[i] for i, ei in enumerate(e)]
            t = self.net[:-1](torch.cat([x] + e, 1))     # 공유 몸통
            return self.net[-1](t).squeeze(1)

        def forward_aux(self, x, cc):
            """(주 로짓, 보조 로짓 2개). 학습에서만 쓴다."""
            x = self.numemb_apply(x)
            if self.snum is not None:
                x = x * self.snum
            e = [emb(cc[:, self.keep[i]]) for i, emb in enumerate(self.emb)]
            if self.semb is not None:
                e = [ei * self.semb[i] for i, ei in enumerate(e)]
            t = self.net[:-1](torch.cat([x] + e, 1))
            return self.net[-1](t).squeeze(1), self.aux(t)

    dev = _device()
    m = Net(A.shape[1]).to(dev)
    m.ecols = ecols                    # export_weights가 npz에 같이 넣는다
    # scale["wd"]는 **배수**다: True=1.0(같은 wd) / False=0.0(빼기) / 10=10배.
    # AdamW의 decoupled decay는 스텝당 (1 − lr·wd)라 9에폭(약 1,320스텝)에서
    # 배수 1.0이면 s가 1.6%밖에 안 줄어든다 — group lasso 축을 실제로 흔들려면
    # 배수를 키워야 한다. 그래서 이 축을 배수로 열어둔다.
    wdm = None if c["scale"] is None else c["scale"].get("wd", True)
    wdm = None if wdm is None else (1.0 if wdm is True else
                                    (0.0 if wdm is False else float(wdm)))
    if wdm is not None and wdm != 1.0:
        sn = {"snum", "semb"}
        opt = torch.optim.AdamW(
            [{"params": [p for n_, p in m.named_parameters() if n_ not in sn]},
             {"params": [p for n_, p in m.named_parameters() if n_ in sn],
              "weight_decay": c["wd"] * wdm}], lr=c["lr"], weight_decay=c["wd"])
    else:
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

    # 보조 타깃 (몰림, 반대). `src/modelabel.py`가 asof 증분에서 복원한다.
    # **학습 전용**이다 — 추론에는 없고 필요도 없다. 피처로 쓰면 실격이다.
    taux = (None if y_aux is None
            else torch.from_numpy(y_aux.astype(np.float32)).to(dev))
    auxf = nn.BCEWithLogitsLoss(reduction="none")
    lam = float(c["aux"] or 0.0)

    tA = torch.from_numpy(A).to(dev)
    tC = torch.from_numpy(Ctr).to(dev)
    ty = torch.from_numpy(y.astype(np.float32)).to(dev)
    tw = torch.from_numpy((w / w.mean()).astype(np.float32)).to(dev)
    g = torch.Generator().manual_seed(seed)

    t0, curve = time.time(), []
    for ep in range(ep_n):
        m.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH].to(dev)
            if len(idx) < 2:                       # BatchNorm은 배치 1을 못 받는다
                continue
            opt.zero_grad(set_to_none=True)
            if lam and taux is not None:
                z_, za_ = m.forward_aux(tA[idx], tC[idx])
                l = (lossf(z_, ty[idx]) * tw[idx]).mean()
                l = l + lam * (auxf(za_, taux[idx]).mean(1) * tw[idx]).mean()
            else:
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
        out[f"emb{i}"] = e.weight.detach().cpu().numpy().astype(np.float32)

    # net = [Linear, BN, GELU, Dropout, Linear, BN, GELU, Dropout, Linear]
    #
    # ⚠️ 인덱스가 하드코딩돼 있다. depth≠2 나 resid=True 면 이 인덱스가 **다른 층**을
    #    가리킨다 — depth=3이면 net[8]이 헤드가 아니라 셋째 블록의 Linear라서
    #    **에러 없이 틀린 가중치**가 나간다. 조용히 틀리느니 여기서 죽는다.
    assert len(m.net) == 9, (
        f"export_weights는 depth=2·resid=False 전용이다 (net 길이 {len(m.net)}). "
        "다른 구조를 배포하려면 여기와 mlp_infer.MLPPredictor를 같이 일반화할 것.")
    assert getattr(m, "ne", None) is None, (
        "수치 임베딩(numemb)이 붙은 모델은 아직 배포 경로가 없다. "
        "채택하려면 numemb.apply_np를 mlp_infer 순전파 앞에 넣고 경계를 npz에 담을 것.")
    for j, (li, bi) in enumerate(((0, 1), (4, 5))):
        lin, bn = m.net[li], m.net[bi]
        W = lin.weight.detach().cpu().numpy()
        b = lin.bias.detach().cpu().numpy()
        s = bn.weight.detach().cpu().numpy() / np.sqrt(
            bn.running_var.detach().cpu().numpy() + bn.eps)
        out[f"W{j}"] = (W.T * s).astype(np.float32)
        out[f"b{j}"] = ((b - bn.running_mean.detach().cpu().numpy()) * s
                        + bn.bias.detach().cpu().numpy()).astype(np.float32)

    out["W2"] = m.net[8].weight.detach().cpu().numpy().T.astype(np.float32)
    out["b2"] = m.net[8].bias.detach().cpu().numpy().astype(np.float32)

    # ── 스케일링 층을 접어 넣는다 → 추론 코드(mlp_infer)는 한 줄도 안 바뀐다 ──
    # W0는 (입력, 출력) 배치이고 입력 순서는 [수치 n_num개 | ecols 순 임베딩 블록]이다
    # (mlp_infer.MLPPredictor.logit의 concatenate와 같은 순서).
    # 행 스케일(여기)과 BN이 만든 열 스케일(위)은 교환되므로 순서는 상관없다.
    if getattr(m, "snum", None) is not None:
        s = m.snum.detach().cpu().numpy().astype(np.float32)
        assert len(s) == m.n_num == out["W0"].shape[0] - sum(
            out[f"emb{i}"].shape[1] for i in range(len(m.ecols)))
        out["W0"][:len(s)] *= s[:, None]
    if getattr(m, "semb", None) is not None:
        se = m.semb.detach().cpu().numpy().astype(np.float32)
        for i in range(len(m.ecols)):
            out[f"emb{i}"] = out[f"emb{i}"] * se[i]
    return out


def infer(m, B, Cva, batch=65536):
    import torch
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(B), batch):
            dv = next(m.parameters()).device        # 모델이 있는 곳으로 입력을 보낸다
            out.append(m(torch.from_numpy(B[i:i + batch]).to(dv),
                         torch.from_numpy(Cva[i:i + batch]).to(dv))
                       .cpu().numpy())
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
