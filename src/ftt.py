"""FT-Transformer + **헤드 분할 어텐션** — 이 저장소의 첫 토큰 기반 파이프라인.

    base: src/foldcache.py 의 fold 하네스 (배포와 같은 피처 설정 · soft label 포함).
          챔피언 층이 아니다.
    격차: 챔피언(`hj05_r047f330.zip`)은 LGBM 20 + CatBoost 12 + MLP 8 + δ축 + prior
          shift 가 얹힌 스택이고, 여기서 재는 것은 그 스택의 **MLP 슬롯 후보**로서의
          단독 점수와 탈상관이다. 여기 이득이 챔피언 위에서 그대로 재현되지 않는다
          (§7.207 층 격차). 재베이스 경로는 아래 "배포로 가는 길" 참조.

왜 트랜스포머인가 — 이 저장소가 아직 안 물어본 질문

    지금 스택의 모든 팔(LGBM · CatBoost · MLP · TabM · NODE · GRANDE)은 입력을
    **하나의 평평한 벡터**로 받는다. 71개 수치 + 7개 범주가 concat 되어 첫 Linear
    로 들어가고, 거기서부터는 어떤 값이 어느 피처에서 왔는지가 사라진다. 피처 간
    상호작용은 은닉층이 "우연히" 만들어내야 한다.

    FT-Transformer(Gorishniy et al., NeurIPS 2021)는 **피처 1개 = 토큰 1개**로 놓는다.
    상호작용이 은닉층의 부산물이 아니라 **어텐션이라는 명시적 연산**이 된다.
    이 데이터에서 그게 말이 되는 이유: 제구 성공은 "이 투수가 이 카운트에서"처럼
    **그룹 사이의 곱**으로 읽히는 과제고, 실제로 §7 에서 채택된 피처 상당수가
    사람이 손으로 만든 상호작용이다 (`hand_same`, `matchup_*_edge`, `form_*`).

### 헤드 분할 (이 파일의 고유한 부분)

    표준 MHA 는 헤드가 8개든 32개든 **전부 같은 78개 토큰을 본다.** 헤드는 표현
    부분공간만 다르고 시야는 같다. 여기서는 헤드마다 **key/value 를 피처 그룹으로
    제한한다** — 헤드 h 는 자기 그룹 G(h) 의 토큰만 읽는다.

        표준 MHA        head h:  softmax(q_i · K_all) V_all
        헤드 분할       head h:  softmax(q_i · K_{G(h)}) V_{G(h)}

    쿼리는 전 토큰이 그대로 던진다(= 죽는 토큰이 없다). 제한되는 것은 **읽히는 쪽**
    뿐이다. 그래서 CLS 토큰의 출력은 문자 그대로

        [카운트 요약 ‖ 주자·점수 요약 ‖ 투수 커리어 요약 ‖ 최근 폼 요약 ‖ …]

    가 되고, 헤드 concat 이 곧 **그룹별 요약의 병치**다. 표준 MHA 는 이걸 만들 수는
    있지만 만들어야 할 이유가 없고, 실제로는 어텐션 질량이 강한 피처 몇 개로 쏠린다.
    분할은 **구조적 정규화**다: 파라미터를 하나도 안 늘리고 "각 그룹을 최소 한 번은
    읽어라"를 강제한다.

    ⚠️ 그래서 이 변경은 §7.25·§7.26 의 용량 법칙과 방향이 **반대**다. 용량을 더하는
       게 아니라 어텐션 가중치의 자유도를 **뺀다**. 이 저장소에서 이긴 변경들이
       대체로 그쪽이었다 (game_type 제거, noid 멤버, feature_fraction 0.45).

### 팔 — 분할이 일한 건지, 트랜스포머가 일한 건지, 그냥 마스킹이 일한 건지

    full          표준 FT-Transformer. **같은 헤드 수·같은 파라미터**, 마스크만 없다.
    sem8          의미 축 8그룹 × 2헤드            ┐
    sem8ov        같은 8그룹 + 축 걸침 중복 배치   ├ `src/ftt_groups.py`
    subj3         주체 축 3그룹 (투수6·타자5·상황5)┘
    shuf@<팔>     **위약.** 그 팔과 그룹 크기 분포는 같고 소속만 무작위로 섞는다.

    `scheme > full` 만으로는 부족하다. 마스킹 자체가 (내용과 무관하게) 정규화라서
    무작위 분할도 오를 수 있다 — 그게 `shuf` 다. 실제로 첫 스모크(60k·2에폭)에서
    **위약이 진짜를 이겼다** (462.9 vs 419.4). 판정은 두 조건을 같이 본다:

        ① scheme > full   분할이 일한다
        ② scheme > shuf@scheme   그룹 **구성**이 일한다 (①만이면 그냥 마스킹 효과)

    그리고 fold 2023 · 2024 에서 부호가 같아야 한다 (§7 규칙). 어긋나면 그것 자체가
    결과다 — 한쪽만 보고하지 않는다.

### 연속 / 범주 임베딩을 다르게 준다

    연속  x_j  ->  x_j · W_j + b_j          (피처마다 자기 방향 벡터. FT-T 표준)
          "plr" 를 켜면 주기 임베딩 -> Linear -> ReLU (Gorishniy 2022)
    범주  c_j  ->  Embedding(size_j, r) -> Linear(r, d)

    범주 쪽 **저랭크 r 이 이 저장소 고유의 수정**이다. 그냥 `Embedding(793, 64)` 를
    쓰면 pitcher_id 하나가 5만 파라미터로 모델을 지배한다. §7.27 에서 배포 MLP 의
    ID 임베딩 최적 차원은 **4** 였다 (2019~2024 로 배운 선수 벡터가 2025 에 낡는다).
    r 로 병목을 두면 "표현은 r 차원, 토큰은 d 차원" 이 되어 두 제약을 다 만족한다.

### soft label

    2019–22 F 의 y=1 라벨에 숨은 reverse 가 섞여 있다 (§3.1a · §7.70). `soften` 이
    행 복제로 soft target CE 와 동치를 만든다 — LGBM 과 MLP 가 이미 쓰는 경로를
    **그대로** 호출하므로 여기서 새로 짜는 것이 없다. 기본 ON (`SOFT_LABEL_F=True`).

    ⚠️ 켜고 끄고를 섞지 마라. 캐시 파일명에 박아서 다른 파일이 되게 했다.

### 배포로 가는 길 (없으면 후보가 아니다 — §7)

    이 팔이 MLP 슬롯을 이기면 → `mlp_infer.py` 규약대로 numpy 순전파를 짜서
    (마스크가 상수라 어텐션은 `einsum` 몇 줄이다) `tools_hj/rebase_*.py` 로
    챔피언 zip 의 mlp 멤버 자리에 얹는다. 245,789행 × 79토큰 × d64 는 numpy 로
    대략 20~40초 예상 — 10분 예산 안이다. 단 **먼저 이겨야 한다.**

실행 (GPU):
    # 배선 스모크
    MLP_DEVICE=cuda .venv-gpu/Scripts/python.exe -u src/ftt.py --rows 60000 --epochs 2 \
        --folds 2024 --seeds 42 --arms full sem8

    # 정규화 스크리닝
    MLP_DEVICE=cuda .venv-gpu/Scripts/python.exe -u src/ftt.py --screen

    # 본 실험
    MLP_DEVICE=cuda .venv-gpu/Scripts/python.exe -u src/ftt.py \
        --arms full sem8 sem8ov subj3 shuf@sem8 shuf@subj3 2>&1 | tee experiments/ftt1.log
"""
import argparse
import zlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import mlp
from ftt_groups import N_HEAD, SCHEMES
from mlp_infer import EMB_COLS

FOLDS = [2023, 2024]
SEEDS = (42, 7)

# 인코딩된 fold 행렬 캐시. **다른 세션과 안 겹치게** 이름에 ftt 를 박는다.
CACHE = Path(__file__).resolve().parents[1] / "experiments" / "cache_ftt"

DEFAULT = dict(
    d=64,              # 토큰 차원
    dh=8,              # 헤드당 차원 (내부 폭 = N_HEAD * dh)
    depth=2,           # 트랜스포머 블록 수
    ffn_mult=2.0,      # FFN 확장비 (원논문 4/3~2)
    # 범주 임베딩 병목 차원. int 하나 또는 컬럼별 dict.
    # ⚠️ **챔피언 실물 값이다** (`hj05_r047f330.zip` 의 `mlp_emb4_*.npz`:
    #    pitcher_id (793,8) · batter_id (831,4) · 나머지 4).
    #    `config.MLP_VARIANTS` 는 아직 pitcher_id=4 라고 적혀 있는데 그건 낡은 값이고,
    #    4→8 그 한 줄이 **LB +7.44** 였다 (§7.197). config 를 그대로 믿으면
    #    챔피언이 아닌 모델을 재게 된다 (§7.211 이 같은 사고를 기록해뒀다).
    id_rank=dict(pitcher_id=8, batter_id=4, _default=4),
    # 스크리닝 실측 (fold2024 · sem8 · 4에폭 · 1시드, `experiments/ftt_screen.log`):
    #   A 보수 (drop .30, lr 5e-4)  727.7
    #   B 저lr (drop .30, lr 2e-4)  690.4    ← lr 을 더 낮추면 진다 (−37.3)
    #   C 약드롭 (drop .10, lr 5e-4) 779.0   ← **살린 것은 wd 1e-2 지 dropout 이 아니다**
    # 1차 캘리브레이션(drop .10 · **wd 1e-5**)이 무너진 범인은 wd 였다. dropout 0.3 은
    # 배포 MLP 눈금을 그대로 옮긴 것인데 이 구조에서는 과했다 (+51.3 손해).
    p_attn=0.10,
    p_ffn=0.10,
    p_tok=0.00,        # 토큰 드롭아웃 (임베딩 직후)
    cont="linear",     # "linear" | "plr"
    plr_k=12,
    plr_sigma=0.05,
    lr=5e-4,
    wd=1e-2,           # 배포 MLP 와 같은 눈금
    epochs=6,
    aux=None,          # 보조 헤드 λ. None/0 이면 헤드를 **안 만든다** (비트 동일)
    n_aux=2,           # 축 개수 — `config.MLP_AUX_MODES` 가 정본
    inner=None,        # 내부 폭 고정(표준 MHA). None 이면 H·dh (용량이 H 에 비례)
    batch=2048,
    cls_key=True,      # CLS 를 모든 헤드의 key 로 허용 (빈 key 집합 방지)
    amp=True,          # bf16 autocast (cuda 일 때만)
)


# ──────────────────────────────────────────────────────────────────────
#  그룹 → 어텐션 마스크
# ──────────────────────────────────────────────────────────────────────
def token_names(num_cols):
    """토큰 순서 = [CLS] + 수치 + 범주. 마스크 인덱스의 단일 진실 소스."""
    return ["[CLS]"] + list(num_cols) + list(EMB_COLS)


def check_cover(groups, num_cols, tag="", verbose=True):
    """오타와 미포함 피처를 **조용히 넘기지 않는다.**

    미포함 피처는 아무 헤드도 읽지 않아 출력에서 사라진다 — 의도한 삭제가
    아니라면 그냥 성능 손실이다. 오타는 KeyError 로 즉시 죽인다.
    """
    names = set(token_names(num_cols))
    used = set()
    for g, cols in groups.items():
        for c in cols:
            if c not in names:
                raise KeyError(f"{tag} '{g}' 그룹의 '{c}' 는 존재하지 않는 컬럼이다")
            used.add(c)
    miss = [c for c in token_names(num_cols)[1:] if c not in used]
    if verbose:
        n_tok = len(num_cols) + len(EMB_COLS)
        print(f"    {tag:<8s} 그룹 {len(groups)}개 · 커버 {len(used)}/{n_tok}"
              + (f"   ⚠️ 미포함 {len(miss)}개: {miss}" if miss else "   (전부 포함)"))
    return miss


def build_mask(groups, head_plan, num_cols, *, cls_key=True, rng=None, shuffle=False):
    """(H, T) bool. True = 그 헤드가 그 토큰을 **읽을 수 있다**(key/value).

    `shuffle=True` 면 **위약**: 그룹 크기 분포는 유지한 채 소속만 무작위로 섞는다.
    크기까지 바꾸면 "작은 그룹이라 좋았다/나빴다"와 구성 효과가 섞인다.

    ⚠️ 위약도 진짜와 **같은 개수의 토큰**을 읽어야 한다. 아니면 두 팔이 읽는
       정보량 자체가 달라져 §7.190(위약은 공유 항을 남겨야 한다)에 걸린다.
    """
    import torch

    names = token_names(num_cols)
    idx = {n: i for i, n in enumerate(names)}
    T = len(names)
    feat = list(range(1, T))          # CLS 를 뺀 실제 피처 토큰

    if shuffle:
        assert rng is not None, "위약 팔에는 rng 가 필요하다"
        groups = {g: [names[i] for i in rng.choice(feat, size=len(cols),
                                                   replace=False)]
                  for g, cols in groups.items()}

    H = sum(head_plan.values())
    m = torch.zeros(H, T, dtype=torch.bool)
    h, plan = 0, []
    for g, nh in head_plan.items():
        keys = feat if g == "*" else [idx[c] for c in groups[g]]
        for _ in range(nh):
            m[h, keys] = True
            if cls_key:
                m[h, 0] = True
            plan.append((g, int(m[h].sum())))
            h += 1
    assert h == H
    assert bool(m.any(1).all()), "key 가 하나도 없는 헤드가 있다 (SDPA 가 NaN 을 낸다)"
    return m, plan


def arm_mask(arm, num_cols, rng, cls_key=True):
    """팔 이름 -> (마스크, plan). `full` 은 None, `shuf@X` 는 X 의 위약."""
    if arm == "full":
        return None, []
    shuffle = arm.startswith("shuf@")
    name = arm.split("@", 1)[1] if shuffle else arm
    if name not in SCHEMES:
        raise KeyError(f"모르는 scheme '{name}' — {list(SCHEMES)} 중 하나여야 한다")
    groups, plan = SCHEMES[name]
    return build_mask(groups, plan, num_cols, cls_key=cls_key,
                      rng=rng, shuffle=shuffle)


def arm_pool(arm, num_cols):
    """`fgtt@X` 면 (G, T) 평균 풀링 행렬, 아니면 None.

    ### ② FGTT 식 — 그룹 **자체**가 토큰이다 (arXiv:2412.06825)

    같은 "의미 그룹" 아이디어인데 구현이 내 마스크와 정확히 반대편이다:

        마스크(`sem8`)   토큰 79 개 유지, **헤드의 시야**를 그룹으로 자른다
                         → 그룹 안 세부가 어텐션에 그대로 남는다
        FGTT(`fgtt@sem8`) 그룹 하나를 토큰 하나로 **풀링**한다 (79 → 1+8)
                         → 그룹 안 세부는 평균으로 뭉개지고, 어텐션은
                           **그룹 사이 상호작용만** 본다

    FGTT 원논문은 교통사고 데이터에서 XGBoost·CatBoost 를 이겼다고 보고한다.
    토큰이 79 → 9 라 어텐션 비용도 훨씬 싸다.

    ⚠️ 풀링은 **평균**이다. 학습 가중을 주면 그건 어텐션을 한 겹 더 넣는 것이라
       "풀링이냐 마스크냐" 대조가 흐려진다. 원논문도 그룹 토큰을 그냥 만든다.
    ⚠️ 겹치는 그룹(`sem8ov`)도 그대로 된다 — 한 피처가 여러 그룹 토큰에 들어간다.
    """
    if not arm.startswith("fgtt@"):
        return None
    import torch

    name = arm.split("@", 1)[1]
    if name not in SCHEMES:
        raise KeyError(f"모르는 scheme '{name}'")
    groups = SCHEMES[name][0]
    names = token_names(num_cols)
    idx = {n: i for i, n in enumerate(names)}
    P = torch.zeros(len(groups), len(names))
    for gi, cols in enumerate(groups.values()):
        for c in cols:
            P[gi, idx[c]] = 1.0
        P[gi] /= P[gi].sum()
    return P


# ──────────────────────────────────────────────────────────────────────
#  모델
# ──────────────────────────────────────────────────────────────────────
def _rank_of(col, spec):
    """`id_rank` 가 int 면 전 컬럼 공통, dict 면 컬럼별(`_default` 가 나머지)."""
    if isinstance(spec, dict):
        return int(spec.get(col, spec.get("_default", 4)))
    return int(spec)


def build_net(n_num, cat_sizes, mask, c, pool=None):
    """mask=None 이면 표준 FT-Transformer(`full` 팔)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    d = c["d"]
    # `full` 팔은 마스크가 없지만 헤드 수는 분할 팔과 **똑같이** 맞춘다.
    # 안 맞추면 재는 것이 "분할 효과"가 아니라 "헤드 수 효과"가 된다.
    H = int(c.get("_n_head") or N_HEAD) if mask is None else mask.shape[0]

    # ── 헤드당 차원 `dh` 를 어떻게 정하는가 — 이게 헤드 수 축의 성격을 바꾼다 ──
    # `inner` 를 주면 내부 폭을 고정하고 dh = inner // H 로 **쪼갠다** (표준 MHA 규약).
    #   → 헤드 수가 **순수한 분할 노브**가 된다. 파라미터가 안 변하므로 H 를 바꿔도
    #     재는 것이 용량이 아니라 "몇 갈래로 나눠 보는가" 다.
    # `inner` 가 없으면 dh 고정 → 내부 폭 = H·dh 라 **H 를 늘리면 용량도 는다.**
    #   → 그 경우 H 비교는 용량과 교란된다. 지금 실험이 H=16 을 전 팔에 고정한 이유다.
    if c.get("inner"):
        if int(c["inner"]) % H:
            raise ValueError(f"inner {c['inner']} 가 헤드 {H} 로 안 나눠떨어진다")
        dh = int(c["inner"]) // H
    else:
        dh = c["dh"]

    class Tokenizer(nn.Module):
        """연속과 범주에 **다른 임베딩**을 준다 — 이 파이프라인의 전제."""

        def __init__(self):
            super().__init__()
            self.cont = c["cont"]
            if self.cont == "plr":
                # 주기 임베딩: [cos(2πfx), sin(2πfx)] -> Linear -> ReLU
                # (Gorishniy et al. 2022). f 는 학습되는 파라미터다.
                k = c["plr_k"]
                self.freq = nn.Parameter(torch.randn(n_num, k) * c["plr_sigma"])
                self.plin = nn.Parameter(torch.empty(n_num, 2 * k, d))
                nn.init.uniform_(self.plin, -(2 * k) ** -0.5, (2 * k) ** -0.5)
                self.pb = nn.Parameter(torch.zeros(n_num, d))
            else:
                # 선형 토큰화: 피처마다 자기 방향 벡터 W_j 를 갖고 값이 그 크기다.
                self.wn = nn.Parameter(torch.empty(n_num, d))
                self.bn = nn.Parameter(torch.empty(n_num, d))
                nn.init.uniform_(self.wn, -d ** -0.5, d ** -0.5)
                nn.init.uniform_(self.bn, -d ** -0.5, d ** -0.5)

            # 컬럼마다 다른 랭크를 준다. pitcher_id 만 8 인 것이 챔피언 구조다 —
            # 793명을 4차원에 밀어 넣는 것이 실제로 손실이었고 그 한 줄이 LB +7.44 였다.
            ranks = [_rank_of(col, c["id_rank"]) for col in EMB_COLS]
            self.emb = nn.ModuleList([nn.Embedding(s, r)
                                      for s, r in zip(cat_sizes, ranks)])
            self.cproj = nn.ModuleList([nn.Linear(r, d) for r in ranks])
            for e, r in zip(self.emb, ranks):
                nn.init.normal_(e.weight, std=r ** -0.5)

            self.cls = nn.Parameter(torch.empty(1, 1, d))
            nn.init.uniform_(self.cls, -d ** -0.5, d ** -0.5)

        def forward(self, x, cc):
            b = x.shape[0]
            if self.cont == "plr":
                a = 2.0 * np.pi * x.unsqueeze(-1) * self.freq          # (B,n,k)
                z = torch.cat([torch.cos(a), torch.sin(a)], -1)        # (B,n,2k)
                tn = F.relu(torch.einsum("bnk,nkd->bnd", z, self.plin) + self.pb)
            else:
                tn = x.unsqueeze(-1) * self.wn + self.bn               # (B,n,d)
            tc = torch.stack([p(e(cc[:, i])) for i, (e, p)
                              in enumerate(zip(self.emb, self.cproj))], 1)
            return torch.cat([self.cls.expand(b, -1, -1), tn, tc], 1)

    class GroupedMHA(nn.Module):
        """헤드마다 key/value 를 그룹으로 제한한다. 쿼리는 전 토큰이 던진다.

        마스크는 쿼리 위치에 안 의존하므로 (1, H, 1, T) 로 브로드캐스트한다 —
        (B, H, T, T) 를 만들지 않아 메모리가 T 배 싸다.
        """

        def __init__(self):
            super().__init__()
            self.H, self.dh = H, dh
            self.q = nn.Linear(d, H * dh)
            self.k = nn.Linear(d, H * dh)
            self.v = nn.Linear(d, H * dh)
            self.o = nn.Linear(H * dh, d)
            if mask is None:
                self.m = None
            else:
                self.register_buffer("m", mask.view(1, H, 1, -1))

        def forward(self, t):
            b, T, _ = t.shape

            def sp(u):
                return u.view(b, T, self.H, self.dh).transpose(1, 2)

            att = F.scaled_dot_product_attention(
                sp(self.q(t)), sp(self.k(t)), sp(self.v(t)),
                attn_mask=(None if self.m is None else self.m),
                dropout_p=(c["p_attn"] if self.training else 0.0))
            return self.o(att.transpose(1, 2).reshape(b, T, self.H * self.dh))

    class Block(nn.Module):
        """pre-LN. post-LN 은 이 깊이에서 warmup 없이 잘 안 붙는다."""

        def __init__(self):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.att = GroupedMHA()
            hf = int(d * c["ffn_mult"])
            self.ffn = nn.Sequential(nn.Linear(d, hf), nn.GELU(approximate="tanh"),
                                     nn.Dropout(c["p_ffn"]), nn.Linear(hf, d))
            self.drop = nn.Dropout(c["p_ffn"])

        def forward(self, t):
            t = t + self.drop(self.att(self.n1(t)))
            return t + self.drop(self.ffn(self.n2(t)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = Tokenizer()
            self.tdrop = nn.Dropout(c["p_tok"])
            # FGTT 모드 — 학습 파라미터가 아니라 상수 행렬이다 (평균 풀링).
            self.register_buffer("pool", pool if pool is not None
                                 else torch.zeros(0), persistent=False)
            self.blocks = nn.ModuleList([Block() for _ in range(c["depth"])])
            self.nf = nn.LayerNorm(d)
            self.head = nn.Linear(d, 1)
            # 보조 헤드 — 몸통(CLS 표현)을 공유하고 실패 모드를 같이 맞힌다.
            # ⚠️ **주 헤드 뒤에** 만든다. 앞에 두면 aux 를 켤 때 RNG 소비 순서가
            #    달라져 몸통 초기값이 통째로 바뀌고, 2×2 가 재는 것이 "aux 효과"가
            #    아니라 "aux 효과 + 다른 초기값"이 된다 (mlp.py 가 같은 규칙을 쓴다).
            # ⚠️ aux=None 이면 **아무것도 안 만든다** → 기존 실행과 비트 동일.
            self.aux = nn.Linear(d, int(c["n_aux"])) if c["aux"] else None

        def _trunk(self, x, cc):
            t = self.tok(x, cc)
            if self.pool.numel():
                t = torch.cat([t[:, :1],
                               torch.einsum("gt,btd->bgd", self.pool, t)], 1)
            t = self.tdrop(t)
            for b in self.blocks:
                t = b(t)
            return F.gelu(self.nf(t[:, 0]))

        def forward_aux(self, x, cc):
            h = self._trunk(x, cc)
            return self.head(h).squeeze(1), self.aux(h)

        def forward(self, x, cc):
            # 읽기는 CLS 토큰에서. 헤드 분할이면 이 벡터가 그룹 요약의 병치다.
            # (FGTT 모드면 피처 토큰이 그룹으로 접힌 뒤다 — `_trunk` 참조)
            return self.head(self._trunk(x, cc)).squeeze(1)

    return Net()


# ──────────────────────────────────────────────────────────────────────
#  학습
# ──────────────────────────────────────────────────────────────────────
def train(A, C, y, w, B, Cva, cat_sizes, mask, seed=42, yva=None,
          verbose=True, pool=None, y_aux=None, offset=None, zadd=None, **cfg):
    """에폭마다 valid 로짓을 남긴다 (`mlp.train_mlp` 와 같은 규약).

    ⚠️ 판정에는 **마지막 에폭**을 쓴다. argmax 는 valid 를 보고 고르는 것이라
       낙관 편향이다 (mlp3 의 규율). OneCycle 이 마지막 스텝에서 LR 0 으로
       어닐링되므로 마지막 값이 수렴점이고, 곡선은 진단용으로만 출력한다.
    """
    import torch
    import torch.nn as nn

    c = dict(DEFAULT, **cfg)
    if y_aux is not None:
        c["n_aux"] = int(y_aux.shape[1])
    torch.manual_seed(seed)
    dev = mlp._device()

    net = build_net(A.shape[1], cat_sizes,
                    None if mask is None else mask.to(dev), c,
                    pool=None if pool is None else pool.to(dev)).to(dev)
    n_par = sum(p.numel() for p in net.parameters())

    opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=c["wd"])
    n, bs = len(A), c["batch"]
    steps = c["epochs"] * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=steps)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    # 보조 타깃 — **학습 전용**이다. 추론에는 없고 피처로 쓰면 실격이다.
    lam = float(c["aux"] or 0.0)
    taux = (None if (y_aux is None or not lam)
            else torch.from_numpy(y_aux.astype(np.float32)).to(dev))
    auxf = nn.BCEWithLogitsLoss(reduction="none")

    # `offset` — 학습 행별 시작 로짓 (LGBM `init_score` 규약, MODEL.md §7.114).
    # 손실을 sigmoid(z + offset) 으로 계산하므로 모델은 **δ 가 잡는 몫을 안 배우고**
    # 나머지에 용량을 쓴다. ⚠️ 반환 로짓에는 **안 더한다** — 블렌드가 편차 공간에서
    # 이뤄지고 레벨은 prior shift 가 맡는다 (mlp.train_mlp 와 같은 규약).
    toff = (None if offset is None
            else torch.from_numpy(np.asarray(offset, dtype=np.float32)).to(dev))
    if toff is not None and len(toff) != len(A):
        raise ValueError(f"offset {len(toff)} != 학습행 {len(A)} — 정렬이 깨졌다")

    tA = torch.from_numpy(A).to(dev)
    tC = torch.from_numpy(C).to(dev)
    ty = torch.from_numpy(y.astype(np.float32)).to(dev)
    tw = torch.from_numpy((w / w.mean()).astype(np.float32)).to(dev)
    g = torch.Generator().manual_seed(seed)
    amp = bool(c["amp"]) and dev.type == "cuda"

    t0, curve = time.time(), []
    for ep in range(c["epochs"]):
        net.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs].to(dev)
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                if taux is None:
                    z, za = net(tA[idx], tC[idx]), None
                else:
                    z, za = net.forward_aux(tA[idx], tC[idx])
            zo = z.float() if toff is None else z.float() + toff[idx]
            l = (lossf(zo, ty[idx]) * tw[idx]).mean()
            if za is not None:
                l = l + lam * (auxf(za.float(), taux[idx]).mean(1)
                               * tw[idx]).mean()
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(l.detach()) * len(idx)
        z = predict(net, B, Cva, amp=amp)
        curve.append(z)
        if verbose:
            # 곡선 BSS 는 **배포와 같은 자리**에서 읽는다 — valid δ 를 더한 뒤다.
            zs_ = z if zadd is None else z + zadd
            s = "" if yva is None else f"  validBSS={mlp.score(zs_, yva):>8.1f}"
            print(f"      ep{ep+1:>2d}  loss={tot/n:.5f}  ({time.time()-t0:.0f}s){s}")
    return net, curve, n_par


def predict(net, B, Cva, batch=16384, amp=False):
    import torch
    net.eval()
    dev = next(net.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(B), batch):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                z = net(torch.from_numpy(B[i:i + batch]).to(dev),
                        torch.from_numpy(Cva[i:i + batch]).to(dev))
            out.append(z.float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


# ──────────────────────────────────────────────────────────────────────
#  fold 데이터 (인코딩까지 캐시 — 피처 빌드가 fold 당 수 분이다)
# ──────────────────────────────────────────────────────────────────────
def fold_arrays(df, fold, rows=None, soft=True):
    """`foldcache.fold_data` 를 그대로 쓴다 — 배포와 같은 피처 설정이다.

    soft label 은 `soften.soften` 을 **배포와 같은 순서로** 호출한다
    (원본 행에서 라벨을 읽고 → 행 복제 → 그 다음 인코더 fit).
    """
    import foldcache as fc

    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"fold{fold}_{'soft' if soft else 'raw'}_{fc._key()}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        out = (z["A"], z["C"], z["y"], z["w"], z["B"], z["Cva"], z["yva"],
               [str(x) for x in z["num_cols"]], list(z["sizes"]))
    else:
        import soften
        Xtr, ytr, w, Xva, yva, art, tr = fc.fold_data(df, fold)
        if soft:
            n0 = len(Xtr)
            Xtr, ytr, w, _, h = soften.soften(tr, Xtr, ytr, w)
            print(f"  soft label: {n0:,} → {len(Xtr):,}행  h={h}")
        A, C, B, Cva, num_cols, sizes = mlp.encode(Xtr, Xva, art)
        np.savez(f, A=A, C=C, y=ytr, w=w, B=B, Cva=Cva, yva=yva,
                 num_cols=np.array(num_cols), sizes=np.array(sizes))
        out = (A, C, ytr, w, B, Cva, yva, num_cols, list(sizes))
    if rows:
        A, C, y, w, B, Cva, yva, nc, sz = out
        # ⚠️ 학습은 뒤에서 자른다 — 최근 시즌을 남겨야 스모크가 같은 레짐이다
        out = (A[-rows:], C[-rows:], y[-rows:], w[-rows:],
               B[:rows], Cva[:rows], yva[:rows], nc, sz)
    return out


# ──────────────────────────────────────────────────────────────────────
#  ① 정규화 스크리닝 — 1차 캘리브레이션이 과적합으로 죽어서 필요해졌다
# ──────────────────────────────────────────────────────────────────────
#   lr 1e-3 / dropout .10 / wd 1e-5 / id_rank 8 / 10에폭:
#       ep1 611.7 → ep2 591.0 → ep3 579.4 → ep4 572.7 → ep5 453.5   (단조 하락)
#   학습 손실은 계속 내려가는데 valid 가 무너진다 = 전형적 과적합이다.
#   이 과제의 총 신호는 0.004 nat 규모라(§7.13) 용량보다 정규화가 먼저다.
#   ep8 은 226.9 까지 무너졌다. 원설정을 다시 잴 필요는 없다 — A 를 기준으로 놓고
#   **한 번에 한 축씩만** 되돌려 무엇이 범인인지 가른다 (교란을 안 만든다).
SCREEN = {
    "A_보수":   dict(lr=5e-4, p_attn=0.30, p_ffn=0.30, wd=1e-2, id_rank=4, epochs=4),
    "B_저lr":   dict(lr=2e-4, p_attn=0.30, p_ffn=0.30, wd=1e-2, id_rank=4, epochs=4),
    "C_약드롭": dict(lr=5e-4, p_attn=0.10, p_ffn=0.10, wd=1e-2, id_rank=4, epochs=4),
    "D_큰ID":   dict(lr=5e-4, p_attn=0.30, p_ffn=0.30, wd=1e-2, id_rank=16, epochs=4),
}


def fold_aux(df, fold, soft=True):
    """배포 MLP 가 쓰는 **보조 타깃**(몰림·반대)을 캐시된 학습행에 정렬해 돌려준다.

    ⚠️ 이걸 빼먹고 잰 앵커는 **배포 MLP 슬롯이 아니다.** `train.py` 와
       `foldcache.compute` 는 둘 다 `y_aux=aux, aux=MLP_AUX_LAMBDA` 로 학습하고,
       §7.41 이 그 값을 +20.55 / +9.91 로 기록해뒀다.

    순서는 배포와 같아야 한다 — 원본 행에서 라벨 복원 → soften 이 (X,y,w,aux) 를
    **같이** 늘린다. 순서가 바뀌면 복제 행의 보조 타깃이 빈다 (foldcache 주석 참조).
    """
    import foldcache as fc

    f = CACHE / f"aux_fold{fold}_{'soft' if soft else 'raw'}_{fc._key()}.npz"
    if f.exists():
        return np.load(f)["aux"]

    import modelabel
    import soften
    if df is None:
        # ⚠️ 호출부(`need_df`)가 **fold_arrays 캐시만** 보고 df 로드를 건너뛴다.
        #    여기는 캐시가 따로라 df 가 None 으로 올 수 있다. 직접 읽는다.
        from config import TRAIN_CSV
        from data import load_csv
        print("  train.csv 로드 (보조 라벨용) …")
        df = load_csv(TRAIN_CSV)
    Xtr, ytr, w, _, _, _, tr = fc.fold_data(df, fold)
    aux, cov = modelabel.aux_labels(tr)
    if soft:
        _, _, _, aux, _ = soften.soften(tr, Xtr, ytr, w, aux)
    np.savez(f, aux=aux)
    print(f"  보조 라벨 복원 {cov:.1%}  shape {aux.shape}")
    return aux


def fold_delta(df, fold, soft=True):
    """δ 축(`features.logit_adjust`)을 **fold 층에서 fit** 해 (학습행 offset, valid δ) 반환.

    ⚠️ `foldcache.fold_data` 의 `art` 에는 δ 축이 **없다** — 그건 `train.py` 가
       ctxfit / asofdelta / mixent 로 만든다. 그래서 하네스가 배포의 사후 로짓
       보정을 통째로 빼놓고 잰다 (정정 4). [[run-diagnostics-on-the-deployed-layer]]

    ⚠️ **학습 시즌으로만 fit 한다.** train.py 는 전체 df 로 하지만 그건 홀드아웃이
       없는 최종 빌드라서다. valid 행으로 축을 만들면 그게 리크다.

    ⚠️ soft label 이 켜져 있으면 학습행이 복제로 늘어난다 — δ 도 **같은 확장**을
       거쳐야 정렬이 맞는다. `soften.soften` 을 그대로 통과시켜 그 로직을 재사용한다
       (직접 복제 마스크를 짜면 조용히 어긋난다).
    """
    import foldcache as fc

    f = CACHE / f"delta_fold{fold}_{'soft' if soft else 'raw'}_{fc._key()}.npz"
    if f.exists():
        z = np.load(f)
        return z["off"], z["zva"]

    import pandas as pd
    import asofdelta
    import ctxfit
    import mixent
    import soften
    from config import (ASOF_AXES, ASOF_STRENGTH, CONTAM_AXES, CTX_AXES,
                        CTX_HALF_LIFE, CTX_K, GT_F_BETA, GT_F_DELTA, MIX_AXES,
                        RECENCY_HALF_LIFE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                        TARGET_COL, TRAIN_CSV)
    from data import load_csv, recency_weight
    from features import fit_artifacts, logit_adjust

    if df is None:
        print("  train.csv 로드 (δ 축용) …")
        df = load_csv(TRAIN_CSV)
    mtr, mva = fc.SPLITS[fold](df.season.to_numpy(), df.game_month.to_numpy())
    tr, va = df[mtr], df[mva]
    ref = int(tr.season.max())
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=ref)

    # δ 표는 모델과 **다른 half-life** 를 쓴다 (§7.61 — 셀당 표본이 필요해 완만하게)
    ctx = ctxfit.fit_ctx(tr, TARGET_COL, CTX_AXES, CTX_K,
                         weights=recency_weight(tr.season.to_numpy(), CTX_HALF_LIFE,
                                                ref_season=ref))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, ctx=ctx, contam_axes=CONTAM_AXES)
    art["asof"] = asofdelta.fit_asof_axes(tr, ASOF_AXES, ASOF_STRENGTH,
                                          ct=art.get("contam"))
    art["mix"] = mixent.fit_mixent(tr, MIX_AXES)
    art["gt"] = {"delta": GT_F_BETA * GT_F_DELTA}

    off, hits = logit_adjust(tr, art)
    zva, hva = logit_adjust(va, art)
    if soft:
        # 복제 확장을 soften 에 맡긴다 — δ 를 컬럼으로 태워 보낸다
        ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
        d2, _, _, _, _ = soften.soften(tr, pd.DataFrame({"d": off}), ytr, w)
        off = d2["d"].to_numpy()
    np.savez(f, off=off, zva=zva)
    print(f"  δ 축 fit: 학습 {len(off):,}행 (sd {off.std():.4f}) · "
          f"valid sd {zva.std():.4f} · 적중률 "
          f"{ {k: round(v, 3) for k, v in hva.items()} }")
    return off, zva


# ──────────────────────────────────────────────────────────────────────
#  ② 앵커 — **같은 층의 배포 MLP**. 이게 없으면 트랜스포머 점수는 뜻이 없다
# ──────────────────────────────────────────────────────────────────────
# ⚠️ `config.MLP_VARIANTS` 를 그대로 쓰면 안 된다. 거기 `pitcher_id` 가 아직 4 인데
#    챔피언 실물은 **8** 이다 (`hj05_r047f330.zip` → `mlp_emb4_*.npz`). 그 한 줄이
#    LB +7.44 였고(§7.197), config 를 믿으면 챔피언이 아닌 모델을 앵커로 삼게 된다.
#    [[audit-inherited-constants-not-just-registered-gates]]
# ⚠️ 넘어야 할 바는 `emb4` **한 멤버가 아니다.** 배포 MLP 슬롯은 8멤버다:
#       z_mlp = 0.875 · mean(emb4 5시드) + 0.125 · mean(noid 3시드)
#    (`config.MLP_VARIANT_WEIGHT` · §7.120 — 이 가중 자체가 LB +1.85 였다)
#    한 멤버짜리를 앵커로 쓰면 **바를 낮춰놓고 넘었다고 말하게 된다.**
_EMB = {"pitcher_id": 8, "batter_id": 4, "pitcher_team_id": 4, "batter_team_id": 4,
        "base_state": 4, "hand_matchup": 4, "count_state": 4}
ANCHOR = {
    # 이름: (cfg, 시드들, 가중)   — cfg 는 config.MLP_VARIANTS 와 같되 dim 만 챔피언 값
    "emb4": (dict(epochs=9, hid=128, lr=2e-3, dropout=0.3, wd=1e-2,
                  use_id_emb=True, emb_dim=_EMB),
             (42, 7, 2024, 1234, 31337), 0.875),
    "noid": (dict(epochs=6, hid=128, lr=2e-3, dropout=0.3, wd=1e-2,
                  use_id_emb=False,
                  emb_dim={k: v for k, v in _EMB.items() if not k.endswith("er_id")}),
             (42, 7, 2024), 0.125),
}


def run_anchor(df, fold, soft, seeds=None, use_aux=True):
    """배포 **MLP 슬롯 전체**(8멤버 가중 로짓평균)를 같은 배열 위에서 재현한다.

    멤버별 단독 점수도 같이 출력한다 — 시드 산포가 곧 **이 층의 노이즈 대역**이고,
    그걸 모르면 트랜스포머와의 격차를 읽을 수 없다.
    """
    A, C, y, w, B, Cva, yva, num_cols, sizes = fold_arrays(df, fold, soft=soft)
    # 보조 헤드는 배포의 일부다 — 빼면 앵커가 낮게 잡히고 후보가 유리해진다.
    from config import MLP_AUX_LAMBDA
    ya = fold_aux(df, fold, soft) if use_aux else None
    extra = dict(aux=MLP_AUX_LAMBDA) if use_aux else {}
    if ya is not None and len(ya) != len(A):
        raise ValueError(f"보조 라벨 {len(ya)} != 학습행 {len(A)} — 정렬이 깨졌다")
    print(f"    aux={'ON' if use_aux else 'OFF'}"
          + (f" (λ={MLP_AUX_LAMBDA}, 축 {ya.shape[1]}개)" if use_aux else ""))
    zslot, parts = 0.0, {}
    for vname, (cfg, vseeds, wt) in ANCHOR.items():
        zs = []
        for s in vseeds:
            _, curve = mlp.train_mlp(A, C, y, w, B, Cva, sizes, seed=s,
                                     verbose=False, y_aux=ya, **dict(cfg, **extra))
            zs.append(curve[-1])
        solo = [mlp.score(z, yva) for z in zs]
        zm = np.mean(zs, axis=0)
        parts[vname] = zm
        zslot = zslot + wt * zm
        print(f"    {vname:<5s} {len(vseeds)}시드  단독 "
              + " ".join(f"{v:.1f}" for v in solo)
              + f"   sd {np.std(solo, ddof=1):>5.1f}"
              + f"   시드평균 {mlp.score(zm, yva):>8.1f}")
    sc = mlp.score(zslot, yva)
    print(f"  ▶ 앵커 = 배포 MLP 슬롯 (emb4 .875 + noid .125) fold{fold} = {sc:>8.1f}")
    # z 를 남긴다 — 블렌드/상관 질문은 재학습 없이 z 만 있으면 끝난다 (foldcache 의 교훈).
    np.savez(CACHE / f"z_anchor_fold{fold}"
             f"{'' if use_aux else '_noaux'}.npz", z=zslot, yva=yva,
             **{f"z_{k}": v for k, v in parts.items()})
    return zslot, yva


# ──────────────────────────────────────────────────────────────────────
#  ③ 에폭 예산 — **후보가 아닌 팔에서만** 고른다
# ──────────────────────────────────────────────────────────────────────
# 팔마다 최적 에폭이 다를 수 있다는 것은 맞다. 그런데 판정 fold 에서 팔마다 고르면
# 각 팔이 노이즈 곡선의 max 를 갖게 되고 **곡선이 시끄러운 팔이 더 자주 이긴다** —
# mlp3 가 막으려고 쓰인 편향이고, 위약이 진짜를 이기는 기전이 정확히 그것이다.
# 그리고 OneCycle 이라 한 번 돌린 곡선의 중간점은 LR 이 안 식은 **다른 모델**이다.
# 제대로 고르려면 격자마다 재학습해야 한다 (6팔 × 3점 = 2h45m, 선택에만).
#
# → **전 팔에 격자를 돌리고 팔마다 argmax 를 준다** (사용자 지적, 2026-08-30).
#
#   내 원래 반론("판정 fold 에서 팔마다 고르면 편향")에는 답이 있다:
#   **위약에도 똑같이 적용하면 그 편향이 상쇄된다.**
#
#       sem8(E*) − shuf@sem8(E*)      ← 양쪽 다 같은 횟수만큼 격자에서 뽑았다
#
#   각 팔이 max 를 갖는 것은 맞지만 위약도 같은 수의 기회를 갖는다. 위약 팔의
#   존재 이유가 정확히 이것이다 — **그룹 내용과 무관한 모든 절차적 이점을 흡수**.
#
#   ⛔ 대신 두 가지를 못 박는다:
#     1. 에폭은 **규칙으로** 고른다. `E*(팔) = fold2024 격자 argmax`. 눈으로
#        "얘는 6이 안정적이네" 하고 고르면 그 순간 다시 편향이다.
#     2. fold2023 은 그 `E*` 로 읽는다 → **fold2023 이 에폭 선택의 표본 외 확인**이
#        된다 (mlp3 가 쓰는 패턴: "에폭은 fold2024 곡선에서 골랐다 → 2023 이 OOS").
BUDGET_ARMS = ("full", "sem8", "sem8ov", "subj3", "shuf@sem8", "shuf@subj3")
BUDGET_GRID = (4, 7, 10)


def run_budget(df, fold, soft, over, arms=BUDGET_ARMS, grid=BUDGET_GRID, seeds=(42,)):
    """격자점마다 **재학습**한다 — OneCycle 이라 곡선 중간점은 다른 모델이다."""
    A, C, y, w, B, Cva, yva, num_cols, sizes = fold_arrays(df, fold, soft=soft)
    print(f"\n=== 에폭 예산 격자  fold {fold} · soft={soft} ===")
    print(f"  팔 {list(arms)}   격자 {list(grid)}   시드 {list(seeds)}")
    print("  규칙: E*(팔) = 격자 argmax. 위약도 같은 규칙을 받으므로 선택 편향이 상쇄된다.")
    tab, best = {}, {}
    for arm in arms:
        rng = np.random.default_rng(zlib.crc32(f"{arm}|{fold}".encode()) & 0xFFFFFFFF)
        mask, _ = arm_mask(arm, num_cols, rng, cls_key=DEFAULT["cls_key"])
        row = {}
        for e in grid:
            zs = [train(A, C, y, w, B, Cva, sizes, mask, seed=s, yva=yva,
                        verbose=False, **dict(over, epochs=e))[1][-1] for s in seeds]
            row[e] = mlp.score(np.mean(zs, axis=0), yva)
            print(f"    {arm:<12s} epochs={e:<3d} {row[e]:>8.1f}"
                  f"   ({time.strftime('%H:%M:%S')})")
        tab[arm], best[arm] = row, max(row, key=row.get)
        print(f"  ▶ {arm:<12s} E*={best[arm]}  {row}")

    print(f"\n{'팔':<12s} " + " ".join(f"{e:>8d}ep" for e in grid) + f" {'E*':>5s}")
    print("-" * (13 + 11 * len(grid) + 6))
    for arm in arms:
        print(f"{arm:<12s} " + " ".join(f"{tab[arm][e]:>10.1f}" for e in grid)
              + f" {best[arm]:>5d}")
    print("\n  E* 요약:", best)
    return best, tab


# ──────────────────────────────────────────────────────────────────────
#  ④ 시드 산포 — **설계의 검출력**을 재는 것이지 성능을 재는 게 아니다
# ──────────────────────────────────────────────────────────────────────
# 앵커 실측(fold2024): emb4 5시드 단독 sd **22.0**, fold2023 은 **28.9**.
# 계획했던 2시드 설계로는 그 위에서 팔 간 ±20 을 못 읽는다. 규율을 다 지켜놓고
# 동전 던지기를 하게 된다. [[preregister-power-not-just-thresholds]]
#
# ⚠️ MLP 의 22 를 트랜스포머에 그대로 가져다 쓰면 안 된다 — 노이즈 바닥은 학습기마다
#    다르다 [[noise-learner-specific]]. 그래서 **이 학습기에서 직접 잰다.**
# 출력: ① 단독 sd  ② k 시드 평균 점수가 시드 조합에 따라 얼마나 흔들리는가
#       ②가 판정에 쓰이는 양이다 (팔마다 k 시드 평균 로짓으로 비교하므로).
def run_seedspread(df, fold, soft, over, arm="full", seeds=(42, 7, 2024, 1234, 31337),
                   epochs=4, name_tag=""):
    from itertools import combinations

    A, C, y, w, B, Cva, yva, num_cols, sizes = fold_arrays(df, fold, soft=soft)
    ya = fold_aux(df, fold, soft) if over.get("aux") else None
    # `use_delta` 는 cfg 가 아니라 별도 키다 (모델 구조가 아니라 **학습 타깃**을 바꾼다)
    off, zva = (fold_delta(df, fold, soft) if over.pop("use_delta", False)
                else (None, None))
    if ya is not None and len(ya) != len(A):
        raise ValueError(f"보조 라벨 {len(ya)} != 학습행 {len(A)} — 정렬이 깨졌다")
    rng = np.random.default_rng(zlib.crc32(f"{arm}|{fold}".encode()) & 0xFFFFFFFF)
    # `fgtt@X` 는 마스크가 아니라 **풀링**이다 (토큰을 그룹으로 접는다). 둘은
    # 배타적이다 — 접고 나면 헤드가 자를 그룹이 애초에 없다.
    pool = arm_pool(arm, num_cols)
    mask = (None if pool is not None
            else arm_mask(arm, num_cols, rng, cls_key=DEFAULT["cls_key"])[0])
    print(f"\n=== 시드 산포  fold{fold} · 팔 {arm} · {epochs}에폭 · {len(seeds)}시드"
          + (f" · {name_tag}" if name_tag else "") + " ===")
    _cc = dict(DEFAULT, **over)
    print(f"  d={_cc['d']} dh={_cc['dh']} inner={_cc['inner']} depth={_cc['depth']}"
          f" ffn_mult={_cc['ffn_mult']} lr={_cc['lr']} wd={_cc['wd']}"
          f" drop={_cc['p_attn']}/{_cc['p_ffn']}"
          f" aux={_cc['aux']}")

    zs = []
    for s in seeds:
        _, curve, _ = train(A, C, y, w, B, Cva, sizes, mask, seed=s, yva=yva,
                            verbose=False, pool=pool, y_aux=ya,
                            offset=off, zadd=zva, **dict(over, epochs=epochs))
        zs.append(curve[-1])
        print(f"    seed{s:<6d} "
              f"{mlp.score(curve[-1] if zva is None else curve[-1] + zva, yva):>8.1f}"
              f"   ({time.strftime('%H:%M:%S')})")
    _sc = (lambda z: mlp.score(z, yva)) if zva is None else           (lambda z: mlp.score(z + zva, yva))
    solo = np.array([_sc(z) for z in zs])
    print(f"  ① 단독      평균 {solo.mean():>7.1f}  sd {solo.std(ddof=1):>5.1f}"
          f"  범위 {solo.max()-solo.min():>5.1f}")

    for k in (2, 3, 5):
        if k > len(zs):
            continue
        vals = np.array([_sc(np.mean([zs[i] for i in c], axis=0))
                         for c in combinations(range(len(zs)), k)])
        # 이 sd 가 **판정에 쓰이는 노이즈**다. 두 팔의 차이는 그 √2 배로 흔들린다.
        print(f"  ② {k}시드평균  평균 {vals.mean():>7.1f}  sd {vals.std(ddof=1):>5.1f}"
              f"  → 팔 간 차이의 SE ≈ {vals.std(ddof=1)*np.sqrt(2):>5.1f}"
              f"   ({len(vals)}개 조합)")
    tag = arm.replace("@", "_") + (f"_{name_tag}" if name_tag else "")
    # ⚠️ 저장하는 z 는 **δ 를 안 더한 원본**이다 (다른 팔과 같은 규약).
    #    δ 는 분석 쪽에서 더한다 — 안 그러면 파일마다 규약이 갈린다.
    np.savez(CACHE / f"z_ftt_{tag}_fold{fold}_ep{epochs}.npz",
             z=np.mean(zs, axis=0), yva=yva,
             **{f"z_s{s}": z for s, z in zip(seeds, zs)})
    return zs, yva


def run_screen(df, fold, soft, arm="sem8"):
    A, C, y, w, B, Cva, yva, num_cols, sizes = fold_arrays(df, fold, soft=soft)
    rng = np.random.default_rng(20260830 + fold)
    mask, _ = arm_mask(arm, num_cols, rng)
    print(f"\n=== 정규화 스크리닝  fold {fold} · 팔 {arm} · soft={soft} ===")
    for name, cfg in SCREEN.items():
        _, curve, npar = train(A, C, y, w, B, Cva, sizes, mask, seed=42,
                               yva=yva, verbose=True, **cfg)
        sc = [mlp.score(z, yva) for z in curve]
        print(f"  {name:<8s} 마지막 {sc[-1]:>8.1f}   곡선 "
              f"{'  '.join(f'{v:.0f}' for v in sc)}   ({npar:,} 파라미터)")


# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=0, help="스모크용 행 제한")
    ap.add_argument("--epochs", type=int, default=DEFAULT["epochs"])
    ap.add_argument("--folds", type=int, nargs="*", default=FOLDS)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--arms", nargs="*",
                    default=["full", "sem8", "sem8ov", "subj3",
                             "shuf@sem8", "shuf@subj3"])
    ap.add_argument("--cont", default=DEFAULT["cont"])
    ap.add_argument("--lr", type=float, default=DEFAULT["lr"])
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--wd", type=float, default=DEFAULT["wd"])
    ap.add_argument("--no-soft", action="store_true", help="soft label 끄기")
    ap.add_argument("--screen", action="store_true", help="정규화 스크리닝만")
    ap.add_argument("--budget", action="store_true", help="에폭 예산 격자만")
    ap.add_argument("--budget-arms", nargs="*", default=None)
    ap.add_argument("--budget-grid", type=int, nargs="*", default=None)
    ap.add_argument("--anchor", action="store_true", help="배포 MLP 앵커만")
    ap.add_argument("--seedspread", action="store_true", help="시드 산포(검출력)만")
    ap.add_argument("--spread-arm", default="full")
    ap.add_argument("--spread-seeds", type=int, nargs="*", default=None)
    # ── 크기 축 (2026-08-30 사용자 요청: d 와 ffn_mult 를 잰다) ──
    # ⚠️ `d` 는 **토큰 개수가 아니라 토큰 하나의 두께**다. 토큰은 79 개로 고정.
    #    수치 71 개를 71×d 차원으로 펴는 것이라 §7.26(PLE 73→381 기각)의 방향이고
    #    d=64 는 그보다 12 배 극단적이다 — 지금 −68 핸디캡의 유력 용의자.
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--ffn-mult", type=float, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--inner", type=int, default=None)
    ap.add_argument("--tag", default="", help="z 저장 파일명 접미사 (설정 구분용)")
    # 보조 헤드 λ. 배포 MLP 는 1.0 을 쓴다 (`config.MLP_AUX_LAMBDA`).
    # ⚠️ 이걸 안 켜면 트랜스포머는 배포 MLP 와 **손실이 다른 모델**이다.
    ap.add_argument("--aux", type=float, default=None)
    # δ 축을 **학습 offset 으로** 넣는다 (사후 가산이 아니라). 채점도 δ 자리에서.
    ap.add_argument("--delta", action="store_true")
    ap.add_argument("--no-anchor", action="store_true", help="본 실험에서 앵커 생략")
    ap.add_argument("--anchor-noaux", action="store_true", help="앵커에서 보조 헤드 끄기")
    # 예산 격자에서 `full` 과 위약이 갈리면 계열별로 다른 에폭을 준다.
    ap.add_argument("--epochs-full", type=int, default=None)
    ap.add_argument("--epochs-mask", type=int, default=None)
    a = ap.parse_args()

    from config import TRAIN_CSV
    from data import load_csv

    soft = not a.no_soft
    t0 = time.time()
    over = dict(epochs=a.epochs, cont=a.cont, lr=a.lr, wd=a.wd)
    if a.dropout is not None:
        over.update(p_attn=a.dropout, p_ffn=a.dropout)
    for _k, _v in (("d", a.d), ("ffn_mult", a.ffn_mult), ("depth", a.depth),
                   ("inner", a.inner), ("aux", a.aux),
                   ("use_delta", True if a.delta else None)):
        if _v is not None:
            over[_k] = _v

    print(f"FT-Transformer 헤드 분할  —  헤드 {N_HEAD}개 (전역 헤드 0 · 완전 분할)")
    print(f"  팔 {a.arms}\n  folds {a.folds}  seeds {a.seeds}  soft={soft}  "
          f"cfg {over}")
    print("  base: src/foldcache.py fold 하네스 (챔피언 층 아님 — §7.207 층 격차 감수)")

    df = None

    def need_df(fold):
        nonlocal df
        import foldcache as fc
        if not (CACHE / f"fold{fold}_{'soft' if soft else 'raw'}_{fc._key()}.npz").exists() \
                and df is None:
            print(f"  train.csv 로드 + 피처 빌드 … ({time.time()-t0:.0f}s)")
            df = load_csv(TRAIN_CSV)

    if a.screen or a.budget or a.anchor or a.seedspread:
        for fold in a.folds:
            need_df(fold)
            if a.screen:
                run_screen(df, fold, soft)
            if a.anchor:
                run_anchor(df, fold, soft, seeds=tuple(a.seeds),
                           use_aux=not a.anchor_noaux)
            if a.seedspread:
                run_seedspread(df, fold, soft, over, arm=a.spread_arm,
                               epochs=a.epochs, name_tag=a.tag,
                               **({} if a.spread_seeds is None
                                  else {"seeds": tuple(a.spread_seeds)}))
            if a.budget:
                run_budget(df, fold, soft, over,
                           arms=tuple(a.budget_arms or BUDGET_ARMS),
                           grid=tuple(a.budget_grid or BUDGET_GRID),
                           seeds=tuple(a.seeds))
        return

    # 팔 계열별 에폭. 지정이 없으면 전 팔 공통(`--epochs`).
    ep_full = a.epochs_full or a.epochs
    ep_mask = a.epochs_mask or a.epochs

    res = {}
    for fold in a.folds:
        need_df(fold)
        A, C, y, w, B, Cva, yva, num_cols, sizes = fold_arrays(
            df, fold, a.rows or None, soft=soft)
        print(f"\n=== valid={fold} ===  train {len(A):,}행 · valid {len(B):,}행 · "
              f"수치 {len(num_cols)} + 범주 {len(sizes)}   ({time.time()-t0:.0f}s)")
        for name, (groups, _) in SCHEMES.items():
            check_cover(groups, num_cols, tag=name)

        row = {}
        if not a.no_anchor:
            # ⚠️ 앵커 없이 트랜스포머 절대 점수를 읽지 않는다. 이 층(foldcache +
            #    soft label)의 BSS 눈금은 배포 층과 다르므로 비교 대상이 필요하다.
            z_anc, _ = run_anchor(df, fold, soft, seeds=tuple(a.seeds))
            row["anchor_emb4"] = dict(solo=mlp.score(z_anc, yva),
                                      per_seed=[], npar=0, z=z_anc, keys=0.0)
        for arm in a.arms:
            # 위약 rng 는 **팔마다 새로 만든다** — 실행 순서가 바뀌어도 같은 위약이
            # 나오게. 안 그러면 팔 목록을 바꿀 때마다 위약이 달라져 대조가 깨진다.
            # ⚠️ 파이썬 `hash()` 는 문자열에 대해 프로세스마다 다르다(PYTHONHASHSEED).
            #    그걸 쓰면 재실행 때 위약이 바뀌어 쌍체 비교가 조용히 깨진다.
            rng = np.random.default_rng(
                zlib.crc32(f"{arm}|{fold}".encode()) & 0xFFFFFFFF)
            mask, plan = arm_mask(arm, num_cols, rng, cls_key=DEFAULT["cls_key"])
            ep = ep_full if arm == "full" else ep_mask
            zs, npar = [], 0
            for s in a.seeds:
                _, curve, npar = train(A, C, y, w, B, Cva, sizes, mask, seed=s,
                                       yva=yva, verbose=True,
                                       **dict(over, epochs=ep))
                zs.append(curve[-1])
                print(f"    [{arm}/seed{s}] {mlp.score(curve[-1], yva):>8.1f}"
                      f"   ({time.time()-t0:.0f}s)")
            zm = np.mean(zs, axis=0)
            row[arm] = dict(solo=mlp.score(zm, yva),
                            per_seed=[mlp.score(z, yva) for z in zs],
                            npar=npar, z=zm,
                            keys=(float(np.mean([k for _, k in plan])) if plan
                                  else len(num_cols) + len(sizes) + 1))
            print(f"  {arm:<12s} 시드평균 {row[arm]['solo']:>8.1f}"
                  f"   (헤드당 key {row[arm]['keys']:.1f} · 파라미터 {npar:,})")
        res[fold] = row

    # ── 판정표 ────────────────────────────────────────────────────────
    shown = ([] if a.no_anchor else ["anchor_emb4"]) + list(a.arms)
    print(f"\n{'팔':<12s} " + " ".join(f"{f:>10d}" for f in a.folds)
          + f" {'평균':>10s} {'시드폭':>8s}")
    print("-" * (13 + 11 * len(a.folds) + 20))
    for arm in shown:
        v = np.array([res[f][arm]["solo"] for f in a.folds])
        ps = [res[f][arm]["per_seed"] for f in a.folds]
        sd = (np.mean([np.std(p, ddof=1) for p in ps])
              if len(a.seeds) > 1 and all(len(p) > 1 for p in ps) else float("nan"))
        print(f"{arm:<12s} " + " ".join(f"{x:>10.1f}" for x in v)
              + f" {v.mean():>10.1f} {sd:>8.1f}")

    print("\n=== 사전 등록 판정 ===")
    print("  ① scheme > full        분할이 일한다")
    print("  ② scheme > shuf@scheme 그룹 **구성**이 일한다 (①만이면 그냥 마스킹 효과)")
    print("  둘 다 fold 전부에서 부호가 같아야 한다. 시드폭보다 작은 차이는 안 읽는다.")
    for arm in a.arms:
        if arm == "full" or arm.startswith("shuf@"):
            continue
        for ref in ("full", f"shuf@{arm}"):
            if ref not in a.arms:
                continue
            d = np.array([res[f][arm]["solo"] - res[f][ref]["solo"] for f in a.folds])
            same = bool(np.all(d > 0) or np.all(d < 0))
            print(f"  {arm} − {ref:<12s}: {np.round(d,1)}  평균 {d.mean():+.1f}"
                  f"   부호일치 {'O' if same else 'X'}")

    for f in a.folds:
        ks = list(shown)
        print(f"\n  fold{f} 팔 간 로짓 상관"
              + ("   (앵커 열이 낮을수록 앙상블 멤버로 값어치가 있다 — §7.27)"
                 if not a.no_anchor else ""))
        for i in range(len(ks)):
            print("    " + f"{ks[i]:<12s}" + " ".join(
                f"{np.corrcoef(res[f][ks[i]]['z'], res[f][ks[j]]['z'])[0,1]:>6.3f}"
                for j in range(len(ks))))

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
