"""§7.163 — **NODE** (Popov·Morozov·Babenko, ICLR 2020, arXiv:1909.06312).

미분가능 oblivious 결정트리(ODT) 몸통. `mlp.Net` 의 `self.net` 자리에 꽂힌다
(`u` (B, d) → 로짓 (B,)). `make_adapters` 와 같은 규약이다.

### 왜 이 저장소에 이게 없었나 — 기각이 아니라 **목록에 오른 적이 없다**

§7.32 의 16개 항목 전수 대조는 LG Aimers 강의자료 6종을 목록으로 썼는데,
그 자료에 **미분가능 트리 계열이 통째로 없다**. `git log --all -S "NODE"` 0건,
`grep -ri oblivious src/` 0건. 감사의 사각지대다.

### 기전 — 스택에 비어 있는 칸

    LightGBM    하드 · 축정렬 · 탐욕 성장               배포 α = 0.15
    CatBoost    하드 · 축정렬 · **oblivious** · 탐욕     배포 α = 0.10
    MLP/TabM    부드러움 · end-to-end 경사              배포 α = 0.75
    NODE        **oblivious · end-to-end 경사**         ← 빈칸

CatBoost 와 같은 구조(oblivious)를 **탐욕이 아니라 경사로** 배운다. §7.14 에서
CatBoost 가 corr .985 로 +3.5 밖에 못 준 이유가 "같은 탐욕 분할을 두 번 하기
때문"이라면 여기서 갈려야 하고, "oblivious 구조 자체가 이미 포화"라면 여기서도
같은 corr 이 나온다. **어느 쪽이든 답이 나오는 자리다.**

### 사망 위험 (사전에 적는다)

  §7.25 "폭은 재앙"  — 용량 법칙. m=128·D=6 이면 파라미터 66k 로 MLP 몸통(26k)의
                      2.5배다. 그래서 **용량-반응**으로 두 점을 잰다 (§7.151 교훈:
                      단일 지점이면 또 "미판정" 으로 끝난다).
  Brier             — 하드 게이트는 출력을 계단함수로 만든다. proper scoring rule
                      에서 계단은 캘리브레이션을 깎는다. `gate="sigmoid"` 대조군이
                      이 축을 가른다.

### 파라미터 예산 (d = 99 실측: 수치 71 + 임베딩 7×4)

    F  피처선택 로짓   d × m × D
    b  임계값          m × D
    lt log 온도        m × D
    R  잎 응답         m × 2^D
    ---------------------------------------------
    m=64,  D=4        25,344 +   512 + 1,024 =  26,880   ← MLP 몸통 26,113 과 동급
    m=128, D=6        76,032 + 1,536 + 8,192 =  85,760   ← 3.3배. 용량 법칙 시험대

### 실측 비용 (RTX 5060 Laptop, B=8192, 9 에폭 = 180 step × 9)

    m=64,  D=4     step  13ms  → 멤버당 20s   학습 peak 0.26 GiB
    m=128, D=6     step  53ms  → 멤버당 86s   학습 peak 1.37 GiB
    추론 65536행 청킹 결과가 참조와 `allclose` True, peak 0.43 GiB

자기검사:  .venv-gpu/Scripts/python.exe src/nodenet.py
"""
import math

import torch
import torch.nn as nn
from torch.autograd import Function

# choice 텐서 (B, m, 2^D) 의 원소 수 상한. 넘으면 배치를 쪼갠다.
# 왜 필요한가: `mlp.infer` 가 batch=65536 으로 부른다. m=128·D=6 이면
# 65536 × 128 × 64 = 5.4e8 원소 = 2.1GB — 그냥 죽는다.
# 주의: **no_grad 일 때만** 쪼갠다. autograd 가 켜져 있으면 중간값을 어차피 다
#      들고 있으므로 쪼개도 메모리가 안 줄고 그래프만 길어진다.
CHUNK_CELLS = 24_000_000


# ─────────────────────────────────────────────────────────────────────
#  entmax_1.5  (Peters·Niculae·Martins, ACL 2019, arXiv:1905.05702)
# ─────────────────────────────────────────────────────────────────────
def _entmax_tau(z, dim):
    """정확 해 tau*. 정렬 기반 O(d log d).  p_i = [(z_i − tau)_+]^2, Σp = 1"""
    zs, _ = torch.sort(z, dim=dim, descending=True)
    shape = [1] * z.dim()
    shape[dim] = -1
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype).view(shape)
    mean = zs.cumsum(dim) / k
    mean_sq = (zs ** 2).cumsum(dim) / k
    ss = k * (mean_sq - mean ** 2)
    delta = torch.clamp((1 - ss) / k, min=0)
    tau = mean - torch.sqrt(delta)
    support = (tau <= zs).sum(dim, keepdim=True)
    return tau.gather(dim, support - 1)


class _Entmax15(Function):
    """forward 는 정확 해, backward 는 해석적 야코비안.

        J v = s ⊙ (v − ⟨s,v⟩/⟨s,1⟩ · 1),   s = p^(2−α) = sqrt(p)

    자동미분으로 정렬·gather 를 타고 내려가면 틀린다 — 그래서 직접 쓴다.
    `_selftest` 의 gradcheck 가 이 두 줄을 검산한다.
    """

    @staticmethod
    def forward(ctx, z, dim):
        ctx.dim = dim
        z = z - z.max(dim=dim, keepdim=True).values      # 수치 안정
        z = z / 2                                        # (α−1) = 0.5
        p = torch.clamp(z - _entmax_tau(z, dim), min=0) ** 2
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, g):
        (p,) = ctx.saved_tensors
        s = p.sqrt()
        dz = g * s
        q = dz.sum(ctx.dim, keepdim=True) / s.sum(ctx.dim, keepdim=True)
        return dz - q * s, None


def entmax15(z, dim=-1):
    return _Entmax15.apply(z, dim)


# ─────────────────────────────────────────────────────────────────────
#  분할 게이트  c → [0, 1]
# ─────────────────────────────────────────────────────────────────────
def _g_entmax(c):
    """2원소 entmax_1.5 의 **닫힌 해**. 원논문 NODE 의 게이트.

    entmax15([c, 0])[0] 을 풀면 δ = c/2 에 대해

        g = ((δ + sqrt(2 − δ²)) / 2)²,   δ ∈ [−1, 1]

    이고 |c| ≥ 2 에서 정확히 0 / 1 로 **포화**한다 (경사도 정확히 0).
    d 원소 일반형을 부르면 길이 2 짜리를 정렬하느라 10배 느리다.
    """
    d = torch.clamp(c * 0.5, -1.0, 1.0)
    return ((d + torch.sqrt(2.0 - d * d)) * 0.5) ** 2


def _g_sigmoid(c):
    """대조군. 절대 포화하지 않는다 → ODT 가 "구조화된 MLP" 가 된다.

    이 팔이 `entmax` 를 이기면 이득의 출처는 **트리 구조가 아니라 곱셈 게이팅**
    이라는 뜻이다. 그러면 §7.26(입력차원 단조 법칙) 쪽 해석으로 넘어간다.
    """
    return torch.sigmoid(c)


GATES = {"entmax": _g_entmax, "sigmoid": _g_sigmoid}


# ─────────────────────────────────────────────────────────────────────
#  ODT 몸통
# ─────────────────────────────────────────────────────────────────────
class NodeBody(nn.Module):
    """oblivious 미분가능 트리 m 개. 출력은 **트리 평균** + 편향.

    ⚠️ 합이 아니라 평균이다. LightGBM `boosting=rf` 사고(raw_score 가 평균이
       아니라 합이라 로짓이 300배 부풀고 BSS −83,000) 와 정확히 같은 자리다.
       `_selftest` 가 init 로짓 std 를 MLP 대역과 대조한다.
    """

    def __init__(self, d, m, depth, gate="entmax", leaf_std=1.0, n_aux=0):
        super().__init__()
        self.d, self.m, self.D = d, m, depth
        self.gate = GATES[gate]
        self.F = nn.Parameter(torch.randn(d, m * depth) * (1.0 / math.sqrt(d)))
        self.b = nn.Parameter(torch.zeros(m, depth))
        self.lt = nn.Parameter(torch.zeros(m, depth))
        self.R = nn.Parameter(torch.randn(m, 1 << depth) * leaf_std)
        self.bias = nn.Parameter(torch.zeros(1))
        # ⚠️ 보조 잎은 **맨 마지막**에 만든다. `R` 을 (m, L, 1+n_aux) 로 넓히면
        #    randn 소비량이 달라져 aux OFF 팔의 초기값이 통째로 바뀌고, 이미
        #    캐시된 §7.163 실행과 대조가 깨진다. 별도 파라미터로 뒤에 붙이면
        #    aux ON/OFF 두 팔의 fsel·R 이 **비트 단위로 같다** — 2×2 가 재는 것이
        #    "aux 효과" 지 "aux 효과 + 다른 초기값" 이 아니게 된다.
        self.R_aux = (nn.Parameter(torch.randn(m, 1 << depth, n_aux) * leaf_std)
                      if n_aux else None)
        self.register_buffer("ready", torch.zeros((), dtype=torch.bool))

    # ── ★ 사용자 결정 지점 ────────────────────────────────────────────
    @torch.no_grad()
    def _init_thresholds(self, f):
        """데이터 인지 초기화. `f` 는 첫 배치의 분할값 (B, m, D).

        `self.b` (m, D) 와 `self.lt` (m, D) 를 **제자리에서** 채운다
        (`self.b.copy_(...)` / `self.lt.copy_(...)`). `lt` 는 log τ 다:
        게이트 입력이 `c = (f − b)·exp(−lt)` 이고 `_g_entmax` 가 |c| ≥ 2 에서
        포화하므로 **`lt` 가 초기 포화 비율을 정한다.**

        NODE 재구현이 실제로 죽는 자리가 여기다. τ 가 너무 작으면 첫 스텝부터
        전 게이트가 포화해 경사가 0 이고, 너무 크면 모든 게이트가 0.5 근처라
        choice 텐서가 균등해져 출력이 "잎 평균 + 작은 흔들림" = 사실상 선형이 된다.

        ── ⚠️ 처음에 "용량 법칙 vs 다양성 법칙이 맞선다" 고 적었는데 **틀렸다**

        fold2023 실데이터(8192행) 위에서 세 후보를 재보니 **트리간 출력 상관이
        전부 ≈ 0** 이었다. `R ~ N(0,1)` 이 잎값을 트리마다 독립으로 뽑기 때문에
        임계값을 어디에 놓든 초기 다양성은 이미 최대다.
        → §7.27·§7.120 의 다양성 법칙은 **이 노브에 할 말이 없다.**
          남는 축은 포화율 하나고, 그건 다양성이 아니라 **정규화 다이얼**이다.

        후보 (fold2023 실측. 포화% 는 m=64,D=4 / m=128,D=6):

          A. 원논문   b ~ f 의 무작위 분위수 U(0,1),  τ = mean|f − b|
                      포화 9.0 / 8.9 %   b흩뿌림 0.70   corr −0.007 / +0.000
                      → 가장 트리답다. 원논문 설정이라 음수가 나와도 **내 튜닝이
                        아니라 방법의 책임**으로 귀속된다. `gate="sigmoid"` 대조군
                        과의 대비도 최대라 4차 관문이 가장 날카롭다.
          B. 중앙값   b = median(f),                 τ = mean|f − b|
                      포화 10.9 / 11.0 %  b흩뿌림 0.15   corr −0.002 / +0.002
                      → A 의 포화율에 C 의 흩뿌림. 어느 논증도 이걸 고르지 않는다.
          C. 좁은 띠  b ~ f 의 40~60 분위수,         τ = std(f)
                      포화 4.7 / 4.7 %   b흩뿌림 0.19   corr −0.003 / +0.002
                      → 가장 부드럽다 = 가장 조인 쪽. §7.25 의 "조이는 수가 이긴다"
                        가 지지하지만, NSIG(sigmoid) 와 구별이 흐려진다.

        ⚠️ τ 에 하한을 둔다 (`clamp_min(1e-3)`). 어떤 (tree, level) 은 entmax 가
           피처 하나에 몰빵해 f 의 분산이 0 이 될 수 있고, τ=0 → c=±inf → NaN 이다.
        ⚠️ 여기서 `torch.Generator` 를 새로 만들지 마라. `torch.manual_seed(seed)`
           가 이미 잡혀 있고, 새로 만들면 팔 사이 쌍체 비교가 깨진다 (§7.161b).

        f: (B, m, D) float32   →   반환 없음 (제자리 수정)

        ── 채택: **A (원논문)**. 정규화 논증이 아니라 실험 설계 논증이다.
           ① 원논문 설정이면 음수가 나와도 "NODE 가 졌다" 로 귀속된다. C 로 조여
              놓고 지면 "내가 잘못 조인 건가" 가 안 풀려 §7.163 을 못 닫는다.
           ② 포화 9% vs NSIG(sigmoid, 포화 0%) 의 대비가 최대 → 4차 관문이 날카롭다.
           ③ 조이는 축은 N64 vs N128 용량-반응이 이미 따로 재고 있다.
        """
        B = f.shape[0]
        # 분위수는 정렬 + gather 로 뽑는다. torch.quantile 은 입력 원소 수에
        # 상한(2^24)이 있어 B·m·D 가 이걸 넘으면 조용히 죽는다.
        q = torch.rand(self.m, self.D, device=f.device, dtype=f.dtype)
        fs, _ = torch.sort(f, dim=0)                          # (B, m, D)
        i = (q * (B - 1)).round().long().clamp_(0, B - 1)
        b = fs.gather(0, i.unsqueeze(0)).squeeze(0)           # (m, D)
        # τ = mean|f − b|. 하한이 없으면 entmax 가 피처 하나에 몰빵한 (tree, level)
        # 에서 f 의 분산이 0 → τ=0 → c=±inf → NaN 이다.
        tau = (f - b).abs().mean(0).clamp_min(1e-3)
        self.b.copy_(b)
        self.lt.copy_(torch.log(tau))

    # ──────────────────────────────────────────────────────────────────

    def _split(self, u):
        """(B, d) → (B, m, D) 분할값 f = u · entmax15(F)

        `entmax15(F, dim=0)` 이 피처 축 위에서 희소해진다 — 대부분의 (tree, level)
        이 피처 **한두 개**만 보게 되고, 그게 이 모델을 "축정렬 트리" 로 만든다.
        여기를 softmax 로 바꾸면 oblique(선형결합) 트리가 되고 MLP 에 가까워진다.
        """
        w = entmax15(self.F, dim=0)                     # (d, m*D)
        return (u @ w).view(-1, self.m, self.D)

    def _reduce(self, u, want_aux=False):
        f = self._split(u)
        if not bool(self.ready):
            self._init_thresholds(f)
            self.ready.fill_(True)
        c = (f - self.b) * torch.exp(-self.lt)          # (B, m, D)
        g = self.gate(c)
        # choice 텐서를 깊이마다 한 칸씩 넓힌다: (B, m, 2^i) → (B, m, 2^(i+1)).
        # oblivious 라 레벨마다 게이트가 **트리 전체에 하나**다 — 이게 일반
        # 이진트리(GRANDE)보다 2^D 배 싼 이유다.
        ch = torch.ones(u.shape[0], self.m, 1, device=u.device, dtype=u.dtype)
        for i in range(self.D):
            gi = g[:, :, i].unsqueeze(-1)               # (B, m, 1)
            ch = torch.cat([ch * gi, ch * (1.0 - gi)], dim=-1)
        z = (ch * self.R).sum(-1).mean(-1)              # 트리 평균 → (B,)
        if not want_aux:
            return z
        # einsum 으로 (B, m, L, n_aux) 중간값을 안 만든다 — m=128·D=6·B=8192 에서
        # 그 텐서만 537MB 다. 주 경로는 **손대지 않는다**: 축약 순서를 바꾸면
        # 부동소수점 결과가 미세하게 달라져 캐시와 안 맞는다.
        za = torch.einsum("bml,mla->ba", ch, self.R_aux) / self.m
        return z, za

    def forward(self, u, want_aux=False):
        if want_aux:                       # 학습 전용 — 항상 grad 가 켜져 있다
            z, za = self._reduce(u, want_aux=True)
            return z + self.bias, za
        cells = u.shape[0] * self.m * (1 << self.D)
        if torch.is_grad_enabled() or cells <= CHUNK_CELLS:
            return self._reduce(u) + self.bias
        step = max(1, CHUNK_CELLS // (self.m * (1 << self.D)))
        out = [self._reduce(u[i:i + step]) for i in range(0, len(u), step)]
        return torch.cat(out) + self.bias


def build(d, cfg, n_aux=0):
    """`mlp.Net.__init__` 이 부른다. cfg = {"m": 64, "depth": 4, "gate": "entmax"}"""
    return NodeBody(d, int(cfg["m"]), int(cfg["depth"]),
                    gate=cfg.get("gate", "entmax"),
                    leaf_std=float(cfg.get("leaf_std", 1.0)), n_aux=n_aux)


# ─────────────────────────────────────────────────────────────────────
#  자기검사 — 메모리 규칙: "간이 재구현은 알려진 수치 재현부터"
# ─────────────────────────────────────────────────────────────────────
def _selftest():
    import sys
    try:                        # 콘솔 기본이 cp949 라 ⊙·− 같은 글자에서 죽는다
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    torch.manual_seed(0)
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'OK ' if cond else 'FAIL'}  {name}   {detail}")

    # ① entmax15 는 단체(simplex) 위에 있다
    z = torch.randn(200, 73, dtype=torch.float64)
    p = entmax15(z, dim=-1)
    chk("Σp = 1", torch.allclose(p.sum(-1), torch.ones(200, dtype=torch.float64)),
        f"max|Σp−1| = {(p.sum(-1) - 1).abs().max():.2e}")
    chk("p ≥ 0", (p >= 0).all())
    chk("희소하다 (softmax 는 안 그렇다)", (p == 0).float().mean() > 0.5,
        f"0 비율 {(p == 0).float().mean():.3f} vs softmax 0.000")

    # ② 균등 입력 → 균등 출력 (알려진 수치)
    p0 = entmax15(torch.zeros(1, 10, dtype=torch.float64), dim=-1)
    chk("entmax15(0) = 1/d", torch.allclose(p0, torch.full((1, 10), 0.1,
                                                           dtype=torch.float64)))

    # ③ 해석적 backward vs 수치미분
    zg = torch.randn(4, 12, dtype=torch.float64, requires_grad=True)
    chk("gradcheck", torch.autograd.gradcheck(lambda t: entmax15(t, -1), (zg,),
                                              eps=1e-6, atol=1e-7))

    # ④ 닫힌 해 게이트 == 일반형 entmax15([c, 0])
    c = torch.linspace(-4, 4, 401, dtype=torch.float64)
    gen = entmax15(torch.stack([c, torch.zeros_like(c)], -1), dim=-1)[:, 0]
    chk("닫힌해 == 일반형", torch.allclose(_g_entmax(c), gen, atol=1e-12),
        f"max|Δ| = {(_g_entmax(c) - gen).abs().max():.2e}")
    chk("|c| ≥ 2 에서 포화", (_g_entmax(torch.tensor([-2.0, 2.0])) ==
                              torch.tensor([0.0, 1.0])).all())
    chk("g(0) = 1/2", abs(float(_g_entmax(torch.tensor(0.0, dtype=torch.float64)))
                          - 0.5) < 1e-12,
        f"float32 판은 {float(_g_entmax(torch.tensor(0.0))):.8f} — sqrt 반올림")

    # ⑤ ★ 로짓 스케일 — LightGBM rf 모드 사고(합 vs 평균, BSS −83,000) 자리
    #
    # MLP 대역(0.2~0.6)과 맞대는 것은 **틀린 자**다. 잎이 iid N(0,1) 이라 트리 하나의
    # 분산은 Σ_leaf c² 이고, 이건 레벨마다 인수분해된다:
    #
    #     Σ_leaf c² = Π_i (g_i² + (1−g_i)²)
    #
    # 게이트가 0.5 근처면 인수가 0.5 라 2^(−D) 지만, 옵션 A 는 임계값을 분위수
    # 전역에 흩뿌려 choice 가 균등보다 뾰족하다 — 실측 인수 0.655 (레벨당 1.31배).
    # 그래서 **가정하지 않고 트리별 출력에서 직접 잰다.** 이렇게 하면 이 검사가
    # 초기화 선택과 무관해지고, 검사하는 대상이 축약 산술 하나로 좁혀진다.
    d, B = 99, 4096
    u = torch.randn(B, d)
    for m_, D_ in ((64, 4), (128, 6)):
        torch.manual_seed(0)
        net = NodeBody(d, m_, D_)
        zz = net(u)
        n_par = sum(p_.numel() for p_ in net.parameters())
        with torch.no_grad():                      # 트리별 출력 (평균 전)
            f_ = net._split(u)
            g_ = net.gate((f_ - net.b) * torch.exp(-net.lt))
            ch = torch.ones(B, m_, 1)
            for i in range(D_):
                ch = torch.cat([ch * g_[:, :, i:i + 1],
                                ch * (1 - g_[:, :, i:i + 1])], -1)
            per_tree = (ch * net.R).sum(-1)                       # (B, m)
        # 트리가 서로 독립이고 축약이 평균이면 std 는 정확히 1/sqrt(m) 로 준다.
        # ⚠️ 분산은 **행 방향으로** 재야 한다 (`var(0)`). 전체 원소에 대고 재면
        #    트리별 평균 오프셋이 섞이는데, 그건 행 방향으로 상수라 std(z) 에
        #    기여하지 않는다 — 그대로 쓰면 예측이 17~35% 부풀어 검사가 헛돈다.
        vbar = float(per_tree.var(0).mean())
        pred = math.sqrt(vbar / m_)
        got = float(zz.std())
        # 밴드 25%: 트리는 **정확히** 독립이 아니다 (같은 u 를 보고 entmax 선택이
        # 겹친다). 실측 편차는 +11% / −10% 로 부호가 갈려 계통 오차가 아니다.
        # 이 검사가 잡으려는 것은 축약 사고의 m 배(64·128배)지 독립성 인증이 아니다.
        chk(f"std(z) m={m_} D={D_} == sqrt(트리분산 / m)",
            abs(got - pred) < 0.25 * pred,
            f"예측 {pred:.4f}  실측 {got:.4f}  "
            f"레벨인수 {vbar ** (1.0 / D_):.3f}  파라미터 {n_par:,}")

        # 합/평균 판별자. 비율이 m 이면 축약이 평균, 1 이면 이미 합이라 rf 사고다.
        ratio = float(per_tree.sum(-1).std() / per_tree.mean(-1).std())
        chk(f"축약이 합이 아니라 평균 (비율 == m={m_})", abs(ratio - m_) < 0.01 * m_,
            f"sum/mean std 비율 {ratio:.1f}")

    # ⑥ 청킹 경로가 통짜 계산과 같은 답을 준다 (mlp.infer 가 batch=65536)
    torch.manual_seed(0)
    net = NodeBody(d, 128, 6)
    with torch.no_grad():
        big = torch.randn(70_000, d)
        chk("청킹 == 통짜", torch.allclose(net(big), net._reduce(big) + net.bias,
                                           atol=1e-6),
            f"cells {70_000 * 128 * 64:,} > 상한 {CHUNK_CELLS:,}")

    print("\n  " + ("전부 통과" if ok else "★ 실패 있음"))
    return ok


if __name__ == "__main__":
    _selftest()
