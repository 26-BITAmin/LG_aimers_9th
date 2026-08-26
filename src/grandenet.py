"""§7.164 — **GRANDE** (Marton·Lüdtke·Bartelt·Stuckenschmidt, ICLR 2024,
arXiv:2309.17130). 경사 학습 하드 축정렬 결정트리 앙상블.

`mlp.Net` 의 `self.net` 자리에 꽂힌다 (`u` (B, d) → 로짓 (B,)). `nodenet` 과 같은 규약.

### ★ 설계 원칙 — NODE 와 **부품을 공유**해서 차이를 두 개로 좁힌다

공식 구현은 TF2/Keras 라 어차피 재구현이다. 그렇다면 §7.163(NODE) 과 **게이트
파라미터화를 똑같이** 맞추는 편이 값이 크다 — 그래야 두 실험의 차이가 방법의
차이지 내 구현 습관의 차이가 아니게 된다.

    공유하는 것   `c = (f − b)·exp(−lt)`, 데이터 인지 임계값 초기화(무작위 분위수),
                  잎값 N(0,1), 트리 축약, 청킹 규약
    GRANDE 만     ① **비-oblivious** — 레벨 하나에 게이트 하나가 아니라 노드마다 하나
                  ② **하드 STE** — forward 는 1[s>0], backward 는 sigmoid 경사
                  ③ **instance-wise weighting** — 행마다 추정기 가중이 다르다

그래서 판정이 이렇게 쪼개진다:

    G  vs N128e3    ①+② 의 값                (구조·경사 추정량)
    G  vs GU        ③ 의 값                   ← §7.45 가 죽은 바로 그 축
    G32 vs G64      용량-반응                  (§7.151)

### ⚠️ ③ 은 이 저장소에서 이미 한 번 죽은 축이다

§7.45 "입력 의존 α"(트리/MLP 귀납편향에 따라 행마다 블렌드 비중을 다르게) 는
**요일 대조 버킷이 같은 크기의 이득(+22.8)을 내면서** 죽었다. GRANDE 의 간판
기여가 정확히 그 모양이다. → `weighted=False` 팔(GU)을 반드시 같이 돌린다.
GU 는 위약이 아니라 **절제(ablation)** 다. 위약(무작위 가중)까지 필요하면
GU 가 이긴 뒤에 붙인다.

### ⚠️ §7.163 에서 배운 것을 미리 적용한다 — 에폭 예산을 두 개 돌린다

NODE 는 `MLP_VARIANTS["emb4"]` 에서 물려받은 `epochs=9` 에서 과적합했고
(정점 ep3, 최종이 정점보다 f2023 −46), 그걸 몰라서 실험을 두 번 돌렸다.
이번엔 ep3 팔과 ep9 팔을 처음부터 같이 넣는다.

### 파라미터 예산 (d = 99, 깊이 4 → 내부노드 15 · 잎 16)

    fsel  피처선택 로짓   E × I × d
    thr   분할 임계값     E × I × d        (선택된 피처의 것만 쓴다 — 원논문 규약)
    lt    log 온도        E × I
    leaf  잎값            E × L
    lw    잎 가중         E × L            (weighted=True 일 때만)
    ------------------------------------------------------------------
    E=32, D=4     47,520 + 47,520 + 480 + 512 + 512 =  96,544
    E=64, D=4     95,040 + 95,040 + 960 + 1,024 + 1,024 = 193,088

    비교: MLP 몸통 26,113 · NODE m128/D6 85,761 (ep3 에서 손해 없었다)

자기검사:  .venv-gpu/Scripts/python.exe src/grandenet.py
"""
import math

import torch
import torch.nn as nn

# (B, E, L) 원소 수 상한. nodenet 과 같은 규약 — no_grad 일 때만 쪼갠다.
CHUNK_CELLS = 24_000_000


# ─────────────────────────────────────────────────────────────────────
#  Straight-through 추정량
# ─────────────────────────────────────────────────────────────────────
def hard_onehot_ste(logits, dim=-1):
    """forward 는 argmax one-hot, backward 는 softmax 경사.

    이게 GRANDE 를 "축정렬" 로 만드는 부품이다 — 각 내부 노드가 피처를 **하나만**
    본다. softmax 그대로 쓰면 선형결합(oblique) 트리가 되고 MLP 에 가까워진다.
    """
    p = torch.softmax(logits, dim)
    hard = torch.zeros_like(p).scatter_(dim, logits.argmax(dim, keepdim=True), 1.0)
    return (hard - p).detach() + p


def hard_step_ste(s, lt):
    """forward 는 1[s > 0] (진짜 하드 분할), backward 는 sigmoid(s·e^−lt) 경사.

    ⚠️ 항등 경사 STE(Bengio 2013 원형)를 안 쓰는 이유: `s` 가 비유계라 경사가
       분할에서 얼마나 떨어졌든 똑같이 흐른다. 온도를 끼우면 `lt` 가 학습되면서
       경사 폭이 조절되고, **nodenet 의 게이트와 같은 파라미터화**가 된다.
       round(sigmoid(z)) == 1[z>0] 이므로 forward 는 lt 와 무관하게 정확히 하드다.
    """
    soft = torch.sigmoid(s * torch.exp(-lt))
    return (torch.round(soft) - soft).detach() + soft


# ─────────────────────────────────────────────────────────────────────
#  GRANDE 몸통
# ─────────────────────────────────────────────────────────────────────
class GrandeBody(nn.Module):
    """비-oblivious 하드 트리 E 개 + (선택) 행별 추정기 가중.

    ⚠️ 잎 마스크는 **정확히 one-hot** 이다 (하드 분할이므로). `_selftest` 가
       max == 1 · sum == 1 을 확인한다 — 여기가 깨지면 트리가 아니라 소프트
       혼합이고, 그러면 NODE 와 같은 것을 두 번 재는 셈이 된다.
    """

    def __init__(self, d, n_est, depth, weighted=True, leaf_std=1.0, n_aux=0):
        super().__init__()
        self.d, self.E, self.D = d, n_est, depth
        self.I = (1 << depth) - 1           # 내부 노드
        self.L = 1 << depth                 # 잎
        self.weighted = bool(weighted)
        self.fsel = nn.Parameter(torch.randn(n_est, self.I, d) * 0.1)
        self.thr = nn.Parameter(torch.zeros(n_est, self.I, d))
        self.lt = nn.Parameter(torch.zeros(n_est, self.I))
        self.leaf = nn.Parameter(torch.randn(n_est, self.L) * leaf_std)
        # 가중을 안 써도 파라미터는 만든다 — 만들지 않으면 위쪽 RNG 소비가 달라져
        # weighted True/False 두 팔의 fsel/leaf 초기값이 갈린다. 그러면 ③ 을
        # 분리하려던 절제가 "초기값도 다른 두 모델" 비교가 되어 버린다.
        # (torch.zeros 는 RNG 를 안 쓰지만 순서를 고정해 두는 편이 안전하다.)
        self.lw = nn.Parameter(torch.zeros(n_est, self.L))
        self.bias = nn.Parameter(torch.zeros(1))
        # ⚠️ nodenet 과 같은 규칙 — 보조 잎은 맨 마지막. aux ON/OFF 두 팔의
        #    fsel·thr·leaf 가 비트 단위로 같아야 2×2 가 aux 만 잰다.
        self.leaf_aux = (nn.Parameter(torch.randn(n_est, self.L, n_aux) * leaf_std)
                         if n_aux else None)
        self.register_buffer("ready", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def _init_thresholds(self, u):
        """데이터 인지 초기화. §7.163 이 채택한 옵션 A 와 **같은 규칙**이다.

        thr[e,i,f] ← 피처 f 의 무작위 분위수 (q ~ U(0,1))
        lt[e,i]    ← log(mean|s|),  s = 선택된 피처값 − 그 임계값

        NODE 에서는 `f = u·entmax(F)` 가 이미 합성된 값이라 그것의 분위수를 썼다.
        여기서는 노드가 **원 피처 하나**를 보므로 원 피처의 분위수가 같은 뜻이다.
        """
        B = u.shape[0]
        us, _ = torch.sort(u, dim=0)                          # (B, d)
        q = torch.rand(self.E, self.I, self.d, device=u.device, dtype=u.dtype)
        idx = (q * (B - 1)).round().long().clamp_(0, B - 1)   # (E, I, d)
        self.thr.copy_(us.gather(0, idx.reshape(-1, self.d)).reshape_as(idx))
        s = self._pre(u)                                      # (B, E, I)
        self.lt.copy_(torch.log(s.abs().mean(0).clamp_min(1e-3)))

    def _pre(self, u):
        """(B, d) → (B, E, I) 분할 여유값 s = x[선택피처] − 그 피처의 임계값"""
        p = hard_onehot_ste(self.fsel, dim=-1)                # (E, I, d) one-hot
        xs = torch.einsum("bd,eid->bei", u, p)
        th = (p * self.thr).sum(-1)                           # (E, I)
        return xs - th

    def _reduce(self, u, want_aux=False):
        s = self._pre(u)
        g = hard_step_ste(s, self.lt)                         # (B, E, I) ∈ {0,1}
        # 잎 마스크. **`stack`+`reshape` 여야 한다** — `cat` 을 쓰면 다음 레벨에서
        # 형제 노드가 인접하지 않아 힙 인덱스와 어긋난다. oblivious(nodenet) 에서는
        # 레벨당 게이트가 하나뿐이라 순서가 잎 재명명일 뿐이지만 여기서는 아니다.
        mask = torch.ones(u.shape[0], self.E, 1, device=u.device, dtype=u.dtype)
        off = 0
        for l in range(self.D):
            n_l = 1 << l
            gl = g[:, :, off:off + n_l]                       # (B, E, n_l)
            mask = torch.stack([mask * (1 - gl), mask * gl], -1).reshape(
                u.shape[0], self.E, n_l << 1)
            off += n_l
        out = (mask * self.leaf).sum(-1)                      # (B, E)
        if self.weighted:
            a = torch.softmax((mask * self.lw).sum(-1), -1)    # (B, E) 행별 가중
            z = (a * out).sum(-1)
        else:
            a = None
            z = out.mean(-1)
        if not want_aux:
            return z
        # 보조 헤드는 **같은 가중**을 쓴다 — 몸통(=선택된 잎)을 공유하는 것이
        # 다중과제의 요점이고, 가중을 따로 두면 공유가 반쪽이 된다.
        wm = mask * (a.unsqueeze(-1) if a is not None else (1.0 / self.E))
        return z, torch.einsum("bel,ela->ba", wm, self.leaf_aux)

    def forward(self, u, want_aux=False):
        if not bool(self.ready):
            self._init_thresholds(u)
            self.ready.fill_(True)
        if want_aux:                       # 학습 전용
            z, za = self._reduce(u, want_aux=True)
            return z + self.bias, za
        cells = u.shape[0] * self.E * self.L
        if torch.is_grad_enabled() or cells <= CHUNK_CELLS:
            return self._reduce(u) + self.bias
        step = max(1, CHUNK_CELLS // (self.E * self.L))
        return torch.cat([self._reduce(u[i:i + step])
                          for i in range(0, len(u), step)]) + self.bias


def build(d, cfg, n_aux=0):
    """`mlp.Net.__init__` 이 부른다. cfg = {"E":32,"depth":4,"weighted":True}"""
    return GrandeBody(d, int(cfg["E"]), int(cfg["depth"]),
                      weighted=bool(cfg.get("weighted", True)),
                      leaf_std=float(cfg.get("leaf_std", 1.0)), n_aux=n_aux)


# ─────────────────────────────────────────────────────────────────────
#  자기검사 — "간이 재구현은 알려진 수치 재현부터"
# ─────────────────────────────────────────────────────────────────────
def _selftest():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    torch.manual_seed(0)
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'OK ' if cond else 'FAIL'}  {name}   {detail}")

    # ① STE — forward 는 진짜 하드, backward 는 부드러운 경사
    lg = torch.randn(5, 7, requires_grad=True)
    oh = hard_onehot_ste(lg, -1)
    chk("one-hot STE forward 가 정확히 one-hot",
        bool(((oh == 0) | (oh == 1)).all() and torch.allclose(oh.sum(-1),
                                                              torch.ones(5))))
    # ⚠️ `oh.sum()` 으로 미분하면 안 된다 — one-hot 의 합은 **항상 1** 이라
    #    상수를 미분하는 것이고 경사가 정당하게 0 이다. 내가 여기서 한 번 속았다.
    #    비상수 스칼라로 걸어야 STE 가 경사를 흘리는지 실제로 보인다.
    (oh * torch.randn(5, 7)).sum().backward()
    chk("one-hot STE backward 가 softmax 경사", lg.grad.abs().max() > 0,
        f"|grad| max {float(lg.grad.abs().max()):.2e} (0 이면 경사가 끊긴 것)")

    sx = torch.randn(200, requires_grad=True)
    lt0 = torch.zeros(200)
    gg = hard_step_ste(sx, lt0)
    chk("step STE forward == 1[s>0]", bool((gg == (sx > 0).float()).all()))
    gg.sum().backward()
    chk("step STE backward 가 sigmoid'", sx.grad.abs().max() > 0,
        f"|grad| max {float(sx.grad.abs().max()):.3f} (sigmoid' 최대 0.25)")

    # ② 잎 마스크가 **정확히 one-hot** — 하드 분할의 정의다
    d, B = 99, 2048
    u = torch.randn(B, d)
    for E_, D_ in ((32, 4), (64, 4)):
        torch.manual_seed(0)
        net = GrandeBody(d, E_, D_)
        with torch.no_grad():
            net._init_thresholds(u)
            net.ready.fill_(True)
            g = hard_step_ste(net._pre(u), net.lt)
            mask = torch.ones(B, E_, 1)
            off = 0
            for l in range(D_):
                n_l = 1 << l
                gl = g[:, :, off:off + n_l]
                mask = torch.stack([mask * (1 - gl), mask * gl], -1).reshape(
                    B, E_, n_l << 1)
                off += n_l
        chk(f"잎 마스크 one-hot E={E_}",
            bool(torch.allclose(mask.sum(-1), torch.ones(B, E_))
                 and float(mask.max()) == 1.0 and float(mask.min()) == 0.0),
            f"활성 잎 {int(mask.sum())} == B·E {B * E_}")

        # ③ 힙 인덱싱 — 잎 번호를 무식하게 다시 계산해서 대조한다.
        #    `cat` 으로 쌓으면 여기서 어긋난다.
        leaf_id = mask.argmax(-1)                              # (B, E)
        ref = torch.zeros(B, E_, dtype=torch.long)
        node = torch.zeros(B, E_, dtype=torch.long)            # 힙 인덱스
        for l in range(D_):
            bit = g.gather(2, node.unsqueeze(-1)).squeeze(-1).long()
            ref = (ref << 1) | bit
            node = ((node + 1) << 1) - 1 + bit                 # 힙 자식
        chk(f"잎 번호 == 힙 순회 재계산 E={E_}", bool((leaf_id == ref).all()),
            f"불일치 {int((leaf_id != ref).sum())} / {B * E_}")

    # ④ 스케일 — 하드 분할이라 추정기 하나의 출력은 잎값 하나다 (std = leaf_std).
    #    가중평균하면 sqrt(Σa²) 배 → 균등이면 1/sqrt(E).
    for E_, D_ in ((32, 4), (64, 4)):
        for wt in (True, False):
            torch.manual_seed(0)
            net = GrandeBody(d, E_, D_, weighted=wt)
            zz = net(u)
            pred = 1.0 / math.sqrt(E_)
            chk(f"std(z) E={E_} weighted={wt} == 1/sqrt(E)",
                abs(float(zz.std()) - pred) < 0.25 * pred,
                f"예측 {pred:.4f}  실측 {float(zz.std()):.4f}  "
                f"파라미터 {sum(p.numel() for p in net.parameters()):,}")

    # ⑤ ★ 절제의 공정성 — weighted True/False 가 **같은 초기값**에서 출발해야 한다.
    #    안 그러면 ③ 을 분리하려던 팔이 "초기값도 다른 두 모델" 비교가 된다.
    torch.manual_seed(0); a = GrandeBody(d, 32, 4, weighted=True)
    torch.manual_seed(0); b = GrandeBody(d, 32, 4, weighted=False)
    same = all(torch.equal(pa, pb) for (na, pa), (nb, pb)
               in zip(a.named_parameters(), b.named_parameters()) if na != "lw")
    chk("weighted 절제가 초기값을 안 건드린다", same,
        "lw 를 제외한 전 파라미터 비트 동일")
    # ⚠️ 임계값 초기화는 **첫 forward 에서 전역 RNG 를 소비한다.** 그냥 a(u), b(u)
    #    를 연달아 부르면 두 번째가 다른 분위수를 뽑아 max|Δ| 0.82 가 나온다
    #    (실제로 여기서 한 번 걸렸다). 배포 경로에서는 `train_mlp` 이 팔마다
    #    `manual_seed(seed)` 로 시작하고 `lw` 를 항상 만들어 두어 파라미터 생성
    #    순서가 같으므로 두 팔의 RNG 상태가 첫 forward 시점에 일치한다.
    #    검사도 그 조건을 재현해야 의미가 있다.
    torch.manual_seed(1); za = a(u)
    torch.manual_seed(1); zb = b(u)
    # lw = 0 이면 softmax 가 균등이라 두 팔이 **초기에는 같은 함수**여야 한다
    chk("lw=0 초기엔 weighted == uniform", torch.allclose(za, zb, atol=1e-5),
        f"max|Δ| {float((za - zb).abs().max()):.2e}")

    # ⑥ 청킹
    torch.manual_seed(0)
    net = GrandeBody(d, 64, 4)
    with torch.no_grad():
        big = torch.randn(70_000, d)
        net(big[:1000])                                        # 초기화 먼저
        chk("청킹 == 통짜", torch.allclose(net(big), net._reduce(big) + net.bias,
                                           atol=1e-6),
            f"cells {70_000 * 64 * 16:,} > 상한 {CHUNK_CELLS:,}")

    print("\n  " + ("전부 통과" if ok else "★ 실패 있음"))
    return ok


if __name__ == "__main__":
    _selftest()
