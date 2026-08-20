"""Factorization Machine — **제3의 함수족.**

지금 앙상블에는 두 종류의 귀납 편향만 있다.
    트리   축 정렬 분할 + 잎마다 상수      (계단함수)
    MLP    조밀 선형변환 + GELU            (매끄러운 일반 비선형)

FM은 셋째다. **모든 쌍 상호작용을 저랭크 내적으로 명시한다.**

    z = w0 + Σ w_i x_i + Σ_{i<j} <v_i, v_j> x_i x_j

핵심은 마지막 항이 **곱셈 형태로 강제**된다는 것이다. MLP도 상호작용을 배울 수 있지만
층을 쌓아 근사할 뿐이고, 트리는 분할의 중첩으로 흉내낼 뿐이다. FM은 그걸 구조로 박는다.
야구에서 (투수 성향 × 볼카운트), (좌우 매치업 × 구속) 같은 곱셈적 관계가
자연스럽다는 점에서 사전지식과도 맞는다.

계산은 O(n·d·k)로 싸다 — 쌍을 직접 돌지 않는다:

    Σ_{i<j} <v_i,v_j> x_i x_j = ½ Σ_f [ (Σ_i v_if x_i)² − Σ_i v_if² x_i² ]

인코딩은 MLP와 **완전히 같은 것**을 쓴다 (mlp.fit_encoder / encode_apply).
같은 정보 위에서 함수족만 다르게 하는 것이 이 실험의 요점이기 때문이다.
범주형은 임베딩을 그대로 v_i로 쓰고 x_i = 1로 둔다 (원-핫의 FM 표현과 동일).
"""
import numpy as np

SEED = 42
THREADS = 6

DEFAULT = dict(k=8, epochs=8, lr=3e-3, wd=1e-2, emb_dim=4, batch=8192)


def _torch():
    import torch
    torch.set_num_threads(THREADS)
    return torch


class FM:
    """torch 모듈. numpy 추론용 가중치는 export()로 뽑는다."""

    def __init__(self, n_num, sizes, ecols, k, emb_dim, seed):
        """sizes는 **ecols와 같은 순서의 리스트**다 (mlp.encode가 그렇게 준다).
        dict도 받아준다 — 합성 테스트에서 dict를 넘겼다가 실전에서 리스트라
        TypeError로 죽은 적이 있다."""
        torch = _torch()
        nn = torch.nn
        g = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)

        if isinstance(sizes, dict):
            card = [sizes[c] for c in ecols]
        else:
            if len(sizes) != len(ecols):
                raise ValueError(f"sizes({len(sizes)}) != ecols({len(ecols)}) — "
                                 "순서가 어긋나면 임베딩이 조용히 뒤섞인다")
            card = list(sizes)

        self.torch, self.nn = torch, nn
        self.ecols = ecols
        self.n_num = n_num
        self.k = k

        # 수치 피처: 1차 가중치 w_i 와 잠재벡터 v_i
        self.w = nn.Parameter(torch.zeros(n_num))
        self.V = nn.Parameter(torch.randn(n_num, k, generator=g) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))
        # 범주형: 카드마다 1차 항 + 잠재벡터 (x_i = 1 이므로 임베딩 조회가 곧 기여)
        self.cw = nn.ParameterList([nn.Parameter(torch.zeros(c)) for c in card])
        self.cV = nn.ParameterList([nn.Parameter(torch.randn(c, k, generator=g) * 0.01)
                                    for c in card])
        self.params = [self.w, self.V, self.b] + list(self.cw) + list(self.cV)

    def forward(self, xn, xc):
        """xn: (B, n_num) 표준화된 수치, xc: (B, n_cat) 범주 인덱스."""
        torch = self.torch
        lin = self.b + xn @ self.w
        # (B, k): 수치 기여 Σ v_i x_i  와  범주 기여 Σ v_c
        s = xn @ self.V
        sq = (xn * xn) @ (self.V * self.V)
        for j, (cw, cV) in enumerate(zip(self.cw, self.cV)):
            idx = xc[:, j]
            lin = lin + cw[idx]
            v = cV[idx]
            s = s + v
            sq = sq + v * v
        return lin + 0.5 * (s * s - sq).sum(dim=1)

    def export(self):
        d = {"w": self.w.detach().numpy(), "V": self.V.detach().numpy(),
             "b": self.b.detach().numpy(), "ecols": np.array(self.ecols)}
        for j, c in enumerate(self.ecols):
            d[f"cw{j}"] = self.cw[j].detach().numpy()
            d[f"cV{j}"] = self.cV[j].detach().numpy()
        return d


def fm_logit_numpy(W, A, C, eidx):
    """numpy 순전파 — torch와 일치하는지 검증하고, 제출 경로에서도 이걸 쓴다."""
    lin = W["b"][0] + A @ W["w"]
    s = A @ W["V"]
    sq = (A * A) @ (W["V"] * W["V"])
    for j in range(len(W["ecols"])):
        idx = C[:, eidx[j]]
        lin = lin + W[f"cw{j}"][idx]
        v = W[f"cV{j}"][idx]
        s = s + v
        sq = sq + v * v
    return lin + 0.5 * (s * s - sq).sum(axis=1)


def train_fm(A, C, y, w, B, Cv, sizes, ecols, seed=SEED, verbose=False, **cfg):
    """반환: (모델, [에폭별 valid 로짓]). 최적 에폭은 호출자가 고른다."""
    torch = _torch()
    c = dict(DEFAULT, **cfg)
    m = FM(A.shape[1], sizes, ecols, c["k"], c["emb_dim"], seed)

    Xn = torch.from_numpy(A)
    Xc = torch.from_numpy(C)
    Y = torch.from_numpy(y.astype(np.float32))
    Wt = torch.from_numpy((w / w.mean()).astype(np.float32))
    Bn, Bc = torch.from_numpy(B), torch.from_numpy(Cv)

    opt = torch.optim.AdamW(m.params, lr=c["lr"], weight_decay=c["wd"])
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")
    n, bs = len(A), c["batch"]
    g = torch.Generator().manual_seed(seed)
    curve = []
    for ep in range(c["epochs"]):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            z = m.forward(Xn[j], Xc[j])
            (lossf(z, Y[j]) * Wt[j]).mean().backward()
            opt.step()
        with torch.no_grad():
            zs = [m.forward(Bn[i:i + 65536], Bc[i:i + 65536]).numpy()
                  for i in range(0, len(B), 65536)]
        curve.append(np.concatenate(zs).astype(np.float64))
        if verbose:
            print(f"    ep{ep+1} done")
    return m, curve
