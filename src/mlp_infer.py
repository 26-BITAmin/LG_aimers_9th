"""MLP 순전파 — **numpy만 쓴다.** 제출 zip 루트에 복사된다.

왜 torch를 안 쓰는가
    torch를 `requirements.txt`에 넣으면 오프라인 설치 10분 제약, pickle 버전 호환,
    CUDA 유무가 전부 위험이 된다. 그런데 이 모델의 추론은 행렬곱 세 번이다.
    학습만 torch로 하고 가중치를 npz로 내보내면 그 위험이 통째로 사라진다.

    245,789 x 80 -> 128 -> 128 -> 1 은 numpy로 1초 미만이다.

⚠️ 행 독립성
    `encode`가 쓰는 mu / sd / 범주 LUT / 결측 대체값은 전부 **학습 때 동결된 상수**다.
    이 모듈에는 test 전체에 대한 통계 계산이 없다. `verify_pipeline.py`의
    조각 분할 검사가 이를 매 빌드마다 확인한다.

⚠️ BatchNorm
    학습 때의 배치 통계가 아니라 **running_mean / running_var**를 쓴다
    (torch의 eval 모드와 같다). 내보낼 때 그 둘을 가중치에 접어 넣으므로
    여기서는 별도 처리가 없다 — Linear의 W, b에 이미 반영돼 있다.
"""
import numpy as np

# 학습 코드(mlp.py)와 **같은 목록이어야 한다**. 순서까지 같아야 한다.
EMB_COLS = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
            "base_state", "hand_matchup", "count_state"]


# 순전파 조각 크기. 이 값이 바뀌면 예측이 마지막 자리에서 달라지므로
# 학습·검증·추론이 모두 같은 값을 써야 한다 (그래서 상수로 고정한다).
BATCH = 8192

_C = np.float32(0.7978845608028654)      # sqrt(2/pi)


def _gelu(x):
    """torch `nn.GELU(approximate="tanh")`와 **같은 식**이다.

    ⚠️ torch의 기본값은 정확한 erf 버전인데 numpy에는 erf가 없다.
       math.erf를 frompyfunc로 감싸면 3천만 원소를 파이썬 루프로 돌게 되고,
       여기서만 tanh 근사를 쓰면 train/serve skew가 난다.
       그래서 **학습 쪽을 approximate="tanh"로 맞췄다** (mlp.py).
       두 파일 중 하나만 바꾸면 조용히 어긋난다.
    """
    return 0.5 * x * (1.0 + np.tanh(_C * (x + np.float32(0.044715) * x * x * x)))


def encode_apply(X, enc):
    """학습 때 동결된 인코더 상수로 (수치행렬, 범주인덱스행렬)을 만든다.

    enc = {num_cols, mu, sd, nan_fill, nan_src, cat_luts}
    """
    num_cols = enc["num_cols"]
    A = X[num_cols].to_numpy(dtype=np.float32).astype(np.float64)

    # 원본 rate의 결측 -> 같은 통계의 `_sm` 컬럼값 -> 그래도 없으면 학습 prior
    for c, src, fill in enc["nan_plan"]:
        j = num_cols.index(c)
        m = np.isnan(A[:, j])
        if m.any():
            if src is not None:
                A[m, j] = A[m, num_cols.index(src)]
            A[:, j] = np.nan_to_num(A[:, j], nan=fill)

    A = np.nan_to_num(A, nan=0.0)
    A = (A - enc["mu"]) / enc["sd"]

    C = np.empty((len(X), len(EMB_COLS)), dtype=np.int64)
    for i, col in enumerate(EMB_COLS):
        keys, idx = enc["cat_luts"][col]          # 정렬된 키 배열, 대응 인덱스
        v = X[col].to_numpy(dtype=np.float64)
        pos = np.clip(np.searchsorted(keys, v), 0, len(keys) - 1)
        C[:, i] = np.where(keys[pos] == v, idx[pos], 0)   # 미지값 -> 0
    return A.astype(np.float32), C


class MLPPredictor:
    """npz에 담긴 가중치로 순전파. 반환값은 **로짓**이다."""

    def __init__(self, npz, enc):
        # 층 수는 **npz 가 정한다** (§7.217). 예전에는 3으로 하드코딩돼 있어
        # depth 2 전용이었다. depth d plain 이면 은닉 W0..W{d-1} + 헤드 W{d} 다.
        # depth 2 npz 는 키가 W0/W1/W2 라 이 루프가 **예전과 같은 값**을 만든다.
        n = 0
        while f"W{n}" in npz:
            n += 1
        assert n >= 2, f"가중치 층이 {n}개다 (은닉 1개 + 헤드 1개가 최소)"
        self.W = [npz[f"W{i}"] for i in range(n)]
        self.b = [npz[f"b{i}"] for i in range(n)]
        # ⚠️ 멤버마다 임베딩 컬럼이 다르다 (`use_id_emb=False`면 투수/타자 임베딩이 없다).
        #    EMB_COLS 전체를 가정하면 KeyError가 나거나 — 더 나쁘게는 —
        #    다른 컬럼의 인덱스를 엉뚱한 임베딩 테이블에 넣는다.
        #    그래서 **어떤 컬럼을 쓰는지 가중치와 함께 저장**한다.
        self.ecols = [str(c) for c in npz["ecols"]]
        self.eidx = [EMB_COLS.index(c) for c in self.ecols]
        self.emb = [npz[f"emb{i}"] for i in range(len(self.ecols))]
        self.enc = enc

    def logit(self, X, batch=BATCH):
        """⚠️ 마지막 조각을 **패딩해서 모든 행렬 모양을 (batch, d)로 고정**한다.

        패딩이 없으면 행 순서 독립성 검사가 1e-8 수준에서 실패한다. 정보 누출이
        아니라 BLAS 때문이다 — 행렬 모양이 바뀌면 블로킹·벡터화 경로가 달라지고
        float32 덧셈은 결합법칙이 성립하지 않는다. 실측:

            253,507행:  배치 200k vs 통짜 = 7.45e-09   (순서를 전혀 안 바꿔도)
                        셔플 후 되돌림     = 7.45e-09   (정확히 같은 크기)

        검사 임계값을 푸는 대신 원인을 없앤다. 모양이 항상 같으면 어떤 행이 어느
        위치에 오든 비트 단위로 같은 값이 나오고, `== 0` 검사를 그대로 쓸 수 있다.
        비용은 마지막 조각의 패딩분(<3%)뿐이다.
        """
        n = len(X)
        out = np.empty(n, dtype=np.float64)
        for i in range(0, n, batch):
            j = min(i + batch, n)
            A, C = encode_apply(X.iloc[i:j], self.enc)
            k = j - i
            if k < batch:
                A = np.pad(A, ((0, batch - k), (0, 0)))
                C = np.pad(C, ((0, batch - k), (0, 0)))
            h = np.concatenate(
                [A] + [e[C[:, self.eidx[m]]] for m, e in enumerate(self.emb)], axis=1)
            for Wi, bi in zip(self.W[:-1], self.b[:-1]):
                h = _gelu(h @ Wi + bi)
            out[i:j] = (h @ self.W[-1] + self.b[-1]).ravel()[:k]
        return out
