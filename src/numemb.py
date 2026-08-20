"""수치 피처 임베딩 — Gorishniy et al., "On Embeddings for Numerical Features in
Tabular Deep Learning" (NeurIPS 2022, arXiv:2203.05556)

왜 이걸 보는가
    우리 MLP는 표준화된 스칼라를 그대로 먹는다. 트리는 같은 값을 **구간으로 쪼갠다.**
    fold2024에서 MLP 단독 700.6 vs LGBM 752.5 — **100점 격차**가 있고, 이 문제는
    극단적으로 GBDT 친화적이다(축 정렬 임계값이 잘 먹는다). 논문의 주장이 정확히
    이 지점이다: 수치 임베딩을 붙이면 MLP가 GBDT 친화 벤치마크에서 격차를 메운다.

    §7.24가 정한 후보 조건("단독 점수가 오르는가")을 정면으로 겨냥하는 첫 후보다.

두 가지 인코딩 (공식 구현 `rtdl_num_embeddings` 기준)

    PLE  구간 경계는 분위수로 잡고, 성분 t는 "구간 t가 얼마나 찼는가"다.
             x_t = (x - b_{t-1}) / (b_t - b_{t-1})
         clamp가 **비대칭**이다 — 첫 성분은 위로만, 마지막은 아래로만 자른다.
         양끝을 열어둬야 학습 범위 밖 값의 정보가 남는다 (2025 신인·미지 구간).
             첫  clamp_max(1)   중간  clamp(0,1)   마지막  clamp_min(0)

    RFF  z = 2π·w·x  →  concat[cos z, sin z],  w ~ truncnorm(0, σ), ±3σ 절단.
         논문의 Periodic은 w를 **학습**하지만 여기서는 **고정**한다 (Rahimi & Recht의
         random Fourier features). 이유: 학습 파라미터를 모델 안에 넣으면
         `export_weights`/`mlp_infer`를 같이 일반화해야 하고, 그건 채택된 뒤에 할 일이다.
         σ는 표준화 입력 기준으로 잡는다 — rtdl 기본 0.01은 분위수 변환 입력용이라
         우리 스케일에선 2π·0.01·x ≈ 0.06x로 사실상 선형이라 아무것도 안 한다.

메모리 (이 기계는 RAM 15.3GB, 여유 2.4GB다)
    PLE를 미리 펼치면 fold2024 학습셋이 1,221,585 × ~480 × 4B = **2.3GB**다.
    그래서 **펼치지 않는다.** 경계만 들고 있다가 배치 안에서 만든다 (8192행 = 15MB).
    공식 구현도 같은 형태다 (affine + clamp, 룩업 없음).

배포
    PLE는 **고정 변환**이다 (경계는 학습셋 분위수). numpy로 몇 줄이면 되고
    `mlp_infer`의 순전파에 그대로 얹힌다. 채택되면 그때 옮긴다.
"""
import numpy as np

SUB = 200_000          # 분위수·카디널리티 판정용 표본. 경계 추정엔 차고 넘친다


def _continuous(A, n_bins, rng_seed=0):
    """구간을 나눌 만한 컬럼만 고른다.

    이진·저카디널리티(is_two_strikes, runner_on_1b, miss_* …)는 그대로 통과시킨다.
    구간을 쪼갤 값이 없어서 분위수가 퇴화하고, 쪼개봐야 원핫의 중복이다.
    """
    n = min(SUB, len(A))
    step = max(1, len(A) // n)
    S = A[::step][:n]
    return S, [j for j in range(A.shape[1])
               if len(np.unique(S[:, j])) > n_bins]


def fit_ple(A, n_bins=8):
    """학습셋 분위수로 구간 경계를 잡는다 (학습에서 fit, 추론에서 transform)."""
    S, cand = _continuous(A, n_bins)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    idx, W, B = [], [], []
    for j in cand:
        e = np.quantile(S[:, j].astype(np.float64), qs)
        # 퇴화 구간(폭 0)이 하나라도 있으면 나눗셈이 터진다. 그 컬럼은 raw로 보낸다.
        if np.any(np.diff(e) < 1e-9):
            continue
        w = 1.0 / np.diff(e)
        idx.append(j)
        W.append(w)
        B.append(-e[:-1] * w)
    keep = set(idx)
    return dict(kind="ple", n_bins=n_bins,
                idx=np.array(sorted(idx), dtype=np.int64),
                raw=np.array([j for j in range(A.shape[1]) if j not in keep],
                             dtype=np.int64),
                w=np.array(W, dtype=np.float32),
                b=np.array(B, dtype=np.float32))


def fit_rff(A, d=8, sigma=0.2, seed=42):
    S, cand = _continuous(A, 8)
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, sigma, size=(len(cand), d))
    np.clip(w, -3 * sigma, 3 * sigma, out=w)          # truncated normal
    keep = set(cand)
    return dict(kind="rff", d=d, sigma=sigma,
                idx=np.array(sorted(cand), dtype=np.int64),
                raw=np.array([j for j in range(A.shape[1]) if j not in keep],
                             dtype=np.int64),
                w=w.astype(np.float32), b=None)


def out_dim(spec, n_num):
    if spec is None:
        return n_num
    k = len(spec["idx"])
    per = spec["n_bins"] if spec["kind"] == "ple" else 2 * spec["d"]
    return len(spec["raw"]) + k * per


def apply_np(spec, X):
    """numpy 참조 구현. 배포(mlp_infer)와 검증에 쓴다.

    학습 쪽 torch 구현(`mlp.train_mlp`)과 **같은 식**이어야 한다.
    features.py의 `_col64` 사고와 같은 종류의 함정이라 검증으로 대조한다.
    """
    if spec is None:
        return X
    v = X[:, spec["idx"]][:, :, None]
    if spec["kind"] == "ple":
        e = v * spec["w"] + spec["b"]
        e = np.concatenate([np.minimum(e[:, :, :1], 1.0),
                            np.clip(e[:, :, 1:-1], 0.0, 1.0),
                            np.maximum(e[:, :, -1:], 0.0)], axis=2)
    else:
        z = 2.0 * np.pi * v * spec["w"]
        e = np.concatenate([np.cos(z), np.sin(z)], axis=2)
    return np.concatenate([X[:, spec["raw"]], e.reshape(len(X), -1)],
                          axis=1).astype(np.float32)
