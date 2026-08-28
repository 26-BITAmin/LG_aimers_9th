"""미학습 투수(임베딩 index 0) 자리를 asof 프로파일 유사도로 채워 본다.

동기
    배포 임베딩 테이블은 투수 792명 + **미지값 1칸**이다. 그런데 매 시즌 행의
    14~22% 가 그 전에 없던 투수다 (2024 = 19.86%, 5년 평균 18.5%).
    → 2025 test 행의 약 20% 가 index 0 벡터 **하나**로 들어간다.
      알려진 투수 1명당 행 0.1% vs 미지값 1칸당 행 20% — **200배 과부하**다.

    그리고 index 0 은 학습 중 어떤 행도 매핑되지 않아 (LUT 를 학습 데이터로 적합)
    사실상 초기값에 머문다. 실측: ||emb0[0]|| 이 시드마다 1.29~3.90 (3배 산포),
    같은 시드의 나머지 792행 중앙값은 2.658~2.721 (거의 고정).

제안
    index 0 행의 투수 임베딩을 고정 벡터 대신, **그 행 자신의 asof_pitcher_* 와
    가까운 알려진 투수들의 임베딩 가중평균**으로 대체한다.

    쓰는 것은 (a) 학습 때 동결된 임베딩·프로파일 표, (b) 그 행 자신의 피처뿐이다.
    test 전체 통계도 행 간 정보도 없다 — `predict.py` 의 라우팅과 같은 성격이다.

평가 설계 — 마스킹 실험
    2024 행은 배포 모델의 in-sample 이라 절대값을 믿으면 안 된다. 그래서 투수의
    일부를 **인위적으로 미지값 처리**하고 세 팔을 같은 행에서 비교한다.

      UNK   마스킹된 행 -> index 0            (현재 배포가 미학습 투수에 하는 것)
      COLD  마스킹된 행 -> 유사도 가중평균     (제안)
      TRUE  마스킹된 행 -> 원래 임베딩         (상한 참고. in-sample 이라 부풀려져 있다)

    **판정은 UNK vs COLD 다.** 둘 다 그 투수 자신의 벡터를 안 쓰므로 공평하다.
    TRUE 는 천장 감각용이고 채택 근거로 쓰지 않는다.
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = "/Users/hyejin/Aimers_9th/data/train.csv"


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-dir", required=True, help="풀어 둔 제출 zip 경로")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--mask-frac", type=float, default=0.20, help="미지값 처리할 투수 비율")
    ap.add_argument("--tau", type=float, default=1.0, help="유사도 softmax 온도")
    ap.add_argument("--topk", type=int, default=25, help="가중평균에 쓸 이웃 수")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, a.zip_dir)
    from features import build_features
    from mlp_infer import EMB_COLS, encode_apply

    bundle = joblib.load(os.path.join(a.zip_dir, "model", "artifacts.joblib"))
    enc = bundle["mlp_enc"]
    keys, lut_idx = enc["cat_luts"]["pitcher_id"]

    df = pd.read_csv(TRAIN_CSV)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    df = df[df["season"] == a.season].reset_index(drop=True)
    y = df["control_success"].to_numpy(dtype=np.float64)
    # 배포와 같은 경로로 피처를 만든다 (predict.py 와 동일: build_features -> feature_cols)
    src = df.drop(columns=["control_success"])
    X = build_features(src, bundle["artifacts"])
    if bundle["artifacts"].get("use_player_ids"):        # predict.py:115 와 같은 처리
        for c in ("pitcher_id", "batter_id"):
            X[c] = src[c].to_numpy(dtype=np.float32)
    X = X[bundle["feature_cols"]]
    print(f"{a.season} 행 {len(X):,} · 피처 {X.shape[1]}")

    A, C = encode_apply(X, enc)
    p_idx = C[:, EMB_COLS.index("pitcher_id")]          # 0 = 미지값
    print(f"  실제 미지값 행 {(p_idx == 0).sum():,} ({(p_idx == 0).mean():.2%})")

    # ── 투수 프로파일 표 (학습 시점 동결 상수로 취급) ──────────────
    num_cols = list(enc["num_cols"])
    prof_cols = [i for i, c in enumerate(num_cols) if c.startswith("asof_pitcher")]
    print(f"  프로파일 피처 {len(prof_cols)}개 (asof_pitcher_*)")
    n_emb = int(enc["sizes"][0])
    prof = np.zeros((n_emb, len(prof_cols)), dtype=np.float64)
    cnt = np.zeros(n_emb)
    np.add.at(prof, p_idx, A[:, prof_cols].astype(np.float64))
    np.add.at(cnt, p_idx, 1.0)
    seen = cnt > 0
    prof[seen] /= cnt[seen, None]

    # ── 마스킹: 투수의 일부를 미지값 처리 ────────────────────────
    rng = np.random.RandomState(a.seed)
    known = np.flatnonzero(seen)
    known = known[known != 0]
    masked = set(rng.choice(known, int(len(known) * a.mask_frac), replace=False).tolist())
    is_masked = np.array([i in masked for i in p_idx])
    print(f"  마스킹 투수 {len(masked)}/{len(known)}명 -> 행 {is_masked.sum():,} ({is_masked.mean():.2%})")

    # 마스킹된 투수는 이웃 후보에서도 제외한다 (누출 방지)
    cand = np.array([i for i in known if i not in masked])
    cprof = prof[cand]

    # ── 유사도 가중평균 (마스킹 행 각각에 대해) ──────────────────
    rows = np.flatnonzero(is_masked)
    q = A[rows][:, prof_cols].astype(np.float64)
    W = np.zeros((len(rows), len(cand)))
    for i in range(0, len(rows), 4096):
        j = min(i + 4096, len(rows))
        d2 = ((q[i:j, None, :] - cprof[None, :, :]) ** 2).sum(-1)
        kth = np.partition(d2, a.topk, axis=1)[:, a.topk][:, None]
        d2 = np.where(d2 <= kth, d2, np.inf)
        w = np.exp(-(d2 - d2.min(1, keepdims=True)) / a.tau)
        W[i:j] = w / w.sum(1, keepdims=True)

    # ── 세 팔의 emb4 앙상블 로짓 ─────────────────────────────────
    import glob
    files = sorted(glob.glob(os.path.join(a.zip_dir, "model", "mlp_emb4_*.npz")))
    res = {}
    for arm in ("UNK", "MEAN", "COLD", "TRUE"):
        z = np.zeros(len(X))
        for f in files:
            npz = np.load(f)
            Wm = [npz[f"W{i}"] for i in range(3)]
            bm = [npz[f"b{i}"] for i in range(3)]
            ecols = [str(c) for c in npz["ecols"]]
            embs = [npz[f"emb{i}"] for i in range(len(ecols))]
            pe = EMB_COLS.index("pitcher_id")
            Cm = C.copy()
            if arm == "UNK":
                Cm[rows, pe] = 0
            E = [e[Cm[:, EMB_COLS.index(c)]] for e, c in zip(embs, ecols)]
            pi = ecols.index("pitcher_id")
            if arm == "COLD":
                E[pi][rows] = W @ embs[pi][cand]
            elif arm == "MEAN":
                E[pi][rows] = embs[pi][cand].mean(0)
            h = np.concatenate([A] + E, axis=1)
            h = gelu(h @ Wm[0] + bm[0])
            h = gelu(h @ Wm[1] + bm[1])
            z += (h @ Wm[2] + bm[2]).ravel()
        res[arm] = z / len(files)

    print(f"\n마스킹 행 {is_masked.sum():,} 개에서의 Brier (낮을수록 좋다)")
    print(f"{'팔':<6}{'Brier':>12}{'vs UNK':>12}")
    base = None
    for arm in ("UNK", "MEAN", "COLD", "TRUE"):
        p = 1.0 / (1.0 + np.exp(-res[arm][rows]))
        br = float(((p - y[rows]) ** 2).mean())
        if base is None:
            base = br
        print(f"{arm:<6}{br:>12.6f}{(base - br) * 1e5:>+12.2f}")
    print("\n(vs UNK 는 Brier 차이 x 1e5 — 대회 점수와 같은 축척)")
    print("판정은 UNK vs COLD. TRUE 는 in-sample 이라 상한 참고용이다.")


if __name__ == "__main__":
    main()
