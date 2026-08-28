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

TRAIN_CSV = "/Users/hyejin/Aimers_9th/data/train.csv"

def load_from_zip(zip_dir, names):
    """제출 zip 을 푼 디렉터리에서 모듈을 가져온다.

    ⚠️ `features` · `mlp_infer` 는 **이 저장소에 없다.** 배포 시점의 스냅숏이라
       제출 zip 안에만 존재하고, 그래서 정적 분석기가 못 찾는다 (실행은 된다).
       배포와 **같은 코드로** 피처·인코딩을 해야 비교가 성립하므로 일부러 이렇게 둔다
       (원우 §7.175 `t1core.py` 의 방침 — "T1 자신의 features.py + artifacts.joblib").
    """
    need = ["features.py", "mlp_infer.py", os.path.join("model", "artifacts.joblib")]
    missing = [n for n in need if not os.path.exists(os.path.join(zip_dir, n))]
    if missing:
        raise SystemExit(
            f"[중단] --zip-dir 가 풀어 둔 제출 zip 이 아니다: {zip_dir}\n"
            f"  없는 파일: {missing}\n"
            f"  예) mkdir -p /tmp/z && cd /tmp/z && unzip -oq outputs/hj02_base.zip")
    sys.path.insert(0, zip_dir)
    import importlib
    return [importlib.import_module(n) for n in names]


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-dir", required=True, help="풀어 둔 제출 zip 경로")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--mask-frac", type=float, default=0.20, help="미지값 처리할 투수 비율")
    ap.add_argument("--tau", type=float, default=1.0, help="유사도 softmax 온도")
    ap.add_argument("--topk", type=int, default=25, help="가중평균에 쓸 이웃 수")
    ap.add_argument("--seeds", default="0", help="쉼표 구분. 마스킹 시드 목록")
    a = ap.parse_args()

    feat, mi = load_from_zip(a.zip_dir, ["features", "mlp_infer"])
    build_features, EMB_COLS, encode_apply = feat.build_features, mi.EMB_COLS, mi.encode_apply

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

    # ── 시드별 마스킹 → 채점 (마스킹 행만 순전파한다) ──────────
    import glob
    files = sorted(glob.glob(os.path.join(a.zip_dir, "model", "mlp_emb4_*.npz")))
    nets = []
    for f in files:
        z = np.load(f)
        nets.append(dict(W=[z[f"W{i}"] for i in range(3)], b=[z[f"b{i}"] for i in range(3)],
                         ecols=[str(c) for c in z["ecols"]],
                         emb=[z[f"emb{i}"] for i in range(len(z["ecols"]))]))
    known_all = np.flatnonzero(seen); known_all = known_all[known_all != 0]

    print(f"\n{'시드':>4} {'마스킹행':>9} {'UNK':>10} {'MEAN':>9} {'COLD':>9} {'TRUE':>9}")
    res = {k: [] for k in ("MEAN", "COLD", "TRUE")}
    for sd in [int(x) for x in a.seeds.split(",")]:
        rng = np.random.RandomState(sd)
        masked = set(rng.choice(known_all, int(len(known_all) * a.mask_frac), replace=False).tolist())
        rows = np.flatnonzero(np.array([i in masked for i in p_idx]))
        cand = np.array([i for i in known_all if i not in masked])
        cprof = prof[cand]

        q = A[rows][:, prof_cols].astype(np.float64)
        Wt = np.zeros((len(rows), len(cand)))
        for i in range(0, len(rows), 4096):
            j = min(i + 4096, len(rows))
            d2 = ((q[i:j, None, :] - cprof[None, :, :]) ** 2).sum(-1)
            kth = np.partition(d2, a.topk, axis=1)[:, a.topk][:, None]
            d2 = np.where(d2 <= kth, d2, np.inf)
            wgt = np.exp(-(d2 - d2.min(1, keepdims=True)) / a.tau)
            Wt[i:j] = wgt / wgt.sum(1, keepdims=True)

        Asub, Csub, ysub = A[rows], C[rows], y[rows]
        br = {}
        for arm in ("UNK", "MEAN", "COLD", "TRUE"):
            z = np.zeros(len(rows))
            for nt in nets:
                pe = nt["ecols"].index("pitcher_id")
                E = [nt["emb"][m][Csub[:, EMB_COLS.index(c)]] for m, c in enumerate(nt["ecols"])]
                if arm == "UNK":
                    E[pe] = np.repeat(nt["emb"][pe][0][None, :], len(rows), 0)
                elif arm == "MEAN":
                    E[pe] = np.repeat(nt["emb"][pe][cand].mean(0)[None, :], len(rows), 0)
                elif arm == "COLD":
                    E[pe] = Wt @ nt["emb"][pe][cand]
                h = np.concatenate([Asub] + E, axis=1)
                h = gelu(h @ nt["W"][0] + nt["b"][0])
                h = gelu(h @ nt["W"][1] + nt["b"][1])
                z += (h @ nt["W"][2] + nt["b"][2]).ravel()
            pr = 1.0 / (1.0 + np.exp(-z / len(nets)))
            br[arm] = float(((pr - ysub) ** 2).mean())
        for k in res:
            res[k].append((br["UNK"] - br[k]) * 1e5)
        print("%4d %9d %10.6f %+9.2f %+9.2f %+9.2f"
              % (sd, len(rows), br["UNK"], res["MEAN"][-1], res["COLD"][-1], res["TRUE"][-1]))

    print(f"\n{len(res['MEAN'])}시드 요약 (Brier차 x 1e5, UNK 대비)")
    print("%-6s %8s %8s %8s %8s %6s" % ("팔", "평균", "std", "SE", "t", "양수"))
    for k in ("MEAN", "COLD", "TRUE"):
        v = np.array(res[k]); se = v.std(ddof=1) / np.sqrt(len(v))
        print("%-6s %+8.2f %8.2f %8.2f %8.2f %4d/%d"
              % (k, v.mean(), v.std(ddof=1), se, v.mean() / se, (v > 0).sum(), len(v)))
    return


if __name__ == "__main__":
    main()
