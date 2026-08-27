"""§7.188 — 행 공간 검색(kNN) 멤버 — 인스턴스 상관 채널의 **행 유사도** 형태 실측

사용자 재감사("5개 중 놓친 것이 있을 것")에서 나온 구멍 ①: §7.184/185 는 엔티티
그래프 형태만 죽였고, **피처 공간 행 유사도**(TabR/kNN 가족)는 TabReD 문헌으로만
닫혀 있었다 — 이 표에서 유일한 무실측 폐쇄. 여기서 실측한다.

왜 죽은 가족이 아닌가: 질의가 test 행의 **갱신되는 asof 피처**다 (얼린 엔티티
판정이 아님). 트리 리프도 "질의로 고른 train 행 집계"지만 축 정렬 기하 제약이
있고, kNN 은 구면 기하 = 다른 귀납 편향 = 탈상관 멤버 후보 (§7.16 MLP 채택 논리).

구성 (fold 학습 시즌만 — 시간 정직)
    피처: fold_data 의 78컬럼 − ID/팀 4개 → train 통계로 표준화 · NaN→0
          → train 20만 행 표본 SVD 16차원 투영 (전부 train 에서 동결)
    인덱스: train 행 전체. 질의: 검증 행별 k=128 이웃 (sklearn, 행 독립)
    멤버:  p̂ = (Σ w·y + k0·ȳ_w)/(Σ w + k0),  w = recency(hl 0.5), k0 = 25
    위약:  같은 k·같은 평활에서 **무작위 이웃** (기하 파괴 — 검색이 일하는지 판정)

평가 (사전 등록 · 배포 블렌드 z_b = .25·z_lgb + .75·z_mlp 캐시 위)
    단독 BSS · corr(z_knn, z_b) · 블렌드 α ∈ {.02, .05, .10, .15} 곡선(로짓 혼합)
    판정: ⓐ 양 fold 공통 양수 α 구간(폭 ≥ 2칸) ⓑ 위약은 그 구간 없음
          ⓒ α=.05 쌍체(투수 클러스터 B=2000) P ≥ 0.9 양 fold
    전부 통과 → TabReD 반증·멤버 후보 (추론 예산 검증 후 LB 논의).
    미달 → 인스턴스 상관 채널의 두 형태 모두 실측 폐쇄 (문헌과 일치).

실행:  .venv/Scripts/python.exe -u src/knnmember.py 2>&1 | tee experiments/knnmember.log
"""
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

import foldcache as fc
from config import MLP_ALPHA, RECENCY_HALF_LIFE, TARGET_COL, TRAIN_CSV
from data import load_csv, recency_weight
from validation import bss, shift_to_prior

K_NBR = 128
K0 = 25.0
N_SVD = 200_000
DIMS = 16
ALPHAS = (0.02, 0.05, 0.10, 0.15)
ALPHA_GATE = 0.05
B_BOOT = 2000
DROP_ID = ("pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id")
EPS = 1e-6


def project(Xtr, Xva, rng):
    """train 통계 표준화 → NaN 0 → train 표본 SVD 16차원. 전 상수 train 동결."""
    cols = [c for c in Xtr.columns if c not in DROP_ID]
    A = Xtr[cols].to_numpy(dtype=np.float32)
    B = Xva[cols].to_numpy(dtype=np.float32)
    mu = np.nanmean(A, axis=0)
    sd = np.nanstd(A, axis=0)
    sd[sd < 1e-9] = 1.0

    def std(M):
        M = (M - mu) / sd
        return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    A, B = std(A), std(B)
    sub = A[rng.choice(len(A), size=min(N_SVD, len(A)), replace=False)]
    _, _, Vt = np.linalg.svd(sub, full_matrices=False)
    P = Vt[:DIMS].T.astype(np.float32)
    return A @ P, B @ P


def knn_logit(idx, ytr, wtr, ybar_w):
    sw = wtr[idx].sum(axis=1)
    sy = (wtr[idx] * ytr[idx]).sum(axis=1)
    p = (sy + K0 * ybar_w) / (sw + K0)
    return np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


def main():
    t0 = time.time()
    from sklearn.neighbors import NearestNeighbors
    df = load_csv(TRAIN_CSV)
    base = fc.load([2023, 2024], verbose=True)

    res = {}
    for fold in (2023, 2024):
        print(f"\n=== fold {fold} ===")
        Xtr, ytr, wtr, Xva, yva, art, tr = fc.fold_data(df, fold)
        rng = np.random.default_rng(fold)
        Atr, Ava = project(Xtr, Xva, rng)
        del Xtr, Xva
        gc.collect()
        wtr = np.asarray(wtr, dtype=np.float64)
        ybar_w = float((wtr * ytr).sum() / wtr.sum())
        print(f"  투영 {Atr.shape} → 질의 {Ava.shape}  ȳ_w={ybar_w:.4f}"
              f"  ({time.time()-t0:.0f}s)")

        nn = NearestNeighbors(n_neighbors=K_NBR, n_jobs=6).fit(Atr)
        idx = nn.kneighbors(Ava, return_distance=False)
        print(f"  kNN 질의 완료  ({time.time()-t0:.0f}s)")
        z_knn = knn_logit(idx, ytr, wtr, ybar_w)
        idx_r = rng.integers(0, len(Atr), size=idx.shape)
        z_rnd = knn_logit(idx_r, ytr, wtr, ybar_w)
        del Atr, Ava, idx, idx_r
        gc.collect()

        zl, zm, y = base[fold][0], base[fold][1], base[fold][2]
        assert np.allclose(y, yva), "캐시 y 불일치"
        zb = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm

        def sc(z):
            return bss(y, shift_to_prior(1 / (1 + np.exp(-z)), y.mean()))

        b0 = sc(zb)
        print(f"  블렌드 기준 {b0:.1f}   kNN 단독 {sc(z_knn):.1f}"
              f" (위약 단독 {sc(z_rnd):.1f})"
              f"   corr(z_knn, z_b) {np.corrcoef(z_knn, zb)[0, 1]:.3f}")
        row = {"base": b0}
        print("  %-6s" % "α" + "".join(f"{a:>10.2f}" for a in ALPHAS))
        for nm, zc in (("kNN", z_knn), ("위약", z_rnd)):
            ds = [sc((1 - a) * zb + a * zc) - b0 for a in ALPHAS]
            row[nm] = ds
            print(f"  {nm:<6s}" + "".join(f"{d:>+10.2f}" for d in ds))

        # 쌍체 부트스트랩 (α_GATE, 투수 클러스터)
        cl = pd.factorize(tr := df[fc.SPLITS[fold](df.season.to_numpy(),
                          df.game_month.to_numpy())[1]]["pitcher_id"].to_numpy())[0]
        n_id = cl.max() + 1
        p0 = shift_to_prior(1 / (1 + np.exp(-zb)), y.mean())
        p1 = shift_to_prior(1 / (1 + np.exp(-((1 - ALPHA_GATE) * zb
                                              + ALPHA_GATE * z_knn))), y.mean())
        ref = y.mean() * (1 - y.mean())
        n = np.bincount(cl, minlength=n_id)
        s0 = np.bincount(cl, weights=(p0 - y) ** 2, minlength=n_id)
        s1 = np.bincount(cl, weights=(p1 - y) ** 2, minlength=n_id)
        seen = np.flatnonzero(n > 0)
        rngb = np.random.default_rng(42)
        D = np.empty(B_BOOT)
        for i in range(B_BOOT):
            pick = rngb.choice(seen, size=len(seen), replace=True)
            D[i] = 1e5 * (s0[pick].sum() - s1[pick].sum()) / (n[pick].sum() * ref)
        lo, hi = np.percentile(D, [2.5, 97.5])
        P = float(np.mean(D > 0))
        print(f"  쌍체 Δ(α={ALPHA_GATE}) {bss(y, p1)-b0:+.2f}"
              f"  [{lo:+.2f}, {hi:+.2f}]  P {P:.3f}")
        res[fold] = dict(knn=row["kNN"], plb=row["위약"], P=P)

    print("\n" + "=" * 90)
    both = [i for i in range(len(ALPHAS))
            if res[2023]["knn"][i] > 0 and res[2024]["knn"][i] > 0]
    plb_both = [i for i in range(len(ALPHAS))
                if res[2023]["plb"][i] > 0 and res[2024]["plb"][i] > 0]
    ok_a = len(both) >= 2
    ok_b = len(plb_both) == 0
    ok_c = all(res[f]["P"] >= 0.9 for f in (2023, 2024))
    print(f"판정  ⓐ 공통 양수 α 폭≥2: {'통과' if ok_a else '미달'} ({len(both)}칸)"
          f"   ⓑ 위약 구간 없음: {'통과' if ok_b else '미달'} ({len(plb_both)}칸)"
          f"   ⓒ 쌍체 P≥.9 양 fold: {'통과' if ok_c else '미달'}"
          f" ({res[2023]['P']:.3f}/{res[2024]['P']:.3f})")
    print("→ " + ("**채널 생존 — TabReD 반증. 추론 예산 검증 후 멤버 후보**"
                  if ok_a and ok_b and ok_c else
                  "기각 — 인스턴스 상관의 행 유사도 형태도 실측 폐쇄 (TabReD 와 일치)"))
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
