"""MLP 잔차의 계단 구조 프로브 — PLE 부분집합 후보 컬럼 선별 (§7.26 재개봉 사전 조사)

질문: 어떤 연속 컬럼 축을 따라, MLP 잔차에 "매끄러운 곡선으로 설명 안 되는"
구조가 남아 있고, 같은 자리의 LGBM 잔차에는 없는가?
그런 컬럼만 PLE(구간선형 인코딩)를 부분 적용할 근거가 있다. 전 컬럼 인코딩은
§7.26에서 차원 단조로 기각됐지만, 그 실험은 부분집합 축을 덮지 않았다.

방법 (fold별, foldcache의 z 재사용 — 학습 없음):
    r = y − σ(z)                          (확률 공간 잔차)
    컬럼마다 검증행을 분위수 24구간으로 나누고, 구간별 평균 잔차 r̄_b 를
    구간 누적질량 좌표에 대한 3차 다항(= MLP가 표현할 수 있는 매끄러운
    주효과 대조군)으로 적합한 뒤 초과 χ²를 z점수로 요약한다:
        χ²  = Σ_b n_b (r̄_b − fit_b)² / σ̄²_b     σ̄²_b = 구간 내 mean(p(1−p))
        z   = (χ² − dof) / √(2·dof)              dof = B − 4
    모델이 옳다면 Var(y−p) = p(1−p) 이므로 z ≈ 0. 양수 z는 그 축을 따라
    체계적 편차(계단 포함)가 남아 있다는 뜻이다.

대조 둘:
    · 같은 통계를 LGBM 잔차로 — 트리도 못 잡으면 못 배우는 구조다.
      후보 조건은 z_mlp 큼 && z_lgb 작음 (트리만 잡는 구조).
    · 컬럼을 행 방향으로 섞은 널 — 통계 교정 확인 (z ≈ 0 이어야 한다).

실행:  .venv/Scripts/python.exe -u src/stepprobe.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache
from config import TRAIN_CSV
from data import load_csv
from mlp_infer import EMB_COLS

FOLDS = [2023, 2024]
NBINS = 24
SMOOTH_DF = 4            # 3차 다항 = 파라미터 4개
MIN_BINS = 12            # 분위수 병합 후 이보다 적으면 저카디널리티로 보고 건너뛴다
SEED = 42


def excess_z(v, r, var_row, nbins=NBINS):
    """v 축을 따라 잔차 r의 '매끄러운 곡선 초과' 구조를 z점수로 잰다.

    반환 (z, 사용 구간 수). 저카디널리티/퇴화 컬럼은 (nan, b)로 보고한다.
    """
    ok = np.isfinite(v)
    v, r, var_row = v[ok], r[ok], var_row[ok]
    edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, nbins + 1)))
    nb = len(edges) - 1
    if nb < MIN_BINS:
        return np.nan, nb
    b = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, nb - 1)

    n = np.bincount(b, minlength=nb).astype(np.float64)
    keep = n > 0
    n = n[keep]
    rbar = (np.bincount(b, weights=r, minlength=nb)[keep]) / n
    vbar = (np.bincount(b, weights=var_row, minlength=nb)[keep]) / n

    # 매끄러운 대조군: 구간 누적질량 중심 좌표에 대한 가중 3차 다항.
    # 등질량 구간이 아닐 수 있으므로(동률 병합) 구간 인덱스가 아니라 질량 좌표를 쓴다.
    x = (np.cumsum(n) - n / 2.0) / n.sum()
    fit = np.polyval(np.polyfit(x, rbar, SMOOTH_DF - 1, w=np.sqrt(n)), x)

    chi2 = float(np.sum(n * (rbar - fit) ** 2 / vbar))
    dof = len(n) - SMOOTH_DF
    return (chi2 - dof) / np.sqrt(2.0 * dof), len(n)


def main():
    t0 = time.time()
    cache = foldcache.load(FOLDS)
    df = load_csv(TRAIN_CSV)
    rng = np.random.default_rng(SEED)

    results = {}          # col -> {fold: (z_mlp, z_lgb, z_null, nb)}
    for f in FOLDS:
        z_lgb, z_mlp, y, _ = cache[f]
        _, _, _, Xva, yva, _, _ = foldcache.fold_data(df, f)
        assert np.array_equal(y, yva), "캐시 y와 fold_data y가 어긋난다"
        print(f"\nfold {f}: 검증 {len(y):,}행  피처 준비 ({time.time()-t0:.0f}s)")

        for tag, z in (("mlp", z_mlp), ("lgb", z_lgb)):
            p = 1.0 / (1.0 + np.exp(-z))
            r = y - p
            var_row = p * (1.0 - p)
            for c in Xva.columns:
                if c in EMB_COLS:
                    continue
                v = Xva[c].to_numpy(dtype=np.float64)
                zc, nb = excess_z(v, r, var_row)
                results.setdefault(c, {}).setdefault(f, {})[tag] = (zc, nb)

        # 셔플 널 (MLP 잔차 기준): 통계 교정이 맞는지 컬럼마다 1회
        p = 1.0 / (1.0 + np.exp(-z_mlp))
        r, var_row = y - p, p * (1.0 - p)
        for c in Xva.columns:
            if c in EMB_COLS:
                continue
            v = rng.permutation(Xva[c].to_numpy(dtype=np.float64))
            zc, _ = excess_z(v, r, var_row)
            results[c][f]["null"] = (zc, 0)

    # ── 보고 ────────────────────────────────────────────────────────
    rows = []
    for c, byf in results.items():
        if any("mlp" not in byf.get(f, {}) for f in FOLDS):
            continue
        zm = [byf[f]["mlp"][0] for f in FOLDS]
        zl = [byf[f]["lgb"][0] for f in FOLDS]
        zn = [byf[f]["null"][0] for f in FOLDS]
        if any(np.isnan(zm)) or any(np.isnan(zl)):
            continue
        rows.append((c, zm, zl, zn, min(zm)))
    rows.sort(key=lambda t: -t[4])

    nulls = [z for _, _, _, zn, _ in rows for z in zn if np.isfinite(z)]
    print(f"\n셔플 널 교정: median {np.median(nulls):+.2f}  "
          f"p95 {np.quantile(nulls, 0.95):+.2f}  max {max(nulls):+.2f}"
          f"  (컬럼 {len(rows)}개 × fold {len(FOLDS)})")

    hdr = (f"{'컬럼':<34} {'mlp23':>7} {'mlp24':>7} {'lgb23':>7} {'lgb24':>7}"
           f" {'Δ23':>7} {'Δ24':>7}")
    print("\n초과구조 z점수 — min(fold) 내림차순, 상위 25")
    print(hdr)
    for c, zm, zl, _, _ in rows[:25]:
        d = [m - l for m, l in zip(zm, zl)]
        print(f"{c:<34} {zm[0]:>7.1f} {zm[1]:>7.1f} {zl[0]:>7.1f} {zl[1]:>7.1f}"
              f" {d[0]:>7.1f} {d[1]:>7.1f}")

    print("\n판독 기준:")
    print("  · 후보 = mlp 양 fold 큼(≥5) && lgb 양 fold 그보다 뚜렷이 작음")
    print("  · mlp ≈ lgb 로 같이 크면: 아무도 못 배운 구조 (PLE 무관)")
    print("  · 널 p95가 ~2 안이면 통계 교정 정상")
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
