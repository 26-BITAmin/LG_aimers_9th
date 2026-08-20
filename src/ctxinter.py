"""δ 보정항끼리의 상호작용 스크린 — MLP 대신 명시적 저차 항 (학습 없음)

§7.43은 δ(투수, hand×count4) **교차 테이블**을 쟀다 — 투수별 결합 편차.
여기서 재는 것은 다른 물건이다: 이미 배포된 두 δ **스칼라의 결합 방식**.
배포는 z + β_h·δ_h + β_c·δ_c 로 가법인데, 남은 신호가 있다면

    δh·δc      두 보정이 겹칠 때 가법이 과/과소인가 (순수 곱 상호작용)
    δh·z, δc·z 모델 확신도에 따라 δ를 다르게 실어야 하는가
    δh², δc²   β가 δ 크기에 대해 휘어야 하는가 (자기 곡률)

"LayerNorm 얕은 MLP로 상호작용을 보자"는 제안의 검증 가능한 알맹이가 이
다섯 항이다 — 신호 총량 ≈0.004 nat에서 MLP가 표현할 수 있는 것은 저차 항뿐이고,
명시항은 학습 0초에 홀드아웃 부풀림(파라미터 수 비례)도 안 산다.

설계
    기준선   z0 = 블렌드(α) + 1.875·δ_hand + 1.25·δ_count4  (배포 상태 전체)
    T2       잔차 r = y − p0 를 [1, δh, δc, 항] 에 OLS → 항의 partial t
             (주효과를 공변량으로 넣는다 — 곱항이 주효과를 흉내내는 것을 차단)
    위약     죽은 축 δ(홈/원정, 이닝3 — §7.36 기각)와의 곱 4개.
             같은 rate(p) 기반·같은 투수 축이라 공유 항 구조를 유지한 위약이다.
             후보의 |t|가 위약 |t| 분포를 못 넘으면 그 t는 구조가 주는 공짜다.
    T3       유의 후보만: 항을 std(δh)로 정규화해 γ 스윕, 양 fold BSS 이득.

다중검정: 후보 5개 중 최댓값은 편향돼 있다. 위약 4개가 경험적 영분포다.

실행:  .venv/Scripts/python.exe -u src/ctxinter.py 2>&1 | tee experiments/ctxinter.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache
from config import CTX_AXES, CTX_K, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from ctxsplit import build_delta, lookup
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
K = CTX_K
GAMMAS = [-2.0, -1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 2.0]


def partial_t(r, controls, term):
    """r ~ [1, controls..., term] OLS에서 term 계수의 t."""
    X = np.column_stack([np.ones_like(r)] + controls + [term])
    beta, _, _, _ = np.linalg.lstsq(X, r, rcond=None)
    res = r - X @ beta
    dof = len(r) - X.shape[1]
    sigma2 = float((res ** 2).sum() / dof)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = float(np.sqrt(sigma2 * XtX_inv[-1, -1]))
    return float(beta[-1]), (float(beta[-1]) / se if se > 0 else 0.0)


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    pid = df["pitcher_id"].to_numpy(dtype=np.int64)

    bh = df["batter_hand"].to_numpy(dtype=np.int64)
    b = df["balls_before"].to_numpy(dtype=np.int64)
    st = df["strikes_before"].to_numpy(dtype=np.int64)
    c4 = (b >= 3).astype(np.int64) + 2 * (st >= 2).astype(np.int64)
    home = (df["top_bottom"].astype(str).to_numpy() == "T").astype(np.int64)
    inn3 = np.clip((df["inning"].to_numpy(dtype=np.int64) - 1) // 3, 0, 2)

    beta_h = dict(CTX_AXES)["hand"]
    beta_c = dict(CTX_AXES)["count4"]
    print(f"train {df.shape}   k={K}   배포 β: hand {beta_h} · count4 {beta_c}")

    cached = foldcache.load(FOLDS, verbose=False)
    rows = {}

    for fold in FOLDS:
        tr, va = s < fold, s == fold
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]

        def drow(code, m):
            u, d, _ = build_delta(pid[tr], code[tr], y[tr], m, K)
            out, hit = lookup(u, d, pid[va], code[va], m)
            return out, hit

        dh, hit_h = drow(bh, 8)
        dc, hit_c = drow(c4, 4)
        dd_home, _ = drow(home, 2)
        dd_inn, _ = drow(inn3, 4)

        z0 = z + beta_h * dh + beta_c * dc
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean())
        b0 = bss(yv, p0)
        r = yv - p0
        zc = (z - z.mean()) / z.std()

        print(f"\n── fold {fold}   기준선 BSS {b0:.2f}   "
              f"적중 hand {hit_h.mean():.3f} / count4 {hit_c.mean():.3f}")

        # 주효과 자체의 partial t (재적용 대조 — ≈0이어야 나머지를 믿는다)
        _, t_h = partial_t(r, [dc], dh)
        _, t_c = partial_t(r, [dh], dc)
        print(f"   재적용 대조   δh t {t_h:+.2f}   δc t {t_c:+.2f}")

        cands = [
            ("δh·δc",  dh * dc,      "후보"),
            ("δh·z",   dh * zc,      "후보"),
            ("δc·z",   dc * zc,      "후보"),
            ("δh²",    dh * dh,      "후보"),
            ("δc²",    dc * dc,      "후보"),
            ("δh·δ홈", dh * dd_home, "위약"),
            ("δc·δ홈", dc * dd_home, "위약"),
            ("δh·δ이닝", dh * dd_inn, "위약"),
            ("δc·δ이닝", dc * dd_inn, "위약"),
        ]

        for name, term, role in cands:
            nz = float((term != 0).mean())
            _, t = partial_t(r, [dh, dc], term)
            rows.setdefault(name, {"role": role})[fold] = (t, nz)
            print(f"   {role} {name:10s}  partial t {t:+6.2f}   비영 {nz:.3f}")

        # T3: γ 스윕 (항을 std(δh) 스케일로 정규화 → γ가 β와 같은 눈금)
        sd_h = dh[dh != 0].std() if (dh != 0).any() else 1.0
        for name, term, role in cands:
            sd_t = term[term != 0].std() if (term != 0).any() else 0.0
            if sd_t == 0:
                continue
            tn = term * (sd_h / sd_t)
            gains = []
            for g in GAMMAS:
                p1 = shift_to_prior(1.0 / (1.0 + np.exp(-(z0 + g * tn))), yv.mean())
                gains.append(bss(yv, p1) - b0)
            best = max(gains)
            rows[name].setdefault("gain", {})[fold] = (best, GAMMAS[int(np.argmax(gains))], gains)

    print("\n══ 요약 (partial t f23 / f24 · 최적 γ 이득 f23 / f24) ══")
    print("   시드 노이즈 대역 ±3.3 · 위약 4개가 경험적 영분포")
    for name, d in rows.items():
        t23, _ = d.get(2023, (float("nan"), 0))
        t24, _ = d.get(2024, (float("nan"), 0))
        g = d.get("gain", {})
        g23 = g.get(2023, (float("nan"), 0, []))[0]
        g24 = g.get(2024, (float("nan"), 0, []))[0]
        print(f"   {d['role']} {name:10s}  t {t23:+6.2f} / {t24:+6.2f}   "
              f"이득 {g23:+6.2f} / {g24:+6.2f}")


if __name__ == "__main__":
    main()
