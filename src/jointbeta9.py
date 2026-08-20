"""외부 제안 검증 4 — ① `β_deploy` 순환 라벨의 결정적 검정 ② recency 가중 pooling.

배경 (사용자가 가져온 지적)
    §7.87 표의 `Adeploy` 행에 ⚠️순환 라벨을 붙였는데, 근거를 확인 안 하고 붙였다.
    확인했다:
      config.py:407/438/501 · MODEL.md §7.37 — **β는 전부 fold2023·2024 스윕에서
      나왔고, LB로 다시 고르는 것은 규칙 2로 금지**되어 있으며 실제로 §7.37에서
      "실현율 30%니까 β를 올리자"를 규칙 2 위반으로 명시 기각했다.
      → **fold2024 평가에 대해서는 순환이 맞다.** (LB 평가에 대해서는 순환이 아니다.
        평가 대상이 뭐냐에 따라 갈리는 문제였다.)

    그런데 지적의 핵심은 살아 있다 — **누출 채널이 매우 좁다.** β_deploy는 성긴
    격자에서 고원 기하 규칙으로 뽑은 **4개 숫자**고, 앵커 적합은 연속 4-벡터를 받는다.
    §7.86의 6.5점이 전부 그 4개 숫자 탓인지는 재지 않았다.

① 결정적 검정 — **밴드 규칙을 평가 fold 이전 fold만으로 적용**
    평가 e → {f < e} 의 고원으로 밴드 규칙 → β_rule(정직) → e 평가.
      β_rule ≈ β_deploy 이고 전이도 비슷  →  누출이 아니라 **규칙이 일한다**
      훨씬 나쁨                            →  누출이 실제로 무겁게 작용했다
    ⚠️ 다른 축을 뭘로 고정하고 스윕하냐가 문제다. `β_deploy`로 고정하면 또 순환이다.
       → **과거 fold pooled 적합값**으로 고정한다 (정직). 좌표 1회 갱신.

② recency 가중 pooling (사용자 제안)
    β̂(γ) = argmin Σ_k γ^(e−k)·BS_k(β)
      γ=1 현재 pooled / γ→0 단일 fold / 중간 = 드리프트 부분 추종
    `Atrend`와 결정적으로 다른 점: **외삽하지 않는다.** f2023의 gap_success 튐(.073)은
    가중평균에서 감쇠될 뿐 기울기로 승격되지 않는다 (§7.87의 .0836 폭주가 구조적으로 차단).
    γ는 축별 선택이 아니라 **파라미터 하나**이고, λ와 같은 이유로 홀드아웃으로
    고르면 안 된다 → **전 구간 곡선을 보고한다.**

실행:  .venv/Scripts/python.exe -u src/jointbeta9.py 2>&1 | tee experiments/jointbeta9.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import jointbeta as J
from jointbeta2 import GRID
from config import ASOF_AXES, CTX_AXES, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import _logit, bss, shift_to_prior

FOLDS = [2021, 2022, 2023, 2024]
TASKS = [(2023, [2021, 2022]), (2024, [2021, 2022, 2023])]   # 과거 2개 이상만
GAMMAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.35, 0.2, 0.1]


def wfit(X, z0, y, w, iters=60):
    """가중 Brier 최소화. 변형 A(β4, 절편·c 고정). X는 표준화 단위."""
    W = w.sum()
    th = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(z0 + X @ th)))
        v = p * (1 - p)
        r = p - y
        g = X.T @ (w * 2.0 * r * v) / W
        h = w * 2.0 * (v * v + r * v * (1.0 - 2.0 * p))
        H = (X.T * (h / W)) @ X + np.eye(X.shape[1]) * 1e-10
        try:
            st = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        th -= st
        if np.max(np.abs(st)) < 1e-13:
            break
    return th


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = J.foldcache.load(FOLDS, verbose=False)
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])

    D = {}
    for f in FOLDS:
        X, names, _, va = J.axis_matrix(df, s, f)
        y = cached[f][2]
        z0 = (1 - MLP_ALPHA) * cached[f][0] + MLP_ALPHA * cached[f][1]
        assert np.array_equal(y, va[TARGET_COL].to_numpy(dtype=np.float64))
        m = va["game_type"].to_numpy() != "F"
        X, y, z0 = X[m], y[m], z0[m]
        z0 = _logit(shift_to_prior(1 / (1 + np.exp(-z0)), y.mean()))
        D[f] = dict(X=X, y=y, z0=z0, sd=X.std(axis=0), n=len(y))

    def sc(f, b):
        z = D[f]["z0"] + D[f]["X"] @ b
        return bss(D[f]["y"], shift_to_prior(1 / (1 + np.exp(-z)), D[f]["y"].mean()))

    base = {f: sc(f, dep) for f in FOLDS}
    print(f"축 {names}   배포 β {dict(zip(names, dep))}")
    print("배포 기준선  " + "   ".join(f"{f}: {base[f]:.1f}" for f in FOLDS))

    def pooled(fs, e, gamma=1.0):
        """Σ_k γ^(e−k)·BS_k 를 최소화. 각 fold를 **자기 n으로 정규화**해서 합친다."""
        Xq = np.vstack([D[f]["X"] for f in fs])
        zq = np.concatenate([D[f]["z0"] for f in fs])
        yq = np.concatenate([D[f]["y"] for f in fs])
        wq = np.concatenate([np.full(D[f]["n"], gamma ** (e - f) / D[f]["n"])
                             for f in fs])
        sq = Xq.std(axis=0)
        return wfit(Xq / sq, zq, yq, wq) / sq

    # ── ① 밴드 규칙을 과거 fold만으로 ───────────────────────────────────
    print(f"\n{'='*80}\n■ ① 밴드 규칙 — 평가 fold **이전** fold만으로 적용 (정직)\n{'='*80}")
    print("  다른 축은 `과거 pooled 적합값`으로 고정한다 (β_deploy로 고정하면 또 순환).")
    print("  규칙: 고원(양 fold 전부 β=0 대비 양수) → 상한/1.5 이하 최대 격자값 → 한 칸 아래\n")
    rule_b = {}
    for e, fs in TASKS:
        p0 = pooled(fs, e)
        b = p0.copy()
        print(f"  ── 평가 {e}  ← 고원을 {fs} 에서만 ──")
        print(f"    {'축':<15}{'고정(pooled)':>14}{'고원':>20}{'허용상한':>10}"
              f"{'규칙 β':>9}{'배포':>9}")
        for i, nm in enumerate(names):
            g = GRID[nm]
            d = {}
            for gv in g:
                bb = p0.copy()
                bb[i] = gv
                d[gv] = [sc(f, bb) for f in fs]
            z0v = d[0.0]
            ok = [gv for gv in g if gv > 0
                  and all(d[gv][k] - z0v[k] > 0 for k in range(len(fs)))]
            if not ok:
                rule = p0[i]
                print(f"    {nm:<15}{p0[i]:>14.4f}{'없음':>20}{'—':>10}"
                      f"{rule:>9.4f}{dep[i]:>9.4f}")
            else:
                hi = max(ok)
                allow = [gv for gv in ok if gv * 1.5 <= hi]
                if allow:
                    mx = max(allow)
                    j = g.index(mx)
                    rule = g[j - 1] if j > 0 else mx
                else:
                    rule = min(ok)
                print(f"    {nm:<15}{p0[i]:>14.4f}"
                      f"{f'[{min(ok):g}, {hi:g}]':>20}{hi/1.5:>10.4f}"
                      f"{rule:>9.4f}{dep[i]:>9.4f}")
            b[i] = rule
        rule_b[e] = b
        print(f"    → β_rule(정직) {dict(zip(names, np.round(b, 4)))}")
        print(f"      전이  β_rule {sc(e, b)-base[e]:+.1f}"
              f"   |  pooled {sc(e, p0)-base[e]:+.1f}"
              f"   |  배포 0.0 (기준)\n")

    # ── ② recency 가중 pooling ──────────────────────────────────────────
    print(f"{'='*80}\n■ ② recency 가중 pooling  β̂(γ)=argmin Σ_k γ^(e−k)·BS_k\n{'='*80}")
    print("  γ=1 현재 pooled · γ→0 최근 fold 단독. **외삽 없음** (가중평균이라 튐이 감쇠).\n")
    print(f"  {'γ':>6} │ " + " │ ".join(f"{e:>13}" for e, _ in TASKS)
          + " │  최소   평균  양수")
    keep = {}
    for gm in GAMMAS:
        ds = []
        for e, fs in TASKS:
            b = pooled(fs, e, gm)
            keep[(e, gm)] = b
            ds.append(sc(e, b) - base[e])
        print(f"  {gm:>6.2f} │ " + " │ ".join(f"{d:>+13.1f}" for d in ds)
              + f" │ {min(ds):>+6.1f} {np.mean(ds):>+6.1f}"
              f"  {sum(d > 0 for d in ds)}/{len(ds)}")

    print(f"\n  γ에 따른 β 궤적 (평가 2024, 적합 21·22·23) — 주장된 기전 확인")
    print(f"    {'γ':>6}" + "".join(f"{n:>14}" for n in names))
    for gm in GAMMAS:
        print(f"    {gm:>6.2f}" + "".join(f"{x:>14.4f}" for x in keep[(2024, gm)]))
    print(f"    {'실제2024':>6}" + "".join(f"{x:>14.4f}" for x in
          (J.fit(D[2024]["X"] / D[2024]["sd"], D[2024]["z0"], D[2024]["y"],
                 [False, False, True, True, True, True], 0.0, "brier")[2:]
           / D[2024]["sd"])))
    print(f"    {'배포':>6}" + "".join(f"{x:>14.4f}" for x in dep))


if __name__ == "__main__":
    main()
