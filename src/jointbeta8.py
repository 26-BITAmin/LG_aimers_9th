"""외부 제안 검증 3 — 수축 중심을 `β̄_folds`로 (사용자가 가져옴) + 추세 외삽 앵커.

받은 제안
    β̂ = argmin  Σ_folds BS(z_blend + Xβ + c)  +  λ‖β − β̄_folds‖²

    §7.86에서 `β_deploy` 앵커가 순환(평가 fold를 봄)이라 기각됐으니 중심을
    **fold 간 β 평균**으로 바꾸자는 것. 순환은 사라진다 (적합 fold만 쓰면).

미리 아는 것 — 경로가 평평할 가능성
    §7.86 측정:  pooled({21,22,23}) β=(1.657,1.475,.0482,.0212) → 2024 −5.3
                 solo 3개 단순평균  β=(1.667,1.480,.0497,.0215) → 2024 −5.8
    **양 끝점(λ=0의 pooled, λ=∞의 β̄)이 이미 같은 자리다.** 그래도 잰다.

같이 재는 것 — 추세 외삽 앵커 (여기가 앵커 재구성이 값을 낼 수 있는 곳)
    §7.84: `count4`의 fold별 최적이 **단조 증가**한다 (1.28 → 1.46 → 1.71 → 2.02).
    그러면 **평균은 구조적으로 뒤처진다** — {21,22,23} 평균 1.48 vs 2024 실제 2.02.
    앵커를 β̄ 대신 **시즌에 대한 선형 외삽**으로 두면 그 편향이 없어진다.
    (memory: 감쇠 모형에 개체 회전을 넣어라 — 추세가 있는 양을 평균으로 대표하면
     체계적으로 틀린다. 같은 종류의 실수를 앵커에서 반복하지 않는다.)

앵커 셋을 **완전히 같은 절차**에서 비교한다 (§7.86의 대조 규약):
    A0      0            — 종전 ridge. 순환 없음
    Amean   β̄_fit       — 사용자 제안. 순환 없음
    Atrend  선형 외삽     — 추세 보정. 순환 없음 (적합 fold의 시즌만 쓴다)

⛔ 전방향만. 평가 fold마다 **그 이전 fold 전부**로 적합한다 (§7.86 절차).
   λ→∞ 행은 앵커 자신의 전이 점수다 — 앵커 품질을 그대로 읽을 수 있다.

실행:  .venv/Scripts/python.exe -u src/jointbeta8.py 2>&1 | tee experiments/jointbeta8.log
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
from jointbeta7 import fit_anchor
from config import ASOF_AXES, CTX_AXES, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import _logit, bss, shift_to_prior

FOLDS = [2021, 2022, 2023, 2024]
TASKS = [(2022, [2021]), (2023, [2021, 2022]), (2024, [2021, 2022, 2023])]
LAMS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 1e2, 1e4]


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
        D[f] = dict(X=X, y=y, z0=z0, sd=X.std(axis=0))

    def ev(e, b):
        z = D[e]["z0"] + D[e]["X"] @ b
        return bss(D[e]["y"], shift_to_prior(1 / (1 + np.exp(-z)), D[e]["y"].mean()))

    base = {e: ev(e, dep) for e, _ in TASKS}
    solo = {}
    for f in FOLDS:
        solo[f] = fit_anchor(D[f]["X"] / D[f]["sd"], D[f]["z0"], D[f]["y"],
                             0.0, np.zeros(len(dep))) / D[f]["sd"]

    print(f"축 {names}   배포 β {dict(zip(names, dep))}")
    print(f"배포 기준선  " + "   ".join(f"{e}: {base[e]:.1f}" for e, _ in TASKS))
    print(f"\n  fold 단독 적합 β (R-only)")
    print(f"    {'축':<16}{'배포':>9}" + "".join(f"{f:>10}" for f in FOLDS))
    for i, nm in enumerate(names):
        print(f"    {nm:<16}{dep[i]:>9.4f}"
              + "".join(f"{solo[f][i]:>10.4f}" for f in FOLDS))

    # ── 앵커 셋 ─────────────────────────────────────────────────────────
    def anchors(e, fs):
        a = {"A0 (0)": np.zeros(len(dep)),
             "Amean (β̄_fit)": np.mean([solo[f] for f in fs], axis=0)}
        if len(fs) >= 2:                       # 선형 외삽은 점 2개부터
            x = np.array(fs, dtype=float)
            Y = np.array([solo[f] for f in fs])
            sl = ((x - x.mean()) @ (Y - Y.mean(0))) / ((x - x.mean()) ** 2).sum()
            a["Atrend (선형외삽)"] = Y.mean(0) + sl * (e - x.mean())
        return a

    print(f"\n{'='*80}\n■ 앵커 값 자체 (적합 fold만 사용 — 순환 없음)\n{'='*80}")
    for e, fs in TASKS:
        print(f"\n  평가 {e}  ← 적합 {fs}")
        print(f"    {'앵커':<20}" + "".join(f"{n:>14}" for n in names)
              + f"{'앵커 단독 전이':>16}")
        for an, av in anchors(e, fs).items():
            print(f"    {an:<20}" + "".join(f"{x:>14.4f}" for x in av)
                  + f"{ev(e, av)-base[e]:>+16.1f}")
        print(f"    {'(참고) 그 fold 실제':<20}"
              + "".join(f"{solo[e][i]:>14.4f}" for i in range(len(names))))

    # ── λ 경로 ──────────────────────────────────────────────────────────
    keys = ["A0 (0)", "Amean (β̄_fit)", "Atrend (선형외삽)"]
    for an in keys:
        print(f"\n{'='*80}\n■ 앵커 = {an}   ·  합산 적합 + λ‖β−앵커‖²\n{'='*80}")
        print(f"  {'λ':>8} │ " + " │ ".join(f"{e:>14}" for e, _ in TASKS)
              + " │   최소    평균  양수")
        best = None
        for lam in LAMS:
            ds, ok = [], True
            for e, fs in TASKS:
                A = anchors(e, fs)
                if an not in A:
                    ok = False
                    ds.append(float("nan"))
                    continue
                Xq = np.vstack([D[f]["X"] for f in fs])
                zq = np.concatenate([D[f]["z0"] for f in fs])
                yq = np.concatenate([D[f]["y"] for f in fs])
                sq = Xq.std(axis=0)
                b = fit_anchor(Xq / sq, zq, yq, lam, A[an] * sq) / sq
                ds.append(ev(e, b) - base[e])
            fin = [d for d in ds if np.isfinite(d)]
            line = (f"  {lam:>8.4g} │ " + " │ ".join(
                ("      —       " if not np.isfinite(d) else f"{d:>+14.1f}")
                for d in ds)
                + f" │ {min(fin):>+6.1f}  {np.mean(fin):>+6.1f}"
                f"  {sum(d > 0 for d in fin)}/{len(fin)}")
            print(line)
            if best is None or min(fin) > best[1]:
                best = (lam, min(fin), np.mean(fin), sum(d > 0 for d in fin), len(fin))
        print(f"  → 최소 기준 최선: λ={best[0]:g}  최소 {best[1]:+.1f}"
              f"  평균 {best[2]:+.1f}  양수 {best[3]}/{best[4]}"
              + ("   ✅ 전 fold 양수" if best[3] == best[4] else ""))


if __name__ == "__main__":
    main()
