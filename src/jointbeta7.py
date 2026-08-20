"""외부 제안 검증 2 — 앵커 있는 벌점과 **다중 fold 합산 적합** (사용자가 가져옴).

받은 제안
    β̂ = argmin  Σ_folds BS(z_blend + Xβ + c)  +  λ‖β − β_deploy‖²

두 가지가 섞여 있다. 하나는 못 쓰고 하나는 진짜 개선이다.

⛔ 못 쓰는 쪽 — 앵커가 `β_deploy`인 것
    `β_deploy`는 **fold 2023·2024를 보고** 골라진 값이다 (§7.60·§7.61·§7.63).
    fold2023에 적합하면서 그쪽으로 당기면 ① 2023을 두 번 쓰고 ② **2024 정보가
    벌점항을 타고 "2023에 적합한" 추정량 안으로 샌다.**
    → `적합2023→평가2024`가 좋아져도 누출인지 진짜인지 구분 불가. **평가 도구가 아니다.**
    (배포 절차로는 말이 된다: "현재 값에서 출발해 증거가 셀 때만 움직인다."
     다만 λ를 평가 fold를 보고 고르면 안 되고, λ→∞면 그냥 현행 유지다.)

✅ 진짜 개선 — `Σ_folds`
    지금까지 β를 **fold 하나**에만 적합했다. 전방향을 지키면서 {2021,2022,2023}에
    합쳐 적합하고 2024로 평가할 수 있다. 표본 3배 + fold 고유 노이즈 평균.

✅ 순환하지 않는 앵커
    `β_deploy` 대신 **직전 fold의 적합값** β̂(t−1)을 앵커로 쓰면 미래를 안 본다.
    제안의 의도(급격한 이동 억제)는 살리고 순환은 없앤다.

재는 것 (전부 R-only. §7.84 게이트 통과분)
    ① 방향 검정 — β(t) = β_dep + t·(β̂₂₀₂₃ − β_dep) 를 t로 훑어 fold2024 점수.
       t>0 어디서도 배포를 못 넘으면 "그 방향 자체가 무익"이고 λ 조정은 무의미하다.
    ② 합산 적합 {2021,2022,2023} → 2024.  λ‖β‖² (앵커 0, 깨끗) 스윕.
    ③ 앵커 β̂(직전 fold), 순환 없음.  fold2023 적합 + λ‖β−β̂₂₀₂₂‖² → 2024.
    ④ 앵커 β_deploy (순환. 대조군으로만).  같은 설정에서 ③과 비교.

⚠️ 합산 적합 전에 각 fold의 z0를 **자기 시즌 prior로 먼저 이동**시킨다.
   시즌마다 ȳ가 다른데(.5128/.5037/.5031/.4897) 그냥 이어붙이면 절편이 고정된
   변형 A에서 **β가 레벨 차이를 흡수**한다.

실행:  .venv/Scripts/python.exe -u src/jointbeta7.py 2>&1 | tee experiments/jointbeta7.log
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
from config import ASOF_AXES, CTX_AXES, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import _logit, bss, shift_to_prior

FOLDS = [2021, 2022, 2023, 2024]
FIT = [2021, 2022, 2023]          # 전방향 — 평가 fold(2024)보다 전부 과거
EVAL = 2024
LAMS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]


def fit_anchor(X, z0, y, lam, anchor, iters=60):
    """변형 A(β4, 절편·c 고정) + `λ‖β − anchor‖²`. X·anchor 둘 다 **표준화 단위**.

    앵커가 0이면 종전 ridge와 동일하다 (`J.fit`과 비트 동일할 필요는 없고
    같은 해에 수렴하면 된다 — 아래 assert로 확인한다).
    """
    n, k = X.shape
    th = np.array(anchor, dtype=float).copy()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(z0 + X @ th)))
        v = p * (1 - p)
        r = p - y
        g1 = 2.0 * r * v
        g2 = 2.0 * (v * v + r * v * (1.0 - 2.0 * p))
        g = X.T @ g1 / n + 2.0 * lam * (th - anchor)
        H = (X.T * (g2 / n)) @ X + 2.0 * lam * np.eye(k) + np.eye(k) * 1e-10
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
        m = va["game_type"].to_numpy() != "F"           # R 전용
        X, y, z0 = X[m], y[m], z0[m]
        # ⚠️ 합산 전 레벨 맞춤 — 시즌 ȳ가 다르므로 자기 prior로 먼저 민다
        z0 = _logit(shift_to_prior(1 / (1 + np.exp(-z0)), y.mean()))
        D[f] = dict(X=X, y=y, z0=z0, sd=X.std(axis=0))
        print(f"fold {f}  R n={len(y):,}  ȳ={y.mean():.4f}  std(z0)={z0.std():.4f}")
    base = bss(D[EVAL]["y"], shift_to_prior(
        1 / (1 + np.exp(-(D[EVAL]["z0"] + D[EVAL]["X"] @ dep))), D[EVAL]["y"].mean()))
    print(f"\n축 {names}   배포 β {dict(zip(names, dep))}")
    print(f"평가 fold {EVAL} 배포 기준선 {base:.1f}")

    def ev(b):
        z = D[EVAL]["z0"] + D[EVAL]["X"] @ b
        return bss(D[EVAL]["y"], shift_to_prior(
            1 / (1 + np.exp(-z)), D[EVAL]["y"].mean()))

    # 각 fold 단독 적합 (원단위)
    solo = {}
    for f in FOLDS:
        t = fit_anchor(D[f]["X"] / D[f]["sd"], D[f]["z0"], D[f]["y"], 0.0,
                       np.zeros(len(dep)))
        solo[f] = t / D[f]["sd"]
    print(f"\n  fold 단독 적합 β (R-only)")
    print(f"    {'축':<16}{'배포':>9}" + "".join(f"{f:>10}" for f in FOLDS))
    for i, nm in enumerate(names):
        print(f"    {nm:<16}{dep[i]:>9.4f}" + "".join(f"{solo[f][i]:>10.4f}"
                                                     for f in FOLDS))

    # ── ① 방향 검정 ─────────────────────────────────────────────────────
    print(f"\n{'='*78}\n■ ① 방향 검정 — β(t) = β_dep + t·(β̂₂₀₂₃ − β_dep)\n{'='*78}")
    print("  t=0 배포, t=1 결합적합. **t>0 어디서도 배포를 못 넘으면 그 방향이 무익하고,")
    print("  λ를 어떻게 잡든(=t를 어떻게 줄이든) 배포보다 나아질 수 없다.\n")
    d23 = solo[2023] - dep
    ts = [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3]
    vs = [ev(dep + t * d23) for t in ts]
    print("      t   " + "".join(f"{t:>9.2f}" for t in ts))
    print("    BSS   " + "".join(f"{v:>9.1f}" for v in vs))
    print("  배포Δ   " + "".join(f"{v-base:>+9.1f}" for v in vs))
    bt = ts[int(np.argmax(vs))]
    print(f"    → t* = {bt:+.2f}  (배포Δ {max(vs)-base:+.1f})"
          f"   {'배포가 이미 최적 방향' if bt <= 0 else '그 방향에 여지 있음'}")

    # ── ② 합산 적합 ─────────────────────────────────────────────────────
    print(f"\n{'='*78}\n■ ② 합산 적합 {FIT} → {EVAL}  (앵커 0, 순환 없음)\n{'='*78}")
    Xp = np.vstack([D[f]["X"] for f in FIT])
    zp = np.concatenate([D[f]["z0"] for f in FIT])
    yp = np.concatenate([D[f]["y"] for f in FIT])
    sp = Xp.std(axis=0)
    print(f"  합산 n={len(yp):,}\n")
    print(f"  {'λ':>8}{'BSS':>9}{'배포Δ':>9}   " + "".join(f"{n:>13}" for n in names))
    for lam in LAMS:
        b = fit_anchor(Xp / sp, zp, yp, lam, np.zeros(len(dep))) / sp
        print(f"  {lam:>8.4f}{ev(b):>9.1f}{ev(b)-base:>+9.1f}   "
              + "".join(f"{x:>13.4f}" for x in b))
    # 단독 적합 3개의 단순 평균 — 합산과 다른가
    avg = np.mean([solo[f] for f in FIT], axis=0)
    print(f"\n  참고: 단독 적합 3개의 평균 → BSS {ev(avg):.1f} ({ev(avg)-base:+.1f})"
          f"   β {dict(zip(names, np.round(avg, 4)))}")

    # ── ③④ 앵커 비교 ───────────────────────────────────────────────────
    for tag, anc, note in (
            ("③ 앵커 = β̂(2022)", solo[2022], "순환 없음 (미래를 안 본다)"),
            ("④ 앵커 = β_deploy", dep, "⚠️ 순환 — β_dep가 2023·2024를 봤다")):
        print(f"\n{'='*78}\n■ {tag}  ·  fold2023 적합 → {EVAL} 평가\n"
              f"  {note}\n{'='*78}")
        print(f"  {'λ':>8}{'BSS':>9}{'배포Δ':>9}   " + "".join(f"{n:>13}" for n in names))
        for lam in LAMS:
            b = fit_anchor(D[2023]["X"] / D[2023]["sd"], D[2023]["z0"], D[2023]["y"],
                           lam, anc * D[2023]["sd"]) / D[2023]["sd"]
            print(f"  {lam:>8.4f}{ev(b):>9.1f}{ev(b)-base:>+9.1f}   "
                  + "".join(f"{x:>13.4f}" for x in b))

    # ── ⑤ 전방향 누적 합산 — 평가 fold마다 **그 이전 전부**로 적합 ──────
    #    ①②③④는 전부 2024 하나로 평가했다. t*나 λ*를 그 fold를 보고 고르면
    #    in-fold 최적이지 전이 추정이 아니다. 평가 fold를 3개로 늘린다.
    print(f"\n{'='*78}\n■ ⑤ 전방향 누적 합산 — 평가 fold마다 그 이전 fold 전부로 적합\n{'='*78}")
    tasks = [(2022, [2021]), (2023, [2021, 2022]), (2024, [2021, 2022, 2023])]
    bases = {e: bss(D[e]["y"], shift_to_prior(
        1 / (1 + np.exp(-(D[e]["z0"] + D[e]["X"] @ dep))), D[e]["y"].mean()))
        for e, _ in tasks}
    print("  배포 기준선  " + "   ".join(f"{e}: {bases[e]:.1f}" for e, _ in tasks))
    print(f"\n  {'λ':>8} │ " + " │ ".join(f"{e}←{'+'.join(str(x) for x in fs)}"
                                          .ljust(20) for e, fs in tasks)
          + " │  평균  양수")
    pooled = {}
    for lam in LAMS:
        cells, ds = [], []
        for e, fs in tasks:
            Xq = np.vstack([D[f]["X"] for f in fs])
            zq = np.concatenate([D[f]["z0"] for f in fs])
            yq = np.concatenate([D[f]["y"] for f in fs])
            sq = Xq.std(axis=0)
            b = fit_anchor(Xq / sq, zq, yq, lam, np.zeros(len(dep))) / sq
            pooled[(e, lam)] = b
            v = bss(D[e]["y"], shift_to_prior(
                1 / (1 + np.exp(-(D[e]["z0"] + D[e]["X"] @ b))), D[e]["y"].mean()))
            ds.append(v - bases[e])
            cells.append(f"{v:>8.1f} {v-bases[e]:>+6.1f}")
        print(f"  {lam:>8.4f} │ " + " │ ".join(f"{c:<20}" for c in cells)
              + f" │ {np.mean(ds):>+6.1f}  {sum(x > 0 for x in ds)}/3")

    # ── ⑥ 방향 검정을 3 fold에서 — 공통 t가 있는가 ──────────────────────
    print(f"\n{'='*78}\n■ ⑥ 방향 검정 (3 fold) — β(t) = β_dep + t·(β_pooled(과거) − β_dep)\n{'='*78}")
    print("  ①의 t*=+0.20은 fold2024를 **보고** 고른 값이라 전이 추정이 아니다.")
    print("  세 평가 fold에서 **같은 t가 전부 양수**여야 채택 후보가 된다.\n")
    ts2 = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    print(f"  {'t':>7} │ " + " │ ".join(f"{e:>8}" for e, _ in tasks) + " │  최소  양수")
    for t in ts2:
        ds = []
        for e, fs in tasks:
            b = dep + t * (pooled[(e, 0.0)] - dep)
            v = bss(D[e]["y"], shift_to_prior(
                1 / (1 + np.exp(-(D[e]["z0"] + D[e]["X"] @ b))), D[e]["y"].mean()))
            ds.append(v - bases[e])
        print(f"  {t:>+7.2f} │ " + " │ ".join(f"{d:>+8.1f}" for d in ds)
              + f" │ {min(ds):>+6.1f}  {sum(x > 0 for x in ds)}/3")


if __name__ == "__main__":
    main()
