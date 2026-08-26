"""§7.162 2단계 — `tabm.py` 의 교란 둘을 닫는다.

1단계(`experiments/tabm.log`)가 T5 에 배포층 Δ +21.5 / +19.8 (양 fold 구간 0 배제)
을 줬다. 그런데 그 비교에 교란이 두 개 있다. 둘 다 **내 하네스 설계 탓**이다.

  ① 멤버 수 불일치
     T5 팔 = 3 draw × 5멤버 = **15멤버** 풀링.  CTRL = **5멤버**.
     §7.161(b) 장부(b=33.43)면 CTRL 5→15 는 b(1/5−1/15) = +4.5 뿐이라
     +21.5 의 21% 밖에 설명 못 하지만, **그건 외삽이다.**
     memory: "국소 기울기 × 옥타브는 깨진 외삽기" (§7.159 에서 17배 빗나갔다).
     → CTRL 을 실제로 20멤버까지 학습해 `BSS(k) = A − b/k` 를 **이 하네스에서**
       적합한다. 판정 열은 **점근값 A** 다:

           A(CTRL) < T5   →  가중치 공유가 **편향**을 줄인다. 멤버를 무한히
                             늘려도 독립 풀이 못 따라잡는다 → 진짜 후보
           A(CTRL) ≥ T5   →  전부 멤버 수 효과. emb4 확장이 더 싸다 → 종결

  ② 하네스 타당성 대조 누락
     CTRL 이 배포 emb4 풀을 재현하는지 확인하지 않았다. CTRL 이 약하면 "+21.5" 는
     TabM 이 아니라 내 버그를 잰 것이다 (memory: "간이 재구현은 알려진 수치 재현부터").
     → fold2023 의 `mem_emb4_f2023_{KEY}_{seed}.npz` (배포 캐시, aux **ON**) 와
       같은 시드끼리 로짓 상관을 본다. aux ON/OFF 차이만큼만 벌어져야 한다
       (§7.118-E 가 잰 축). corr < 0.99 면 코드가 어긋난 것이다.

⚠️ 1단계와 **같은 캐시**를 쓴다 (`tabm_f{fold}_ctrl_{seed}.npz` 등) — 재학습 없이
   이어붙는다. 새로 도는 것은 CTRL 의 6~20번째 멤버뿐이다.

실행 (GPU):
    MLP_DEVICE=cuda .venv-gpu/Scripts/python.exe -u src/tabm2.py 2>&1 | tee experiments/tabm2.log
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
from scipy.optimize import curve_fit

import tabm as T1
from config import (ASOF_AXES, CAT_ALPHA, CAT_SEEDS, CAT_VARIANTS, CTX_AXES,
                    MLP_ALPHA, MLP_SEEDS, TARGET_COL, TRAIN_CSV)
from data import load_csv
from predict import PROB_CENTER, PROB_SCALE
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
CACHE = T1.CACHE
W_E, W_N = T1.W_E, T1.W_N
N_MAX = 20                       # CTRL 을 여기까지 늘린다
KS = (1, 2, 3, 5, 8, 12, 16, 20)  # 곡선을 찍을 지점
# 1단계의 5개 + 새 15개. 앞 5개는 순서·값이 같아야 캐시가 맞는다.
SEEDS = tuple(MLP_SEEDS[:5]) + tuple(10_000 + i for i in range(N_MAX - 5))


def curve_point(zs, k, rng, reps=12):
    """k 멤버 부분집합을 reps 번 뽑아 평균. draw 노이즈를 평균해 없앤다."""
    if k >= len(zs):
        return [np.mean(zs, axis=0)]
    return [np.mean([zs[i] for i in rng.choice(len(zs), k, replace=False)], axis=0)
            for _ in range(reps)]


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    mclean = ~((gt == "F") & (s < 2023))
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    import jointbeta as J
    print(f"CTRL 을 {N_MAX} 멤버까지 · 시드 {SEEDS[:3]}...{SEEDS[-1]}\n")

    OUT = {}
    for fold in FOLDS:
        print(f"  -- fold {fold} --")
        inp = T1.fold_inputs(df, fold)
        Xax, _, _, _ = J.axis_matrix(df, s, fold)
        vi = np.where(s == fold)[0]
        mm, yv = mclean[vi], y[vi]
        ax = (Xax @ dep)[mm]
        yc = yv[mm]
        ybar = yc.mean()
        z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
        z_cat = np.mean([np.load(CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz")["z"]
                         for v in CAT_VARIANTS for sd in CAT_SEEDS], axis=0)
        z_noid = T1.noid_z(inp, fold, t0)

        def deployed(ze):
            """emb4 슬롯 로짓 -> 배포층 BSS."""
            zm = W_E * ze + W_N * z_noid
            z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * zm
            z = (1 - CAT_ALPHA) * z + CAT_ALPHA * z_cat
            p = shift_to_prior(1 / (1 + np.exp(-(z[mm] + ax))), ybar)
            p = shift_to_prior(PROB_CENTER + PROB_SCALE * (p - PROB_CENTER), ybar)
            return bss(yc, p)

        ctrl = [T1._train(inp, "emb4", sd, None, "ctrl", fold, t0) for sd in SEEDS]
        t5 = [T1._train(inp, "emb4", sd, dict(k=5), "k5", fold, t0) for sd in T1.DRAWS]
        t32 = [T1._train(inp, "emb4", sd, dict(k=32), "k32", fold, t0) for sd in T1.DRAWS]

        pts = {k: [deployed(z) for z in curve_point(ctrl, k, rng)] for k in KS}
        OUT[fold] = dict(pts=pts, t5=deployed(np.mean(t5, axis=0)),
                         t32=deployed(np.mean(t32, axis=0)),
                         t5_1=[deployed(z) for z in t5],
                         ctrl_raw=ctrl)
        print(f"     CTRL k=5 {np.mean(pts[5]):.1f}  k={N_MAX} {np.mean(pts[N_MAX]):.1f}"
              f"   T5(15멤버) {OUT[fold]['t5']:.1f}   T32(96멤버) {OUT[fold]['t32']:.1f}")

    bar = "=" * 86
    print(f"\n{bar}\n[교란 ①] CTRL 확장 곡선  BSS(k) = A - b/k   판정 열은 **A**\n{bar}")
    print(f"  {'fold':>6}" + "".join(f"{f'k={k}':>9}" for k in KS)
          + f"{'A(k→∞)':>10}{'b':>8}{'잔차':>7}")
    for fold in FOLDS:
        pts = OUT[fold]["pts"]
        xs = np.array(KS, dtype=float)
        ys = np.array([np.mean(pts[k]) for k in KS])
        (A, b), _ = curve_fit(lambda k, A, b: A - b / k, xs, ys, p0=(ys[-1], 30.0))
        res = float(np.sqrt(np.mean((ys - (A - b / xs)) ** 2)))
        OUT[fold]["A"], OUT[fold]["b"] = A, b
        print(f"  {fold:>6}" + "".join(f"{v:>9.1f}" for v in ys)
              + f"{A:>10.1f}{b:>8.1f}{res:>7.2f}")

    print(f"\n{bar}\n[교란 ①] 판정 — 독립 풀의 **점근값**이 TabM 을 따라잡는가\n{bar}")
    print(f"  {'fold':>6}{'A(CTRL,k→∞)':>14}{'T5':>10}{'T32':>10}"
          f"{'A − T5':>10}{'판정':>8}")
    for fold in FOLDS:
        A, t5, t32 = OUT[fold]["A"], OUT[fold]["t5"], OUT[fold]["t32"]
        v = "공유O" if A < t5 else "멤버수"
        print(f"  {fold:>6}{A:>14.1f}{t5:>10.1f}{t32:>10.1f}{A - t5:>10.1f}{v:>8}")
    print("\n  A < T5  -> 멤버를 무한히 늘려도 독립 풀이 못 따라잡는다 = 편향 이득")
    print("  A >= T5 -> 전부 멤버 수 효과. emb4 확장이 싸다 -> 종결")

    print(f"\n{bar}\n[교란 ①b] 비용 정렬 — 학습 **런** 수를 맞추면\n{bar}")
    print(f"  {'fold':>6}{'CTRL 3런(3멤버)':>18}{'T5 3런(15멤버)':>17}"
          f"{'CTRL 5런':>11}{'T5 1런(5멤버)':>15}")
    for fold in FOLDS:
        p = OUT[fold]["pts"]
        print(f"  {fold:>6}{np.mean(p[3]):>18.1f}{OUT[fold]['t5']:>17.1f}"
              f"{np.mean(p[5]):>11.1f}{np.mean(OUT[fold]['t5_1']):>15.1f}")

    print(f"\n{bar}\n[교란 ②] 하네스 타당성 — CTRL 이 배포 emb4 캐시를 재현하는가\n{bar}")
    print("  fold2023 만 가능하다 (fold2024 는 멤버별 배포 캐시가 없다).")
    print("  배포 캐시는 aux **ON**, CTRL 은 aux OFF — 그 차이만큼만 벌어져야 한다.")
    ok = True
    for sd in MLP_SEEDS[:5]:
        g = sorted(CACHE.glob(f"mem_emb4_f2023_*_{sd}.npz"))
        if not g:
            print(f"     seed {sd}: 배포 캐시 없음")
            continue
        zd = np.load(g[0])["z"]
        zc = OUT[2023]["ctrl_raw"][list(SEEDS).index(sd)]
        c = float(np.corrcoef(zd, zc)[0, 1])
        ok &= c > 0.99
        print(f"     seed {sd:>6}: 로짓 상관 {c:.5f}   sd(배포) {zd.std():.4f}"
              f"  sd(CTRL) {zc.std():.4f}   {'OK' if c > 0.99 else '!! 어긋남'}")
    print(f"     -> {'통과. CTRL 은 배포 emb4 다.' if ok else '실패 — 위 표를 읽지 말 것.'}")
    print(f"\n  총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
