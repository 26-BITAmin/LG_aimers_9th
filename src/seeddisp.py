"""emb_dim 사다리(§7.197 다음 수 ②) 판정에 필요한 **시드 산포**를 배포 블렌드에서 잰다.

검토 의견: "fold2023 값보다 시드 산포가 먼저다. 5시드면 Δ의 std 를 달라.
20 근처면 표 전체가 무효." — 그 20 은 §7.3 의 판정 규칙(단일 실행 차이 <20 BSS 면
채택 근거 불가)에서 왔는데, §7.3 은 **2019 시절 LGBM 단독 oracle 3시드**로 잰 값이다.
지금 스택은 MLP 8멤버 + LGBM 20 + CatBoost 12 를 평균한다 → 노이즈 바닥이 다르다.
값을 기다리기 전에 **우리 캐시로 예상 산포를 먼저 계산**한다.

핵심 단순화 (§7.197 층별 실측): dim 을 바꾼 sy02_c4 는 LGBM 4변형·CatBoost 2변형이
corr=1.00000000, `mlp_noid` 3개가 **npz 바이트 동일**이었다. 즉 사다리 셀 간에
새로 뽑히는 것은 **emb4 5시드뿐**이다. 그 sub-pool 만 흔들어 셀 점수의 산포를 잰다.

읽기 전용 — 캐시만 읽는다 (다른 세션과 안전 공존).
실행: .venv/Scripts/python.exe -u src/seeddisp.py 2>&1 | tee experiments/seeddisp.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import deploy29 as D                                            # noqa: E402
from config import ASOF_AXES, CTX_AXES, MLP_ALPHA, MLP_SEEDS    # noqa: E402
from validation import bss, shift_to_prior                      # noqa: E402

N_EMB, N_NOID = 5, 3          # MLP_VARIANTS: emb4 5시드 + noid 3시드
B = 4000


def fold_state(df, s, y, fold):
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    vi = np.where(s == fold)[0]
    with np.load(D.CACHE / f"dep29_fold{fold}_{D.KEY}.npz", allow_pickle=True) as z:
        Xax = z["Xax"]
    with np.load(D.CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
        M, yv_all = d0["members"], d0["y"]
    z_lgb = np.load(D.CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
    gt = df["game_type"].to_numpy()
    mm = (~((gt == "F") & (s < 2023)))[vi]
    return dict(M=M[:, mm], z_lgb=z_lgb[mm], yv=yv_all[mm], adj=(Xax @ dep)[mm])


def scorer(st):
    z_lgb, yv, adj = st["z_lgb"], st["yv"], st["adj"]
    r = float(yv.mean())

    def score(z_mlp):
        z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp + adj
        p = 1 / (1 + np.exp(-z))
        return bss(yv, shift_to_prior(np.clip(p, 1e-6, 1 - 1e-6), r))
    return score


def main():
    df, s, y = D.load_df()
    print(f"KEY={D.KEY}  α={MLP_ALPHA}  emb4 시드 {MLP_SEEDS[:N_EMB]}")
    out = {}
    for fold in (2023, 2024):
        st = fold_state(df, s, y, fold)
        score = scorer(st)
        M = st["M"]
        assert M.shape[0] == N_EMB + N_NOID, M.shape
        E, noid = M[:N_EMB], M[N_EMB:].mean(axis=0)

        # 배포 셀 = (5·emb4평균 + 3·noid평균)/8  — 가중은 배포 그대로 두고
        # emb4 sub-pool 만 다른 뽑기로 갈아끼운다.
        def cell(emb_pool):
            return score((N_EMB * emb_pool + N_NOID * noid) / (N_EMB + N_NOID))

        base = cell(E.mean(axis=0))
        print(f"\n=== fold {fold}  n={len(st['yv']):,}  기준 BSS {base:.2f} "
              f"(기대 {D.EXPECT_BASE[fold]}) ===")
        assert abs(base - D.EXPECT_BASE[fold]) < 0.5, "하네스 어긋남"

        # ① 단일 시드 셀 — k=1 의 draw 산포
        s1 = np.array([cell(E[i]) for i in range(N_EMB)])
        sd1 = s1.std(ddof=1)
        print("  ① 셀=emb4 1시드 (배포 가중 유지)")
        print("     " + "  ".join(f"{v:.2f}" for v in s1))
        print(f"     평균 {s1.mean():.2f}   sd {sd1:.2f}   범위 {np.ptp(s1):.2f}")

        # ② 5시드 셀의 SE — 잭나이프 + 부트스트랩(시드 위)
        loo = np.array([cell(np.delete(E, i, axis=0).mean(axis=0))
                        for i in range(N_EMB)])
        se_jack = np.sqrt((N_EMB - 1) / N_EMB * ((loo - loo.mean()) ** 2).sum())
        rng = np.random.default_rng(0)
        bs = np.array([cell(E[rng.integers(0, N_EMB, N_EMB)].mean(axis=0))
                       for _ in range(B)])
        print("  ② 셀=emb4 5시드 (배포 그대로) 의 뽑기 SE")
        print(f"     잭나이프 {se_jack:.2f}   부트스트랩(B={B}) {bs.std(ddof=1):.2f}"
              f"   1/√k 예측 {sd1 / np.sqrt(N_EMB):.2f}")

        # ③ 1/k 법칙 — 부분집합 sd (유한모집단 보정 후 무한모집단 환산)
        print("  ③ 1/√k 점검   k   sd(부분집합)   FPC보정   sd1/√k")
        for k in (1, 2, 3, 4):
            idx = [rng.choice(N_EMB, k, replace=False) for _ in range(200)]
            v = np.array([cell(E[i].mean(axis=0)) for i in idx])
            # 유한모집단(5개)에서 비복원 추출이라 관측 sd 는 √((N−k)/(N−1)) 만큼
            # 축소돼 있다. 무한모집단 환산은 그 역수를 **곱한다**.
            fpc = np.sqrt((N_EMB - 1) / (N_EMB - k))
            print(f"                {k}      {v.std(ddof=1):.2f}        "
                  f"{v.std(ddof=1) * fpc:.2f}"
                  f"       {sd1 / np.sqrt(k):.2f}")

        out[fold] = dict(sd1=sd1, se5=max(se_jack, bs.std(ddof=1)), base=base)

    print("\n=== dim 사다리 셀 비교에 붙는 잡음 (짝 없음: dim 이 바뀌면 RNG 스트림이 갈린다) ===")
    print(f"  {'fold':>6s}{'sd(1시드셀)':>14s}{'SE(5시드셀)':>14s}"
          f"{'SD(Δ) 1시드':>14s}{'SD(Δ) 5시드':>14s}")
    for f, o in out.items():
        print(f"  {f:>6d}{o['sd1']:>14.2f}{o['se5']:>14.2f}"
              f"{np.sqrt(2)*o['sd1']:>14.2f}{np.sqrt(2)*o['se5']:>14.2f}")
    print("\n  ※ noid 3멤버·LGBM·CatBoost 는 셀 간 고정으로 뒀다 (§7.197 실측: 바이트/기능 동일).")
    print("     sinyoung 이 전 층을 재학습했다면 실제 산포는 이보다 크다.")


if __name__ == "__main__":
    main()
