"""§7.133 후속 5 — **투구 수가 적은 구간에서 예측이 좋은가**. 재학습 없음.

발단 (사용자)
    "신입들(결측치 없더라도 공 얼마 안 던진 선수들)에 대해서는 예측을 잘 하나?"

§7.79·§7.80 의 "신인"과 **다른 축이다.** 거기서는 커리어 **시즌 수**(0 = 신인)로
잘랐다. 여기서는 행 단위 `asof_pitcher_n`(그 투구 **이전까지** 누적 투구 수)으로
자른다. 확인: `asof_pitcher_n` 은 시즌을 넘어 단조증가하는 커리어 누적이다
(투수 21372: 2019 [0,1082] -> 2020 [1083,1432] -> ... -> 2024 [3952,4454]).
-> 저-n 행 = **최근 데뷔한 투수의 행**이고, 한 투수가 시즌 안에서 버킷을 넘어간다.

무엇을 재나 (버킷 x fold)
    ① 편향  p̄ - ȳ   <- Brier 에 직결된다. 배포 파이프라인의 `shift_to_prior` 는
                        **전역** 평균만 맞추므로 구간별 편향은 남을 수 있다.
    ② 로컬 BSS (그 버킷 자신의 기준율 대비)
    ③ §7.80 식 분해 — 원시 로컬 BSS 를 그대로 읽으면 **코호트 성질을 모델 성질로
       오독한다** ([[raw-transfer-confounds-level-with-structure]]). 그래서:
           Var_true(θ) = Var(ȳ_j) - E[ȳ_j(1-ȳ_j)/n_j]      (이항 노이즈 제거)
           투수간 상한 = 1e5 · Var_true / ref                (각 투수의 진짜 θ 를 알 때)
           포착률      = 실현 투수간 / 상한
       "고를 게 많았나"(상한)와 "잘 골랐나"(포착률)를 가른다.

⚠️ SE 는 **투수 클러스터 부트스트랩**으로 낸다. 이건 **그룹 비교**이지 쌍체가 아니다 —
   §7.133 후속 4 에서 확인했듯 쌍체면 클러스터가 거의 불필요하지만, 서로 다른 행
   집합을 맞대는 여기서는 클러스터가 지배한다
   ([[paired-comparisons-absorb-cluster-correlation]]).

⚠️ 이건 **진단**이다. 편향이 보여도 곧바로 구간별 보정으로 가지 않는다 — 세그먼트
   레벨 보정은 이 저장소에서 실패 전적이 있다(다른 세션 §7.123/§7.124 s31 -1.90).
   전적이 근거는 아니지만([[track-record-is-not-an-argument]]), 같은 함정을 다시
   밟지 않으려면 **전이 검정**이 먼저다.

실행:  .venv/Scripts/python.exe -u src/lowndiag.py 2>&1 | tee experiments/lowndiag.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

import jointbeta as J                                                # noqa: E402
from config import (ASOF_AXES, CTX_AXES, MLP_ALPHA, TARGET_COL, TRAIN_CSV)   # noqa: E402
from data import load_csv                                            # noqa: E402
from validation import shift_to_prior                                # noqa: E402

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
FOLDS = [2023, 2024]
SEED = 42
NBOOT = 400
EDGES = [0, 50, 200, 500, 1500, 4000, 10 ** 9]
MIN_ROWS = 20            # 투수간 분해에 쓸 최소 행 수 (버킷 안에서)


def label(i):
    lo, hi = EDGES[i], EDGES[i + 1]
    return f"[{lo:,}, {hi:,})" if hi < 10 ** 9 else f"[{lo:,}, ∞)"


def between_within(p, y, cf):
    """§7.80 식 분해. 투수 j 안의 평균으로 나눠 '고를 게 많았나 / 잘 골랐나'를 가른다.

    Var_true(θ) = Var(ȳ_j) - E[ȳ_j(1-ȳ_j)/n_j]   <- 이항 노이즈를 뺀 진짜 실력 분산
    잔차_true   = E[(ȳ_j-p̄_j)²] - E[ȳ_j(1-ȳ_j)/n_j]  <- 같은 노이즈를 잔차에서도 뺀다
    포착률      = 1 - 잔차_true / Var_true
    """
    df = pd.DataFrame({"p": p, "y": y, "c": cf})
    g = df.groupby("c").agg(pm=("p", "mean"), ym=("y", "mean"), n=("y", "size"))
    g = g[g["n"] >= MIN_ROWS]
    if len(g) < 10:
        return None
    ym, pm, n = g["ym"].to_numpy(), g["pm"].to_numpy(), g["n"].to_numpy(float)
    noise = float(np.mean(ym * (1 - ym) / n))
    var_true = float(np.var(ym, ddof=1)) - noise
    resid_true = float(np.mean((ym - pm) ** 2)) - noise
    return dict(J=len(g), sd_theta=np.sqrt(max(var_true, 0.0)),
                var_true=var_true, resid_true=resid_true,
                capture=(1 - resid_true / var_true) if var_true > 1e-9 else np.nan)


def main():
    rng = np.random.default_rng(SEED)
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    y_all = df[TARGET_COL].to_numpy(dtype=np.float64)
    n_all = df["asof_pitcher_n"].to_numpy(dtype=np.float64)
    codes, _ = pd.factorize(df["pitcher_id"].to_numpy())
    mclean = ~((gt == "F") & (s < 2023))
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    a = MLP_ALPHA
    print(f"a={a}  배포층(delta축 + prior shift, clean 행)  부트스트랩 {NBOOT}회")
    print(f"재학습 없음 — §7.133 base 캐시(현행 fold 스택)를 읽는다\n")

    for fold in FOLDS:
        Xax, _, _, _ = J.axis_matrix(df, s, fold)
        vi = np.where(s == fold)[0]
        mm = mclean[vi]
        yv = y_all[vi][mm]
        nv = n_all[vi][mm]
        cf = codes[vi][mm]
        with np.load(CACHE / f"histdepth_fold{fold}_base.npz") as d:
            z = ((1 - a) * d["z_lgb"] + a * d["z_mlp"])[mm] + (Xax @ dep)[mm]
        # 배포와 같은 전역 prior shift. **전역**이라는 점이 이 진단의 전제다.
        p = shift_to_prior(1 / (1 + np.exp(-z)), yv.mean())
        ref_all = yv.mean() * (1 - yv.mean())

        print(f"{'='*104}\nfold {fold}   행 {len(yv):,}  투수 {len(np.unique(cf))}"
              f"   전역 p̄-ȳ = {p.mean()-yv.mean():+.6f} (shift_to_prior 로 0 근처여야 정상)")
        print(f"{'='*104}")
        print(f"  {'asof_n 구간':<16}{'행':>8}{'투수':>6}{'ȳ':>8}{'p̄':>8}"
              f"{'편향':>9}{'95% 구간':>18}{'로컬BSS':>9}"
              f"{'투수':>6}{'sd(θ)':>8}{'상한':>8}{'포착률':>8}")

        # 투수 클러스터 부트스트랩 준비
        order = np.argsort(cf, kind="stable")
        n_id = cf.max() + 1
        bnd = np.searchsorted(cf[order], np.arange(n_id + 1))
        seen = np.where(np.diff(bnd) > 0)[0]
        boots = [np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                                 rng.choice(seen, size=len(seen), replace=True)])
                 for _ in range(NBOOT)]

        for i in range(len(EDGES) - 1):
            m = (nv >= EDGES[i]) & (nv < EDGES[i + 1])
            if m.sum() < 200:
                continue
            yb, pb, cb = yv[m], p[m], cf[m]
            bias = pb.mean() - yb.mean()
            refb = yb.mean() * (1 - yb.mean())
            lbss = 1e5 * (1 - ((pb - yb) ** 2).mean() / refb) if refb > 0 else np.nan
            bs = []
            for idx in boots:
                mi = m[idx]
                if mi.sum() < 50:
                    continue
                bs.append(p[idx][mi].mean() - yv[idx][mi].mean())
            lo, hi = np.percentile(bs, [2.5, 97.5])
            bw = between_within(pb, yb, cb)
            cap = f"{bw['capture']*100:7.1f}%" if bw and np.isfinite(bw["capture"]) else "      -"
            print(f"  {label(i):<16}{int(m.sum()):>8,}{len(np.unique(cb)):>6}"
                  f"{yb.mean():>8.4f}{pb.mean():>8.4f}{bias:>+9.4f}"
                  f"  [{lo:+.4f},{hi:+.4f}]{lbss:>9.0f}"
                  + (f"{bw['J']:>6}{bw['sd_theta']:>8.4f}"
                     f"{1e5*bw['var_true']/ref_all:>8.0f}{cap:>8}" if bw else f"{'-':>6}{'-':>8}{'-':>8}{'-':>8}"))
        print()

    print("읽기")
    print("  · 편향 구간이 0 을 포함하면 그 구간에 체계적 과대/과소예측이 없다는 뜻이다.")
    print("  · 로컬 BSS 는 버킷마다 기준율이 달라 **직접 비교하면 안 된다** — 상한·포착률로 읽는다.")
    print("  · 상한 = 그 구간 투수들의 진짜 실력 분산(고를 게 얼마나 많나).")
    print("    포착률 = 그중 모델이 실제로 가려낸 비율(얼마나 잘 고르나).")
    print("    상한이 낮아서 점수가 낮은 것과, 포착률이 낮아서 낮은 것은 처방이 다르다.")


if __name__ == "__main__":
    main()
