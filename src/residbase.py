"""잔차 t의 **기준선을 층별로 쪼갠다** (사용자 질문: "잔차가 뭐에 대한 잔차인가").

§7.90은 `b_j`(투수 절편 EB)를 **배포 전체**의 잔차에 물었고 t가 죽었다.
그런데 기준선이 뭐냐에 따라 결론의 문장이 달라진다:

    블록만(z0)에서도 죽음   →  "모델의 **피처**가 이미 안다"      (b_j는 정보가 아님)
    z0에서는 살고 배포에서 죽음 →  "**보정축이 이미 먹었다**"        (b_j는 정보인데 중복)

두 진술은 다르다. 후자면 "축을 b_j로 바꾸면?"이 열리고, 전자면 닫힌다.

기준선 4층 (전부 홀드아웃. fold f의 모델은 <f 로만 학습)
    lgb        LGBM 단독
    blend      0.6·lgb + 0.4·mlp                      ← 사후 보정 없음
    +ctx       블렌드 + δ_hand + δ_count4
    deploy     + gap_success + gap_bsuccess           ← §7.90이 쓴 것

프로브 3종 (같은 코드로 재서 서로 비교 가능하게)
    b_j            투수 절편 EB (λ = 적률법)
    gap_success    ★ 하네스 검증용 — §7.54가 블렌드 기준 +16.05/+8.02를 보고했다.
                     재현되면 이 측정 코드를 믿을 수 있다.
    gap_bsuccess   타자판
    shuf(b_j)      대조군

⚠️ 잔차는 **확률 스케일**(y − p)이고 `resid_t`는 OLS다. 이 데이터는 p가 .4~.6이라
   p(1−p)이 거의 상수(.24~.25)여서 이분산 보정 없이도 SE가 거의 맞다.
   확률 구간이 넓은 문제였다면 가중을 넣어야 한다.

실행:  .venv/Scripts/python.exe -u src/residbase.py 2>&1 | tee experiments/residbase.log
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

import jointbeta as J
from ctxsplit import resid_t
from ebshrink import lam_eb
from config import ASOF_AXES, CTX_AXES, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = [2021, 2022, 2023, 2024]
SEED = 42


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y_all = df[TARGET_COL].to_numpy(dtype=np.float64)
    pid_all = df["pitcher_id"].to_numpy(dtype=np.int64)
    cached = J.foldcache.load(FOLDS, verbose=False)
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    n_ctx = len(CTX_AXES)
    rng = np.random.default_rng(SEED)

    for f in FOLDS:
        X, names, _, va = J.axis_matrix(df, s, f)
        yv = cached[f][2]
        m = va["game_type"].to_numpy() != "F"
        X, yv, va = X[m], yv[m], va[m]
        zl, zm = cached[f][0][m], cached[f][1][m]

        # 기준선 4층 — 전부 같은 prior shift 를 거쳐 확률로 만든다
        b_ctx = dep.copy(); b_ctx[n_ctx:] = 0.0
        layers = {
            "lgb": zl,
            "blend": (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm,
            "+ctx": (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm + X @ b_ctx,
            "deploy": (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm + X @ dep,
        }
        P = {k: shift_to_prior(1 / (1 + np.exp(-z)), yv.mean())
             for k, z in layers.items()}

        # b_j — train(<f)에서만
        tr_m = s < f
        lam = lam_eb(pid_all[tr_m], y_all[tr_m], 30)[0]
        g = pd.DataFrame({"p": pid_all[tr_m], "y": y_all[tr_m]}).groupby("p")["y"] \
            .agg(["mean", "count"])
        ybar = float(y_all[tr_m].mean())
        vp = va["pitcher_id"].to_numpy(dtype=np.int64)
        gm = g["mean"].reindex(vp).fillna(ybar).to_numpy()
        gn = g["count"].reindex(vp).fillna(0.0).to_numpy()
        bj = (gn / (gn + lam)) * (gm - ybar)
        perm = rng.permutation(len(vp))
        bj_s = (gn[perm] / (gn[perm] + lam)) * (gm[perm] - ybar)

        probes = {"b_j (EB)": bj, "shuf(b_j)": bj_s,
                  "gap_success": X[:, n_ctx], "gap_bsuccess": X[:, n_ctx + 1]}

        print(f"\n{'='*84}")
        print(f"■ fold {f}   R n={len(yv):,}   λ_EB={lam:.1f}")
        print(f"{'='*84}")
        print(f"  {'기준선':<10}" + "".join(f"{k:>9}" for k in ("BSS",))
              + "".join(f"{k:>15}" for k in probes))
        for k, p in P.items():
            print(f"  {k:<10}{bss(yv, p):>9.1f}"
                  + "".join(f"{resid_t(yv - p, v)[1]:>15.2f}" for v in probes.values()))

    print(f"\n{'='*84}\n■ 읽는 법\n{'='*84}")
    print("  `gap_success` 열의 blend 행 = §7.54가 보고한 +16.05 / +8.02 와 대조된다.")
    print("     → 재현되면 이 측정 코드가 옳다는 증거 (하네스 검증).")
    print("     → deploy 행에서 거의 0으로 떨어져야 한다 (이미 배포된 축이니 소진됨).")
    print("  `b_j` 열이 **모든 층에서** 작으면 → 모델 피처가 이미 안다. 채널 닫힘.")
    print("  `b_j` 열이 blend에서 크고 deploy에서 작으면 → 보정축이 먹은 것. 다른 얘기.")
    print("  `shuf(b_j)` 는 어느 층에서도 작아야 한다 (아니면 측정 자체가 틀린 것).")


if __name__ == "__main__":
    main()
