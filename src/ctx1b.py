"""§7.209 — `runner_on_1b`(세트포지션) ctx δ 축을 **챔피언 등가 fold 층**에서 재측정.

§7.37 판정: T1 corr .111/.037 · 잔차 t 0.6/0.7 · **f2023 +0.20 / f2024 +0.20**
· **크기비 1.03** · 대조군 ~0 → 조건 ②③ 통과, **조건 ①(효과 < 시드 노이즈 ±3.3)로 기각**.

★ 그 조건 ①이 §7.201 이 철회한 틀이다 ("잡음보다 작으면 채택 근거 없음").
★ 그리고 크기비 1.03 은 그 표에서 **가장 좋다** — 채택된 `count4` 가 1.30 이었다.
  크기비는 fold 간 전이의 대리변수이므로, 작지만 가장 잘 옮겨갈 후보였다.

이번에 다른 것
    ① 층: §7.208 챔피언 등가 fold (emb_dim 8) — §7.37 은 s11 시절 층이었다
    ② **β 적합**: 고원 기하로 고른다 (`config.py` 규칙 — 허용 구간 최대값 한 칸 아래)
    ③ **α 스윕**: MLP 블렌딩 비율을 축과 **함께** 다시 찾는다 (사용자 지시)
    ④ 위약: 투수 재배정 순열 · 투수 클러스터 부트스트랩

δ 표는 배포와 같은 방식으로 적합한다 — `ctxfit.fit_ctx` + recency 가중 (§7.61).

실행: .venv/Scripts/python.exe -u src/ctx1b.py 2>&1 | tee experiments/ctx1b.log
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

import ctxfit
import deploy29 as D
import features as F
import hillclimb as H
from config import (CTX_K, MLP_ALPHA, MLP_SEEDS, RECENCY_HALF_LIFE, TARGET_COL,
                    TRAIN_CSV)
from data import load_csv, recency_weight
from validation import bss, shift_to_prior

CACHE = D.CACHE
ARM = {2023: "D4", 2024: "D8"}          # §7.208 각 fold 최선 팔
BETAS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.875, 2.25, 2.75, 3.5]
ALPHAS = [0.65, 0.70, 0.75, 0.80, 0.85]
B_BOOT = 600


def _code_set1b(df):
    """세트포지션 — 1루 주자 유무. m=2 (§7.37 `ctx_runner1b` 와 같은 정의)."""
    v = pd.to_numeric(df["runner_on_1b"], errors="coerce").fillna(0)
    return (v.to_numpy() > 0).astype(np.int64), 2


def main():
    F.CTX_CODERS.setdefault("set1b", _code_set1b)
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y_all = df[TARGET_COL].to_numpy(dtype=np.float64)
    out = {}

    for fold in (2023, 2024):
        st = D.state(df, s, y_all, fold, check=False)
        mm, ax, yv = st["mm"], st["Xax"] @ st["dep"], st["yv"]
        r = float(yv.mean())
        zl = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"][mm]
        cat = np.mean([np.load(CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz")["z"]
                       for v in H.CAT_VARIANTS for sd in H.CAT_SEEDS], axis=0)[mm]
        e = [np.load(CACHE / f"champ_{ARM[fold]}_f{fold}_emb4_{sd}.npz")["z"]
             for sd in MLP_SEEDS]
        noid = np.load(CACHE / f"champ_noid_f{fold}.npz")["z"]
        zm = (0.875 * np.mean(e, axis=0) + 0.125 * np.mean(noid, axis=0))[mm]

        def blend(alpha):
            return ((1 - H.W_CAT) * ((1 - alpha) * zl + alpha * zm)
                    + H.W_CAT * cat + ax)

        def sc(z):
            return bss(yv, shift_to_prior(1 / (1 + np.exp(-z)), r))

        # ── δ 표: 배포와 같은 적합 (train<fold · recency 가중)
        tr = df[s < fold]
        w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                           ref_season=int(tr.season.max()))
        axes = ctxfit.fit_ctx(tr, TARGET_COL, [("set1b", 1.0)], CTX_K, weights=w)
        d = df.iloc[st["vi"][mm]].copy().reset_index(drop=True)
        dlt = F.ctx_lookup(d, axes[0]).astype(np.float64)

        rng = np.random.default_rng(0)
        dsh = d.copy()
        dsh["pitcher_id"] = rng.permutation(d["pitcher_id"].to_numpy())
        dsh_delta = F.ctx_lookup(dsh, axes[0]).astype(np.float64)

        print("=" * 96)
        print(f"fold {fold}  팔 {ARM[fold]}  행 {len(yv):,}   "
              f"δ 비영 {np.mean(dlt != 0):.1%}  sd(δ) {dlt.std():.5f}")
        print("=" * 96)

        # ── β × α 격자
        print(f"  {'α\β':<7}" + "".join(f"{b:>8.3g}" for b in BETAS) + f"{'기준':>9}")
        grid = {}
        for a in ALPHAS:
            z0 = blend(a)
            b0 = sc(z0)
            row = [sc(z0 + b * dlt) - b0 for b in BETAS]
            grid[a] = (b0, row)
            print(f"  {a:<7.2f}" + "".join(f"{v:>+8.2f}" for v in row) + f"{b0:>9.2f}")

        # ── 배포 α 에서의 위약·구간
        z0 = blend(MLP_ALPHA)
        b0 = sc(z0)
        row = grid[MLP_ALPHA][1]
        i = int(np.argmax(row))
        bstar, dv = BETAS[i], row[i]
        pl = sc(z0 + bstar * dsh_delta) - b0
        codes, uniq = pd.factorize(d["pitcher_id"].to_numpy())
        order = np.argsort(codes, kind="stable")
        bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
        seen = np.where(np.diff(bnd) > 0)[0]
        br = np.random.default_rng(7)
        zc = z0 + bstar * dlt
        bs = []
        for _ in range(B_BOOT):
            j = np.concatenate([order[bnd[k]:bnd[k + 1]] for k in
                                br.choice(seen, size=len(seen), replace=True)])
            bs.append(bss(yv[j], shift_to_prior(1 / (1 + np.exp(-zc[j])), r))
                      - bss(yv[j], shift_to_prior(1 / (1 + np.exp(-z0[j])), r)))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"\n  배포 α={MLP_ALPHA}: β*={bstar}  Δ={dv:+.2f}  "
              f"95% [{lo:+.2f}, {hi:+.2f}]  P(Δ>0)={np.mean(np.array(bs)>0):.3f}"
              f"  위약 {pl:+.2f}")
        out[fold] = dict(row=row, best=(bstar, dv), ci=(lo, hi), plac=pl, grid=grid)
        print()

    print("=" * 96)
    print("판정")
    print("=" * 96)
    a, b = out[2023], out[2024]
    print(f"  {'β':>8}{'f2023':>9}{'f2024':>9}{'크기비':>9}  (배포 α)")
    for i, bb in enumerate(BETAS):
        x, y_ = a["row"][i], b["row"][i]
        rr = x / y_ if y_ else float("nan")
        print(f"  {bb:>8.3g}{x:>+9.2f}{y_:>+9.2f}{rr:>9.2f}")
    print(f"\n  f2023 β*={a['best'][0]} Δ{a['best'][1]:+.2f} "
          f"[{a['ci'][0]:+.2f},{a['ci'][1]:+.2f}] 위약{a['plac']:+.2f}")
    print(f"  f2024 β*={b['best'][0]} Δ{b['best'][1]:+.2f} "
          f"[{b['ci'][0]:+.2f},{b['ci'][1]:+.2f}] 위약{b['plac']:+.2f}")
    print("\n  ⚠️ β 는 argmax 가 아니라 **고원 기하**로 고른다 (config.py 규칙).")
    print("     α 격자는 축과 α 의 상호작용을 보려는 것 — α 를 fold 로 다시 고르지 않는다.")


if __name__ == "__main__":
    main()
