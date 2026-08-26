"""클리핑을 **설계로 살릴 수 있는가** (사용자 질문). 세 갈래를 전부 건다.

§7.115 는 "상위/하위 1% → 1/0" 한 형태만 쟀다. 그 실패에는 두 원인이 있었다:

    ① REL 폭발 : 꼬리 실제율이 .616 인데 1.0 이라 부른다 → 비용 q·(1−ō)²·400000
    ② RES 잠김 : f(p) 는 같은 p 를 가진 행을 못 쪼갠다 → RES 를 살 방법이 없다

설계로 고칠 수 있다면 ①은 **덜 밀어서**, ②는 **x 를 보게 해서** 고쳐야 한다. 둘 다 건다.

  §1  f(p) 안에서 설계 개선  — winsorize · 부분 클리핑 λ 스윕 · prior shift 전/후 · 로짓공간
      예측: 전부 REL_true(+10~17 BSS)가 상한. 최적점은 λ=0(= 아무것도 안 함)일 것.
  §2  클래스 탈출            — 멤버 단위 클리핑(blend **전** z_lgb)은 최종 예보 입장에서
      f(z_lgb, z_mlp) 라 **2인자**다 → 셀을 쪼갤 수 있다. RES 가 실제로 움직이는지 본다.
  §3  ★ 핵심                — 손익분기(ō > 1 − E[(p−y)²] ≈ .764)를 넘는 **식별 가능한**
      부분모집단이 존재하는가. 존재하지 않으면 어떤 조건부 설계도 이길 수 없다.
      selector 를 train 에서만 적합해 val 에서 평가한다 (셀 라벨로 고르면 노이즈가
      혼자 .76 을 만든다 — n=20 이면 ō 의 sd 가 0.11 이다).

기준선: s29 배포 스택 (blend α=0.75 + δ 4축 + prior shift, clean 행). 학습 없음.

실행:  .venv/Scripts/python.exe -u src/clipdesign.py 2>&1 | tee experiments/clipdesign.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache as _fc

KEY = _fc._key()

import jointbeta as J                                             # noqa: E402
from config import (ASOF_AXES, CTX_AXES, MLP_ALPHA, TARGET_COL,
                    TRAIN_CSV)                                    # noqa: E402
from data import load_csv                                         # noqa: E402
from pclip import murphy_rank_debias                              # noqa: E402
from validation import brier, bss, shift_to_prior                 # noqa: E402

FOLDS = [2023, 2024]
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED = 42
Q = 0.01                      # 사용자 요청값


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def cell_code(d):
    """(count4, batter_hand) 상황코드. 8가지. 행 독립 — 각 행이 자기 값만 본다."""
    b = d["balls_before"].to_numpy(dtype=np.int64)
    s = d["strikes_before"].to_numpy(dtype=np.int64)
    c4 = (b >= 3).astype(np.int64) + 2 * (s >= 2).astype(np.int64)
    bh = d["batter_hand"].to_numpy(dtype=np.int64)
    return c4 * 2 + (bh == 2).astype(np.int64), 8


def main():
    rng = np.random.default_rng(SEED)
    import pandas as pd

    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    mclean = ~((gt == "F") & (s < 2023))
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])

    print(f"train {df.shape}   기준선 = s29 배포 스택   지문 {KEY}")
    print()

    D = {}
    for f in FOLDS:
        Xax, _, _, _ = J.axis_matrix(df, s, f)
        vi = np.where(s == f)[0]
        mm = mclean[vi]
        with np.load(CACHE / f"fold{f}_{KEY}.npz") as d:
            z_mlp, yc = d["z_mlp"], d["y"]
        assert np.allclose(yc, y[vi])
        z_lgb = np.load(CACHE / f"initdep_fold{f}_ctr0.npz")["z"]
        zl, zm, ax = z_lgb[mm], z_mlp[mm], (Xax @ dep)[mm]
        yv = y[vi][mm]
        p = shift_to_prior(_sig((1 - MLP_ALPHA) * zl + MLP_ALPHA * zm + ax), yv.mean())
        D[f] = dict(zl=zl, zm=zm, ax=ax, y=yv, p=p, vi=vi[mm])
        print(f"  fold {f}: n={len(p):,}  BSS {bss(yv, p):8.1f}  "
              f"p∈[{p.min():.4f}, {p.max():.4f}]")
    print()

    # ══ §1  f(p) 안에서 설계를 바꾼다 ═══════════════════════════════════
    print("=" * 104)
    print("█ §1  f(p) 안에서 설계 개선 — 덜 밀거나, 다른 곳에서 밀거나")
    print("=" * 104)
    print("  부분 클리핑:  상위 꼬리  p' = p + λ(1−p),  하위 꼬리  p' = p(1−λ)")
    print("  λ=1 이 §7.115 의 하드 클리핑. λ=0 이 무변경. winsorize 는 경계값으로 눌러앉히기.")
    print()
    print(f"  {'설계':<34}" + "".join(f"{f'fold{f} BSS':>14}{'Δ':>9}" for f in FOLDS))

    def report(lab, fn):
        cells = []
        for f in FOLDS:
            d = D[f]
            pp = fn(d)
            v = bss(d["y"], pp)
            cells.append(f"{v:>14.1f}{v - bss(d['y'], d['p']):>+9.1f}")
        print(f"  {lab:<34}" + "".join(cells))

    report("기준선 (무변경)", lambda d: d["p"])

    def partial(d, lam):
        p = d["p"].copy()
        lo, hi = np.quantile(p, Q), np.quantile(p, 1 - Q)
        up, dn = p >= hi, p <= lo
        p[up] = p[up] + lam * (1.0 - p[up])
        p[dn] = p[dn] * (1.0 - lam)
        return p

    for lam in (0.05, 0.1, 0.25, 0.5, 1.0):
        report(f"부분 클리핑 λ={lam}", lambda d, l=lam: partial(d, l))

    def winsor(d, q):
        p = d["p"]
        return np.clip(p, np.quantile(p, q), np.quantile(p, 1 - q))

    for q in (0.01, 0.05, 0.10):
        report(f"winsorize (경계값으로) q={q}", lambda d, qq=q: winsor(d, qq))

    def clip_before_shift(d):
        z = (1 - MLP_ALPHA) * d["zl"] + MLP_ALPHA * d["zm"] + d["ax"]
        p0 = _sig(z)
        lo, hi = np.quantile(p0, Q), np.quantile(p0, 1 - Q)
        p0 = np.where(p0 >= hi, 1.0 - 1e-9, np.where(p0 <= lo, 1e-9, p0))
        return shift_to_prior(p0, d["y"].mean())

    report("하드 클리핑 (prior shift **전**)", clip_before_shift)

    def clip_logit(d, c):
        z = _logit(d["p"])
        zc = np.clip(z, np.quantile(z, c), np.quantile(z, 1 - c))
        return shift_to_prior(_sig(zc), d["y"].mean())

    report("로짓 공간 winsorize q=0.01", lambda d: clip_logit(d, 0.01))
    print()
    print("  → 어느 설계든 f(p) 다. §7.115 가 잰 REL_true 상한(+10~17 BSS)을 못 넘는다.")
    print()

    # ══ §2  클래스 탈출 : 멤버 단위 (2인자가 된다) ══════════════════════
    print("=" * 104)
    print("█ §2  클래스 탈출 — blend **전** 멤버에 클리핑하면 최종 예보 입장에서 f(z_lgb, z_mlp)")
    print("=" * 104)
    print("  같은 최종 p 를 가진 두 행도 (z_lgb, z_mlp) 조합이 다를 수 있다 → 셀이 쪼개진다.")
    print("  RES 가 **움직일 수 있는** 설계다. 움직이는지, 그리고 이득인지는 별개 문제.")
    print()
    print(f"  {'설계':<34}{'fold':>6}{'RES_true':>11}{'ΔRES(BSS)':>12}"
          f"{'ΔREL(BSS)':>12}{'BSS':>10}{'Δ':>9}")
    for f in FOLDS:
        d = D[f]
        R0, S0, nz0, U, _ = murphy_rank_debias(d["y"], d["p"], 1000)
        c = 100000.0 / U
        base = bss(d["y"], d["p"])
        arms = {}
        for who in ("lgb", "mlp"):
            zz = d["zl"].copy() if who == "lgb" else d["zm"].copy()
            lo, hi = np.quantile(zz, Q), np.quantile(zz, 1 - Q)
            zz = np.where(zz >= hi, 20.0, np.where(zz <= lo, -20.0, zz))
            other = d["zm"] if who == "lgb" else d["zl"]
            wz = (1 - MLP_ALPHA) if who == "lgb" else MLP_ALPHA
            z = wz * zz + (1 - wz) * other + d["ax"]
            arms[f"멤버 클리핑 ({who}, blend 전)"] = shift_to_prior(_sig(z), d["y"].mean())
        for lab, pp in arms.items():
            R1, S1, nz1, _, _ = murphy_rank_debias(d["y"], pp, 1000)
            v = bss(d["y"], pp)
            print(f"  {lab:<34}{f:>6}{S1 - nz1:>11.6f}"
                  f"{c * ((S1 - nz1) - (S0 - nz0)):>+12.1f}"
                  f"{-c * (max(R1 - nz1, 0) - max(R0 - nz0, 0)):>+12.1f}"
                  f"{v:>10.1f}{v - base:>+9.1f}")
    print()

    # ══ §3  손익분기를 넘는 부분모집단이 존재하는가 ═════════════════════
    print("=" * 104)
    print("█ §3  ★ 조건부 클리핑이 가능하려면 — 손익분기를 넘는 **식별 가능한** 집단이 있어야")
    print("=" * 104)
    print("  selector 는 **train 에서만** 적합한다. val 라벨로 고르면 노이즈가 혼자 .76 을 만든다")
    print("  (n=20 셀이면 ō 의 sd 가 0.11). 각 selector 로 상위 1% 를 뽑아 val 실제율을 본다.")
    print()

    for f in FOLDS:
        d = D[f]
        yv, p = d["y"], d["p"]
        need_hi = 1.0 - float(np.mean((p[p >= np.quantile(p, 1 - Q)]
                                       - yv[p >= np.quantile(p, 1 - Q)]) ** 2))
        m = int(round(Q * len(p)))
        tr_m = (s < f) & mclean
        tr = df[tr_m]
        va = df.iloc[d["vi"]]

        # train 셀 성공률 (투수 × 상황8), EB 수축
        pid_tr = tr["pitcher_id"].to_numpy(dtype=np.int64)
        code_tr, M = cell_code(tr)
        y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
        key_tr = pid_tr * M + code_tr
        u, inv = np.unique(key_tr, return_inverse=True)
        n = np.bincount(inv).astype(np.float64)
        t = np.bincount(inv, weights=y_tr)
        mu = float(y_tr.mean())

        pid_va = va["pitcher_id"].to_numpy(dtype=np.int64)
        code_va, _ = cell_code(va)
        key_va = pid_va * M + code_va

        print(f"  ── fold {f}   손익분기 필요 ō > {need_hi:.4f}   상위 1% = {m:,}행 "
              + "─" * 22)
        print(f"  {'selector (train 적합)':<40}{'n':>9}{'val ō':>9}"
              f"{'필요치':>9}{'여유':>9}{'클립 ΔBSS':>12}")

        def show(lab, sel_val):
            idx = np.argsort(-sel_val, kind="stable")[:m]
            o = float(yv[idx].mean())
            pc = p.copy()
            pc[idx] = 1.0
            print(f"  {lab:<40}{len(idx):>9,}{o:>9.4f}{need_hi:>9.4f}"
                  f"{o - need_hi:>+9.4f}{bss(yv, pc) - bss(yv, p):>+12.1f}")

        show("A  배포 p (§7.115 와 동일)", p)
        for k in (0.0, 10.0, 50.0):
            r = (t + k * mu) / (n + k)
            j = np.clip(np.searchsorted(u, key_va), 0, len(u) - 1)
            hit = u[j] == key_va
            v = np.where(hit, r[j], mu)
            show(f"B  train 셀 성공률 (투수×상황8) k={k:g}", v)
        r = (t + 10.0 * mu) / (n + 10.0)
        j = np.clip(np.searchsorted(u, key_va), 0, len(u) - 1)
        cellv = np.where(u[j] == key_va, r[j], mu)
        show("C  p 와 셀 성공률의 곱순위", (p - p.mean()) / p.std()
             + (cellv - cellv.mean()) / cellv.std())
        show("D  대조 : 무작위 1%", rng.random(len(p)))
        print()

        # 최상위 셀은 어디까지 가나
        big = n >= 200
        r10 = (t + 10.0 * mu) / (n + 10.0)
        print(f"    train 셀(n≥200, {int(big.sum()):,}개) 성공률 최댓값 "
              f"{r10[big].max():.4f}   상위 0.1% 분위 {np.quantile(r10[big], 0.999):.4f}")
        hi_cells = big & (r10 > 0.60)
        print(f"    train 성공률 > .60 인 셀 {int(hi_cells.sum()):,}개 → "
              f"이 셀들에 속한 val 행의 실제율 ", end="")
        sel = np.isin(key_va, u[hi_cells])
        print(f"{yv[sel].mean() if sel.any() else float('nan'):.4f} (n={int(sel.sum()):,})")
        print()

    print("=" * 104)
    print("█ 규모 환산 — 언제쯤이면 상위 1% 클리핑이 손익분기에 닿나")
    print("=" * 104)
    for f in FOLDS:
        d = D[f]
        yv, p = d["y"], d["p"]
        R, S, nz, U, _ = murphy_rank_debias(yv, p, 1000)
        sd = np.sqrt(max(S - nz, 1e-12))
        mu = yv.mean()
        need = 1.0 - float(np.mean((p[p >= np.quantile(p, 1 - Q)]
                                    - yv[p >= np.quantile(p, 1 - Q)]) ** 2))
        cur99 = float(np.quantile(p, 0.99))
        zs = (cur99 - mu) / sd                       # 현재 99분위가 몇 sd 인가
        need_sd = (need - mu) / max(zs, 1e-9)        # 같은 분위가 need 에 닿는 sd
        need_res = need_sd ** 2
        print(f"  fold {f}: sd(π)={sd:.4f}  99분위 p={cur99:.4f} (= {zs:.2f} sd)  "
              f"필요 ō={need:.4f}")
        print(f"           → 같은 분위가 손익분기에 닿으려면 sd(π) {need_sd:.4f} "
              f"(현재의 {need_sd/sd:.1f}배), RES {need_res:.5f} (현재의 "
              f"{need_res/max(S-nz,1e-12):.0f}배)")
        print(f"           → 그때의 BSS ≈ {100000*need_res/U:,.0f}  "
              f"(현재 {bss(yv, p):.0f}, {100000*need_res/U/bss(yv, p):.1f}배)")


if __name__ == "__main__":
    main()
