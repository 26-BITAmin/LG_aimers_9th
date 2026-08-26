"""예측 확률의 **꼬리를 0/1로 밀어붙이는** 후처리를 배포 층에서 잰다 (사용자 요청).

    "상위 1% → 1, 하위 1% → 0"

⚠️ 규정 — 왜 두 가지 버전을 재나
    test 예측 분포에서 분위를 뽑는 순간 **행 독립성이 깨진다** (CLAUDE.md §3.2:
    "분포 통계 ... 각 행은 완전히 독립적으로 예측되어야 한다"). 그래서
      팔 A  in-fold 분위  = 사용자가 말한 그대로. **제출 불가**지만 이 아이디어의 **상한**이다.
      팔 B  얼린 경계     = 직전 시즌 홀드아웃 p 분포에서 경계 2개를 뽑아 다음 fold에 적용.
                           `clipfit.py`가 피처에 쓰는 패턴 그대로. **제출 가능**.
    A가 지면 B는 볼 것도 없다 (B ⊂ A + 전이 손실).

무엇이 판정하나 — 손익분기는 산수로 닫힌다
    상위 q를 1로 밀면 그 빈의 평균 손실이  E[(p-y)²]  →  (1 - rate)  로 바뀐다.
    (rate = 그 빈의 실제 성공률). 즉 **이득이 나려면 rate > 1 - E[(p-y)²]** 여야 한다.
    p̄≈.85 인 빈에서 E[(p-y)²] ≈ .13 이면 rate가 **.87을 넘어야** 본전이다.
    → 빈별로 (실제 rate, 필요 rate)를 나란히 찍는다. 점수보다 이게 기전을 보여준다.

    같이 찍는 것: 그 빈을 **자기 실제 rate 상수로** 바꿨을 때의 이득(= 완전 재캘리브레이션
    헤드룸). 0/1 밀어붙이기가 지더라도 이게 크면 "꼬리가 틀렸다"는 진단은 살아 있다.

모델: s29 배포 층 (LGBM init_score + MLP α=0.75 + δ축 4개 + prior shift, clean 행).
      `initdeploy.py` 팔 B 와 같은 z 를 캐시에서 읽는다 — 학습 없음.
      F전문가(a=0.15)는 initdeploy 와 같은 이유로 빠져 있다 (절대값만 ~2.5 낮다).

실행:  .venv/Scripts/python.exe -u src/pclip.py 2>&1 | tee experiments/pclip.log
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

KEY = _fc._key()          # 배포 지문 (CENTER_ASOF=False)

import jointbeta as J                                              # noqa: E402
from config import (ASOF_AXES, CTX_AXES, MLP_ALPHA, TARGET_COL,    # noqa: E402
                    TRAIN_CSV)
from data import load_csv                                          # noqa: E402
from validation import brier, bss, shift_to_prior                  # noqa: E402

FOLDS = [2023, 2024]
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED = 42
QS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]      # 사용자 요청은 0.01
N_BOOT = 200


def deploy_p(df, s, y, fold):
    """s29 배포 층 홀드아웃 확률. (p, y, pitcher_code, clean_mask) 를 돌려준다."""
    Xax, _, _, _ = J.axis_matrix(df, s, fold)
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    vi = np.where(s == fold)[0]

    with np.load(CACHE / f"fold{fold}_{KEY}.npz") as d:
        z_mlp, yc = d["z_mlp"], d["y"]
    assert np.allclose(yc, y[vi]), "캐시 y 불일치 — 지문이 틀렸다"
    z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]    # ★ s29 팔 B

    z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp + Xax @ dep
    return z, vi


def clip_hard(p, lo_thr, hi_thr):
    """경계 **밖**을 0/1 로 민다. 행 단위 연산 — 행 간 정보 없음."""
    out = p.copy()
    out[p <= lo_thr] = 0.0
    out[p >= hi_thr] = 1.0
    return out


def murphy(y, p, h):
    """Murphy 분해.  BS = REL + WBV - RES + UNC  (항등식, 오차 0)

    격자 h로 예보값을 묶는다. 분위 빈이 아니라 **값 격자**를 쓰는 이유: 클리핑한
    예보는 0/1에 원자(atom)가 생겨 분위 빈이 무너진다. 격자는 두 예보에 같은
    규칙을 적용하고 원자를 자기 빈으로 보낸다.

    WBV = 빈 안 예보 분산. 이게 있어야 항등식이 **정확히** 닫힌다 (h→0에서 0).
    이 항을 빼먹고 REL을 보면 빈 폭에 따라 REL이 WBV를 흡수해 커 보인다.
    """
    k = np.round(np.asarray(p) / h).astype(np.int64)
    order = np.argsort(k, kind="stable")
    ks, ps, ys = k[order], np.asarray(p)[order], np.asarray(y)[order]
    cut = np.flatnonzero(np.diff(ks)) + 1
    n = np.diff(np.concatenate([[0], cut, [len(ks)]])).astype(np.float64)
    pbar = np.add.reduceat(ps, np.concatenate([[0], cut])) / n
    obar = np.add.reduceat(ys, np.concatenate([[0], cut])) / n
    p2 = np.add.reduceat(ps ** 2, np.concatenate([[0], cut])) / n
    w = n / n.sum()
    o = float(np.mean(y))
    REL = float(np.sum(w * (pbar - obar) ** 2))
    WBV = float(np.sum(w * (p2 - pbar ** 2)))
    RES = float(np.sum(w * (obar - o) ** 2))
    UNC = o * (1 - o)
    return REL, WBV, RES, UNC, len(n)


def murphy_rank(y, p, nbins=1000):
    """순위(등개수) 분할로 잰 Murphy 분해.

    왜 이게 필요한가 — 고정폭 격자 `murphy()` 는 **순증가 변환에 불변이 아니다.**
    p 를 넓게 벌리면 같은 행들이 더 많은 셀로 흩어져 분할이 세분되고, RES 가
    기계적으로 오른다. 그 상승은 해상도 획득이 아니라 **binning 아티팩트**다.

    등개수 분할은 순서만 보므로 z*T (T>0) 같은 **순증가** 변환에서 셀이 글자 그대로
    같다 → RES 가 **정확히** 불변이어야 한다. 이게 "값 변환은 RES 를 못 산다" 의
    조작적 형태다. 반대로 하드 클리핑은 순증가지만 **단사가 아니라** 셀을 병합하므로
    RES 가 내려간다 (Jensen). 두 경우를 같은 자로 갈라 보여준다.

    동점은 같은 셀에 묶는다 (클리핑이 만든 0/1 원자가 쪼개지면 안 된다).
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], y[order]
    N = len(ps)
    change = np.flatnonzero(np.diff(ps)) + 1          # 값이 바뀌는 위치만 절단 후보
    if len(change):
        tgt = np.arange(1, nbins) * (N / nbins)
        pick = np.clip(np.searchsorted(change, tgt), 0, len(change) - 1)
        cuts = np.unique(change[pick])
    else:
        cuts = np.empty(0, dtype=np.int64)
    starts = np.concatenate([[0], cuts]).astype(np.int64)
    n = np.diff(np.concatenate([starts, [N]])).astype(np.float64)
    pbar = np.add.reduceat(ps, starts) / n
    obar = np.add.reduceat(ys, starts) / n
    p2 = np.add.reduceat(ps ** 2, starts) / n
    w = n / N
    o = float(y.mean())
    return (float(np.sum(w * (pbar - obar) ** 2)),      # REL
            float(np.sum(w * (p2 - pbar ** 2))),        # WBV
            float(np.sum(w * (obar - o) ** 2)),         # RES
            o * (1 - o), len(n))


def murphy_rank_debias(y, p, nbins):
    """등개수 분할 + **노이즈 보정**.

    유한 셀에서 ō_k 는 표본추정치라 E[REL_hat] = REL_true + E[Var(ō_k)] 이고
    RES 도 **같은 양**만큼 부풀려진다 (그래서 REL−RES 는 불편이고 BSS 는 맞는다).
    각 항을 따로 읽으려면 그 노이즈를 빼야 한다:

        noise = Σ w_k · ō_k(1−ō_k)/(n_k − 1)      (Var(ō_k) 의 불편추정)

    REL_true 가 노이즈 바닥에 붙어 있으면 "캘리브레이션에 남은 몫이 없다" 는
    주장이 **측정으로** 뒷받침된다. 셀 수를 바꿔가며 바닥이 따라오는지 본다.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], y[order]
    N = len(ps)
    change = np.flatnonzero(np.diff(ps)) + 1
    tgt = np.arange(1, nbins) * (N / nbins)
    pick = np.clip(np.searchsorted(change, tgt), 0, max(len(change) - 1, 0))
    cuts = np.unique(change[pick]) if len(change) else np.empty(0, dtype=np.int64)
    starts = np.concatenate([[0], cuts]).astype(np.int64)
    n = np.diff(np.concatenate([starts, [N]])).astype(np.float64)
    pbar = np.add.reduceat(ps, starts) / n
    obar = np.add.reduceat(ys, starts) / n
    w = n / N
    o = float(y.mean())
    REL = float(np.sum(w * (pbar - obar) ** 2))
    RES = float(np.sum(w * (obar - o) ** 2))
    noise = float(np.sum(w * obar * (1 - obar) / np.maximum(n - 1, 1)))
    return REL, RES, noise, o * (1 - o), len(n)


def main():
    rng = np.random.default_rng(SEED)
    import pandas as pd

    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    codes, uniq = pd.factorize(df["pitcher_id"].to_numpy())
    n_id = len(uniq)
    mclean = ~((gt == "F") & (s < 2023))

    print(f"train {df.shape}   α={MLP_ALPHA}   지문 {KEY}   모델 = s29 (init_score)\n")

    P = {}
    for fold in FOLDS:
        z, vi = deploy_p(df, s, y, fold)
        mm = mclean[vi]
        yv = y[vi][mm]
        p = shift_to_prior(1 / (1 + np.exp(-z[mm])), yv.mean())
        P[fold] = (p, yv, codes[vi][mm])
        print(f"  fold {fold}: n={len(p):,}  BSS {bss(yv, p):8.1f}  "
              f"p∈[{p.min():.4f}, {p.max():.4f}]  Brier {brier(yv, p):.6f}")

    # ── 꼬리 진단 : 손익분기 ────────────────────────────────────────────
    print(f"\n{'='*100}\n█ 꼬리 진단 — 0/1 로 밀면 이득인가 (손익분기는 산수로 닫힌다)\n{'='*100}")
    print("  상위 q 를 1 로 밀면 그 빈 손실이 E[(p-y)²] → (1-rate).  **rate > 1-E[(p-y)²]** 여야 이득.")
    print("  하위 q 를 0 으로 밀면 손실이 rate 그 자체가 된다.       **rate < E[(p-y)²]** 여야 이득.\n")
    for fold in FOLDS:
        p, yv, _ = P[fold]
        print(f"  ── fold {fold} " + "─" * 84)
        print(f"  {'꼬리':<12}{'n':>8}{'평균 p':>10}{'실제 rate':>11}{'현재 손실':>11}"
              f"{'필요 rate':>11}{'여유':>9}{'0/1 ΔBSS':>11}{'재캘 ΔBSS':>11}")
        for q in QS:
            for side in ("상위", "하위"):
                thr = np.quantile(p, 1 - q if side == "상위" else q)
                m = (p >= thr) if side == "상위" else (p <= thr)
                if m.sum() == 0:
                    continue
                pb, yb = p[m], yv[m]
                cur = float(np.mean((pb - yb) ** 2))
                rate = float(yb.mean())
                need = (1 - cur) if side == "상위" else cur
                slack = (rate - need) if side == "상위" else (need - rate)

                pc = p.copy()
                pc[m] = 1.0 if side == "상위" else 0.0
                d01 = bss(yv, pc) - bss(yv, p)

                pr = p.copy()
                pr[m] = rate                       # 그 빈을 실제 rate 상수로 = 완전 재캘리브레이션
                drc = bss(yv, pr) - bss(yv, p)

                print(f"  {side} {q*100:>4.2f}%{m.sum():>10,}{pb.mean():>10.4f}{rate:>11.4f}"
                      f"{cur:>11.4f}{need:>11.4f}{slack:>+9.4f}{d01:>+11.1f}{drc:>+11.1f}")
        print()

    # ── Murphy 분해 : 손해가 REL 인가 RES 인가 ──────────────────────────
    print()
    print("=" * 100)
    print("█ Murphy 분해 — BS = REL + WBV − RES + UNC.   "
          "ΔBSS = −(100000/UNC)·(ΔREL + ΔWBV − ΔRES)")
    print("=" * 100)
    print("  UNC 는 y 만의 함수라 후처리로 안 변한다. 손해가 어느 항에서 오는지 두 개로 갈린다.")
    print("  '캘리브레이션은 멀쩡하니 sharpening 해도 된다' 가 맞으려면 **ΔREL ≈ 0** 이어야 한다.")
    print()
    for f in FOLDS:
        pv, yv, _ = P[f]
        lo, hi = np.quantile(pv, 0.01), np.quantile(pv, 0.99)
        pc = clip_hard(pv, lo, hi)
        print(f"  ── fold {f} " + "─" * 84)
        print(f"  {'격자 h':<10}{'예보':<12}{'빈수':>8}{'REL':>12}{'WBV':>12}{'RES':>12}"
              f"{'UNC':>12}{'BS(항등)':>12}{'오차':>10}")
        for h in (0.01, 0.002, 0.0005):
            row = {}
            for lab, pp in (("s29 원본", pv), ("q=1% 클립", pc)):
                R, W, S, U, nb = murphy(yv, pp, h)
                row[lab] = (R, W, S, U)
                ident = R + W - S + U
                print(f"  {h:<10.4f}{lab:<12}{nb:>8,}{R:>12.6f}{W:>12.6f}{S:>12.6f}"
                      f"{U:>12.6f}{ident:>12.6f}{ident - brier(yv, pp):>10.1e}")
            (R0, W0, S0, U0) = row["s29 원본"]
            (R1, W1, S1, _) = row["q=1% 클립"]
            c = 100000.0 / U0
            print(f"  {'':<10}{'Δ (BSS)':<12}{'':>8}{-c*(R1-R0):>+12.1f}{-c*(W1-W0):>+12.1f}"
                  f"{c*(S1-S0):>+12.1f}{0.0:>+12.1f}"
                  f"{-c*((R1+W1-S1)-(R0+W0-S0)):>+12.1f}")
            print()

    # ── 꼬리 구간의 REL 기여도 ──────────────────────────────────────────
    print("=" * 100)
    print("█ 꼬리 구간의 REL 기여도 — '꼬리가 안 맞아서' 클리핑할 이유가 있나 (클리핑 **전**)")
    print("=" * 100)
    print(f"  {'fold':<8}{'구간':<14}{'w':>9}{'p̄':>9}{'ō':>9}{'(p̄−ō)':>10}"
          f"{'REL 기여':>13}{'BSS 환산':>11}")
    for f in FOLDS:
        pv, yv, _ = P[f]
        o = yv.mean()
        U = o * (1 - o)
        lo, hi = np.quantile(pv, 0.01), np.quantile(pv, 0.99)
        for lab, m in (("하위 1%", pv <= lo), ("중간 98%", (pv > lo) & (pv < hi)),
                       ("상위 1%", pv >= hi)):
            w = m.mean()
            pb = pv[m].mean()
            ob = yv[m].mean()
            rel = w * (pb - ob) ** 2
            print(f"  {f:<8}{lab:<14}{w:>9.4f}{pb:>9.4f}{ob:>9.4f}{pb-ob:>+10.4f}"
                  f"{rel:>13.8f}{100000*rel/U:>11.2f}")
        print()

    # ── REL 에 남은 몫이 있나 : 노이즈 바닥과 대조 ──────────────────────
    print("=" * 100)
    print("█ REL 에 남은 몫이 있나 — 측정 REL 을 **셀 표본노이즈 바닥**과 대조")
    print("=" * 100)
    print("  E[REL_hat] = REL_true + E[Var(ō_k)].  RES 도 같은 양만큼 부풀려진다(차는 불편).")
    print("  REL_hat ≈ noise 면 REL_true ≈ 0 → 캘리브레이션 축에 남은 점수가 없다.")
    print()
    print(f"  {'fold':<8}{'셀수':>8}{'REL_hat':>12}{'노이즈바닥':>12}{'REL_true':>12}"
          f"{'RES_hat':>12}{'RES_true':>12}{'REL 몫(BSS)':>13}{'RES 몫(BSS)':>13}")
    for f in FOLDS:
        pv, yv, _ = P[f]
        for nb in (100, 1000, 5000):
            R, S, nz, U, cells = murphy_rank_debias(yv, pv, nb)
            c = 100000.0 / U
            print(f"  {f:<8}{cells:>8,}{R:>12.6f}{nz:>12.6f}{max(R-nz,0.0):>12.6f}"
                  f"{S:>12.6f}{S-nz:>12.6f}{c*max(R-nz,0.0):>13.1f}{c*(S-nz):>13.1f}")
        print()
    print("  읽는 법: 'REL 몫' = 캘리브레이션을 완벽히 고쳤을 때 벌 수 있는 BSS.")
    print("           'RES 몫' = 지금 점수 중 실제로 해상도가 벌고 있는 BSS.")
    print("  1 BSS = ΔRES 2.5e-06. 즉 +100 BSS 를 벌려면 RES 를 그만큼 늘려야 한다.")
    print()

    # ── 온도 스케일링 : **순증가** 변환은 RES 를 사줄 수 있나 ────────────
    print("=" * 100)
    print("█ 온도 스케일링 (순증가 sharpening) — 값 변환으로 RES 를 살 수 있나")
    print("=" * 100)
    print("  z*T 는 순서를 보존한다 → 등개수 분할(1000셀)이 글자 그대로 같다.")
    print("  RES 는 셀의 관측률 ō_k 만의 함수이므로 **정확히 불변**이어야 한다. 확인한다.")
    print("  (매번 fold prior 로 되돌려 레벨을 고정했다. 비교 대상은 sharpness 뿐이다.)")
    print()
    print(f"  {'fold':<8}{'T':>6}{'std(p)':>10}{'p 범위':>18}{'REL(순위)':>13}"
          f"{'RES(순위)':>13}{'RES 변화':>12}{'BSS':>11}{'Δ':>10}")
    for f in FOLDS:
        pv, yv, _ = P[f]
        z0 = np.log(pv / (1 - pv))
        base_bss = bss(yv, pv)
        res0 = None
        for T in (1.0, 1.25, 1.5, 2.0, 3.0):
            pt = shift_to_prior(1 / (1 + np.exp(-z0 * T)), yv.mean())
            R, W, S, U, _ = murphy_rank(yv, pt)
            if res0 is None:
                res0 = S
            v = bss(yv, pt)
            print(f"  {f:<8}{T:>6.2f}{pt.std():>10.4f}"
                  f"{f'[{pt.min():.3f}, {pt.max():.3f}]':>18}{R:>13.6f}"
                  f"{S:>13.6f}{S-res0:>12.2e}{v:>11.1f}{v - base_bss:>+10.1f}")
        print()

    # ── 같은 자로 하드 클리핑 : 순증가지만 **단사가 아니다** ──────────────
    print("=" * 100)
    print("█ 같은 등개수 분할로 본 하드 클리핑 — 여기서는 RES 가 내려간다 (셀 병합)")
    print("=" * 100)
    print(f"  {'fold':<8}{'예보':<14}{'셀수':>8}{'REL':>12}{'RES':>12}"
          f"{'ΔREL(BSS)':>12}{'ΔRES(BSS)':>12}{'Δ합':>10}")
    for f in FOLDS:
        pv, yv, _ = P[f]
        pc = clip_hard(pv, np.quantile(pv, 0.01), np.quantile(pv, 0.99))
        R0, W0, S0, U, nb0 = murphy_rank(yv, pv)
        R1, W1, S1, _, nb1 = murphy_rank(yv, pc)
        c = 100000.0 / U
        print(f"  {f:<8}{'s29 원본':<14}{nb0:>8,}{R0:>12.6f}{S0:>12.6f}"
              f"{'':>12}{'':>12}{'':>10}")
        print(f"  {f:<8}{'q=1% 클립':<14}{nb1:>8,}{R1:>12.6f}{S1:>12.6f}"
              f"{-c*(R1-R0):>+12.1f}{c*(S1-S0):>+12.1f}"
              f"{-c*((R1-S1)-(R0-S0)):>+10.1f}")
        print()

    # ── 팔 A : in-fold 분위 (사용자 요청 그대로, 제출 불가) ──────────────
    print(f"{'='*100}\n█ 팔 A — in-fold 분위로 양끝 클리핑 (요청 그대로 · **제출 불가**, 이 아이디어의 상한)\n{'='*100}")
    print(f"  {'q':<10}" + "".join(f"{f'fold {f} BSS':>16}{f'Δ':>10}" for f in FOLDS))
    base = {f: bss(P[f][1], P[f][0]) for f in FOLDS}
    print(f"  {'클리핑 없음':<10}" + "".join(f"{base[f]:>16.1f}{0.0:>+10.1f}" for f in FOLDS))
    for q in QS:
        cells = []
        for f in FOLDS:
            p, yv, _ = P[f]
            pc = clip_hard(p, np.quantile(p, q), np.quantile(p, 1 - q))
            v = bss(yv, pc)
            cells.append(f"{v:>16.1f}{v-base[f]:>+10.1f}")
        print(f"  {q*100:>4.2f}%{'':<5}" + "".join(cells))

    # ── 팔 B : 얼린 경계 (제출 가능) ────────────────────────────────────
    print(f"\n{'='*100}\n█ 팔 B — 경계를 fold2023 p 분포에서 얼려 fold2024 에 적용 (**제출 가능** 형태)\n{'='*100}")
    p23, y23, _ = P[2023]
    p24, y24, _ = P[2024]
    print(f"  {'q':<10}{'lo(2023)':>12}{'hi(2023)':>12}{'2024 적중률':>14}"
          f"{'fold2024 BSS':>16}{'Δ':>10}")
    for q in QS:
        lo, hi = np.quantile(p23, q), np.quantile(p23, 1 - q)
        hit = float(((p24 <= lo) | (p24 >= hi)).mean())
        pc = clip_hard(p24, lo, hi)
        v = bss(y24, pc)
        print(f"  {q*100:>4.2f}%{'':<5}{lo:>12.4f}{hi:>12.4f}{hit*100:>13.2f}%"
              f"{v:>16.1f}{v-base[2024]:>+10.1f}")

    # ── q=1% 양끝 클리핑의 오차막대 (투수 클러스터 부트스트랩) ───────────
    print(f"\n{'='*100}\n█ 요청값 q=1% (양끝) 클러스터 부트스트랩 — 투수 복원추출 {N_BOOT}회\n{'='*100}")
    print(f"  {'fold':<10}{'Δ BSS':>12}{'95% 구간':>26}")
    for f in FOLDS:
        p, yv, cf = P[f]
        lo, hi = np.quantile(p, 0.01), np.quantile(p, 0.99)
        pc = clip_hard(p, lo, hi)
        order = np.argsort(cf, kind="stable")
        bnd = np.searchsorted(cf[order], np.arange(n_id + 1))
        seen = np.where(np.diff(bnd) > 0)[0]
        bs = []
        for _ in range(N_BOOT):
            pick = rng.choice(seen, size=len(seen), replace=True)
            idx = np.concatenate([order[bnd[j]:bnd[j + 1]] for j in pick])
            yy = yv[idx]
            bs.append(bss(yy, pc[idx]) - bss(yy, p[idx]))
        l, h = np.percentile(bs, [2.5, 97.5])
        print(f"  {f:<10}{bss(yv, pc)-base[f]:>+12.1f}{f'[{l:+.1f}, {h:+.1f}]':>26}")


if __name__ == "__main__":
    main()
