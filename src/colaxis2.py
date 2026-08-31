"""§7.223b 생존 3축의 **0 근처 정밀 β** + 결합 (`src/colaxis2.py`).

§7.223(`colaxis.py`) 1차: β 격자가 ±0.01 이 최소여서 후보 전원의 β* 가 **격자의
0 쪽 끝**에 걸렸다. 고원 기하(G4)가 하나도 안 잡혔다 = 최적점이 |β| < 0.01 안에 있다.
여기서 0 근처를 20배 촘촘히 다시 판다.

살아남은 것만 본다 (1차에서 `prev3`·`prev5` 는 시즌 전이 23→24 가 음수 −3.98/−4.57,
f2024 Δ 가 널바닥 +2.06 아래 → 폐쇄):
    asof_pitcher_middle_rate · _sm   (같은 양의 raw/smooth — 한 축으로 취급)
    matchup_success_edge

추가로 묻는 것:
  ■ 결합 — `middle_rate_sm` 과 `matchup_success_edge` 는 엔티티가 다르다(투수 vs 매치업).
    가산 예측 대비 상호작용을 찍는다.
  ■ 널뱅크도 같은 정밀 격자로 다시 — 바닥이 격자 해상도에 따라 움직이는지 본다.
    (1차에서 f2023 바닥 +7.81 이 **드랍된 `game_month`** 였다)

실행: .venv/Scripts/python.exe -u src/colaxis2.py --on-grev 2>&1 | tee experiments/colaxis2.log
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

import asofdelta
import champfold as CF
import contam_fit
import features as F
import hillclimb as H
import modelabel
from colaxis import ANCHOR, ANCHOR_BETA, ANCHOR_EXPECT, std_axis
from config import (CONTAM_AXES, MLP_ALPHA, MLP_SEEDS, RECENCY_HALF_LIFE, SEED,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from gapstrike import EXTRA_SPEC
from grevaxis import PLATEAU_FRAC, plateau_beta
from validation import bss, shift_to_prior

ON_GREV = "--on-grev" in sys.argv
# --scale: 챔피언의 확률공간 아핀(PROB_CENTER .4801 · PROB_SCALE 1.05)을 얹는다.
#   §7.210 이 이 샤프닝만으로 홀드아웃 +2.5~4.1 을 쟀으므로 δ 축의 여지를 먹을 수
#   있다 — `gbmid.py --scale` 과 같은 스위치. 이걸 켜면 남는 층 격차는 라우팅 29파일뿐.
SCALE = "--scale" in sys.argv


def to_prob(praw, r):
    if SCALE:
        import champspec as CS
        return CS.prob_transform(praw, CS.solve_shift(praw, r))
    return shift_to_prior(praw, r)
FOLDS = (2023, 2024)
N_TRAIN_SUB = 400_000
B_BOOT = 1000

CANDS = ["asof_pitcher_middle_rate", "asof_pitcher_middle_rate_sm",
         "matchup_success_edge"]
NULLS = ["outs_before", "strikes_before", "num_runners_on", "is_first_pitch",
         "is_late_close", "top_bottom", "game_dayofweek", "game_month"]
PAIR = ("asof_pitcher_middle_rate_sm", "matchup_success_edge")

FINE = np.array([-0.0140, -0.0120, -0.0100, -0.0085, -0.0070, -0.0060, -0.0050,
                 -0.0040, -0.0030, -0.0025, -0.0020, -0.0015, -0.0010, -0.0005,
                 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0040, 0.0050,
                 0.0060, 0.0070, 0.0085, 0.0100, 0.0120, 0.0140])


def main():
    for k, v in EXTRA_SPEC.items():
        asofdelta.ASOF_SPEC.setdefault(k, v)
    print("§7.223b  생존 3축 0 근처 정밀 beta + 결합 (챔피언 등가 fold 층, 팔 D8C)"
          + ("  [기준선 = D8C + gap_reverse = **hj08 등가** (타자 emb1[0] 없음)]" if ON_GREV else "")
          + ("  [+ 챔피언 아핀 PROB_SCALE 1.05]" if SCALE else "  [shift_to_prior]"))
    print(f"  beta 격자 {len(FINE)}점  [{FINE.min():+.4f}, {FINE.max():+.4f}]  "
          f"1차 최소 스텝의 1/20 · B_BOOT {B_BOOT}\n")

    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y_all = df[TARGET_COL].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(SEED)
    curves, vs, ctxs = {}, {}, {}

    for fold in FOLDS:
        st, mm, yv, d, core, cat, noid, ax, tr, w = CF.fold_state(df, s, y_all, fold)
        r = float(yv.mean())
        e = [np.load(CF.CACHE / f"champ_D8C_f{fold}_emb4_{sd}.npz")["z"]
             for sd in MLP_SEEDS]
        zm = (0.875 * np.mean(e, axis=0) + 0.125 * np.mean(noid, axis=0))[mm]
        z0 = ((1 - H.W_CAT) * ((1 - MLP_ALPHA) * core + MLP_ALPHA * zm)
              + H.W_CAT * cat + ax)

        def sc(z):
            return bss(yv, to_prob(1 / (1 + np.exp(-z)), r))

        lab, _, _ = modelabel.recover(tr)
        rev = np.where(np.isnan(lab["reverse"]), 0.0, np.round(lab["reverse"]))
        ct = contam_fit.fit_contam(tr, rev)
        a_r = asofdelta.fit_asof_axes(tr, [(ANCHOR, 1.0)], CF.ASOF_S, ct=ct)[0]
        v_r = F.asof_lookup(d, a_r, ct=ct).astype(np.float64)
        # G0 은 **두 조건 모두** 잰다. ANCHOR_EXPECT 는 §7.216 의 shift_to_prior
        # 하네스 값이므로 그쪽으로 assert 하고, 챔피언 아핀 쪽은 **측정**으로 남긴다.
        # 두 값의 비 = 아핀이 δ 축의 여지를 얼마나 먹는가 (gbmid.py 가 남긴 미결).
        praw0 = 1 / (1 + np.exp(-z0))
        prawA = 1 / (1 + np.exp(-(z0 + ANCHOR_BETA * v_r)))
        d_ns = bss(yv, shift_to_prior(prawA, r)) - bss(yv, shift_to_prior(praw0, r))
        d_sc = sc(prawA * 0 + (z0 + ANCHOR_BETA * v_r)) - sc(z0)
        err = abs(d_ns - ANCHOR_EXPECT[fold]) / abs(ANCHOR_EXPECT[fold])
        print("=" * 118)
        print(f"fold {fold}   valid {len(yv):,}행")
        print(f"  G0 앵커 {ANCHOR} beta={ANCHOR_BETA:+.3f}")
        print(f"     shift_to_prior  D {d_ns:+.2f}  (§7.216 {ANCHOR_EXPECT[fold]:+.2f}, "
              f"오차 {err:.2%})  {'OK' if err < 0.02 else 'FAIL'}")
        print(f"     +PROB_SCALE1.05 D {d_sc:+.2f}   아핀이 먹는 몫 "
              f"{1 - d_sc / d_ns:+.1%}" if abs(d_ns) > 1e-9 else "")
        assert err < 0.02, "G0 실패 (shift_to_prior 기준)"
        if ON_GREV:
            z0 = z0 + ANCHOR_BETA * v_r
        base = sc(z0)
        print(f"기준선 {base:.2f}")
        print("=" * 118)

        wt = recency_weight(tr["season"].to_numpy(), RECENCY_HALF_LIFE,
                            ref_season=int(tr["season"].max()))
        art = F.fit_artifacts(tr, weights=wt, k_pitcher=SMOOTH_K_PITCHER,
                              k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)
        Xv = F.build_features(d, art)
        idx = np.sort(rng.choice(len(tr), size=min(N_TRAIN_SUB, len(tr)),
                                 replace=False))
        Xt = F.build_features(tr.iloc[idx], art)

        print(f"\n  {'축':<42}{'종류':>5}{'beta*':>10}{'D@beta*':>10}{'G4':>4}"
              f"   0 근처 곡선")
        for col in CANDS + NULLS:
            if col not in Xv.columns:
                continue
            t = np.asarray(Xt[col], dtype=np.float64)
            t = t[np.isfinite(t)]
            v = std_axis(np.asarray(Xv[col], dtype=np.float64),
                         float(t.mean()), float(t.std()))
            dd = np.array([sc(z0 + b * v) - base for b in FINE])
            bs = plateau_beta(FINE, dd, PLATEAU_FRAC)[0]
            g4 = bs is not None
            j = int(np.argmin(np.abs(FINE - bs))) if g4 else int(np.argmax(dd))
            curves.setdefault(col, {})[fold] = (FINE.copy(), dd, float(FINE[j]),
                                                float(dd[j]))
            if col in CANDS:
                vs.setdefault(fold, {})[col] = v
            head = "  ".join(f"{FINE[i]:+.4f}:{dd[i]:+.2f}"
                             for i in (2, 6, 9, 12, 15, 19, 25))
            print(f"  {col:<42}{'후보' if col in CANDS else '널':>5}"
                  f"{FINE[j]:>+10.4f}{dd[j]:>+10.2f}{'Y' if g4 else 'n':>4}   {head}")
        ctxs[fold] = (z0, base, yv, r, d, sc)

    nb = {f: max(curves[c][f][3] for c in NULLS if c in curves) for f in FOLDS}
    print(f"\n  널뱅크 max (정밀격자):  f2023 {nb[2023]:+.2f}   f2024 {nb[2024]:+.2f}")

    # ── 결합 + 부트스트랩 (fold2024 주 판단) ───────────────────────────
    z0, base, yv, r, d, sc = ctxs[2024]
    codes, uniq = pd.factorize(d["pitcher_id"].to_numpy())
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    seen = np.where(np.diff(bnd) > 0)[0]
    br = np.random.default_rng(SEED)
    idxs = [np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                            br.choice(seen, size=len(seen), replace=True)])
            for _ in range(B_BOOT)]
    p0 = to_prob(1 / (1 + np.exp(-z0)), r)

    def boot(dz, tag, pt):
        pa = to_prob(1 / (1 + np.exp(-(z0 + dz))), r)
        dif = np.array([bss(yv[i], pa[i]) - bss(yv[i], p0[i]) for i in idxs])
        lo, hi = np.percentile(dif, [2.5, 97.5])
        print(f"    {tag:<52}D {pt:+7.2f}  95% [{lo:+7.2f}, {hi:+7.2f}]  "
              f"P(D>0) {(dif > 0).mean():.3f}")
        return pt

    print(f"\n  결합·부트스트랩 (fold2024, 투수 클러스터, B={B_BOOT})")
    solo = {}
    for c in CANDS:
        b = curves[c][2024][2]
        solo[c] = boot(b * vs[2024][c], f"{c}  beta*={b:+.4f}", curves[c][2024][3])
    ca, cb = PAIR
    ba, bb = curves[ca][2024][2], curves[cb][2024][2]
    dz = ba * vs[2024][ca] + bb * vs[2024][cb]
    both = sc(z0 + dz) - base
    boot(dz, f"[결합] {ca.split('_')[-2]}_{ca.split('_')[-1]} + matchup", both)
    print(f"    가산 예측 {solo[ca] + solo[cb]:+.2f}  실측 {both:+.2f}  "
          f"상호작용 {both - solo[ca] - solo[cb]:+.2f}   "
          f"corr(v) {np.corrcoef(vs[2024][ca], vs[2024][cb])[0, 1]:+.3f}")

    print("\n  시즌 전이 (f2023 beta* -> f2024 채점, 그 반대)")
    for c in CANDS:
        b23, d23, s23, v23 = curves[c][2023]
        b24, d24, s24, v24 = curves[c][2024]
        f_ = float(d24[int(np.argmin(np.abs(b24 - s23)))])
        b_ = float(d23[int(np.argmin(np.abs(b23 - s24)))])
        print(f"    {c:<42}f23 {s23:+.4f}/{v23:+6.2f}   f24 {s24:+.4f}/{v24:+6.2f}"
              f"   23->24 {f_:+6.2f}   24->23 {b_:+6.2f}")


if __name__ == "__main__":
    main()
