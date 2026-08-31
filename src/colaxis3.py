"""§7.226 `asof_pitcher_success_rate` — **`hj15` 위의 증분**으로 (`src/colaxis3.py`).

base: `hj15_pmid.zip` (현 챔피언, LB 1144.9773230404)
측정층: `champfold.py` fold 재학습 층 + `gap_reverse` −0.017 + **`col` 축
        `asof_pitcher_middle_rate_sm` β −0.0070** = **hj15 등가**
        (타자 `emb1[0]` 콜드패치 · 라우팅 11% 는 여전히 없다 — §7.223d ④)
확률변환: **`--scale` 기본 ON** (§7.223d — 끄면 축 선별이 뒤집힌다)

■ 왜 증분으로 묻나
`success = 1 − (몰림 + 반대 + 벗어남)` 이라 `success_rate` 는 방금 배포한
`middle_rate_sm` 과 **구조적으로 겹친다.** 단독 Δ 를 보면 이미 사고 있는 것을
두 번 세게 된다. §7.57 이 `gap_bsuccess` 를 `gap_success` **위의 증분**으로 다시 잰
그 규약을 그대로 쓴다. [[placebo-pass-doesnt-mean-incremental]]

■ 후보
    asof_pitcher_success_rate(_sm)   §7.91 잔차 t +2.8 / +4.2 — C군 중 유일 미측정
    asof_pitcher_middle_rate (raw)   배포된 `_sm` 위의 증분 (갈아타기가 아니라 증분)
    asof_pitcher_reverse_rate(_sm)   덤 — 같은 분해의 나머지 축, 한 번도 안 물었다

■ 관문 (§7.223 과 동일)
    G0  앵커 `gap_reverse` β=−0.017 이 §7.216 의 +30.61 / +4.75 를 재현 (shift_to_prior 기준)
    G0b **배포 축 재현** — `middle_rate_sm` β −0.0070 이 §7.223 의 f2024 +2.94 를 내는가
    G1  양 fold 부호 일치 · G2 널뱅크(§7.91 죽은 축 8개) 초과
    G3  투수 클러스터 부트스트랩 · G4 고원 기하 · G5 시즌 전이 양방향 양수

실행: .venv/Scripts/python.exe -u src/colaxis3.py 2>&1 | tee experiments/colaxis3.log
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
import champspec as CS
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

NO_SCALE = "--no-scale" in sys.argv          # 기본은 아핀 ON
FOLDS = (2023, 2024)
N_TRAIN_SUB = 400_000
B_BOOT = 1000

DEPLOYED = ("asof_pitcher_middle_rate_sm", -0.0070)   # hj15 가 들고 있는 col 축
DEPLOY_EXPECT = {2023: None, 2024: +2.94}             # §7.223d 표 (아핀 ON)

CANDS = ["asof_pitcher_success_rate", "asof_pitcher_success_rate_sm",
         "asof_pitcher_middle_rate",
         "asof_pitcher_reverse_rate", "asof_pitcher_reverse_rate_sm"]
NULLS = ["outs_before", "strikes_before", "num_runners_on", "is_first_pitch",
         "is_late_close", "top_bottom", "game_dayofweek", "game_month"]

FINE = np.array([-0.0140, -0.0120, -0.0100, -0.0085, -0.0070, -0.0060, -0.0050,
                 -0.0040, -0.0030, -0.0025, -0.0020, -0.0015, -0.0010, -0.0005,
                 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0040, 0.0050,
                 0.0060, 0.0070, 0.0085, 0.0100, 0.0120, 0.0140])


def to_prob(praw, r):
    if NO_SCALE:
        return shift_to_prior(praw, r)
    return CS.prob_transform(praw, CS.solve_shift(praw, r))


def main():
    for k, v in EXTRA_SPEC.items():
        asofdelta.ASOF_SPEC.setdefault(k, v)
    print("§7.226  `asof_pitcher_success_rate` 외 4축 — **hj15 위의 증분** "
          "(챔피언 등가 fold 층, 팔 D8C)")
    print(f"  기준선 = D8C + gap_reverse −0.017 + col {DEPLOYED[0]} β {DEPLOYED[1]:+.4f}"
          f"  [{'shift_to_prior' if NO_SCALE else '+ 챔피언 아핀 PROB_SCALE 1.05'}]")
    print(f"  후보 {len(CANDS)} · 널뱅크 {len(NULLS)} · beta 격자 {len(FINE)}점 · "
          f"B_BOOT {B_BOOT}")
    print("  ⚠️ 타자 emb1[0] 콜드패치 · 라우팅 11% 는 이 층에 없다 — 크기 전이 안 됨\n")

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

        print("=" * 118)
        print(f"fold {fold}   valid {len(yv):,}행")

        # ── G0 앵커 (shift_to_prior 기준으로 assert) ────────────────────
        a_r = asofdelta.fit_asof_axes(tr, [(ANCHOR, 1.0)], CF.ASOF_S, ct=ct)[0]
        v_r = F.asof_lookup(d, a_r, ct=ct).astype(np.float64)
        pr0 = 1 / (1 + np.exp(-z0))
        prA = 1 / (1 + np.exp(-(z0 + ANCHOR_BETA * v_r)))
        d_ns = bss(yv, shift_to_prior(prA, r)) - bss(yv, shift_to_prior(pr0, r))
        err = abs(d_ns - ANCHOR_EXPECT[fold]) / abs(ANCHOR_EXPECT[fold])
        print(f"  G0  앵커 {ANCHOR} {ANCHOR_BETA:+.3f}  shift_to_prior D {d_ns:+.2f} "
              f"(§7.216 {ANCHOR_EXPECT[fold]:+.2f}, 오차 {err:.2%})  "
              f"{'OK' if err < 0.02 else 'FAIL'}")
        assert err < 0.02, "G0 실패"
        z0 = z0 + ANCHOR_BETA * v_r

        # ── 피처 · 배포 col 축 재현 (G0b) ──────────────────────────────
        wt = recency_weight(tr["season"].to_numpy(), RECENCY_HALF_LIFE,
                            ref_season=int(tr["season"].max()))
        art = F.fit_artifacts(tr, weights=wt, k_pitcher=SMOOTH_K_PITCHER,
                              k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)
        Xv = F.build_features(d, art)
        idx = np.sort(rng.choice(len(tr), size=min(N_TRAIN_SUB, len(tr)),
                                 replace=False))
        Xt = F.build_features(tr.iloc[idx], art)

        def axis_of(col):
            t = np.asarray(Xt[col], dtype=np.float64)
            t = t[np.isfinite(t)]
            return std_axis(np.asarray(Xv[col], dtype=np.float64),
                            float(t.mean()), float(t.std()))

        v_dep = axis_of(DEPLOYED[0])
        base_pre = sc(z0)
        got = sc(z0 + DEPLOYED[1] * v_dep) - base_pre
        exp = DEPLOY_EXPECT[fold]
        tag = "" if exp is None else (f" (§7.223d {exp:+.2f}, 오차 "
                                      f"{abs(got - exp) / abs(exp):.1%})")
        print(f"  G0b 배포 col {DEPLOYED[0]} {DEPLOYED[1]:+.4f}  D {got:+.2f}{tag}")
        z0 = z0 + DEPLOYED[1] * v_dep
        base = sc(z0)
        print(f"  기준선 {base:.2f}  (= hj15 등가)")
        print("=" * 118)

        print(f"\n  {'축':<40}{'종류':>5}{'beta*':>10}{'D@beta*':>10}{'G4':>4}"
              f"{'corr(배포)':>11}   0 근처 곡선")
        for col in CANDS + NULLS:
            if col not in Xv.columns:
                print(f"  {col:<40}  — 없음")
                continue
            v = axis_of(col)
            dd = np.array([sc(z0 + b * v) - base for b in FINE])
            bs = plateau_beta(FINE, dd, PLATEAU_FRAC)[0]
            g4 = bs is not None
            j = int(np.argmin(np.abs(FINE - bs))) if g4 else int(np.argmax(dd))
            curves.setdefault(col, {})[fold] = (FINE.copy(), dd, float(FINE[j]),
                                                float(dd[j]))
            if col in CANDS:
                vs.setdefault(fold, {})[col] = v
            cr = float(np.corrcoef(v, v_dep)[0, 1])
            head = "  ".join(f"{FINE[i]:+.4f}:{dd[i]:+.2f}"
                             for i in (2, 6, 9, 12, 15, 19, 25))
            print(f"  {col:<40}{'후보' if col in CANDS else '널':>5}"
                  f"{FINE[j]:>+10.4f}{dd[j]:>+10.2f}{'Y' if g4 else 'n':>4}"
                  f"{cr:>+11.3f}   {head}")
        ctxs[fold] = (z0, base, yv, r, d, sc)

    nb = {f: max(curves[c][f][3] for c in NULLS if c in curves) for f in FOLDS}
    print(f"\n  널뱅크 max:  f2023 {nb[2023]:+.2f}   f2024 {nb[2024]:+.2f}")

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
    print(f"\n  G3 투수 클러스터 부트스트랩 (fold2024, B={B_BOOT})")
    for c in CANDS:
        if c not in vs.get(2024, {}):
            continue
        b = curves[c][2024][2]
        pa = to_prob(1 / (1 + np.exp(-(z0 + b * vs[2024][c]))), r)
        dif = np.array([bss(yv[i], pa[i]) - bss(yv[i], p0[i]) for i in idxs])
        lo, hi = np.percentile(dif, [2.5, 97.5])
        print(f"    {c:<40}beta*={b:+.4f}  D {curves[c][2024][3]:+7.2f}  "
              f"95% [{lo:+7.2f}, {hi:+7.2f}]  P(D>0) {(dif > 0).mean():.3f}")

    print("\n  G5 시즌 전이 (f2023 beta* -> f2024 채점, 그 반대)")
    print(f"    {'축':<40}{'f23 b*':>9}{'f23 D':>8}{'f24 b*':>9}{'f24 D':>8}"
          f"{'23->24':>9}{'24->23':>9}  판정")
    for col in CANDS + NULLS:
        if col not in curves or len(curves[col]) < 2:
            continue
        b23, d23, s23, v23 = curves[col][2023]
        b24, d24, s24, v24 = curves[col][2024]
        f_ = float(d24[int(np.argmin(np.abs(b24 - s23)))])
        w_ = float(d23[int(np.argmin(np.abs(b23 - s24)))])
        ok = (np.sign(s23) == np.sign(s24)) and v24 > nb[2024] and f_ > 0 and w_ > 0
        print(f"    {col:<40}{s23:>+9.4f}{v23:>+8.2f}{s24:>+9.4f}{v24:>+8.2f}"
              f"{f_:>+9.2f}{w_:>+9.2f}  {'PASS' if ok else 'fail'}"
              f"  [{'후보' if col in CANDS else '널'}]")


if __name__ == "__main__":
    main()
