"""§7.91 C·B군 잔차 컬럼을 **배포 δ 축**으로 실측 (`src/colaxis.py`).

base: `hj05_r047f330.zip` → 기준선은 `--on-grev` 로 hj09 등가(+ `gap_reverse` −0.017).
측정층: `champfold.py` fold 재학습 층 (코어 LGBM 20 · MLP D8C · CatBoost 12 · δ 5축).
층 격차: 라우팅 29파일 · `PROB_SCALE 1.05` 없음 — 크기는 챔피언에서 그대로 안 나온다.
재베이스 계획: 컬럼형 축이라 `tools_hj/rebase_asof_axis.py` 가 그대로는 안 된다.
        `artifacts['asof']` 는 조회형 전용이므로 **새 축 종류**의 서빙 훅이 필요하다
        (`predict.logit_adjust` 에 선형항 1개). 후보가 살면 그때 뚫는다.

■ 후보 (§7.91 C군 = "잔차에 부호 안정 구조 (참고, 질문 밖)" · B군 = ycorr 낮음)
    asof_pitcher_middle_rate(_sm)               t −4.3~−5.3  양 fold 음수  ← C군 유일 눈금 위
    matchup_success_edge                        t +3.8/+2.4
    asof_pitcher_prev5/prev3_game_middle_rate   t −6.4/−2.9 · −5.1/−2.6   ← B군

■ ★ 위약이 gbmid 와 다르다
`gap_*` 축은 (pitcher_id → 커리어 통계) **조회**라 ID 순열이 축을 지운다. 컬럼형 축은
값이 행에 붙어 있어 **ID 순열이 `max|Δv| = 0`** 이다. 그래서 널을 §7.91 이 실측으로
확정한 **죽은 축**으로 만든다 (A군 6개 + 드랍된 달력 2개). 셔플은 0 을 주지만 죽은
진짜 축은 후보와 같은 프로파일을 준다. [[use-a-known-dead-axis-as-the-null]]

■ 관문
  G0  앵커 — `gap_reverse` β=−0.017 이 §7.216 의 +30.61 / +4.75 를 재현하는가
  G1  양 fold 부호 일치
  G2  후보 Δ@β* > **널뱅크 max** (같은 β 격자)
  G3  투수 클러스터 부트스트랩 95% 하한 > 0 (fold2024)
  G4  β 는 argmax 가 아니라 고원 기하 (frac 0.5)
  G5  **시즌 전이** — f2023 β* 로 f2024 채점(그 반대도). 위약은 지속성을 못 잰다
      [[placebo-cannot-test-persistence]]

실행:  .venv/Scripts/python.exe -u src/colaxis.py --on-grev 2>&1 | tee experiments/colaxis.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import asofdelta
import champfold as CF
import contam_fit
import features as F
import hillclimb as H
import modelabel
from config import (CONTAM_AXES, MLP_ALPHA, MLP_SEEDS, RECENCY_HALF_LIFE, SEED,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from gapstrike import EXTRA_SPEC
from grevaxis import PLATEAU_FRAC, plateau_beta
from validation import bss, shift_to_prior

ON_GREV = "--on-grev" in sys.argv
ANCHOR, ANCHOR_BETA = "gap_reverse", -0.017
ANCHOR_EXPECT = {2023: +30.61, 2024: +4.75}
FOLDS = (2023, 2024)
N_TRAIN_SUB = 400_000
B_BOOT = 1000

CANDS = ["asof_pitcher_middle_rate", "asof_pitcher_middle_rate_sm",
         "matchup_success_edge",
         "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate"]
NULLS = ["outs_before", "strikes_before", "num_runners_on", "is_first_pitch",
         "is_late_close", "top_bottom", "game_dayofweek", "game_month"]

BETAS = np.array([-0.30, -0.25, -0.20, -0.15, -0.12, -0.10, -0.08, -0.06, -0.05,
                  -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                  0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30])


def std_axis(raw, mu, sd):
    v = np.asarray(raw, dtype=np.float64)
    v = np.where(np.isfinite(v), v, mu)
    return (v - mu) / (sd if sd > 0 else 1.0)


def main():
    for k, v in EXTRA_SPEC.items():
        asofdelta.ASOF_SPEC.setdefault(k, v)
    print("§7.223  §7.91 잔차 컬럼 -> 배포 delta 축 실측 (챔피언 등가 fold 층, 팔 D8C)"
          + ("  [기준선 = D8C + gap_reverse -0.017 = **hj08 등가** — 타자 emb1[0] 콜드패치 없음]" if ON_GREV else ""))
    print(f"  후보 {len(CANDS)} · 널뱅크 {len(NULLS)} (§7.91 죽은 축) · "
          f"beta 격자 {len(BETAS)}점 · B_BOOT {B_BOOT}")
    print("  주의: 라우팅·PROB_SCALE 없음 — 크기 전이 안 됨 · mu/sd 는 train 부분표집\n")

    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y_all = df[TARGET_COL].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(SEED)
    curves, boot_src = {}, {}

    for fold in FOLDS:
        st, mm, yv, d, core, cat, noid, ax, tr, w = CF.fold_state(df, s, y_all, fold)
        r = float(yv.mean())
        e = [np.load(CF.CACHE / f"champ_D8C_f{fold}_emb4_{sd}.npz")["z"]
             for sd in MLP_SEEDS]
        zm = (0.875 * np.mean(e, axis=0) + 0.125 * np.mean(noid, axis=0))[mm]
        z0 = ((1 - H.W_CAT) * ((1 - MLP_ALPHA) * core + MLP_ALPHA * zm)
              + H.W_CAT * cat + ax)

        def sc(z):
            return bss(yv, shift_to_prior(1 / (1 + np.exp(-z)), r))

        lab, _, _ = modelabel.recover(tr)
        rev = np.where(np.isnan(lab["reverse"]), 0.0, np.round(lab["reverse"]))
        ct = contam_fit.fit_contam(tr, rev)

        print("=" * 118)
        print(f"fold {fold}   valid {len(yv):,}행   기준(D8C) {sc(z0):.2f}   "
              f"train < {fold} (홀드아웃)")
        print("=" * 118)

        a_r = asofdelta.fit_asof_axes(tr, [(ANCHOR, 1.0)], CF.ASOF_S, ct=ct)[0]
        v_r = F.asof_lookup(d, a_r, ct=ct).astype(np.float64)
        got = sc(z0 + ANCHOR_BETA * v_r) - sc(z0)
        err = abs(got - ANCHOR_EXPECT[fold]) / abs(ANCHOR_EXPECT[fold])
        print(f"  G0 앵커 {ANCHOR} beta={ANCHOR_BETA:+.3f}  D {got:+.2f}  "
              f"(§7.216 {ANCHOR_EXPECT[fold]:+.2f}, 오차 {err:.2%})  "
              f"{'OK' if err < 0.02 else 'FAIL - 아래 전부 무효'}")
        assert err < 0.02, "G0 실패"
        if ON_GREV:
            z0 = z0 + ANCHOR_BETA * v_r
        base = sc(z0)
        print(f"  기준선 {base:.2f}{'  (hj09 등가)' if ON_GREV else ''}")

        wt = recency_weight(tr["season"].to_numpy(), RECENCY_HALF_LIFE,
                            ref_season=int(tr["season"].max()))
        art = F.fit_artifacts(tr, weights=wt, k_pitcher=SMOOTH_K_PITCHER,
                              k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)
        Xv = F.build_features(d, art)
        idx = np.sort(rng.choice(len(tr), size=min(N_TRAIN_SUB, len(tr)),
                                 replace=False))
        Xt = F.build_features(tr.iloc[idx], art)
        have = [c for c in CANDS + NULLS if c in Xv.columns]
        missing = [c for c in CANDS + NULLS if c not in Xv.columns]
        if missing:
            print(f"  없는 컬럼: {missing}")

        print(f"\n  {'축':<42}{'종류':>5}{'beta*':>9}{'D@beta*':>10}{'결측%':>7}"
              f"   beta 곡선")
        for col in have:
            t = np.asarray(Xt[col], dtype=np.float64)
            t = t[np.isfinite(t)]
            mu, sd = float(t.mean()), float(t.std())
            raw = np.asarray(Xv[col], dtype=np.float64)
            v = std_axis(raw, mu, sd)
            dd = np.array([sc(z0 + b * v) - base for b in BETAS])
            bs = plateau_beta(BETAS, dd, PLATEAU_FRAC)[0]
            j = (int(np.argmin(np.abs(BETAS - bs))) if bs is not None
                 else int(np.argmax(dd)))
            g4 = bs is not None
            curves.setdefault(col, {})[fold] = (BETAS.copy(), dd,
                                                float(BETAS[j]), float(dd[j]))
            if col in CANDS:
                boot_src.setdefault(fold, {})[col] = v
            head = "  ".join(f"{BETAS[i]:+.2f}:{dd[i]:+.1f}"
                             for i in (0, 5, 9, 12, 16, 20, 25))
            print(f"  {col:<42}{'후보' if col in CANDS else '널':>5}"
                  f"{BETAS[j]:>+9.3f}{dd[j]:>+10.2f}"
                  f"{100 * (~np.isfinite(raw)).mean():>7.1f}  "
                  f"{'G4' if g4 else 'ar'} {head}")

        if fold == 2024:
            import pandas as pd
            codes, uniq = pd.factorize(d["pitcher_id"].to_numpy())
            order = np.argsort(codes, kind="stable")
            bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
            seen = np.where(np.diff(bnd) > 0)[0]
            br = np.random.default_rng(SEED)
            idxs = [np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                                    br.choice(seen, size=len(seen), replace=True)])
                    for _ in range(B_BOOT)]
            print(f"\n  G3 투수 클러스터 부트스트랩 (fold2024, B={B_BOOT})")
            for col, v in boot_src[2024].items():
                bstar = curves[col][2024][2]
                pz = shift_to_prior(1 / (1 + np.exp(-z0)), r)
                pa = shift_to_prior(1 / (1 + np.exp(-(z0 + bstar * v))), r)
                dif = np.array([bss(yv[i], pa[i]) - bss(yv[i], pz[i]) for i in idxs])
                lo, hi = np.percentile(dif, [2.5, 97.5])
                print(f"    {col:<42}beta*={bstar:+.3f}  "
                      f"D {curves[col][2024][3]:+7.2f}  95% [{lo:+7.2f}, {hi:+7.2f}]  "
                      f"P(D>0) {(dif > 0).mean():.3f}")

    print("\n" + "=" * 118)
    print("종합 — G1 부호 · G2 널뱅크 초과 · G5 시즌 전이")
    print("=" * 118)
    nb = {f: max(curves[c][f][3] for c in NULLS if c in curves and f in curves[c])
          for f in FOLDS}
    print(f"  널뱅크 max:  f2023 {nb[2023]:+.2f}   f2024 {nb[2024]:+.2f}")
    print(f"\n  {'축':<42}{'f23 b*':>9}{'f23 D':>9}{'f24 b*':>9}{'f24 D':>9}"
          f"{'23->24':>10}{'24->23':>10}  판정")
    for col in CANDS + NULLS:
        if col not in curves or len(curves[col]) < 2:
            continue
        b23, d23, s23, v23 = curves[col][2023]
        b24, d24, s24, v24 = curves[col][2024]
        t_fwd = float(d24[int(np.argmin(np.abs(b24 - s23)))])
        t_bwd = float(d23[int(np.argmin(np.abs(b23 - s24)))])
        g1 = (np.sign(s23) == np.sign(s24)) and s23 != 0
        g2 = (v24 > nb[2024]) and (v23 > nb[2023])
        g5 = (t_fwd > 0) and (t_bwd > 0)
        tag = ("PASS" if (g1 and g2 and g5) else
               "FAIL:" + "".join(x for x, ok in (("1", g1), ("2", g2), ("5", g5))
                                 if not ok))
        print(f"  {col:<42}{s23:>+9.3f}{v23:>+9.2f}{s24:>+9.3f}{v24:>+9.2f}"
              f"{t_fwd:>+10.2f}{t_bwd:>+10.2f}  {tag}  "
              f"[{'후보' if col in CANDS else '널'}]")


if __name__ == "__main__":
    main()
