"""§7.216 `gap_reverse` — **배포 축 단위**로 β 를 고른다 (챔피언 등가 fold 층).

base: `hj05_r047f330.zip` (사용자 지정 챔피언, LB 1123.7187284865)
측정층: `champfold.py` 의 fold 재학습 층 (코어 LGBM 20 · MLP D8C · CatBoost 12 · δ 5축)
층 격차: compact paired · general_lgb 12 · user_fspec 라우팅(예측 가중 ~11%) 과
        PROB_SCALE 1.05 가 이 층에 없다. 크기는 챔피언에서 그대로 안 나온다.
재베이스 계획: `tools_hj/rebase_asof_axis.py` — 챔피언 zip 의 `artifacts['asof']` 에
        축 하나를 append 하고 PROB_SHIFT 를 재보정한다. 이 경로가 없으면 후보가 아니다.

■ 왜 다시 재는가 — 단위가 다르다
`foldchamp_rev*.log`(직전 세션)의 β 격자는 `mixresid.build` 의 **원단위 gap** 위에서
잰 것이다. 그런데 배포 경로(`features.asof_lookup`)는
    gap = (k_szn + S*prior)/(n_szn + S) - prior      (S = 35, 챔피언 아티팩트 실측)
    v   = (gap - mu)/sd                              <- **표준화**
    z  += beta * v
로 **표준화된** 축에 β 를 곱한다. 두 단위는 sd 배(~0.02) 만큼 다르다. 원단위 -0.60 을
그대로 아티팩트에 적으면 수십 배 과다다. 여기서는 처음부터 **배포 함수로** 축을 만들어
배포 단위 β 를 고른다 — 그래야 고른 값이 zip 에 그대로 들어간다.

■ 오염 보정 (§7.60)
`gap_reverse` 는 pre-2023 F 오염이 가장 심한 컬럼이다. `CONTAM_OF['gap_reverse']`
= ('p', +1.0). 보정판/무보정판을 **둘 다** 재서 어느 쪽을 배포할지 자료로 정한다.
(챔피언의 `gap_success`·`gap_bsuccess` 는 cadj 有, `gap_ball` 은 無 — 혼재돼 있다.)

■ 관문
  G0 재현: 우리 코드로 챔피언의 `gap_success` 축을 다시 만들어 mu/sd 가 맞는가
     (`--repro`). 안 맞으면 아래 전부 무효다.
  G1 양 fold 에서 채택 β 의 Δ 부호가 같은가
  G2 엔티티 순열 위약이 후보보다 작은가 (같은 β 에서)
  G3 투수 클러스터 부트스트랩 95% 하한 > 0 인가
  G4 β 는 argmax 가 아니라 **고원 기하**로 고른다 (config.py 규칙):
     고원 = Δ 가 최대의 90% 이상인 연속 구간, 허용 = [2*lo, hi/1.5],
     채택 = 허용 안 격자점 중 최대 |β| 의 **한 칸 아래**

실행:
  .venv/Scripts/python.exe -u src/grevaxis.py --repro
  .venv/Scripts/python.exe -u src/grevaxis.py 2>&1 | tee experiments/grevaxis.log
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd

import asofdelta
import contam_fit
import features as F
import hillclimb as H
import modelabel
from config import MLP_ALPHA, MLP_SEEDS, TARGET_COL, TRAIN_CSV
from data import load_csv
from gapstrike import EXTRA_SPEC
from validation import bss, shift_to_prior

import champfold as CF

CACHE = CF.CACHE
FOLDS = (2023, 2024)
AXIS = "gap_reverse"
STRENGTH = CF.ASOF_S                       # 35.0 — 챔피언 아티팩트에서 읽는다
# 배포 단위 β. gap_success 가 +0.03 이므로 같은 눈금에서 시작해 넓게 편다.
BETAS = np.array([-0.060, -0.050, -0.045, -0.040, -0.035, -0.030, -0.025, -0.020,
                  -0.017, -0.015, -0.0125, -0.010, -0.0075, -0.005, -0.0025,
                  0.005, 0.010, 0.020])
B_BOOT = 1000


def fit_axis(tr, name, beta, ct):
    """배포와 **같은 함수**로 축 하나를 만든다."""
    return asofdelta.fit_asof_axes(tr, [(name, beta)], STRENGTH, ct=ct)[0]


def repro_gate():
    """G0 — 챔피언 아티팩트의 asof 축을 우리 코드로 재현할 수 있는가."""
    for k, v in EXTRA_SPEC.items():
        asofdelta.ASOF_SPEC.setdefault(k, v)
    hj = joblib.load(CF.HJ / "model" / "artifacts.joblib")["artifacts"]
    df = load_csv(TRAIN_CSV)
    lab, _, _ = modelabel.recover(df)
    rev = np.where(np.isnan(lab["reverse"]), 0.0, np.round(lab["reverse"]))
    ct = contam_fit.fit_contam(df, rev)
    print(f"G0 재현 관문 — 전 시즌 train {len(df):,}행 · S={STRENGTH}")
    print(f"  {'축':<14}{'cadj':>10}{'mu(hj)':>15}{'mu(재현)':>15}"
          f"{'sd(hj)':>15}{'sd(재현)':>15}{'판정':>6}")
    ok = True
    for a in hj["asof"]:
        nm = a["name"]
        asofdelta.ASOF_SPEC.setdefault(nm, {k: a[k] for k in ("id", "n_col", "r_col")})
        mine = fit_axis(df, nm, float(a["beta"]), ct if "cadj" in a else None)
        good = (abs(mine["mu"] - a["mu"]) < 1e-9 and abs(mine["sd"] - a["sd"]) < 1e-9
                and len(mine["keys"]) == len(a["keys"]))
        ok &= good
        print(f"  {nm:<14}{str(a.get('cadj', '-')):>10}{a['mu']:>15.9f}"
              f"{mine['mu']:>15.9f}{a['sd']:>15.9f}{mine['sd']:>15.9f}"
              f"{'OK' if good else 'NG':>6}")
    print(f"  -> {'통과' if ok else '실패 — 아래 측정은 무효다'}")
    return ok


PLATEAU_FRAC = 0.5     # 고원 = Δ 가 최대의 이 비율 이상인 **연속** 구간


def plateau_beta(betas, deltas, frac=PLATEAU_FRAC):
    """G4 — 고원 기하로 β 를 고른다.

    ⚠️ `frac` 는 관례가 아니라 선택이다. config.py 가 `gap_success` 에 적은 고원
    [0.005, 0.06](폭 12배)은 90% 등고선으로는 절대 나오지 않는다 — 그건 **반값
    등고선**(또는 양수 구간) 규모다. 여기서는 0.5 로 고정하고 그 사실을 적어 둔다.
    90% 로 잡으면 이 축은 고원 폭이 1.6배뿐이라 ±50% 밴드가 아예 안 들어간다.
    """
    d = np.asarray(deltas, dtype=np.float64)
    i = int(np.argmax(d))
    if d[i] <= 0:
        return None, None, None
    thr = frac * d[i]
    lo, hi = i, i
    while lo - 1 >= 0 and d[lo - 1] >= thr:
        lo -= 1
    while hi + 1 < len(d) and d[hi + 1] >= thr:
        hi += 1
    a, b = sorted((abs(betas[lo]), abs(betas[hi])))
    allow = (2.0 * a, b / 1.5)                       # ±50% 전이 오차 밴드가 고원 안
    if allow[0] > allow[1]:
        return None, (a, b), allow
    sgn = np.sign(betas[i])
    cand = sorted([float(x) for x in betas
                   if allow[0] <= abs(x) <= allow[1] and np.sign(x) == sgn], key=abs)
    if not cand:
        return None, (a, b), allow
    pick = cand[-2] if len(cand) >= 2 else cand[-1]  # 허용 최대 **한 칸 아래**
    return pick, (a, b), allow


def main():
    for k, v in EXTRA_SPEC.items():
        asofdelta.ASOF_SPEC.setdefault(k, v)
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y_all = df[TARGET_COL].to_numpy(dtype=np.float64)

    print(f"§7.216  {AXIS} — 배포 축 단위 β 선정 · 챔피언 등가 fold 층 (팔 D8C)")
    print(f"  base = hj05_r047f330.zip · 축 강도 S={STRENGTH} · "
          f"챔피언 asof {CF.ASOF_DEPLOY}")
    print("  ⚠️ 이 층에는 라우팅 29파일·PROB_SCALE 이 없다 — 크기는 전이 안 된다\n")

    res = {}
    for fold in FOLDS:
        st, mm, yv, d, core, cat, noid, ax, tr, w = CF.fold_state(df, s, y_all, fold)
        r = float(yv.mean())
        e = [np.load(CACHE / f"champ_D8C_f{fold}_emb4_{sd}.npz")["z"] for sd in MLP_SEEDS]
        zm = (0.875 * np.mean(e, axis=0) + 0.125 * np.mean(noid, axis=0))[mm]
        z0 = ((1 - H.W_CAT) * ((1 - MLP_ALPHA) * core + MLP_ALPHA * zm)
              + H.W_CAT * cat + ax)

        def sc(z):
            return bss(yv, shift_to_prior(1 / (1 + np.exp(-z)), r))

        base = sc(z0)
        lab, _, _ = modelabel.recover(tr)
        rev = np.where(np.isnan(lab["reverse"]), 0.0, np.round(lab["reverse"]))
        ct = contam_fit.fit_contam(tr, rev)

        rng = np.random.default_rng(0)
        dsh = d.copy()
        dsh["pitcher_id"] = rng.permutation(d["pitcher_id"].to_numpy())

        codes, uniq = pd.factorize(d["pitcher_id"].to_numpy())
        order = np.argsort(codes, kind="stable")
        bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
        seen = np.where(np.diff(bnd) > 0)[0]
        br = np.random.default_rng(7)
        IDX = [np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                               br.choice(seen, size=len(seen), replace=True)])
               for _ in range(B_BOOT)]

        print("=" * 108)
        print(f"fold {fold}   valid {len(yv):,}행   기준(D8C) {base:.2f}   "
              f"⚠️ train < {fold} — 홀드아웃")
        print("=" * 108)

        for tag, use_ct in (("보정", ct), ("무보정", None)):
            a = fit_axis(tr, AXIS, 1.0, use_ct)
            v, hit = F.asof_lookup(d, a, with_hit=True, ct=use_ct)
            v = v.astype(np.float64)
            vs = F.asof_lookup(dsh, a, ct=use_ct).astype(np.float64)
            row = np.array([sc(z0 + b * v) - base for b in BETAS])
            plc = np.array([sc(z0 + b * vs) - base for b in BETAS])
            print(f"\n  [{tag}]  조회 적중 {hit.mean():.1%} · mu {a['mu']:+.6f} "
                  f"sd {a['sd']:.6f} · v: 평균 {v.mean():+.4f} SD {v.std():.4f}")
            print(f"  {'β':>6}" + "".join(f"{b:>9.4f}" for b in BETAS))
            print(f"  {'Δ':>6}" + "".join(f"{x:>+9.2f}" for x in row))
            print(f"  {'위약':>6}" + "".join(f"{x:>+9.2f}" for x in plc))
            res[(fold, tag)] = dict(row=row, plc=plc, base=base, z0=z0, v=v,
                                    yv=yv, r=r, IDX=IDX)
        print()

    # ── 판정 ────────────────────────────────────────────────────────────
    picks = {}
    for tag in ("보정", "무보정"):
        print("=" * 108)
        print(f"판정  [{tag}]")
        print("=" * 108)
        r23, r24 = res[(2023, tag)]["row"], res[(2024, tag)]["row"]
        p23, p24 = res[(2023, tag)]["plc"], res[(2024, tag)]["plc"]
        print(f"  {'β':>8}{'f2023':>10}{'f2024':>10}{'크기비':>10}"
              f"{'위약23':>10}{'위약24':>10}")
        for i, b in enumerate(BETAS):
            ratio = (r23[i] / r24[i]) if abs(r24[i]) > 1e-9 else float("nan")
            print(f"  {b:>8.4f}{r23[i]:>+10.2f}{r24[i]:>+10.2f}{ratio:>10.2f}"
                  f"{p23[i]:>+10.2f}{p24[i]:>+10.2f}")

        pick, span, allow = plateau_beta(BETAS, r24)   # 주 판단 근거 = fold2024
        if pick is None:
            print(f"\n  fold2024 고원 |β| {span} · 허용 {allow} -> 채택 불가\n")
            continue
        print(f"\n  fold2024 고원 |β| [{span[0]:.2f}, {span[1]:.2f}] -> "
              f"허용 [{allow[0]:.4f}, {allow[1]:.4f}] -> **채택 β = {pick:+.4f}**")
        j = int(np.argmin(np.abs(BETAS - pick)))
        okboot = True
        for fold in FOLDS:
            R = res[(fold, tag)]
            zc = R["z0"] + pick * R["v"]
            p1 = shift_to_prior(1 / (1 + np.exp(-zc)), R["r"])
            p0 = shift_to_prior(1 / (1 + np.exp(-R["z0"])), R["r"])
            bs = np.array([bss(R["yv"][k], p1[k]) - bss(R["yv"][k], p0[k])
                           for k in R["IDX"]])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            okboot &= lo > 0
            print(f"  fold{fold}  Δ {R['row'][j]:+.2f}  95% [{lo:+.2f}, {hi:+.2f}]  "
                  f"P(Δ>0) {float((bs > 0).mean()):.3f}  위약 {R['plc'][j]:+.2f}")
        g1 = r23[j] * r24[j] > 0
        g2 = (r24[j] > p24[j]) and (r23[j] > p23[j])
        print(f"  G1 부호 일치 {'통과' if g1 else '실패'}   "
              f"G2 위약 우위 {'통과' if g2 else '실패'}   "
              f"G3 부트 하한>0 {'통과' if okboot else '실패'}")
        picks[tag] = (pick, g1, g2, okboot, r23[j], r24[j])
        print()

    print("=" * 108)
    print("요약 — 빌드에 쓸 값")
    for tag, (pick, g1, g2, g3, d23, d24) in picks.items():
        print(f"  [{tag}]  β = {pick:+.4f}   f2023 {d23:+.2f} / f2024 {d24:+.2f}   "
              f"관문 {int(g1)+int(g2)+int(g3)}/3")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", action="store_true")
    if ap.parse_args().repro:
        repro_gate()
    else:
        main()
