"""§7.58 재방문 — 구종 예측 축은 **현 배포 스택(s40)** 잔차에서도 죽어 있는가

사용자 질문: "이전에 구종 예측을 물어봤는데 지금은 어떤지."
§7.58(8/17)의 판정 — ptdev_*(증류된 상호작용) 기각, pt_*는 주효과(r3_*)와 동일 —
은 당시 스택(LGBM+MLP 블렌드 + δ + asof) 잔차 기준이다. 그 뒤 스택에 soft label,
보조 헤드, CatBoost, fspec, SMOTE-LGBM, 확률 스케일이 들어왔다 (s18→s40, LB
1044→1111). 메모리 "닫은 진단도 새 도구로 다시 물어라" + "진단은 배포층에서 걸어라".

기준선 2개
    A 블렌드층  (1−α)·z_lgb + α·z_mlp + δ(ctx+asof)          — §7.58과 같은 조건.
                fold 2023·2024, foldcache(현 config 지문) 재사용.
    B 배포층    s40 전체: SMOTE-LGBM 팔 + base MLP·CatBoost + fspec(F행)
                + δ + 확률 스케일 — fold 2024만 (sfull 캐시가 2024뿐).

판정 규칙은 §7.58 그대로: 양 fold 부호 일치 + |t| > 2|셔플 대조군 t|.
β 스윕은 참고용 (채택 제안이 아니다 — r3_ent가 이 관문을 다 통과하고 LB −11.01).

실행:  .venv/Scripts/python.exe -u src/pitchnow.py 2>&1 | tee experiments/pitchnow.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import lightgbm as lgb
import numpy as np
import pandas as pd

import foldcache
import jointbeta as J
import pitchtype as PT
from config import (ASOF_AXES, CAT_ALPHA, CTX_AXES, FSPEC_ALPHA, MLP_ALPHA,
                    SEED, TARGET_COL, TRAIN_CSV)
from ctxsplit import resid_t
from data import load_csv
from hbeta2 import train_fspec
from predict import PROB_CENTER, PROB_SCALE
from smotenc import prep
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
CACHE = Path(__file__).parent.parent / "experiments" / "cache"


def pt_features(d, m_full, m_nomix, league, rng):
    """§7.58 ④와 동일 — main 행 예측 피처 + 믹스 셔플 대조군."""
    xm, r3m, tot = PT.main_matrix(d, league)
    pt = m_full.predict(xm)
    lgp = m_nomix.predict(xm[:, PT.NOMIX])
    feat = {}
    for i, nm in enumerate(("ff", "br", "os")):
        feat["pt_" + nm] = pt[:, i]
        feat["ptdev_" + nm] = pt[:, i] - r3m[:, i]
        feat["lg_" + nm] = lgp[:, i]
    feat["pt_ent"] = PT.entropy(pt)
    for i, nm in enumerate(("ff", "br", "os")):
        feat["r3_" + nm] = r3m[:, i]
    feat["r3_ent"] = PT.entropy(r3m)
    feat["mix3_sum"] = np.nan_to_num(tot, nan=0.0)

    perm = rng.permutation(len(xm))
    xs = xm.copy()
    xs[:, 6:10] = xm[perm][:, 6:10]
    pts = m_full.predict(xs)
    fsh = {}
    for i, nm in enumerate(("ff", "br", "os")):
        fsh["pt_" + nm] = pts[:, i]
        fsh["ptdev_" + nm] = pts[:, i] - r3m[perm][:, i]
        fsh["lg_" + nm] = feat["lg_" + nm]
    fsh["pt_ent"] = PT.entropy(pts)
    for i, nm in enumerate(("ff", "br", "os")):
        fsh["r3_" + nm] = r3m[perm][:, i]
    fsh["r3_ent"] = PT.entropy(r3m[perm])
    fsh["mix3_sum"] = feat["mix3_sum"][perm]
    return feat, fsh


def gate_table(tag, r, feat, fsh, keep):
    print(f"\n  [{tag}]  {'피처':<12}{'잔차 t':>10}{'대조군 t':>11}{'차':>9}")
    for k in feat:
        _, t = resid_t(r, feat[k])
        _, tc = resid_t(r, fsh[k])
        keep.setdefault((tag, k), []).append((t, tc))
        mark = "  ✓" if abs(t) > 2 and abs(t) > 2 * abs(tc) else ""
        print(f"  {'':<7}{k:<12}{t:>10.2f}{tc:>11.2f}{t - tc:>9.2f}{mark}")


def beta_sweep(tag, z_logit, yv, feat, to_prob):
    """z_logit + β·x → to_prob(확률 변환은 층마다 다르다). 참고용."""
    base = bss(yv, to_prob(z_logit))
    print(f"\n  [{tag}] β 스윕 (ΔBSS vs {base:.1f})")
    print("  " + f"{'피처':<12}" + "".join(f"{b:>8.3f}" for b in PT.BETAS))
    for k in PT.SWEEP:
        x = feat[k]
        xc = (x - x.mean()) / max(float(x.std()), 1e-9)
        row = [bss(yv, to_prob(z_logit + b * xc)) - base for b in PT.BETAS]
        print("  " + f"{k:<12}" + "".join(f"{v:>8.2f}" for v in row))


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(SEED)

    tm = PT.load_trackman()
    ok = tm["y"].to_numpy() >= 0
    print(f"trackman {len(tm):,}행  3그룹 {ok.sum():,}행 ({ok.mean():.3f})")
    print(f"기준선 A = 블렌드층 (α={MLP_ALPHA} + δ{CTX_AXES}+{ASOF_AXES})  fold {FOLDS}")
    print(f"기준선 B = s40 배포층 (SMOTE-LGBM·MLP·CatBoost α={CAT_ALPHA}"
          f"·fspec α={FSPEC_ALPHA}·스케일 {PROB_SCALE})  fold 2024만")

    keep = {}
    for fold in FOLDS:
        # ── 구종 예측기: §7.58과 동일 (tm season < fold) ──────────────
        fit = ok & (tm["season"].to_numpy() < fold)
        yf = tm.loc[fit, "y"].to_numpy()
        league = np.array([float((yf == i).mean()) for i in range(3)])
        xf = PT.tm_matrix(tm[fit], league)
        m_full = lgb.train(PT.PT_PARAMS, lgb.Dataset(xf, label=yf), PT.PT_ROUNDS)
        m_nomix = lgb.train(PT.PT_PARAMS, lgb.Dataset(xf[:, PT.NOMIX], label=yf),
                            PT.PT_ROUNDS)
        print("\n" + "=" * 78)
        print(f"fold {fold}  구종 예측기 학습 {fit.sum():,}행")

        va_m = s == fold
        d = df[va_m].copy()
        yv = cached[fold][2]
        feat, fsh = pt_features(d, m_full, m_nomix, league, rng)

        # ── 기준선 A: 블렌드층 (δ는 배포 코드 경로 jointbeta로) ────────
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
        Xax, _, _, _ = J.axis_matrix(df, s, fold)
        ax = Xax @ dep
        zA = z + ax

        def probA(zz, m=yv.mean()):
            return shift_to_prior(1.0 / (1.0 + np.exp(-zz)), m)

        pA = probA(zA)
        rA = yv - pA
        print(f"  A 블렌드층  BSS {bss(yv, pA):.1f}  행 {len(d):,}")
        gate_table("A", rA, feat, fsh, keep)
        beta_sweep("A", zA, yv, feat, probA)

        # ── 기준선 B: s40 배포층 전체 (fold 2024만, sfull 캐시) ────────
        if fold == 2024:
            need = {"lgb": CACHE / "sfull_f2024_SMOTE_lgb.npz",
                    "cat": CACHE / "sfull_f2024_base_cat.npz"}
            mlps = sorted(CACHE.glob("sfull_f2024_base_mlp_*.npz"))
            if all(p.exists() for p in need.values()) and mlps:
                zl = np.load(need["lgb"])["z"]
                cb = np.load(need["cat"])["z"]
                mb = np.mean([np.load(p)["z"] for p in mlps], axis=0)
                tr2, va2, w2, Xfn, art2 = prep(df, fold)
                z_fs, _ = train_fspec(tr2, Xfn, Xfn(va2))
                isF = va2["game_type"].to_numpy().astype(str) == "F"
                zB = (1 - CAT_ALPHA) * ((1 - MLP_ALPHA) * zl + MLP_ALPHA * mb) \
                    + CAT_ALPHA * cb
                zB = zB.copy()
                zB[isF] = (1 - FSPEC_ALPHA) * zB[isF] + FSPEC_ALPHA * z_fs[isF]
                zB = zB + ax

                def probB(zz, m=yv.mean()):
                    p = shift_to_prior(1.0 / (1.0 + np.exp(-zz)), m)
                    return shift_to_prior(
                        PROB_CENTER + PROB_SCALE * (p - PROB_CENTER), m)

                pB = probB(zB)
                rB = yv - pB
                print(f"\n  B 배포층    BSS {bss(yv, pB):.1f}  (MLP {len(mlps)}멤버)")
                gate_table("B", rB, feat, fsh, keep)
                beta_sweep("B", zB, yv, feat, probB)
            else:
                print("\n  B 배포층: sfull 캐시 없음 — 건너뜀")

    print("\n" + "=" * 78)
    print("판정 — §7.58 규칙 (A는 양 fold 부호 일치 + 대조군 2배 초과, B는 단일 fold 참고)")
    print("=" * 78)
    a_keys = sorted({k for (tag, k) in keep if tag == "A"})
    print(f"  {'피처':<12}{'A t23':>9}{'A t24':>9}{'부호':>6}{'대조군':>8}"
          f"{'B t24(배포층)':>14}")
    for k in a_keys:
        (t1, c1), (t2, c2) = keep[("A", k)]
        same = (t1 > 0) == (t2 > 0)
        over = abs(t1) > 2 * abs(c1) and abs(t2) > 2 * abs(c2)
        b = keep.get(("B", k))
        btxt = f"{b[0][0]:>10.2f}" if b else "      —"
        bover = ""
        if b:
            tb, cb2 = b[0]
            bover = " ✓" if abs(tb) > 2 and abs(tb) > 2 * abs(cb2) else " ✗"
        print(f"  {k:<12}{t1:>9.2f}{t2:>9.2f}{'✓' if same else '✗':>6}"
              f"{'✓' if over else '✗':>8}{btxt}{bover}")
    print("\n  읽는 법: §7.58 당시 A층 판정 — ptdev_* ✗ / pt_*·r3_ent ✓✓ (이후 r3_ent")
    print("  LB −11.01 로 기각). B열이 지금의 진짜 질문 — 배포층 잔차에 남은 게 있는가.")


if __name__ == "__main__":
    main()
