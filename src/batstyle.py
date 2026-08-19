"""§7.73 — 타자 **성질** 축 (접근 스타일 + 상대 ball/reverse). 사용자 제안.

모델의 타자 입력은 `asof_batter_{n,success_rate,middle_rate}` + 손잡이 + 팀뿐이다.
타자 군집화 EDA(notebooks/player_stats)가 보여준 성질 스펙트럼 중 모델에 없는 것:

    ppa          타석당 투구 수 (승부 길이). PA 시작 = (balls==0 & strikes==0) 행
                 이므로 game_id 복원 없이 n / n_first로 계산된다.
    two_strike   2스트라이크 카운트로 가는 성향
    three_ball   3볼 카운트로 가는 성향
    ball_rate    이 타자 상대로 투구가 볼이 되는 비율      (라벨 역산, README 검증)
    reverse_rate 이 타자 상대로 포수 요구 반대로 가는 비율  (오염 구간 제외로 계산)

운반체는 §7.51과 같은 「타자별 표 + ID 조인」이므로 같은 함정(이름표 효과)이 있다.
게이트: ① 두 fold 잔차 t 부호 일치  ② 셔플 대조군(타자 ID 순열)의 2배 초과
       ③ 타자 클러스터 SE 기준 |t|>2 (§7.44 — 회귀자가 타자 내 상수라 유효 표본은
         행 수가 아니라 타자 수다)  ④ 배포 피처 최대상관 < 0.5 (§7.59의 companion check)

오염 처리: reverse 라벨은 2019–22 F에서 검출 실패(README) → clean 행
          (R 전체 + F 2023~)에서만 집계한다. ball은 오염 채널이 아니다.

사전 예측 (실행 전에 박아둔다)
    P1  접근 3종(ppa/two_strike/three_ball)은 배포 피처 상관 < 0.5   예측: **그렇다**
    P2  4게이트 전부 통과하는 축 ≤ 1개                              예측: **그렇다**
    P3  셔플 대조군 |t| < 2 전 축·전 fold                           예측: **그렇다**

실행:  .venv/Scripts/python.exe -u src/batstyle.py 2>&1 | tee experiments/batstyle.log
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
import ctxfit
import features as F
import foldcache
from config import (ASOF_AXES, ASOF_STRENGTH, CONTAM_AXES, CTX_AXES, CTX_HALF_LIFE,
                    CTX_K, MLP_ALPHA, SEED, TARGET_COL, TRAIN_CSV)
from ctxsplit import resid_t
from data import load_csv, recency_weight
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
MIN_N = 200                      # EDA와 같은 기준. 미달이면 리그 평균으로 fallback


def pitch_labels(df):
    """asof 누적 증분으로 투구 단위 라벨 복원 (README에서 100% 일치 검증된 기전)."""
    out = {}
    for m in ("success", "ball", "reverse"):
        cnt = df["asof_pitcher_n"] * df[f"asof_pitcher_{m}_rate"].fillna(0)
        out[m] = (cnt.groupby(df["pitcher_id"], sort=False).shift(-1) - cnt).round()
    chk = out["success"].notna()
    ok = (out["success"][chk] == df[TARGET_COL][chk]).mean()
    assert ok == 1.0, f"라벨 역산 검증 실패 ({ok:.6f})"
    print(f"[label] 역산 검증 OK ({chk.sum():,}행 100% 일치)")
    return out["ball"].to_numpy(np.float64), out["reverse"].to_numpy(np.float64)


def fit_bat(tr, ball, rev, clean):
    """타자별 성질을 train 행에서만 얼린다."""
    bid = tr["batter_id"].to_numpy(np.int64)
    first = ((tr["balls_before"].to_numpy() == 0)
             & (tr["strikes_before"].to_numpy() == 0)).astype(np.float64)
    ts = (tr["strikes_before"].to_numpy() == 2).astype(np.float64)
    tb = (tr["balls_before"].to_numpy() == 3).astype(np.float64)

    u, inv = np.unique(bid, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)

    def rate(x, mask=None):
        if mask is None:
            return np.bincount(inv, weights=x) / n, n
        nm = np.bincount(inv, weights=mask.astype(np.float64))
        s = np.bincount(inv, weights=np.where(mask, x, 0.0))
        return s / np.maximum(nm, 1), nm

    ppa = n / np.maximum(np.bincount(inv, weights=first), 1)
    ts_r, _ = rate(ts)
    tb_r, _ = rate(tb)
    bl_r, bl_n = rate(np.nan_to_num(ball), ~np.isnan(ball))
    rv_r, rv_n = rate(np.nan_to_num(rev), clean & ~np.isnan(rev))

    tab = {"keys": u}
    for k, v, nn in (("ppa", ppa, n), ("two_strike", ts_r, n), ("three_ball", tb_r, n),
                     ("ball_rate", bl_r, bl_n), ("reverse_rate", rv_r, rv_n)):
        lg = float(np.average(v, weights=np.maximum(nn, 1)))
        tab[k] = np.where(nn >= MIN_N, v, lg)
        tab[k + "_lg"] = lg
        tab[k + "_cov"] = nn >= MIN_N
    return tab


def apply_bat(d, tab):
    bid = d["batter_id"].to_numpy(np.int64)
    j = np.clip(np.searchsorted(tab["keys"], bid), 0, len(tab["keys"]) - 1)
    hit = tab["keys"][j] == bid
    feat, cov = {}, {}
    for k in ("ppa", "two_strike", "three_ball", "ball_rate", "reverse_rate"):
        feat[k] = np.where(hit, tab[k][j], tab[k + "_lg"])
        cov[k] = float((hit & tab[k + "_cov"][j]).mean())
    return feat, cov


def clust_t(r, v, bid):
    """타자 클러스터 robust t. 회귀자가 타자 내 상수이므로 타자 수준 WLS와 동치."""
    u, inv = np.unique(bid, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    rb = np.bincount(inv, weights=r) / n
    vb = np.bincount(inv, weights=v) / n
    vw = np.average(vb, weights=n)
    rw = np.average(rb, weights=n)
    dc = vb - vw
    den = float((n * dc * dc).sum())
    if den <= 0:
        return 0.0
    sl = float((n * dc * (rb - rw)).sum() / den)
    res = (rb - rw) - sl * dc
    se = float(np.sqrt((n * n * dc * dc * res * res).sum()) / den)
    return sl / se if se > 0 else 0.0


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    ball, rev = pitch_labels(df)
    clean_all = ((df["game_type"].to_numpy() == "R") | (s >= 2023))
    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(SEED)

    for fold in FOLDS:
        tr_m, va = s < fold, s == fold
        tr, d = df[tr_m], df[va].copy()
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ref = int(s[tr_m].max())
        ctx = ctxfit.fit_ctx(tr, TARGET_COL, CTX_AXES, CTX_K,
                             weights=recency_weight(s[tr_m], CTX_HALF_LIFE, ref_season=ref))
        art = F.fit_artifacts(tr, ctx=ctx, contam_axes=CONTAM_AXES)
        art["asof"] = asofdelta.fit_asof_axes(tr, ASOF_AXES, ASOF_STRENGTH,
                                              ct=art.get("contam"))
        adj, _ = F.logit_adjust(d, art)
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-(z + adj))), yv.mean())
        r = yv - p0

        tab = fit_bat(tr, ball[tr_m], rev[tr_m], clean_all[tr_m])
        feat, cov = apply_bat(d, tab)
        dsh = d.copy()
        dsh["batter_id"] = rng.permutation(d["batter_id"].to_numpy())
        fsh, _ = apply_bat(dsh, tab)

        bid = d["batter_id"].to_numpy(np.int64)
        print()
        print(f"fold {fold}  기준 BSS {bss(yv, p0):.1f}  행 {len(d):,}")
        print(f"  {'피처':<14}{'행 t':>8}{'클러스터 t':>11}{'셔플 t':>9}"
              f"{'커버리지':>9}{'배포피처 최대상관':>18}")
        X = F.build_features(d, art)
        for k, v in feat.items():
            _, t = resid_t(r, v)
            ct = clust_t(r, v, bid)
            _, tc = resid_t(r, fsh[k])
            cc, cn = 0.0, ""
            for c in X.columns:
                xv = X[c].to_numpy(np.float64)
                m = np.isfinite(xv)
                if m.sum() < 1000 or np.nanstd(xv[m]) == 0:
                    continue
                cr = abs(float(np.corrcoef(v[m], xv[m])[0, 1]))
                if cr > cc:
                    cc, cn = cr, c
            mark = "  ✓" if (abs(ct) > 2 and abs(t) > 2 * abs(tc) and cc < 0.5) else ""
            print(f"  {k:<14}{t:>8.2f}{ct:>11.2f}{tc:>9.2f}"
                  f"{cov[k]:>8.1%} {cc:>10.3f} {cn[:16]:<16}{mark}")


if __name__ == "__main__":
    main()
