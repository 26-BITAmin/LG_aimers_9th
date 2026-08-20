"""배포 피처 전수 잔차 스크린 — "지금 들어가는 피처 중 효과가 없는 것"을 찾는다.

질문 (사용자)
    지금 모델에 들어가는 feature 중, 잔차를 봤을 때 효과가 없는 것은?

방법
    배포(s24) foldcache의 홀드아웃 예측으로 fold별 잔차 r = y − p 를 만들고,
    피처 컬럼 하나하나를 **축**으로 보고 세 가지를 잰다:

        rz     잔차 구조: 축 버킷별 잔차 평균의 χ² → z 정규화.
               시간 홀드아웃이므로, 학습 시즌에서 배운 패턴이 검증 시즌에
               안 맞으면 **들어간 피처라도** 여기서 다시 살아난다 (§7.77의
               game_month가 그랬다).
        t      잔차 선형 t (ctxsplit.resid_t). 부호의 fold 간 안정성도 본다.
        yz     타깃 구조: (y − ȳ)의 같은 χ²z. 축이 애초에 신호를 갖는가.
        ycorr  타깃 곡선의 fold 간 평균 상관. **낮으면 시즌 국한 패턴** —
               달력형 드리프트 채널이고, 이게 진짜 드랍 후보의 프로파일이다.
        zR²    블렌드 로짓이 이 축에서 움직이는 분산 비율. 모델이 축을
               얼마나 타는지 (상관 피처 경유 포함 — 인과 아님, 서술용).

영가설 눈금 (§7.33 · 메모리 "죽은 축을 영가설로")
    CTL_dow      드랍된 game_dayofweek — 상황 입도의 무의미 축
    CTL_month    드랍된 game_month — 시즌 내 구조는 있지만 전이 안 되는 축
    CTL_pidhash  hash(pitcher_id)%10 — 투수 입도의 무의미 축.
                 투수 단위 클러스터 상관이 χ²를 얼마나 부풀리는지의 눈금.

행 필터
    전 fold에서 R행만 (residbase 관례). 2019–22 F 검증 라벨은 오염이라
    fold2021·2022를 쓰려면 어차피 빼야 하고, fold 간 곡선 비교는 같은
    모집단이어야 한다.

⚠️ 잔차는 확률 스케일(y − p), OLS/χ² 는 등분산 근사 (p가 .4~.6이라 괜찮다
   — residbase 헤더 참조). 투수 레벨 축의 χ²는 클러스터 상관으로 부풀 수
   있으므로 CTL_pidhash 와 비교해서 읽는다.

실행:  LGA_FOLD_KEY=5488ef1b18 \
       .venv/Scripts/python.exe -u src/featnull.py 2>&1 | tee experiments/featnull.log
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import features as F
import foldcache
from ctxsplit import resid_t
from rookie import deployed_pred          # ← LGA_FOLD_KEY 핀도 이 import가 건다
from config import (CONTAM_AXES, DROP_FEATURES, MLP_ALPHA, RECENCY_HALF_LIFE,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from validation import bss

FOLDS = [2021, 2022, 2023, 2024]
KQ = 10          # 연속 축 분위 버킷 수
CAT_MAX = 12     # 유니크 ≤ 이면 범주로 취급
MIN_N = 200      # fold 간 곡선 상관에 넣는 버킷의 최소 표본


def codes(x):
    """축 → (라벨, 정수코드). 범주는 값 그대로, 연속은 KQ분위, NaN은 별도 버킷."""
    x = np.asarray(x, dtype=np.float64)
    nan = ~np.isfinite(x)
    xs = x[~nan]
    u = np.unique(xs)
    if len(u) <= CAT_MAX:
        c = np.searchsorted(u, np.where(nan, u[0], x))
        labels = [f"{v:g}" for v in u]
    else:
        e = np.unique(np.quantile(xs, np.arange(1, KQ) / KQ))
        c = np.searchsorted(e, np.where(nan, e[0], x))
        labels = [f"q{i}" for i in range(len(e) + 1)]
    c = c.astype(np.int64)
    if nan.any():
        c[nan] = len(labels)
        labels = labels + ["nan"]
    return labels, c


def curve(v, c, B):
    n = np.bincount(c, minlength=B)
    m = np.bincount(c, weights=v, minlength=B) / np.maximum(n, 1)
    return n, m


def chi2z(v, c, B):
    """버킷 평균의 χ² → z. v는 이미 de-mean 상태여야 한다."""
    n, m = curve(v, c, B)
    s2 = v.var()
    dof = int((n > 0).sum()) - 1
    if dof <= 0 or s2 <= 0:
        return 0.0
    return float(((n * m * m).sum() / s2 - dof) / np.sqrt(2 * dof))


def fold_features(df, s, fold, va):
    """foldcache.fold_data와 같은 아티팩트 구성으로 **검증행만** 피처를 만든다
    (fold_data는 학습행 1.4M도 만들기 때문에 여기선 안 쓴다)."""
    tr = df[s < fold]
    w = recency_weight(tr["season"].to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr["season"].max()))
    art = F.fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                          k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)
    X = F.build_features(va, art)
    if USE_PLAYER_IDS:
        for c in ("pitcher_id", "batter_id"):
            X[c] = va[c].to_numpy(dtype=np.float32)
    return X.drop(columns=[c for c in DROP_FEATURES if c in X.columns])


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    res = {}                             # name -> fold -> dict(통계 + 곡선)

    for fold in FOLDS:
        va, y, p = deployed_pred(df, s, fold, cached)
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        X = fold_features(df, s, fold, va)
        m = va["game_type"].astype("object").to_numpy() != "F"

        r = (y - p)[m];  r -= r.mean()
        yc = y[m] - y[m].mean()
        zc = z[m] - z[m].mean()
        zvar = zc.var()
        print(f"fold {fold}  n(R)={m.sum():,}  BSS(전체)={bss(y, p):.1f}"
              f"  피처 {X.shape[1]}개  ({time.time()-t0:.0f}s)")

        axes = {c: X[c].to_numpy(dtype=np.float64)[m] for c in X.columns}
        axes["CTL_dow"] = va["game_dayofweek"].to_numpy(dtype=np.float64)[m]
        axes["CTL_month"] = va["game_month"].to_numpy(dtype=np.float64)[m]
        axes["CTL_pidhash"] = ((va["pitcher_id"].to_numpy(dtype=np.int64)
                                * 2654435761) % (2 ** 32) % 10).astype(np.float64)[m]

        for name, x in axes.items():
            labels, c = codes(x)
            B = len(labels)
            n, my = curve(yc, c, B)
            _, mz = curve(zc, c, B)
            fin = np.isfinite(x)
            _, t = resid_t(r[fin], x[fin]) if fin.sum() > 10 and x[fin].std() > 0 else (0., 0.)
            res.setdefault(name, {})[fold] = dict(
                rz=chi2z(r, c, B), t=t, yz=chi2z(yc, c, B),
                zr2=float((n * mz * mz).sum() / len(r) / zvar * 100) if zvar > 0 else 0.0,
                labels=labels, n=n, my=my)

    # ── fold 간 타깃 곡선 상관 (라벨 교집합, 양쪽 n ≥ MIN_N) ──────────
    def ycorr(name):
        cs = []
        fs = [f for f in FOLDS if f in res[name]]
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                a, b = res[name][fs[i]], res[name][fs[j]]
                da = {l: (v, k) for l, v, k in zip(a["labels"], a["my"], a["n"]) if k >= MIN_N}
                shared = [l for l, v, k in zip(b["labels"], b["my"], b["n"])
                          if k >= MIN_N and l in da]
                if len(shared) < 3:
                    continue
                va_ = np.array([da[l][0] for l in shared])
                vb_ = np.array([v for l, v, k in zip(b["labels"], b["my"], b["n"])
                                if k >= MIN_N and l in da])
                if va_.std() > 0 and vb_.std() > 0:
                    cs.append(float(np.corrcoef(va_, vb_)[0, 1]))
        return float(np.mean(cs)) if cs else float("nan")

    print("\n" + "=" * 118)
    print(f"{'피처':<34}{'잔차 rz 21/22/23/24':>26}{'t23':>7}{'t24':>7}"
          f"{'yz̄':>8}{'ycorr':>7}{'zR²%':>7}")
    rows = []
    for name in res:
        st = res[name]
        rzs = [st[f]["rz"] if f in st else float("nan") for f in FOLDS]
        yzm = float(np.nanmean([st[f]["yz"] for f in FOLDS if f in st]))
        zr2 = float(np.nanmean([st[f]["zr2"] for f in FOLDS if f in st]))
        t23 = st.get(2023, {}).get("t", float("nan"))
        t24 = st.get(2024, {}).get("t", float("nan"))
        rows.append((name, rzs, t23, t24, yzm, ycorr(name), zr2))
    rows.sort(key=lambda r: -max(abs(v) for v in r[1] if np.isfinite(v)))
    for name, rzs, t23, t24, yzm, yco, zr2 in rows:
        print(f"{name:<34}" + "".join(f"{v:>6.1f}" for v in rzs)
              + f"  {t23:>+6.1f}{t24:>+7.1f}{yzm:>8.1f}{yco:>7.2f}{zr2:>7.2f}")

    # ── 판정 요약 ──────────────────────────────────────────────────
    # ⚠️ 눈금은 CTL_dow(상황 입도의 죽은 축)만 쓴다. CTL_pidhash는 같은 투수가
    #    해마다 같은 버킷에 남아 **투수 실력의 지속성**을 실어 나른다 — yz·ycorr가
    #    커도 그게 "무의미 축의 노이즈"가 아니라, 투수 입도 축의 χ²가 클러스터
    #    상관으로 얼마나 부풀 수 있는지의 눈금이다 (판정이 아니라 각주로 쓴다).
    dow = res["CTL_dow"]
    rz_null = max(abs(dow[f]["rz"]) for f in dow)
    yz_null = float(np.mean([dow[f]["yz"] for f in dow]))
    ph = res["CTL_pidhash"]
    print(f"\n눈금(CTL_dow): |rz| ≤ {rz_null:.1f}   yz̄ ≈ {yz_null:.1f}"
          f"   (투수 입도 각주 CTL_pidhash: |rz| ≤ "
          f"{max(abs(ph[f]['rz']) for f in ph):.1f})")

    dead, drift, live = [], [], []
    for name, rzs, t23, t24, yzm, yco, zr2 in rows:
        if name.startswith("CTL_"):
            continue
        mrz = max(abs(v) for v in rzs if np.isfinite(v))
        if yzm <= yz_null * 1.5 and mrz <= rz_null * 1.5:
            dead.append((name, yzm, zr2))
        elif np.isfinite(yco) and yco < 0.2 and yzm >= yz_null * 2:
            drift.append((name, yco, yzm))
        if abs(t23) >= 2 and abs(t24) >= 2 and t23 * t24 > 0:
            live.append((name, t23, t24))

    print(f"\nA. 축 자체가 죽어 있음 (yz̄ ≤ 눈금×1.5, 잔차도 평평) — {len(dead)}개")
    for name, yzm, zr2 in sorted(dead, key=lambda r: r[1]):
        print(f"   {name:<34} yz̄ {yzm:>5.1f}   zR² {zr2:.2f}%")
    print(f"\nB. 신호는 있는데 시즌 간 전이 안 됨 (ycorr < 0.2) — {len(drift)}개")
    for name, yco, yzm in sorted(drift, key=lambda r: r[1]):
        print(f"   {name:<34} ycorr {yco:>+5.2f}   yz̄ {yzm:>5.1f}")
    print(f"\nC. 잔차에 남은 구조 (부호 안정 |t|≥2 양쪽) — {len(live)}개")
    for name, t23, t24 in live:
        print(f"   {name:<34} t23 {t23:+.1f}   t24 {t24:+.1f}")
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
