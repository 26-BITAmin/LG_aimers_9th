"""pre-2023 F 오염이 **현재 모델의 잔차**에 남아 있는가

배경 (§4 + §7.40의 합류)
    §4는 `game_type='F'`가 2022→2023에 붕괴(.709→.473)하는 것을 보고 컬럼을 지웠다.
    §7.40은 투구별 실패 모드 라벨을 복원했다. 둘을 겹쳐보니 붕괴분이 **전부 반대 축**이다:

        F reverse:  .125 .151 .092 **.048** → **.300** .288     (몰림·벗어남은 평평)
        R reverse:  .188 .215 .262  .259     .251  .243        (평평)

    즉 2019~2022 F의 104,903행은 "포수 요구 반대" 기준이 걸리지 않은 상태로 라벨링됐고,
    운영측은 그 이력을 **재계산하지 않았다** (재계산했다면 asof 증분에서 .25가 나와야 한다).

왜 2025까지 가나
    `asof_pitcher_n`은 시즌마다 리셋되지 않는 **커리어 카운터**이고 F도 같이 센다
    (검증: 한 투수의 R/F 혼합 행을 n으로 정렬하면 증분이 전부 정확히 1).
    test 2025도 그 카운터를 이어받는다 (pitcher 21813: train 최대 3084 → test 3465).
    → 2025 행의 `asof_pitcher_reverse_rate` 분자에 누락분이 그대로 들어 있다.

    2024 투구 기준 왜곡 크기: Δreverse q90 +.034 / q99 +.105 (투수간 std .0445).

이 파일이 묻는 것
    같은 왜곡이 **train에도 있으므로** 이건 train/serve skew가 아니다. 그래서
    "고치면 이득"이 자동으로 따라오지 않는다. 먼저 물어야 할 것은 하나다 —
    **모델의 잔차가 오염도와 상관이 있는가.** 없으면 세 갈래(행 제거·피처 보정·
    오염도 피처) 전부 근거 없이 닫힌다.

    ⚠️ 오염도는 **경력 구성과 상관이 있다.** t가 크게 나와도 "옛 이력 비율"이라는
       일반 축을 잡은 것일 수 있다. 그래서 같은 모양의 **대조 축**을 함께 잰다:
         · share_preR   : pre-2023 R 비율    (같은 시기, 정의 변경 없음)
         · share_postF  : 2023+ F 비율        (같은 리그, 정의 정상)
         · log n        : 커리어 표본 수      (경력 그 자체)
       오염 축의 t가 대조들과 비슷하면 그건 오염의 증거가 아니다.
       (메모리: "무의미한 버킷이 서브그룹 튜닝의 영가설")

실행:  .venv/Scripts/python.exe -u src/fcontam.py 2>&1 | tee experiments/fcontam.log
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
import foldcache
from config import CTX_AXES, CTX_K, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from ctxsplit import resid_t
from data import load_csv
from features import logit_adjust
from modelabel import recover
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]


def pitcher_table(hist, rev, suc):
    """fold의 학습 구간(hist)에서 투수별 오염 지표를 만든다.

    train에서 fit → valid/test에 조회하는 형태라 규정상 안전하다
    (test 행끼리 아무것도 공유하지 않는다).
    """
    d = pd.DataFrame({
        "pid": hist["pitcher_id"].to_numpy(dtype=np.int64),
        "season": hist["season"].to_numpy(dtype=np.int64),
        "gtype": hist["game_type"].to_numpy(),
        "rev": rev, "suc": suc,
    })
    dirty = (d["season"] <= 2022) & (d["gtype"] == "F")
    pre_r = (d["season"] <= 2022) & (d["gtype"] == "R")
    post_f = (d["season"] >= 2023) & (d["gtype"] == "F")

    g = d.groupby("pid")
    tot = g.agg(n=("rev", "size"), rev=("rev", "sum"))
    bad = d[dirty].groupby("pid").agg(n=("rev", "size"), rev=("rev", "sum")) \
                  .reindex(tot.index).fillna(0.0)

    out = pd.DataFrame(index=tot.index)
    out["n"] = tot.n
    out["n_dirty"] = bad.n
    out["share_preF"] = bad.n / tot.n
    out["share_preR"] = d[pre_r].groupby("pid").size().reindex(tot.index).fillna(0) / tot.n
    out["share_postF"] = d[post_f].groupby("pid").size().reindex(tot.index).fillna(0) / tot.n
    # 왜곡의 직접 추정: 오염 구간을 뺐을 때 커리어 reverse_rate가 얼마나 오르나
    clean_n = (tot.n - bad.n).replace(0, np.nan)
    out["bias_rev"] = ((tot.rev - bad.rev) / clean_n) - (tot.rev / tot.n)
    out["bias_rev"] = out["bias_rev"].fillna(0.0)
    return out


AXES = ["bias_rev", "share_preF", "share_preR", "share_postF", "logn"]
IS_CONTROL = {"bias_rev": False, "share_preF": False,
              "share_preR": True, "share_postF": True, "logn": True}


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    lab, _, _ = recover(df)
    rev = np.round(lab["reverse"])
    suc = np.round(lab["success"])
    ok = ~(np.isnan(rev) | np.isnan(suc))
    rev = np.where(ok, rev, 0.0)
    suc = np.where(ok, suc, 0.0)
    print(f"train {df.shape}   모드 라벨 복원 {ok.mean():.1%}")
    print(f"기준선 = 배포 상태 (블렌드 α={MLP_ALPHA} + δ 축 {len(CTX_AXES)}개)\n")

    cached = foldcache.load(FOLDS, verbose=False)
    slopes = {a: {} for a in AXES}

    for fold in FOLDS:
        tr, va = s < fold, s == fold
        d = df[va]
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        adj, _ = logit_adjust(d, {"ctx": ctx})
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-(z + adj))), yv.mean())
        base = bss(yv, p0)
        r = yv - p0                                   # 배포 상태의 잔차

        tab = pitcher_table(df[tr], rev[tr], suc[tr])
        tab["logn"] = np.log1p(tab["n"])
        pid = d["pitcher_id"].to_numpy(dtype=np.int64)
        hit = np.isin(pid, tab.index.to_numpy())

        print("=" * 92)
        print(f"fold {fold}   train {tr.sum():,} → valid {va.sum():,}"
              f"   기준 BSS {base:.1f}   투수 조회 적중 {hit.mean():.1%}")
        print("=" * 92)
        print(f"  {'축':<14}{'':<6}{'평균':>9}{'q90':>9}{'기울기':>11}{'잔차 t':>9}")

        vals = {}
        for a in AXES:
            v = tab[a].reindex(pid).to_numpy(dtype=np.float64)
            v = np.where(np.isnan(v), np.nanmean(v), v)     # 신인 = 오염 없음/평균
            vals[a] = v
            sl, t = resid_t(r, v)
            slopes[a][fold] = sl
            tag = "대조" if IS_CONTROL[a] else "가설"
            print(f"  {a:<14}{tag:<6}{v.mean():>9.4f}{np.quantile(v,.9):>9.4f}"
                  f"{sl:>11.4f}{t:>9.2f}{'  ✓' if abs(t) > 2 else ''}")

        # 실행 가능한 형태 — 오염도 구간별 잔차 평균 (양수 = 모델이 과소예측)
        sh = vals["share_preF"]
        edges = [0.0, 0.01, 0.05, 0.20, 1.01]
        print(f"\n  {'share_preF 구간':<18}{'행 수':>10}{'실제':>9}{'예측':>9}"
              f"{'잔차평균':>11}{'SE':>9}{'t':>8}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (sh >= lo) & (sh < hi) if lo > 0 else (sh < hi)
            if m.sum() < 100:
                continue
            rr = r[m]
            se = rr.std(ddof=1) / np.sqrt(m.sum())
            print(f"  [{lo:.2f}, {hi:.2f}){'':<6}{m.sum():>10,}{yv[m].mean():>9.4f}"
                  f"{p0[m].mean():>9.4f}{rr.mean():>+11.5f}{se:>9.5f}"
                  f"{rr.mean()/se:>8.2f}")
        print()

    print("=" * 92)
    print("판정")
    print("=" * 92)
    for a in AXES:
        v = [slopes[a][f] for f in FOLDS]
        same = "일치" if (v[0] > 0) == (v[1] > 0) else "**불일치**"
        tag = "대조" if IS_CONTROL[a] else "가설"
        print(f"  {a:<14}{tag:<6}기울기 {v[0]:+.4f} / {v[1]:+.4f}   부호 {same}")
    print("\n  · 가설 축의 |t|가 대조 축들과 비슷하면 → 오염이 아니라 '경력' 축이다. 닫는다.")
    print("  · 가설 축만 크고 두 fold 부호가 같으면 → 다음 단계(보정 δ)로 간다.")
    print("  · 어느 쪽이든 t는 '모델이 모른다'만 말한다. 2025 전이는 별개다 (§7.9).")


BETAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]
SHRINK_K = [0, 20, 50, 100, 200, 500]


def shrink_bias(bias_rev, n, n_dirty, k):
    """오염 편향 추정치를 표본 수로 수축한다. ← **여기가 비어 있다**

    왜 필요한가
        `bias_rev`는 투수별 추정량이다. 20구 던진 투수가 그 20구를 전부 2019 F에서
        던졌다면 `share=1.0`이고 `bias_rev`가 .2 넘게 나온다 — 실제 왜곡이 아니라
        표본이 없어서 나온 값이다. 3단계 β 스윕은 그 꼬리에 지배당한다.
        (메모리: "비율 조건에는 노이즈 게이트가 먼저다" — 9개 중 7개가 해석 불가였다.)

    선택지 (배타적이지 않다)
        · 경험적 베이즈 수축   bias * n_dirty / (n_dirty + k)
            – 오염 구간의 표본 수로 가른다. 이 저장소가 `asof_*`에 쓰는 방식과 동일
              (`features.SMOOTH_SPEC`, k=200). 부드럽고 임계값이 없다.
        · 하드 게이트         n_dirty < k 이면 0
            – 해석이 쉽고 꼬리를 확실히 죽인다. 대신 k 근처에서 계단이 생긴다.
        · 상한(winsorize)     np.clip(bias, 0, c)
            – 꼬리만 자른다. 저표본 투수의 **부호**는 살려둔다.

    트레이드오프
        세게 수축할수록 δ는 안전해지지만, 정작 왜곡이 가장 큰 집단이
        **2019~2022를 2군에서 보낸 저표본 투수**다 — 표적과 노이즈원이 같은 사람들이다.
        너무 세게 수축하면 고칠 것이 남지 않고, 안 하면 노이즈를 증폭한다.

    인자
        bias_rev : (P,) 투수별 편향 추정 (오염 구간 제외 시 reverse_rate 상승분)
        n        : (P,) 그 투수의 학습 구간 총 투구 수
        n_dirty  : (P,) 그중 pre-2023 F 투구 수
        k        : 수축 강도 (0 = 수축 없음)
    반환
        (P,) 수축된 편향. 이 값이 δ = −(x − mean) 으로 들어간다.

    경험적 베이즈를 골랐다
        저장소가 `asof_*` 비율에 이미 쓰는 방식과 같고(`features.SMOOTH_SPEC`),
        임계값이 없어 k를 스윕해도 계단이 안 생긴다. 하드 게이트는 k 근처에서
        점수가 튀어 스윕 자체를 못 믿게 된다.

        수축의 분모를 `n`(총 투구)이 아니라 **`n_dirty`(오염 구간 투구)**로 둔다.
        불확실한 것은 "이 투수를 아는가"가 아니라 **"누락분을 잴 만큼 오염 구간을
        봤는가"**다. 통산 3000구를 던졌어도 오염 구간이 15구면 그 15구에서 잰
        누락분은 못 믿는다.
    """
    nd = np.asarray(n_dirty, dtype=np.float64)
    if k <= 0:
        return np.asarray(bias_rev, dtype=np.float64)
    return np.asarray(bias_rev, dtype=np.float64) * nd / (nd + k)


def stage2():
    """1단계의 두 반론을 처리한다.

    반론 A — **fold2023은 대조가 성립하지 않는다.** train이 ≤2022뿐이라
        share_preF + share_preR ≡ 1 이고, 실제로 t가 ∓26.41로 완전 대칭이었다.
        같은 수를 두 번 읽은 것이지 대조가 아니다. → fold2024만 대조로 읽는다.

    반론 B — **'퓨처스 이력이 많은 투수' 축일 수 있다.** 2군 상대 성적은 1군에서
        과대평가되므로 라벨 정의와 무관하게 같은 부호가 나온다. 이걸 가르는 것이
        `share_postF`다 — 같은 퓨처스인데 라벨은 정상인 구간. 편회귀로 분리한다.

    그 다음, 통과하면 배포 기계(사후 로짓 δ + β 스윕)로 실제 이득을 잰다.
    """
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    lab, _, _ = recover(df)
    rev, suc = np.round(lab["reverse"]), np.round(lab["success"])
    ok = ~(np.isnan(rev) | np.isnan(suc))
    rev, suc = np.where(ok, rev, 0.0), np.where(ok, suc, 0.0)
    cached = foldcache.load(FOLDS, verbose=False)

    print("\n" + "=" * 92)
    print("2단계 A — 편회귀: 오염 축 vs '퓨처스 이력' 축 (fold2024만 유효)")
    print("=" * 92)
    gains = {}

    for fold in FOLDS:
        tr, va = s < fold, s == fold
        d = df[va]
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        adj, _ = logit_adjust(d, {"ctx": ctx})
        z0 = z + adj
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean())
        base = bss(yv, p0)
        r = yv - p0

        tab = pitcher_table(df[tr], rev[tr], suc[tr])
        tab["logn"] = np.log1p(tab["n"])
        pid = d["pitcher_id"].to_numpy(dtype=np.int64)

        def col(a):
            v = tab[a].reindex(pid).to_numpy(dtype=np.float64)
            return np.where(np.isnan(v), np.nanmean(v), v)

        names = ["share_preF", "share_postF", "logn"]
        X = np.column_stack([col(a) for a in names] + [np.ones(len(pid))])
        if fold == 2023:
            X = np.delete(X, 1, axis=1)          # post-2023 F가 없어 상수열이다
            names2 = ["share_preF", "logn"]
        else:
            names2 = names
        b, *_ = np.linalg.lstsq(X, r, rcond=None)
        res = r - X @ b
        dof = len(r) - X.shape[1]
        cov = np.linalg.inv(X.T @ X) * (res @ res) / dof
        se = np.sqrt(np.diag(cov))
        print(f"\n  fold {fold}   기준 BSS {base:.1f}")
        for i, nm in enumerate(names2):
            print(f"    {nm:<14} 편기울기 {b[i]:>+9.4f}   SE {se[i]:.4f}"
                  f"   t {b[i]/se[i]:>+7.2f}")

        # ── 2단계 B: 배포 기계로 실제 이득 (δ = −(오염 편향), k × β 격자) ──
        raw = tab["bias_rev"].to_numpy(dtype=np.float64)
        for k in SHRINK_K:
            sb = pd.Series(shrink_bias(raw, tab["n"].to_numpy(),
                                       tab["n_dirty"].to_numpy(), k), index=tab.index)
            v = sb.reindex(pid).to_numpy(dtype=np.float64)
            v = np.where(np.isnan(v), np.nanmean(v), v)
            delta = -(v - v.mean())              # 오염이 클수록 로짓을 내린다
            g = []
            for beta in BETAS:
                p = shift_to_prior(1.0 / (1.0 + np.exp(-(z0 + beta * delta))), yv.mean())
                g.append(bss(yv, p) - base)
            gains.setdefault(k, {})[fold] = g
        # 대조 — 기각된 거친 대리변수(share_preF)를 같은 격자에 올려둔다
        v = col("share_preF")
        delta = -(v - v.mean())
        gains.setdefault("share_preF", {})[fold] = [
            bss(yv, shift_to_prior(1.0 / (1.0 + np.exp(-(z0 + b * delta))), yv.mean())) - base
            for b in BETAS]

    print("\n" + "=" * 92)
    print("2단계 B — δ = −(수축된 bias_rev − 평균), k × β 격자   (BSS 이득)")
    print("=" * 92)
    print(f"  {'k':<12}{'fold':>6}" + "".join(f"{b:>8.2f}" for b in BETAS)
          + f"{'최적β':>8}{'최고':>8}")
    for k in list(SHRINK_K) + ["share_preF"]:
        for fold in FOLDS:
            g = gains[k][fold]
            i = int(np.argmax(g))
            lbl = f"k={k}" if k != "share_preF" else "share_preF"
            print(f"  {lbl:<12}{fold:>6}" + "".join(f"{x:>8.1f}" for x in g)
                  + f"{BETAS[i]:>8.2f}{g[i]:>8.1f}")

    print("\n" + "=" * 92)
    print("판정 — β별로 두 fold를 나란히 (양수 여부 · 크기비)")
    print("=" * 92)
    print(f"  {'k':<12}" + "".join(f"{b:>10.2f}" for b in BETAS[1:]))
    for k in SHRINK_K:
        cells = []
        for j in range(1, len(BETAS)):
            a, b = gains[k][2023][j], gains[k][2024][j]
            if a <= 0 or b <= 0:
                cells.append("      음수")
            else:
                cells.append(f"{max(a,b)/min(a,b):>9.1f}배")
        print(f"  k={k:<10}" + "".join(cells))
    print("\n  조건: 두 fold 양수 · 크기비 ≤3배 (§7.9의 Trackman 6배가 LB에서 뒤집혔다)")


OP_K, OP_BETA = 50, 0.5          # 운영점 — 근거는 stage3 도크스트링


def stage3():
    """운영점 하나를 고정하고 **쌍체 SE**를 붙인다.

    왜 β=0.5, k=50 인가
        · 수축은 실질적으로 무의미했다 (k=0 → 500에서 fold2024 20.5 → 14.8,
          크기비는 오히려 3.7 → 4.4배로 **악화**). 꼬리 지배가 아니었다.
          k=50은 β=2.0 같은 사고 구간의 손실만 줄여주는 보험이라 싸게 남긴다.
        · β는 fold2024 최적(0.8)보다 **한 칸 낮게** 잡는다. 2025는 오염이 가장
          옅은 시점이므로 실효 β가 홀드아웃보다 크게 나올 이유가 없고,
          과보정은 β=2.0에서 보듯 부호가 뒤집힌다. 하방 비대칭 → 보수적으로.
        · δ는 투수별 편향 추정치 자체를 쓰므로 **2025에서 오염이 옅어지면
          δ가 자동으로 작아진다.** β를 시즌에 맞춰 다시 고를 필요가 없다.

    무엇을 확인하나
        ① 쌍체 SE — 행마다 (보정 전 Brier − 보정 후 Brier)의 평균과 t
           (메모리: "효과 크기에는 쌍체 SE를 붙일 것")
        ② 표적 집단에서 실제로 miscalibration이 줄었는가 (구간별 잔차 before/after)
        ③ 부작용 — 오염 없는 투수(δ>0으로 밀려 올라가는 쪽)가 손해 보지 않는가
    """
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    lab, _, _ = recover(df)
    rev, suc = np.round(lab["reverse"]), np.round(lab["success"])
    ok = ~(np.isnan(rev) | np.isnan(suc))
    rev, suc = np.where(ok, rev, 0.0), np.where(ok, suc, 0.0)
    cached = foldcache.load(FOLDS, verbose=False)

    print("\n" + "=" * 92)
    print(f"3단계 — 운영점 k={OP_K}, β={OP_BETA} 고정. 쌍체 SE와 부작용 점검")
    print("=" * 92)

    for fold in FOLDS:
        tr, va = s < fold, s == fold
        d = df[va]
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        adj, _ = logit_adjust(d, {"ctx": ctx})
        z0 = z + adj

        tab = pitcher_table(df[tr], rev[tr], suc[tr])
        sb = pd.Series(shrink_bias(tab["bias_rev"].to_numpy(), tab["n"].to_numpy(),
                                   tab["n_dirty"].to_numpy(), OP_K), index=tab.index)
        pid = d["pitcher_id"].to_numpy(dtype=np.int64)
        v = sb.reindex(pid).to_numpy(dtype=np.float64)
        v = np.where(np.isnan(v), np.nanmean(v), v)
        delta = -(v - v.mean())

        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean())
        p1 = shift_to_prior(1.0 / (1.0 + np.exp(-(z0 + OP_BETA * delta))), yv.mean())
        b0, b1 = bss(yv, p0), bss(yv, p1)

        # ① 쌍체 — 행별 Brier 차이
        e0, e1 = (p0 - yv) ** 2, (p1 - yv) ** 2
        dd = e0 - e1                                  # 양수 = 보정이 이득
        se = dd.std(ddof=1) / np.sqrt(len(dd))
        # BSS 단위로 환산 (BSS = 1e5 * (1 - BS/BSref))
        bsref = yv.mean() * (1 - yv.mean())
        print(f"\n  fold {fold}   BSS {b0:.1f} → {b1:.1f}  ({b1-b0:+.1f})")
        print(f"    쌍체 ΔBrier {dd.mean():+.3e} ± {se:.3e}   t {dd.mean()/se:+.2f}"
              f"   (BSS 환산 {1e5*dd.mean()/bsref:+.1f} ± {1e5*se/bsref:.1f})")

        # ② 표적 집단의 miscalibration
        sh = tab["share_preF"].reindex(pid).to_numpy(dtype=np.float64)
        sh = np.where(np.isnan(sh), 0.0, sh)
        print(f"    {'share_preF':<14}{'행 수':>9}{'실제':>8}{'전':>8}{'후':>8}"
              f"{'잔차 전':>10}{'잔차 후':>10}")
        for lo, hi in [(0.0, 0.01), (0.01, 0.05), (0.05, 0.20), (0.20, 1.01)]:
            m = (sh >= lo) & (sh < hi) if lo > 0 else (sh < hi)
            if m.sum() < 100:
                continue
            print(f"    [{lo:.2f},{hi:.2f}){'':<5}{m.sum():>9,}{yv[m].mean():>8.4f}"
                  f"{p0[m].mean():>8.4f}{p1[m].mean():>8.4f}"
                  f"{(yv[m]-p0[m]).mean():>+10.5f}{(yv[m]-p1[m]).mean():>+10.5f}")

        # ③ 부작용 — δ 부호별 이득 분해
        for nm, m in [("δ<0 (오염, 내림)", delta < 0), ("δ>0 (깨끗, 올림)", delta > 0)]:
            print(f"    {nm:<20} 행 {m.sum():>8,}   ΔBrier 기여"
                  f" {1e5*dd[m].sum()/len(dd)/bsref:>+8.1f} BSS")


def stage4():
    """2025 이득을 외삽한다 — δ의 산포가 fold마다 줄어드는 속도로.

    이 보정의 이득은 δ의 **크기**에서 나온다. Brier 개선은 편향의 제곱에
    비례하므로 이득 ≈ c · std(δ)² 이어야 한다. 두 fold가 그 관계를 만족하면
    (70.2 / 15.8 = 4.44 ≈ std비²), 같은 식으로 2025를 예측할 수 있다.

    δ 산포가 fold마다 주는 이유는 둘 다 기전이다:
      · 오염 구간이 과거로 밀려나면서 분모(커리어 투구 수)가 매년 커진다
      · 학습 구간에 깨끗한 시즌이 하나씩 늘어난다 (fold2023: 0개, 2024: 1개, 2025: 2개)
    """
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    lab, _, _ = recover(df)
    rev, suc = np.round(lab["reverse"]), np.round(lab["success"])
    ok = ~(np.isnan(rev) | np.isnan(suc))
    rev, suc = np.where(ok, rev, 0.0), np.where(ok, suc, 0.0)

    def dstd(hist_max, row_season):
        """train ≤ hist_max 로 만든 표를 row_season 행에 적용했을 때 δ의 산포."""
        h = s <= hist_max
        tab = pitcher_table(df[h], rev[h], suc[h])
        sb = pd.Series(shrink_bias(tab["bias_rev"].to_numpy(), tab["n"].to_numpy(),
                                   tab["n_dirty"].to_numpy(), OP_K), index=tab.index)
        pid = df[s == row_season]["pitcher_id"].to_numpy(dtype=np.int64)
        v = sb.reindex(pid).to_numpy(dtype=np.float64)
        v = np.where(np.isnan(v), np.nanmean(v), v)
        return float(v.std()), float(np.mean(v > 0.005))

    print("\n" + "=" * 92)
    print("4단계 — 2025 외삽 (δ 산포가 매년 얼마나 줄어드는가)")
    print("=" * 92)
    rows = [("fold2023", 2022, 2023, 70.2), ("fold2024", 2023, 2024, 15.8),
            ("2025 배포(대리)", 2024, 2024, None)]
    print(f"  {'':<16}{'학습 이력':>10}{'적용 행':>9}{'std(δ)':>10}"
          f"{'|δ|>.005 비율':>14}{'실측 이득':>10}")
    out = []
    for nm, hmax, rs, g in rows:
        sd, frac = dstd(hmax, rs)
        out.append((nm, sd, g))
        print(f"  {nm:<16}{'≤'+str(hmax):>10}{rs:>9}{sd:>10.5f}{frac:>14.1%}"
              f"{('' if g is None else f'{g:>10.1f}')}")

    (_, sd23, g23), (_, sd24, g24), (_, sd25, _) = out
    print(f"\n  검산 — 이득이 std(δ)² 에 비례하는가:")
    print(f"    실측 이득비 {g23/g24:.2f}배   std비² {(sd23/sd24)**2:.2f}배")
    pred = g24 * (sd25 / sd24) ** 2
    print(f"\n  → 2025 예측 이득 = {g24:.1f} × ({sd25:.5f}/{sd24:.5f})²"
          f" = **{pred:.1f} BSS**")
    print("    (홀드아웃 과대 성향은 별도다 — 정보 추가 계열은 LB에서 더 작게 나온다)")


if __name__ == "__main__":
    main()
    stage2()
    stage3()
    stage4()
