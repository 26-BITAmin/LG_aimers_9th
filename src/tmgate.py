"""Trackman **상황 게이트** 1단계 — 부호 반전이 상황 버킷 안에서 사라지나

왜 묻나
    §7.39가 Trackman 을 닫은 근거는 **관계의 부호가 fold 마다 뒤집힌다**였다.
    `tm_relpoint_disp` f2023 +1.50 / f2024 −2.96 처럼.
    §7.116a E1 이 한 층 아래를 채웠다 — 산포 2종은 형질이 아니라 **상태**다
    (인접 시즌 상관 0.225 / 0.326, 천장 0.995).

    상위권 참가자가 "상황에 따라 게이트를 뒀다"고 했다는 전언이 들어왔다.
    게이트가 이 채널의 구제책이 되려면 조건이 하나다 —
    **게이트 변수가 곧 부호를 결정하는 변수여야 한다.**
    부호가 시즌 축으로 갈리는데 카운트로 게이트하면 아무 일도 안 일어난다.
    반대로 "2스트라이크 행에서는 양 fold 다 +, 아닌 행에서는 양 fold 다 −"라면
    전역 β 는 두 반대 부호의 **평균**이라 0 근처에서 흔들렸던 것이고,
    게이트가 정확한 구제책이 된다.

    이 파일은 그 조건 하나만 판별한다. 채택 실험이 아니다.

무엇을 재나 (학습 0회 — 캐시된 z 만 쓴다)
    기준선   블렌드(α=MLP_ALPHA) + δ(hand) + δ(count4)   ← 배포 상태, tmresid.py 와 동일
    r        그 상태의 잔차
    각 (트랙맨 컬럼 c, 상황축 A, 버킷 k) 에 대해 그 버킷 행만으로 r ~ c 회귀 →
    **투수 클러스터 CR0 t** 를 양 fold 에서 낸다.

    통계량  S(c, A) = max over k [ 부호 일치 버킷의 min(|t23|, |t24|) ]
            = "이 축 위에서 가장 잘 살아남은 버킷의 약한 쪽 증거력"

⚠️ 두 개의 함정을 코드로 막는다
    ① 투수 안에서 상수인 컬럼을 행 단위 OLS 로 재면 t 가 부풀어 있다.
       §7.116a E2 실측 **2~2.5배**. → 전부 투수 CR0.
    ② 버킷 m 개면 우연한 부호 일치가 공짜로 생긴다.
       → 죽은 축(`game_dayofweek`)을 **같은 칸 수로 랜덤 재배정**해 같은 코드
       경로로 돌린다. NULL_DRAWS 번 뽑아 S 의 널 분포를 만든다.
       (전례: 요일로 쪼갰더니 +22.8 이 나왔다.)

컬럼 선택은 사전 등록이다 — fold 를 보고 고르지 않는다
    형질군(E1 통과)  relh_mean · rels_mean · ext_mean · ext_std · relh_std
    상태군(E1 즉사)  rels_std · relpoint_disp
    상태군을 빼지 않는 이유: §7.39 가 부호 반전을 본 컬럼이 정확히 이것들이라
    **게이트 가설이 겨냥하는 대상 자체**다. 다만 여기서 살아도 2단계(버킷별
    trackman 재집계)에서 형질성을 다시 물어야 한다 — 시즌 집계값은 상황 혼합값이다.

실행:  .venv/Scripts/python.exe -u src/tmgate.py 2>&1 | tee experiments/tmgate.log
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
from config import CTX_AXES, CTX_K, MLP_ALPHA, ROOT, SEED, TARGET_COL, TRAIN_CSV
from data import load_csv
from features import logit_adjust
from gapstrike2 import cluster_t
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
MIN_ROWS = 5000          # 버킷이 이보다 작으면 t 를 믿지 않는다
MIN_PITCHERS = 50        # 클러스터 수 하한 — CR0 는 G 가 작으면 못 쓴다
NULL_DRAWS = 40          # 죽은 축 랜덤 재배정 횟수

# ── trackman 집계 테이블 위치 ────────────────────────────────────
#    submit/model 에는 없다 (TRACKMAN_MODE="none" 으로 되돌리면서 빠졌다).
#    §7.9 당시 빌드인 model_s28 에 남아 있다.
TM_CSV = ROOT / "model_s28" / "trackman_pitcher.csv"

# 사전 등록 컬럼 — E1 형질성 표 기준으로 두 군으로 나눠 찍는다
TRAIT_COLS = ["tm_relh_mean", "tm_rels_mean", "tm_ext_mean",
              "tm_ext_std", "tm_relh_std"]
STATE_COLS = ["tm_rels_std", "tm_relpoint_disp"]
COLS = TRAIT_COLS + STATE_COLS


# ── 상황 축 ──────────────────────────────────────────────────────
#    전부 **투구 직전** 상태다. 기전이 없는 축은 올리지 않는다 (ctxsplit.py 관행).
def ax_runner1b(df):
    """세트포지션. 시즌 집계 릴리스 산포는 와인드업/셋 **혼합값**이다 —
    주자로 가르는 것은 그 혼합을 직접 푸는 조작이다."""
    return df["runner_on_1b"].to_numpy().astype(np.int64), 2


def ax_runners(df):
    """주자 유무 전반 (1루 한정이 아닌 넓은 판)."""
    return (df["num_runners_on"].to_numpy().astype(np.int64) > 0).astype(np.int64), 2


def ax_count4(df):
    """0=일반 1=3볼 2=2스트라이크 3=풀카운트급.
    '코너를 노리는가 vs 존에 넣는가' — 제구 지표가 걸리는 조건이 다르다."""
    b = df["balls_before"].to_numpy().astype(np.int64)
    s = df["strikes_before"].to_numpy().astype(np.int64)
    return (b >= 3).astype(np.int64) + 2 * (s >= 2).astype(np.int64), 4


def ax_count12(df):
    b = np.clip(df["balls_before"].to_numpy().astype(np.int64), 0, 3)
    s = np.clip(df["strikes_before"].to_numpy().astype(np.int64), 0, 2)
    return b * 3 + s, 12


def ax_inning3(df):
    """피로·등판 국면. 산포는 피로에 따라 커진다."""
    i = df["inning"].to_numpy().astype(np.int64)
    return np.clip((i - 1) // 3, 0, 2), 3


def ax_hand(df):
    """§7.116a E2 의 B칸이 쓴 축 (슬롯 × same_hand, 통과선 미달).
    같은 기계로 재현되는지 보는 자리."""
    return df["batter_hand"].to_numpy().astype(np.int64), 2


def ax_bases(df):
    """base_state 전체 — 주자 조합."""
    v = df["base_state"].to_numpy()
    u = np.unique(v)
    return np.searchsorted(u, v).astype(np.int64), len(u)


AXES = [
    ("runner_1b",  ax_runner1b, "세트포지션 — 투구 메커니즘이 다르다"),
    ("주자유무",   ax_runners,  "주자 있음 전반"),
    ("count4",     ax_count4,   "3볼/2스트라이크 압박 운용"),
    ("count12",    ax_count12,  "카운트 전체 (칸이 잘다)"),
    ("inning3",    ax_inning3,  "피로·등판 국면"),
    ("hand",       ax_hand,     "§7.116a B칸 재현"),
    ("base_state", ax_bases,    "주자 조합"),
]


def bucket_ts(r, v, pid, code, m):
    """버킷별 (기울기, CR0 t, 행수, 클러스터수). 표본 미달 버킷은 nan."""
    out = []
    for k in range(m):
        msk = code == k
        n = int(msk.sum())
        if n < MIN_ROWS:
            out.append((np.nan, np.nan, n, 0))
            continue
        sl, t, _se, G = cluster_t(r[msk], v[msk], pid[msk])
        if G < MIN_PITCHERS:
            out.append((np.nan, np.nan, n, G))
            continue
        out.append((sl, t, n, G))
    return out


def stat_S(t23, t24):
    """부호 일치 버킷 중 약한 쪽 |t| 의 최댓값. 없으면 nan."""
    best, arg = np.nan, -1
    for k in range(len(t23)):
        a, b = t23[k], t24[k]
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        if np.sign(a) != np.sign(b):
            continue
        sv = min(abs(a), abs(b))
        if not np.isfinite(best) or sv > best:
            best, arg = sv, k
    return best, arg


def main():
    df = load_csv(TRAIN_CSV)
    s_all = df["season"].to_numpy()
    tm = pd.read_csv(TM_CSV)
    cols = [c for c in COLS if c in tm.columns]
    rng = np.random.default_rng(SEED)

    print(f"trackman 표 {tm.shape}   컬럼 {len(cols)}/{len(COLS)}   {TM_CSV}")
    print(f"기준선 = 블렌드(α={MLP_ALPHA}) + δ{CTX_AXES}   ← 배포 상태 (tmresid.py 와 동일)")
    print(f"버킷 하한 {MIN_ROWS:,}행 / {MIN_PITCHERS}투수   "
          f"널 뽑기 {NULL_DRAWS}회 (요일 랜덤 재배정)\n")

    cached = foldcache.load(FOLDS, verbose=False)

    # fold 별 재료를 먼저 다 만든다 (r, v, pid, 축 코드)
    mat = {}
    for fold in FOLDS:
        tr, va = s_all < fold, s_all == fold
        d = df[va]
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        adj, _ = logit_adjust(d, {"ctx": ctx})
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-(z + adj))), yv.mean())
        base = bss(yv, p0)
        r = yv - p0

        sub = tm[tm.season == fold].set_index("pitcher_id")
        pid = d["pitcher_id"].to_numpy(dtype=np.int64)
        vals = {}
        for c in cols:
            v = sub[c].reindex(pid).to_numpy(dtype=np.float64)
            miss = np.isnan(v)
            if miss.all():
                continue
            # 결측 → 관측 평균 (tmresid.py 와 같은 처리)
            vals[c] = np.where(miss, np.nanmean(v), v)
        hit = np.isin(pid, sub.index.to_numpy())
        mat[fold] = dict(r=r, pid=pid, d=d, vals=vals, base=base, hit=hit.mean())
        print(f"  fold {fold}  기준 BSS {base:.1f}  n {len(r):,}  "
              f"trackman 적중 {hit.mean():.1%}  투수 {len(np.unique(pid))}명")

    cols = [c for c in cols if c in mat[FOLDS[0]]["vals"] and c in mat[FOLDS[1]]["vals"]]

    # ── 0단계: 게이트 없는 전역 CR0 t — §7.39 를 클러스터 SE 로 다시 본다 ──
    print("\n" + "=" * 96)
    print("0단계  게이트 없음 — 전역 CR0 t (§7.39 는 OLS 였다. E2 실측 2~2.5배 부풀림)")
    print("=" * 96)
    print(f"  {'컬럼':<22}{'f2023 t':>10}{'f2024 t':>10}{'부호':>8}   군")
    glob = {}
    for c in cols:
        ts = []
        for fold in FOLDS:
            M = mat[fold]
            _, t, _, _ = cluster_t(M["r"], M["vals"][c], M["pid"])
            ts.append(t)
        glob[c] = ts
        agree = "일치" if np.sign(ts[0]) == np.sign(ts[1]) else "★반전"
        grp = "형질" if c in TRAIT_COLS else "상태"
        print(f"  {c:<22}{ts[0]:>10.2f}{ts[1]:>10.2f}{agree:>8}   {grp}")

    # ── 1단계: 축별 버킷 t + 죽은 축 널 ──────────────────────────
    print("\n" + "=" * 96)
    print("1단계  상황 버킷별 CR0 t → S = max_k[부호일치 버킷의 min(|t23|,|t24|)]")
    print("       NULL = 요일을 같은 칸 수로 랜덤 재배정한 죽은 축 (같은 코드 경로)")
    print("=" * 96)

    dow = df["game_dayofweek"].to_numpy()
    udow = np.unique(dow)
    dow_i = {fold: np.searchsorted(udow, mat[fold]["d"]["game_dayofweek"].to_numpy())
             for fold in FOLDS}
    nd = len(udow)

    summary = []
    for name, fn, why in AXES:
        code_all, m = fn(df)
        codes = {fold: code_all[s_all == fold] for fold in FOLDS}

        print(f"\n── {name}  ({m}칸)  — {why}")
        print(f"  {'컬럼':<22}{'S':>7}{'버킷':>6}{'t23':>8}{'t24':>8}"
              f"{'NULL중앙':>10}{'NULL최대':>10}{'초과':>6}")

        for c in cols:
            per = {}
            for fold in FOLDS:
                M = mat[fold]
                per[fold] = bucket_ts(M["r"], M["vals"][c], M["pid"], codes[fold], m)
            t23 = [x[1] for x in per[FOLDS[0]]]
            t24 = [x[1] for x in per[FOLDS[1]]]
            S, arg = stat_S(t23, t24)

            # 널 — 요일을 m 칸으로 랜덤 재배정 (매 draw 새 배정, 시드 고정)
            nulls = []
            for _ in range(NULL_DRAWS):
                assign = rng.integers(0, m, size=nd)
                n23 = bucket_ts(mat[FOLDS[0]]["r"], mat[FOLDS[0]]["vals"][c],
                                mat[FOLDS[0]]["pid"], assign[dow_i[FOLDS[0]]], m)
                n24 = bucket_ts(mat[FOLDS[1]]["r"], mat[FOLDS[1]]["vals"][c],
                                mat[FOLDS[1]]["pid"], assign[dow_i[FOLDS[1]]], m)
                ns, _ = stat_S([x[1] for x in n23], [x[1] for x in n24])
                if np.isfinite(ns):
                    nulls.append(ns)
            nmed = float(np.median(nulls)) if nulls else np.nan
            nmax = float(np.max(nulls)) if nulls else np.nan
            ok = "✓" if (np.isfinite(S) and np.isfinite(nmax) and S > nmax) else ""

            a = t23[arg] if arg >= 0 else np.nan
            b = t24[arg] if arg >= 0 else np.nan
            print(f"  {c:<22}{S:>7.2f}{arg:>6}{a:>8.2f}{b:>8.2f}"
                  f"{nmed:>10.2f}{nmax:>10.2f}{ok:>6}")
            summary.append((name, c, S, arg, a, b, nmed, nmax, bool(ok)))

    # ── 판정 ────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("판정")
    print("=" * 96)
    win = [x for x in summary if x[8]]
    print(f"  죽은 축 최대를 넘긴 (축, 컬럼) 조합: {len(win)}/{len(summary)}")
    for name, c, S, arg, a, b, nmed, nmax, _ in sorted(win, key=lambda x: -x[2]):
        g = "일치" if np.sign(glob[c][0]) == np.sign(glob[c][1]) else "★전역반전"
        print(f"    {name:<12}{c:<22} S {S:.2f} (버킷 {arg}, t {a:+.2f}/{b:+.2f})"
              f"  NULLmax {nmax:.2f}   전역 {glob[c][0]:+.2f}/{glob[c][1]:+.2f} {g}")
    print()
    print("  읽는 법")
    print("   · 살아남은 조합이 0개  → 게이트 가설 사망. 부호 반전은 상황 축이 아니다.")
    print("   · 살아남되 **전역 부호가 이미 일치**하는 컬럼뿐 → 게이트가 한 일이 없다.")
    print("     (게이트의 값어치는 '전역 반전인데 버킷 안에서 일치'일 때만 생긴다)")
    print("   · 전역 반전 컬럼이 버킷 안에서 일치 → **2단계로**. 단 그때도 채택이 아니다 —")
    print("     시즌 집계값은 상황 혼합값이라 버킷별 재집계로 형질성(E1)을 다시 물어야 한다.")
    print()
    print("  ⚠️ S 는 채택 근거가 아니다. §7.9 가 홀드아웃 +13.3 → LB −13.6 을 두 번 냈다.")
    print("     이 단계가 답하는 것은 '게이트가 부호 반전을 설명하는가' 하나뿐이다.")


if __name__ == "__main__":
    main()
