"""투수 × 상황 스플릿 δ — 플래툰(§7.36)이 연 계열을 전수 조사한다

무엇이 열렸나
    `asof_*` 컬럼은 전부 **상황을 뭉갠(pooled) 투수 요약**이다.
    성공률·reverse·middle·직전 N경기 — 어느 것도 "이 투수가 **어떤 상황에서**
    더 흔들리는가"를 담지 않는다. 그래서 손잡이 스플릿이 비어 있었고,
    δ(투수, 손잡이)를 사후 로짓에 더한 것만으로 LB +5.93이 나왔다 (§7.36).

    같은 논리가 손잡이에만 걸릴 이유가 없다. **상황 축은 여러 개다.**
    이 파일은 후보 축을 전부 같은 잣대로 재서, 만들 값이 있는지부터 가른다.

기준선이 중요하다
    baseline z = 블렌드 + **PLATOON_BETA · δ_hand**  ← 배포와 같은 상태
    이미 배포된 손잡이 δ를 뺀 z에 대고 재면 새 축의 이득에 손잡이 몫이 섞인다.
    그래서 `hand(재적용)`을 대조군으로 같이 돌린다 — **이건 0이 나와야 정상이다.**
    0이 아니면 기준선 구성이 틀린 것이고, 나머지 숫자도 전부 못 믿는다.

재는 것 (전부 학습 없음 — 캐시된 z만 쓴다)
    T1  corr(δ_과거, δ_실제)    스플릿이 시즌을 넘어 유지되나
    T2  잔차 회귀 t             배포 모델이 아직 안 쓰고 있나  ← 결정적
    T3  β 스윕 BSS 이득         실제로 점수가 오르나 (양 fold + 크기비)
    대조군  리그평균 상황효과    주효과 재발견인지 가른다 (0이어야 진짜)

다중검정 주의
    축을 9개 재고 최댓값을 고르면 최댓값은 편향돼 있다. 조건 ①②③(2·SE / 양 fold
    부호일치 / 크기비 ≤3)은 축 **하나**를 위한 잣대다. 여기서 살아남은 것은
    후보일 뿐이고, 채택은 `platoon2.py`처럼 단독 재측정으로 결정한다.

실행:  .venv/Scripts/python.exe -u src/ctxsplit.py 2>&1 | tee experiments/ctxsplit.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache
from config import MLP_ALPHA, PLATOON_BETA, PLATOON_K, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
K = PLATOON_K                 # 200. 배포 상수 — fold를 보고 고르지 않는다
BETAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
MIN_CELL = 100                # T1에서 검증 셀 최소 표본


# ── 후보 상황 축 ──────────────────────────────────────────────────
# 각 함수는 df → 정수 코드 배열. 카디널리티가 작아야 셀당 표본이 산다.
# 전부 **투구 직전** 상태다 (data_description.md 3~5절).
def _q(df, col):
    return df[col].to_numpy()


def ctx_hand(df):        # 대조군 — 이미 배포 중. 추가 이득 0이어야 정상
    return _q(df, "batter_hand").astype(np.int64)


def ctx_runner1b(df):    # 세트포지션. 와인드업과 투구 메커니즘이 실제로 다르다
    return _q(df, "runner_on_1b").astype(np.int64)


def ctx_runners(df):     # 주자 유무 (1루 한정이 아닌 넓은 판)
    return (_q(df, "num_runners_on").astype(np.int64) > 0).astype(np.int64)


def ctx_count4(df):      # 0=일반 1=3볼 2=2스트라이크 3=풀카운트급
    b = _q(df, "balls_before").astype(np.int64)
    s = _q(df, "strikes_before").astype(np.int64)
    return (b >= 3).astype(np.int64) + 2 * (s >= 2).astype(np.int64)


def ctx_count12(df):     # 카운트 전체 12칸
    b = np.clip(_q(df, "balls_before").astype(np.int64), 0, 3)
    s = np.clip(_q(df, "strikes_before").astype(np.int64), 0, 2)
    return b * 3 + s


def ctx_inning3(df):     # 1~3 / 4~6 / 7+  — 피로·역할
    i = _q(df, "inning").astype(np.int64)
    return np.clip((i - 1) // 3, 0, 2)


def ctx_home(df):        # 초=원정팀 공격=홈 투수. 마운드·심판 성향
    return (_q(df, "top_bottom").astype(str) == "T").astype(np.int64)


def ctx_li2(df, thr=None):   # 승부처 여부 (임계는 train 중앙값으로 동결)
    return (_q(df, "li").astype(np.float64) > thr).astype(np.int64)


def ctx_bteam(df):       # 상대 팀 친숙도
    return _q(df, "batter_team_id").astype(np.int64)


# (이름, 함수, 한 줄 기전) — 기전이 없는 축은 여기 올리지 않는다
CONTEXTS = [
    ("hand(대조·기배포)", ctx_hand,    "이미 배포 중 → 0이어야 정상"),
    ("runner_on_1b",      ctx_runner1b, "세트포지션 — 투구 메커니즘이 다르다"),
    ("주자유무",          ctx_runners,  "주자 있음 전반"),
    ("count4",            ctx_count4,   "3볼/2스트라이크 압박 운용"),
    ("count12",           ctx_count12,  "카운트 전체 (칸이 잘다)"),
    ("inning3",           ctx_inning3,  "피로·등판 국면"),
    ("홈/원정",           ctx_home,     "마운드·심판 성향"),
    ("li 고저",           ctx_li2,      "승부처 압박"),
    ("상대팀",            ctx_bteam,    "타선 친숙도"),
]


def build_delta(pid, code, y, m, k):
    """train에서만 fit. (pitcher, code) → 수축된 자기평균 대비 편차."""
    key = pid * m + code
    u, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    t = np.bincount(inv, weights=y)

    upid, invp = np.unique(pid, return_inverse=True)
    n_p = np.bincount(invp).astype(np.float64)
    rate_p = np.bincount(invp, weights=y) / n_p
    rp = rate_p[np.searchsorted(upid, u // m)]
    return u, (n / (n + k)) * (t / n - rp), n


def lookup(u, d, pid, code, m):
    kv = pid * m + code
    j = np.clip(np.searchsorted(u, kv), 0, len(u) - 1)
    hit = u[j] == kv
    return np.where(hit, d[j], 0.0), hit


def actual_delta(pid, code, y, m):
    """검증 시즌 자기 데이터로 잰 실제 편차 (T1의 참값)."""
    key = pid * m + code
    u, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    t = np.bincount(inv, weights=y)
    upid, invp = np.unique(pid, return_inverse=True)
    n_p = np.bincount(invp).astype(np.float64)
    rate_p = np.bincount(invp, weights=y) / n_p
    rp = rate_p[np.searchsorted(upid, u // m)]
    return u, t / n - rp, n


def resid_t(r, d):
    """잔차 r에 d를 회귀했을 때의 t. 모델이 안 쓰는 정보면 0이 아니다."""
    dc = d - d.mean()
    den = float((dc ** 2).sum())
    if den <= 0:
        return 0.0, 0.0
    sl = float((dc * r).sum() / den)
    res = (r - r.mean()) - sl * dc
    se = float(np.sqrt((res ** 2).sum() / (len(r) - 2) / den))
    return sl, (sl / se if se > 0 else 0.0)


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    pid = df["pitcher_id"].to_numpy(dtype=np.int64)
    cached = foldcache.load(FOLDS, verbose=False)

    li_thr = float(np.median(df["li"].to_numpy(dtype=np.float64)))
    print(f"train {df.shape}   k = {K} (배포 PLATOON_K)   li 임계 {li_thr:.4f}")
    print(f"기준선 = 블렌드(α={MLP_ALPHA}) + {PLATOON_BETA}·δ_hand  ← 배포 상태\n")

    # 축 → fold → β별 이득 (투수별 / 대조군)
    res = {}

    for fold in FOLDS:
        tr, va = s < fold, s == fold
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]

        # ── 기준선에 배포 중인 손잡이 δ를 얹는다 ──────────────────
        bh = df["batter_hand"].to_numpy(dtype=np.int64)
        uh, dh, _ = build_delta(pid[tr], bh[tr], y[tr], 8, K)
        dh_row, _ = lookup(uh, dh, pid[va], bh[va], 8)
        z0 = z + PLATOON_BETA * dh_row

        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean())
        base = bss(yv, p0)
        r = yv - p0                                    # 배포 상태의 잔차

        print("=" * 100)
        print(f"fold {fold}   train {tr.sum():,} → valid {va.sum():,}"
              f"   기준 BSS(배포상태) {base:.1f}")
        print("=" * 100)
        print(f"  {'축':<18}{'칸':>7}{'적중':>7}{'δ산포':>9}{'T1 corr':>9}"
              f"{'T2 t':>8}{'대조 t':>8}   " + " ".join(f"β{b:<5.2f}" for b in BETAS))

        for name, fn, _why in CONTEXTS:
            code_all = fn(df, li_thr) if fn is ctx_li2 else fn(df)
            uc = np.unique(code_all)
            m = int(uc.max()) + 1
            if m > 64:                                  # 팀 ID 등 코드가 성길 때
                code_all = np.searchsorted(uc, code_all)
                m = len(uc)

            u, d, n_cell = build_delta(pid[tr], code_all[tr], y[tr], m, K)
            d_row, hit = lookup(u, d, pid[va], code_all[va], m)

            # 대조군 — 리그 평균 상황효과 (투수별이 아니다)
            lg = np.zeros(m)
            for c in range(m):
                msk = code_all[tr] == c
                if msk.any():
                    lg[c] = y[tr][msk].mean() - y[tr].mean()
            d_lg = lg[code_all[va]]

            # T1 — 과거 δ vs 검증 시즌 실제 δ (셀 가중)
            uv, dv, nv = actual_delta(pid[va], code_all[va], y[va], m)
            j = np.clip(np.searchsorted(u, uv), 0, len(u) - 1)
            ok = (u[j] == uv) & (nv >= MIN_CELL)
            if ok.sum() >= 20:
                w, a, b = nv[ok], d[j][ok], dv[ok]
                am = np.average(a, weights=w)
                bm = np.average(b, weights=w)
                cov = np.average((a - am) * (b - bm), weights=w)
                c1 = cov / np.sqrt(np.average((a - am) ** 2, weights=w)
                                   * np.average((b - bm) ** 2, weights=w))
            else:
                c1 = float("nan")

            _, t2 = resid_t(r, d_row)
            _, t2c = resid_t(r, d_lg)

            gains, gains_c = [], []
            for be in BETAS:
                gains.append(bss(yv, shift_to_prior(
                    1.0 / (1.0 + np.exp(-(z0 + be * d_row))), yv.mean())) - base)
                gains_c.append(bss(yv, shift_to_prior(
                    1.0 / (1.0 + np.exp(-(z0 + be * d_lg))), yv.mean())) - base)
            res.setdefault(name, {})[fold] = (gains, gains_c)

            print(f"  {name:<18}{len(u):>7,}{hit.mean():>6.0%}{d_row.std():>9.4f}"
                  f"{c1:>9.3f}{t2:>8.1f}{t2c:>8.1f}   "
                  + " ".join(f"{g:>+6.1f}" for g in gains))

    # ── 판정표 ────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("판정 — 양 fold 동시 양수 + 크기비 ≤ 3 (조건 ②③). 대조군이 0 근처여야 진짜다")
    print("=" * 100)
    print(f"  {'축':<18}{'최적β':>7}{'f2023':>9}{'f2024':>9}{'크기비':>8}"
          f"{'대조23':>9}{'대조24':>9}  판정")
    for name, byf in res.items():
        a = np.array(byf[FOLDS[0]][0])
        b = np.array(byf[FOLDS[1]][0])
        ca = np.array(byf[FOLDS[0]][1])
        cb = np.array(byf[FOLDS[1]][1])
        lo = np.minimum(a, b)
        i = int(np.argmax(lo))               # 두 fold 중 나쁜 쪽이 가장 큰 β
        ratio = max(abs(a[i]), abs(b[i])) / max(min(abs(a[i]), abs(b[i])), 1e-9)
        ok = a[i] > 0 and b[i] > 0 and ratio <= 3
        print(f"  {name:<18}{BETAS[i]:>7.2f}{a[i]:>+9.2f}{b[i]:>+9.2f}{ratio:>8.2f}"
              f"{ca[i]:>+9.2f}{cb[i]:>+9.2f}  {'✓ 후보' if ok else '✗'}")

    print("\n  ⚠️ 여기서 ✓는 **후보**일 뿐이다. 9축을 재고 최댓값을 고르면 그 값은 편향돼 있다.")
    print("     채택은 단독 재측정(platoon2.py 방식) + 기존 δ와의 **합동** 측정으로 결정한다.")
    print("     δ끼리 정보가 겹치면 개별 이득의 합보다 작다.")


if __name__ == "__main__":
    main()
