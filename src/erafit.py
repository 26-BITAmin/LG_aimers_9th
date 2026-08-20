"""era 보정 스크린 — 투수 이력 통계의 **리그 수준 오염**을 제거하면 전이가 살아나는가.

가설 (사베르메트릭스의 표준 처방)
--------------------------------
`asof_pitcher_success_rate`는 **커리어 누적**이다 (`src/eraprobe.py`에서 확인:
시즌 경계 연속성 100%, 중앙 |Δ| = 1). 그런데 리그 제구율은 6년간
.5647 → .4861로 8%p 내려갔다. 따라서 같은 0.52라도

    2019~2021에 쌓은 0.52  =  리그(.53) 대비 −0.01   →  평균 이하 투수
    2023~2024에 쌓은 0.52  =  리그(.49) 대비 +0.03   →  좋은 투수

**같은 숫자가 반대의 실력을 뜻한다.** 야구 통계가 ERA−·FIP−·OPS+·wRC+처럼
전부 리그 상대값을 쓰는 이유가 정확히 이것이다 (FanGraphs: "scale to league
average so that you can compare players during different eras").

전역 logit shift는 **평균** 레벨 오차만 지운다. 누적 구간의 조성이 투수마다
다르면 편향도 투수마다 달라서 스칼라 하나로는 지울 수 없다.
실측 (eraprobe): 2024 시즌 투수들의 누적 era 수준은 가중 std 0.0121,
5~95 분위 폭 **0.0407** — 시즌 드리프트 6년치와 맞먹는 산포다.

이 저장소가 이미 재놓은 것과 정확히 맞물린다 (MODEL.md §7.34 G)
--------------------------------------------------------------
    pitcher_id   천장 685.2   **전이 −93.5**   ← 과거 시즌 투수평균은 상수보다 나쁘다
    pitcher×hand 천장 820.2   전이  +81.2     ← δ는 자기평균 대비 **편차**라 레벨이 없다

전이 −93.5는 "투수 개인차가 시즌을 못 넘는다"로 읽혔다. **다른 읽기가 가능하다** —
넘어가지 못한 것은 개인차가 아니라 **레벨**이고, 개인차는 그 밑에 깔려 있다.
δ가 통한 이유도 같다 (구성상 레벨이 없다). 이 스크립트가 그 둘을 가른다.

측정 (학습 없음 — 캐시된 z만 쓴다)
----------------------------------
  Part A  전이 분해.  2×2 요인설계로 **탈레벨**과 **recency**를 분리한다.
          raw/del × career/marcel.  del ≫ raw 면 가설 지지.
  Part B  잔차 회귀 T2.  배포 모델(블렌드+δ_hand+δ_count4)이 **아직 안 쓰는가**.
          이게 0이면 트리가 asof_n·pitcher_id로 이미 흡수한 것이고 끝이다.
  Part C  β 스윕 → 조건 ①②③ (양 fold 부호일치 · 크기비 ≤3 · 시드노이즈 초과).

대조군
------
  raw_*        주효과 대조. "탈레벨이 일한 것인가, 그냥 투수 평균이 일한 것인가"
  shuffle      투수 라벨 무작위. 구조적으로 0이 나와야 한다
  era_row      per-row era 수준 (asof_n으로 누적 구간을 복원) — 편향의 직접 프록시

리키지
------
  모든 표는 **train 시즌으로만** fit 한다. valid 시즌의 리그율은 쓰지 않는다
  (검증 시즌 이후 구간은 R4 규칙으로 **추정**한다 — 배포와 같은 방식).
  조회 키는 `pitcher_id` (와 `asof_pitcher_n`) 뿐이라 행 단위로 독립이다.

실행:  .venv/Scripts/python.exe -u src/erafit.py 2>&1 | tee experiments/erafit.log
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
from config import CTX_AXES, CTX_K, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from ctxsplit import build_delta, lookup, resid_t
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
K = CTX_K                      # 수축 강도. 배포 δ와 같은 값을 쓴다 (새 노브를 안 만든다)
BETAS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

# Marcel(투수)의 3/2/1 가중을 그대로 쓴다. 직전 시즌부터 3시즌.
# 이 값은 **문헌 그대로**이고 이 데이터로 고른 것이 아니다 (튜닝 자유도 0).
MARCEL_W = [3.0, 2.0, 1.0]


def league_rates(s, y, seasons):
    """시즌별 리그 제구율. train 시즌만 넘긴다."""
    return {int(t): float(y[s == t].mean()) for t in seasons}


def next_level(lg, seasons):
    """R4 규칙 — 직전 시즌 + 시즌간 변화량의 평균. 배포 prior 추정과 같은 규칙.

    valid 시즌(과 그 이후)의 리그 수준을 **train만 보고** 추정한다.
    """
    v = [lg[t] for t in seasons]
    d = float(np.mean(np.diff(v))) if len(v) > 1 else 0.0
    return v[-1] + d


def season_table(pid, s, y, seasons):
    """(투수, 시즌) → (투구수, 성공률). train 시즌만."""
    n = {}
    r = {}
    for t in seasons:
        m = s == t
        u, inv = np.unique(pid[m], return_inverse=True)
        cnt = np.bincount(inv).astype(np.float64)
        tot = np.bincount(inv, weights=y[m])
        n[t] = (u, cnt)
        r[t] = (u, tot / cnt)
    return n, r


def pitcher_estimate(upit, ntab, rtab, seasons, lg, weights, delevel):
    """투수별 **편차** 추정치. 항상 0 중심 (= 리그 평균이 기준).

        delevel=True   Σ w_s n_ps (r_ps − lg_s) / (Σ w_s n_ps + k)     ← ERA− 방식
        delevel=False  Σ w_s n_ps  r_ps        / (Σ w_s n_ps)  − lg_all
                       (레벨을 안 지운 것. 주효과 대조군)

    수축 분모의 k는 배포 δ와 같은 CTX_K. 표본이 적으면 0(리그 평균)으로 당긴다.
    """
    num = np.zeros(len(upit))
    den = np.zeros(len(upit))
    lg_all_num = 0.0
    lg_all_den = 0.0
    for t, w in zip(seasons, weights):
        if w <= 0:
            continue
        u, cnt = ntab[t]
        _, rate = rtab[t]
        j = np.searchsorted(upit, u)
        ok = (j < len(upit)) & (upit[np.clip(j, 0, len(upit) - 1)] == u)
        j = j[ok]
        c = cnt[ok] * w
        val = rate[ok] - lg[t] if delevel else rate[ok]
        np.add.at(num, j, c * val)
        np.add.at(den, j, c)
        lg_all_num += float((cnt[ok] * w * lg[t]).sum())
        lg_all_den += float(c.sum())

    est = np.zeros(len(upit))
    nz = den > 0
    est[nz] = num[nz] / den[nz]
    if not delevel:
        est[nz] -= lg_all_num / max(lg_all_den, 1e-9)   # 전체 리그 수준만 뺀다
    return est * (den / (den + K)), den                 # 경험적 베이즈 수축


def era_level_per_row(pid_v, asof_n, upit, ntab, seasons, lg, lg_next):
    """행 단위 era 수준 — `asof_pitcher_n`으로 누적 구간을 복원한다.

    투수 p의 시즌별 투구수를 시간순으로 쌓아, 첫 N개가 어느 시즌에서 왔는지
    비례 배분한다. N이 train 총합을 넘는 몫은 **평가 시즌**의 것이므로
    `lg_next`(R4 추정) 수준으로 친다.

    → 이 행의 `asof_*` rate가 어느 리그 수준에서 쌓였는지의 추정치.
    """
    cum = np.zeros((len(upit), len(seasons) + 1))
    lvl = np.zeros(len(seasons))
    for i, t in enumerate(seasons):
        u, cnt = ntab[t]
        j = np.searchsorted(upit, u)
        ok = (j < len(upit)) & (upit[np.clip(j, 0, len(upit) - 1)] == u)
        col = np.zeros(len(upit))
        col[j[ok]] = cnt[ok]
        cum[:, i + 1] = cum[:, i] + col
        lvl[i] = lg[t]

    j = np.clip(np.searchsorted(upit, pid_v), 0, len(upit) - 1)
    known = upit[j] == pid_v
    N = np.maximum(asof_n.astype(np.float64), 1.0)

    wsum = np.zeros(len(pid_v))
    used = np.zeros(len(pid_v))
    for i in range(len(seasons)):
        lo = np.where(known, cum[j, i], 0.0)
        hi = np.where(known, cum[j, i + 1], 0.0)
        take = np.clip(np.minimum(hi, N) - lo, 0.0, None)
        wsum += take * lvl[i]
        used += take
    rest = np.maximum(N - used, 0.0)                     # 평가 시즌 몫
    return (wsum + rest * lg_next) / N


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    pid = df["pitcher_id"].to_numpy(dtype=np.int64)
    an = df["asof_pitcher_n"].to_numpy()
    bh = df["batter_hand"].to_numpy(dtype=np.int64)
    b = df["balls_before"].to_numpy(dtype=np.int64)
    st = df["strikes_before"].to_numpy(dtype=np.int64)
    c4 = (b >= 3).astype(np.int64) + 2 * (st >= 2).astype(np.int64)

    print(f"train {df.shape}   k={K}   Marcel 가중 {MARCEL_W} (문헌값, 튜닝 안 함)")
    print(f"배포 기준선 = 블렌드(α={MLP_ALPHA}) + {CTX_AXES}\n")

    rng = np.random.default_rng(42)
    summary = {}

    for fold in FOLDS:
        tr, va = s < fold, s == fold
        seasons = sorted(int(t) for t in np.unique(s[tr]))
        lg = league_rates(s[tr], y[tr], seasons)
        lg_next = next_level(lg, seasons)

        upit = np.unique(pid[tr])
        ntab, rtab = season_table(pid[tr], s[tr], y[tr], seasons)

        # recency 가중 벡터: 직전 시즌에 3, 그 앞에 2, 그 앞에 1, 나머지 0
        w_mar = [0.0] * len(seasons)
        for i, wv in enumerate(MARCEL_W):
            if len(seasons) - 1 - i >= 0:
                w_mar[len(seasons) - 1 - i] = wv
        w_flat = [1.0] * len(seasons)

        est = {}
        for tag, wts, dl in (("raw_career", w_flat, False),
                             ("del_career", w_flat, True),
                             ("raw_marcel", w_mar, False),
                             ("del_marcel", w_mar, True)):
            est[tag], den = pitcher_estimate(upit, ntab, rtab, seasons, lg, wts, dl)

        # ── 행 단위로 조회 ────────────────────────────────────────
        jv = np.clip(np.searchsorted(upit, pid[va]), 0, len(upit) - 1)
        hit = upit[jv] == pid[va]
        row = {t: np.where(hit, v[jv], 0.0) for t, v in est.items()}

        # 대조군 — 투수 라벨 셔플 (구조적으로 0이어야 한다)
        perm = rng.permutation(len(upit))
        row["shuffle"] = np.where(hit, est["del_marcel"][perm][jv], 0.0)

        # per-row era 수준 (편향의 직접 프록시). 중심화해서 레벨 성분을 뺀다.
        era = era_level_per_row(pid[va], an[va], upit, ntab, seasons, lg, lg_next)
        row["era_row"] = -(era - era.mean())      # 부호: era가 높으면 과대예측 → 음의 보정

        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        z0 = z.copy()
        for name, beta in CTX_AXES:
            code, m = (bh, 8) if name == "hand" else (c4, 4)
            u, d, _ = build_delta(pid[tr], code[tr], y[tr], m, K)
            drow, _ = lookup(u, d, pid[va], code[va], m)
            z0 = z0 + beta * drow
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean())
        base = bss(yv, p0)
        r = yv - p0

        print("=" * 110)
        print(f"fold {fold}   train {tr.sum():,} → valid {va.sum():,}   "
              f"기준 BSS(배포상태) {base:.1f}   투수 적중 {hit.mean():.1%}")
        print(f"  train 리그율 " + " ".join(f"{t}:{lg[t]:.4f}" for t in seasons)
              + f"   → R4 추정 {lg_next:.4f} (실제 {y[va].mean():.4f})")
        print("=" * 110)

        # ── Part A — 전이 분해 (§7.34 G 재현 + 분해) ─────────────
        print("\n  [A] 전이 — 이 추정치 **하나만으로** 검증 시즌을 예측 (오라클 shift 후)")
        print(f"      {'추정치':<14}{'산포':>9}{'BSS':>10}")
        for tag in ("raw_career", "raw_marcel", "del_career", "del_marcel", "shuffle"):
            p = shift_to_prior(np.clip(yv.mean() + row[tag], 1e-6, 1 - 1e-6), yv.mean())
            print(f"      {tag:<14}{row[tag].std():>9.5f}{bss(yv, p):>10.1f}")

        # ── Part B — 잔차 회귀 (결정적) ───────────────────────────
        print("\n  [B] 배포 모델 잔차 회귀 T2 — 모델이 **아직 안 쓰는** 정보인가")
        print(f"      {'추정치':<14}{'기울기':>11}{'t':>9}")
        for tag in ("raw_career", "raw_marcel", "del_career", "del_marcel",
                    "era_row", "shuffle"):
            sl, t2 = resid_t(r, row[tag])
            print(f"      {tag:<14}{sl:>+11.4f}{t2:>+9.2f}")

        # ── Part C — β 스윕 ──────────────────────────────────────
        print("\n  [C] β 스윕 — 최종 로짓에 β·추정치를 더했을 때의 BSS 이득")
        print(f"      {'추정치':<14}" + "".join(f"β{x:<7.2f}" for x in BETAS))
        for tag in ("raw_career", "raw_marcel", "del_career", "del_marcel",
                    "era_row", "shuffle"):
            g = []
            for beta in BETAS:
                p = shift_to_prior(1.0 / (1.0 + np.exp(-(z0 + beta * row[tag]))),
                                   yv.mean())
                g.append(bss(yv, p) - base)
            summary.setdefault(tag, {})[fold] = g
            print(f"      {tag:<14}" + "".join(f"{v:>+8.2f}" for v in g))
        print()

    # ── 요약 — 조건 ②③ ──────────────────────────────────────────
    print("=" * 110)
    print("요약 — 양 fold 공통 이득 구간과 크기비 (조건 ②③)")
    print("=" * 110)
    print(f"  {'추정치':<14}" + "".join(f"β{x:<12.2f}" for x in BETAS))
    for tag, per in summary.items():
        cells = ""
        for i, beta in enumerate(BETAS):
            a, bb = per[FOLDS[0]][i], per[FOLDS[1]][i]
            ok = a > 0 and bb > 0
            ratio = max(abs(a), abs(bb)) / max(min(abs(a), abs(bb)), 1e-9)
            mark = "✓" if (ok and ratio <= 3.0) else ("~" if ok else " ")
            cells += f"{a:>+5.1f}/{bb:>+5.1f}{mark}"
        print(f"  {tag:<14}{cells}")
    print("\n  ✓ = 양 fold 양수 **이고** 크기비 ≤3   ~ = 부호만 일치")
    print("  ⚠ 후보 6개를 동시에 쟀다. 여기서 사는 것은 **후보**일 뿐이고,")
    print("    채택은 단독 재측정 + 시드 노이즈(±3.3) 대조로 정한다.")


if __name__ == "__main__":
    cached = foldcache.load(FOLDS, verbose=True)
    main()
