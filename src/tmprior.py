"""E3 — 콜드스타트 스무딩 prior 를 trackman 개인 prior 로 바꾸면 무엇이 좋아지나 (§7.116).

아이디어 (재학습 없이 상한만 잰다)
    `features._smooth`는 asof_pitcher_{fastball,breaking,offspeed}_rate 를
    pitchmix_n 으로 **리그 평균**에 수축한다. 저n(콜드스타트) 행에서는 사실상
    리그 평균이 피처가 된다. trackman 은 같은 투수의 믹스를 더 큰 표본(퓨처스
    포함)으로 갖고 있고, 믹스 레벨은 실측 형질이다 (인접 시즌 corr 0.92~0.96).
    → 수축 타깃을 리그 → trackman 개인 prior 로 바꾸는 후보.

    새 컬럼 0개, 새 관계 0개 — 이미 학습·LB 검증된 관계에 들어가는 **값의 추정치**만
    바꾼다. §7.39의 사망 기전(관계의 시즌 반전)이 낄 자리가 구조적으로 없다.

무엇이 죽이나 (사전 예측)
    커버리지다. 저n 투수는 신인이고, 신인은 trackman(≤2024)에도 없다 — 콜드스타트
    역설. 관문(사전 등록): fold 행 중 pitchmix_n<500 이면서 trackman prior 사용
    가능(매핑 + tm 과거 n≥200)인 행 비율 **< 2% → 종료**. MAE 상대 개선 <20% → 종료.

측정 설계
    truth  = 그 투수의 시즌 t 실측 믹스 (trackman 시즌 t 로그, 3그룹 재정규화, n≥100)
             ⚠️ 퓨처스 투구 포함 — 1군 믹스의 근사다. 한계로 기록한다.
    현행   = smooth(raw_r3, n, 리그 prior_r3, K=200)
    대안   = smooth(raw_r3, n, trackman as-of 개인 prior_r3, K=200)  (없으면 리그)
    비교는 3그룹 재정규화(r3) 공간 — 'other'(1.24%, 시즌 간 무상관)는 제외한다.
    배포 피처는 원공간 rate 이므로 이 측정은 근사다. 상한 판정에는 충분하다.

실행:  .venv/Scripts/python.exe -u src/tmprior.py 2>&1 | tee experiments/tmprior.log
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

from config import DATA_DIR, RECENCY_HALF_LIFE, ROOT, SMOOTH_K_PITCHER, TRAIN_CSV
from data import load_csv, recency_weight

FOLDS = [2023, 2024]
K = SMOOTH_K_PITCHER            # 200 — 배포 상수
RATE_COLS = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
             "asof_pitcher_offspeed_rate"]
GROUPS = ["fastball", "breaking", "offspeed"]
TM_N_MIN = 200                  # trackman 개인 prior 최소 과거 표본
TRUTH_N_MIN = 100               # truth 최소 시즌 표본
THRESHOLDS = [1, 200, 500, 1000]   # thr=1 → n=0(결측) 행만


def renorm3(a):
    """(N,3) → 행별 3그룹 재정규화. 합이 0/NaN이면 NaN."""
    s = a.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = a / s
    out[~np.isfinite(out)] = np.nan
    return out


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()

    mp = pd.read_csv(ROOT / "model_s28" / "pitcher_id_map.csv")
    m2t = dict(zip(mp.pitcher_trackman_id, mp.pitcher_id))

    tmr = pd.read_csv(DATA_DIR / "trackman_history.csv",
                      usecols=["season", "pitcher_trackman_id", "pitch_type_group"])
    tmr["pitcher_id"] = tmr.pitcher_trackman_id.map(m2t)
    tmr = tmr.dropna(subset=["pitcher_id"])
    tmr["pitcher_id"] = tmr.pitcher_id.astype(np.int64)

    tmp = pd.read_csv(ROOT / "model_s28" / "trackman_pitcher.csv",
                      usecols=["pitcher_id", "season", "tm_n", "tm_share_fastball",
                               "tm_share_breaking", "tm_share_offspeed"])
    print(f"train {df.shape}   매핑 {len(m2t)}명   K={K}   "
          f"tm prior 게이트 n≥{TM_N_MIN}   truth 게이트 시즌 n≥{TRUTH_N_MIN}\n")

    for fold in FOLDS:
        va = s == fold
        d = df[va]
        n_rows = int(va.sum())

        # ── 리그 prior (배포와 같은 recency 가중, 학습 시즌만) ──────────
        tr = df[s < fold]
        w = recency_weight(tr["season"].to_numpy(), RECENCY_HALF_LIFE,
                           ref_season=fold - 1)
        pri = []
        for c in RATE_COLS:
            v = tr[c].to_numpy(dtype=np.float64)
            m = np.isfinite(v)
            pri.append(float(np.average(v[m], weights=w[m])))
        prior_l = renorm3(np.array(pri)[None, :])[0]

        # ── truth: 시즌 t 실측 믹스 (3그룹) ────────────────────────────
        tt = tmr[tmr.season == fold]
        ct = (tt.groupby(["pitcher_id", "pitch_type_group"], observed=True).size()
                .unstack("pitch_type_group", fill_value=0))
        for gcol in GROUPS:
            if gcol not in ct:
                ct[gcol] = 0
        n3 = ct[GROUPS].sum(axis=1)
        truth = ct[GROUPS].to_numpy(dtype=np.float64)
        truth = renorm3(truth)
        truth_map = {pid: truth[i] for i, (pid, nn) in enumerate(zip(ct.index, n3))
                     if nn >= TRUTH_N_MIN}

        # ── trackman as-of 개인 prior (시즌 fold 행 = ≤fold−1 누적) ────
        tp = tmp[(tmp.season == fold) & (tmp.tm_n >= TM_N_MIN)].set_index("pitcher_id")
        tm3 = renorm3(tp[["tm_share_fastball", "tm_share_breaking",
                          "tm_share_offspeed"]].to_numpy(dtype=np.float64))
        tm_map = {pid: tm3[i] for i, pid in enumerate(tp.index)
                  if np.isfinite(tm3[i]).all()}

        # ── 행 단위 ────────────────────────────────────────────────────
        pid = d["pitcher_id"].to_numpy(dtype=np.int64)
        n = np.nan_to_num(d["asof_pitcher_pitchmix_n"].to_numpy(dtype=np.float64), nan=0.0)
        raw = renorm3(d[RATE_COLS].to_numpy(dtype=np.float64))

        has_tm = np.array([p in tm_map for p in pid])
        has_truth = np.array([p in truth_map for p in pid])
        tmv = np.vstack([tm_map.get(p, prior_l) for p in pid])
        trv = np.vstack([truth_map.get(p, np.full(3, np.nan)) for p in pid])

        raw_f = np.where(np.isfinite(raw), raw, prior_l[None, :])        # _smooth: NaN→prior
        cur = (n[:, None] * raw_f + K * prior_l[None, :]) / (n[:, None] + K)
        raw_a = np.where(np.isfinite(raw), raw, tmv)
        alt = (n[:, None] * raw_a + K * tmv) / (n[:, None] + K)

        print("=" * 100)
        print(f"fold {fold}   행 {n_rows:,}   리그 prior(r3) "
              f"[{prior_l[0]:.3f} {prior_l[1]:.3f} {prior_l[2]:.3f}]   "
              f"truth 투수 {len(truth_map)}명   tm prior 투수 {len(tm_map)}명")
        print("=" * 100)
        print(f"  {'조건':<14}{'행수':>9}{'행비율':>8}{'tm있음':>8}{'truth있음':>10}"
              f"{'MAE 현행':>10}{'MAE 대안':>10}{'개선':>8}{'|Δ피처|':>9}")
        for thr in THRESHOLDS:
            m = n < thr
            nm = int(m.sum())
            if nm == 0:
                print(f"  n<{thr:<11}{0:>9}")
                continue
            mt = m & has_tm
            sc = m & has_truth
            mae_c = mae_a = imp = np.nan
            if sc.sum() > 0:
                mae_c = float(np.mean(np.abs(cur[sc] - trv[sc])))
                mae_a = float(np.mean(np.abs(alt[sc] - trv[sc])))
                imp = (mae_c - mae_a) / mae_c * 100 if mae_c > 0 else np.nan
            dshift = float(np.mean(np.abs(alt[mt] - cur[mt]))) if mt.sum() else np.nan
            print(f"  n<{thr:<11}{nm:>9,}{nm/n_rows:>8.2%}{mt.sum()/nm:>8.1%}"
                  f"{sc.sum()/nm:>10.1%}{mae_c:>10.4f}{mae_a:>10.4f}"
                  f"{imp:>+7.1f}%{dshift:>9.4f}")

        # 관문 수치 — 사전 등록: n<500 & tm 있음 행 비율 < 2% → 종료
        m500 = (n < 500) & has_tm
        print(f"\n  ★ 관문: n<500 이면서 tm prior 있는 행 = {m500.sum():,} "
              f"({m500.sum()/n_rows:.2%} of fold)   [사전 등록 종료선: <2%]")
        mt0 = (n < 1) & has_tm
        print(f"     참고: n=0(결측) 행 {int((n < 1).sum()):,}개 중 tm prior 있는 행 "
              f"{mt0.sum():,} ({mt0.sum()/max(int((n < 1).sum()), 1):.1%})"
              f"  ← 콜드스타트 역설의 직접 측정")


if __name__ == "__main__":
    main()
