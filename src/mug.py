"""계층적 μ 검정 — Lever 1 에서 유일하게 살아남은 조각 (§7.155).

배포 `_smooth(r, n, μ, k) = (n·r + kμ)/(n+k)` 의 **μ 가 전역 상수 하나**다
(`art["priors"][col]`, recency 가중). 이것이 그룹 조건부여야 했다면 배포 피처의
오차는 **정확히** 다음 꼴이다 — 두 수축식의 차:

    f_g − f = k(μ_g − μ)/(n + k)  ≡  Δ_g

그래서 **잔차에 Δ_g 를 직접 물어본다.** 학습 없이.

### 왜 3항인가

`a_g`(버킷 절편) · 깎임항 · Δ_g 는 **전부 n 의 함수라 서로 상관돼 있다.**
따로 넣으면 신호의 소유권을 못 가린다. 동시에 넣어야 한다:

    logit(y) = offset(배포 로짓) + Σ a_g·1[버킷 g] + b·깎임 + c·Δ_g

  · 깎임 = (f − μ)·k/(n+k)   ← **k** 오설정 방향. §7.31 스윕이 이미 닫은 채널
    (200→50 fold2024 −8.58 t −2.7 / 200→800 −7.39 t −2.4). 대조군으로 넣는다.
  · Δ_g                       ← **μ** 오설정 방향. 살아 있는 유일한 가설
  · 버킷 절편                 ← §7.153 이 배포했다 실패한 그 항

**채택 조건은 `c` 의 생존이다.** 그리고 버킷 절편이 같이 약해지면 "`a_g` 는 사실
μ 오설정의 그림자였다"가 된다.

### 세 관문 (전부 통과해야 다음 단계)

  ① 잔차에서 Δ_g 생존 (투수 클러스터 SE, 양 fold)
  ② 위약 그룹(요일 μ_g) 분리 — §7.45 의 요일 +22.8 이 이 자리의 영가설
  ③ **시즌 안정성** — (μ_g − μ) 를 2019~2024 각각 따로 재서 부호·크기가 유지되는가.
     s38 의 치명점은 "패턴의 시즌 지속성"을 제출 전에 못 쟀다는 것이었다.
     μ_g 는 다르다 — 라벨만 있으면 시즌별로 6번 계산된다. **이 검사가 판정력이
     제일 높다.** 이 저장소가 반복해서 진 곳이 "지금 신호가 있나"가 아니라
     "그 신호가 넘어가나"이기 때문이다 (§7.124 c=0.168, §7.154 c=0.162).

⚠️ μ_g 는 **train(시즌 < fold)에서만** 적합한다. test 행 간 집계가 아니다.

실행:  .venv/Scripts/python.exe -u src/mug.py 2>&1 | tee experiments/mug.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np
import pandas as pd
from scipy import stats

import deploy29 as D
from calslope import EPS, _cluster_cov, _glm, build_layers
from calsub import EDGE_SEASON, make_buckets
from config import RECENCY_HALF_LIFE, SMOOTH_K_PITCHER
from data import recency_weight
from predict import PROB_CENTER, PROB_SCALE
from validation import shift_to_prior

REF_SEASON = 2024                       # train.py:52 와 같은 값

RATE = "asof_pitcher_success_rate"
NCOL = "asof_pitcher_n"
K = SMOOTH_K_PITCHER


# ── μ_g 후보 그룹 (전부 행 독립) ─────────────────────────────────────
def group_codes(df, name, art_role=None):
    """(코드, 라벨). 각 행이 자기 값으로 자기 그룹을 안다 (CLAUDE.md §3.2)."""
    if name == "hand":
        # ⚠️ `pitcher_hand` 는 **정수 코드(1/2)** 다. 문자열로 가정하면 전 행이
        #    한 그룹에 몰려 Δ_g ≡ 0 이 되고 설계행렬이 특이해진다 (한 번 밟았다).
        u = art_role["hands"]
        v = df["pitcher_hand"].to_numpy()
        return np.searchsorted(u, v).clip(0, len(u) - 1), [f"hand{h}" for h in u]
    if name == "mix3":
        # 구종믹스 — 자기 행의 asof 값만 본다. 경계는 train 에서 동결.
        v = df["asof_pitcher_fastball_rate"].to_numpy(dtype=np.float64)
        e = art_role["mix_edges"]
        c = np.digitize(np.nan_to_num(v, nan=e[0]), e)
        return c, ["mix:FB저", "mix:FB중", "mix:FB고"]
    if name == "role":
        # 역할 대리 — train 에서 얼린 투수별 중앙 이닝. 미지 투수는 전역(코드 -1→0 처리)
        m = art_role["role_map"]
        v = pd.Series(df["pitcher_id"].to_numpy()).map(m).to_numpy(dtype=np.float64)
        c = np.where(np.isnan(v), 2, (v <= 3.0).astype(int))   # 2 = 미지
        return c, ["role:선발형", "role:구원형", "role:미지"]
    if name == "team":
        # §7.138 이 "팀 효과는 시즌 간 부호가 뒤집힌다"고 기록한 축 —
        # 관문 ③(시즌 안정성)의 스트레스 케이스로 일부러 넣는다.
        v = df["pitcher_team_id"].to_numpy()
        u = art_role["teams"]
        c = np.searchsorted(u, v).clip(0, len(u) - 1)
        return c, [f"team{t}" for t in u]
    if name == "dow":                              # ★ 위약 (§7.45)
        v = df["game_dayofweek"].to_numpy(dtype=np.int64)
        return v - v.min(), [f"dow{i}" for i in range(7)]
    raise ValueError(name)


def fit_mu_g(tr, w, codes, ng):
    """train 행에서 그룹별 recency 가중 μ_g. 배포 `art["priors"]` 와 같은 방식."""
    v = tr[RATE].to_numpy(dtype=np.float64)
    m = np.isfinite(v)
    mu = float(np.average(v[m], weights=w[m]))
    out = np.full(ng, mu)
    for g in range(ng):
        s = m & (codes == g)
        if s.sum() >= 2000:
            out[g] = float(np.average(v[s], weights=w[s]))
    return mu, out


def main():
    df, s, y = D.load_df()
    gt = df["game_type"].to_numpy()
    clean_all = ~((gt == "F") & (s < 2023))          # §7.48 오염 F 제외

    edges = np.quantile(df.loc[s < EDGE_SEASON, NCOL].to_numpy(float), [.2, .4, .6, .8])
    CAND = ["hand", "mix3", "role", "team", "dow"]

    # ══ 관문 ③ 먼저 — 시즌 안정성 (판정력이 제일 높다) ═══════════════
    print("=" * 100)
    print("관문 ③  시즌별 (μ_g − μ) 안정성 — clean 행, 시즌마다 독립 계산")
    print("=" * 100)
    seasons = list(range(2019, 2025))
    art_all = {"mix_edges": np.nanquantile(
                   df.loc[s < EDGE_SEASON, "asof_pitcher_fastball_rate"].to_numpy(float),
                   [1/3, 2/3]),
               "teams": np.unique(df["pitcher_team_id"].to_numpy()),
               "hands": np.unique(df["pitcher_hand"].to_numpy()),
               "role_map": df[s < EDGE_SEASON].groupby("pitcher_id")["inning"].median()}
    stab = {}
    for nm in CAND:
        rows = []
        for yr in seasons:
            m = (s == yr) & clean_all
            sub = df[m]
            c, lab = group_codes(sub, nm, art_all)
            w = np.ones(len(sub))
            mu, mug = fit_mu_g(sub, w, c, len(lab))
            rows.append(mug - mu)
        A = np.array(rows)                            # 시즌 × 그룹
        stab[nm] = (A, lab)
        print(f"\n  ── {nm} ──")
        print(f"    {'시즌':<7s}" + "".join(f"{l:>13s}" for l in lab[:6]))
        for yr, r in zip(seasons, A):
            print(f"    {yr:<7d}" + "".join(f"{v:>+13.5f}" for v in r[:6]))
        # 부호 일치율 + 연속 시즌 스케일비 (§7.151 의 원점 회귀와 같은 도구)
        base = A[-2]                                   # 2023
        cs = []
        for i in range(len(seasons) - 1):
            a0, a1 = A[i], A[i + 1]
            if (a0 * a0).sum() > 0:
                cs.append((a0 * a1).sum() / (a0 * a0).sum())
        sign = float(np.mean(np.sign(A[-1]) == np.sign(A[-2])))
        r_last = np.corrcoef(A[-1], A[-2])[0, 1] if len(lab) > 2 else np.nan
        print(f"    스케일비 c (연속): " + " ".join(f"{v:+.3f}" for v in cs))
        print(f"    2023→2024 부호 일치 {sign:.0%}"
              + (f" · corr {r_last:+.3f}" if np.isfinite(r_last) else "")
              + f" · ‖Δ‖ 2019→2024 " + " ".join(f"{np.sqrt((r*r).sum()):.4f}" for r in A))

    # ══ 관문 ①② — 3항 동시 회귀 ═════════════════════════════════════
    print("\n" + "=" * 100)
    print("관문 ①②  3항 동시 회귀:  offset(배포 로짓) + a_g 버킷절편 + 깎임 + Δ_g")
    print("=" * 100)
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        yv, cl, n_id, mm = st["yv"], st["cl"], st["n_id"], st["mm"]
        ax = st["Xax"] @ st["dep"]
        ybar = yv.mean()
        nm_l, zb = build_layers(df, s, fold, mm)[-1]
        pb = shift_to_prior(1 / (1 + np.exp(-(zb + ax))), ybar)
        p = shift_to_prior(PROB_CENTER + PROB_SCALE * (pb - PROB_CENTER), ybar)
        off = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))

        d = df.iloc[st["vi"][mm]]
        n = d[NCOL].to_numpy(dtype=np.float64)
        r = d[RATE].to_numpy(dtype=np.float64)

        tr = df[(s < fold) & clean_all]
        wtr = recency_weight(tr["season"].to_numpy(), RECENCY_HALF_LIFE,
                             ref_season=REF_SEASON)
        art_f = {"mix_edges": np.nanquantile(
                     tr["asof_pitcher_fastball_rate"].to_numpy(float), [1/3, 2/3]),
                 "teams": np.unique(df["pitcher_team_id"].to_numpy()),
                 "hands": np.unique(df["pitcher_hand"].to_numpy()),
                 "role_map": tr.groupby("pitcher_id")["inning"].median()}
        v = tr[RATE].to_numpy(dtype=np.float64)
        mfin = np.isfinite(v)
        mu = float(np.average(v[mfin], weights=wtr[mfin]))

        f = (n * np.nan_to_num(r, nan=mu) + K * mu) / (n + K)
        shave = (f - mu) * K / (n + K)
        bk, blab = make_buckets(df, s, fold, mm, "apn5", edges)
        Dm = np.column_stack([(bk == g).astype(np.float64) for g in range(1, 5)])

        print(f"\n{'─'*100}\nfold {fold}  층 {nm_l}  clean {len(yv):,}행 · 투수 "
              f"{len(np.unique(cl)):,}  μ={mu:.5f}  K={K:.0f}")
        print(f"  {'그룹':<10s}{'|μ_g−μ| 최대':>13s}{'c(Δ_g)':>10s}{'cluster t':>11s}"
              f"{'b(깎임)':>11s}{'t':>9s}{'‖a_g‖ 전':>10s}{'‖a_g‖ 후':>10s}{'판정':>8s}")
        for gn in CAND:
            cg, lab = group_codes(d, gn, art_f)
            ctr, _ = group_codes(tr, gn, art_f)
            _, mug = fit_mu_g(tr, wtr, ctr, len(lab))
            dg = (mug[cg] - mu) * K / (n + K)

            if np.std(dg) < 1e-12:            # μ_g 가 전부 fallback → 검정 불가
                print(f"  {gn:<10s}{'Δ_g 퇴화 (그룹별 표본 부족)':>60s}")
                continue
            X2 = np.column_stack([np.ones(len(yv)), Dm, shave])
            b2, mu2, H2 = _glm(X2, yv, offset=off)
            X3 = np.column_stack([np.ones(len(yv)), Dm, shave, dg])
            b3, mu3, H3 = _glm(X3, yv, offset=off)
            V3, _ = _cluster_cov(X3, yv, mu3, H3, cl, n_id)
            se = np.sqrt(np.diag(V3))
            t_dg, t_sh = b3[-1] / se[-1], b3[-2] / se[-2]
            na_b, na_a = np.sqrt((b2[1:5] ** 2).sum()), np.sqrt((b3[1:5] ** 2).sum())
            ok = "✅" if abs(t_dg) > 2 else "—"
            if gn == "dow":
                ok = "위약" + ("❌누수" if abs(t_dg) > 2 else "✅")
            print(f"  {gn:<10s}{np.abs(mug-mu).max():>13.5f}{b3[-1]:>10.3f}"
                  f"{t_dg:>11.2f}{b3[-2]:>11.3f}{t_sh:>9.2f}"
                  f"{na_b:>10.4f}{na_a:>10.4f}{ok:>8s}")


if __name__ == "__main__":
    main()
