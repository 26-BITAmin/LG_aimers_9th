"""EDA 2부 — 상한과 형태. `eda.py`가 "신호가 어디 있나"를 쟀다면 여기는
"최대 얼마이고, 그중 얼마가 전이되나"와 "관계의 모양이 시즌마다 같은가"를 본다.

G. 그룹별 상한 vs 전이
   각 그룹핑(투수/카운트/매치업…)에 대해 두 숫자를 낸다.
     · 상한  — 검증 시즌 **자기 데이터**의 그룹 평균으로 예측 (LOO 보정).
               "이 그룹 구조에 정보가 얼마나 있나"의 천장.
     · 전이  — 과거 시즌 그룹 평균(k=200 스무딩)으로 예측.
               "그중 실제로 쓸 수 있는 몫".
   둘의 격차가 **드리프트 세금**이다. 이 프로젝트가 33번 진 이유가 여기 보일 것이다.

   ⚠️ LOO 보정을 하는 이유: 그룹이 작으면 자기 자신이 평균에 들어가 상한이 부풀려진다.
      투수 1명당 수백 투구라 보정 없이는 천장이 가짜로 높게 나온다.

H. 타깃 반응 곡선 — 상위 피처별 P(y=1) vs 값, **시즌마다 따로**.
   모양이 겹치면 관계가 안정적이고, 벌어지면 그 피처가 드리프트원이다.

I. 분포 이동 — 시즌별 분위수. 입력 자체가 이동했는지.
J. 카운트 12칸 · K. 시즌 내 월별 추이 · L. 투수별 성공률 분포

실행:  .venv/Scripts/python.exe -u src/eda2.py 2>&1 | tee experiments/eda2.log
산출:  experiments/eda2.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

from config import EXP_DIR, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

SMOOTH_K = 200.0
OUT = {}


def codes(a):
    """임의의 키 배열 → 0..K-1 정수 코드."""
    u, inv = np.unique(a, return_inverse=True)
    return inv, len(u)


def ceiling(key_va, y_va):
    """LOO 그룹 평균으로 예측했을 때의 BSS oracle = 이 그룹 구조의 천장.

    LOO를 쓰는 이유: 자기 자신이 그룹 평균에 포함되면 y와 p가 상관되어
    천장이 위로 부풀려진다. 그룹이 작을수록 심하다.
    """
    c, k = codes(key_va)
    cnt = np.bincount(c, minlength=k).astype(np.float64)
    tot = np.bincount(c, weights=y_va, minlength=k)
    n_i, t_i = cnt[c], tot[c]
    gm = y_va.mean()
    p = np.where(n_i > 1, (t_i - y_va) / np.maximum(n_i - 1, 1), gm)
    return float(bss(y_va, shift_to_prior(p, y_va.mean()))), k


def transfer(key_tr, y_tr, key_va, y_va):
    """과거 시즌 그룹 평균(스무딩)으로 검증 시즌을 예측 → 실제로 쓸 수 있는 몫."""
    u, inv_tr = np.unique(key_tr, return_inverse=True)
    cnt = np.bincount(inv_tr, minlength=len(u)).astype(np.float64)
    tot = np.bincount(inv_tr, weights=y_tr, minlength=len(u))
    prior = float(y_tr.mean())
    rate = (tot + SMOOTH_K * prior) / (cnt + SMOOTH_K)

    pos = np.searchsorted(u, key_va)
    pos_c = np.clip(pos, 0, len(u) - 1)
    hit = u[pos_c] == key_va
    p = np.where(hit, rate[pos_c], prior)
    return float(bss(y_va, shift_to_prior(p, y_va.mean()))), float(hit.mean())


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    print(f"train {df.shape}  ({time.time()-t0:.0f}s)")

    pid = df["pitcher_id"].to_numpy(dtype=np.int64)
    bid = df["batter_id"].to_numpy(dtype=np.int64)
    cs = (df["balls_before"].to_numpy(dtype=np.int64) * 3
          + df["strikes_before"].to_numpy(dtype=np.int64))
    hm = (df["pitcher_hand"].to_numpy(dtype=np.int64) * 4
          + df["batter_hand"].to_numpy(dtype=np.int64))
    pteam = df["pitcher_team_id"].to_numpy(dtype=np.int64)
    month = df["game_month"].to_numpy(dtype=np.int64)

    GROUPS = {
        "pitcher_id": pid,
        "batter_id": bid,
        "count_state": cs,
        "hand_matchup": hm,
        "pitcher_team": pteam,
        "pitcher x count": pid * 16 + cs,
        "pitcher x hand": pid * 16 + hm,
        "pitcher x month": pid * 16 + month,
        "pitcher x batter": pid * 100000 + bid,
    }

    # ══ G. 상한 vs 전이 ════════════════════════════════════════════
    print("\n" + "=" * 92)
    print("G. 그룹별 상한 vs 전이 (valid = 2024, 상한은 LOO 보정)")
    print("=" * 92)
    print(f"{'그룹':>18} {'그룹수':>9} {'상한':>9} {'전이':>9} {'드리프트세금':>13} {'적중':>7}")
    va, tr = s == 2024, s < 2024
    g_out = []
    for name, key in GROUPS.items():
        ub, k = ceiling(key[va], y[va])
        tf, hit = transfer(key[tr], y[tr], key[va], y[va])
        g_out.append({"name": name, "k": k, "ceiling": ub, "transfer": tf,
                      "tax": ub - tf, "hit": hit})
        print(f"{name:>18} {k:>9,} {ub:>9.1f} {tf:>9.1f} {ub-tf:>13.1f} {hit:>7.1%}")
    OUT["groups"] = g_out
    print("\n  배포 시스템은 fold2024에서 약 770이다. 상한이 그보다 낮은 그룹은")
    print("  그 축만으로는 현재 모델을 못 이긴다는 뜻이다.")

    # ══ H. 타깃 반응 곡선 ══════════════════════════════════════════
    print("\n" + "=" * 92)
    print("H. 타깃 반응 곡선 — 시즌마다 모양이 같은가")
    print("=" * 92)
    FEATS = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
             "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
             "asof_pitcher_prev5_game_success_rate", "asof_batter_success_rate",
             "asof_pitcher_n", "li"]
    seasons = [2019, 2020, 2021, 2022, 2023, 2024]
    curves = {}
    for c in FEATS:
        v = df[c].to_numpy(dtype=np.float64)
        fin = np.isfinite(v)
        # 경계는 **전 시즌 공통**으로 잡는다 — 시즌마다 다르면 모양 비교가 불가능하다
        edges = np.unique(np.quantile(v[fin], np.linspace(0, 1, 21)[1:-1]))
        b = np.searchsorted(edges, v, side="right")
        cur = {"edges": edges.tolist(), "by_season": {}}
        for ss in seasons:
            m = (s == ss) & fin
            cnt = np.bincount(b[m], minlength=len(edges) + 1).astype(np.float64)
            tot = np.bincount(b[m], weights=y[m], minlength=len(edges) + 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                rate = np.where(cnt >= 200, tot / np.maximum(cnt, 1), np.nan)
            cur["by_season"][str(ss)] = [None if not np.isfinite(x) else float(x)
                                         for x in rate]
            cur.setdefault("n", {})[str(ss)] = cnt.tolist()
        curves[c] = cur
        # 시즌 간 모양 차이 = 각 빈의 (시즌 양성률 − 시즌 평균)의 시즌간 표준편차
        rel = []
        for ss in seasons:
            r = np.array([np.nan if x is None else x
                          for x in cur["by_season"][str(ss)]], dtype=float)
            rel.append(r - np.nanmean(r))
        rel = np.array(rel)
        spread = float(np.nanmean(np.nanstd(rel, axis=0)))
        print(f"  {c:<45} 빈별 시즌간 산포 {spread:.4f}")
    OUT["curves"] = curves
    print("   (산포가 작을수록 관계의 모양이 안정적이다)")

    # ══ I. 분포 이동 ═══════════════════════════════════════════════
    print("\n" + "=" * 92)
    print("I. 분포 이동 — 입력 자체가 시즌마다 움직였나 (중앙값)")
    print("=" * 92)
    dist = {}
    print(f"{'컬럼':<45} " + " ".join(f"{ss:>8}" for ss in seasons))
    for c in FEATS:
        v = df[c].to_numpy(dtype=np.float64)
        row = []
        for ss in seasons:
            m = (s == ss) & np.isfinite(v)
            row.append(float(np.quantile(v[m], 0.5)))
        dist[c] = row
        print(f"{c:<45} " + " ".join(f"{x:>8.3f}" for x in row))
    OUT["dist_median"] = dist

    # ══ J. 카운트 12칸 ═════════════════════════════════════════════
    print("\n" + "=" * 92)
    print("J. 카운트 상태별 양성률 (행=볼, 열=스트라이크)")
    print("=" * 92)
    cnt_tab = []
    print(f"{'':>6} " + " ".join(f"{'S'+str(k):>16}" for k in range(3)))
    for b_ in range(4):
        cells = []
        for k in range(3):
            m = (df["balls_before"].to_numpy() == b_) & (df["strikes_before"].to_numpy() == k)
            m24 = m & (s == 2024)
            cells.append({"balls": b_, "strikes": k, "n": int(m.sum()),
                          "rate_all": float(y[m].mean()),
                          "rate_2024": float(y[m24].mean()) if m24.any() else None})
        cnt_tab += cells
        print(f"B{b_:>4} " + " ".join(
            f"{c['rate_all']:>7.4f}({c['n']/1e3:>5.0f}k)" for c in cells))
    OUT["count_table"] = cnt_tab

    # ══ K. 시즌 내 월별 추이 ═══════════════════════════════════════
    print("\n" + "=" * 92)
    print("K. 시즌 내 월별 양성률 — 시즌 안에서도 계속 내려가나")
    print("=" * 92)
    months = sorted(set(month.tolist()))
    mo = {}
    print(f"{'시즌':>6} " + " ".join(f"{m_:>7}" for m_ in months))
    for ss in seasons:
        row = []
        for m_ in months:
            k = (s == ss) & (month == m_)
            row.append(float(y[k].mean()) if k.sum() >= 500 else None)
        mo[str(ss)] = row
        print(f"{ss:>6} " + " ".join("      —" if x is None else f"{x:>7.4f}"
                                     for x in row))
    OUT["month_rate"] = {"months": months, "by_season": mo}

    # ══ L. 투수별 성공률 분포 ══════════════════════════════════════
    print("\n" + "=" * 92)
    print("L. 투수별 시즌 성공률 분포 — 선수 간 실력 차가 실제로 얼마나 벌어져 있나")
    print("=" * 92)
    pl = {}
    for ss in (2023, 2024):
        m = s == ss
        c, k = codes(pid[m])
        cnt = np.bincount(c).astype(np.float64)
        tot = np.bincount(c, weights=y[m])
        big = cnt >= 200
        r = (tot[big] / cnt[big])
        qs = np.quantile(r, [0.05, 0.25, 0.5, 0.75, 0.95])
        pl[str(ss)] = {"n_pitchers": int(big.sum()), "q": qs.tolist(),
                       "std": float(r.std()), "rates": r.tolist(),
                       "counts": cnt[big].tolist()}
        print(f"  {ss}: 200투구 이상 투수 {big.sum()}명   "
              f"5% {qs[0]:.3f} / 25% {qs[1]:.3f} / 50% {qs[2]:.3f} /"
              f" 75% {qs[3]:.3f} / 95% {qs[4]:.3f}   std {r.std():.4f}")
    OUT["pitcher_rates"] = pl
    print("\n  std가 곧 선수 축의 신호 상한과 직결된다 — 실력 차가 없으면 뭘 해도 못 번다.")

    path = EXP_DIR / "eda2.json"
    path.write_text(json.dumps(OUT, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {path}\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
