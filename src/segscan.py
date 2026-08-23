"""세그먼트 축 전수 스캔 — **모델이 이미 얼마나 맞히나** (`MODEL.md §7.125`).

§7.124 의 교훈: `game_type` 의 "이득 +2.85" 는 전부 `r_F−r_R = −0.0304` 라는
**가정**에서 나온 값이었고, 실제 천장은 **+0.08** 이었다. 모델이 이미 격차를
거의 맞히고 있었기 때문이다. 축을 고르기 전에 **천장부터** 재야 한다.

    천장 = K · Σ_g w_g (b_g − b̄)²        b_g = p̄_g − r_g   (fold 라벨 = oracle)

⚠️ **천장은 그룹 수만으로 커진다.** r̂_g 의 표본오차가 Var(b̂) 에 그대로 들어가서
   영가설 아래 기댓값이 `≈ 1e5·(G−1)/n` 이다 (n=253,507 이면 그룹당 0.39 BSS).
   10그룹 축은 신호가 0이어도 +3.5 가 나온다. 그래서 세 가지를 나란히 낸다.

    oracle    fold 자신의 라벨로 잰 천장            (노이즈로 부풀어 있다)
    floor     그 부풀림의 해석적 기댓값 1e5(G−1)/n
    transfer  **직전 시즌**의 격차를 가져다 쓴 실제 이득 (노이즈가 도와주지 않는다)

`transfer` 만이 실제로 쓸 수 있는 수치다. `oracle − floor` 는 상한이고,
그마저도 §7.124 처럼 **전이율이 1보다 작으면 못 받는다.**
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import deploy29 as D                                            # noqa: E402
from config import TARGET_COL                                   # noqa: E402
from validation import bss                                      # noqa: E402


def qcodes(v, edges):
    return np.searchsorted(edges, v, side="right")


def axis_defs(train_lt_fold):
    """축 이름 → (설명, 코드 함수). 분위 경계는 **fold 이전 시즌에서만** 만든다."""
    q = {}
    for c in ("asof_pitcher_n", "asof_pitcher_success_rate", "asof_batter_n", "li",
              "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"):
        v = train_lt_fold[c].to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        q[c] = np.quantile(v, [0.2, 0.4, 0.6, 0.8])

    def num(c):
        return lambda x, c=c: qcodes(np.nan_to_num(x[c].to_numpy(dtype=np.float64),
                                                   nan=-1e18), q[c])

    def cat(c):
        return lambda x, c=c: pd.factorize(x[c].to_numpy())[0]

    return {
        "game_type":       ("정규 vs 그 외 (§7.124 실측 축)", cat("game_type")),
        "pitcher_hand":    ("투수 손", cat("pitcher_hand")),
        "batter_hand":     ("타자 손", cat("batter_hand")),
        "hand_matchup":    ("투타 손 조합 4", lambda x: pd.factorize(
            x["pitcher_hand"].astype(str) + x["batter_hand"].astype(str))[0]),
        "balls":           ("볼 카운트 0~3", lambda x: x["balls_before"].to_numpy()),
        "strikes":         ("스트라이크 0~2", lambda x: x["strikes_before"].to_numpy()),
        "count12":         ("볼×스트라이크 12칸", lambda x: (
            x["balls_before"].to_numpy() * 3 + x["strikes_before"].to_numpy())),
        "outs":            ("아웃 0~2", lambda x: x["outs_before"].to_numpy()),
        "base_state":      ("주자 상태 8", cat("base_state")),
        "top_bottom":      ("초/말", cat("top_bottom")),
        "inning_b":        ("이닝 구간", lambda x: np.clip(
            x["inning"].to_numpy(), 1, 10) // 3),
        "month":           ("월", cat("game_month")),
        "pitcher_exp":     ("투수 경험 5분위 (asof_n)", num("asof_pitcher_n")),
        "pitcher_skill":   ("투수 실력 5분위 (asof 성공률)",
                            num("asof_pitcher_success_rate")),
        "pitcher_middle":  ("투수 몰림률 5분위", num("asof_pitcher_middle_rate")),
        "pitcher_ball":    ("투수 볼률 5분위", num("asof_pitcher_ball_rate")),
        "batter_exp":      ("타자 경험 5분위", num("asof_batter_n")),
        "leverage":        ("LI 5분위", num("li")),
        "pitcher_team":    ("투수 팀", cat("pitcher_team_id")),
        # ── 위약 ────────────────────────────────────────────────────
        "PLACEBO_dow":     ("[위약] 요일 (§7.45 에서 가짜 +22.8)", cat("game_dayofweek")),
        "PLACEBO_pid2":    ("[위약] pitcher_id 홀짝",
                            lambda x: x["pitcher_id"].to_numpy().astype(np.int64) % 2),
        "PLACEBO_pid5":    ("[위약] pitcher_id mod 5",
                            lambda x: x["pitcher_id"].to_numpy().astype(np.int64) % 5),
    }


def scan(fold=2024):
    df, s, y = D.load_df()
    st = D.state(df, s, y, fold)
    d, p0, yv = st["d"], st["p0"], st["yv"]
    n = len(yv)
    r = float(yv.mean())
    K = 1e5 / (r * (1 - r))

    gt = df["game_type"].to_numpy()
    clean = ~((gt == "F") & (s < 2023))
    prev = df[(s == fold - 1) & clean]
    defs = axis_defs(df[s < fold])

    print(f"fold {fold}   n={n:,}   r={r:.5f}   K={K:,.0f}   기준 BSS {st['base']:.2f}")
    print(f"직전 시즌 {fold-1} 참조행 {len(prev):,}\n")
    print(f"  {'축':<16s}{'G':>3s}{'oracle':>9s}{'floor':>8s}{'천장':>9s}"
          f"{'transfer':>10s}{'α=0.5':>9s}   설명")

    rows = []
    for name, (desc, fn) in defs.items():
        g = np.asarray(fn(d))
        G = len(np.unique(g))
        if G < 2 or G > 40:
            continue
        w = np.bincount(g, minlength=G) / n
        pg = np.bincount(g, weights=p0, minlength=G) / np.maximum(np.bincount(g, minlength=G), 1)
        rg = np.bincount(g, weights=yv, minlength=G) / np.maximum(np.bincount(g, minlength=G), 1)
        b = pg - rg
        oracle = K * float(np.sum(w * (b - np.sum(w * b)) ** 2))
        floor = 1e5 * (G - 1) / n

        gp = np.asarray(fn(prev))
        yp = prev[TARGET_COL].to_numpy(dtype=np.float64)
        cnt = np.bincount(gp, minlength=G)
        rp = np.where(cnt > 0, np.bincount(gp, weights=yp, minlength=G) / np.maximum(cnt, 1), rg)
        bh = pg - rp
        dev = bh - np.sum(w * bh)                      # 전역 중립
        out = []
        for a in (1.0, 0.5):
            p = np.clip(p0 - a * dev[g], 1e-6, 1 - 1e-6)
            out.append(bss(yv, p) - st["base"])
        rows.append((name, G, oracle, floor, oracle - floor, out[0], out[1], desc))
        print(f"  {name:<16s}{G:>3d}{oracle:>9.2f}{floor:>8.2f}{oracle-floor:>9.2f}"
              f"{out[0]:>+10.2f}{out[1]:>+9.2f}   {desc}")
    return rows


def main():
    """두 fold 를 나란히 낸다. **§7.124 의 교훈이 여기 있다** — fold 하나짜리
    transfer 는 전이율의 표본이 1개고, `game_type` 이 fold2024 에서 +3.13 을 내고
    2025 에서 +0.08 이었다. 채택 조건은 **두 fold 모두 양수**다."""
    res = {f: {r[0]: r for r in scan(f)} for f in (2023, 2024)}
    names = [n for n in res[2024] if n in res[2023]]
    print()
    print("=" * 82)
    print("두 fold 대조 — transfer 만 본다 (oracle 은 노이즈로 부풀어 있다)")
    print()
    print(f"  {'축':<16s}{'f2023 α1':>10s}{'f2023 α.5':>11s}"
          f"{'f2024 α1':>10s}{'f2024 α.5':>11s}{'판정':>8s}")
    keep = []
    for n in sorted(names, key=lambda x: -min(res[2023][x][5], res[2024][x][5])):
        a, b = res[2023][n], res[2024][n]
        both = a[5] > 0 and b[5] > 0
        both5 = a[6] > 0 and b[6] > 0
        v = "✅ 통과" if both else ("△ α.5만" if both5 else "❌")
        if both or both5:
            keep.append(n)
        print(f"  {n:<16s}{a[5]:>+10.2f}{a[6]:>+11.2f}{b[5]:>+10.2f}{b[6]:>+11.2f}{v:>8s}")
    print()
    print(f"두 fold 모두 양수인 축: {keep or '없음'}")


if __name__ == "__main__":
    main()
