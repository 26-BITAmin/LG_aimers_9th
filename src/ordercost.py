"""§7.137 M2/M3 재실행 — **파생 범주형의 순서 비용** (`count_state`, `hand_matchup`).

왜 다시 재나
    §7.137 은 `count_state` 를 대조군으로 넣어놓고 **측정에 실패했다.**
    `catcost.py` 가 `deploy29.load_df()` 의 **원본 df** 에서 컬럼을 찾는데
    `count_state` 는 `build_features` 가 만드는 파생 컬럼이라 없다
    → `KeyError: 'count_state'` (experiments/catcost.log:56).
    그래서 MODEL.md §7.137 순서비용 표에 `count_state` 행이 없고,
    §7.138 A/B 팔에도 `count_state`·`hand_matchup` 이 들어간 적이 없다.

    여기서 고치는 것은 **컬럼을 어디서 가져오는가** 하나뿐이다. 추정량
    (`best_splits`)은 `catcost.py` 에서 그대로 import 한다 — 재구현하지 않는다.

측정
    M2  단일 분할 최적성. Fisher(1958): 이진 타깃 단일 분할의 최적 부분집합은
        비율 정렬 후 접두사. `임의 부분집합 최적 / ID연속 구간 최적` 이 순서 비용.
    M3  파편화. 최적 부분집합이 코드 공간에서 몇 개 연속구간으로 흩어지는가.
    N   **순서 순열 널** (§7.137 에 없던 것). 같은 컬럼의 코드 순서를 R 회 섞어
        비용 분포를 만든다. "3.30배가 큰가"는 다른 컬럼이 아니라 **이 분포** 와
        비교해야 한다 — L 이 다르면 랜덤 순서의 기대 비용도 다르다.
        관측이 널 중앙값 근처면 "순서가 정보를 안 담는다",
        널 하단(≈1.0)이면 "순서가 실제로 일한다".

재현 검사 (하네스 자기검증)
    같은 스크립트로 §7.137 의 세 수치가 다시 나와야 한다:
        batter_team_id 1.00 · pitcher_team_id 1.24 · base_state 3.30
    안 나오면 새 수치도 못 믿는다. 아래 REPRO 딕트가 문턱을 들고 있다.

실행:  .venv/Scripts/python.exe -u src/ordercost.py 2>&1 | tee experiments/ordercost.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

from catcost import best_splits          # ⚠️ 추정량은 재구현하지 않는다
from config import TRAIN_CSV, TARGET_COL, SEED
from data import load_csv

# §7.137 이 발표한 값. 재현 실패하면 새 수치도 무효다.
REPRO = {"batter_team_id": 1.00, "pitcher_team_id": 1.24, "base_state": 3.30}
REPRO_TOL = 0.03

N_PERM = 2000


def codes_of(v):
    """값 배열 → 0..L-1 코드. 순서는 값의 정렬 순서 (= cat_maps 와 같다).

    `features.fit_artifacts` 의 `cat_maps` 는 문자열이면 `sorted(str)`,
    정수면 `sorted(int)` 로 코드를 매긴다. 여기서도 같은 규칙을 쓴다.
    파생 컬럼(`count_state` 등)은 값 자체가 이미 0..L-1 정수라 항등이다.
    """
    if v.dtype.kind in "iuf":
        vals = sorted(np.unique(v).tolist())
    else:
        v = v.astype(str)
        vals = sorted(np.unique(v).tolist())
    code_of = {x: i for i, x in enumerate(vals)}
    return np.array([code_of[x] for x in v], dtype=np.int64), vals


def order_cost(codes, y, L):
    """(순서비용, 임의최적, 연속최적, 최적부분집합크기, 구간수, 레벨별 n·rate)"""
    cnt = np.bincount(codes, minlength=L).astype(np.float64)
    pos = np.bincount(codes, weights=y, minlength=L)
    rate = np.where(cnt > 0, pos / np.maximum(cnt, 1), 0.5)
    ba, bc, bset = best_splits(rate, cnt, np.arange(L))
    srt = np.argsort(rate)
    k = int(bset.sum())
    ks = sorted(srt[:k].tolist())
    runs = 1 + sum(1 for a, b in zip(ks, ks[1:]) if b != a + 1)
    return ba / max(bc, 1e-9), ba, bc, k, runs, cnt, rate


def perm_null(rate, cnt, rng, R=N_PERM):
    """코드 순서를 섞었을 때의 순서 비용 분포.

    ⚠️ 섞는 것은 **코드 배정**이지 라벨이 아니다. 레벨별 (n, rate) 는 그대로 두고
       '어떤 레벨이 코드 몇 번인가'만 바꾼다 → 임의 부분집합 최적은 불변이고
       연속 구간 최적만 움직인다. 그래서 이 널이 정확히 **순서의 값어치**를 잰다.
    """
    L = len(rate)
    ba, _, _ = best_splits(rate, cnt, np.arange(L))
    out = np.empty(R)
    for i in range(R):
        p = rng.permutation(L)            # p[j] = 코드 j 자리에 올 레벨
        _, bc, _ = best_splits(rate[p], cnt[p], np.arange(L))
        out[i] = ba / max(bc, 1e-9)
    return out


def run(df, s, y, upto, tag, *, check_repro=False):
    m = s <= upto
    yv = y[m]
    print("\n" + "=" * 98)
    print(f"{tag} — 학습 시즌 ≤ {upto}  ({m.sum():,}행, 양성률 {yv.mean():.4f})")
    print("=" * 98)

    balls = df["balls_before"].to_numpy(dtype=np.int16)[m]
    strikes = df["strikes_before"].to_numpy(dtype=np.int16)[m]
    ph = df["pitcher_hand"].to_numpy(dtype=np.int16)[m]
    bh = df["batter_hand"].to_numpy(dtype=np.int16)[m]

    # ⚠️ features.py:699 / :710 과 **같은 식**이어야 한다. 값이 어긋나면
    #    측정 대상이 배포 컬럼이 아니게 된다.
    derived = {
        "count_state":  (balls * 3 + strikes).astype(np.float32),
        "hand_matchup": (ph * 4 + bh).astype(np.float32),
    }

    cols = [
        # ── 재현 검사 (§7.137 발표값) ──
        ("batter_team_id",  df["batter_team_id"].to_numpy()[m],  "ID 번호순 = 임의"),
        ("pitcher_team_id", df["pitcher_team_id"].to_numpy()[m], "ID 번호순 = 임의"),
        ("base_state",      df["base_state"].to_numpy()[m],      "ASCII 정렬"),
        # ── 미측정이었던 파생 컬럼 ──
        ("count_state",     derived["count_state"],   "볼×3+스트 — 의미 있는 순서"),
        ("hand_matchup",    derived["hand_matchup"],  "투수손×4+타자손"),
        ("top_bottom",      df["top_bottom"].to_numpy()[m],      "2레벨 — 바닥 대조"),
    ]

    rng = np.random.default_rng(SEED)
    results = {}
    for col, v, note in cols:
        codes, vals = codes_of(v)
        L = len(vals)
        cost, ba, bc, k, runs, cnt, rate = order_cost(codes, yv, L)
        null = perm_null(rate, cnt, rng) if L > 2 else np.array([1.0])
        q = float((null <= cost).mean())
        results[col] = cost

        # ⚠️ REPRO 는 §7.137 이 **≤2022 창에서** 발표한 값이다. 다른 창에 대면
        #    당연히 어긋나고, 그걸 FAIL 로 찍으면 로그가 거짓말을 한다.
        flag = ""
        if check_repro and col in REPRO:
            ok = abs(cost - REPRO[col]) <= REPRO_TOL
            flag = f"   [§7.137 재현 {'OK' if ok else 'FAIL'} — 발표값 {REPRO[col]:.2f}]"

        print(f"\n  {col}  ({L}레벨, {note}){flag}")
        print(f"    레벨당 평균 {cnt.mean():>10,.0f}행 · 비율 "
              f"{rate.min():.4f}~{rate.max():.4f} (폭 {rate.max()-rate.min():.4f})")
        print(f"    임의 부분집합 최적 gain {ba:>10.1f}   ← 범주형이 1회로 가능")
        print(f"    ID연속 구간   최적 gain {bc:>10.1f}   ← 숫자 코드가 1회로 가능")
        print(f"    **순서 비용 = {cost:>.2f}배**")
        if L > 2:
            print(f"    순열 널 (R={N_PERM}): 중앙 {np.median(null):.2f}  "
                  f"[{np.percentile(null,5):.2f}, {np.percentile(null,95):.2f}]  "
                  f"최대 {null.max():.2f}   → 관측 분위 {q:>5.1%}")
        print(f"    최적 부분집합 = 하위 {k}/{L} 레벨 → 코드공간 연속구간 {runs}개"
              f" → 최소 분할 {runs if runs == 1 else 2*runs-1}회")
    return results


def main():
    print("불러오는 중 ...")
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    print(f"  {df.shape}")

    # ⚠️ 두 학습 창에서 잰다. §7.137 은 ≤2022 하나뿐이었다.
    #    순서 비용은 적합값이 아니라 데이터 구조량이지만, 창이 바뀌면
    #    레벨별 rate 가 움직이므로 안정성은 따로 확인해야 한다.
    r22 = run(df, s, y, 2022, "fold2023 학습창", check_repro=True)
    r23 = run(df, s, y, 2023, "fold2024 학습창")

    print("\n" + "=" * 98)
    print("요약 — 순서 비용 (1.0 = 순서가 공짜)")
    print("=" * 98)
    print(f"  {'컬럼':<20s}{'≤2022':>10s}{'≤2023':>10s}{'변동':>10s}")
    for c in r22:
        print(f"  {c:<20s}{r22[c]:>10.2f}{r23[c]:>10.2f}"
              f"{abs(r22[c]-r23[c]):>10.2f}")

    bad = [c for c, want in REPRO.items() if abs(r22[c] - want) > REPRO_TOL]
    print("\n재현 검사:", "전부 통과" if not bad else f"❌ 실패 {bad} — 새 수치 무효")


if __name__ == "__main__":
    main()
