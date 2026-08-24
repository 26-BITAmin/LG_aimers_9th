"""임의 코드 순서가 트리에 실제로 얼마를 물리는가 — 사용자 반문 검증.

내가 근거 없이 말한 것 두 개:
  (가) "12레벨이면 트리가 구간 분할로 잘 근사한다"   ← 세어본 적 없다
  (나) "범주형 전환은 과적합 위험"                    ← 12레벨·123k행/레벨에서?

측정 (학습 0회):
  M1  gain/split 효율 — 팀 ID 가 분할 하나당 얼마를 버는가. 낮으면 쪼개느라 낭비 중.
  M2  단일 분할 최적성 — 임의 부분집합(범주형이 할 수 있는 것) vs
      ID 연속 구간(숫자 코드가 할 수 있는 것). Fisher(1958): 이진 타깃 단일 분할의
      최적 부분집합은 비율 정렬 후 접두사다. 두 최적값의 비가 순서의 비용이다.
  M3  파편화 — 비율 상위 k팀이 ID 공간에서 몇 개의 연속 구간으로 흩어지는가.
      구간 r개 = 최소 2r−1 회 분할. 이게 '예산 비용'이다.
  M4  배포 잔차에도 팀 구조가 남아 있는가 (남아 있어야 개선 여지가 있다)

대조군: base_state (ASCII 순서가 우연히 r1 MSB — '좋은 순서'의 예)
        count_state (0..11, 볼×3+스트 — 의미 있는 순서)

실행:  .venv/Scripts/python.exe -u src/catcost.py 2>&1 | tee experiments/catcost.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import lightgbm as lgb

import deploy29 as D
from tmslot import resid_t_cr0

MODEL_DIR = Path(__file__).resolve().parent.parent / "submit" / "model"


def m1_efficiency():
    print("=" * 96)
    print("M1. gain / split 효율 — 분할 하나당 버는 gain (23개 모델 합산)")
    print("=" * 96)
    G = S = None
    names = None
    for f in sorted(MODEL_DIR.glob("lgbm_*.txt")):
        b = lgb.Booster(model_str=f.read_text(encoding="utf-8"))
        names = names or b.feature_name()
        g = np.asarray(b.feature_importance("gain"), np.float64)
        s = np.asarray(b.feature_importance("split"), np.float64)
        G = g if G is None else G + g
        S = s if S is None else S + s
    eff = np.where(S > 0, G / np.maximum(S, 1), 0.0)
    med = np.median(eff[S > 50])
    order = np.argsort(-eff)
    rk = {names[i]: r + 1 for r, i in enumerate(order)}
    print(f"  분할 50회 이상인 피처의 중앙 효율 = {med:.1f} (이걸 1.0 으로 정규화)\n")
    print(f"  {'컬럼':<34s}{'gain합':>10s}{'분할합':>9s}{'효율':>9s}{'중앙대비':>9s}{'효율순위':>9s}")
    watch = ["asof_pitcher_success_rate", "hand_same", "batter_team_id",
             "pitcher_team_id", "pitcher_id", "batter_id", "base_state",
             "count_state", "num_runners_on", "runner_on_1b", "li"]
    for c in watch:
        if c not in names:
            continue
        i = names.index(c)
        print(f"  {c:<34s}{G[i]:>10.3g}{S[i]:>9.0f}{eff[i]:>9.1f}"
              f"{eff[i]/med:>9.2f}{rk[c]:>9d}")
    print(f"\n  (효율 순위는 {len(names)}개 중. 낮을수록 '분할을 많이 쓰는데 조금씩만 번다')")


def best_splits(rate, n, order_codes):
    """이진 타깃 단일 분할의 가중 분산 감소.
    returns (최적 임의 부분집합, 최적 ID연속 구간, 최적 부분집합의 팀 목록)."""
    N = n.sum()
    mu = (rate * n).sum() / N

    def gain(mask):
        n1 = n[mask].sum()
        n0 = N - n1
        if n1 == 0 or n0 == 0:
            return -1.0
        m1 = (rate[mask] * n[mask]).sum() / n1
        m0 = (rate[~mask] * n[~mask]).sum() / n0
        return n1 * (m1 - mu) ** 2 + n0 * (m0 - mu) ** 2

    # 임의 부분집합 최적 = 비율 정렬 후 접두사 (Fisher 1958)
    srt = np.argsort(rate)
    best_any, best_any_set = -1.0, None
    for k in range(1, len(rate)):
        m = np.zeros(len(rate), bool)
        m[srt[:k]] = True
        g = gain(m)
        if g > best_any:
            best_any, best_any_set = g, m.copy()
    # ID 연속 구간 최적 = code <= t
    best_ctg = -1.0
    for t in range(len(rate) - 1):
        m = order_codes <= t
        best_ctg = max(best_ctg, gain(m))
    return best_any, best_ctg, best_any_set


def main():
    m1_efficiency()

    df, s, y = D.load_df()
    tr = s <= 2022                       # 학습 시즌만 (fold2023 기준)
    print("\n" + "=" * 96)
    print("M2/M3. 코드 순서가 단일 분할과 분할 예산에 물리는 비용 (학습 시즌 2019-22)")
    print("=" * 96)
    for col, note in [("batter_team_id", "ID 번호순 = 임의"),
                      ("pitcher_team_id", "ID 번호순 = 임의"),
                      ("base_state", "ASCII 정렬 — 우연히 r1 MSB"),
                      ("count_state", "볼×3+스트 — 의미 있는 순서")]:
        v = df[col].to_numpy()
        if v.dtype.kind not in "iuf":
            vals = sorted(str(x) for x in np.unique(v.astype(str)))
            code_of = {x: i for i, x in enumerate(vals)}
            codes = np.array([code_of[str(x)] for x in v])
        else:
            vals = sorted(np.unique(v).tolist())
            code_of = {x: i for i, x in enumerate(vals)}
            codes = np.array([code_of[x] for x in v])
        L = len(vals)
        cnt = np.bincount(codes[tr], minlength=L).astype(np.float64)
        pos = np.bincount(codes[tr], weights=y[tr], minlength=L)
        rate = np.where(cnt > 0, pos / np.maximum(cnt, 1), 0.5)
        ba, bc, bset = best_splits(rate, cnt, np.arange(L))
        # 파편화: 비율 상위 k 가 ID 공간에서 몇 구간인가
        srt = np.argsort(rate)
        runs = []
        for k in range(1, L):
            ks = sorted(srt[:k].tolist())
            runs.append(1 + sum(1 for a, b in zip(ks, ks[1:]) if b != a + 1))
        opt_k = int(bset.sum())
        opt_runs = runs[opt_k - 1]
        print(f"\n  {col}  ({L}레벨, {note})")
        print(f"    레벨당 평균 {cnt.mean():,.0f}행 · 비율 범위 "
              f"{rate.min():.4f}~{rate.max():.4f} (폭 {rate.max()-rate.min():.4f})")
        print(f"    최적 임의 부분집합 gain {ba:>10.1f}   ← 범주형이 1회로 가능")
        print(f"    최적 ID연속 구간   gain {bc:>10.1f}   ← 숫자 코드가 1회로 가능")
        print(f"    **순서 비용 = {ba/max(bc,1e-9):>.2f}배**"
              f"   (1.0 이면 순서가 공짜, 클수록 손해)")
        print(f"    최적 부분집합 = 하위 {opt_k}개 레벨 → ID 공간에서 연속구간 {opt_runs}개"
              f" → 최소 분할 {opt_runs if opt_runs==1 else 2*opt_runs-1}회")
        print(f"    상위 k별 구간수: {runs}")

    print("\n" + "=" * 96)
    print("M4. 배포 잔차에 팀 구조가 남아 있는가 (남아야 개선 여지가 있다)")
    print("=" * 96)
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        d = st["d"]
        line = []
        for col in ("batter_team_id", "pitcher_team_id"):
            v = d[col].to_numpy(np.float64)
            # 팀별 잔차 평균의 분산이 우연 수준인가 (셔플 널 대비 z)
            u = np.unique(v)
            g = np.array([st["r"][v == t].mean() for t in u])
            n = np.array([(v == t).sum() for t in u], float)
            obs = float(np.average(g ** 2, weights=n))
            rng = np.random.default_rng(42)
            null = []
            for _ in range(40):
                p = rng.permutation(st["r"])
                gg = np.array([p[v == t].mean() for t in u])
                null.append(float(np.average(gg ** 2, weights=n)))
            null = np.array(null)
            z = (obs - null.mean()) / max(null.std(), 1e-12)
            line.append(f"{col} z={z:>6.1f}")
        print(f"  fold {fold} (기준 BSS {st['base']:.1f}):  " + "   ".join(line))
    print("\n  z 가 크면 팀별 잔차 편차가 우연보다 크다 = 배포 층이 팀을 다 못 먹었다.")


if __name__ == "__main__":
    main()
