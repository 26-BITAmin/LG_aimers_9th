"""§7.184 — 상대 질량 보정 (p1/b1) β 선정 — **사전 등록** (측정 전 작성)

§7.183 프로브의 후속. 후보는 사전 관문(min|t|≥3)을 미달했으나 4/4 방향 양수·
위약 초과·직교·기전 정합이라 별도 등록으로 승격한다.

후보
    v_p = −std(p1_opp)   투수가 상대해온 타자 난이도 (LOO, hl=0.5, k=50) — §7.183 과 동일
    v_b = −std(b1_opp)   타자가 상대해온 투수 수준 (LOO)
    dz = β_p·v_p + β_b·v_b   사후 로짓 가산 (gap 축과 같은 전달 형태 — 전이 전례 0.93)
    표준화 상수는 **엔티티 표의 비가중 평균/표준편차** (train 에서만 계산, 얼림).
    미등장 엔티티 → v=0 (보정 없음).

사전 등록 판정 (측정 전 고정)
    ① 각 축 단독: 양 fold 공통 양수 β 구간(고원)이 폭 ≥ 3칸(0.0075) 존재
    ② β 는 §7.37 밴드 규칙 — [0.5β, 1.5β] ⊂ 고원. argmax 금지. 곡선 비대칭이면 왼쪽.
    ③ 결합 (β_p, β_b): fold 간 크기비 ≤ 3
    ④ 위약(기하 보존 순열 쌍둥이)에 같은 스윕 — **양 fold 공통 양수 고원이 없어야** 한다.
       그리고 결합 후보의 쌍체 Δ(후보 − 위약, 같은 draw) P ≥ 0.9 (fold2024, B=4000)
       [수정 1회차, 실행 후] 존재 검사가 크기 무통제였다 — p_plb+ 가 +0.08~+0.46
       짜리 3칸 런으로 전체를 기각시켰다 (후보 크기의 1/10~1/30). §7.183 과 같은
       **크기 관문**으로 교체: 위약 바닥 = 위약 4팔의 min-fold Δ 최댓값,
       후보의 min-fold Δ 가 그 3배를 초과해야 통과. ⚠️ 데이터를 본 후의 수정이므로
       이 관문의 증거 등급은 한 단계 아래로 친다 (로그에 원판정도 남긴다).
    ⑤ 2×2 상호작용 |Δ_both − Δ_p − Δ_b| 가 주효과 합의 25% 이내
       ([[count-what-a-new-correction-double-subtracts]] — 이중 차감 검사)
    전부 통과 → LB 슬롯 1개. 기대 LB Δ = fold2024 Δ × [0.5, 0.93] (gap 전례 0.93,
    T1 코어 겹침 희석 전례 0.51). **LB 가 챔피언 대비 음수면 되돌리고 기록한다.**
    ⛔ LB 를 보고 β 를 다시 고르지 않는다.

알려진 위험 (장부)
    - 잔차층이 s29+cat10+스케일 등가 — 현 챔피언(T1 계보)의 잔차가 아니다 (§7.180 제약).
      T1 코어가 이 축을 이미 일부 샀다면 희석된다.
    - §7.58 전례: 전 관문 통과 후 LB −11. / 반대로 §7.55 gap 축은 0.93 으로 전이.
    - fold 측정의 표는 ≤2022/≤2023 적합, 배포 표는 ≤2024 적합 — 상수는 각자 얼린다
      ([[training-side-correction-cleaner-than-inference-is-skew]] 아님: 양쪽 다 사후 가산).

실행:  .venv/Scripts/python.exe -u src/oppbeta.py 2>&1 | tee experiments/oppbeta.log
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import deploy29 as D
import gnnprobe as G
import resprobe as R
from validation import bss, shift_to_prior

GRID = np.round(np.arange(0.0, 0.0425, 0.0025), 4)      # 후보는 부호 고정(+)
B_BOOT = 4000
PLATEAU_MIN = 3          # 칸 (= 0.0075)


def std_from_table(tab):
    """엔티티 표의 비가중 평균/표준편차 — train 에서만 나오는 얼린 상수."""
    v = tab.to_numpy(dtype=np.float64)
    return float(v.mean()), float(v.std())


def make_v(tabs, name, dva, sign=-1.0):
    axis, tab = tabs[name]
    mu, sd = std_from_table(tab)
    keys = dva.pitcher_id if axis == "p" else dva.batter_id
    import pandas as pd
    v = pd.Series(keys.to_numpy()).map(tab).to_numpy(dtype=np.float64)
    v = np.where(np.isfinite(v), sign * (v - mu) / sd, 0.0)
    return v, mu, sd


def delta(b, dz):
    p = shift_to_prior(1 / (1 + np.exp(-(np.log(b["p"] / (1 - b["p"])) + dz))),
                       b["ybar"])
    return bss(b["yv"], p) - b["base"], p


def plateau(d23, d24):
    """양 fold 공통 양수 β 구간 (0 제외). (lo, hi, 폭칸수) 또는 None."""
    ok = (d23 > 0) & (d24 > 0)
    ok[0] = False
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return None
    # 가장 긴 연속 구간
    runs, start = [], idx[0]
    for a, bb in zip(idx, idx[1:]):
        if bb != a + 1:
            runs.append((start, a))
            start = bb
    runs.append((start, idx[-1]))
    lo, hi = max(runs, key=lambda r: r[1] - r[0])
    return GRID[lo], GRID[hi], hi - lo + 1


def band_pick(lo, hi, d23, d24):
    """[0.5β,1.5β] ⊂ [lo,hi] 이고, **하강부를 밟지 않는** β 중 최대.

    등록문 "곡선 비대칭이면 왼쪽": 꺾이는 fold(여기선 f2023)의 argmax 를 peak 로 잡고
    1.5β ≤ peak 를 요구한다. [수정 1회차: 최초 구현이 cands[-1] 만 취해 등록문과
    어긋났다 — 등록문이 이긴다. §7.58 이 하강부 가장자리에서 LB −11 로 죽은 전례.]"""
    peak = min(GRID[int(np.argmax(d23))], GRID[int(np.argmax(d24))])
    cands = [b for b in GRID if b > 0 and 0.5 * b >= lo - 1e-9
             and 1.5 * b <= hi + 1e-9 and 1.5 * b <= peak + 1e-9]
    return cands[-1] if cands else None


def sweep(F, V, name):
    d23 = np.array([delta(F[2023], b * V[2023][name])[0] for b in GRID])
    d24 = np.array([delta(F[2024], b * V[2024][name])[0] for b in GRID])
    return d23, d24


def main():
    t0 = time.time()
    df, s, y = D.load_df()
    F, V = {}, {}
    for fold in D.FOLDS:
        print(f"\n=== fold {fold} ===")
        F[fold] = R.deploy_base(df, s, y, fold)
        rng = np.random.default_rng(1000 + fold)          # §7.183 과 같은 시드 → 같은 위약
        tabs = G.graph_tables(df, s, fold, rng)
        dva = F[fold]["st"]["d"]
        V[fold] = {}
        for nm, src, sign in (("p", "p1_opp", -1.0), ("b", "b1_opp", -1.0),
                              ("p_plb", "p1_plb", -1.0), ("b_plb", "b1_plb", -1.0),
                              ("p_plb+", "p1_plb", +1.0), ("b_plb+", "b1_plb", +1.0)):
            V[fold][nm], mu, sd = make_v(tabs, src, dva, sign)
            if fold == 2024 and not nm.endswith("+"):
                print(f"    {nm:<7s} mu={mu:.4f} sd={sd:.4f}")
    del df

    print("\n" + "=" * 96)
    print("① 축별 β 스윕 (다른 축 0) — ΔBSS")
    print("=" * 96)
    SW = {nm: sweep(F, V, nm) for nm in ("p", "b", "p_plb", "b_plb", "p_plb+", "b_plb+")}
    hdr = "  β      " + "".join(f"{b:>8.4f}" for b in GRID[1:])
    for nm in ("p", "b", "p_plb", "b_plb", "p_plb+", "b_plb+"):
        d23, d24 = SW[nm]
        print(f"\n  [{nm}]")
        print(hdr)
        print("  f2023  " + "".join(f"{v:>+8.2f}" for v in d23[1:]))
        print("  f2024  " + "".join(f"{v:>+8.2f}" for v in d24[1:]))

    picks = {}
    print("\n" + "=" * 96)
    print("② 관문 판정")
    print("=" * 96)
    ok_all = True
    for nm in ("p", "b"):
        pl = plateau(*SW[nm])
        if pl is None or pl[2] < PLATEAU_MIN:
            print(f"  [{nm}] 고원 없음/부족 ({pl}) → 관문① 미달")
            ok_all = False
            continue
        lo, hi, w = pl
        beta = band_pick(lo, hi, *SW[nm])
        print(f"  [{nm}] 고원 [{lo:.4f}, {hi:.4f}] 폭 {w}칸  → 밴드 규칙(좌편향) β = {beta}")
        if beta is None:
            ok_all = False
        picks[nm] = beta
    # 위약 바닥 — [수정 1회차] 존재 → 크기 관문. 원판정(존재)도 같이 찍는다.
    floor = 0.0
    for nm in ("p_plb", "b_plb", "p_plb+", "b_plb+"):
        d23, d24 = SW[nm]
        pl = plateau(d23, d24)
        best_min = float(np.maximum(0, np.minimum(d23, d24))[1:].max())
        floor = max(floor, best_min)
        st = "없음" if pl is None or pl[2] < PLATEAU_MIN else f"있음 {pl}"
        print(f"  [위약 {nm}] 공통 양수 고원(원관문): {st}   min-fold Δ 최대 {best_min:+.2f}")
    if ok_all and len(picks) == 2:
        cand_min = min(min(SW[nm][0][list(GRID).index(picks[nm])],
                           SW[nm][1][list(GRID).index(picks[nm])])
                       for nm in ("p", "b"))
        ok_plb = cand_min > 3 * floor
        print(f"  위약 바닥 {floor:+.2f} × 3 = {3*floor:+.2f}  vs  후보 min-fold Δ "
              f"{cand_min:+.2f}  → {'통과' if ok_plb else '미달'} [수정 관문 — 증거 등급 하향]")
        ok_all = ok_all and ok_plb

    if not ok_all or len(picks) < 2:
        print("\n→ 사전 등록 관문 미달 — 기각. LB 슬롯 쓰지 않는다.")
        print(f"\n  총 {time.time()-t0:.0f}s")
        return

    bp, bb_ = picks["p"], picks["b"]
    print(f"\n채택 후보: β_p={bp}  β_b={bb_}")

    print("\n" + "=" * 96)
    print("③ 결합 2×2 + 크기비 + 위약 쌍체 부트스트랩 (B=4000)")
    print("=" * 96)
    res = {}
    for fold in D.FOLDS:
        b = F[fold]
        arms = {"p만": bp * V[fold]["p"],
                "b만": bb_ * V[fold]["b"],
                "결합": bp * V[fold]["p"] + bb_ * V[fold]["b"],
                # 위약은 부호가 임의 → 위약에 **유리한** +부호 팔로 결합 (적대적 바닥)
                "위약결합": bp * V[fold]["p_plb+"] + bb_ * V[fold]["b_plb+"]}
        ps, ds = [], {}
        for nm, dz in arms.items():
            d, p = delta(b, dz)
            ds[nm] = d
            ps.append(p)
        inter = ds["결합"] - ds["p만"] - ds["b만"]
        main_sum = abs(ds["p만"] + ds["b만"])
        point, draws = R.boot_delta(b["yv"], b["p"], ps, b["cl"], b["n_id"],
                                    np.random.default_rng(42), B=B_BOOT)
        names = list(arms)
        print(f"\n  ── fold {fold} ──")
        for j, nm in enumerate(names):
            lo, hi = np.percentile(draws[:, j], [2.5, 97.5])
            print(f"  {nm:<8s}{point[j]:>+8.2f}   [{lo:>+7.2f}, {hi:>+7.2f}]"
                  f"   P {np.mean(draws[:, j] > 0):.3f}")
        dp = draws[:, 2] - draws[:, 3]
        lo, hi = np.percentile(dp, [2.5, 97.5])
        print(f"  Δ(결합−위약) 쌍체 {point[2]-point[3]:>+8.2f}   [{lo:>+7.2f}, {hi:>+7.2f}]"
              f"   P {np.mean(dp > 0):.3f}")
        print(f"  상호작용 {inter:+.2f} (주효과 합 {main_sum:.2f} 의 "
              f"{abs(inter)/max(main_sum,1e-9):.0%})")
        res[fold] = dict(d=ds["결합"], pp=np.mean(dp > 0),
                         inter_ok=abs(inter) <= 0.25 * main_sum)

    ratio = max(res[2023]["d"], res[2024]["d"]) / max(min(res[2023]["d"],
                                                          res[2024]["d"]), 1e-9)
    ok3 = res[2023]["d"] > 0 and res[2024]["d"] > 0 and ratio <= 3
    ok4 = res[2024]["pp"] >= 0.9
    ok5 = res[2023]["inter_ok"] and res[2024]["inter_ok"]
    print("\n" + "=" * 96)
    print(f"판정  ③ 양fold 양수·크기비≤3: {'통과' if ok3 else '미달'} "
          f"(f23 {res[2023]['d']:+.2f} / f24 {res[2024]['d']:+.2f}, 비 {ratio:.2f})")
    print(f"      ④ 위약 쌍체 P≥.9 (f2024): {'통과' if ok4 else '미달'} ({res[2024]['pp']:.3f})")
    print(f"      ⑤ 상호작용 ≤25%: {'통과' if ok5 else '미달'}")
    if ok3 and ok4 and ok5:
        print(f"\n→ **후보 생존.** 기대 LB Δ = {res[2024]['d']:+.2f} × [0.5, 0.93] "
              f"= [{res[2024]['d']*0.5:+.2f}, {res[2024]['d']*0.93:+.2f}]. "
              f"챔피언 위 빌드로 진행.")
    else:
        print("\n→ 기각 — LB 슬롯 쓰지 않는다.")
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
