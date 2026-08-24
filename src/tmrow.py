"""Lever 3 · 1단계 — **투구 단위** 매칭율·커버리지 감사 (§7.156).

§4 Tier 3 이 푼 것은 **선수** 매핑이다 (706/792, 투구 99.7% 커버). distillation 은
그것보다 강한 것을 요구한다 — **train 행 하나하나에 그 투구의 trackman 측정**이
붙어야 teacher 가 `(x, x*)` 를 볼 수 있다.

### 정렬 재료

  · 메인 테이블은 경기·이닝·투구 순으로 정렬돼 있다 (§4 에서 확인·복원됨)
  · 경기 매칭은 이미 풀려 있다 (하프이닝 투구수 24차원 지문, 70.5% 가 L1=0)
  · trackman 은 `pitch_no` 로 경기 내 순서를 직접 준다

→ 매칭된 경기의 **같은 하프이닝 안에서 순서로 짝짓는다.**

### ★ 독립 검증이 가능하다 — 이게 이 감사의 핵심이다

양쪽 테이블이 **둘 다** `balls_before` · `strikes_before` · `outs_before` 를 갖는다.
정렬에 **쓰지 않고** 남겨두면, 짝지어진 두 행의 (b,s,o) 삼중 일치율이
**정렬이 맞았는지에 대한 외부 증거**가 된다. 순서로만 맞춘 뒤 사후에 채점한다.

무작위 짝짓기의 기대 일치율을 같이 낸다 (같은 하프이닝 안에서 셔플) — 그것이
이 지표의 영가설이다. 저장소 관행대로 위약을 붙인다.

### 기각 조건 (이 단계)

사용자 설계: "매칭율이 무의미하게 낮다" 하나뿐이다. 커버리지가 부분적이어도
커버된 행만 soft, 나머지는 hard 로 λ-혼합하면 되므로 100% 가 필요하지 않다.

실행:  .venv/Scripts/python.exe -u src/tmrow.py 2>&1 | tee experiments/tmrow.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np
import pandas as pd

import trackman_map as TM

MAPS = Path(__file__).parent.parent / "model_s28"      # §4 ID 매핑 자산
CACHE = Path(__file__).parent.parent / "experiments" / "cache"

BSO = ["balls_before", "strikes_before", "outs_before"]


def main():
    print("=== 1. 로드 · 경기 복원 ===")
    m = TM.load_main(extra_cols=BSO + ["game_type", "control_success", "batter_id"])
    t = TM.load_tm(extra_cols=BSO + ["pitch_no", "pitch_of_pa", "rel_speed",
                                     "spin_rate", "pitch_type_group",
                                     "batter_trackman_id"])
    print(f"  main {len(m):,}행 · 복원 경기 {m.game.nunique():,}")
    print(f"  tm   {len(t):,}행 · 경기 {t.game.nunique():,}")

    print("\n=== 2. 경기 매칭 (§4 자산 재사용) ===")
    mmeta, mprof = TM.profiles(m)
    tmeta, tprof = TM.profiles(t)
    pairs, costs = TM.match_games(mmeta, mprof, tmeta, tprof)
    cost = pd.Series({k: v[1] for k, v in pairs.items()})
    tgame = pd.Series({k: v[0] for k, v in pairs.items()})
    print(f"  매칭된 main 경기 {len(pairs):,} / {m.game.nunique():,}")
    for c in (0, 2, 5, 10, 30):
        print(f"    L1 ≤ {c:<3d}  {(cost <= c).mean():6.1%}")

    # 경기 비용을 행에 붙인다. 비용이 큰 경기는 정렬 자체를 신뢰할 수 없다.
    m["tgame"] = m.game.map(tgame)
    m["gcost"] = m.game.map(cost)

    print("\n=== 3. 하프이닝 투구 수 일치 ===")
    mc = m.groupby(["game", "half"], observed=True).size().rename("n_main")
    tc = t.groupby(["game", "half"], observed=True).size().rename("n_tm")
    hh = mc.reset_index()
    hh["tgame"] = hh.game.map(tgame)
    hh = hh.merge(tc.reset_index().rename(columns={"game": "tgame"}),
                  on=["tgame", "half"], how="left")
    hh["ok"] = hh.n_main == hh.n_tm
    hh["gcost"] = hh.game.map(cost)
    print(f"  하프이닝 {len(hh):,}개 중 투구 수 정확히 일치 {hh.ok.mean():6.1%}")
    for c in (0, 5, 30):
        s = hh[hh.gcost <= c]
        print(f"    경기 L1 ≤ {c:<3d} 인 하프이닝 {len(s):>8,}  일치 {s.ok.mean():6.1%}")

    print("\n=== 4. 투구 단위 정렬 — 하프이닝 안 순서로 짝짓기 ===")
    m = m.sort_index()                       # 파일 순서 = 경기 내 투구 순서 (§4)
    t = t.reset_index(drop=True)
    t["tm_idx"] = np.arange(len(t), dtype=np.int64)
    m["seq"] = m.groupby(["game", "half"], observed=True).cumcount()
    t = t.sort_values(["game", "half", "pitch_no"])
    t["seq"] = t.groupby(["game", "half"], observed=True).cumcount()

    left = m[["tgame", "half", "seq", "season", "game_type", "gcost"] + BSO].copy()
    right = t[["game", "half", "seq", "tm_idx", "pitch_type_group", "rel_speed"] + BSO].rename(
        columns={"game": "tgame", **{c: c + "_tm" for c in BSO}})
    j = left.merge(right, on=["tgame", "half", "seq"], how="left")
    j["hit"] = j.rel_speed.notna() | j.pitch_type_group.notna()
    agree = np.ones(len(j), dtype=bool)
    for c in BSO:
        agree &= (j[c].to_numpy() == j[c + "_tm"].to_numpy())
    j["agree"] = agree & j.hit

    print(f"  짝이 붙은 행      {j.hit.mean():6.1%}  ({int(j.hit.sum()):,} / {len(j):,})")
    print(f"  ★ (b,s,o) 삼중 일치 {j.agree.mean():6.1%}  ← 정렬에 안 쓴 컬럼. 외부 증거")
    hit = j[j.hit]
    print(f"  짝이 붙은 것 중 일치 {hit.agree.mean():6.1%}")

    # 위약 — 같은 하프이닝 안에서 seq 를 셔플한 뒤 같은 채점
    rng = np.random.default_rng(42)
    perm = t.copy()
    perm["seq"] = perm.groupby(["game", "half"], observed=True)["seq"].transform(
        lambda s: rng.permutation(s.to_numpy()))
    pr = perm[["game", "half", "seq"] + BSO].rename(
        columns={"game": "tgame", **{c: c + "_tm" for c in BSO}})
    jp = left.merge(pr, on=["tgame", "half", "seq"], how="left")
    ap = np.ones(len(jp), dtype=bool)
    for c in BSO:
        ap &= (jp[c].to_numpy() == jp[c + "_tm"].to_numpy())
    print(f"  위약(하프이닝 내 셔플) 삼중 일치 {ap.mean():6.1%}  ← 영가설")

    # ── ★ 독립 신호 2 — 투수·타자 정체 대조 ───────────────────────
    # (b,s,o) 삼중은 엔트로피가 낮다. **2스트라이크 파울은 삼중이 그대로**라
    # 파울 연속 구간의 off-by-one 을 구분하지 못한다. §4 의 ID 매핑 자산으로
    # 완전히 다른 신호(선수 정체)를 한 번 더 건다.
    pm = pd.read_csv(MAPS / "pitcher_id_map.csv").set_index("pitcher_id")["pitcher_trackman_id"]
    bm = pd.read_csv(MAPS / "batter_id_map.csv").set_index("batter_id")["batter_trackman_id"]
    left2 = m[["tgame", "half", "seq", "pitcher_id", "batter_id"]].copy()
    left2["p_exp"] = left2.pitcher_id.map(pm)
    left2["b_exp"] = left2.batter_id.map(bm)
    right2 = t[["game", "half", "seq", "pitcher_trackman_id",
                "batter_trackman_id"]].rename(columns={"game": "tgame"})
    j2 = left2.merge(right2, on=["tgame", "half", "seq"], how="left")
    okp = (j2.p_exp.notna() & j2.pitcher_trackman_id.notna()).to_numpy()
    okb = (j2.b_exp.notna() & j2.batter_trackman_id.notna()).to_numpy()
    j["p_ok"] = okp & (j2.p_exp == j2.pitcher_trackman_id).to_numpy()
    j["b_ok"] = okb & (j2.b_exp == j2.batter_trackman_id).to_numpy()
    print("\n  ── 독립 신호 2: 선수 정체 (§4 ID 매핑 자산) ──")
    print(f"    투수 정체 일치 {j.p_ok.to_numpy()[okp].mean():6.1%}  (매핑 가능 {okp.mean():.1%})")
    print(f"    타자 정체 일치 {j.b_ok.to_numpy()[okb].mean():6.1%}  (매핑 가능 {okb.mean():.1%})")
    j["trust"] = j.agree & j.b_ok & j.p_ok
    print(f"    ★ (b,s,o) + 투수 + 타자 **동시** 일치 {j.trust.mean():6.1%}"
          f"  ← 파울 off-by-one 까지 막는다")

    print("\n=== 5. 커버리지 — 시즌 × game_type (삼중 일치 기준) ===")
    piv = j.pivot_table(index="season", columns="game_type", values="trust",
                        aggfunc="mean")
    cnt = j.pivot_table(index="season", columns="game_type", values="trust",
                        aggfunc="size")
    print(f"  {'시즌':<7s}{'R 행':>10s}{'R 일치':>9s}{'F 행':>10s}{'F 일치':>9s}")
    for yr in piv.index:
        rr = piv.loc[yr].get("R", np.nan)
        ff = piv.loc[yr].get("F", np.nan)
        nr = cnt.loc[yr].get("R", 0)
        nf = cnt.loc[yr].get("F", 0)
        print(f"  {yr:<7d}{nr:>10,d}{rr:>9.1%}{nf:>10,d}{ff:>9.1%}")

    print("\n=== 6. 경기 매칭 품질별 — 어디를 버려야 하나 ===")
    print(f"  {'경기 L1':<10s}{'행':>12s}{'짝':>9s}{'삼중 일치':>11s}{'+정체':>11s}")
    for lo, hi in ((0, 0), (1, 2), (3, 5), (6, 30), (31, 10**9)):
        s = j[(j.gcost >= lo) & (j.gcost <= hi)]
        if len(s):
            print(f"  {f'{lo}–{hi}':<10s}{len(s):>12,d}{s.hit.mean():>9.1%}"
                  f"{s.agree.mean():>11.1%}{s.trust.mean():>11.1%}")
    keep = j[j.gcost <= 30]
    print(f"\n  ⇒ 경기 L1 ≤ 30 만 남기면 {len(keep):,}행 중 신뢰 {keep.trust.mean():.2%}")
    ok = (j.trust & (j.gcost <= 30)).to_numpy()
    print(f"  ⇒ **soft label 을 붙일 수 있는 train 행: {ok.mean():.1%} ({int(ok.sum()):,}행)**")

    # ── 2단계가 쓸 조인 결과를 얼린다 (train 행 순서 그대로) ──────────
    # tm_idx = `load_tm()` 이 읽은 순서에서의 위치. 같은 usecols 로 다시 읽으면
    # 같은 순서가 나온다 (pandas read_csv 는 결정적). trust=False 행은 −1.
    out = CACHE / "tmrow_join.npz"
    tm_idx = j.tm_idx.to_numpy(dtype=np.float64)
    tm_idx = np.where(ok & np.isfinite(tm_idx), tm_idx, -1).astype(np.int64)
    np.savez_compressed(out, tm_idx=tm_idx, trust=ok)
    print(f"  조인 저장: {out.name}  (신뢰 {int(ok.sum()):,}행)")


if __name__ == "__main__":
    main()
