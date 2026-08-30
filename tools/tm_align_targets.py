"""train 행 ↔ 트랙맨 투구 정렬 → **투구 단위 보조 타깃** (§7.116 E4 · 논문 동기)

동기
    Manzi et al. 2021 (J Orthop 27:28-33) 은 제구 **일관성** 이 높은 투수와 낮은 투수를
    가르는 것이 구속(p=0.055, 유의하지 않음)이 아니라 **릴리스 기하(arm slot, p=0.009)**
    라고 보고한다. 우리 잔차 스크린도 구속에서 t≈0 을 냈다 — 논문과 정합.

    그런데 우리가 앞서 만든 것은 **투수-시즌 평균**이었고 0/13 이었다. 논문이 말하는
    "일관성" 은 평균이 아니라 **그 투구가 평소에서 얼마나 벗어났나** 다. 평균으로
    뭉개면 사라진다.

    투구 단위 값은 **피처로 못 쓴다** (2025 트랙맨 없음). 그러나 **학습 보조 타깃**
    으로는 쓸 수 있다 — 추론에 트랙맨이 0개 들어간다. 규정 확인됨:
    DACON.GM #26(8/14) · #48(8/20) · #10(8/10) 모두 허용.

타깃
    dev  = 그 투구의 릴리스점이 (투수, 시즌) 중심에서 벗어난 거리
    라벨 = dev > 그 (투수, 시즌) 의 중앙값        → {0,1}
           "이 투구는 평소보다 릴리스가 흔들렸는가"

    보조 헤드가 BCEWithLogits 이므로 이진화한다. 주 타깃은 hard 라벨 그대로다.

리키지
    타깃 전용. 피처로 쓰지 않는다 (test 에 트랙맨이 없어 재현 불가가 된다).
    정렬은 `control_success` 를 전혀 쓰지 않는다 — 경기·하프이닝·순서만 쓴다.
"""
import sys

import numpy as np
import pandas as pd


def build(work_src, out_pkl):
    sys.path.insert(0, work_src)
    import trackman_map as TM

    BSO = ["balls_before", "strikes_before", "outs_before"]
    m = TM.load_main(extra_cols=BSO + ["control_success", "row_id"])
    t = TM.load_tm(extra_cols=BSO + ["pitch_no", "rel_height", "rel_side",
                                     "rel_speed", "spin_rate", "pitch_type_group"])
    print(f"main {len(m):,} · tm {len(t):,}")

    mmeta, mprof = TM.profiles(m)
    tmeta, tprof = TM.profiles(t)
    pairs, _ = TM.match_games(mmeta, mprof, tmeta, tprof)
    tgame = pd.Series({k: v[0] for k, v in pairs.items()})
    cost = pd.Series({k: v[1] for k, v in pairs.items()})
    print(f"경기 매칭 {len(pairs):,}")

    # §7.156 과 같은 정렬: 파일 순서 = 경기 내 투구 순서. 하프이닝 안에서 순번으로 짝짓기
    m = m.sort_index()
    m["tgame"] = m.game.map(tgame)
    m["gcost"] = m.game.map(cost)
    m["seq"] = m.groupby(["game", "half"], observed=True).cumcount()
    t = t.sort_values(["game", "half", "pitch_no"]).reset_index(drop=True)
    t["seq"] = t.groupby(["game", "half"], observed=True).cumcount()

    right = t[["game", "half", "seq", "rel_height", "rel_side", "rel_speed",
               "spin_rate", "pitch_type_group"] + BSO].rename(
        columns={"game": "tgame", **{c: c + "_tm" for c in BSO}})
    j = m.merge(right, on=["tgame", "half", "seq"], how="left")

    ok = j["rel_height"].notna().to_numpy()
    agree = np.ones(len(j), dtype=bool)
    for c in BSO:
        agree &= (j[c].to_numpy() == j[c + "_tm"].to_numpy())
    trust = ok & agree
    print(f"짝 붙은 행 {ok.mean():.1%} · (b,s,o) 삼중 일치 {agree.mean():.1%} · 신뢰 {trust.mean():.1%}")

    # ── 릴리스 이탈 (논문의 "일관성") ────────────────────────
    j["abs_side"] = j["rel_side"].abs()
    g = j.groupby(["pitcher_id", "season"])
    for c in ("rel_height", "abs_side"):
        j[c + "_c"] = j[c] - g[c].transform("median")
    j["dev"] = np.hypot(j["rel_height_c"], j["abs_side_c"])
    j["dev_hi"] = (j["dev"] > g["dev"].transform("median")).astype(np.float32)
    j.loc[~trust, ["dev", "dev_hi"]] = np.nan
    print(f"dev_hi 평균 {j.loc[trust, 'dev_hi'].mean():.3f}  (0.5 근처여야 정상)")

    out = j[["row_id", "dev", "dev_hi", "rel_speed", "spin_rate", "pitch_type_group"]].copy()
    out["trust"] = trust
    out.to_pickle(out_pkl)
    print(f"저장 {out_pkl}  ({len(out):,}행)")
    return out


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
