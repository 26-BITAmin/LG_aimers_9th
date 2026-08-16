"""Trackman ↔ main 투수 ID 매핑.

`pitcher_trackman_id`와 `pitcher_id`는 값 범위가 완전히 다른 별개 체계다 (교집합 0).
직접 조인은 불가능하지만, 두 테이블은 **같은 경기들을 기록하고 있다.**
경기 → 하프이닝 → 투수 순으로 정렬해 대응을 읽어낸다.

파이프라인
    1. main의 행 순서로 경기를 복원한다 (행이 경기·이닝·투구 순으로 정렬돼 있다).
       검증: 복원된 경기의 하프이닝 수가 17/18에 몰리고 경기당 투구 수가 trackman과 일치.
    2. (시즌, 월, 요일) 버킷 안에서 **하프이닝별 투구 수 벡터(24차원)**를 지문으로
       경기를 매칭한다. 한 경기의 이닝별 투구 수 시퀀스는 거의 고유하다.
    3. 매칭이 옳은지 **팀 일관성**으로 검증한다 — 경기 매칭이 맞다면 홈/원정 팀 대응이
       모든 경기에서 같은 답을 내야 한다. (팀 매핑은 부산물로 공짜로 얻는다.)
    4. 매칭된 경기 안에서 (이닝, 초/말)별 투수의 투구 수를 대조해 ID 대응에 투표한다.
    5. 전역 투표 행렬에 헝가리안 배정을 적용해 최종 1:1 매핑을 만든다.
    6. 시즌 홀드아웃으로 검증한다 (2019~2022로 만든 매핑이 2023~2024에서도 맞는가).

리키지: 이 매핑은 `control_success`를 전혀 쓰지 않는다. 순수하게 구조적 대응이다.
       Trackman 피처를 만들 때의 시즌 경계는 별도로 지킨다.

실행:  .venv/Scripts/python.exe -u src/trackman_map.py
"""
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from config import DATA_DIR, MODEL_DIR

MAIN_COLS = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
             "pitcher_id", "pitcher_team_id", "batter_team_id", "pitcher_hand"]
TM_COLS = ["season", "game_month", "game_dayofweek", "trackman_game_id",
           "inning", "top_bottom", "pitcher_trackman_id", "pitcher_team",
           "batter_team", "pitcher_hand"]

N_HALF = 24          # 12이닝까지 (연장 포함). 그 이상은 잘라낸다.
GAME_COST_MAX = 30   # 하프이닝 투구 수 벡터의 L1 거리 허용치


def _read_csv(path, usecols, tries=6, wait=5.0):
    """PermissionError 재시도 래퍼.

    Windows에서 350MB급 CSV는 백신 실시간 검사가 공유 모드를 잠가 간헐적으로
    `PermissionError`를 던진다 (파일 권한 문제가 아니다 — os.access는 True다).
    수 초 뒤 풀리므로 재시도한다.
    """
    for i in range(tries):
        try:
            return pd.read_csv(path, usecols=usecols)
        except PermissionError:
            if i == tries - 1:
                raise
            print(f"    [retry {i+1}/{tries-1}] {Path(path).name} 잠김 — {wait:.0f}s 대기")
            time.sleep(wait)


def load_main(extra_cols=None):
    """extra_cols: 경기 복원에 필요한 컬럼 외에 함께 읽을 컬럼 (읽기 1회로 끝낸다)."""
    df = _read_csv(DATA_DIR / "train.csv", MAIN_COLS + list(extra_cols or []))
    pair = (np.minimum(df.pitcher_team_id, df.batter_team_id).astype(np.int64) * 100
            + np.maximum(df.pitcher_team_id, df.batter_team_id))
    key = df.season.astype(np.int64) * 10000 + df.game_month * 100 + df.game_dayofweek
    half = df.inning.astype(np.int64) * 2 + (df.top_bottom == "B")
    newgame = (pair != pair.shift()) | (key != key.shift()) | (half < half.shift())
    df["game"] = newgame.cumsum().astype(np.int32)
    df["half"] = half.astype(np.int16)
    df["is_top"] = (df.top_bottom == "T")
    return df


def load_tm(extra_cols=None):
    df = _read_csv(DATA_DIR / "trackman_history.csv", TM_COLS + list(extra_cols or []))
    df["half"] = (df.inning.astype(np.int64) * 2 + (df.top_bottom == "Bottom")).astype(np.int16)
    df["is_top"] = (df.top_bottom == "Top")
    df = df.rename(columns={"trackman_game_id": "game"})
    return df


def profiles(df):
    """경기별 (하프이닝 투구 수 24차원) 지문 + 버킷 키 + 홈/원정 팀."""
    piv = (df.groupby(["game", "half"], observed=True).size()
             .unstack("half", fill_value=0)
             .reindex(columns=range(2, N_HALF + 2), fill_value=0))
    meta = df.groupby("game", observed=True).agg(
        season=("season", "first"), month=("game_month", "first"),
        dow=("game_dayofweek", "first"))
    top = df[df.is_top]
    home = top.groupby("game", observed=True).agg(
        home=(("pitcher_team_id" if "pitcher_team_id" in df.columns else "pitcher_team"), "first"),
        away=(("batter_team_id" if "batter_team_id" in df.columns else "batter_team"), "first"))
    meta = meta.join(home)
    piv = piv.reindex(meta.index)
    return meta, piv.to_numpy(dtype=np.int32)


def match_games(mmeta, mprof, tmeta, tprof):
    """(시즌,월,요일) 버킷별로 헝가리안 배정. 반환: main game -> trackman game, cost."""
    out = {}
    mgroups = mmeta.groupby(["season", "month", "dow"], observed=True).indices
    tgroups = tmeta.groupby(["season", "month", "dow"], observed=True).indices
    costs = []
    for key, mi in mgroups.items():
        ti = tgroups.get(key)
        if ti is None:
            continue
        A, B = mprof[mi], tprof[ti]
        C = np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)
        r, c = linear_sum_assignment(C)
        for a, b in zip(r, c):
            out[mmeta.index[mi[a]]] = (tmeta.index[ti[b]], int(C[a, b]))
            costs.append(C[a, b])
    return out, np.array(costs)


def pitcher_votes(m, t, game_pairs):
    """매칭된 경기 안에서 하프이닝별 투수를 대조해 (pitcher_id, trackman_id)에 투표.

    한 하프이닝에 양쪽 모두 투수가 1명뿐이면 그 대응은 거의 확실하다 → 가중치 2.
    여러 명이면 투구 수 차이로 헝가리안 배정해 가중치 1로 투표한다.
    """
    mg = (m.groupby(["game", "half", "pitcher_id"], observed=True).size()
           .rename("n").reset_index())
    tg = (t.groupby(["game", "half", "pitcher_trackman_id"], observed=True).size()
           .rename("n").reset_index())
    mby = {k: v for k, v in mg.groupby("game", observed=True)}
    tby = {k: v for k, v in tg.groupby("game", observed=True)}

    vote = Counter()
    for gm, gt in game_pairs.items():
        A, B = mby.get(gm), tby.get(gt)
        if A is None or B is None:
            continue
        Ah = {h: d for h, d in A.groupby("half", observed=True)}
        Bh = {h: d for h, d in B.groupby("half", observed=True)}
        for h, da in Ah.items():
            db = Bh.get(h)
            if db is None:
                continue
            if len(da) == 1 and len(db) == 1:
                vote[(int(da.pitcher_id.iloc[0]), int(db.pitcher_trackman_id.iloc[0]))] += 2
            else:
                C = np.abs(da.n.to_numpy()[:, None] - db.n.to_numpy()[None, :])
                r, c = linear_sum_assignment(C)
                for a, b in zip(r, c):
                    vote[(int(da.pitcher_id.iloc[a]),
                          int(db.pitcher_trackman_id.iloc[b]))] += 1
    return vote


def solve_mapping(m, t, vote, min_votes=3, min_share=0.5):
    """전역 헝가리안 배정 + 손잡이 제약 + 득표 임계.

    채택 조건: 배정된 쌍의 득표가 min_votes 이상이고, 그 pitcher_id의 전체 득표 중
    min_share 이상을 차지할 것. 애매하면 매핑하지 않는다(피처는 결측 처리).
    """
    mids = sorted(m.pitcher_id.unique())
    tids = sorted(t.pitcher_trackman_id.dropna().unique().astype(int))
    mi = {v: i for i, v in enumerate(mids)}
    ti = {v: i for i, v in enumerate(tids)}
    V = np.zeros((len(mids), len(tids)), dtype=np.float64)
    for (a, b), n in vote.items():
        if a in mi and b in ti:
            V[mi[a], ti[b]] = n

    # 손잡이 제약: main 2 == Right (§tm_pitcher_map 실측)
    mh = m.groupby("pitcher_id").pitcher_hand.agg(lambda s: s.mode().iloc[0]).reindex(mids)
    th = (t.dropna(subset=["pitcher_trackman_id"])
            .astype({"pitcher_trackman_id": int})
            .groupby("pitcher_trackman_id").pitcher_hand
            .agg(lambda s: s.mode().iloc[0]).reindex(tids))
    ok = ((mh.to_numpy() == 2)[:, None] == (th.to_numpy() == "Right")[None, :])
    V = np.where(ok, V, 0.0)

    r, c = linear_sum_assignment(-V)
    row_tot = V.sum(axis=1)
    mapping = {}
    for a, b in zip(r, c):
        n = V[a, b]
        if n >= min_votes and row_tot[a] > 0 and n / row_tot[a] >= min_share:
            mapping[mids[a]] = tids[b]

    npitch = m.groupby("pitcher_id").size()
    cov = npitch.reindex(list(mapping)).sum() / npitch.sum()
    return mapping, {"pitch_cov": float(cov)}


def main():
    t0 = time.time()
    print("Loading ...")
    m, t = load_main(), load_tm()
    print(f"  main {m.shape}  trackman {t.shape}   ({time.time()-t0:.0f}s)")

    mmeta, mprof = profiles(m)
    tmeta, tprof = profiles(t)
    print(f"경기 수: main {len(mmeta):,}  trackman {len(tmeta):,}")

    print("\n=== 2. 경기 매칭 ===")
    pairs, costs = match_games(mmeta, mprof, tmeta, tprof)
    print(f"  배정된 경기 {len(pairs):,}쌍")
    print(f"  L1 비용 분위: {np.round(np.quantile(costs, [0.1,0.25,0.5,0.75,0.9,0.95]),1)}")
    for thr in (0, 5, 10, 20, 30, 50, 100):
        print(f"    cost <= {thr:>3d}: {(costs <= thr).sum():>5,}  ({(costs<=thr).mean():>5.1%})")

    good = {k: v[0] for k, v in pairs.items() if v[1] <= GAME_COST_MAX}
    print(f"  채택 (cost <= {GAME_COST_MAX}): {len(good):,}경기")

    # 정밀도 우선: 비용 0인 경기만 투수 투표에 쓴다
    exact = {k: v[0] for k, v in pairs.items() if v[1] == 0}
    print(f"  정밀 집합 (cost == 0): {len(exact):,}경기")

    print("\n=== 3. 팀 일관성 검증 (경기 매칭이 옳은지의 독립 증거) ===")
    print("  ※ 구단 명칭 변경(SK_WYV→SSG_LAN 등)이 있으므로 **시즌별**로 본다.")
    rows = []
    for mg, tg in good.items():
        s = mmeta.loc[mg, "season"]
        rows.append((s, mmeta.loc[mg, "home"], tmeta.loc[tg, "home"]))
        rows.append((s, mmeta.loc[mg, "away"], tmeta.loc[tg, "away"]))
    tv = pd.DataFrame(rows, columns=["season", "tid", "tstr"])
    agg = tv.groupby(["season", "tid", "tstr"]).size().rename("n").reset_index()
    best = agg.sort_values("n").groupby(["season", "tid"]).tail(1)
    tot = agg.groupby(["season", "tid"]).n.sum().rename("tot").reset_index()
    chk = best.merge(tot, on=["season", "tid"])
    chk["rate"] = chk.n / chk.tot
    print(f"  시즌별 (시즌,team_id) 조합 {len(chk)}개, 일치율 중앙값 {chk.rate.median():.1%}, "
          f"최소 {chk.rate.min():.1%}")
    print(f"  가중 일치율: {chk.n.sum()/chk.tot.sum():.2%}   ← 100%에 가까우면 경기 매칭이 옳다")
    print("\n  시즌별 team_id → trackman 팀 (상위 1개):")
    piv = chk.pivot_table(index="tid", columns="season", values="tstr", aggfunc="first")
    print(piv.to_string())
    print("\n  경기 수가 적은 (시즌,team_id):")
    print(chk.nsmallest(8, "tot")[["season", "tid", "tstr", "n", "tot", "rate"]].to_string(index=False))

    print("\n=== 4. 투수 매핑 투표 ===")
    vote = pitcher_votes(m, t, exact)
    print(f"  투표 총량 {int(sum(vote.values())):,}  (pitcher_id, trackman_id) 조합 {len(vote):,}개")

    print("\n=== 5. 시즌 홀드아웃 검증 ===")
    print("  2019~2022 경기로만 만든 매핑과 2023~2024 경기로만 만든 매핑이 일치하는가.")
    early = {k: v for k, v in exact.items() if mmeta.loc[k, "season"] <= 2022}
    late = {k: v for k, v in exact.items() if mmeta.loc[k, "season"] >= 2023}
    me = solve_mapping(m, t, pitcher_votes(m, t, early))[0]
    ml = solve_mapping(m, t, pitcher_votes(m, t, late))[0]
    both = set(me) & set(ml)
    agree = sum(me[k] == ml[k] for k in both)
    print(f"  early {len(me)}명, late {len(ml)}명, 양쪽 모두 매핑된 투수 {len(both)}명")
    print(f"  **일치 {agree}/{len(both)} = {agree/max(len(both),1):.2%}**"
          f"   ← 100%에 가까우면 매핑이 진짜다")

    mapping, diag = solve_mapping(m, t, vote)

    print("\n  투구 수 정합성 (매핑된 쌍의 시즌별 투구 수 비교):")
    mn = m.groupby("pitcher_id").size().rename("main_n")
    tn = t.dropna(subset=["pitcher_trackman_id"]).astype({"pitcher_trackman_id": int}) \
          .groupby("pitcher_trackman_id").size().rename("tm_n")
    cmp = pd.DataFrame({"main_n": mn.reindex(list(mapping)).to_numpy(),
                        "tm_n": tn.reindex([mapping[k] for k in mapping]).to_numpy()})
    ratio = cmp.tm_n / cmp.main_n.clip(lower=1)
    print(f"    tm/main 투구수 비율 분위: {np.round(np.nanquantile(ratio,[0.1,0.25,0.5,0.75,0.9]),2)}")
    print(f"    (퓨처스 경기가 trackman에만 있으므로 1.0 이상이 정상)")

    print(f"\n=== 6. 최종 매핑 ===")
    print(f"  매핑된 투수 {len(mapping):,}명 / main 투수 {m.pitcher_id.nunique():,}명")
    print(f"  main 투구 커버리지 {diag['pitch_cov']:.1%}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pd.Series(mapping, name="pitcher_trackman_id").rename_axis("pitcher_id") \
        .to_csv(MODEL_DIR / "pitcher_id_map.csv")
    print(f"  저장: {MODEL_DIR / 'pitcher_id_map.csv'}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
