"""실패 모드 라벨 복원 — `asof_*` 누적 통계의 증분에서 투구별 라벨을 되찾는다

착상
    `control_success`는 **세 실패의 합집합**이다 (CLAUDE.md §1).
      ① 한가운데 몰림   ② 크게 벗어남   ③ 포수 요구 반대
    드라이버가 전부 다르다 — 3볼에서는 ①이 늘고 ②가 줄고, 2스트라이크에서는 반대다.
    이진 분류기 하나는 이 상반된 움직임을 **평균 내서** 학습한다.

    그런데 운영측이 준 `asof_pitcher_middle_rate` / `_reverse_rate`는 그 투구
    **직전까지의 누적** 비율이다. 같은 투수의 연속한 두 투구에서

        cnt = asof_pitcher_n × asof_pitcher_(mode)_rate

    의 **증분이 곧 그 사이 투구의 라벨**이다 (n이 정확히 1 늘었을 때).

리키지 판단
    복원된 라벨은 **학습 타깃**으로만 쓴다. 추론에는 라벨이 없고 필요하지도 않다.
    타깃을 더 잘게 쪼개는 것은 "현재 투구의 결과를 피처로 쓰는 것"과 다르다.
    (피처로 쓰면 실격이다. 타깃으로 쓰는 것은 그냥 다중 과제 학습이다.)

    ⚠️ 단, 복원 라벨로 **피처를 만들면 안 된다.** 그 순간 test에서 재현 불가가 되고
       (test에는 후속 투구가 없다) train/serve skew가 생긴다.

이 파일이 확인하는 것
    ① 증분이 실제로 {0, 1}에 떨어지는가 (부동소수점 반올림이 살아남는가)
    ② 몇 %의 투구에 대해 라벨을 복원할 수 있는가
    ③ 복원 라벨이 `control_success`와 정합하는가
       — 성공한 투구는 middle=0, reverse=0 이어야 한다 (정의상)
    ④ 모드별 비율이 카운트에 따라 실제로 반대로 움직이는가 (착상의 전제)

실행:  .venv/Scripts/python.exe -u src/modelabel.py 2>&1 | tee experiments/modelabel.log
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

from config import TARGET_COL, TRAIN_CSV
from data import load_csv

# (모드 이름, 누적 비율 컬럼)
MODES = [("middle", "asof_pitcher_middle_rate"),
         ("reverse", "asof_pitcher_reverse_rate"),
         ("ball", "asof_pitcher_ball_rate"),
         ("strike", "asof_pitcher_strike_rate"),
         ("success", "asof_pitcher_success_rate")]


def recover(df):
    """투구별 모드 라벨을 복원한다. 복원 불가 행은 NaN.

    같은 투수 안에서 `asof_pitcher_n` 으로 정렬한 뒤, n이 정확히 1 늘어난
    **다음 행**의 누적 카운트에서 현재 행의 카운트를 뺀다.
    """
    n = df["asof_pitcher_n"].to_numpy(dtype=np.float64)
    pid = df["pitcher_id"].to_numpy(dtype=np.int64)

    # 투수별 · n 오름차순 정렬 순서
    order = np.lexsort((n, pid))
    n_s, pid_s = n[order], pid[order]

    # 다음 행이 같은 투수이고 n이 정확히 1 큰가
    nxt_ok = np.zeros(len(df), dtype=bool)
    nxt_ok[:-1] = (pid_s[1:] == pid_s[:-1]) & (np.abs(n_s[1:] - n_s[:-1] - 1.0) < 1e-9)

    out = {}
    for name, col in MODES:
        r = df[col].to_numpy(dtype=np.float64)[order]
        cnt = n_s * r                      # 누적 개수 (실수)
        d = np.full(len(df), np.nan)
        d[:-1] = np.where(nxt_ok[:-1], (n_s[1:] * r[1:]) - cnt[:-1], np.nan)
        lab = np.full(len(df), np.nan)
        lab[order] = d                     # 원래 행 순서로 되돌린다
        out[name] = lab
    return out, order, nxt_ok


def aux_labels(df, modes=None):
    """(n, len(modes)) float32 보조 타깃. 복원 불가 행은 0으로 채운다.

    축 목록은 **`config.MLP_AUX_MODES` 하나가 정본이다.** 인자로 덮어쓰는 건
    실험 스크립트뿐이고, 배포 경로(train.py / verify_pipeline.py)는 인자를 주지
    않아 둘이 어긋날 수가 없다. (§7.36의 사고는 같은 값을 두 곳에서 따로 넘긴
    데서 났다 — 여기서는 아예 넘길 자리를 만들지 않는다.)

    **학습 타깃 전용이다.** 추론에는 없고 필요도 없다 (보조 헤드는 버린다).
    피처로 쓰면 test에 후속 투구가 없어 재현 불가고 실격이다.
    """
    from config import MLP_AUX_MODES
    modes = tuple(MLP_AUX_MODES if modes is None else modes)
    known = {m for m, _ in MODES}
    bad = [m for m in modes if m not in known]
    assert not bad, f"모르는 보조 축 {bad} (가능: {sorted(known)})"

    lab, _, _ = recover(df)
    cols = [np.round(lab[m]) for m in modes]
    # `success`는 축이 아니라 **복원 유효성 검사**다 (§③에서 100% 일치를 확인했다).
    ok = ~np.any([np.isnan(c) for c in cols] + [np.isnan(np.round(lab["success"]))],
                 axis=0)
    a = np.stack(cols, 1)
    a[~ok] = 0.0
    return a.astype(np.float32), float(ok.mean())


def main():
    df = load_csv(TRAIN_CSV)
    print(f"train {df.shape}")
    lab, order, ok = recover(df)
    print(f"후속 투구가 있는 행 {ok.sum():,} / {len(df):,} = {ok.mean():.1%}\n")

    print("=" * 88)
    print("① 증분이 {0,1}에 떨어지는가 — 부동소수점이 살아남았는가")
    print("=" * 88)
    for name, col in MODES:
        v = lab[name]
        m = ~np.isnan(v)
        if m.sum() == 0:
            print(f"  {name:<10} 복원 불가")
            continue
        vv = v[m]
        near = np.abs(vv - np.round(vv))
        r = np.round(vv)
        clean = (near < 0.01) & np.isin(r, [0.0, 1.0])
        print(f"  {name:<10} 복원 {m.sum():>9,}  "
              f"|증분−반올림| 중앙값 {np.median(near):.2e}  "
              f"최대 {near.max():.2e}   "
              f"**{{0,1}}에 떨어짐 {clean.mean():>6.2%}**   "
              f"양성률 {r[clean].mean():.4f}")

    print("\n" + "=" * 88)
    print("③ 복원 라벨이 control_success와 정합하는가")
    print("=" * 88)
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    mid, rev, suc = lab["middle"], lab["reverse"], lab["success"]
    m = ~(np.isnan(mid) | np.isnan(rev) | np.isnan(suc))
    mid_b, rev_b, suc_b = np.round(mid[m]), np.round(rev[m]), np.round(suc[m])
    yb = y[m]
    print(f"  복원 success == control_success ?  일치 {np.mean(suc_b == yb):.4%}")
    print(f"    ← 이게 100%면 복원 자체가 검증된다 (다른 경로로 같은 값을 얻음)")
    ok1 = suc_b == 1
    print(f"  성공 투구 중 middle=1 인 비율 {mid_b[ok1].mean():.4%}  "
          f"(정의상 0이어야 한다)")
    print(f"  성공 투구 중 reverse=1 인 비율 {rev_b[ok1].mean():.4%}")
    fail = suc_b == 0
    print(f"\n  실패 투구 {fail.sum():,}개의 모드 구성:")
    print(f"    middle  {mid_b[fail].mean():.4f}")
    print(f"    reverse {rev_b[fail].mean():.4f}")
    third = 1 - mid_b[fail] - rev_b[fail]
    print(f"    나머지(크게 벗어남) {third.mean():.4f}"
          f"   음수 {np.mean(third < 0):.2%}")

    print("\n" + "=" * 88)
    print("④ 착상의 전제 — 카운트에 따라 모드가 **반대로** 움직이는가")
    print("=" * 88)
    d = df.loc[m].copy()
    d["mid"], d["rev"], d["suc"] = mid_b, rev_b, suc_b
    d["off"] = 1 - d.suc - d.mid - d.rev
    b = d["balls_before"].clip(0, 3)
    s = d["strikes_before"].clip(0, 2)
    d["cnt"] = b.astype(str) + "-" + s.astype(str)
    g = d.groupby("cnt", observed=True)[["suc", "mid", "rev", "off"]].mean()
    g["n"] = d.groupby("cnt", observed=True).size()
    print(g.round(4).to_string())
    print("\n  → mid와 off가 카운트에 따라 **반대 방향**으로 움직이면,")
    print("     단일 이진 분류기는 그 둘을 상쇄시키며 학습하고 있는 것이다.")


if __name__ == "__main__":
    main()
