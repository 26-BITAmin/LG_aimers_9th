"""라운드 5 — 세그먼트 레벨 분해의 **전이** 검정 (`MODEL.md §7.123`).

전역 레벨은 이미 최적이다(§7.122). 남은 것은 세그먼트 간 편차:

    이득 = K · w_1 · w_2 · (b_1 − b_2)²,    b_g = p̄_g − r_g

핵심 질문은 "격차가 있나"가 아니라 **"직전 시즌의 격차가 다음 시즌에 전이되나"**다.
`§7.71`이 학습기에게 F−R 레벨차를 맡기는 것을 기각한 이유가 **시즌마다 뒤집힌다**
였으므로, 그 기각을 뒤집으려면 전이를 직접 보여야 한다.

세 팔을 나란히 잰다.
  oracle    : 그 fold 자신의 라벨로 Δ_g 를 잡는다        → 천장 (전이 아님)
  transfer  : **직전 시즌**의 격차를 그대로 가져다 쓴다   → 실제로 쓸 수 있는 것
  placebo   : 요일 버킷. §7.45 에서 가짜 이득 +22.8 을 낸 축 → 영가설

전역 중립성을 지킨다: Δ_R = −w_F·D̂, Δ_F = +w_R·D̂ (가중평균 이동 = 0).
그래서 이 팔들은 **레벨 축이 이미 가져간 이득을 다시 세지 않는다.**
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import deploy29 as D                                            # noqa: E402
from config import TARGET_COL, TRAIN_CSV                        # noqa: E402
from validation import bss                                      # noqa: E402


def seg_gain(p0, yv, is_a, dhat):
    """전역 중립 차등 시프트 D̂ 를 확률공간에서 적용했을 때의 ΔBSS.

    is_a 가 True 인 쪽이 `1`, 나머지가 `2`.  Δ_1 = −w_2·D̂,  Δ_2 = +w_1·D̂.
    """
    w1 = float(is_a.mean()); w2 = 1.0 - w1
    p = p0 + np.where(is_a, -w2 * dhat, w1 * dhat)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return bss(yv, p) - bss(yv, p0)


def buckets(d):
    """축 이름 → 불리언 마스크. 요일·월은 위약(§7.45 에서 죽은 것이 확인됨)."""
    dow = d["game_dayofweek"].to_numpy()
    mon = d["game_month"].to_numpy()
    out = {"game_type": d["game_type"].to_numpy().astype(str) == "F"}
    # 위약은 **질량을 F 에 맞춘다** — 이득이 w1·w2 에 비례하므로 안 맞추면 비교 불가
    target = out["game_type"].mean()
    for name, arr in (("placebo_dow", dow), ("placebo_month", mon)):
        vals, cnt = np.unique(arr, return_counts=True)
        frac = cnt / cnt.sum()
        pick = vals[np.argmin(np.abs(frac - target))]
        out[name] = arr == pick
    return out


def prev_gap(df, season, mask_fn):
    """직전 시즌의 (그룹1 − 그룹2) 라벨 평균 격차. **평가 시즌 라벨은 안 본다.**"""
    prev = df[df.season == season - 1]
    if len(prev) == 0:
        return None
    m = mask_fn(prev)
    y = prev[TARGET_COL].to_numpy(dtype=np.float64)
    if m.sum() == 0 or (~m).sum() == 0:
        return None
    return float(y[m].mean() - y[~m].mean())


def main():
    df, s, y = D.load_df()
    # 위약 정의는 fold 와 무관하게 같아야 한다 — 전체에서 한 번 고른다
    ref = buckets(df)
    picks = {k: None for k in ref}
    dow_v = df["game_dayofweek"].to_numpy(); mon_v = df["game_month"].to_numpy()
    tgt = ref["game_type"].mean()
    for name, arr in (("placebo_dow", dow_v), ("placebo_month", mon_v)):
        vals, cnt = np.unique(arr, return_counts=True)
        picks[name] = vals[np.argmin(np.abs(cnt / cnt.sum() - tgt))]
    print(f"위약 선택: 요일={picks['placebo_dow']}  월={picks['placebo_month']}  "
          f"(목표 질량 {tgt:.4f})")

    def mask_of(name):
        if name == "game_type":
            return lambda x: x["game_type"].to_numpy().astype(str) == "F"
        col = "game_dayofweek" if name == "placebo_dow" else "game_month"
        return lambda x, c=col, v=picks[name]: x[c].to_numpy() == v

    print("\n시즌별 격차 (그룹1 − 그룹2, 라벨 실측):")
    hdr = f"  {'시즌':>6s}" + "".join(f"{n:>16s}" for n in ref)
    print(hdr)
    for ss in sorted(df.season.unique()):
        sub = df[df.season == ss]
        row = f"  {ss:>6d}"
        for n in ref:
            m = mask_of(n)(sub)
            yy = sub[TARGET_COL].to_numpy(dtype=np.float64)
            row += f"{yy[m].mean() - yy[~m].mean():>+16.5f}"
        print(row)

    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        d, p0, yv = st["d"], st["p0"], st["yv"]
        print(f"\n{'='*74}\nfold {fold}   기준 BSS {st['base']:.2f}   n={len(yv):,}")
        print(f"  {'축':<16s}{'w1':>8s}{'p̄1−p̄2':>10s}{'r1−r2':>10s}"
              f"{'oracle':>10s}{'transfer':>10s}{'천장대비':>9s}")
        for name in ref:
            m = mask_of(name)(d)
            w1 = m.mean()
            dp = p0[m].mean() - p0[~m].mean()
            dr = yv[m].mean() - yv[~m].mean()
            g_or = seg_gain(p0, yv, m, dp - dr)                    # 천장
            gp = prev_gap(df, fold, mask_of(name))
            g_tr = seg_gain(p0, yv, m, dp - gp) if gp is not None else np.nan
            ratio = (g_tr / g_or * 100) if g_or > 1e-9 else np.nan
            print(f"  {name:<16s}{w1:>8.4f}{dp:>+10.5f}{dr:>+10.5f}"
                  f"{g_or:>+10.3f}{g_tr:>+10.3f}{ratio:>8.1f}%")


if __name__ == "__main__":
    main()
