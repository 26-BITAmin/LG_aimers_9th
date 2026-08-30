"""구종군 라벨 복원 — `asof_pitcher_pitchmix_*` 누적의 증분에서 (§7.116 E4 1단계)

착상은 `modelabel.py` 와 같다. 다만 기준 카운터가 `asof_pitcher_n` 이 아니라
**`asof_pitcher_pitchmix_n`** 이다 (구종이 기록된 투구만 세는 별도 카운터).

    cnt = asof_pitcher_pitchmix_n × asof_pitcher_{fastball,breaking,offspeed}_rate
    같은 투수에서 pitchmix_n 이 정확히 1 늘 때, 그 증분이 그 투구의 구종군

왜 이걸 하는가
    §7.116a E4 = "그 투구의 실제 구종을 **학습 시에만** 보조 타깃으로 쓴다".
    원우는 트랙맨 정렬을 전제로 미실행 상태로 뒀는데, **구종군은 train.csv 안에서
    복원된다** (라운드 13 이 99.89% 확인). 트랙맨 정렬 없이 먼저 이 축을 잴 수 있다.

    ⚠️ `§7.159` 의 distillation 과 다른 기전이다. 저기는 교사의 soft 확률로 **주 타깃을
       대체**해서 교사 편향에 죽었다. 여기는 주 타깃이 hard 라벨 그대로이고 보조 헤드가
       하나 늘 뿐이다 — 배포 중인 `middle`/`reverse` 와 같은 배선(§7.42).

    라벨은 **학습 타깃 전용**이다. 피처로 쓰면 test 에 후속 투구가 없어 재현 불가가 된다.
"""
import numpy as np

MODES = [("fastball", "asof_pitcher_fastball_rate"),
         ("breaking", "asof_pitcher_breaking_rate"),
         ("offspeed", "asof_pitcher_offspeed_rate")]
NCOL = "asof_pitcher_pitchmix_n"


def recover(df):
    n = df[NCOL].to_numpy(dtype=np.float64)
    pid = df["pitcher_id"].to_numpy(dtype=np.int64)
    order = np.lexsort((n, pid))
    n_s, pid_s = n[order], pid[order]
    nxt_ok = np.zeros(len(df), dtype=bool)
    nxt_ok[:-1] = (pid_s[1:] == pid_s[:-1]) & (np.abs(n_s[1:] - n_s[:-1] - 1.0) < 1e-9)

    out = {}
    for name, col in MODES:
        r = df[col].to_numpy(dtype=np.float64)[order]
        cnt = n_s * r
        d = np.full(len(df), np.nan)
        d[:-1] = np.where(nxt_ok[:-1], (n_s[1:] * r[1:]) - cnt[:-1], np.nan)
        lab = np.full(len(df), np.nan)
        lab[order] = d
        out[name] = lab
    return out


def aux_labels(df, round_tol=0.02):
    """(n, 3) float32 보조 타깃. 복원 불가·비정상 행은 0 으로 채운다.

    반환 (labels, coverage, diag)
    """
    rec = recover(df)
    M = np.stack([rec[k] for k, _ in MODES], axis=1)
    ok = np.isfinite(M).all(1)
    # 증분은 {0,1} 에 떨어져야 한다 (부동소수점 반올림 허용치 안에서)
    near = np.abs(M - np.round(M)) <= round_tol
    onehot = np.abs(M.sum(1) - 1.0) <= round_tol      # 정확히 한 구종
    good = ok & near.all(1) & onehot
    lab = np.zeros_like(M, dtype=np.float32)
    lab[good] = np.clip(np.round(M[good]), 0, 1)
    diag = dict(finite=float(ok.mean()), near01=float((ok & near.all(1)).mean()),
                onehot=float(good.mean()),
                mix=lab[good].mean(0) if good.any() else None)
    return lab, float(good.mean()), diag


if __name__ == "__main__":
    import sys
    import pandas as pd
    df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1
                     else "/Users/hyejin/Aimers_9th/data/train.csv", encoding="utf-8-sig")
    lab, cov, d = aux_labels(df)
    print(f"행 {len(df):,}")
    print(f"  유한 증분        {d['finite']:.4%}")
    print(f"  {{0,1}} 근접      {d['near01']:.4%}")
    print(f"  원핫(합=1)       {d['onehot']:.4%}   <- 최종 커버리지")
    print(f"  복원된 구종 분포  fastball {d['mix'][0]:.3f} · breaking {d['mix'][1]:.3f} · offspeed {d['mix'][2]:.3f}")
    for s in (2023, 2024):
        m = (df.season == s).to_numpy()
        l2, c2, _ = aux_labels(df[m].reset_index(drop=True))
        print(f"  [{s} 단독 복원] {c2:.4%}")
