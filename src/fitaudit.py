"""§7.62 — **train에서 fit하는 모든 상수를 전수 감사한다.**

왜: 직전 두 이득이 둘 다 여기서 나왔다.
    §7.60  asof gap 축이 오염 안 보정된 컬럼을 읽었다      → LB +19.28
    §7.61  δ 표에 시즌 가중이 없었다                      → LB  +2.30

두 가지를 묻는다.
    ① **시간 가중** — 모델은 recency hl=0.5로 학습한다. 이 상수도 최근 시즌을
       반영하나, 아니면 2019~2022가 66%를 차지하나.
    ② **오염 보정** — pre-2023 F는 라벨과 `asof_*`가 둘 다 오염돼 있다.
       이 상수가 원본 컬럼을 보나, 보정된 값을 보나.

**모든 상수에 둘 다 필요한 것은 아니다.** 판정 기준:

    값이 예측에 **직접** 들어가면 (δ, gap, 스무딩 prior) → 편향이 그대로 남는다. 중요.
    값이 **학습되는 층 앞의 affine 변환**이면 (MLP mu/sd) → 1층 가중·바이어스가
        흡수한다. 편향이 아니라 컨디셔닝 문제다. **안 중요.**
    값이 **경계/코드**면 (clip, cat_maps) → 순서만 쓰이므로 둔감하다.

실행:  .venv/Scripts/python.exe -u src/fitaudit.py 2>&1 | tee experiments/fitaudit.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import contam
import features as F
from config import (CONTAM_AXES, RECENCY_HALF_LIFE, SMOOTH_K_PITCHER, TARGET_COL,
                    TRAIN_CSV)
from data import load_csv, recency_weight

# 오염이 어느 컬럼을 어느 방향으로 틀리게 했나 (§7.49).
#   성공률은 반대성 실패를 성공으로 세었으므로 과대 → −1
#   반대성 비율은 그만큼 과소                      → +1
CADJ = {
    "asof_pitcher_success_rate": ("p", -1.0),
    "asof_pitcher_reverse_rate": ("p", +1.0),
    "asof_batter_success_rate": ("b", -1.0),
}


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    w = recency_weight(s, RECENCY_HALF_LIFE, ref_season=2024)
    ct = F.fit_artifacts(df, contam_axes=CONTAM_AXES)["contam"]

    print("=" * 96)
    print("감사 대상 — train에서 fit되어 추론까지 얼려지는 상수")
    print("=" * 96)
    print(f"  {'상수':<34}{'시간가중':>10}{'오염보정':>10}   {'예측에 직접 쓰이나'}")
    rows = [
        ("art['priors'] (스무딩 리그평균)", "✅ w", "❌ 원본", "예 — 저표본 행의 대체값"),
        ("art['smooth_k']", "—", "—", "상수 (200)"),
        ("art['cat_maps']", "❌", "—", "코드값. 순서만 쓰인다"),
        ("art['clip']", "미사용", "미사용", "train.py가 clip= 을 안 넘긴다"),
        ("art['contam']  (q_s)", "❌", "자기 자신", "예 — 보정량 자체"),
        ("art['ctx'] δ 표", "✅ hl=1.0", "상쇄됨", "예 — §7.61에서 고쳤다"),
        ("art['asof'] gap (mu/sd/offset)", "—", "✅ §7.60", "예 — §7.60에서 고쳤다"),
        ("mlp enc mu/sd", "❌", "❌", "**아니오** — 1층이 흡수한다"),
        ("mlp enc cat_luts", "❌", "—", "코드값. 미등장은 0"),
        ("modelabel.aux_labels", "—", "❌ 원본", "학습 타깃 (추론엔 없음)"),
        ("logit_delta (prior shift)", "2024 참조행", "—", "예 — 스칼라"),
    ]
    for a, b, c, d in rows:
        print(f"  {a:<34}{b:>10}{c:>10}   {d}")

    # ── ① art['priors'] — 오염이 얼마나 밀어올렸나 ────────────────────────
    print("\n" + "=" * 96)
    print("① art['priors'] — 스무딩 리그평균이 오염된 컬럼에서 나온다")
    print("=" * 96)
    print("  저표본 투수/타자는 자기 통계 대신 이 값으로 끌려간다 (강도 200).")
    print("  CLAUDE.md가 'asof_* 결측 처리가 성능에 직결'이라 짚어둔 그 지점이다.\n")
    print(f"  {'컬럼':<34}{'원본':>10}{'보정':>10}{'차':>10}{'차/sd':>8}")
    worst = []
    for col, (ax, sign) in CADJ.items():
        v = df[col].to_numpy(dtype=np.float64)
        dm = contam.adjust(df, ct, ax)
        vc = v + sign * dm
        m = np.isfinite(v)
        p0 = float(np.average(v[m], weights=w[m]))
        p1 = float(np.average(vc[m], weights=w[m]))
        sd = float(np.std(v[m]))
        worst.append((col, p0, p1, p1 - p0, (p1 - p0) / sd))
        print(f"  {col:<34}{p0:>10.5f}{p1:>10.5f}{p1-p0:>+10.5f}{(p1-p0)/sd:>8.3f}")
    print("\n  스무딩이 실제로 얼마나 움직이나 — n이 작을 때가 최대다")
    print(f"  {'n(통산 투구)':<14}" + "".join(f"{c.split('_')[1][:1]+c.split('_')[2][:3]:>12}"
                                              for c, *_ in worst))
    for n in (0, 10, 50, 200, 1000):
        line = f"  {n:<14}"
        for col, p0, p1, d, _ in worst:
            k = SMOOTH_K_PITCHER
            # (k·prior + n·r) / (n + k) 의 prior 부분만 바뀐다 → 변화 = k/(n+k) · Δprior
            line += f"{k / (n + k) * d:>+12.5f}"
        print(line)

    # ── ② modelabel — 보조 타깃의 '반대' 라벨 ─────────────────────────────
    print("\n" + "=" * 96)
    print("② modelabel.aux_labels — '반대' 축은 **오염된 그 컬럼**에서 복원된다")
    print("=" * 96)
    import modelabel
    lab, _, _ = modelabel.recover(df)
    gt = df["game_type"].astype("object").to_numpy()
    con = (s <= 2022) & (gt == "F")
    for m in ("middle", "reverse"):
        if m not in lab:
            continue
        v = np.round(lab[m])
        ok = np.isfinite(v)
        print(f"  {m:<10} 전체 {np.nanmean(v[ok]):.4f}   "
              f"오염구간 {np.nanmean(v[ok & con]):.4f}   "
              f"깨끗한구간 {np.nanmean(v[ok & ~con]):.4f}")
    print("\n  → 오염 구간의 '반대' 라벨이 낮으면 그 행들이 **틀린 타깃**으로 학습된다.")
    print("     다만 MLP 손실에 recency 가중이 들어가므로 pre-2023의 실효 비중은")
    print(f"     {w[con].sum() / w.sum() * 100:.2f}% 다 (행 비중 {con.mean()*100:.1f}%).")

    print("\n" + "=" * 96)
    print("판정")
    print("=" * 96)
    print("  ✅ 후보 1 — art['priors'] 를 오염 보정된 컬럼에서 뽑는다.")
    print("     피처를 바꾸므로 **사후 가산이 아니다** → evaluate 하네스로 재학습 측정 필요.")
    print("  ❌ mlp enc mu/sd — 편향이 아니다. 1층 가중·바이어스가 흡수한다.")
    print("     (학습·추론이 같은 상수를 쓰므로 affine 재매개화일 뿐이다.)")
    print("  ❌ art['clip'] — train.py가 clip= 을 넘기지 않아 애초에 비활성이다.")
    print("  ⚠️ 후보 2 — modelabel 보조 타깃. 실효 비중이 작아 기대값도 작다.")


if __name__ == "__main__":
    main()
