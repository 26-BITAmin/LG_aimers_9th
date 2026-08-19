"""game_type 피처의 R/F 행별 효과 — soft label 이후 재평가 (§7.70 후속 질문)

질문 (사용자)
    "game_type을 넣으면 R에서는 적합도가 올라가고 F에서는 폭락하는가?"

기존 기록의 한계
    §4·evaluate.py의 fold 단위 숫자(유지 −798.4 vs 제거 −11.1, fold2023)는
    hl·ids가 같이 바뀐 비교였고, R행/F행 분해가 없다. 단일 변수로 잰 적이 없다.

새 변수
    soft label(§7.70)이 오염 구간 F의 유효 성공률을 .71 → ~.50으로 내렸다.
    game_type이 독이었던 이유가 "학습기 F 레벨(.71) ≠ 배포기 F 레벨(.46)"이라면,
    soft label이 그 간극을 거의 지웠으므로 **독성이 사라졌을 수 있다.**

설계 — 2×2 (gt 유지/제거 × soft 유무), fold 2024·2023, R/F 행별 subBSS

실행:  .venv/Scripts/python.exe -u src/gtcheck.py 2>&1 | tee experiments/gtcheck.log
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

import soften as soften_mod
from config import (CONTAM_AXES, RECENCY_HALF_LIFE, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from softlabel import train_eval

FOLDS = [2024, 2023]
SEEDS = (42, 7, 2024)
ARMS = [("드랍+soft (배포)", False, True), ("유지+soft", True, True),
        ("드랍+원라벨", False, False), ("유지+원라벨", True, False)]


def main():
    df = load_csv(TRAIN_CSV)
    t0 = time.time()
    res = {}
    for fold in FOLDS:
        print(f"\n=== fold {fold} ===")
        tr = df[df.season < fold]
        va = df[df.season == fold]
        w0 = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                            ref_season=int(tr.season.max()))
        art = fit_artifacts(tr, weights=w0, k_pitcher=SMOOTH_K_PITCHER,
                            k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

        def X(d, keep_gt):
            x = build_features(d, art)
            if USE_PLAYER_IDS:
                for c in ("pitcher_id", "batter_id"):
                    x[c] = d[c].to_numpy(dtype=np.float32)
            return x if keep_gt else x.drop(columns=["game_type"])

        yva = va[TARGET_COL].to_numpy(dtype=np.float64)
        va_f = va["game_type"].to_numpy().astype(str) == "F"
        ytr0 = tr[TARGET_COL].to_numpy(dtype=np.float64)

        for name, keep, soft in ARMS:
            Xtr, ytr, w = X(tr, keep), ytr0, w0
            if soft:
                Xtr, ytr, w, _, _ = soften_mod.soften(tr, Xtr, ytr, w, None)
            r = train_eval(Xtr, ytr, w, X(va, keep), yva, va_f, SEEDS)
            res[(name, fold)] = r
            print(f"  {name:<14} BSS {r[:,0].mean():>8.1f} ± {r[:,0].std(ddof=1):>5.1f}"
                  f"   F행 {r[:,1].mean():>8.1f}   R행 {r[:,2].mean():>8.1f}"
                  f"   ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 76)
    print("game_type **유지 − 드랍** 쌍체 Δ (양수 = 유지가 이득)")
    print("=" * 76)
    for soft, tag in [(True, "soft"), (False, "원라벨")]:
        keep_n = f"유지+{tag}" if soft else "유지+원라벨"
        drop_n = f"드랍+{tag}" if soft else "드랍+원라벨"
        keep_n = "유지+soft" if soft else "유지+원라벨"
        drop_n = "드랍+soft (배포)" if soft else "드랍+원라벨"
        for fold in FOLDS:
            d = res[(keep_n, fold)] - res[(drop_n, fold)]
            print(f"  {tag:<5} f{fold}  전체 {d[:,0].mean():>+7.1f}±{d[:,0].std(ddof=1)/np.sqrt(3):.1f}"
                  f"   F행 {d[:,1].mean():>+8.1f}   R행 {d[:,2].mean():>+7.1f}")


if __name__ == "__main__":
    main()
