"""F 전문가 하이퍼파라미터 튜닝 — **위약 대조와 함께** (§7.102)

배경 (사용자 지적)
    "R하고 F를 구분시켰으니 early stopping·하이퍼파라미터 튜닝을 할 수 있지 않냐."
    R쪽(N_ROUNDS·SMOOTH_K·max_bin)은 §7.30·§7.31에서 이미 스윕됐고 평지였다
    — fold2024를 **커닝해서 고른** 변형별 최적조차 t=+0.5였다.

    그런데 지적이 가리키는 빈자리는 반대쪽이다. **FSPEC_PARAMS는 한 번도 스윕된 적이
    없다** — "5만 행이니 이 정도"로 손으로 정한 값이다. 그리고 F 전문가는 R쪽과 달리
    커닝 없이 고를 수단이 이미 있다 (fold "2023H2").

헤드룸의 크기
    fold2024에서 α=0.15(정직) → +2.53,  α=0.45(커닝) → +4.4.
    **그 차이 +1.9가 "선택을 더 잘하면 얻을 수 있는 상한"**이다.

설계 — 2023H2에서만 고르고 2024는 끝까지 안 본다
    좌표 하강 2회 (num_leaves · min_data_in_leaf · lr · feature_fraction · lambda_l2).
    각 후보마다 α를 같이 스윕한다 — 전문가가 좋아지면 최적 α도 올라가므로 두 축이
    같이 움직인다. 선택 기준은 **fold 전체 BSS**(fspec_verify와 같은 목적함수).

⚠️ 위약 대조가 필수다
    MODEL.md: 무의미한 버킷으로 서브그룹을 쪼개 튜닝하면 요일 같은 아무 축에서도
    +22.8이 나온다. 그래서 **행 태그만 F가 아닌 무작위 11.8%로 바꾼 팔**을 똑같이
    돌린다. 위약도 이득이 나오면 이건 F 얘기가 아니라 튜닝 자유도의 산물이다.

실행:  .venv/Scripts/python.exe -u src/fspec_tune.py 2>&1 | tee experiments/fspec_tune.log
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import lightgbm as lgb
import numpy as np

import foldcache
from config import (FSPEC_ALPHA, FSPEC_MIN_SEASON, FSPEC_PARAMS, FSPEC_SEEDS,
                    MLP_ALPHA, TARGET_COL, TRAIN_CSV)
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = ["2023H2", 2024]
INNER = "2023H2"                 # 여기서만 고른다
ALPHA_GRID = np.round(np.arange(0.0, 0.801, 0.05), 3)
PLACEBO_RATE = 0.118             # F 비율과 맞춘다
PLACEBO_SEED = 20260820

# 좌표 하강 축 — 현재 배포값에서 출발한다
AXES = [
    ("num_leaves", [7, 15, 31, 63]),
    ("min_data_in_leaf", [30, 100, 300, 1000]),
    ("learning_rate", [0.015, 0.03, 0.06]),
    ("feature_fraction", [0.5, 0.7, 0.9]),
    ("lambda_l2", [3.0, 10.0, 30.0, 100.0]),
]
N_PASS = 2


def fit_spec(Xtr, ytr, tt, Xva, params):
    """라운드 수는 학습셋 내부 시간분할(뒤 25%)로. 3시드 로짓 평균."""
    cut = np.quantile(tt, 0.75)
    itr, iva = tt <= cut, tt > cut
    if iva.sum() < 300 or len(np.unique(ytr[iva])) < 2:
        itr, iva = np.ones(len(ytr), bool), np.zeros(len(ytr), bool)
    zs = []
    for s in FSPEC_SEEDS:
        prm = dict(params, seed=s, bagging_seed=s, feature_fraction_seed=s,
                   data_random_seed=s)
        if iva.any():
            b0 = lgb.train(prm, lgb.Dataset(Xtr[itr], ytr[itr]), num_boost_round=3000,
                           valid_sets=[lgb.Dataset(Xtr[iva], ytr[iva])],
                           callbacks=[lgb.early_stopping(100, verbose=False)])
            nr = max(50, b0.best_iteration)
        else:
            nr = 200
        zs.append(lgb.train(prm, lgb.Dataset(Xtr, ytr),
                            num_boost_round=nr).predict(Xva, raw_score=True))
    return np.mean(zs, axis=0)


def main():
    df = load_csv(TRAIN_CSV)
    cache = foldcache.load(FOLDS)
    rng = np.random.default_rng(PLACEBO_SEED)
    placebo_tag = rng.random(len(df)) < PLACEBO_RATE      # 행 고정 태그
    t0 = time.time()

    D = {}
    for f in FOLDS:
        z_lgb, z_mlp, y, _ = cache[f]
        z_dep = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
        mtr, mva = foldcache.SPLITS[f](df.season.to_numpy(), df.game_month.to_numpy())
        tr, va = df[mtr], df[mva]
        Xtr, ytr, _, Xva, yva, _, _ = foldcache.fold_data(df, f)
        Xtr_np = Xtr.to_numpy(dtype=np.float32)
        Xva_np = Xva.to_numpy(dtype=np.float32)
        tt = (tr["season"].to_numpy(dtype=np.int64) * 12
              + tr["game_month"].to_numpy(dtype=np.int64))
        recent = tr["season"].to_numpy() >= FSPEC_MIN_SEASON
        tag = {
            "F": ((tr["game_type"].to_numpy().astype(str) == "F"),
                  (va["game_type"].to_numpy().astype(str) == "F")),
            "위약": (placebo_tag[mtr], placebo_tag[mva]),
        }
        D[f] = dict(z_dep=z_dep, y=yva, Xtr=Xtr_np, ytr=ytr, Xva=Xva_np,
                    tt=tt, recent=recent, tag=tag)
        print(f"  fold {f}: 검증 {len(yva):,}행  "
              + "  ".join(f"{k} 학습 {int((m & recent).sum()):,}/검증 {int(v.sum()):,}"
                          for k, (m, v) in tag.items())
              + f"   ({time.time()-t0:.0f}s)")

    def score(f, z_spec, vmask, a):
        d = D[f]
        z = d["z_dep"].copy()
        z[vmask] = (1 - a) * z[vmask] + a * z_spec
        return bss(d["y"], shift_to_prior(1 / (1 + np.exp(-z)), d["y"].mean()))

    def run(f, group, params):
        """그 fold에서 전문가를 학습하고 (최고점, 최적 α, 기준점)을 돌려준다."""
        d = D[f]
        m, v = d["tag"][group]
        m = m & d["recent"]
        z = fit_spec(d["Xtr"][m], d["ytr"][m], d["tt"][m], d["Xva"][v], params)
        base = score(f, z, v, 0.0)
        sc = [score(f, z, v, a) for a in ALPHA_GRID]
        k = int(np.argmax(sc))
        return sc[k], float(ALPHA_GRID[k]), base, z, v

    print(f"\n  기준: 현재 배포 설정 (α={FSPEC_ALPHA})")
    for group in ("F", "위약"):
        best = dict(FSPEC_PARAMS)
        s0, a0, b0, _, _ = run(INNER, group, best)
        print(f"\n  === {group} — 좌표 하강 (선택은 {INNER}에서만) ===")
        print(f"    출발  {s0:>8.1f} (α={a0:.2f}, 기준 {b0:.1f}, Δ {s0-b0:+.1f})")
        cur = s0
        for p in range(N_PASS):
            for axis, vals in AXES:
                cands = []
                for v_ in vals:
                    if v_ == best[axis]:
                        cands.append((cur, v_, None))
                        continue
                    trial = dict(best, **{axis: v_})
                    sc, a, bb, _, _ = run(INNER, group, trial)
                    cands.append((sc, v_, a))
                cands.sort(key=lambda t: -t[0])
                top, val, aa = cands[0]
                if val != best[axis]:
                    print(f"    p{p+1} {axis:<18} {best[axis]} → {val}   "
                          f"{cur:.1f} → {top:.1f}  (α={aa:.2f})  ({time.time()-t0:.0f}s)")
                    best[axis], cur = val, top
        s1, a1, b1, _, _ = run(INNER, group, best)
        print(f"    최종  {s1:>8.1f} (α={a1:.2f}, Δ {s1-b1:+.1f})   {best}")

        # ── 2024 적용 (여기서 처음 본다) ─────────────────────
        print(f"    ── fold2024 적용 ──")
        for name, prm, a in (("현재 배포", dict(FSPEC_PARAMS), FSPEC_ALPHA),
                             ("튜닝 결과", best, a1)):
            _, _, b24, z24, v24 = run(2024, group, prm)
            s24 = score(2024, z24, v24, a)
            print(f"      {name:<10} α={a:.2f}   {b24:.1f} → {s24:.1f}   Δ {s24-b24:+.2f}")


if __name__ == "__main__":
    main()
