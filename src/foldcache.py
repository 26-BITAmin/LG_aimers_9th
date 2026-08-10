"""fold별 예측(z_lgb, z_mlp, y)을 한 번만 계산해 디스크에 캐시한다.

ablate17·18·19가 **같은 fold 모델을 세 번 학습**했다. 한 번에 20분씩이다.
블렌드 가중·캘리브레이션·후처리처럼 **z만 있으면 되는 실험**은 학습이 필요 없다.
여기서 z를 얼려두면 그런 실험이 즉시 끝난다.

캐시 키에 설정을 박아 넣는다 — 피처 수·앙상블 변형·시드·MLP 구성이 바뀌면
다른 파일이 된다. 안 그러면 낡은 z로 새 설정을 평가하게 된다.

⚠️ 여기 담기는 것은 **train 시즌으로 만든 홀드아웃 예측**이다. test와 무관하다.

실행:  .venv/Scripts/python.exe -u src/foldcache.py 2023 2024
"""
import gc
import hashlib
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

import mlp
from config import (DROP_FEATURES, ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS,
                    MLP_SEEDS, MLP_VARIANTS, N_ROUNDS, RECENCY_HALF_LIFE,
                    SEED_ENSEMBLE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
LGB_SEEDS = SEED_ENSEMBLE[:3]        # fold 실험은 3시드 (배포는 5시드)


def _key():
    """설정 지문. 하나라도 바뀌면 캐시가 무효가 된다."""
    s = repr([sorted(FINAL_LGB_PARAMS.items()), N_ROUNDS, RECENCY_HALF_LIFE,
              sorted(DROP_FEATURES), USE_PLAYER_IDS, LGB_SEEDS,
              sorted((k, sorted(v.items())) for k, v in ENSEMBLE_VARIANTS.items()),
              sorted((k, sorted(c.items()), n) for k, (c, n) in MLP_VARIANTS.items()),
              MLP_SEEDS, SMOOTH_K_PITCHER, SMOOTH_K_BATTER])
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def fold_data(df, valid_season):
    tr = df[df.season < valid_season]
    va = df[df.season == valid_season]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    return (X(tr), tr[TARGET_COL].to_numpy(dtype=np.float64), w,
            X(va), va[TARGET_COL].to_numpy(dtype=np.float64), art)


def part(f, kind, tag):
    return CACHE / f"part_fold{f}_{_key()}_{kind}_{tag}.npz"


def compute(f, df, t0):
    Xtr, ytr, w, Xva, yva, art = fold_data(df, f)

    pl = part(f, "lgb", "all")
    if pl.exists():
        z_lgb = np.load(pl)["z"]
        print(f"  lgbm 재사용  단독 {mlp.score(z_lgb, yva):.1f}  ({time.time()-t0:.0f}s)")
    else:
        z_lgb, n = np.zeros(len(Xva)), 0
        for over in ENSEMBLE_VARIANTS.values():
            for s in LGB_SEEDS:
                p = dict(FINAL_LGB_PARAMS, **over, seed=s, bagging_seed=s,
                         feature_fraction_seed=s, data_random_seed=s)
                bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w),
                                num_boost_round=N_ROUNDS)
                z_lgb += bst.predict(Xva, raw_score=True)
                n += 1
                del bst
        z_lgb /= n
        np.savez_compressed(pl, z=z_lgb)
        print(f"  lgbm {n}개  단독 {mlp.score(z_lgb, yva):.1f}  ({time.time()-t0:.0f}s)")

    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    del Xtr, Xva
    gc.collect()

    # MLP도 멤버 단위로 즉시 저장한다. 20분짜리 루프 도중에 끊기면
    # (세션 종료·OOM 어느 쪽이든) 여기까지 한 일이 통째로 날아간다.
    zs = []
    for vname, (cfg, nseed) in MLP_VARIANTS.items():
        for s in MLP_SEEDS[:nseed]:
            pm = part(f, "mlp", f"{vname}_{s}")
            if pm.exists():
                zs.append(np.load(pm)["z"])
                print(f"  mlp {vname}_{s} 재사용  ({time.time()-t0:.0f}s)")
                continue
            _, curve = mlp.train_mlp(A, C, ytr, w, B, Cv, sizes, seed=s,
                                     verbose=False, **cfg)
            zs.append(curve[-1])
            np.savez_compressed(pm, z=zs[-1])
            print(f"  mlp {vname}_{s} 완료  단독 {mlp.score(zs[-1], yva):.1f}"
                  f"  ({time.time()-t0:.0f}s)")
    z_mlp = np.mean(zs, axis=0)
    print(f"  mlp {len(zs)}개 평균  단독 {mlp.score(z_mlp, yva):.1f}"
          f"  ({time.time()-t0:.0f}s)")
    return z_lgb, z_mlp, yva


def load(folds, verbose=True):
    """캐시가 있으면 읽고, 없으면 계산해서 저장한 뒤 돌려준다.

    fold 전체가 끝나야 저장하는 구조였다가 한 번 크게 손해를 봤다 — MLP 학습 중
    세션이 내려가면서 LGBM 12개(349초)가 통째로 날아갔다. 지금은 **부분 산출물을
    그때그때 저장**하고, 다시 돌리면 있는 것부터 주워 담는다."""
    CACHE.mkdir(parents=True, exist_ok=True)
    k = _key()
    out, need = {}, []
    for f in folds:
        p = CACHE / f"fold{f}_{k}.npz"
        if p.exists():
            with np.load(p) as d:
                out[f] = (d["z_lgb"], d["z_mlp"], d["y"])
            if verbose:
                print(f"  캐시 적중: {p.name}")
        else:
            need.append(f)
    if need:
        t0 = time.time()
        df = load_csv(TRAIN_CSV)
        for f in need:
            print(f"\n=== fold {f} 계산 (캐시 없음) ===")
            z_lgb, z_mlp, y = compute(f, df, t0)
            np.savez_compressed(CACHE / f"fold{f}_{k}.npz",
                                z_lgb=z_lgb, z_mlp=z_mlp, y=y)
            for p in CACHE.glob(f"part_fold{f}_{k}_*.npz"):
                p.unlink()          # 완성됐으니 조각은 치운다
            out[f] = (z_lgb, z_mlp, y)
            gc.collect()
    return out


if __name__ == "__main__":
    folds = [int(x) for x in sys.argv[1:]] or [2023, 2024]
    d = load(folds)
    print(f"\n지문 {_key()}   캐시 위치 {CACHE}")
    for f, (a, b, y) in d.items():
        print(f"  fold {f}: n={len(y):,}  lgb {mlp.score(a, y):.1f}"
              f"  mlp {mlp.score(b, y):.1f}")
