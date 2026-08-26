"""정규화 완화 축 — **`min_data_in_leaf` 를 푸는 방향** (사용자 제안).

§7.18 은 `min_data 500` 에서 두 fold 다 양수(+4.9 / +0.1)를 얻고도 **크기비 49** 로
기각했다. 분모(+0.1)가 노이즈라 그 게이트는 여기서 발동하면 안 되는 것이었다.
그 뒤 스택이 150절 넘게 바뀌었고, 완화 방향은 아무도 다시 안 봤다.

**네 개의 독립 신호가 같은 쪽을 가리킨다:**
  §7.18   min_data 6000 (조임)  -1.1 / -1.7   두 fold 음수
  §7.18   min_data  500 (완화)  +4.9 / +0.1   두 fold 양수 (게이트 오작동으로 기각)
  혜진13-4 타깃 평활(정규화 대체)  -11.54  t=-7.62   배포급에서 부호 반전
  배포층   확률 스케일 최적 1.05 > 1.0        압축을 되돌리는 방향

**설계 — 혜진 라운드 13 의 규칙을 따른다** ("재현+위약 먼저, 결론은 나중").
  · 배포급 설정에서 잰다. 프록시 규제 수준이 다르면 부호까지 뒤집힌다 (혜진 13-4).
  · 하이퍼파라미터에는 '정체만 지운 위약'이 없다. 대신 **반대 방향이 대조군**이다 —
    풀면 오르고 조이면 내려가는 **단조 응답**이어야 진짜다. 한쪽만 오르면 불안정성.
  · 시드마다 4변형 미니앙상블을 따로 채점해 **쌍체 관측 n=7** 을 만든다. 같은 모델
    집합이라 추가 비용이 없는데 SE·t·CI 가 생긴다. §7.18 에 없던 것이 이것이다.
  · 측정 대상은 트리 단독이 아니라 **블렌드**다 (§7.18 의 핵심 설계를 유지).
  · `ENSEMBLE_VARIANTS` 가 명시한 값은 그대로 둔다 — 다양성 포트폴리오를 덮으면
    '완화 효과'와 '다양성 감소'가 한 숫자에 섞인다 (ablate19 의 주석 그대로).

실행:  .venv/Scripts/python.exe -u src/regloose.py 2024
"""
import gc
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
from config import (DROP_FEATURES, ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA,
                    N_ROUNDS, RECENCY_HALF_LIFE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts

mlp.THREADS = 8
OUT = Path(__file__).parent.parent / "experiments" / "cache"

SEEDS = (42, 7, 2024, 1234, 31337, 777, 31415)      # n=7 쌍체
EMB4 = {"pitcher_id": 4, "batter_id": 4, "pitcher_team_id": 4, "batter_team_id": 4,
        "base_state": 4, "hand_matchup": 4, "count_state": 4}
CFG_EMB4 = dict(epochs=9, hid=128, lr=2e-3, dropout=0.3, wd=1e-2, use_id_emb=True, emb_dim=EMB4)
CFG_NOID = dict(epochs=6, hid=128, lr=2e-3, dropout=0.3, wd=1e-2, use_id_emb=False, emb_dim=EMB4)

# 완화 사다리 + 반대 방향 대조군. 한 축씩만 흔든다.
CFGS = {
    "기준 md2000":   {},
    "md 1000":       dict(min_data_in_leaf=1000),
    "md 500":        dict(min_data_in_leaf=500),
    "md 250":        dict(min_data_in_leaf=250),
    "md 4000 [대조]": dict(min_data_in_leaf=4000),
}


def fold_data(df, valid_season):
    tr = df[df.season < valid_season]
    va = df[df.season == valid_season]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER, k_batter=SMOOTH_K_BATTER)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    return (X(tr), tr[TARGET_COL].to_numpy(np.float64), w,
            X(va), va[TARGET_COL].to_numpy(np.float64), art)


def main(fold):
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    Xtr, ytr, w, Xva, yva, art = fold_data(df, fold)
    del df; gc.collect()
    print(f"[{time.time()-t0:.0f}s] fold={fold}  train {Xtr.shape}  valid {Xva.shape}")

    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    zs = []
    for cfg, sd in ((CFG_EMB4, (42, 7, 2024)), (CFG_NOID, (42, 7))):
        for s in sd:
            _, curve = mlp.train_mlp(A, C, ytr, w, B, Cv, sizes, seed=s, verbose=False, **cfg)
            zs.append(curve[-1])
    z_mlp = np.mean(zs, axis=0)
    print(f"[{time.time()-t0:.0f}s] MLP {len(zs)}멤버 단독 {mlp.score(z_mlp, yva):.1f}")

    res = {}
    for name, over in CFGS.items():
        per_seed, z_pool = [], np.zeros(len(Xva))
        for s in SEEDS:
            z_s = np.zeros(len(Xva))
            for vname, vover in ENSEMBLE_VARIANTS.items():
                eff = {k: v for k, v in over.items() if k not in vover}   # 변형 값이 이긴다
                p = dict(FINAL_LGB_PARAMS); p.update(vover); p.update(eff)
                p.update(seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
                bst = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=N_ROUNDS)
                z_s += bst.predict(Xva, raw_score=True)
                del bst
            z_s /= len(ENSEMBLE_VARIANTS)
            per_seed.append(mlp.score((1 - MLP_ALPHA) * z_s + MLP_ALPHA * z_mlp, yva))
            z_pool += z_s
            gc.collect()
        z_pool /= len(SEEDS)
        pooled = mlp.score((1 - MLP_ALPHA) * z_pool + MLP_ALPHA * z_mlp, yva)
        res[name] = (np.array(per_seed), pooled)
        print(f"[{time.time()-t0:.0f}s] {name:<15} 풀드 {pooled:8.2f}   "
              f"시드별 {np.mean(per_seed):8.2f} ± {np.std(per_seed, ddof=1):.2f}")

    base_s, base_p = res["기준 md2000"]
    print(f"\n{'설정':<15}{'풀드Δ':>9}{'쌍체평균Δ':>11}{'SE':>8}{'t':>8}{'양수':>7}  95% CI")
    print("-" * 78)
    for name, (s, p) in res.items():
        if name == "기준 md2000":
            continue
        d = s - base_s
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se > 0 else 0.0
        lo, hi = d.mean() - 2.447 * se, d.mean() + 2.447 * se     # t_.975(6)=2.447
        print(f"{name:<15}{p-base_p:>9.2f}{d.mean():>11.2f}{se:>8.2f}{t:>8.2f}"
              f"{f'{(d>0).sum()}/7':>7}  [{lo:+.2f}, {hi:+.2f}]")
    np.savez_compressed(OUT / f"regloose_f{fold}.npz",
                        **{k: v[0] for k, v in res.items()},
                        **{f"pooled_{k}": v[1] for k, v in res.items()})
    print(f"\n총 {time.time()-t0:.0f}s   저장 experiments/cache/regloose_f{fold}.npz")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2024)
