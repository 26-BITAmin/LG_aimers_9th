"""§7.181 — ④류(스택 상대적) 기각의 T1 계보 재감사: 용량축·recency 노브 4팔

배경 (사용자: "지금까지 했던 것들 전수 재조사해야 하는 것 아님?")
    기각 증거를 4류로 분류하면 ① LB 실측(스택 불변) ② 기전·위약(데이터 성질)
    ③ 출력 스크린이 소급 커버 ④ **당시 스택 상대적**(corr·핸디캡·노브). ④만이
    스택이 바뀌면 무효가 되는 부류이고, 실제 역전 전례가 전부 여기다
    (§7.14→§7.140 CatBoost +1.09 · §7.19→§7.126 스케일 +2.45 · §7.172b→§7.176a λ2 +0.26).
    ④ 중 pre-§7.70(soft label 이전) 하네스에서 죽고 T1 계보에서 재측정 안 된
    노브: 용량축(§7.18 9전 9패) · RECENCY_HALF_LIFE(§7.23, §7.113은 init_score
    스택까지만) · SMOOTH_K(§7.31) · CTX_K(§7.64). 앞 둘을 이번에 잰다 —
    뒤 둘은 동결 artifacts 재적합이 필요한 별도 배관이라 다음 라운드.

팔 (전부 코어 LGBM 20 · s41 SMOTE 증강 동일 · rng 777 → 합성행 동일 → 쌍체,
    t1core.build 를 그대로 호출한다 — 재구현 없음 [[simplified-reimplementations]])
    base   s41 등가 (t1nrk_base.npz 재사용, 기지수 1524.1 대조)
    CAP2   num_leaves x2   — 변형별 31→62, 127→254 (min_data 2000 유지, 단일 노브)
    CAPH   num_leaves x1/2 — 31→15, 127→63
    HL035  RECENCY_HALF_LIFE 0.5 → 0.35 (w·soften dup 가중·SMOTE src 가중·init 재계산)
    HL100  RECENCY_HALF_LIFE 0.5 → 1.0

사전 등록 판정 (⚠️ fold2024 하나뿐 — 최종 판정자는 LB 슬롯)
    후보 = Δ vs base 양수 · P(Δ>0) ≥ 0.9 (투수 클러스터 부트스트랩 B=1000).
    4팔 x 게이트 10% → 우연 통과 기대 0.4개. 통과 팔은 B=4000 재확인
    ([[bootstrap-B-can-flip-a-borderline-gate]]) 후에만 LB 후보로 올린다.
    미달이면 "§7.18/§7.23 기각이 T1 계보에서도 유지"로 §7.181 에 기록하고 닫는다.
    노브를 fold 를 보고 다시 고르지 않는다 — 팔 4개가 가설의 전부다.

배관 검증 (사전)
    ⓐ base BSS 가 t1swp 실측 1524.1 과 일치 (같은 npz 재사용이므로 항등)
    ⓑ 각 팔 dup 행 수·SMOTE k 가 base 와 동일 (노브는 행 집합을 못 바꾼다)
    ⓒ 각 팔 vs base 최대|Δp| > 0

실행:  .venv/Scripts/python.exe -u src/t1knob.py > experiments/t1knob.log 2>&1
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
T1 = ROOT / "t1work"
ASCII = Path("D:/t1eval")
sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, N_ROUNDS,
                    RECENCY_HALF_LIFE, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from t1core import RATIO, SEEDS, VARIANTS, WORKER, build as t1core_build, t1_features
from validation import bss, shift_to_prior

VALID = 2024
B_BOOT = 1000
BASE_BSS_KNOWN = 1524.1          # t1swp/t1nrk 실측 — 같은 npz 라 일치해야 한다

# 팔 정의: (num_leaves 배율 | None, half_life | None)
ARMS = {
    "CAP2":  (2.0, None),
    "CAPH":  (0.5, None),
    "HL035": (None, 0.35),
    "HL100": (None, 1.00),
}


def leaf_scale(v, mult):
    """변형 v 의 유효 num_leaves 에 배율을 건다 (기본 31)."""
    nl = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v]).get("num_leaves", 31)
    return max(2, int(round(nl * mult)))


def main():
    t0 = time.time()
    art = joblib.load(T1 / "model" / "artifacts.joblib")["artifacts"]
    cols = lgb.Booster(model_str=(T1 / "model" / "lgbm_base_42.txt")
                       .read_text(encoding="utf-8")).feature_name()
    df = load_csv(TRAIN_CSV)
    tr = df[df.season < VALID].reset_index(drop=True)
    X = t1_features(tr, art, cols)
    print(f"피처 {X.shape}  ({time.time()-t0:.0f}s)")

    w_base = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                            ref_season=int(tr.season.max()))

    # ── 팔별 학습 ────────────────────────────────────────────────
    nd_base = None
    for a, (mult, hl) in ARMS.items():
        dst = ASCII / f"KNB_{a}"
        if (dst / "model" / f"lgbm_lowff_{SEEDS[-1]}.txt").exists():
            print(f"  {a} 재사용"); continue
        shutil.copytree(T1, dst, dirs_exist_ok=True)
        w = (w_base if hl is None else
             recency_weight(tr.season.to_numpy(), hl,
                            ref_season=int(tr.season.max())))
        X2, y2, w2, init, k = t1core_build(tr, X, w, True, np.random.default_rng(777))
        nd = len(X2) - len(X) - k
        if nd_base is None:
            nd_base = (nd, k)
        assert (nd, k) == nd_base, f"배관 ⓑ 실패 — {a} dup/k {(nd, k)} != {nd_base}"
        print(f"  [{a}] 행 {len(X):,} → {len(X2):,} (dup {nd:,} · SMOTE {k:,})"
              + (f"  hl={hl}" if hl else f"  leaves x{mult}"))
        for v in VARIANTS:
            p = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v])
            if mult is not None:
                p["num_leaves"] = leaf_scale(v, mult)
            for sd in SEEDS:
                p2 = dict(p, seed=sd, bagging_seed=sd,
                          feature_fraction_seed=sd, data_random_seed=sd)
                b = lgb.train(p2, lgb.Dataset(X2, y2, weight=w2, init_score=init),
                              num_boost_round=N_ROUNDS)
                # ⚠️ LF — write_text 의 CRLF 가 model_file= 을 죽인다 (§7.173 ③)
                (dst / "model" / f"lgbm_{v}_{sd}.txt").write_text(
                    b.model_to_string(), encoding="utf-8", newline="\n")
                del b
            print(f"    {v}"
                  + (f" nl={p['num_leaves']}" if mult is not None else "")
                  + f" ({time.time()-t0:.0f}s)")
        del X2, y2, w2, init
    del X, tr

    # ── 예측 (T1 자신의 predict.py, 팔 디렉터리에서) ─────────────
    wk = ROOT / "experiments" / "cache" / "_t1knob_worker.py"
    wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/")),
                  encoding="utf-8", newline="\n")
    P = {}
    for a, dname in [("base", "D_aug")] + [(x, f"KNB_{x}") for x in ARMS]:
        f = ROOT / "experiments" / "cache" / ("t1nrk_base.npz" if a == "base"
                                              else f"t1knob_{a}.npz")
        if not f.exists():
            r = subprocess.run([sys.executable, str(wk), dname, str(f)],
                               capture_output=True, text=True)
            if not f.exists():
                print(f"  ❌ {a}: {r.stderr.strip()[-300:]}"); return
        P[a] = np.load(f)["p"]
    print("  배관 ⓒ OK — base 대비 최대|Δp|: " +
          " ".join(f"{a} {np.abs(P[a]-P['base']).max():.5f}" for a in ARMS))
    for a in ARMS:
        assert np.abs(P[a] - P["base"]).max() > 0, f"{a} 가 base 와 같다"

    # ── 평가: fold2024, 투수 클러스터 부트스트랩 ─────────────────
    va = df[df.season == VALID].reset_index(drop=True)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    isF = va["game_type"].to_numpy().astype(str) == "F"
    codes, uniq = pd.factorize(va["pitcher_id"].to_numpy())
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    seen = np.where(np.diff(bnd) > 0)[0]
    rng = np.random.default_rng(42)
    boots = [np.concatenate([order[bnd[j]:bnd[j+1]] for j in
                             rng.choice(seen, size=len(seen), replace=True)])
             for _ in range(B_BOOT)]

    def s_(p, m=None, idx=None):
        pp, yy = (p, yva) if m is None else (p[m], yva[m])
        if idx is not None:
            pp, yy = pp[idx], yy[idx]
        return bss(yy, shift_to_prior(pp, yy.mean()))

    b0 = s_(P["base"])
    assert abs(b0 - BASE_BSS_KNOWN) < 0.5, f"배관 ⓐ 실패 — base {b0:.1f} != {BASE_BSS_KNOWN}"
    print("\n" + "=" * 92)
    print("§7.181 노브 재감사 (④류 소급) · 코어 LGBM 20 · fold2024 · ⚠️ fold 하나뿐 · B=%d" % B_BOOT)
    print("=" * 92)
    print("  %-30s %9s %10s %22s %8s" % ("팔", "BSS", "Δ vs s41", "95% 구간", "P(Δ>0)"))
    print("  %-30s %9.1f   (배관 ⓐ OK — 기지수 %.1f)" % ("base = s41", b0, BASE_BSS_KNOWN))
    desc = {"CAP2": "CAP2   num_leaves x2", "CAPH": "CAPH   num_leaves x0.5",
            "HL035": "HL035  half_life 0.35", "HL100": "HL100  half_life 1.0"}
    for a in ARMS:
        D = np.array([s_(P[a], idx=ix) - s_(P["base"], idx=ix) for ix in boots])
        lo, hi = np.percentile(D, [2.5, 97.5])
        pos = (D > 0).mean()
        flag = "  ← 후보(P≥.9)" if (s_(P[a]) - b0 > 0 and pos >= 0.9) else ""
        print("  %-30s %9.1f %+10.2f   [%+7.2f, %+7.2f] %8.3f%s"
              % (desc[a], s_(P[a]), s_(P[a]) - b0, lo, hi, pos, flag))
    print("\n  F행 / R행 / 예측 std 분해")
    for a in ["base"] + list(ARMS):
        print("    %-6s F %8.1f   R %8.1f   std(p) %.4f"
              % (a, s_(P[a], isF), s_(P[a], ~isF), P[a].std()))
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
