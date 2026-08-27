"""§7.183a — 용량 축 2라운드: leaves × λ2 **2×2 격자** (§7.181 CAP2 승리 후속)

❌ 실측 후기: s46_c4l10 = LB 1119.0384 (−0.4511). 챔피언 s45 유지, 용량 축 종결.

§7.182 실측: CAP2(leaves ×2) LB +1.0189 (전이 배율 0.59, 예측 밴드 정중앙).
이제 챔피언 코어 = CAP2 이므로 판정 기준점을 CAP2 로 옮긴다. 두 질문:
    ① 꼭짓점이 ×2 위인가 — CAP4 (×4)
    ② §7.176a 의 λ2 30→10 (+0.26, leaves ×1 에서 측정) 이 새 코어에서도 유효한가
       — 용량↑ 와 정규화↓ 는 같은 방향 조작이라 주효과 덧셈이 안 된다
       ([[count-what-a-new-correction-double-subtracts]]). 2×2 로 상호작용을 직접 잰다.

팔 (코어 LGBM 20 · s41 SMOTE 동일 · rng 777 → 쌍체 · t1core.build 직접 호출)
    base   s41 등가 (t1nrk_base.npz 캐시)          leaves ×1 · λ30
    CAP2   t1knob_CAP2.npz 캐시 재사용             leaves ×2 · λ30  ← 판정 기준점
    CAP4   신규                                    leaves ×4 · λ30  (31→124 · 127→508)
    C2L10  신규                                    leaves ×2 · λ10
    C4L10  신규                                    leaves ×4 · λ10
    λ10 은 4변형 전부에 적용 (extra 는 원래 λ10 이라 불변).

사전 등록 판정 (⚠️ fold2024 하나뿐 — 최종 판정자는 LB 슬롯)
    후보 = 신규 3칸 중 argmax. 채택 조건: Δ(argmax − CAP2) > 0 · P ≥ 0.9
    (투수 클러스터 부트스트랩 B=1000, 경계선이면 B=4000 재확인).
    통과 → 그 설정으로 s46 빌드. 미달 → 챔피언 코어(×2·λ30) 유지,
    λ2=10 미이식으로 기록 (§7.176a 는 leaves ×1 국한으로 좁힌다).

실행:  .venv/Scripts/python.exe -u src/t1knob2.py > experiments/t1knob2.log 2>&1
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
from t1core import SEEDS, VARIANTS, WORKER, build as t1core_build, t1_features
from validation import bss, shift_to_prior

VALID = 2024
B_BOOT = 1000

# (leaf_mult, lambda_l2 | None=변형 기본값)
ARMS = {"CAP4": (4.0, None), "C2L10": (2.0, 10.0), "C4L10": (4.0, 10.0)}
CACHED = {"base": "t1nrk_base.npz", "CAP2": "t1knob_CAP2.npz"}


def leaf_scale(v, mult):
    nl = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v]).get("num_leaves", 31)
    return max(2, int(round(nl * mult)))


def main():
    t0 = time.time()
    art = joblib.load(T1 / "model" / "artifacts.joblib")["artifacts"]
    cols = lgb.Booster(model_str=(T1 / "model" / "lgbm_base_42.txt")
                       .read_text(encoding="utf-8")).feature_name()
    df = load_csv(TRAIN_CSV)
    tr = df[df.season < VALID].reset_index(drop=True)
    need_train = any(not (ASCII / f"KN2_{a}" / "model" / f"lgbm_lowff_{SEEDS[-1]}.txt").exists()
                     or not (ASCII / f"KN2_{a}" / "_done").exists() for a in ARMS)
    if need_train:
        X = t1_features(tr, art, cols)
        w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                           ref_season=int(tr.season.max()))
        X2, y2, w2, init, k = t1core_build(tr, X, w, True, np.random.default_rng(777))
        print(f"피처 {X.shape} · 행 {len(X2):,} (SMOTE {k:,})  ({time.time()-t0:.0f}s)")
        del X
        for a, (mult, l2) in ARMS.items():
            dst = ASCII / f"KN2_{a}"
            if (dst / "_done").exists():
                print(f"  {a} 재사용"); continue
            shutil.copytree(T1, dst, dirs_exist_ok=True)
            for v in VARIANTS:
                p = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v])
                p["num_leaves"] = leaf_scale(v, mult)
                if l2 is not None:
                    p["lambda_l2"] = l2      # extra 는 원래 10 → 불변
                for sd in SEEDS:
                    p2 = dict(p, seed=sd, bagging_seed=sd,
                              feature_fraction_seed=sd, data_random_seed=sd)
                    b = lgb.train(p2, lgb.Dataset(X2, y2, weight=w2, init_score=init),
                                  num_boost_round=N_ROUNDS)
                    # LF — CRLF 는 model_file= 을 죽인다 (§7.173 ③)
                    (dst / "model" / f"lgbm_{v}_{sd}.txt").write_text(
                        b.model_to_string(), encoding="utf-8", newline="\n")
                    del b
                print(f"    [{a}] {v} nl={p['num_leaves']} l2={p.get('lambda_l2')} "
                      f"({time.time()-t0:.0f}s)")
            (dst / "_done").write_text("ok", encoding="utf-8")
        del X2, y2, w2, init
    del tr

    wk = ROOT / "experiments" / "cache" / "_t1knob2_worker.py"
    wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/")),
                  encoding="utf-8", newline="\n")
    P = {}
    for a in list(CACHED) + list(ARMS):
        f = ROOT / "experiments" / "cache" / CACHED.get(a, f"t1knob2_{a}.npz")
        if not f.exists():
            r = subprocess.run([sys.executable, str(wk), f"KN2_{a}", str(f)],
                               capture_output=True, text=True)
            if not f.exists():
                print(f"  ❌ {a}: {r.stderr.strip()[-300:]}"); return
        P[a] = np.load(f)["p"]
    for a in ARMS:
        assert np.abs(P[a] - P["CAP2"]).max() > 0, f"{a} 가 CAP2 와 같다"
    print("  배관 OK — CAP2 대비 최대|Δp|: " +
          " ".join(f"{a} {np.abs(P[a]-P['CAP2']).max():.5f}" for a in ARMS))

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

    print("\n" + "=" * 96)
    print("§7.183 leaves × λ2 2×2 · 코어 LGBM 20 · fold2024 · 판정 기준 = CAP2 · B=%d" % B_BOOT)
    print("=" * 96)
    print("  %-26s %9s %11s %22s %8s %11s" %
          ("팔", "BSS", "Δ vs CAP2", "95% 구간", "P(Δ>0)", "Δ vs base"))
    b0, c0 = s_(P["base"]), s_(P["CAP2"])
    print("  %-26s %9.1f %11s %22s %8s %+11.2f" % ("base  ×1·λ30", b0, "—", "", "", 0.0))
    print("  %-26s %9.1f %+11.2f %22s %8s %+11.2f"
          % ("CAP2  ×2·λ30 (챔피언)", c0, 0.0, "(기준점)", "", c0 - b0))
    desc = {"CAP4": "CAP4  ×4·λ30", "C2L10": "C2L10 ×2·λ10", "C4L10": "C4L10 ×4·λ10"}
    for a in ARMS:
        D = np.array([s_(P[a], idx=ix) - s_(P["CAP2"], idx=ix) for ix in boots])
        lo, hi = np.percentile(D, [2.5, 97.5])
        pos = (D > 0).mean()
        flag = "  ← 후보" if (s_(P[a]) - c0 > 0 and pos >= 0.9) else ""
        print("  %-26s %9.1f %+11.2f   [%+7.2f, %+7.2f] %8.3f %+11.2f%s"
              % (desc[a], s_(P[a]), s_(P[a]) - c0, lo, hi, pos, s_(P[a]) - b0, flag))
    print("\n  상호작용 (λ10 의 한계효과): ×2 에서 %+.2f · ×4 에서 %+.2f"
          % (s_(P["C2L10"]) - s_(P["CAP2"]), s_(P["C4L10"]) - s_(P["CAP4"])))
    print("  용량의 한계효과 (×2→×4): λ30 에서 %+.2f · λ10 에서 %+.2f"
          % (s_(P["CAP4"]) - s_(P["CAP2"]), s_(P["C4L10"]) - s_(P["C2L10"])))
    print("\n  F행 / R행 / std")
    for a in ["base", "CAP2"] + list(ARMS):
        print("    %-6s F %8.1f   R %8.1f   std(p) %.4f"
              % (a, s_(P[a], isF), s_(P[a], ~isF), P[a].std()))
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
