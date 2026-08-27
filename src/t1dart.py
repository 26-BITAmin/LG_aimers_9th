"""§7.186 — DART 부스팅 T1 재감사 (Kaggle 스윕 §7.185 후속 · 사전 등록)

출처: Amex Default Prediction 상위권의 주력 부스팅 모드가 DART 였다.
이 저장소에서 부스팅 모드는 `rf` 사고(§7.25) 외에 측정된 적이 없다.
④류 재감사 계보(§7.181 CAP2 → LB +1.02)의 다음 노브다.

팔 (코어 LGBM 20 전체 교체 — CAP2 와 같은 전달 형태. t1core.build 재사용,
    rng 777 → 합성행 동일 → 쌍체. leaves 는 **배포와 같은 x2** 유지)
    기준   CAP2 (t1knob_CAP2.npz 재사용, 앵커 1525.8)
    DART320  boosting=dart, N_ROUNDS 320 (배포 라운드 그대로. drop_rate .1 등 기본값)
    DART640  boosting=dart, 640 라운드 (dart 는 드롭 보상으로 라운드가 더 필요하다는
             통설 검정. 배포 시 추론 트리수 2 배 — 예산 내)
    BLEND    사후 산술 — logit 0.8·CAP2 + 0.2·(dart 중 나은 팔).
             "5번째 변형 추가" 근사 (코어 20:5 비중). ⚠️ 라우팅·fspec·δ 가 코어와
             출력 사이에 있어 정확한 변형 추가와 다르다 — 근사임을 명기.

사전 등록 판정 (t1knob 규약 — ⚠️ fold2024 하나뿐, 최종 판정자는 LB)
    후보 = Δ vs CAP2 양수 · P(Δ>0) ≥ 0.9 (투수 클러스터 부트스트랩 B=1000).
    2팔 x 10% → 우연 통과 기대 0.2. 통과 팔만 B=4000 재확인 후 LB 후보.
    미달이면 "DART 는 이 레짐에서 이득 없음"으로 기록하고 닫는다.
    라운드·drop_rate 를 fold 를 보고 다시 고르지 않는다 — 팔 2개가 가설의 전부다.

배관 검증 (사전)
    ⓐ 기준 BSS = t1knob 실측 1525.8 ± 0.5
    ⓑ 각 팔 dup·SMOTE k 가 동일 (부스팅 모드는 행 집합을 못 바꾼다)
    ⓒ 각 팔 vs CAP2 최대|Δp| > 0

실행:  .venv/Scripts/python.exe -u src/t1dart.py 2>&1 | tee experiments/t1dart.log
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
B_CONFIRM = 4000
CAP2_BSS_KNOWN = 1525.8          # §7.181 실측 앵커
LEAF_MULT = 2.0                  # 배포(s45) 코어와 동일
ARMS = {"DART320": 320, "DART640": 640}
BLEND_W = 0.2                    # 5번째 변형(5시드) 추가 근사 = 5/25


def leaf2(v):
    nl = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v]).get("num_leaves", 31)
    return max(2, int(round(nl * LEAF_MULT)))


def main():
    t0 = time.time()
    art = joblib.load(T1 / "model" / "artifacts.joblib")["artifacts"]
    cols = lgb.Booster(model_str=(T1 / "model" / "lgbm_base_42.txt")
                       .read_text(encoding="utf-8")).feature_name()
    df = load_csv(TRAIN_CSV)
    tr = df[df.season < VALID].reset_index(drop=True)

    need = [ASCII / f"DRT_{a}" / "model" / f"lgbm_lowff_{SEEDS[-1]}.txt"
            for a in ARMS]
    if all(f.exists() for f in need):
        print("  전 팔 재사용")
    else:
        X = t1_features(tr, art, cols)
        print(f"피처 {X.shape}  ({time.time()-t0:.0f}s)")
        w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                           ref_season=int(tr.season.max()))
        X2, y2, w2, init, k = t1core_build(tr, X, w, True,
                                           np.random.default_rng(777))
        nd = len(X2) - len(X) - k
        print(f"  행 {len(X):,} → {len(X2):,} (dup {nd:,} · SMOTE {k:,})")
        assert k > 0 and nd > 0, "배관 ⓑ 실패 — 증강/soften 이 안 걸렸다"
        del X
        for a, rounds in ARMS.items():
            dst = ASCII / f"DRT_{a}"
            if (dst / "model" / f"lgbm_lowff_{SEEDS[-1]}.txt").exists():
                print(f"  {a} 재사용"); continue
            shutil.copytree(T1, dst, dirs_exist_ok=True)
            for v in VARIANTS:
                p = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v])
                p["num_leaves"] = leaf2(v)
                p["boosting"] = "dart"          # drop_rate .1 · skip_drop .5 기본값
                for sd in SEEDS:
                    p2 = dict(p, seed=sd, bagging_seed=sd, drop_seed=sd,
                              feature_fraction_seed=sd, data_random_seed=sd)
                    b = lgb.train(p2, lgb.Dataset(X2, y2, weight=w2,
                                                  init_score=init),
                                  num_boost_round=rounds)
                    # LF — CRLF 는 model_file= 을 죽인다 (§7.173 ③)
                    (dst / "model" / f"lgbm_{v}_{sd}.txt").write_text(
                        b.model_to_string(), encoding="utf-8", newline="\n")
                    del b
                print(f"    [{a}] {v} nl={p['num_leaves']} r={rounds}"
                      f" ({time.time()-t0:.0f}s)")
        del X2, y2, w2, init
    del tr

    # ── 예측 (T1 자신의 predict.py) ─────────────────────────────
    wk = ROOT / "experiments" / "cache" / "_t1dart_worker.py"
    wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/")),
                  encoding="utf-8", newline="\n")
    P = {}
    for a, dname in [("CAP2", "KNB_CAP2")] + [(x, f"DRT_{x}") for x in ARMS]:
        f = ROOT / "experiments" / "cache" / ("t1knob_CAP2.npz" if a == "CAP2"
                                              else f"t1dart_{a}.npz")
        if not f.exists():
            r = subprocess.run([sys.executable, str(wk), dname, str(f)],
                               capture_output=True, text=True)
            if not f.exists():
                print(f"  ❌ {a}: {r.stderr.strip()[-300:]}"); return
        P[a] = np.load(f)["p"]
    print("  배관 ⓒ OK — CAP2 대비 최대|Δp|: " +
          " ".join(f"{a} {np.abs(P[a]-P['CAP2']).max():.5f}" for a in ARMS))
    for a in ARMS:
        assert np.abs(P[a] - P["CAP2"]).max() > 0, f"{a} 가 CAP2 와 같다"

    # ── 평가: fold2024, 투수 클러스터 부트스트랩 (t1knob 과 동일) ──
    va = df[df.season == VALID].reset_index(drop=True)
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    isF = va["game_type"].to_numpy().astype(str) == "F"
    codes, uniq = pd.factorize(va["pitcher_id"].to_numpy())
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    seen = np.where(np.diff(bnd) > 0)[0]

    def make_boots(rng, B):
        return [np.concatenate([order[bnd[j]:bnd[j+1]] for j in
                                rng.choice(seen, size=len(seen), replace=True)])
                for _ in range(B)]

    def s_(p, m=None, idx=None):
        pp, yy = (p, yva) if m is None else (p[m], yva[m])
        if idx is not None:
            pp, yy = pp[idx], yy[idx]
        return bss(yy, shift_to_prior(pp, yy.mean()))

    b0 = s_(P["CAP2"])
    assert abs(b0 - CAP2_BSS_KNOWN) < 0.5, f"배관 ⓐ 실패 — {b0:.1f}"

    eps = 1e-6
    zc = np.log(np.clip(P["CAP2"], eps, 1-eps) / (1 - np.clip(P["CAP2"], eps, 1-eps)))
    best = max(ARMS, key=lambda a: s_(P[a]))
    zd = np.log(np.clip(P[best], eps, 1-eps) / (1 - np.clip(P[best], eps, 1-eps)))
    P["BLEND"] = 1 / (1 + np.exp(-((1 - BLEND_W) * zc + BLEND_W * zd)))

    boots = make_boots(np.random.default_rng(42), B_BOOT)
    print("\n" + "=" * 92)
    print(f"§7.186 DART 재감사 · 코어 20 (leaves x2) · fold2024 · ⚠️ fold 하나뿐 · B={B_BOOT}")
    print("=" * 92)
    print("  %-26s %9s %11s %22s %8s" % ("팔", "BSS", "Δ vs CAP2", "95% 구간", "P(Δ>0)"))
    print("  %-26s %9.1f   (배관 ⓐ OK — §7.181 앵커 %.1f)" % ("CAP2 (배포 코어)", b0, CAP2_BSS_KNOWN))
    cands = []
    for a in list(ARMS) + ["BLEND"]:
        tag = a if a != "BLEND" else f"BLEND .8·CAP2+.2·{best}"
        D = np.array([s_(P[a], idx=ix) - s_(P["CAP2"], idx=ix) for ix in boots])
        lo, hi = np.percentile(D, [2.5, 97.5])
        pos = (D > 0).mean()
        d = s_(P[a]) - b0
        flag = "  ← 후보(P≥.9)" if (d > 0 and pos >= 0.9) else ""
        if d > 0 and pos >= 0.9:
            cands.append(a)
        print("  %-26s %9.1f %+11.2f   [%+7.2f, %+7.2f] %8.3f%s"
              % (tag, s_(P[a]), d, lo, hi, pos, flag))

    print("\n  F행 / R행 / 예측 std 분해")
    for a in ["CAP2"] + list(ARMS) + ["BLEND"]:
        print("    %-8s F %8.1f   R %8.1f   std(p) %.4f"
              % (a, s_(P[a], isF), s_(P[a], ~isF), P[a].std()))

    if cands:
        print(f"\n  B={B_CONFIRM} 재확인 ([[bootstrap-B-can-flip-a-borderline-gate]])")
        boots4 = make_boots(np.random.default_rng(4242), B_CONFIRM)
        for a in cands:
            D = np.array([s_(P[a], idx=ix) - s_(P["CAP2"], idx=ix) for ix in boots4])
            lo, hi = np.percentile(D, [2.5, 97.5])
            print(f"    {a}: Δ {s_(P[a])-b0:+.2f}  [{lo:+.2f}, {hi:+.2f}]"
                  f"  P {np.mean(D > 0):.4f}")
    else:
        print("\n→ 전 팔 미달 — DART 는 이 레짐에서 이득 없음. 축 폐쇄.")
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
