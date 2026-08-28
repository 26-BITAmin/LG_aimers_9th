"""§7.19x — CatBoost α 의 R/F 분리 — s45 등가 fold2024 층에서 첫 실측 (`src/catrf45.py`)

배경
    s45 블렌드에서 유일하게 R/F 공통인 노브가 CatBoost α=0.10 이다 (라우팅·F전문가는
    이미 분기). §7.145④ 는 s35 층 fold2024 가법 분해로 "CatBoost 이득은 R 집중,
    F행 손해" (dF −0.534, 초과분 [−1.647, −0.029] 0 배제) 를 확정하고 R-only 라우팅을
    "위약과 함께 별도 슬롯" 후보로 남겼는데 실행되지 않았다. 현 스택(T1 계보)에서는
    한 번도 안 재졌다.

층 구성 (s45 등가 fold2024)
    코어      KNB_CAP2 (§7.181 이 학습한 fold(<2024) leaves×2 코어 20)
    predict   S45REF 의 predict.py (라우팅 0.08125/0.375 + cat 블렌드 + affine)
    cat       이 스크립트가 학습하는 fold(<2024)·무증강 12멤버 (§7.146 레시피 그대로)
    나머지    출하본 (MLP8·compact·general12·F전문가) — 전 팔 공통이라 Δ에서 1차 상쇄

⚠️ 편향 방향 (사전 기록): F행의 채연 F전문가(37.5%)는 2024 를 본 in-sample 이라
   F행 기저가 실제(2025)보다 좋게 잡힌다 → fold-정직한 cat 의 F행 기여는
   **과소평가(손해는 과대평가)** 방향. 즉 이 측정은 α_F=0 쪽으로 기울어 있다.
   통과해도 LB 기대는 보수적으로 잡고, **미달이면 그쪽이 더 강한 증거다.**

기법: predict.py 를 패치해 로짓 조각(z_pre·z_cat·z_adj)을 덤프하고, 임의의
   (α_R, α_F) 를 오프라인 재구성한다. 재구성이 실제 Predictor 출력과 1e-12 로
   일치해야 유효 ([[simplified-reimplementations-need-a-known-number-check]]).

사전 등록 게이트 (결과 보기 전 고정) — 셋 다 통과해야 LB 후보
    G1  Δ(RONLY − BASE) > 0 이고 투수 클러스터 쌍체 부트스트랩 P(Δ>0) ≥ 0.9
    G2  Δ(RONLY − BASE) > 위약 널(무의미 서브그룹에 α=0, F 질량 매칭,
        투수 클러스터형 + 투수 내 행 무작위형 각 20) 의 max
    G3  가법 분해 dF(BASE vs NOCAT) < 0, 95% 구간 0 배제 (§7.145④ 이 층 재현)

실행:  .venv/Scripts/python.exe -u src/catrf45.py > experiments/catrf45.log 2>&1
"""
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
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

from config import (CAT_ALPHA, CAT_ITERS, CAT_PARAMS, CAT_SEEDS, CAT_VARIANTS,
                    RECENCY_HALF_LIFE, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from t1core import build, t1_features
from validation import bss, shift_to_prior

CRLF = chr(13) + chr(10)
LF = chr(10)
VALID = 2024
ARM_DIR = ASCII / "CATRF45"
SRC_DIR = ASCII / "KNB_CAP2"          # fold leaves×2 코어 (§7.181)
REF_PRED = ASCII / "S45REF" / "predict.py"
PIECES = ROOT / "experiments" / "cache" / "catrf45_pieces.npz"
B_BOOT = 2000
N_PLACEBO = 20                        # 기하별

WORKER = r"""
import sys, numpy as np, pandas as pd
d = r"D:/t1eval/CATRF45"
sys.path.insert(0, d)                  # ⚠️ 우리 src 를 넣지 않는다 (§7.174 사고)
va = pd.read_csv(r"{csv}")
va = va[va.season == 2024].reset_index(drop=True)
from predict import Predictor
import predict as _pm
assert _pm.__file__.replace("\\", "/").startswith(d), "CATRF45 predict.py 가 아니다"
pr = Predictor(model_dir=d + "/model")
p = np.asarray(pr.predict(va), dtype=np.float64)
np.savez_compressed(sys.argv[1], p=p, zpre=pr._zpre, zc=pr._zc,
                    zadj=np.asarray(pr._zadj, dtype=np.float64),
                    delta=np.float64(pr.delta))
"""

PATCH = [
    ("            zc /= len(self.cats)",
     "            zc /= len(self.cats)\n"
     "            self._zpre, self._zc = z.copy(), zc.copy()   # catrf45 조각 덤프"),
    ("        z = z + z_adj",
     "        self._zadj = z_adj                                 # catrf45 조각 덤프\n"
     "        z = z + z_adj"),
]


def stage_a():
    """팔 디렉터리 구성 + fold cat 12 학습 (재개 안전 — cat_*.cbm 존재가 재개 키,
    CATRF45 에는 원본 cat 이 하나도 없으므로 §7.176a 함정이 없다)."""
    t0 = time.time()
    if not (ARM_DIR / "predict.py").exists():
        assert (SRC_DIR / "model" / "lgbm_lowff_1234.txt").exists(), "KNB_CAP2 코어 없음"
        shutil.copytree(SRC_DIR, ARM_DIR, dirs_exist_ok=True)
    # predict.py = S45REF 원본 + 조각 덤프 패치 (항상 원본에서 다시 만든다 — 멱등)
    src = REF_PRED.read_text(encoding="utf-8").replace(CRLF, LF)
    for old, new in PATCH:
        assert src.count(old) == 1, f"패치 대상 {src.count(old)}회: {old[:40]!r}"
        src = src.replace(old, new)
    (ARM_DIR / "predict.py").write_text(src, encoding="utf-8", newline="\n")

    tags = [f"{v}_{sd}" for v in CAT_VARIANTS for sd in CAT_SEEDS]
    todo = [t for t in tags if not (ARM_DIR / "model" / f"cat_{t}.cbm").exists()]
    if todo:
        from catboost import CatBoostClassifier, Pool
        art = joblib.load(T1 / "model" / "artifacts.joblib")["artifacts"]
        cols = lgb.Booster(model_str=(T1 / "model" / "lgbm_base_42.txt")
                           .read_text(encoding="utf-8")).feature_name()
        df = load_csv(TRAIN_CSV)
        tr = df[df.season < VALID].reset_index(drop=True)
        X = t1_features(tr, art, cols)
        w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                           ref_season=int(tr.season.max()))
        X2, y2, w2, init, k = build(tr, X, w, False, np.random.default_rng(777))
        assert k == 0, f"증강이 걸렸다 ({k}) — cat 은 무증강 (§7.146)"
        print(f"[{time.time()-t0:.0f}s] fold(<{VALID}) 무증강 {len(X2):,}행 · 피처 {X2.shape[1]}")
        del X, df, tr
        pool = Pool(X2, y2, weight=w2, baseline=init)
        del X2, y2, w2, init
        for tag in tags:
            f = ARM_DIR / "model" / f"cat_{tag}.cbm"
            if f.exists():
                print(f"  {tag:<14} 재사용"); continue
            var, sd = tag.rsplit("_", 1)
            m = CatBoostClassifier(iterations=CAT_ITERS, random_seed=int(sd),
                                   **CAT_PARAMS, **CAT_VARIANTS[var])
            m.fit(pool)
            m.save_model(str(f))
            print(f"[{time.time()-t0:.0f}s]   {tag:<14} 완료 ({f.stat().st_size/1e6:.1f} MB)")
            del m
        del pool
    else:
        print("  cat 12 재사용 (전부 존재)")
    (ARM_DIR / "model" / "cat_members.json").write_text(
        json.dumps({"alpha": CAT_ALPHA, "members": tags}), encoding="utf-8")


def stage_b():
    if PIECES.exists():
        print("  조각 npz 재사용")
        return
    wk = ROOT / "experiments" / "cache" / "_catrf45_worker.py"
    wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/")),
                  encoding="utf-8", newline="\n")
    r = subprocess.run([sys.executable, str(wk), str(PIECES)],
                       capture_output=True, text=True)
    if not PIECES.exists():
        print(f"  ❌ worker: {r.stderr.strip()[-500:]}")
        sys.exit(1)


def stage_c():
    t0 = time.time()
    d = np.load(PIECES)
    p_ref, zpre, zc, zadj, delta = (d["p"], d["zpre"], d["zc"], d["zadj"],
                                    float(d["delta"]))
    src = (ARM_DIR / "predict.py").read_text(encoding="utf-8")
    C = float(re.search(r"PROB_CENTER\s*=\s*([0-9.eE+-]+)", src).group(1))
    S = float(re.search(r"PROB_SCALE\s*=\s*([0-9.eE+-]+)", src).group(1))
    SH = float(re.search(r"PROB_SHIFT\s*=\s*([0-9.eE+-]+)", src).group(1))
    m = re.search(r"CLIP_EPS\s*=\s*([0-9.eE+-]+)", src)
    EPS = float(m.group(1)) if m else 1e-6

    df = load_csv(TRAIN_CSV)
    va = df[df.season == VALID].reset_index(drop=True)
    del df
    assert len(va) == len(zpre), (len(va), len(zpre))
    yva = va[TARGET_COL].to_numpy(dtype=np.float64)
    isF = va["game_type"].to_numpy().astype(str) == "F"
    fshare = float(isF.mean())

    def recon(a_row):
        zb = (1.0 - a_row) * zpre + a_row * zc
        p = 1.0 / (1.0 + np.exp(-(zb + zadj + delta)))
        p = C + S * (p - C) + SH
        return np.clip(p, EPS, 1 - EPS)

    def alpha_vec(aR, aF):
        a = np.full(len(zpre), aR)
        a[isF] = aF
        return a

    # 배관 ⓐ — 재구성이 실제 Predictor 출력과 일치하는가 (알려진 수 검증)
    p_chk = recon(alpha_vec(CAT_ALPHA, CAT_ALPHA))
    dmax = float(np.abs(p_chk - p_ref).max())
    print(f"  배관 ⓐ 재구성 vs Predictor  max|Δp| = {dmax:.2e}  "
          f"{'✅' if dmax < 1e-12 else '❌'}")
    assert dmax < 1e-10, "재구성 불일치 — 오프라인 결과 전체 무효"

    ARMS = {
        "NOCAT": (0.00, 0.00),
        "BASE":  (0.10, 0.10),
        "RONLY": (0.10, 0.00),
        "RHALF": (0.10, 0.05),
        "FONLY": (0.00, 0.10),
    }
    P = {a: recon(alpha_vec(*v)) for a, v in ARMS.items()}
    dm = float(np.abs(P["BASE"] - P["NOCAT"]).mean())
    print(f"  배관 ⓑ cat 이동폭 평균|Δp| = {dm:.5f}  (§7.178 배포층 0.00105 대조)")
    assert np.abs(P["BASE"] - P["NOCAT"]).max() > 0

    # 클러스터 (투수) — t1knob 과 동일 기계
    codes, uniq = pd.factorize(va["pitcher_id"].to_numpy())
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    seen = np.where(np.diff(bnd) > 0)[0]
    rng = np.random.default_rng(42)

    def s_(p, m=None, idx=None):
        pp, yy = (p, yva) if m is None else (p[m], yva[m])
        if idx is not None:
            pp, yy = pp[idx], yy[idx]
        return bss(yy, shift_to_prior(pp, yy.mean()))

    # ── 가법 분해 준비: cat 기여 (BASE − NOCAT) 의 F/R 몫 (§7.145④ catfr3 방식) ──
    ref = yva.mean() * (1 - yva.mean())
    pB = shift_to_prior(P["BASE"], yva.mean())
    pN = shift_to_prior(P["NOCAT"], yva.mean())
    di = 1e5 * ((pN - yva) ** 2 - (pB - yva) ** 2) / (len(yva) * ref)
    dT, dF, dR = float(di.sum()), float(di[isF].sum()), float(di[~isF].sum())

    # ── 부트스트랩: draw 마다 인덱스를 만들고 즉시 버린다 (메모리 O(n)) ──
    BOOT_ARMS = ("NOCAT", "RONLY", "RHALF")     # FONLY 는 점추정만 (진단용)
    D = {a: np.empty(B_BOOT) for a in BOOT_ARMS}
    exc = np.empty(B_BOOT)
    for b in range(B_BOOT):
        ix = np.concatenate([order[bnd[j]:bnd[j+1]] for j in
                             rng.choice(seen, size=len(seen), replace=True)])
        sB = s_(P["BASE"], idx=ix)
        for a in BOOT_ARMS:
            D[a][b] = s_(P[a], idx=ix) - sB
        exc[b] = di[ix][isF[ix]].sum() - di[ix].sum() * fshare
        del ix

    print("\n" + "=" * 96)
    print(f"CatBoost α R/F 분리 · s45 등가 fold2024 층 · fold cat12 · B={B_BOOT}"
          f" · F행 {fshare:.3%}")
    print("  ⚠️ F행 기저에 in-sample F전문가 → α_F=0 쪽 편향 (헤더). fold 하나뿐.")
    print("=" * 96)
    print("  %-22s %9s %10s %22s %8s %9s %9s"
          % ("팔 (α_R, α_F)", "BSS", "Δ vs BASE", "95% 구간", "P(Δ>0)", "F행", "R행"))
    stats = {}
    for a, v in ARMS.items():
        if a == "BASE":
            print("  %-22s %9.1f %10s %22s %8s %9.1f %9.1f"
                  % ("BASE  (.10, .10)", s_(P[a]), "—", "", "",
                     s_(P[a], isF), s_(P[a], ~isF)))
            continue
        dpt = float(s_(P[a]) - s_(P["BASE"]))
        if a in BOOT_ARMS:
            lo, hi = np.percentile(D[a], [2.5, 97.5])
            pos = float((D[a] > 0).mean())
            stats[a] = (dpt, lo, hi, pos)
            print("  %-22s %9.1f %+10.2f   [%+7.2f, %+7.2f] %8.3f %9.1f %9.1f"
                  % (f"{a:<6}({v[0]:.2f}, {v[1]:.2f})", s_(P[a]), dpt, lo, hi, pos,
                     s_(P[a], isF), s_(P[a], ~isF)))
        else:
            print("  %-22s %9.1f %+10.2f   %22s %8s %9.1f %9.1f"
                  % (f"{a:<6}({v[0]:.2f}, {v[1]:.2f})", s_(P[a]), dpt,
                     "(점추정만)", "", s_(P[a], isF), s_(P[a], ~isF)))

    lo, hi = np.percentile(exc, [2.5, 97.5])
    print(f"\n  cat 기여 가법 분해   dTotal {dT:+.3f} = dF {dF:+.3f} + dR {dR:+.3f}"
          f"   균등 기대 dF {dT*fshare:+.3f}")
    print(f"  F 초과분 {dF - dT*fshare:+.3f}  [{lo:+.3f}, {hi:+.3f}]"
          f"   (§7.145④ s35층: dF −0.534, 초과 [−1.647, −0.029])")
    g3 = dF < 0 and hi < 0

    # ── 위약 널: 무의미 서브그룹에 α=0 (F 질량 매칭) ─────────────────
    nulls = {}
    for geom in ("pitcher", "withinrow"):
        rngp = np.random.default_rng(7 if geom == "pitcher" else 2024)
        vals = []
        for _ in range(N_PLACEBO):
            if geom == "pitcher":
                mask = np.zeros(len(yva), bool)
                for j in rngp.permutation(seen):
                    mask[order[bnd[j]:bnd[j+1]]] = True
                    if mask.mean() >= fshare:
                        break
            else:
                mask = rngp.random(len(yva)) < fshare
            a = np.full(len(yva), 0.10)
            a[mask] = 0.0
            vals.append(s_(recon(a)) - s_(P["BASE"]))
        nulls[geom] = np.array(vals)
        print(f"  위약({geom:<9}) n={N_PLACEBO}  중앙값 {np.median(nulls[geom]):+.2f}"
              f"  q95 {np.quantile(nulls[geom], .95):+.2f}  max {nulls[geom].max():+.2f}")

    d_ronly = stats["RONLY"][0]
    g1 = d_ronly > 0 and stats["RONLY"][3] >= 0.9
    g2 = all(d_ronly > n.max() for n in nulls.values())
    print("\n  게이트  G1(Δ>0·P≥.9) %s   G2(위약 max 초과) %s   G3(dF<0·CI 0 배제) %s"
          % tuple("✅" if g else "❌" for g in (g1, g2, g3)))
    print("  판정: " + ("**LB 후보** — 단 F전문가 in-sample 편향으로 기대는 보수적으로"
                      if (g1 and g2 and g3) else
                      "균일 α=0.10 유지 (게이트 미달)"))
    print(f"\n  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    stage_a()
    stage_b()
    stage_c()
